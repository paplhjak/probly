#!/usr/bin/env python3
"""Stage 2a: extract and cache APPA-REAL backbone features.

Loads the pretrained ResNet-101 from
``dataset_config['backbone']['weights_path']``, replaces its head
with :class:`torch.nn.Identity`, and runs one forward pass per image
per split. Caches the resulting features at
``data/appa_real_features/<split>.npz`` with keys ``features``
(``float32 (N, 2048)``) and ``filenames`` (``object`` array).

Idempotent: skips if the cache file exists and its config-hash sidecar
matches the current config-hash. ``--force-recompute`` bypasses the
cache.

Failure modes (fail loudly, exit code 1):

* ``weights_path`` missing on disk.
* ``backbone_meta.yaml`` records a non-null ``weights_sha256`` and the
  weights file's actual sha256 doesn't match.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.methods._base import hash_config  # noqa: E402


_DEFAULT_FEATURE_CACHE = _REPO_ROOT / "data" / "appa_real_features"


def _file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_callable(dotted: str) -> Any:  # noqa: ANN401
    """Resolve a dotted path like ``a.b:c`` or ``a.b.c`` to a callable."""
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    else:
        module_path, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _build_backbone(arch: str, weights_path: Path) -> nn.Module:
    """Instantiate the backbone and load weights into it.

    Loads the full state-dict first (so ``strict=True`` matches the
    saved checkpoint), then replaces the trailing classification head
    with :class:`torch.nn.Identity` so the module returns features.
    Works for the locked default ``torchvision.models.resnet101`` and
    for the smoke-test fixture (a tiny ``nn.Module`` with a ``.fc``).
    """
    factory = _resolve_callable(arch)
    backbone = factory()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    backbone.load_state_dict(state_dict, strict=True)
    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Module):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Module):
        backbone.classifier = nn.Identity()
    backbone.eval()
    return backbone


def _split_dataloader(dataset_config: dict[str, Any], split: str) -> Any:  # noqa: ANN401
    """Build a (dataset, image_size) pair for the given split.

    The function is split-out for testability. It dispatches to
    ``loader.module:class`` per the config and forwards the loader
    kwargs verbatim with ``split=<split>``.
    """
    loader_module = importlib.import_module(dataset_config["loader"]["module"])
    loader_cls = getattr(loader_module, dataset_config["loader"]["class"])
    kwargs = dict(dataset_config.get("loader_kwargs", {}))
    kwargs["split"] = split
    return loader_cls(**kwargs)


def _extract_features_for_split(
    backbone: nn.Module,
    dataset: Any,  # noqa: ANN401
    feature_dim: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``backbone`` over every dataset item.

    Returns ``(features, filenames)``. ``filenames`` is the dataset's
    own per-item filename when exposed; otherwise the integer index
    cast to string.
    """
    n = len(dataset)
    features = np.zeros((n, feature_dim), dtype=np.float32)
    filenames: list[str] = []
    backbone.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            xs = []
            for i in range(start, stop):
                item = dataset[i]
                if isinstance(item, tuple):
                    image = item[0]
                else:
                    image = item
                if not torch.is_tensor(image):
                    image = torch.as_tensor(image)
                xs.append(image)
                fname = getattr(dataset, "filenames", None)
                if fname is not None:
                    filenames.append(str(fname[i]))
                else:
                    filenames.append(str(i))
            batch = torch.stack(xs, dim=0).to(torch.float32)
            out = backbone(batch)
            if out.ndim > 2:
                out = out.flatten(1)
            features[start:stop] = out.detach().cpu().numpy().astype(np.float32, copy=False)
    return features, np.asarray(filenames, dtype=object)


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Dataset YAML config.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the feature-cache directory (mainly for tests).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to extract.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args(argv)

    dataset_config = yaml.safe_load(args.config.read_text())
    backbone_cfg = dataset_config.get("backbone")
    if backbone_cfg is None:
        print("dataset config has no 'backbone' section; nothing to extract.", file=sys.stderr)
        return 1
    weights_path = Path(backbone_cfg["weights_path"])
    feature_dim = int(backbone_cfg["feature_dim"])
    arch = str(backbone_cfg["architecture"])

    # Documented failure modes: print the remediation message to
    # stderr and exit with code 1 instead of a Python traceback.
    if not weights_path.exists():
        print(
            f"APPA-REAL backbone weights not found at {weights_path}.\n"
            f"Place the resnet101_CSFD16_weights.pth file at this path\n"
            f"(or update the dataset config to point elsewhere). See\n"
            f"decisions.md 'APPA-REAL backbone strategy' for provenance.",
            file=sys.stderr,
        )
        return 1
    meta_path = weights_path.parent / "backbone_meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        expected_sha = meta.get("weights_sha256")
        if expected_sha is not None:
            actual_sha = _file_sha256(weights_path)
            if actual_sha.lower() != str(expected_sha).lower():
                print(
                    f"Backbone weights at {weights_path} do not match recorded sha256 in\n"
                    f"backbone_meta.yaml. Refusing to proceed.",
                    file=sys.stderr,
                )
                return 1

    cache_dir = args.cache_dir or _DEFAULT_FEATURE_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = hash_config(dataset_config)

    backbone = _build_backbone(arch, weights_path)

    for split in args.splits:
        cache_file = cache_dir / f"{split}.npz"
        sidecar = cache_dir / f"{split}.config_hash"
        if cache_file.exists() and sidecar.exists() and not args.force_recompute:
            if sidecar.read_text().strip() == cfg_hash:
                print(f"cache hit for split={split}; skipping (sidecar matches).")
                continue
        dataset = _split_dataloader(dataset_config, split)
        features, filenames = _extract_features_for_split(backbone, dataset, feature_dim)
        np.savez_compressed(cache_file, features=features, filenames=filenames)
        sidecar.write_text(cfg_hash)
        print(f"wrote {cache_file} ({features.shape[0]} samples).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
