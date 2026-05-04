#!/usr/bin/env python3
"""Stage 3: extract per-method uncertainty samples on the test split.

Reads ``runs/<run_id>/method.pth`` (saved by :mod:`fit_uncertainty`),
imports the method module via ``importlib.import_module``, calls
``module.extract(...)``, and writes ``runs/<run_id>/predictions.npz``
with keys ``logits`` (``(N, K, S)`` float32) and ``indices``
(``(N,)`` int64). ``S`` is method-specific: MC-Dropout's
``n_samples`` (default 20) or the ensemble's ``n_members``.

A_hat and E_hat are NOT computed here -- they belong to the oracle /
decomposition layer (Tasks 6/6.5). The extraction layer is
loss-agnostic; no loss strings appear in this file or in the method
modules.
"""

from __future__ import annotations

import argparse
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

from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402
from probly.method.head import MlpHead  # noqa: E402
from probly.quantification.decomposition import OutputSchema  # noqa: E402

#: Per-schema set of cache keys that ``predictions.npz`` must contain
#: for the dispatch chain in ``compute_decomposition.py`` to succeed.
#: Every method's ``extract()`` is contractually required to return a
#: dict with these keys + ``"indices"``. Mismatch is a wrapper bug.
_REQUIRED_KEYS_BY_SCHEMA: dict[str, frozenset[str]] = {
    "logits_nks": frozenset({"logits"}),
    "evidential_alpha": frozenset({"alpha", "evidence"}),
    "ddu_probs_density": frozenset({"probs", "density"}),
}


def _resolve_callable(dotted: str) -> Any:  # noqa: ANN401
    """Resolve a dotted path like ``a.b:c`` or ``a.b.c`` to a callable."""
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    else:
        module_path, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _full_network_factory(dataset_config: dict[str, Any]) -> Any:  # noqa: ANN401
    """Build a model factory for ``full_network`` mode."""
    classifier_cfg = dataset_config.get("classifier", {})
    arch = classifier_cfg["architecture"]
    cls_or_fn = _resolve_callable(arch)
    num_classes = int(classifier_cfg.get("num_classes", 0))

    def factory() -> nn.Module:
        try:
            return cls_or_fn(num_classes=num_classes)  # type: ignore[call-arg]
        except TypeError:
            return cls_or_fn()

    return factory


def _head_factory(head_factory_args: dict[str, Any]) -> Any:  # noqa: ANN401
    """Build a model factory for ``linear_probe`` mode from saved args."""
    in_features = int(head_factory_args["in_features"])
    num_classes = int(head_factory_args["num_classes"])
    hidden = int(head_factory_args.get("hidden", 256))
    dropout_p = float(head_factory_args.get("dropout_p", 0.1))

    def factory() -> nn.Module:
        return MlpHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden=hidden,
            dropout_p=dropout_p,
        )

    return factory


def _make_full_network_cifar10h_test_provider(
    dataset_config: dict[str, Any],
    batch_size: int = 256,
) -> FeatureProvider:
    """Build a test-time provider for the CIFAR-10H full-network path.

    Loads the canonical CIFAR-10 test split via :class:`CIFAR10NoMD5`
    (canonical ordering matches ``cifar10h-counts.npy``), applies the
    eval-time normalisation declared in the dataset config's
    ``training.normalization``, and yields ``(image_tensor, label)``
    batches.

    Labels are integer class indices, which is what
    :class:`CIFAR10NoMD5` returns natively; the method modules
    discard them at extract time, so we just keep them as-is.
    """
    from torchvision import transforms as T  # noqa: PLC0415

    from experiments.epistemic_eval.datasets.cifar10_canonical import (  # noqa: PLC0415
        CIFAR10NoMD5,
    )

    norm = dataset_config["training"]["normalization"]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(tuple(float(x) for x in norm["mean"]), tuple(float(x) for x in norm["std"])),
    ])
    cifar_root = dataset_config.get("loader_kwargs", {}).get("root", "data/cifar10h")
    test_ds = CIFAR10NoMD5(root=cifar_root, train=False, transform=transform)
    n = len(test_ds)
    indices = np.arange(n, dtype=np.int64)
    num_classes = int(dataset_config.get("classifier", {}).get("num_classes", 10))

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xs = []
        ys = []
        for i in range(start, stop):
            x, y = test_ds[i]
            xs.append(x)
            ys.append(y)
        batches.append((torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)))
    return FeatureProvider(
        batches=batches,
        mode="full_network",
        feature_dim=None,
        n_classes=num_classes,
        indices=indices,
    )


def _make_test_provider(
    dataset_config: dict[str, Any],
    feature_cache_dir: Path,
    seed: int = 0,
) -> FeatureProvider:
    """Build a test-time provider from the dataset config.

    For linear-probe mode, reads ``data/appa_real_features/test.npz``.
    For full-network CIFAR-10H, loads the canonical-format test split
    via :class:`CIFAR10NoMD5` and applies the dataset's normalisation
    transform. For DCIC datasets, picks the seed-derived test fold
    (per :func:`experiments.epistemic_eval.datasets.dcic.test_fold_for_seed`)
    and builds a no-augmentation DataLoader-backed provider over that
    slice. ImageNet-ReaL full-network is still cluster-only;
    raises ``NotImplementedError`` with a clear message there.
    """
    extraction_mode = dataset_config.get("extraction_mode", "full_network")
    if extraction_mode != "linear_probe":
        if dataset_config.get("name") == "cifar10h":
            return _make_full_network_cifar10h_test_provider(dataset_config)
        if dataset_config.get("family") == "dcic":
            from experiments.epistemic_eval.datasets.dcic import (  # noqa: PLC0415
                build_test_provider,
            )

            training_block = dataset_config.get("training", {}) or {}
            return build_test_provider(
                dataset_config,
                seed=int(seed),
                batch_size=int(training_block.get("batch_size", 32)),
                num_workers=int(training_block.get("num_workers", 0)),
            )
        msg = (
            f"Full-network test FeatureProvider for dataset "
            f"{dataset_config.get('name')!r} is not yet wired here. "
            "Add a per-dataset branch alongside the cifar10h one in "
            "_make_test_provider, or build a FeatureProvider in-process "
            "and call extract() directly."
        )
        raise NotImplementedError(msg)
    cache_file = feature_cache_dir / "test.npz"
    if not cache_file.exists():
        msg = f"feature cache not found at {cache_file}; run extract_features.py first."
        raise FileNotFoundError(msg)
    blob = np.load(cache_file, allow_pickle=True)
    features = np.asarray(blob["features"], dtype=np.float32)
    n = features.shape[0]
    feature_dim = int(features.shape[1])
    num_classes = int(dataset_config.get("metadata", {}).get("num_classes", 0))
    labels = np.full((n, max(num_classes, 1)), 1.0 / max(num_classes, 1), dtype=np.float32)
    indices = np.arange(n, dtype=np.int64)
    batches = []
    batch_size = 128
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = torch.from_numpy(features[start:stop])
        y = torch.from_numpy(labels[start:stop])
        batches.append((x, y))
    return FeatureProvider(
        batches=batches,
        mode="linear_probe",
        feature_dim=feature_dim,
        n_classes=max(num_classes, 1),
        indices=indices,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "appa_real_features",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"missing config at {config_path}", file=sys.stderr)
        return 1
    merged = yaml.safe_load(config_path.read_text())
    method_config = merged["method"]
    dataset_config = merged["dataset"]
    method_module = importlib.import_module(method_config["method_module"])

    handle = method_module.load(run_dir / "method.pth")

    extraction_mode = dataset_config.get("extraction_mode", "full_network")
    if extraction_mode == "linear_probe":
        head_args = handle.head_factory_args or {}
        factory = _head_factory(head_args)
    elif dataset_config.get("family") == "dcic":
        # DCIC datasets need the same torchvision-resnet18 + K-class
        # head that fit_uncertainty.py used at training time. The
        # dataset_config carries num_classes; pretrained=False because
        # the loaded handle's state_dict will overwrite the head and
        # encoder weights anyway, and we don't want to re-download
        # ImageNet weights at extract time.
        from experiments.epistemic_eval.datasets.dcic import (  # noqa: PLC0415
            make_resnet18_factory,
        )

        num_classes = int(dataset_config["classifier"]["num_classes"])
        factory = make_resnet18_factory(num_classes=num_classes, pretrained=False)
    else:
        factory = _full_network_factory(dataset_config)

    seed = int(merged.get("seed", 0))
    provider = _make_test_provider(dataset_config, args.feature_cache_dir, seed=seed)

    n_samples = int(method_config.get("n_samples", method_config.get("n_members", 1)))
    out = method_module.extract(
        handle, provider, n_samples=n_samples, model_factory=factory
    )

    # Schema-aware cache write. Each method's ``extract()`` returns a
    # dict whose keys depend on ``output_schema``: sampling-based
    # methods write ``logits`` of shape ``(N, K, S)``; evidential
    # writes ``alpha`` + ``evidence``; DDU writes ``probs`` + ``density``.
    # ``indices`` is required for every schema. Pre-schema runs
    # (no ``output_schema`` field) default to ``"logits_nks"`` for
    # backwards compatibility with mc_dropout / ensemble caches built
    # before Task 5b.
    output_schema_raw = method_config.get("output_schema", "logits_nks")
    if not isinstance(output_schema_raw, str):
        msg = (
            f"method config 'output_schema' must be a string, got "
            f"{type(output_schema_raw).__name__}: {output_schema_raw!r}."
        )
        raise TypeError(msg)
    valid_schemas = OutputSchema.__args__  # type: ignore[attr-defined]
    if output_schema_raw not in valid_schemas:
        msg = (
            f"unknown output_schema {output_schema_raw!r} in method config; "
            f"expected one of {valid_schemas!r}."
        )
        raise ValueError(msg)
    output_schema: str = output_schema_raw

    required = _REQUIRED_KEYS_BY_SCHEMA[output_schema]
    missing = required - set(out)
    if missing:
        msg = (
            f"method module {method_config['method_module']!r} returned "
            f"keys {sorted(out)}, missing required {sorted(missing)} for "
            f"schema {output_schema!r}."
        )
        raise ValueError(msg)
    if "indices" not in out:
        msg = (
            f"method module {method_config['method_module']!r} returned "
            f"no 'indices' key in extract(); 'indices' is required for "
            f"every schema."
        )
        raise ValueError(msg)

    # Save only the schema-required keys + indices, deterministic
    # alphabetical order for readability of resulting predictions.npz.
    payload = {key: out[key] for key in sorted(required)}
    payload["indices"] = out["indices"]
    np.savez_compressed(run_dir / "predictions.npz", **payload)

    shape_summary = ", ".join(
        f"{k}={tuple(payload[k].shape)}"
        for k in sorted(required)
    )
    print(
        f"wrote {run_dir / 'predictions.npz'} "
        f"(schema={output_schema}, {shape_summary})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
