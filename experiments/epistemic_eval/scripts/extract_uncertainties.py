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


def _make_test_provider(
    dataset_config: dict[str, Any],
    feature_cache_dir: Path,
) -> FeatureProvider:
    """Build a test-time provider from the dataset config.

    For linear-probe mode, reads ``data/appa_real_features/test.npz``.
    For full-network mode, real-data wiring uses the dataset's loader
    (e.g. ``probly.datasets.torch.CIFAR10H``) on the cluster. The
    method-module-level extract path is exercised in-process by the
    Task 5 tests (``test_full_network_{mc_dropout,ensemble}.py``).
    """
    extraction_mode = dataset_config.get("extraction_mode", "full_network")
    if extraction_mode != "linear_probe":
        msg = (
            "Full-network real-data test FeatureProvider construction is a "
            "cluster-side wiring task (Task 9+); it requires the dataset's "
            "test split on disk plus the per-architecture preprocessing "
            "transform. The method-module extract() functions are exercised "
            "in-process by tests/test_full_network_{mc_dropout,ensemble}.py "
            "with synthetic providers; that path validates the (N, K, S) "
            "schema and dropout-vs-ensemble variability without requiring "
            "real data."
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
    else:
        factory = _full_network_factory(dataset_config)

    provider = _make_test_provider(dataset_config, args.feature_cache_dir)

    n_samples = int(method_config.get("n_samples", method_config.get("n_members", 1)))
    out = method_module.extract(
        handle, provider, n_samples=n_samples, model_factory=factory
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=out["logits"],
        indices=out["indices"],
    )
    print(
        f"wrote {run_dir / 'predictions.npz'} "
        f"(logits shape={out['logits'].shape})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
