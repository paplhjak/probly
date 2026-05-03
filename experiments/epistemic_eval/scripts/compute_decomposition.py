#!/usr/bin/env python3
"""Stage 5: decompose a method's cached ``(N, K, S)`` logits.

Reads ``runs/<run_id>/predictions.npz`` (written by
:mod:`extract_uncertainties`) and the dataset config attached to the
run (or supplied via ``--dataset-config``). For every loss declared in
``dataset_config['supported_losses']``, calls
:func:`probly.quantification.decomposition.decompose` and writes a
sidecar cache::

    runs/<run_id>/decomposition_<loss>.npz
        keys: A_hat (N,), E_hat (N,), H_hat (N,) or (N, K), indices (N,)

The user runs this script once per ``run_id`` and gets all losses
declared in the dataset config; downstream code (Task 7) reads the
file matching the chosen loss for that experiment.

Idempotence: per-loss outputs are skipped if the cache file exists
with a matching config hash, unless ``--force-recompute`` is passed.
The hash includes the dataset name, the loss string, the contents of
the ``support`` array, and the ``_DECOMPOSITION_VERSION`` constant; a
math change in
:mod:`probly.quantification.decomposition` bumps that constant and
invalidates every cached file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, cast

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.methods._base import hash_config  # noqa: E402
from probly.quantification.decomposition import (  # noqa: E402
    _DECOMPOSITION_VERSION,
    LossName,
    decompose,
)


def _git_commit() -> str:
    """Return the current git HEAD short SHA, or ``"unknown"`` on failure."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _write_meta(run_dir: Path, hashes: dict[str, str]) -> None:
    """Write ``meta.json`` capturing per-loss config hashes and provenance."""
    existing = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
    existing["decomposition_version"] = _DECOMPOSITION_VERSION
    existing["decomposition_git_commit"] = _git_commit()
    existing["decomposition_timestamp_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    existing["decomposition_host"] = socket.gethostname()
    existing["decomposition_per_loss_hashes"] = hashes
    meta_path.write_text(json.dumps(existing, indent=2))


def _support_fingerprint(support: np.ndarray) -> str:
    """Return a short hex digest of ``support``'s contents."""
    arr = np.asarray(support)
    return hashlib.blake2b(arr.tobytes() + str(arr.dtype).encode(), digest_size=8).hexdigest()


def _per_loss_hash(dataset_name: str, loss: str, support: np.ndarray) -> str:
    """Stable hash of (dataset, loss, support, decomposition version)."""
    payload = {
        "dataset": dataset_name,
        "loss": loss,
        "support_fingerprint": _support_fingerprint(support),
        "decomposition_version": int(_DECOMPOSITION_VERSION),
    }
    return hash_config(payload)


def _load_dataset_config(
    run_dir: Path,
    explicit_path: Path | None,
) -> dict[str, Any]:
    """Read the dataset config from ``--dataset-config`` or from the run's resolved config."""
    if explicit_path is not None:
        return yaml.safe_load(explicit_path.read_text())
    resolved = run_dir / "config.yaml"
    if not resolved.exists():
        msg = (
            f"--dataset-config was not provided and {resolved} does not exist; "
            f"cannot infer the dataset config."
        )
        raise FileNotFoundError(msg)
    merged = yaml.safe_load(resolved.read_text())
    if "dataset" not in merged:
        msg = f"resolved config at {resolved} has no 'dataset' section."
        raise KeyError(msg)
    return merged["dataset"]


def _resolve_support(dataset_config: dict[str, Any], k: int) -> np.ndarray:
    """Pick the support array from the dataset config.

    Priority:
        1. ``loader_kwargs.age_support`` (APPA-REAL pattern).
        2. ``classifier.num_classes`` -> ``arange(K)`` as a stand-in
           for losses that ignore support.
        3. ``metadata.num_classes`` -> same.
        4. ``arange(K)`` if neither field is set; the regression
           losses will then misbehave but the validation layer will
           catch shape mismatches.
    """
    loader_kwargs = dataset_config.get("loader_kwargs") or {}
    age_support = loader_kwargs.get("age_support")
    if age_support is not None:
        return np.asarray(age_support)
    classifier = dataset_config.get("classifier") or {}
    metadata = dataset_config.get("metadata") or {}
    num_classes = classifier.get("num_classes") or metadata.get("num_classes")
    if isinstance(num_classes, int):
        return np.arange(int(num_classes), dtype=np.int64)
    return np.arange(k, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Override; defaults to the run's resolved config.yaml -> 'dataset'.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args(argv)

    run_dir: Path = args.run
    if not run_dir.exists():
        msg = f"run directory not found at {run_dir}."
        raise FileNotFoundError(msg)
    predictions_path = run_dir / "predictions.npz"
    if not predictions_path.exists():
        msg = f"predictions.npz not found at {predictions_path}; run extract_uncertainties.py first."
        raise FileNotFoundError(msg)

    dataset_config = _load_dataset_config(run_dir, args.dataset_config)
    dataset_name = str(dataset_config["name"])
    supported_losses_raw = dataset_config.get("supported_losses")
    if not supported_losses_raw:
        msg = (
            f"dataset config for {dataset_name} is missing the required "
            f"'supported_losses' field."
        )
        raise KeyError(msg)
    supported_losses = list(supported_losses_raw)

    blob = np.load(predictions_path, allow_pickle=False)
    logits = np.asarray(blob["logits"])
    indices = np.asarray(blob["indices"])
    if logits.ndim != 3:
        msg = f"predictions.npz['logits'] must be 3-D, got shape={logits.shape}."
        raise ValueError(msg)
    k = int(logits.shape[1])
    support = _resolve_support(dataset_config, k)

    per_loss_hashes: dict[str, str] = {}
    for loss in supported_losses:
        loss_hash = _per_loss_hash(dataset_name, str(loss), support)
        per_loss_hashes[loss] = loss_hash
        cache_file = run_dir / f"decomposition_{loss}.npz"
        sidecar = run_dir / f"decomposition_{loss}.config_hash"
        if cache_file.exists() and sidecar.exists() and not args.force_recompute:
            if sidecar.read_text().strip() == loss_hash:
                print(f"cache hit for loss={loss}; skipping (sidecar matches).")
                continue
        out = decompose(logits, cast("LossName", str(loss)), support)
        np.savez_compressed(
            cache_file,
            A_hat=out["A_hat"],
            E_hat=out["E_hat"],
            H_hat=out["H_hat"],
            indices=indices.astype(np.int64, copy=False),
        )
        sidecar.write_text(loss_hash)
        print(f"wrote {cache_file} (n={logits.shape[0]}, k={k}).")

    _write_meta(run_dir, per_loss_hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
