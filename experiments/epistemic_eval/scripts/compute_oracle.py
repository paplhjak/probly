#!/usr/bin/env python3
"""Stage 4: compute the oracle ``(H*, A*, E*)`` from a known ``p*(y | x)``.

Reads a dataset config (the same one consumed by the rest of the
pipeline) and a ``p*`` matrix on disk, then runs the per-loss oracle
from :mod:`probly.quantification.oracle` for every loss declared in
``dataset_config['supported_losses']``. Outputs are cached as
``oracle_<loss>.npz`` files under
``experiments/epistemic_eval/runs/<oracle_run_id>/``, where
``oracle_run_id = make_run_id("oracle", dataset_name, seed=0)``.

Stage-2 also emits ``p_star.npz`` (a loss-independent sidecar) so
downstream metric computation in ``compute_metrics.py`` doesn't need
to re-derive ``p*`` from dataset configs. The sidecar holds keys
``p_star`` (``float32 (N, K)``), ``support`` (``int64 (K,)``), and
``indices`` (``int64 (N,)``), and is hashed loss-independently
(dataset + support fingerprint + ``_ORACLE_VERSION``).

Loading ``p*``:

* The default path is to read ``--p-star-path`` (a ``.npz`` file
  containing keys ``p_star`` ``(N, K)``, optional ``support`` ``(K,)``,
  optional ``indices`` ``(N,)``). This is what the smoke test uses
  and what the cluster-side loader-driver scripts in Tasks 4/5 will
  emit alongside the test split's images.
* Loading directly from the real dataset class (e.g. ``CIFAR10H``,
  ``ImageNetReaL``, ``AppaReal``) is the cluster-side wiring path and
  is delegated to a sidecar helper script that materialises the same
  ``.npz`` shape. Doing the real loading here would couple the oracle
  layer to the data-on-disk layout, which violates the strict
  three-layer separation in
  ``experiments/epistemic_eval/oracle/README.md``.

Idempotence: the per-loss output is skipped if its cache file exists
with a matching config hash, unless ``--force-recompute`` is passed.
The hash includes the dataset name, the loss string, the contents of
the ``support`` array, and the ``_ORACLE_VERSION`` constant; a math
change in :mod:`probly.quantification.oracle` bumps that constant and
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

from experiments.epistemic_eval.methods._base import (  # noqa: E402
    hash_config,
    make_run_id,
)
from probly.quantification.oracle import (  # noqa: E402
    _ORACLE_VERSION,
    LossName,
    compute_oracle,
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
    meta = {
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "oracle_version": _ORACLE_VERSION,
        "per_loss_hashes": hashes,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _support_fingerprint(support: np.ndarray | None) -> str:
    """Return a short hex digest of ``support``'s contents (or 'none')."""
    if support is None:
        return "none"
    arr = np.asarray(support)
    return hashlib.blake2b(arr.tobytes() + str(arr.dtype).encode(), digest_size=8).hexdigest()


def _p_star_fingerprint(p_star_path: Path) -> str:
    """Return a short hex digest of the input ``p_star.npz``'s file bytes.

    Hashing the raw bytes of the input sidecar makes the oracle cache
    content-addressable: if a caller passes a different ``p_star``
    (e.g. because the DCIC test fold rotates per seed and each seed
    builds its own ``p_star_seed<N>.npz``), the digest changes and
    the cache invalidates. Without this, two seeds with the same
    dataset name but different test sets would silently share an
    oracle.
    """
    return hashlib.blake2b(p_star_path.read_bytes(), digest_size=8).hexdigest()


def _per_loss_hash(
    dataset_name: str,
    loss: str,
    support: np.ndarray | None,
    p_star_fingerprint: str,
) -> str:
    """Stable hash of (dataset, loss, support, p_star, oracle version)."""
    payload = {
        "dataset": dataset_name,
        "loss": loss,
        "support_fingerprint": _support_fingerprint(support),
        "p_star_fingerprint": p_star_fingerprint,
        "oracle_version": int(_ORACLE_VERSION),
    }
    return hash_config(payload)


def _p_star_sidecar_hash(
    dataset_name: str,
    support: np.ndarray | None,
    p_star_fingerprint: str,
) -> str:
    """Loss-independent hash for the ``p_star.npz`` sidecar.

    The sidecar is shared across all losses (it carries only ``p*``,
    the support, and the row indices). Its hash covers the dataset,
    the support, the input ``p_star`` content, and the oracle
    version, so a fresh p_star (different seed -> different test
    fold for DCIC) invalidates the cached sidecar.
    """
    payload = {
        "dataset": dataset_name,
        "support_fingerprint": _support_fingerprint(support),
        "p_star_fingerprint": p_star_fingerprint,
        "oracle_version": int(_ORACLE_VERSION),
    }
    return hash_config(payload)


def _load_p_star_blob(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a ``.npz`` containing ``p_star`` (and optional ``support``, ``indices``)."""
    if not path.exists():
        msg = f"p_star file not found at {path}."
        raise FileNotFoundError(msg)
    blob = np.load(path, allow_pickle=False)
    if "p_star" not in blob.files:
        msg = f"`p_star` key missing from {path}; got keys={blob.files}."
        raise KeyError(msg)
    p_star = np.asarray(blob["p_star"])
    support = np.asarray(blob["support"]) if "support" in blob.files else None
    indices = np.asarray(blob["indices"]) if "indices" in blob.files else None
    return p_star, support, indices


def _resolve_support(
    dataset_config: dict[str, Any],
    blob_support: np.ndarray | None,
    k: int,
) -> np.ndarray | None:
    """Resolve the ``support`` array.

    Priority:
      1. The blob's ``support`` array if present.
      2. ``dataset_config['loader_kwargs']['age_support']`` if present.
      3. ``None`` (only valid for cross-entropy / zero-one).
    """
    if blob_support is not None:
        return blob_support
    loader_kwargs = dataset_config.get("loader_kwargs") or {}
    age_support = loader_kwargs.get("age_support")
    if age_support is not None:
        return np.asarray(age_support)
    classifier = dataset_config.get("classifier") or {}
    metadata = dataset_config.get("metadata") or {}
    num_classes = classifier.get("num_classes") or metadata.get("num_classes")
    if num_classes is not None and isinstance(num_classes, int):
        return np.arange(num_classes, dtype=np.int64) if int(num_classes) == k else None
    return None


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument(
        "--p-star-path",
        type=Path,
        required=True,
        help=".npz with keys 'p_star' (N, K) and optional 'support' (K,), 'indices' (N,).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "experiments" / "epistemic_eval" / "runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Run seed; used to name the oracle run dir "
            "(``<date>_main_oracle_<dataset>_seed<N>``). For datasets "
            "whose test set is fixed across seeds (CIFAR-10H, "
            "ImageNet-ReaL) this is conventionally 0; for DCIC the test "
            "fold rotates per seed via dcic.pick_test_fold so each "
            "seed needs its own oracle run dir."
        ),
    )
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args(argv)

    dataset_config = yaml.safe_load(args.dataset_config.read_text())
    dataset_name = str(dataset_config["name"])
    supported_losses_raw = dataset_config.get("supported_losses")
    if not supported_losses_raw:
        msg = (
            f"dataset config at {args.dataset_config} is missing the required "
            f"'supported_losses' field."
        )
        raise KeyError(msg)
    supported_losses = list(supported_losses_raw)

    p_star, blob_support, indices = _load_p_star_blob(args.p_star_path)
    if p_star.ndim != 2:
        msg = f"p_star must be 2-D, got shape={p_star.shape}."
        raise ValueError(msg)
    n, k = int(p_star.shape[0]), int(p_star.shape[1])
    support = _resolve_support(dataset_config, blob_support, k)
    if indices is None:
        indices = np.arange(n, dtype=np.int64)
    p_star_fingerprint = _p_star_fingerprint(args.p_star_path)

    run_dir = args.output_root / make_run_id(
        "oracle", dataset_name, seed=int(args.seed)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    per_loss_hashes: dict[str, str] = {}
    for loss in supported_losses:
        loss_hash = _per_loss_hash(dataset_name, str(loss), support, p_star_fingerprint)
        per_loss_hashes[loss] = loss_hash
        cache_file = run_dir / f"oracle_{loss}.npz"
        sidecar = run_dir / f"oracle_{loss}.config_hash"
        if cache_file.exists() and sidecar.exists() and not args.force_recompute:
            if sidecar.read_text().strip() == loss_hash:
                print(f"cache hit for loss={loss}; skipping (sidecar matches).")
                continue
        out = compute_oracle(p_star, cast("LossName", str(loss)), support)
        np.savez_compressed(
            cache_file,
            H_star=out["H_star"],
            A_star=out["A_star"],
            E_star=out["E_star"],
            indices=indices.astype(np.int64, copy=False),
        )
        sidecar.write_text(loss_hash)
        print(f"wrote {cache_file} (n={n}, k={k}).")

    # Write the loss-independent p_star sidecar (Task 7). Written once
    # per dataset so downstream metric computation doesn't need to
    # re-derive p* from dataset configs. The hash is loss-independent;
    # it covers dataset_name, support fingerprint, and _ORACLE_VERSION.
    sidecar_hash = _p_star_sidecar_hash(dataset_name, support, p_star_fingerprint)
    sidecar_path = run_dir / "p_star.npz"
    sidecar_hash_path = run_dir / "p_star.config_hash"
    if (
        not sidecar_path.exists()
        or not sidecar_hash_path.exists()
        or sidecar_hash_path.read_text().strip() != sidecar_hash
        or args.force_recompute
    ):
        if support is None:
            sidecar_support = np.arange(k, dtype=np.int64)
        else:
            support_arr = np.asarray(support)
            if support_arr.dtype.kind in {"i", "u"}:
                sidecar_support = support_arr.astype(np.int64, copy=False)
            else:
                # APPA-REAL ages may be stored as float64; cast only when
                # the values are integral. A non-integral support would
                # be a regression task we don't currently target.
                if not np.all(np.equal(np.mod(support_arr, 1.0), 0.0)):
                    msg = (
                        f"support has non-integer values (dtype={support_arr.dtype}); "
                        f"the p_star sidecar requires an integer support."
                    )
                    raise ValueError(msg)
                sidecar_support = support_arr.astype(np.int64, copy=False)
        np.savez_compressed(
            sidecar_path,
            p_star=p_star.astype(np.float32, copy=False),
            support=sidecar_support,
            indices=indices.astype(np.int64, copy=False),
        )
        sidecar_hash_path.write_text(sidecar_hash)
        print(f"wrote {sidecar_path}.")
    else:
        print(f"cache hit for p_star sidecar; skipping (sidecar matches).")

    _write_meta(run_dir, per_loss_hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
