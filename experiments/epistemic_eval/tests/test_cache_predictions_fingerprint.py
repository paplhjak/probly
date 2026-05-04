"""Pin: per-loss cache hashes invalidate when predictions.npz changes.

Pre-fix, both ``compute_decomposition.py``'s ``_per_loss_hash`` and
``compute_metrics.py``'s ``_config_hash`` covered only the
(dataset, loss, support, schema, method, seed) metadata. If a run
dir's ``predictions.npz`` was overwritten in place -- the
dryrun-then-production overwrite pattern -- the metadata was
unchanged so the sidecar matched and stale decomposition + metrics
were silently kept.

The dryrun-then-production overwrite was real: in the
2026-05-03 SLURM grid, run dirs ended up with predictions.npz at
01:14 (production) and decomposition_*.npz / metrics_*.json at
22:02 / 23:22 from the prior day's dryrun, with all sidecars
matching.

These tests run each script twice, mutating ``predictions.npz``
between runs while keeping every other input fixed, and assert the
output (and its sidecar) is rewritten the second time.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DECOMP_SCRIPT = (
    _REPO_ROOT
    / "experiments"
    / "epistemic_eval"
    / "scripts"
    / "compute_decomposition.py"
)
_METRICS_SCRIPT = (
    _REPO_ROOT
    / "experiments"
    / "epistemic_eval"
    / "scripts"
    / "compute_metrics.py"
)


def _write_decomposition_inputs(
    tmp_path: Path,
    rng_seed: int,
) -> tuple[Path, Path]:
    """Create a synthetic run dir + dataset config for compute_decomposition."""
    rng = np.random.default_rng(rng_seed)
    logits = rng.standard_normal(size=(8, 4, 3)).astype(np.float32)
    indices = np.arange(8, dtype=np.int64)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", logits=logits, indices=indices)

    dataset_cfg = {
        "name": "synthetic_fingerprint",
        "loader_kwargs": {"age_support": [20, 30, 40, 50]},
        "supported_losses": ["squared"],
        "extraction_mode": "linear_probe",
        "metadata": {"num_classes": 4},
    }
    cfg_path = tmp_path / "dataset.yaml"
    cfg_path.write_text(yaml.safe_dump(dataset_cfg))
    return run_dir, cfg_path


def _run_decomp(run_dir: Path, dataset_cfg: Path) -> None:
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_DECOMP_SCRIPT),
            "--run",
            str(run_dir),
            "--dataset-config",
            str(dataset_cfg),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_compute_decomposition_invalidates_cache_when_predictions_change(
    tmp_path: Path,
) -> None:
    """Overwriting predictions.npz with new content rewrites the cache.

    Without the predictions fingerprint in the per-loss hash, the
    second run would short-circuit on a sidecar match and leave the
    stale ``decomposition_squared.npz`` in place. With the fingerprint,
    the hash changes and the file is rewritten with the new contents.
    """
    run_dir, dataset_cfg = _write_decomposition_inputs(tmp_path, rng_seed=1)
    _run_decomp(run_dir, dataset_cfg)
    cache_file = run_dir / "decomposition_squared.npz"
    sidecar = run_dir / "decomposition_squared.config_hash"
    assert cache_file.exists()
    first_hash = sidecar.read_text().strip()
    first_blob = np.load(cache_file)
    first_a_hat = first_blob["A_hat"].copy()

    # Overwrite predictions.npz with materially different content. Same
    # shape, schema, dataset config -- only the bytes differ.
    rng2 = np.random.default_rng(99999)
    new_logits = rng2.standard_normal(size=(8, 4, 3)).astype(np.float32)
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=new_logits,
        indices=np.arange(8, dtype=np.int64),
    )

    # mtime resolution: ensure the next write registers as later.
    time.sleep(0.01)
    _run_decomp(run_dir, dataset_cfg)
    second_hash = sidecar.read_text().strip()
    second_blob = np.load(cache_file)
    second_a_hat = second_blob["A_hat"]

    assert second_hash != first_hash, (
        "_per_loss_hash should depend on predictions.npz contents; "
        "got identical hashes for two different prediction blobs."
    )
    # The decomposition is a deterministic function of predictions, so
    # changed predictions must produce a changed A_hat.
    assert not np.array_equal(first_a_hat, second_a_hat), (
        "decomposition cache appears stale after predictions changed."
    )


def _write_metrics_inputs(
    tmp_path: Path,
    loss: str,
    rng_seed: int,
) -> tuple[Path, Path]:
    """Create a synthetic run dir + oracle dir for compute_metrics."""
    rng = np.random.default_rng(rng_seed)
    n, k, s = 10, 4, 3
    run_dir = tmp_path / f"run_{loss}"
    oracle_dir = tmp_path / f"oracle_{loss}"
    run_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    logits = rng.standard_normal(size=(n, k, s)).astype(np.float32)
    indices = np.arange(n, dtype=np.int64)
    np.savez_compressed(
        run_dir / "predictions.npz", logits=logits, indices=indices
    )
    a_hat = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    e_hat = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    np.savez_compressed(
        run_dir / f"decomposition_{loss}.npz",
        A_hat=a_hat,
        E_hat=e_hat,
        H_hat=np.zeros(n, dtype=np.float32),
        indices=indices,
    )
    a_star = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    np.savez_compressed(
        oracle_dir / f"oracle_{loss}.npz",
        H_star=np.zeros(n, dtype=np.float32),
        A_star=a_star,
        E_star=np.zeros(n, dtype=np.float32),
        indices=indices,
    )
    p_star = rng.dirichlet(np.ones(k), size=n).astype(np.float32)
    np.savez_compressed(
        oracle_dir / "p_star.npz",
        p_star=p_star,
        support=np.arange(k, dtype=np.int64),
        indices=indices,
    )
    config = {
        "method": {"name": "synthetic_method"},
        "dataset": {"name": "synthetic_dataset"},
        "seed": 0,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return run_dir, oracle_dir


def _run_metrics(run_dir: Path, oracle_dir: Path, loss: str) -> None:
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_METRICS_SCRIPT),
            "--run",
            str(run_dir),
            "--loss",
            loss,
            "--oracle-run",
            str(oracle_dir),
            "--p-star-path",
            str(oracle_dir / "p_star.npz"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_compute_metrics_invalidates_cache_when_predictions_change(
    tmp_path: Path,
) -> None:
    """Overwriting predictions.npz with new content rewrites metrics_<loss>.json.

    Without the predictions fingerprint in the metrics ``_config_hash``,
    the second run would skip the rewrite via the sidecar match and
    leave a stale ``metrics_cross_entropy.json`` next to a fresh
    predictions.npz -- the dryrun → production poisoning scenario.
    """
    loss = "cross_entropy"
    run_dir, oracle_dir = _write_metrics_inputs(tmp_path, loss, rng_seed=1)
    _run_metrics(run_dir, oracle_dir, loss)
    sidecar = run_dir / f"metrics_{loss}.config_hash"
    metrics_path = run_dir / f"metrics_{loss}.json"
    assert metrics_path.exists()
    first_hash = sidecar.read_text().strip()
    mtime_before = metrics_path.stat().st_mtime_ns

    # Overwrite predictions.npz with new content; everything else
    # (decomposition, oracle, p_star, config, seed, schema) stays put.
    rng2 = np.random.default_rng(8675309)
    n, k, s = 10, 4, 3
    new_logits = rng2.standard_normal(size=(n, k, s)).astype(np.float32)
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=new_logits,
        indices=np.arange(n, dtype=np.int64),
    )

    time.sleep(0.01)
    _run_metrics(run_dir, oracle_dir, loss)
    second_hash = sidecar.read_text().strip()
    mtime_after = metrics_path.stat().st_mtime_ns

    assert second_hash != first_hash, (
        "_config_hash should depend on predictions.npz contents; "
        "got identical hashes for two different prediction blobs."
    )
    assert mtime_after > mtime_before, (
        "metrics JSON was not rewritten after predictions.npz changed; "
        "the cache is poisoned."
    )
