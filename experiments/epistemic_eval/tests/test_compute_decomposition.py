"""Smoke test for ``experiments/epistemic_eval/scripts/compute_decomposition.py``.

Builds a synthetic ``(N=8, K=4, S=3)`` predictions blob, drives the
script via ``subprocess`` against a synthetic dataset config that
declares all four supported losses, and asserts each per-loss output
file has the documented shape contract.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import yaml  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts" / "compute_decomposition.py"
)


def _write_dataset_config(tmp_path: Path, supported_losses: list[str]) -> Path:
    cfg = {
        "name": "synthetic_decomposition_smoke",
        "loader": {
            "module": "experiments.epistemic_eval.tests.fixture",
            "class": "Synthetic",
        },
        "loader_kwargs": {
            "age_support": [20, 30, 40, 50],
        },
        "supported_losses": supported_losses,
        "extraction_mode": "linear_probe",
        "metadata": {"num_classes": 4},
    }
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _write_predictions(tmp_path: Path) -> Path:
    """Write a synthetic ``predictions.npz`` with non-degenerate logits."""
    rng = np.random.default_rng(123)
    logits = rng.standard_normal(size=(8, 4, 3)).astype(np.float32)
    indices = np.arange(8, dtype=np.int64)
    run_dir = tmp_path / "runs" / "synthetic_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", logits=logits, indices=indices)
    return run_dir


def _run_script(run_dir: Path, dataset_cfg: Path) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_SCRIPT),
            "--run",
            str(run_dir),
            "--dataset-config",
            str(dataset_cfg),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = (
            f"compute_decomposition.py failed: stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}"
        )
        raise AssertionError(msg)


def test_compute_decomposition_writes_all_four_losses(tmp_path: Path) -> None:
    """All four losses produce per-loss caches with the right shape contract."""
    losses = ["cross_entropy", "zero_one", "squared", "absolute"]
    dataset_cfg = _write_dataset_config(tmp_path, supported_losses=losses)
    run_dir = _write_predictions(tmp_path)
    _run_script(run_dir, dataset_cfg)

    n = 8
    for loss in losses:
        cache_file = run_dir / f"decomposition_{loss}.npz"
        assert cache_file.exists(), cache_file
        blob = np.load(cache_file)
        assert {"A_hat", "E_hat", "H_hat", "indices"} <= set(blob.files)
        assert blob["A_hat"].shape == (n,)
        assert blob["E_hat"].shape == (n,)
        if loss == "cross_entropy":
            assert blob["H_hat"].shape == (n, 4)
        elif loss == "zero_one":
            assert blob["H_hat"].shape == (n,)
            assert blob["H_hat"].dtype == np.int64
        else:  # squared, absolute
            assert blob["H_hat"].shape == (n,)


def test_compute_decomposition_is_idempotent(tmp_path: Path) -> None:
    """Re-running with identical inputs hits the cache (no rewrite)."""
    losses = ["squared"]
    dataset_cfg = _write_dataset_config(tmp_path, supported_losses=losses)
    run_dir = _write_predictions(tmp_path)
    _run_script(run_dir, dataset_cfg)
    cache_file = run_dir / "decomposition_squared.npz"
    mtime_before = cache_file.stat().st_mtime_ns
    _run_script(run_dir, dataset_cfg)
    mtime_after = cache_file.stat().st_mtime_ns
    assert mtime_after == mtime_before
