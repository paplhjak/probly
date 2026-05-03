"""Smoke test for ``experiments/epistemic_eval/scripts/compute_oracle.py``.

Builds a synthetic ``(N=5, K=4)`` ``p_star`` matrix, drives the script
via ``subprocess``, and asserts each per-loss output file has the
documented shape contract.

E_star is identically zero on a first-order dataset for every loss
(per Definition 1 of paper_tex/sections/preliminaries.tex and the
"Frequentist Paradox" passage at lines 41-42).
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
_SCRIPT = _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts" / "compute_oracle.py"


def _write_dataset_config(tmp_path: Path, supported_losses: list[str]) -> Path:
    cfg = {
        "name": "synthetic_oracle_smoke",
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


def _write_p_star(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    p_star = rng.dirichlet(np.ones(4), size=5).astype(np.float64)
    support = np.array([20, 30, 40, 50], dtype=np.int64)
    indices = np.arange(5, dtype=np.int64)
    out = tmp_path / "p_star.npz"
    np.savez_compressed(out, p_star=p_star, support=support, indices=indices)
    return out


def _run_script(dataset_cfg: Path, p_star_path: Path, output_root: Path) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_SCRIPT),
            "--dataset-config",
            str(dataset_cfg),
            "--p-star-path",
            str(p_star_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = f"compute_oracle.py failed: stdout={result.stdout!r}, stderr={result.stderr!r}"
        raise AssertionError(msg)


def test_compute_oracle_writes_per_loss_npz(tmp_path: Path) -> None:
    """All four losses produce a populated cache file with the right keys."""
    losses = ["cross_entropy", "zero_one", "squared", "absolute"]
    dataset_cfg = _write_dataset_config(tmp_path, supported_losses=losses)
    p_star_path = _write_p_star(tmp_path)
    output_root = tmp_path / "runs"
    _run_script(dataset_cfg, p_star_path, output_root)

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    for loss in losses:
        cache_file = run_dir / f"oracle_{loss}.npz"
        assert cache_file.exists(), cache_file
        sidecar = run_dir / f"oracle_{loss}.config_hash"
        assert sidecar.exists()
        blob = np.load(cache_file)
        assert {"H_star", "A_star", "E_star", "indices"} <= set(blob.files)
        n = 5
        if loss == "cross_entropy":
            assert blob["H_star"].shape == (n, 4)
        elif loss == "zero_one":
            assert blob["H_star"].shape == (n,)
            assert blob["H_star"].dtype == np.int64
        elif loss == "squared":
            assert blob["H_star"].shape == (n,)
        else:  # absolute
            assert blob["H_star"].shape == (n,)
        assert blob["A_star"].shape == (n,)
        assert blob["E_star"].shape == (n,)
        # E_star is identically zero on a first-order dataset.
        assert np.all(blob["E_star"] == 0.0)

    meta_path = run_dir / "meta.json"
    assert meta_path.exists()


def test_compute_oracle_is_idempotent(tmp_path: Path) -> None:
    """Re-running with identical inputs hits the cache (no rewrite)."""
    losses = ["squared"]
    dataset_cfg = _write_dataset_config(tmp_path, supported_losses=losses)
    p_star_path = _write_p_star(tmp_path)
    output_root = tmp_path / "runs"
    _run_script(dataset_cfg, p_star_path, output_root)

    run_dir = next(iter(output_root.iterdir()))
    cache_file = run_dir / "oracle_squared.npz"
    mtime_before = cache_file.stat().st_mtime_ns

    # Sleep is unnecessary; we directly compare nanosecond mtimes.
    _run_script(dataset_cfg, p_star_path, output_root)
    mtime_after = cache_file.stat().st_mtime_ns
    assert mtime_after == mtime_before, "cache file was rewritten despite a hash hit"
