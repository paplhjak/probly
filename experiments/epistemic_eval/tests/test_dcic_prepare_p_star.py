"""Smoke + contract tests for ``scripts/prepare_dcic_p_star.py``.

Builds a synthetic DCIC dataset on disk, drives the prep script via
``subprocess`` once per seed, and asserts:

* The output ``p_star_seed<N>.npz`` has the locked schema
  (``p_star`` ``(N_test, K) float32``, ``support`` ``(K,) int64``,
  ``indices`` ``(N_test,) int64``).
* Rows of ``p_star`` are row-stochastic.
* Different seeds produce different test indices (the per-seed test
  fold rotates per :func:`test_fold_for_seed`).
* The idempotence cache hits when re-run with identical inputs.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")  # noqa: N816

import yaml  # noqa: E402

from experiments.epistemic_eval.tests.conftest import (  # noqa: E402
    make_synthetic_dcic_fixture,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / "experiments"
    / "epistemic_eval"
    / "scripts"
    / "prepare_dcic_p_star.py"
)


def _write_dataset_config(tmp_path: Path, root: Path) -> Path:
    cfg = {
        "name": "plankton",
        "family": "dcic",
        "loader": {"dcic_name": "Plankton"},
        "loader_kwargs": {"root": str(root)},
        "supported_losses": ["cross_entropy", "zero_one"],
        "extraction_mode": "full_network",
    }
    config_path = tmp_path / "plankton.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def _run_script(
    config_path: Path, seed: int, output_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_SCRIPT),
            "--dataset-config",
            str(config_path),
            "--seed",
            str(seed),
            "--output-path",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_p_star_schema_and_shapes(tmp_path: Path) -> None:
    """Per-seed sidecar matches the locked schema."""
    root = make_synthetic_dcic_fixture(
        tmp_path / "data", num_classes=4, images_per_fold=3
    )
    config_path = _write_dataset_config(tmp_path, root)
    output = tmp_path / "p_star_seed0.npz"
    _run_script(config_path, seed=0, output_path=output)

    blob = np.load(output)
    assert {"p_star", "support", "indices"} <= set(blob.files)
    p_star = blob["p_star"]
    assert p_star.dtype == np.float32
    # 1 fold's worth = images_per_fold rows.
    assert p_star.shape[0] == 3
    assert p_star.ndim == 2
    # Row-stochastic.
    sums = p_star.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1.0e-5)
    # support is arange(K).
    np.testing.assert_array_equal(blob["support"], np.arange(p_star.shape[1]))


def test_per_seed_test_indices_rotate(tmp_path: Path) -> None:
    """Different seeds produce different test fold indices."""
    root = make_synthetic_dcic_fixture(
        tmp_path / "data", num_classes=4, images_per_fold=3
    )
    config_path = _write_dataset_config(tmp_path, root)
    seen_indices: set[tuple[int, ...]] = set()
    for seed in range(5):
        output = tmp_path / f"p_star_seed{seed}.npz"
        _run_script(config_path, seed=seed, output_path=output)
        blob = np.load(output)
        seen_indices.add(tuple(blob["indices"].tolist()))
    # Five seeds -> five distinct test fold index sets.
    assert len(seen_indices) == 5


def test_idempotent_cache_hits(tmp_path: Path) -> None:
    """Re-running with identical inputs hits the cache (no rewrite)."""
    root = make_synthetic_dcic_fixture(
        tmp_path / "data", num_classes=4, images_per_fold=3
    )
    config_path = _write_dataset_config(tmp_path, root)
    output = tmp_path / "p_star_seed0.npz"
    _run_script(config_path, seed=0, output_path=output)
    mtime_before = output.stat().st_mtime_ns
    result = _run_script(config_path, seed=0, output_path=output)
    mtime_after = output.stat().st_mtime_ns
    assert mtime_before == mtime_after, "p_star sidecar was rewritten on cache hit"
    assert "skipped" in result.stdout
