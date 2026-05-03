"""Smoke and contract tests for ``compute_metrics.py``.

For each loss the test builds a synthetic ``(N=10, K=4, S=3)``
predictions blob, the matching decomposition cache, the loss-specific
oracle cache, and a ``p_star.npz`` sidecar. It then drives
``compute_metrics.py`` via ``subprocess`` and asserts the resulting
``metrics_<loss>.json`` has the locked schema with finite numeric
fields.

Idempotence and ``--force-recompute`` are verified separately.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts" / "compute_metrics.py"
)


def _write_synthetic_run(
    tmp_path: Path,
    loss: str,
    seed: int = 0,
    n: int = 10,
    k: int = 4,
    s: int = 3,
) -> tuple[Path, Path]:
    """Write a synthetic run dir + matching oracle dir under ``tmp_path``."""
    rng = np.random.default_rng(123 + hash(loss) % 1000)
    run_dir = tmp_path / f"run_{loss}"
    run_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = tmp_path / f"oracle_{loss}"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    # predictions.npz: random logits, sequential indices.
    logits = rng.standard_normal(size=(n, k, s)).astype(np.float32)
    indices = np.arange(n, dtype=np.int64)
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=logits,
        indices=indices,
    )

    # decomposition_<loss>.npz: A_hat, E_hat, H_hat (shape doesn't matter
    # since compute_metrics ignores H_hat).
    a_hat = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    e_hat = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    h_hat = np.zeros(n, dtype=np.float32)
    np.savez_compressed(
        run_dir / f"decomposition_{loss}.npz",
        A_hat=a_hat,
        E_hat=e_hat,
        H_hat=h_hat,
        indices=indices,
    )

    # oracle_<loss>.npz: A_star, E_star, H_star (H_star ignored by
    # compute_metrics; we still write it for schema parity).
    a_star = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    e_star = np.zeros(n, dtype=np.float32)
    h_star_value = np.zeros(n, dtype=np.float32)
    np.savez_compressed(
        oracle_dir / f"oracle_{loss}.npz",
        H_star=h_star_value,
        A_star=a_star,
        E_star=e_star,
        indices=indices,
    )

    # p_star.npz sidecar.
    p_star = rng.dirichlet(np.ones(k), size=n).astype(np.float32)
    support = np.arange(k, dtype=np.int64)
    np.savez_compressed(
        oracle_dir / "p_star.npz",
        p_star=p_star,
        support=support,
        indices=indices,
    )

    # config.yaml -- written by stage 2b in the real pipeline.
    config = {
        "method": {"name": "synthetic_method"},
        "dataset": {"name": "synthetic_dataset"},
        "seed": int(seed),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))

    return run_dir, oracle_dir


def _run_compute_metrics(
    run_dir: Path,
    oracle_dir: Path,
    loss: str,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    """Drive ``compute_metrics.py`` via subprocess."""
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_SCRIPT),
            "--run",
            str(run_dir),
            "--loss",
            loss,
            "--oracle-run",
            str(oracle_dir),
            "--p-star-path",
            str(oracle_dir / "p_star.npz"),
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "loss", ["cross_entropy", "zero_one", "squared", "absolute"]
)
def test_compute_metrics_writes_locked_schema_per_loss(
    tmp_path: Path, loss: str
) -> None:
    """All four losses produce a metrics JSON with the locked keys and finite numerics."""
    run_dir, oracle_dir = _write_synthetic_run(tmp_path, loss)
    _run_compute_metrics(run_dir, oracle_dir, loss)

    metrics_path = run_dir / f"metrics_{loss}.json"
    assert metrics_path.exists(), metrics_path
    payload = json.loads(metrics_path.read_text())

    expected_keys = {
        "run_id",
        "method",
        "dataset",
        "loss",
        "seed",
        "aurec",
        "aurc",
        "pareto_gap",
        "n_test_points",
        "config_hash",
        "_metrics_version",
        "_aurec_version",
        "_pareto_gap_version",
        "git_commit",
        "timestamp_utc",
    }
    assert expected_keys <= set(payload.keys()), payload

    # Finiteness on numeric fields.
    for key in ("aurec", "aurc", "pareto_gap"):
        assert isinstance(payload[key], float)
        assert np.isfinite(payload[key]), key

    assert payload["loss"] == loss
    assert payload["method"] == "synthetic_method"
    assert payload["dataset"] == "synthetic_dataset"
    assert payload["seed"] == 0
    assert payload["n_test_points"] == 10
    assert len(payload["config_hash"]) == 16

    sidecar = run_dir / f"metrics_{loss}.config_hash"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == payload["config_hash"]


def test_compute_metrics_is_idempotent(tmp_path: Path) -> None:
    """Re-running with identical inputs hits the cache (no rewrite)."""
    run_dir, oracle_dir = _write_synthetic_run(tmp_path, "squared")
    _run_compute_metrics(run_dir, oracle_dir, "squared")
    metrics_path = run_dir / "metrics_squared.json"
    mtime_before = metrics_path.stat().st_mtime_ns
    # Second run must skip the rewrite via sidecar match.
    _run_compute_metrics(run_dir, oracle_dir, "squared")
    mtime_after = metrics_path.stat().st_mtime_ns
    assert mtime_after == mtime_before, "metrics JSON was rewritten despite a hash hit"


def test_compute_metrics_force_recompute_overwrites(tmp_path: Path) -> None:
    """Passing ``--force-recompute`` rewrites the JSON even on cache hit."""
    run_dir, oracle_dir = _write_synthetic_run(tmp_path, "cross_entropy")
    _run_compute_metrics(run_dir, oracle_dir, "cross_entropy")
    metrics_path = run_dir / "metrics_cross_entropy.json"
    mtime_before = metrics_path.stat().st_mtime_ns
    # Sleep briefly so mtime resolution can register the rewrite.
    time.sleep(0.01)
    _run_compute_metrics(
        run_dir, oracle_dir, "cross_entropy", "--force-recompute"
    )
    mtime_after = metrics_path.stat().st_mtime_ns
    assert mtime_after > mtime_before, (
        "metrics JSON should be rewritten under --force-recompute"
    )
