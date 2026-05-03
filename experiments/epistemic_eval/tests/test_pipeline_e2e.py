"""End-to-end keystone test: synthetic predictions through metrics + aggregation.

For each of the four supported losses, this test materialises the
artefacts that the upstream stages of the pipeline (extract,
decompose, oracle) would have written; calls ``compute_metrics.main``
in-process to produce a ``metrics_<loss>.json``; calls
``aggregate_results.main`` to roll all four files into
``results.csv`` and ``results_table.md``; and asserts the locked
schema and shape contracts.

Sizing: ``N=12``, ``K=4``, ``S=3`` with default selector grid sizes
(91 directions, 51 lambdas inside ``pareto_gap``). Per the Task 7
brief's runtime budget (<5 s wall), the test invokes
``compute_metrics.main`` and ``aggregate_results.main`` *in-process*
rather than via ``subprocess``: the bottleneck is Python interpreter
startup (~1.7 s per spawn x 5 spawns = ~8.5 s), not the compute
itself (which finishes in <50 ms per loss). Calling ``main()``
in-process bypasses startup while still exercising the same code
path the cluster shell scripts use (``compute_metrics.py``'s
``__main__`` guard wraps ``main(argv=sys.argv[1:])``).

Runtime budget: <5 s wall on the development machine. If exceeded
on slower machines, downsize per Task 7's R3.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import aggregate_results  # noqa: E402
import compute_metrics  # noqa: E402


def _materialise_run(
    tmp_path: Path,
    loss: str,
    seed: int,
    n: int,
    k: int,
    s: int,
) -> tuple[Path, Path]:
    """Write the synthetic upstream artefacts for one (loss, seed) cell."""
    rng = np.random.default_rng(7 + hash(loss) % 1000 + seed)
    run_dir = tmp_path / "runs" / f"run_{loss}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = tmp_path / "oracles" / f"oracle_{loss}"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    logits = rng.standard_normal(size=(n, k, s)).astype(np.float32)
    indices = np.arange(n, dtype=np.int64)
    np.savez_compressed(
        run_dir / "predictions.npz", logits=logits, indices=indices
    )

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

    p_star = rng.dirichlet(np.ones(k), size=n).astype(np.float32)
    support = np.arange(k, dtype=np.int64)
    np.savez_compressed(
        oracle_dir / "p_star.npz",
        p_star=p_star,
        support=support,
        indices=indices,
    )

    config = {
        "method": {"name": "synthetic_method"},
        "dataset": {"name": "synthetic_dataset"},
        "seed": int(seed),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return run_dir, oracle_dir


def test_pipeline_e2e_all_four_losses(tmp_path: Path) -> None:
    """End-to-end: synthetic predictions through metrics and aggregation for all four losses.

    Asserts:

    * ``results.csv`` has exactly 4 rows (one per loss).
    * Every numeric metric (``aurec``, ``aurc``, ``pareto_gap``) is
      finite and non-negative.
    * ``results_table.md`` contains the four ``(dataset, loss)``
      sections (synthetic_dataset crossed with the four losses).

    Runtime budget: <5 s wall on the development machine; if exceeded,
    downsize per Task 7's R3.
    """
    n, k, s = 12, 4, 3
    losses = ["cross_entropy", "zero_one", "squared", "absolute"]
    seed = 0

    for loss in losses:
        run_dir, oracle_dir = _materialise_run(tmp_path, loss, seed, n, k, s)
        rc = compute_metrics.main(
            [
                "--run",
                str(run_dir),
                "--loss",
                loss,
                "--oracle-run",
                str(oracle_dir),
                "--p-star-path",
                str(oracle_dir / "p_star.npz"),
            ]
        )
        assert rc == 0, (loss, rc)

    output_root = tmp_path / "results"
    rc = aggregate_results.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(output_root),
        ]
    )
    assert rc == 0, rc

    csv_path = output_root / "results.csv"
    md_path = output_root / "results_table.md"
    assert csv_path.exists(), csv_path
    assert md_path.exists(), md_path

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    assert len(rows) == 4, rows
    seen_losses = {row["loss"] for row in rows}
    assert seen_losses == set(losses)

    for row in rows:
        for key in ("aurec", "aurc", "pareto_gap"):
            value = float(row[key])
            assert np.isfinite(value), (key, row)
            assert value >= 0.0, (key, row)

    md_text = md_path.read_text()
    for loss in losses:
        assert f"## synthetic_dataset -- {loss}" in md_text, (loss, md_text)

    # Spot-check: the metrics JSON is also reachable for a developer
    # who wants to inspect a single cell.
    sample_path = (
        tmp_path
        / "runs"
        / f"run_cross_entropy_seed{seed}"
        / "metrics_cross_entropy.json"
    )
    assert sample_path.exists()
    payload = json.loads(sample_path.read_text())
    assert payload["loss"] == "cross_entropy"
    assert payload["n_test_points"] == n
