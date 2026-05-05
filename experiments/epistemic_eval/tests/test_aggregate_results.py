"""Tests for ``aggregate_results.py``.

Builds a synthetic ``runs/`` tree of three pre-existing
``metrics_<loss>.json`` files spanning two ``(method, dataset, loss)``
cells (one of which has 2 seeds), drives the aggregator via
``subprocess``, and asserts the output CSV/Markdown have the right
shape.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / "experiments"
    / "epistemic_eval"
    / "scripts"
    / "aggregate_results.py"
)


def _write_metrics_json(
    run_dir: Path,
    *,
    method: str,
    dataset: str,
    loss: str,
    seed: int,
    aurec: float,
    aurc: float,
    pareto_gap: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_dir.name,
        "method": method,
        "dataset": dataset,
        "loss": loss,
        "seed": seed,
        "aurec": aurec,
        # ``excess_aurec`` and ``n_aurec`` are written by every
        # post-eac4a24b ``compute_metrics`` invocation; the aggregator
        # reads them, so the fixture must populate them too. Use
        # plausible values derived from ``aurec`` to keep the synthetic
        # numbers internally consistent.
        "excess_aurec": aurec * 0.5,
        "n_aurec": aurec * 5.0,
        "aurc": aurc,
        "pareto_gap": pareto_gap,
        "n_test_points": 100,
        "config_hash": "deadbeef00000000",
        "_metrics_version": 1,
        "_aurec_version": 1,
        "_pareto_gap_version": 1,
        "git_commit": "abcdef0",
        "timestamp_utc": "2026-05-03T00:00:00+00:00",
    }
    (run_dir / f"metrics_{loss}.json").write_text(json.dumps(payload))


def _run_aggregate(runs_root: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_SCRIPT),
            "--runs-root",
            str(runs_root),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_aggregate_results_csv_has_three_rows_and_md_has_both_sections(
    tmp_path: Path,
) -> None:
    """The aggregator emits three CSV rows and two Markdown sections."""
    runs_root = tmp_path / "runs"
    output_root = tmp_path / "results"

    _write_metrics_json(
        runs_root / "run_a",
        method="mc_dropout",
        dataset="appa_real",
        loss="squared",
        seed=0,
        aurec=0.10,
        aurc=0.50,
        pareto_gap=0.20,
    )
    _write_metrics_json(
        runs_root / "run_b",
        method="mc_dropout",
        dataset="appa_real",
        loss="squared",
        seed=1,
        aurec=0.12,
        aurc=0.55,
        pareto_gap=0.22,
    )
    _write_metrics_json(
        runs_root / "run_c",
        method="ensemble",
        dataset="cifar10h",
        loss="cross_entropy",
        seed=0,
        aurec=0.05,
        aurc=0.30,
        pareto_gap=0.10,
    )

    _run_aggregate(runs_root, output_root)

    csv_path = output_root / "results.csv"
    md_path = output_root / "results_table.md"
    assert csv_path.exists(), csv_path
    assert md_path.exists(), md_path

    # CSV: header + 3 data rows.
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    assert len(rows) == 3, rows
    methods_per_dataset = {(row["method"], row["dataset"], row["loss"]) for row in rows}
    assert methods_per_dataset == {
        ("mc_dropout", "appa_real", "squared"),
        ("ensemble", "cifar10h", "cross_entropy"),
    }

    md_text = md_path.read_text()
    assert "## appa_real -- squared" in md_text
    assert "## cifar10h -- cross_entropy" in md_text


def test_aggregate_results_n_seeds_per_cell(tmp_path: Path) -> None:
    """The (mc_dropout, appa_real, squared) cell reports n_seeds=2."""
    runs_root = tmp_path / "runs"
    output_root = tmp_path / "results"

    _write_metrics_json(
        runs_root / "run_a",
        method="mc_dropout",
        dataset="appa_real",
        loss="squared",
        seed=0,
        aurec=0.10,
        aurc=0.50,
        pareto_gap=0.20,
    )
    _write_metrics_json(
        runs_root / "run_b",
        method="mc_dropout",
        dataset="appa_real",
        loss="squared",
        seed=1,
        aurec=0.12,
        aurc=0.55,
        pareto_gap=0.22,
    )

    _run_aggregate(runs_root, output_root)
    md_text = (output_root / "results_table.md").read_text()
    # The mc_dropout row should have n_seeds=2 in its trailing column.
    matched = [line for line in md_text.splitlines() if "mc_dropout" in line]
    assert matched, md_text
    assert matched[0].rstrip().endswith("| 2 |")
