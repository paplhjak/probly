#!/usr/bin/env python3
"""Stage 7: aggregate per-run ``metrics_<loss>.json`` files into a results table.

Walks the runs root, collects every ``metrics_<loss>.json`` produced
by :mod:`compute_metrics`, and emits two artefacts under
``--output-root``:

* ``results.csv`` -- flat, one row per
  ``(method, dataset, loss, seed)`` tuple. Numeric fields are kept
  as-is. Useful for downstream scripting.
* ``results_table.md`` -- pivoted Markdown table grouped by
  ``(dataset, loss)``. Each cell shows ``mean +/- std`` across seeds
  with 4 significant figures and the actual ``n_seeds`` count.

The Markdown sections are sorted alphabetically by
``(dataset, loss)``; rows within a section are sorted by ``AuReC``
mean ascending. Cells with missing seeds are simply reported with
their actual ``n_seeds``; we never pad or warn.

Both outputs are gitignored (see ``.gitignore``) so re-running this
script does not produce VCS churn.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Columns written to ``results.csv`` in this exact order.
_CSV_COLUMNS = (
    "run_id",
    "method",
    "dataset",
    "loss",
    "seed",
    "aurec",
    "excess_aurec",
    "n_aurec",
    "aurc",
    "pareto_gap",
    "n_test_points",
    "config_hash",
    "_metrics_version",
    "_aurec_version",
    "_pareto_gap_version",
    "git_commit",
    "timestamp_utc",
)


def _format_mean_std(values: list[float]) -> str:
    """Format ``mean +/- std`` with 4 significant figures."""
    n = len(values)
    if n == 0:
        return "n/a"
    mean = sum(values) / n
    if n == 1:
        return f"{mean:.4g} +/- n/a"
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = variance**0.5
    return f"{mean:.4g} +/- {std:.4g}"


def _collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    """Walk ``runs_root`` and return the list of metrics records."""
    rows: list[dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for path in sorted(runs_root.rglob("metrics_*.json")):
        record = json.loads(path.read_text())
        rows.append(record)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the flat one-row-per-record CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(_CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _CSV_COLUMNS})


def _group_by_dataset_loss(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group ``rows`` by ``(dataset, loss)``."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("loss", "")))
        out.setdefault(key, []).append(row)
    return out


def _aggregate_method_cell(
    method_rows: list[dict[str, Any]],
) -> tuple[str, str, str, str, str, int]:
    """Return (aurec_str, excess_aurec_str, n_aurec_str, aurc_str, pareto_gap_str, n_seeds)."""
    aurec_values = [float(r["aurec"]) for r in method_rows]
    excess_values = [float(r["excess_aurec"]) for r in method_rows]
    n_aurec_values = [float(r["n_aurec"]) for r in method_rows]
    aurc_values = [float(r["aurc"]) for r in method_rows]
    gap_values = [float(r["pareto_gap"]) for r in method_rows]
    return (
        _format_mean_std(aurec_values),
        _format_mean_std(excess_values),
        _format_mean_std(n_aurec_values),
        _format_mean_std(aurc_values),
        _format_mean_std(gap_values),
        len(method_rows),
    )


def _render_markdown(
    rows: list[dict[str, Any]],
    timestamp_utc: str,
) -> str:
    """Render the pivoted Markdown table."""
    lines = [
        "# Results",
        "",
        f"Generated {timestamp_utc}; {len(rows)} rows.",
        "",
    ]
    grouped = _group_by_dataset_loss(rows)
    for (dataset, loss) in sorted(grouped.keys()):
        section_rows = grouped[(dataset, loss)]
        per_method: dict[str, list[dict[str, Any]]] = {}
        for r in section_rows:
            per_method.setdefault(str(r.get("method", "")), []).append(r)
        method_summaries = []
        for method, m_rows in per_method.items():
            (
                aurec_str,
                excess_str,
                n_aurec_str,
                aurc_str,
                gap_str,
                n_seeds,
            ) = _aggregate_method_cell(m_rows)
            aurec_mean = sum(float(r["aurec"]) for r in m_rows) / len(m_rows)
            method_summaries.append(
                {
                    "method": method,
                    "aurec_str": aurec_str,
                    "excess_str": excess_str,
                    "n_aurec_str": n_aurec_str,
                    "aurc_str": aurc_str,
                    "gap_str": gap_str,
                    "n_seeds": n_seeds,
                    "aurec_mean": aurec_mean,
                }
            )
        method_summaries.sort(key=lambda s: s["aurec_mean"])

        lines.append(f"## {dataset} -- {loss}")
        lines.append("")
        lines.append(
            "| method | AuReC (mean +/- std) | excess AuReC (mean +/- std) | "
            "n_AuReC (mean +/- std) | AuRC (mean +/- std) | "
            "Pareto-gap (mean +/- std) | n_seeds |"
        )
        lines.append(
            "|--------|----------------------|------------------------------|"
            "------------------------|----------------------|"
            "---------------------------|---------|"
        )
        for entry in method_summaries:
            lines.append(
                f"| {entry['method']} | {entry['aurec_str']} | "
                f"{entry['excess_str']} | {entry['n_aurec_str']} | "
                f"{entry['aurc_str']} | {entry['gap_str']} | "
                f"{entry['n_seeds']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_REPO_ROOT / "experiments" / "epistemic_eval" / "runs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "experiments" / "epistemic_eval" / "results",
    )
    args = parser.parse_args(argv)

    rows = _collect_rows(args.runs_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "results.csv"
    md_path = args.output_root / "results_table.md"

    _write_csv(rows, csv_path)
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md_path.write_text(_render_markdown(rows, timestamp_utc))
    print(f"wrote {csv_path} ({len(rows)} rows).")
    print(f"wrote {md_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
