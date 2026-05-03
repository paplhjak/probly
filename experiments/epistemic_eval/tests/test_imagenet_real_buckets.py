"""Tests for :mod:`check_imagenet_real_buckets`."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

# conftest.py prepends experiments/epistemic_eval/scripts/ to sys.path.
import check_imagenet_real_buckets as buckets_module  # noqa: E402


def test_compute_buckets_partitions_by_set_cardinality() -> None:
    real_labels = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [42]]
    counts = buckets_module.compute_buckets(real_labels)
    assert counts == {
        "empty": 1,
        "single_label": 2,
        "multi_2": 1,
        "multi_3": 1,
        "multi_4_plus": 1,
    }


def test_render_markdown_contains_counts_and_percentages() -> None:
    real_labels = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [42]]
    counts = buckets_module.compute_buckets(real_labels)
    md = buckets_module.render_markdown(counts)
    # Counts present.
    assert "| empty            |       1 |" in md
    assert "| single-label     |       2 |" in md
    assert "| multi-label (2)  |       1 |" in md
    assert "| multi-label (3)  |       1 |" in md
    assert "| multi-label (4+) |       1 |" in md
    # Total and the canonical percentage formatting.
    assert "| **total**        |       6 | 100.00% |" in md
    # Single-label row reports 33.33% of 6.
    assert "33.33%" in md


def test_cli_writes_markdown_output(tmp_path: Path) -> None:
    real_path = tmp_path / "real.json"
    real_path.write_text(json.dumps([[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [42]]))
    out_path = tmp_path / "out.md"

    saved_argv = sys.argv
    sys.argv = [
        "check_imagenet_real_buckets.py",
        "--real-json",
        str(real_path),
        "--output",
        str(out_path),
    ]
    try:
        # Use runpy to exercise the CLI in-process.
        try:
            runpy.run_module("check_imagenet_real_buckets", run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (0, None)
    finally:
        sys.argv = saved_argv

    text = out_path.read_text()
    assert "ImageNet-ReaL bucket fractions" in text
    assert "| **total**        |       6 | 100.00% |" in text
    assert "| empty            |       1 |" in text
