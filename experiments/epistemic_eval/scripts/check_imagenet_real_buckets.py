"""Compute ImageNet-ReaL multi-label bucket fractions.

Reads the ``real.json`` file from ``reassessed-imagenet-master`` and
emits a markdown table with the count and fraction of images in each
bucket: empty / single-label / multi-label-2 / multi-label-3 /
multi-label-4-or-more. The empty bucket is what
:class:`experiments.epistemic_eval.scripts.imagenet_real_adapter.ImageNetReaLDropEmpty`
removes per ``decisions.md`` "p*(y | x) construction".

Usage::

    python check_imagenet_real_buckets.py \\
        --real-json <path>/real.json \\
        --output experiments/epistemic_eval/imagenet_real_buckets.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_BUCKET_ORDER = ("empty", "single_label", "multi_2", "multi_3", "multi_4_plus")
_BUCKET_LABELS: dict[str, str] = {
    "empty": "empty",
    "single_label": "single-label",
    "multi_2": "multi-label (2)",
    "multi_3": "multi-label (3)",
    "multi_4_plus": "multi-label (4+)",
}


def compute_buckets(real_labels: list[list[int]]) -> dict[str, int]:
    """Bucket the per-image label sets by cardinality.

    Args:
        real_labels: One label set per image from ``real.json``.

    Returns:
        A dict with keys ``empty``, ``single_label``, ``multi_2``,
        ``multi_3``, ``multi_4_plus`` mapping to integer counts.
    """
    buckets: dict[str, int] = dict.fromkeys(_BUCKET_ORDER, 0)
    for labels in real_labels:
        size = len(labels)
        if size == 0:
            buckets["empty"] += 1
        elif size == 1:
            buckets["single_label"] += 1
        elif size == 2:
            buckets["multi_2"] += 1
        elif size == 3:
            buckets["multi_3"] += 1
        else:
            buckets["multi_4_plus"] += 1
    return buckets


def render_markdown(buckets: dict[str, int]) -> str:
    """Render the bucket counts as a markdown table.

    Args:
        buckets: Output of :func:`compute_buckets`.

    Returns:
        A markdown string identical in structure to the placeholder in
        ``experiments/epistemic_eval/imagenet_real_buckets.md``.
    """
    total = sum(buckets.values())
    lines = [
        "# ImageNet-ReaL bucket fractions",
        "",
        "| Bucket           |  Count  | Fraction |",
        "|------------------|--------:|---------:|",
    ]
    for key in _BUCKET_ORDER:
        count = buckets[key]
        fraction = (100.0 * count / total) if total > 0 else 0.0
        lines.append(f"| {_BUCKET_LABELS[key]:<16s} | {count:>7d} | {fraction:>7.2f}% |")
    lines.append(f"| **total**        | {total:>7d} | 100.00% |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Source: ``real.json`` from")
    lines.append("  ``https://github.com/google-research/reassessed-imagenet``.")
    lines.append("- \"Empty\" entries arise when the relabeling team marked an image as")
    lines.append("  unidentifiable. Per ``decisions.md`` we drop these via")
    lines.append("  ``ImageNetReaLDropEmpty``.")
    lines.append("- \"Multi-label\" entries get uniform mass ``1/k`` over the ``k`` labels")
    lines.append("  in the set, per ``decisions.md`` \"p*(y | x) construction\".")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-json",
        type=pathlib.Path,
        required=True,
        help="Path to real.json from reassessed-imagenet-master.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="If given, write the markdown table here; otherwise print to stdout.",
    )
    args = parser.parse_args(argv)

    real_path = args.real_json.expanduser()
    if not real_path.is_file():
        print(f"--real-json does not exist: {real_path}", file=sys.stderr)
        return 1
    with real_path.open() as handle:
        real_labels = json.load(handle)
    if not isinstance(real_labels, list):
        print(f"Expected a JSON list at {real_path}; got {type(real_labels).__name__}.", file=sys.stderr)
        return 1
    buckets = compute_buckets(real_labels)
    markdown = render_markdown(buckets)

    if args.output is None:
        sys.stdout.write(markdown)
    else:
        out_path = args.output.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown)
        print(f"Wrote bucket report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
