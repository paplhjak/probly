#!/usr/bin/env python3
"""Build the APPA-REAL ``p_star.npz`` sidecar from the per-rater CSV.

Reads ``<root>/gt_<split>.csv`` (the per-rater file shipped with the
CVPR 2017 release; one row per (image, rater) with columns
``file_name, real_age, apparent_age, worker_age, worker_gender``)
and emits

    <root>/p_star_<split>.npz

with::

    p_star:  (N_test, 101) float32  -- row-stochastic conditional
                                       p*(age | image), columns indexed
                                       by the locked integer-age
                                       support [0..100].
    support: (101,)        int64    -- np.arange(101).
    indices: (N_test,)     int64    -- np.arange(N_test); the per-row
                                       dataset index in the loader's
                                       iteration order.

Construction per ``decisions.md`` -> "p*(y | x) construction": the
empirical apparent-age vote distribution per image, normalised to
sum to 1. The integer-age support is pinned to [0..100] (the union
of votes across all three splits, verified empirically).

Idempotent by default: if ``p_star_<split>.npz`` already exists with
the correct shape AND its values match what we would produce now to
``atol=1e-12``, the script prints ``skipped: <path> exists`` and
exits 0. Pass ``--force`` to overwrite unconditionally.

Schema invariants asserted at write time:
    * ``p_star.shape == (N_test, 101)``
    * row sums are within ``1e-6`` of 1.0
    * no NaN, no Inf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Locked integer-age support for APPA-REAL -- every age 0..100 inclusive
#: appears at least once across train+valid+test apparent-age votes
#: (verified 2026-05-04). Mirrors ``configs/datasets/appa_real.yaml``'s
#: ``loader_kwargs.age_support``.
_AGE_SUPPORT: tuple[int, ...] = tuple(range(101))

_ROW_SUM_ATOL = 1.0e-6


def build_p_star(csv_path: Path) -> dict[str, np.ndarray]:
    """Compute ``p_star`` over the apparent-age votes in ``csv_path``.

    Args:
        csv_path: Path to one of ``gt_<split>.csv`` shipped with the
            APPA-REAL release.

    Returns:
        Dict with keys ``p_star`` ``(N_test, 101) float32``, ``support``
        ``(101,) int64``, ``indices`` ``(N_test,) int64``.

    Raises:
        ValueError: If a vote falls outside ``[0, 100]``, the CSV is
            missing the expected columns, or any row fails to sum to 1
            within ``1e-6``.
    """
    if not csv_path.is_file():
        msg = f"per-rater CSV not found at {csv_path}."
        raise FileNotFoundError(msg)
    df = pd.read_csv(csv_path)
    required = {"file_name", "apparent_age"}
    missing = required - set(df.columns)
    if missing:
        msg = (
            f"CSV at {csv_path} is missing required columns: "
            f"{sorted(missing)!r}. Got: {list(df.columns)!r}."
        )
        raise ValueError(msg)
    ages = pd.to_numeric(df["apparent_age"], errors="coerce")
    if ages.isna().any():
        msg = f"CSV at {csv_path} contains non-numeric apparent_age values."
        raise ValueError(msg)
    ages_int = ages.astype(int)
    if not (ages == ages_int).all():
        msg = f"CSV at {csv_path} contains non-integer apparent_age values."
        raise ValueError(msg)
    if (ages_int < _AGE_SUPPORT[0]).any() or (ages_int > _AGE_SUPPORT[-1]).any():
        bad = df[(ages_int < _AGE_SUPPORT[0]) | (ages_int > _AGE_SUPPORT[-1])].head(5)
        msg = (
            f"CSV at {csv_path} has apparent_age outside the locked "
            f"support [{_AGE_SUPPORT[0]}, {_AGE_SUPPORT[-1]}]: "
            f"{bad.to_dict(orient='records')}."
        )
        raise ValueError(msg)

    # Stable iteration order: rows of p_star are in the order each
    # image first appears in the CSV (matching what AppaReal's
    # __init__ uses when it builds image_filenames).
    filenames_in_order: list[str] = []
    seen: set[str] = set()
    for f in df["file_name"].tolist():
        f_str = str(f)
        if f_str not in seen:
            seen.add(f_str)
            filenames_in_order.append(f_str)

    n = len(filenames_in_order)
    k = len(_AGE_SUPPORT)
    counts = np.zeros((n, k), dtype=np.int64)
    filename_to_row = {f: i for i, f in enumerate(filenames_in_order)}
    for filename, age in zip(df["file_name"].tolist(), ages_int.tolist(), strict=True):
        counts[filename_to_row[str(filename)], int(age)] += 1
    row_totals = counts.sum(axis=1)
    if (row_totals == 0).any():
        msg = f"CSV at {csv_path} has at least one image with zero apparent-age votes."
        raise ValueError(msg)
    p_star = (counts.astype(np.float64) / row_totals[:, None]).astype(
        np.float32, copy=False
    )

    new_sums = p_star.sum(axis=1, dtype=np.float64)
    drift = float(np.max(np.abs(new_sums - 1.0)))
    if drift > _ROW_SUM_ATOL:
        msg = (
            f"row sums deviate from 1.0 by {drift:.3e}, exceeds atol="
            f"{_ROW_SUM_ATOL:.0e}; check input counts."
        )
        raise ValueError(msg)
    return {
        "p_star": p_star,
        "support": np.asarray(_AGE_SUPPORT, dtype=np.int64),
        "indices": np.arange(n, dtype=np.int64),
    }


def _matches_existing(output_path: Path, candidate: dict[str, np.ndarray]) -> bool:
    """Return True iff ``output_path`` exists and its arrays match ``candidate``."""
    if not output_path.is_file():
        return False
    try:
        cached = np.load(output_path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    required = {"p_star", "support", "indices"}
    if not required.issubset(cached.files):
        return False
    if cached["p_star"].shape != candidate["p_star"].shape:
        return False
    if not np.array_equal(cached["support"], candidate["support"]):
        return False
    if not np.array_equal(cached["indices"], candidate["indices"]):
        return False
    return bool(
        np.allclose(cached["p_star"], candidate["p_star"], atol=1e-12, rtol=0.0)
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "data" / "appa-real-release",
        help=(
            "APPA-REAL release root containing ``gt_<split>.csv`` files. "
            "Default: data/appa-real-release/."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "valid", "test"),
        help="Which split's CSV to read. Defaults to 'test'.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Override; defaults to <root>/p_star_<split>.npz."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the sidecar unconditionally (skip the idempotence check).",
    )
    args = parser.parse_args(argv)

    csv_path: Path = args.root / f"gt_{args.split}.csv"
    candidate = build_p_star(csv_path)

    output_path = args.output_path or (args.root / f"p_star_{args.split}.npz")

    if not args.force and _matches_existing(output_path, candidate):
        print(f"skipped: {output_path} exists")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        p_star=candidate["p_star"],
        support=candidate["support"],
        indices=candidate["indices"],
    )
    n, k = candidate["p_star"].shape
    sidecar = output_path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "split": str(args.split),
                "n_images": int(n),
                "n_ages": int(k),
                "support_min": int(candidate["support"][0]),
                "support_max": int(candidate["support"][-1]),
            },
            indent=2,
        )
    )
    print(f"wrote {output_path} (n={n}, k={k}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
