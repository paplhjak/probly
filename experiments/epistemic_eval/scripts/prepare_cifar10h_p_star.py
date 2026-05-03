#!/usr/bin/env python3
"""Build the CIFAR-10H ``p_star.npz`` sidecar from raw human counts.

Reads ``data/cifar10h/cifar-10h-master/data/cifar10h-counts.npy``
(shape ``(10000, 10)`` ``int64``; per-image vote counts across the 10
CIFAR classes) and emits ``data/cifar10h/p_star.npz`` with::

    p_star:  (10000, 10) float32  -- row-stochastic conditional p*(y | x).
    support: (10,)       int64    -- arange(10), the class index support.
    indices: (10000,)    int64    -- arange(10000), per-row dataset index.

The resulting file is the input to ``compute_oracle.py`` and to
the per-loss decompositions; one-shot prep only -- the output does
not depend on any method or seed.

Idempotent by default: if ``p_star.npz`` already exists with the
correct shape AND its values match what we would produce now to
``atol=1e-12``, the script prints ``skipped: p_star.npz exists`` and
exits 0. Pass ``--force`` to overwrite unconditionally.

Schema invariants asserted at write time:
    * ``p_star.shape == (10000, 10)``
    * row sums are within ``1e-6`` of 1.0
    * no NaN, no Inf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent

# Locked input / output paths. Override-able only via command-line
# args so reusable for synthetic tests under tmp_path.
_DEFAULT_COUNTS_PATH = (
    _REPO_ROOT / "data" / "cifar10h" / "cifar-10h-master" / "data" / "cifar10h-counts.npy"
)
_DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "cifar10h" / "p_star.npz"

_ROW_SUM_ATOL = 1e-6


def build_p_star(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Convert per-image vote counts to a row-stochastic ``p_star`` triple.

    Args:
        counts: ``(N, K)`` integer (or any non-negative real) array of
            per-image vote counts. Must have at least one non-zero
            entry per row -- the CIFAR-10H release satisfies this by
            construction (every test image has ~51 human votes).

    Returns:
        Dict with ``p_star``, ``support``, ``indices`` (see module
        docstring for shapes / dtypes / contracts).

    Raises:
        ValueError: If ``counts`` is not 2-D, contains negative
            entries, has a zero-sum row, or contains non-finite
            values.
    """
    arr = np.asarray(counts)
    if arr.ndim != 2:
        msg = f"`counts` must be 2-D with shape (N, K), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.size == 0:
        msg = f"`counts` must be non-empty, got shape={arr.shape}."
        raise ValueError(msg)
    arr_float = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_float)):
        msg = "`counts` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    if np.any(arr_float < 0):
        msg = "`counts` must be non-negative."
        raise ValueError(msg)
    row_totals = arr_float.sum(axis=1)
    if np.any(row_totals == 0):
        msg = "`counts` has a zero-sum row; cannot normalise to a probability distribution."
        raise ValueError(msg)
    p_star = (arr_float / row_totals[:, None]).astype(np.float32, copy=False)
    n, k = p_star.shape

    # Sanity assertion: rows sum to 1 within atol=1e-6 (the contract
    # in the file's header). Float32 accumulation is the main source
    # of drift; 1e-6 covers it comfortably.
    new_sums = p_star.sum(axis=1, dtype=np.float64)
    drift = float(np.max(np.abs(new_sums - 1.0)))
    if drift > _ROW_SUM_ATOL:
        msg = (
            f"row sums deviate from 1.0 by {drift:.3e}, exceeds atol={_ROW_SUM_ATOL:.0e}; "
            f"check input counts."
        )
        raise ValueError(msg)
    return {
        "p_star": p_star,
        "support": np.arange(k, dtype=np.int64),
        "indices": np.arange(n, dtype=np.int64),
    }


def _matches_existing(output_path: Path, candidate: dict[str, np.ndarray]) -> bool:
    """Return True iff ``output_path`` exists and its arrays match ``candidate``.

    Uses ``np.array_equal`` for indices/support and ``np.allclose``
    (``atol=1e-12``) for ``p_star``. A drift of even 1e-6 between the
    cached and freshly-recomputed ``p_star`` (e.g. from a different
    accumulation order) should NOT trigger a skip; the script's
    contract is a true value-match.
    """
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
    return bool(np.allclose(cached["p_star"], candidate["p_star"], atol=1e-12, rtol=0.0))


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts-path",
        type=Path,
        default=_DEFAULT_COUNTS_PATH,
        help=(
            "Path to cifar10h-counts.npy. Default: "
            "data/cifar10h/cifar-10h-master/data/cifar10h-counts.npy."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_DEFAULT_OUTPUT_PATH,
        help="Path to write p_star.npz. Default: data/cifar10h/p_star.npz.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite p_star.npz unconditionally (skip the idempotence check).",
    )
    args = parser.parse_args(argv)

    counts_path: Path = args.counts_path
    output_path: Path = args.output_path

    if not counts_path.is_file():
        msg = (
            f"counts file not found at {counts_path}; "
            f"run download_cifar10h.py first or pass --counts-path."
        )
        print(msg, file=sys.stderr)
        return 1

    counts = np.load(counts_path, allow_pickle=False)
    candidate = build_p_star(counts)

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
    print(f"wrote {output_path} (n={n}, k={k}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
