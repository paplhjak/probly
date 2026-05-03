"""Align our PNG test set with the canonical CIFAR-10H ordering.

The CIFAR-10H human soft-label counts at
``data/cifar10h/cifar-10h-master/data/cifar10h-counts.npy`` are
indexed in *canonical CIFAR-10 test order* (the order in which images
appear in the canonical pickled ``test_batch``). Our PNG test set at
``data/cifar10-raw-images/images/test/<ClassName>/*.png`` is
organised by class instead.

The CIFAR-10H raw CSV
(``data/cifar10h/cifar-10h-master/data/cifar10h-raw.zip ::
cifar10h-raw.csv``) carries the cross-reference: every annotation
row has both ``image_filename`` (matching our PNG basename) and
``cifar10_test_test_idx`` (the canonical index in
``[0, 10000)``). This module parses that CSV once and returns the
permutation that reorders our PNG set into canonical order.

The alignment also yields the canonical p*(y|x) tensor: row ``i``
of ``cifar10h-counts.npy`` is the human-vote histogram for the
image at canonical index ``i``; normalise to probabilities and
the result is the first-order ground-truth ``p*`` for the paper.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile

import numpy as np

from experiments.epistemic_eval.datasets.cifar10_from_pngs import CIFAR10FromPNGs

CANONICAL_TEST_SIZE = 10_000


def _build_filename_to_canonical_idx(raw_zip: Path) -> dict[str, int]:
    """Parse cifar10h-raw.csv inside ``raw_zip`` -> {filename: canonical_idx}.

    The CSV has many duplicate rows (one per annotation, ~50 per
    image); we keep the first canonical_idx seen for each filename
    and verify any subsequent rows agree.
    """
    if not raw_zip.is_file():
        msg = f"cifar10h-raw.zip not found at {raw_zip}."
        raise FileNotFoundError(msg)
    mapping: dict[str, int] = {}
    with zipfile.ZipFile(raw_zip) as zf:
        with zf.open("cifar10h-raw.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                fname = row["image_filename"]
                idx_str = row["cifar10_test_test_idx"]
                if not idx_str:
                    continue
                idx = int(idx_str)
                if not (0 <= idx < CANONICAL_TEST_SIZE):
                    msg = f"canonical index {idx} for {fname!r} out of [0, {CANONICAL_TEST_SIZE})."
                    raise ValueError(msg)
                if fname in mapping:
                    if mapping[fname] != idx:
                        msg = (
                            f"filename {fname!r} maps to multiple canonical indices "
                            f"({mapping[fname]} and {idx}); CSV is inconsistent."
                        )
                        raise ValueError(msg)
                else:
                    mapping[fname] = idx
    return mapping


def build_canonical_permutation(
    png_root: Path,
    raw_zip: Path,
) -> np.ndarray:
    """Return the permutation that reorders ``CIFAR10FromPNGs(test)`` to canonical order.

    ``perm[i]`` is the index into ``CIFAR10FromPNGs(test)`` of the
    item whose canonical CIFAR-10 test index is ``i``. So:

        perm = build_canonical_permutation(...)
        ds = CIFAR10FromPNGs(png_root, train=False)
        # ds[perm[0]] is the canonical-index-0 image, etc.

    Args:
        png_root: Root containing ``test/<ClassName>/*.png``.
        raw_zip: Path to ``cifar10h-raw.zip``.

    Returns:
        Length-10000 ``int64`` permutation array.

    Raises:
        FileNotFoundError: If either input is missing.
        ValueError: If alignment fails (size mismatch, missing
            filenames, or duplicate canonical indices).
    """
    fname_to_canonical = _build_filename_to_canonical_idx(raw_zip)

    ds = CIFAR10FromPNGs(root=png_root, train=False)
    if len(ds) != CANONICAL_TEST_SIZE:
        msg = (
            f"PNG test set has {len(ds)} images; expected {CANONICAL_TEST_SIZE} "
            f"to match canonical CIFAR-10 test ordering."
        )
        raise ValueError(msg)

    canonical_to_png_idx = np.full(CANONICAL_TEST_SIZE, -1, dtype=np.int64)
    for png_idx, (path, _label) in enumerate(ds._items):  # noqa: SLF001
        fname = path.name
        canonical_idx = fname_to_canonical.get(fname)
        if canonical_idx is None:
            msg = (
                f"PNG basename {fname!r} not found in cifar10h-raw.csv; "
                f"cannot align to canonical order."
            )
            raise ValueError(msg)
        if canonical_to_png_idx[canonical_idx] != -1:
            msg = (
                f"two PNG files claim canonical index {canonical_idx}: "
                f"{ds._items[canonical_to_png_idx[canonical_idx]][0].name!r} "  # noqa: SLF001
                f"and {fname!r}."
            )
            raise ValueError(msg)
        canonical_to_png_idx[canonical_idx] = png_idx

    if (canonical_to_png_idx == -1).any():
        missing = int((canonical_to_png_idx == -1).sum())
        msg = f"{missing} canonical indices have no PNG match."
        raise ValueError(msg)

    return canonical_to_png_idx


def load_p_star_from_counts(counts_path: Path) -> np.ndarray:
    """Load the CIFAR-10H counts and normalise to row-stochastic ``p*``.

    Args:
        counts_path: Path to ``cifar10h-counts.npy``.

    Returns:
        Float32 ``(10000, 10)`` array; each row sums to 1.
    """
    counts = np.load(counts_path)
    if counts.shape != (CANONICAL_TEST_SIZE, 10):
        msg = f"cifar10h-counts.npy has shape {counts.shape}; expected (10000, 10)."
        raise ValueError(msg)
    counts = counts.astype(np.float32)
    rowsum = counts.sum(axis=1, keepdims=True)
    if (rowsum == 0).any():
        msg = "cifar10h-counts.npy contains a row that sums to 0; cannot normalise."
        raise ValueError(msg)
    return counts / rowsum
