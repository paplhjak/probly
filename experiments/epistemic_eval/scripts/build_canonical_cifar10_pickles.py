#!/usr/bin/env python3
"""Build canonical-format CIFAR-10 pickled batches from raw PNGs.

The Toronto host that serves the official ``cifar-10-python.tar.gz``
has been flaking; HuggingFace mirrors don't carry the canonical
tarball; the Web Archive only redirects to dead snapshots. We do
have the raw PNG dump under ``data/cifar10-raw-images/images/``
plus ``cifar10h-raw.zip`` which gives us the (filename ->
canonical CIFAR-10 test index) mapping. This script produces the
five ``data_batch_<i>`` train pickles and the ``test_batch`` test
pickle in a layout that ``torchvision.datasets.CIFAR10`` accepts.

The test pickle preserves canonical CIFAR-10 test order (the order
in which ``cifar10h-counts.npy`` is indexed); train batches are
ordered class-by-class then alphabetically by filename, which is
deterministic and irrelevant for downstream training.

The MD5s of the produced pickles do NOT match torchvision's
hardcoded canonical MD5s -- our pickles are bit-different from the
official tarball even though the underlying image data is the same
(PNG decode is not byte-for-byte invertible against the original
tarball's pickle blobs). Callers therefore use a thin
``CIFAR10NoMD5`` subclass that overrides ``_check_integrity``; see
``experiments/epistemic_eval/datasets/cifar10_canonical.py``.

The script is idempotent: it skips writes when an output file
already exists. Pass ``--force`` to rebuild from scratch.
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

CANONICAL_TEST_SIZE = 10_000
CANONICAL_TRAIN_SIZE = 50_000
CANONICAL_TRAIN_BATCH_SIZE = 10_000

# Canonical CIFAR-10 class order (per the official dataset).
CLASSES_LOWER: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CLASSES_TITLE: tuple[str, ...] = tuple(c.title() for c in CLASSES_LOWER)


def _build_filename_to_canonical_test_idx(raw_zip: Path) -> dict[str, int]:
    """Parse cifar10h-raw.csv -> {test PNG basename: canonical idx in [0, 10000)}.

    Filters out attention-check rows (``is_attn_check == 1`` or
    ``cifar10_test_test_idx`` outside ``[0, 10000)``).
    """
    if not raw_zip.is_file():
        msg = f"cifar10h-raw.zip not found at {raw_zip}."
        raise FileNotFoundError(msg)
    mapping: dict[str, int] = {}
    n_rows = 0
    n_skipped_attn = 0
    n_skipped_oob = 0
    with zipfile.ZipFile(raw_zip) as zf, zf.open("cifar10h-raw.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
        for row in reader:
            n_rows += 1
            if row.get("is_attn_check", "0") == "1":
                n_skipped_attn += 1
                continue
            try:
                idx = int(row["cifar10_test_test_idx"])
            except (KeyError, ValueError):
                n_skipped_oob += 1
                continue
            if not (0 <= idx < CANONICAL_TEST_SIZE):
                n_skipped_oob += 1
                continue
            fname = row["image_filename"]
            existing = mapping.get(fname)
            if existing is None:
                mapping[fname] = idx
            elif existing != idx:
                msg = (
                    f"filename {fname!r} maps to two distinct canonical indices "
                    f"({existing} and {idx}); cifar10h-raw.csv is inconsistent."
                )
                raise ValueError(msg)
    print(
        f"parsed {n_rows} CSV rows: {n_skipped_attn} attn-check, "
        f"{n_skipped_oob} OOB; {len(mapping)} unique test filenames mapped."
    )
    return mapping


def _png_to_canonical_row(png_path: Path) -> np.ndarray:
    """Decode a 32x32 RGB PNG to the canonical CIFAR-10 row layout.

    Canonical layout: ``(R[1024], G[1024], B[1024])`` flattened, dtype
    ``uint8``, total length 3072. ``image.reshape(3, 32, 32)`` yields
    a ``(C, H, W)`` tensor.
    """
    with Image.open(png_path) as img:
        img.load()
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # (32, 32, 3)
    if arr.shape != (32, 32, 3):
        msg = f"{png_path} has shape {arr.shape}; expected (32, 32, 3)."
        raise ValueError(msg)
    # Convert (H, W, C) -> (C, H, W) then flatten C-major.
    return arr.transpose(2, 0, 1).reshape(-1)


def _build_test_batch(
    test_root: Path, fname_to_idx: dict[str, int]
) -> tuple[np.ndarray, list[int], list[str]]:
    """Build canonical-order ``(data, labels, filenames)`` for the test split."""
    # Discover all test PNGs grouped by class (so labels are correct).
    fname_to_label: dict[str, int] = {}
    fname_to_path: dict[str, Path] = {}
    for class_idx, class_name in enumerate(CLASSES_TITLE):
        class_dir = test_root / class_name
        if not class_dir.is_dir():
            msg = f"expected class dir at {class_dir}."
            raise FileNotFoundError(msg)
        for path in sorted(class_dir.glob("*.png")):
            if path.name in fname_to_label:
                msg = f"duplicate test PNG basename: {path.name}."
                raise ValueError(msg)
            fname_to_label[path.name] = class_idx
            fname_to_path[path.name] = path

    if len(fname_to_label) != CANONICAL_TEST_SIZE:
        msg = (
            f"discovered {len(fname_to_label)} test PNGs; "
            f"expected exactly {CANONICAL_TEST_SIZE}."
        )
        raise ValueError(msg)

    # Build the canonical-ordered table: at canonical index i, store the
    # PNG whose basename maps to i in cifar10h-raw.csv.
    data = np.zeros((CANONICAL_TEST_SIZE, 3072), dtype=np.uint8)
    labels: list[int] = [-1] * CANONICAL_TEST_SIZE
    filenames: list[str] = [""] * CANONICAL_TEST_SIZE
    seen: set[int] = set()
    for fname, canonical_idx in fname_to_idx.items():
        if fname not in fname_to_label:
            # CSV references a filename we don't have on disk.
            continue
        if canonical_idx in seen:
            msg = f"duplicate canonical index {canonical_idx} in CSV."
            raise ValueError(msg)
        seen.add(canonical_idx)
        path = fname_to_path[fname]
        data[canonical_idx] = _png_to_canonical_row(path)
        labels[canonical_idx] = fname_to_label[fname]
        filenames[canonical_idx] = fname

    if len(seen) != CANONICAL_TEST_SIZE:
        missing = sorted(set(range(CANONICAL_TEST_SIZE)) - seen)
        msg = (
            f"only {len(seen)} canonical indices populated; "
            f"missing {len(missing)} (first few: {missing[:5]}). "
            f"cifar10h-raw.csv may not cover the full test set."
        )
        raise ValueError(msg)
    return data, labels, filenames


def _build_train_batches(
    train_root: Path,
) -> tuple[list[np.ndarray], list[list[int]], list[list[str]]]:
    """Build five 10000-row ``(data, labels, filenames)`` chunks for the train split.

    Train ordering is not canonically constrained (the canonical
    tarball uses an arbitrary shuffle). We choose: class-by-class
    then alphabetical within class, split into five contiguous
    10000-row chunks. Deterministic across runs.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []
    for class_idx, class_name in enumerate(CLASSES_TITLE):
        class_dir = train_root / class_name
        if not class_dir.is_dir():
            msg = f"expected class dir at {class_dir}."
            raise FileNotFoundError(msg)
        paths = sorted(class_dir.glob("*.png"))
        for path in paths:
            rows.append(_png_to_canonical_row(path))
            labels.append(class_idx)
            filenames.append(path.name)
    if len(rows) != CANONICAL_TRAIN_SIZE:
        msg = f"discovered {len(rows)} train PNGs; expected {CANONICAL_TRAIN_SIZE}."
        raise ValueError(msg)
    data = np.stack(rows, axis=0)

    batch_data: list[np.ndarray] = []
    batch_labels: list[list[int]] = []
    batch_filenames: list[list[str]] = []
    for i in range(5):
        start = i * CANONICAL_TRAIN_BATCH_SIZE
        stop = start + CANONICAL_TRAIN_BATCH_SIZE
        batch_data.append(data[start:stop])
        batch_labels.append(labels[start:stop])
        batch_filenames.append(filenames[start:stop])
    return batch_data, batch_labels, batch_filenames


def _write_pickle(
    path: Path,
    data: np.ndarray,
    labels: list[int],
    filenames: list[str],
    batch_label: str,
) -> None:
    """Pickle the canonical CIFAR-10 batch dict.

    Keys are ``str`` (not ``bytes``) so torchvision's loader sees them
    as such after ``pickle.load(encoding="latin1")``.
    """
    blob = {
        "batch_label": batch_label,
        "labels": list(labels),
        "data": data,
        "filenames": list(filenames),
    }
    with path.open("wb") as handle:
        pickle.dump(blob, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_batches_meta(path: Path) -> None:
    """Write ``batches.meta`` mirroring the canonical tarball."""
    meta = {
        "num_cases_per_batch": CANONICAL_TRAIN_BATCH_SIZE,
        "label_names": list(CLASSES_LOWER),
        "num_vis": 3072,
    }
    with path.open("wb") as handle:
        pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png-root",
        type=Path,
        default=Path("data/cifar10-raw-images/images"),
        help="Root containing train/<ClassName>/*.png and test/<ClassName>/*.png.",
    )
    parser.add_argument(
        "--raw-zip",
        type=Path,
        default=Path("data/cifar10h/cifar-10h-master/data/cifar10h-raw.zip"),
        help="cifar10h-raw.zip (provides canonical test ordering).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cifar10h/cifar-10-batches-py"),
        help="Output directory for the generated pickles.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    expected = (
        [args.output_root / "batches.meta"]
        + [args.output_root / f"data_batch_{i + 1}" for i in range(5)]
        + [args.output_root / "test_batch"]
    )
    if not args.force and all(p.exists() for p in expected):
        print(f"all outputs already present at {args.output_root}; pass --force to rebuild.")
        return 0

    print("building filename -> canonical-test-index map ...")
    fname_to_idx = _build_filename_to_canonical_test_idx(args.raw_zip)
    print(f"  -> {len(fname_to_idx)} mappings")

    print("building test batch (canonical order, 10000 rows) ...")
    test_data, test_labels, test_filenames = _build_test_batch(
        args.png_root / "test", fname_to_idx
    )
    print(f"  -> data shape {test_data.shape} dtype {test_data.dtype}")

    print("building train batches (5 x 10000 rows) ...")
    train_data, train_labels, train_filenames = _build_train_batches(args.png_root / "train")

    print("writing pickles ...")
    _write_batches_meta(args.output_root / "batches.meta")
    for i in range(5):
        out = args.output_root / f"data_batch_{i + 1}"
        _write_pickle(
            out,
            train_data[i],
            train_labels[i],
            train_filenames[i],
            batch_label=f"training batch {i + 1} of 5",
        )
        print(f"  wrote {out}")
    test_out = args.output_root / "test_batch"
    _write_pickle(
        test_out,
        test_data,
        test_labels,
        test_filenames,
        batch_label="testing batch 1 of 1",
    )
    print(f"  wrote {test_out}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
