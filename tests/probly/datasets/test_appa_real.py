"""Tests for :mod:`probly.datasets.appa_real`.

The real APPA-REAL data is gated behind registration, so the on-disk
test runs only if ``data/appa_real/gt_test.csv`` is present. The mock
test always runs and is the load-bearing check for the loader's
counts/support/determinism contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import pytest
import torch

from probly.datasets.appa_real import AppaReal

if TYPE_CHECKING:
    pass


_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_DATA_DIR = _REPO_ROOT / "data" / "appa_real"


def _write_dummy_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build_mock_dataset(tmp_path: Path) -> Path:
    """Construct a minimal APPA-REAL-style directory under ``tmp_path``.

    Three rater rows for two images. Image ``a.jpg`` gets two votes
    (ages 25 and 30); image ``b.jpg`` gets one vote (age 30). The
    age support inferred by the loader should therefore be [25, 30].
    """
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_dummy_image(test_dir / "a.jpg")
    _write_dummy_image(test_dir / "b.jpg")
    csv_path = tmp_path / "gt_test.csv"
    csv_path.write_text("file_name,apparent_age\na.jpg,25\na.jpg,30\nb.jpg,30\n")
    return tmp_path


def test_mock_loader_infers_support_and_counts(tmp_path: Path) -> None:
    root = _build_mock_dataset(tmp_path)
    dataset = AppaReal(root=root, split="test")
    assert dataset.age_support == [25, 30]
    assert len(dataset) == 2
    counts_by_filename = {name: dataset.vote_counts[name].clone() for name in dataset.image_filenames}
    assert torch.equal(counts_by_filename["a.jpg"], torch.tensor([1.0, 1.0]))
    assert torch.equal(counts_by_filename["b.jpg"], torch.tensor([0.0, 1.0]))
    assert dataset.num_raters_per_image == {"a.jpg": 2, "b.jpg": 1}


def test_mock_loader_getitem_returns_image_and_counts(tmp_path: Path) -> None:
    root = _build_mock_dataset(tmp_path)
    dataset = AppaReal(root=root, split="test")
    image, counts = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    assert counts.shape == (len(dataset.age_support),)
    assert counts.sum().item() > 0


def test_mock_loader_getitem_is_deterministic(tmp_path: Path) -> None:
    root = _build_mock_dataset(tmp_path)
    dataset = AppaReal(root=root, split="test")
    image_a, counts_a = dataset[0]
    image_b, counts_b = dataset[0]
    assert torch.equal(image_a, image_b)
    assert torch.equal(counts_a, counts_b)


def test_mock_loader_normalised_counts_sum_to_one(tmp_path: Path) -> None:
    root = _build_mock_dataset(tmp_path)
    dataset = AppaReal(root=root, split="test")
    _, counts = dataset[0]
    p_star = counts / counts.sum()
    assert torch.isclose(p_star.sum(), torch.tensor(1.0))


def test_mock_loader_explicit_support_widens(tmp_path: Path) -> None:
    root = _build_mock_dataset(tmp_path)
    dataset = AppaReal(root=root, split="test", age_support=[10, 20, 25, 30, 99])
    assert dataset.age_support == [10, 20, 25, 30, 99]
    counts = dataset.vote_counts["a.jpg"]
    assert counts.shape == (5,)
    # Index of 25 is 2, index of 30 is 3.
    assert counts[2].item() == 1.0
    assert counts[3].item() == 1.0
    assert counts[0].item() == 0.0


def test_mock_loader_rejects_support_missing_observed_age(tmp_path: Path) -> None:
    root = _build_mock_dataset(tmp_path)
    with pytest.raises(ValueError, match="missing observed ages"):
        AppaReal(root=root, split="test", age_support=[25])  # 30 is missing


def test_mock_loader_rejects_negative_age(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_dummy_image(test_dir / "a.jpg")
    csv_path = tmp_path / "gt_test.csv"
    csv_path.write_text("file_name,apparent_age\na.jpg,-5\n")
    with pytest.raises(ValueError, match="Negative apparent-age values"):
        AppaReal(root=tmp_path, split="test")


def test_mock_loader_rejects_non_integer_age(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_dummy_image(test_dir / "a.jpg")
    csv_path = tmp_path / "gt_test.csv"
    csv_path.write_text("file_name,apparent_age\na.jpg,25.5\n")
    with pytest.raises(ValueError, match="Non-integer apparent-age values"):
        AppaReal(root=tmp_path, split="test")


def test_mock_loader_raises_on_missing_csv(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    with pytest.raises(ValueError, match=r"gt_test\.csv"):
        AppaReal(root=tmp_path, split="test")


def test_mock_loader_raises_on_missing_image_folder(tmp_path: Path) -> None:
    csv_path = tmp_path / "gt_test.csv"
    csv_path.write_text("file_name,apparent_age\na.jpg,25\n")
    with pytest.raises(ValueError, match="split image folder"):
        AppaReal(root=tmp_path, split="test")


def test_mock_loader_raises_on_missing_root(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="root directory"):
        AppaReal(root=nonexistent, split="test")


def test_mock_loader_raises_with_listed_columns_when_no_age_column(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_dummy_image(test_dir / "a.jpg")
    csv_path = tmp_path / "gt_test.csv"
    csv_path.write_text("file_name,score\na.jpg,25\n")
    with pytest.raises(ValueError, match=r"Could not identify required columns"):
        AppaReal(root=tmp_path, split="test")


def test_use_face_crops_rewrites_filenames(tmp_path: Path) -> None:
    """``use_face_crops=True`` resolves ``X.jpg`` to ``X.jpg_face.jpg``.

    The CVPR 2017 release ships a face crop for every original at the
    suffixed filename. The CSV references the originals; the loader's
    flag swaps the in-memory filename so all downstream lookups
    (``__getitem__``, ``vote_counts``, ``image_filenames``) point at
    the crops.
    """
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_dummy_image(test_dir / "a.jpg")
    _write_dummy_image(test_dir / "a.jpg_face.jpg")
    _write_dummy_image(test_dir / "b.jpg")
    _write_dummy_image(test_dir / "b.jpg_face.jpg")
    csv_path = tmp_path / "gt_test.csv"
    csv_path.write_text("file_name,apparent_age\na.jpg,25\na.jpg,30\nb.jpg,30\n")

    dataset = AppaReal(root=tmp_path, split="test", use_face_crops=True)
    assert sorted(dataset.image_filenames) == ["a.jpg_face.jpg", "b.jpg_face.jpg"]
    assert "a.jpg" not in dataset.vote_counts
    assert "a.jpg_face.jpg" in dataset.vote_counts
    # __getitem__ resolves to the crop file (would FileNotFoundError if
    # the rewrite hadn't happened).
    image, counts = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert counts.sum().item() > 0


def test_use_face_crops_default_is_false_back_compat(tmp_path: Path) -> None:
    """Default behaviour is unchanged: lookups go to the CSV's file_name as-is."""
    root = _build_mock_dataset(tmp_path)
    dataset = AppaReal(root=root, split="test")
    assert sorted(dataset.image_filenames) == ["a.jpg", "b.jpg"]
    assert "a.jpg_face.jpg" not in dataset.vote_counts


def test_compute_global_support_unions_all_splits(tmp_path: Path) -> None:
    for split, ages in (("train", [10, 20]), ("valid", [20, 30]), ("test", [25, 30])):
        split_dir = tmp_path / split
        split_dir.mkdir()
        _write_dummy_image(split_dir / "a.jpg")
        rows = "\n".join(f"a.jpg,{age}" for age in ages)
        (tmp_path / f"gt_{split}.csv").write_text("file_name,apparent_age\n" + rows + "\n")
    support = AppaReal.compute_global_support(tmp_path)
    assert support == [10, 20, 25, 30]


@pytest.mark.skipif(
    not (_REAL_DATA_DIR / "gt_test.csv").is_file(),
    reason="APPA-REAL data not available locally; run download_appa_real.py to fetch.",
)
def test_real_loader_smoke() -> None:
    dataset = AppaReal(root=_REAL_DATA_DIR, split="test")
    assert len(dataset) > 0
    image, counts = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    assert counts.sum().item() > 0
    p_star = counts / counts.sum()
    assert torch.isclose(p_star.sum(), torch.tensor(1.0))
    assert counts.shape == (len(dataset.age_support),)
    image_again, counts_again = dataset[0]
    assert torch.equal(image, image_again)
    assert torch.equal(counts, counts_again)
