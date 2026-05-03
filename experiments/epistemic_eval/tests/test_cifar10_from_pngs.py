"""Smoke tests for the PNG-backed CIFAR-10 loader.

These tests build a tiny synthetic PNG tree in ``tmp_path`` rather
than depending on the real on-disk dataset (which is gitignored
under ``data/cifar10-raw-images/``). A single test additionally
exercises the real on-disk dataset if it is present, asserting the
canonical 50000-train / 10000-test counts; that test ``skip``s
otherwise so CI without the data still passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from experiments.epistemic_eval.datasets.cifar10_from_pngs import CIFAR10FromPNGs

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _make_synthetic_png_tree(tmp_path: Path, n_per_class: int = 2) -> Path:
    """Build ``<tmp_path>/<split>/<ClassName>/*.png`` with random images."""
    rng = np.random.default_rng(0)
    root = tmp_path / "synthetic_cifar10"
    for split in ("train", "test"):
        for class_idx, class_name in enumerate(CIFAR10FromPNGs.CLASSES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(
                    class_dir / f"{class_name.lower()}_{class_idx:02d}_{i:04d}.png"
                )
    return root


def test_cifar10_from_pngs_yields_correct_counts(tmp_path: Path) -> None:
    """Synthetic 2-per-class tree -> 20 images per split, 10 classes."""
    root = _make_synthetic_png_tree(tmp_path, n_per_class=2)
    train = CIFAR10FromPNGs(root=root, train=True)
    test = CIFAR10FromPNGs(root=root, train=False)
    assert len(train) == 20
    assert len(test) == 20
    # Class counts: 2 per class for both splits.
    train_labels = [train[i][1] for i in range(len(train))]
    assert sorted(train_labels) == sorted(list(range(10)) * 2)
    test_labels = [test[i][1] for i in range(len(test))]
    assert sorted(test_labels) == sorted(list(range(10)) * 2)


def test_cifar10_from_pngs_returns_pil_image(tmp_path: Path) -> None:
    """Without a transform, items are ``(PIL.Image, int)``."""
    root = _make_synthetic_png_tree(tmp_path, n_per_class=1)
    ds = CIFAR10FromPNGs(root=root, train=True)
    img, label = ds[0]
    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)
    assert img.mode == "RGB"
    assert isinstance(label, int)
    assert 0 <= label < 10


def test_cifar10_from_pngs_applies_transform(tmp_path: Path) -> None:
    """A custom transform is applied; the dataset returns whatever the transform yields."""
    root = _make_synthetic_png_tree(tmp_path, n_per_class=1)

    def transform(img: Image.Image) -> tuple[int, int]:
        return img.size

    ds = CIFAR10FromPNGs(root=root, train=True, transform=transform)
    img_size, _ = ds[0]
    assert img_size == (32, 32)


def test_cifar10_from_pngs_deterministic_ordering(tmp_path: Path) -> None:
    """Two instances over the same tree yield items in the same order."""
    root = _make_synthetic_png_tree(tmp_path, n_per_class=3)
    a = CIFAR10FromPNGs(root=root, train=True)
    b = CIFAR10FromPNGs(root=root, train=True)
    a_labels = [a[i][1] for i in range(len(a))]
    b_labels = [b[i][1] for i in range(len(b))]
    assert a_labels == b_labels


def test_cifar10_from_pngs_missing_split_raises(tmp_path: Path) -> None:
    """Missing ``train/`` or ``test/`` subdir -> FileNotFoundError."""
    root = tmp_path / "empty_cifar10"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="split directory"):
        CIFAR10FromPNGs(root=root, train=True)


def test_cifar10_from_pngs_missing_class_raises(tmp_path: Path) -> None:
    """A split missing one of the 10 class subdirs -> FileNotFoundError."""
    root = tmp_path / "broken_cifar10"
    (root / "train" / "Airplane").mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
        root / "train" / "Airplane" / "p.png"
    )
    # The other 9 class dirs are missing.
    with pytest.raises(FileNotFoundError, match="class directory"):
        CIFAR10FromPNGs(root=root, train=True)


def test_cifar10_from_pngs_real_dataset_counts() -> None:
    """If the real on-disk dataset is present, assert canonical sizes.

    Skips on CI / dev machines without the data.
    """
    real_root = _REPO_ROOT / "data" / "cifar10-raw-images" / "images"
    if not (real_root / "train" / "Airplane").is_dir():
        pytest.skip("real CIFAR-10 PNGs not present at data/cifar10-raw-images/images")
    train = CIFAR10FromPNGs(root=real_root, train=True)
    test = CIFAR10FromPNGs(root=real_root, train=False)
    assert len(train) == 50_000
    assert len(test) == 10_000
