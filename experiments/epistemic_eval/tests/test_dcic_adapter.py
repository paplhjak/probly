"""Tests for ``experiments.epistemic_eval.datasets.dcic``.

Covers:

* Seed-to-fold mapping (:func:`pick_test_fold`) is deterministic
  and covers every fold across the locked 5-seed grid.
* Fold-prefix agnosticism: the adapter works whether the DCIC release
  uses ``fold1``..``fold5`` or ``part1``..``part5`` (per Oleg, the
  Zenodo release uses ``part``-prefixed names).
* Test/train fold partitioning is disjoint and exhaustive.
* Train/val carve-out is deterministic given a seed.
* Class label ordering is stable across DCIC instances that may
  otherwise return ``set()``-derived label_mappings (the
  PYTHONHASHSEED hazard documented in the module docstring).
* End-to-end provider construction against a synthetic on-disk
  fixture: the test provider yields the right number of samples and
  the test indices match the seed's fold partition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")  # noqa: N816

from experiments.epistemic_eval.datasets.dcic import (  # noqa: E402
    DCIC_LOADERS,
    DCIC_NUM_FOLDS,
    _build_loader,
    fold_indices,
    split_test_train_indices,
    split_train_val_indices,
    pick_test_fold,
)
from experiments.epistemic_eval.tests.conftest import (  # noqa: E402
    make_synthetic_dcic_fixture,
)


def test_dcic_loaders_excludes_cifar10hdcic() -> None:
    """The 9 wired DCIC datasets do not include CIFAR10HDCIC.

    Pin: CIFAR10H is wired via the torchvision-derived class on the
    different ``data/cifar10h/`` layout; the DCIC variant is omitted
    intentionally.
    """
    assert "Plankton" in DCIC_LOADERS
    assert "CIFAR10H" not in DCIC_LOADERS
    assert "CIFAR10HDCIC" not in DCIC_LOADERS


def test_pick_test_fold_is_deterministic_modulo_5() -> None:
    """Seeds 0..4 cover all five folds; seed 5 wraps."""
    folds = [f"part{i + 1}" for i in range(DCIC_NUM_FOLDS)]
    seen = {pick_test_fold(s, folds) for s in range(5)}
    assert seen == set(folds)
    # Seed 5 wraps back to fold 1.
    assert pick_test_fold(5, folds) == pick_test_fold(0, folds)


def test_pick_test_fold_sorts_lexicographically() -> None:
    """Fold ordering is by sorted name, not by insertion order."""
    folds_unsorted = ["part5", "part1", "part3", "part2", "part4"]
    assert pick_test_fold(0, folds_unsorted) == "part1"
    assert pick_test_fold(1, folds_unsorted) == "part2"
    assert pick_test_fold(4, folds_unsorted) == "part5"


def test_pick_test_fold_works_with_fold_prefix() -> None:
    """The function does not hardcode 'part' or 'fold' as the prefix."""
    folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    assert pick_test_fold(0, folds) == "fold1"
    assert pick_test_fold(2, folds) == "fold3"


def test_pick_test_fold_rejects_empty_folds() -> None:
    """Empty fold lists raise ValueError, not silently default."""
    with pytest.raises(ValueError, match="no folds found"):
        pick_test_fold(0, [])


def _build_fixture(tmp_path: Path, fold_prefix: str = "part") -> Path:
    return make_synthetic_dcic_fixture(
        tmp_path,
        dataset_name="Plankton",
        num_classes=4,
        images_per_fold=3,
        num_folds=5,
        annotations_per_image=5,
        fold_prefix=fold_prefix,
    )


def test_adapter_loads_synthetic_fixture(tmp_path: Path) -> None:
    """Synthetic fixture loads cleanly through the adapter wrapper."""
    root = _build_fixture(tmp_path)
    loader = _build_loader("Plankton", root, transform=None)
    # 5 folds * 3 images_per_fold = 15 total images.
    assert len(loader) == 15
    assert loader.num_classes <= 4
    # Targets are row-stochastic.
    for target in loader.targets:
        assert torch.allclose(
            target.sum(), torch.tensor(1.0), atol=1.0e-5
        ), target


@pytest.mark.parametrize("fold_prefix", ["part", "fold"])
def test_fold_indices_partition_is_disjoint_and_exhaustive(
    tmp_path: Path, fold_prefix: str
) -> None:
    """fold_indices buckets every image into exactly one fold."""
    root = _build_fixture(tmp_path, fold_prefix=fold_prefix)
    loader = _build_loader("Plankton", root, transform=None)
    indices_by_fold = fold_indices(loader)
    # Each fold has exactly images_per_fold images.
    assert sum(len(v) for v in indices_by_fold.values()) == len(loader)
    all_indices: set[int] = set()
    for fold_name, indices in indices_by_fold.items():
        assert fold_name.startswith(fold_prefix), fold_name
        assert all_indices.isdisjoint(indices), fold_name
        all_indices.update(indices)
    assert all_indices == set(range(len(loader)))


def test_split_test_train_indices_disjoint(tmp_path: Path) -> None:
    """Train and test indices are disjoint and union to the full dataset."""
    root = _build_fixture(tmp_path)
    loader = _build_loader("Plankton", root, transform=None)
    fold, train_idx, test_idx = split_test_train_indices(loader, seed=0)
    assert fold.startswith("part")
    assert set(train_idx).isdisjoint(test_idx)
    assert set(train_idx) | set(test_idx) == set(range(len(loader)))


def test_split_test_train_indices_rotates_per_seed(tmp_path: Path) -> None:
    """Different seeds (0..4) each produce a distinct test fold."""
    root = _build_fixture(tmp_path)
    loader = _build_loader("Plankton", root, transform=None)
    test_folds = {
        split_test_train_indices(loader, seed=s)[0] for s in range(5)
    }
    assert len(test_folds) == 5


def test_split_train_val_indices_disjoint(tmp_path: Path) -> None:
    """The 90/10 carve-out is disjoint and seed-deterministic."""
    root = _build_fixture(tmp_path)
    loader = _build_loader("Plankton", root, transform=None)
    _, train_idx, _ = split_test_train_indices(loader, seed=0)
    train_only, val_only = split_train_val_indices(
        train_idx, seed=0, val_fraction=0.1
    )
    assert set(train_only).isdisjoint(val_only)
    assert set(train_only) | set(val_only) == set(train_idx)
    # Re-running with the same seed yields the same split.
    train2, val2 = split_train_val_indices(train_idx, seed=0, val_fraction=0.1)
    assert train_only == train2
    assert val_only == val2


def test_deterministic_label_ordering(tmp_path: Path) -> None:
    """Two consecutive loader instantiations agree on the label ordering.

    Within a process this is automatic; across processes the upstream
    ``DCICDataset`` would otherwise non-deterministically order
    ``label_mappings`` (it builds it from a ``set()``). Our wrapper
    sorts by ``str(label)`` so the column order is stable.
    """
    root = _build_fixture(tmp_path)
    a = _build_loader("Plankton", root, transform=None)
    b = _build_loader("Plankton", root, transform=None)
    assert a.label_mappings == b.label_mappings
    assert a.num_classes == b.num_classes
    for ta, tb in zip(a.targets, b.targets, strict=True):
        assert torch.equal(ta, tb)


def test_test_provider_yields_expected_count(tmp_path: Path) -> None:
    """The test-provider materialises the right number of samples in n_samples."""
    from experiments.epistemic_eval.datasets.dcic import (
        build_test_provider,
    )

    root = _build_fixture(tmp_path)
    dataset_config = {
        "name": "plankton",
        "family": "dcic",
        "loader": {"dcic_name": "Plankton"},
        "loader_kwargs": {"root": str(root)},
    }
    provider = build_test_provider(
        dataset_config,
        seed=0,
        batch_size=4,
        num_workers=0,
    )
    # 1 fold (3 images) is the test set.
    assert provider.n_samples == 3
    # Iterate one batch to verify shapes flow through.
    batch_count = 0
    for x, y in provider:
        assert x.dtype == torch.float32
        assert x.shape[1] == 3  # RGB
        assert x.shape[2] == 224
        assert x.shape[3] == 224
        assert y.shape[0] == x.shape[0]
        batch_count += 1
    assert batch_count >= 1


def test_full_train_provider_yields_expected_count(tmp_path: Path) -> None:
    """The full-train provider has the four non-test folds' images."""
    from experiments.epistemic_eval.datasets.dcic import (
        build_full_train_provider,
    )

    root = _build_fixture(tmp_path)
    dataset_config = {
        "name": "plankton",
        "family": "dcic",
        "loader": {"dcic_name": "Plankton"},
        "loader_kwargs": {"root": str(root)},
    }
    provider = build_full_train_provider(
        dataset_config,
        seed=0,
        batch_size=4,
        num_workers=0,
    )
    # 4 folds * 3 images = 12 train images.
    assert provider.n_samples == 12


def test_train_val_providers_disjoint(tmp_path: Path) -> None:
    """The train and val providers cover disjoint slices of the 4 train folds."""
    from experiments.epistemic_eval.datasets.dcic import (
        build_train_val_providers,
    )

    root = _build_fixture(tmp_path)
    dataset_config = {
        "name": "plankton",
        "family": "dcic",
        "loader": {"dcic_name": "Plankton"},
        "loader_kwargs": {"root": str(root)},
    }
    train, val = build_train_val_providers(
        dataset_config,
        seed=0,
        val_fraction=0.25,
        batch_size=2,
        num_workers=0,
    )
    train_set = set(train.indices.tolist())
    val_set = set(val.indices.tolist())
    assert train_set.isdisjoint(val_set)
    assert len(train_set) + len(val_set) == 12


def test_make_resnet18_factory_produces_correct_head_size() -> None:
    """The factory's nn.Linear head matches the requested num_classes."""
    from experiments.epistemic_eval.datasets.dcic import make_resnet18_factory
    from torch import nn

    factory = make_resnet18_factory(num_classes=7, pretrained=False)
    model = factory()
    # torchvision resnet18 has model.fc == final Linear; cast for ty.
    fc = model.fc
    assert isinstance(fc, nn.Linear)
    assert fc.out_features == 7
    # in_features matches resnet18's penultimate dim.
    assert fc.in_features == 512
    # Sanity: a forward pass on a (1, 3, 224, 224) input shapes to (1, 7).
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 7)
