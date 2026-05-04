"""DCIC adapter for the epistemic-eval pipeline.

This module bridges the DCIC dataset family
(:cite:`schmarjeIsOne2022`, see ``probly.datasets.torch:98-428``) to
the experiment pipeline that was designed around CIFAR-10H. Three
DCIC-specific concerns are centralised here:

* **Fold-per-seed mapping.** DCIC ships predefined 5-fold splits
  embedded in image paths (the second path component, e.g.
  ``"Plankton/part1/img.png"`` -> ``"part1"``). The Zenodo release
  uses ``part1`` .. ``part5`` (per Oleg, who maintains the
  reference :mod:`experiments.first_order_data.dcic_ensemble_pipeline`).
  We do NOT hardcode the prefix: :func:`test_fold_for_seed` reads
  the actual fold names from the loader's image paths, sorts them
  lexicographically, and maps seed ``N`` to fold ``sorted_folds[N % len(folds)]``.
  Five seeds therefore cover all 5 folds — a full 5-fold
  cross-validation by construction. (Oleg's reference pipeline runs
  one fold per invocation and does not rotate; we deliberately
  diverge here to exercise every fold across the seed grid without
  expanding the run grid further.)

* **Deterministic class label ordering.** The upstream
  :class:`probly.datasets.torch.DCICDataset` builds
  ``label_mappings`` from a ``set()`` (``torch.py:159``), whose
  iteration order depends on Python's PYTHONHASHSEED. Across
  processes this can flip the column ordering of ``targets`` and
  break alignment between cached predictions and cached p*. We wrap
  every loader instance with :class:`_DeterministicDCIC` which
  re-keys ``label_mappings`` by ``str(label)`` order and rebuilds
  ``targets`` accordingly.

* **Backbone factory.** All DCIC datasets are wired through a
  single torchvision ``resnet18`` with ImageNet-pretrained weights
  and a fresh K-class linear head. See :func:`make_resnet18_factory`.

The module exposes provider builders — :func:`build_test_provider`,
:func:`build_full_train_provider`,
:func:`build_train_val_providers` — that wrap a ``DCICDataset`` in a
torch :class:`~torch.utils.data.DataLoader` and adapt it to the
experiment's :class:`FeatureProvider` interface. Image loading is
lazy (per-batch ``__getitem__``) so the providers do not preload
the dataset into memory at fp32; this matters at the 224x224 input
resolution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from probly.datasets.torch import (
    Benthic,
    DCICDataset,
    MiceBone,
    Pig,
    Plankton,
    QualityMRI,
    Synthetic,
    Treeversity1,
    Treeversity6,
    Turkey,
)

from experiments.epistemic_eval.methods._base import FeatureProvider


#: The 9 DCIC datasets we wire through the epistemic-eval pipeline.
#: ``CIFAR10HDCIC`` is intentionally absent: the existing CIFAR-10H
#: wiring already covers the same images via the torchvision-derived
#: ``probly.datasets.torch.CIFAR10H`` class with a different on-disk
#: layout, so the DCIC-format variant would duplicate work without a
#: new image domain.
DCIC_LOADERS: dict[str, type[DCICDataset]] = {
    "Benthic": Benthic,
    "MiceBone": MiceBone,
    "Pig": Pig,
    "Plankton": Plankton,
    "QualityMRI": QualityMRI,
    "Synthetic": Synthetic,
    "Treeversity#1": Treeversity1,
    "Treeversity#6": Treeversity6,
    "Turkey": Turkey,
}

#: Number of folds the DCIC release ships with. The names themselves
#: are read from disk (typically ``part1``..``part5`` per Oleg) so
#: this constant only governs the seed-to-fold modulo arithmetic
#: in :func:`test_fold_for_seed`.
DCIC_NUM_FOLDS: int = 5

#: Default ImageNet normalisation for the torchvision ResNet-18
#: stem; all DCIC datasets use this since they share the
#: ImageNet-pretrained backbone.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Input resolution required by torchvision ResNet-18.
_RESNET18_INPUT_SIZE: int = 224


def pick_test_fold(seed: int, fold_names: Iterable[str]) -> str:
    """Pick the seed-derived test fold from the loader's actual fold set.

    Locked convention: take the lexicographically sorted fold names
    and pick index ``seed % len(folds)``. The DCIC Zenodo release
    uses ``part1``..``part5`` (per Oleg) but we do not hardcode that
    prefix — the function reads whatever is on disk so a re-released
    dataset with renamed folds keeps working.

    The name is ``pick_test_fold`` rather than ``test_fold_for_seed``
    deliberately: pytest auto-collects functions whose names begin
    with ``test_``, which would mis-classify this as a test case.

    Args:
        seed: The run's root seed.
        fold_names: Iterable of fold names found on disk (the keys of
            :func:`fold_indices`).

    Returns:
        One of the fold names from ``fold_names``.

    Raises:
        ValueError: If ``fold_names`` is empty.
    """
    sorted_folds = sorted(set(fold_names))
    if not sorted_folds:
        msg = "no folds found; cannot derive a test fold from the seed."
        raise ValueError(msg)
    return sorted_folds[int(seed) % len(sorted_folds)]


def _resolve_loader(dataset_name: str) -> type[DCICDataset]:
    """Look up a DCIC loader class by dataset name."""
    loader = DCIC_LOADERS.get(dataset_name)
    if loader is None:
        msg = (
            f"unknown DCIC dataset {dataset_name!r}; expected one of "
            f"{sorted(DCIC_LOADERS)!r}."
        )
        raise KeyError(msg)
    return loader


class _DeterministicDCIC(torch.utils.data.Dataset):
    """Wraps a ``DCICDataset`` with a deterministic class label ordering.

    Upstream ``DCICDataset`` builds ``label_mappings`` from a
    ``set()`` whose iteration order depends on PYTHONHASHSEED, so the
    column index of ``targets`` can flip across processes. This
    wrapper enumerates labels by ``str(label)`` order, rebuilds
    ``targets`` to match, and exposes ``image_paths``, ``targets``,
    ``label_mappings``, ``num_classes``, and a ``transform`` slot
    just like the upstream class.

    The wrapped instance is mutated in place — we do not copy ``image_labels``
    or the underlying tensors, only re-derive ``label_mappings`` and
    ``targets``. This keeps memory flat and is safe because
    :class:`DCICDataset` is one-shot constructed in our pipeline.
    """

    def __init__(
        self,
        wrapped: DCICDataset,
        *,
        first_order: bool = True,
    ) -> None:
        self._wrapped = wrapped
        self._first_order = bool(first_order)
        unique_labels = sorted(
            {label for labels in wrapped.image_labels.values() for label in labels},
            key=str,
        )
        self.label_mappings: dict[Any, int] = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.num_classes: int = len(unique_labels)
        self.image_paths: list[str] = list(wrapped.image_paths)
        self.targets: list[torch.Tensor] = []
        for img_path in self.image_paths:
            labels = wrapped.image_labels[img_path]
            label_indices = [self.label_mappings[label] for label in labels]
            dist = torch.bincount(
                torch.tensor(label_indices), minlength=self.num_classes
            ).float()
            dist /= dist.sum()
            if self._first_order:
                self.targets.append(dist)
            else:
                self.targets.append(torch.multinomial(dist, 1).squeeze())

    @property
    def transform(self) -> Callable[..., Any] | None:
        """Pass-through to the wrapped instance's transform slot."""
        return self._wrapped.transform

    @transform.setter
    def transform(self, value: Callable[..., Any] | None) -> None:
        self._wrapped.transform = value

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        # Reuse the wrapped __getitem__ for image loading + transforms,
        # but substitute our deterministic-ordering target.
        image, _ = self._wrapped[index]
        return image, self.targets[index]


def _build_loader(
    dataset_name: str,
    data_root: Path,
    transform: Callable[..., Any] | None,
) -> _DeterministicDCIC:
    """Instantiate ``dataset_name``'s loader against ``data_root`` with ``transform``.

    ``data_root`` is the parent dir, e.g. ``data/`` — the loader
    appends ``<DatasetName>/annotations.json``.
    """
    loader_cls = _resolve_loader(dataset_name)
    raw = loader_cls(Path(data_root), transform=transform)
    return _DeterministicDCIC(raw)


def _extract_fold_name(image_path: str) -> str:
    """Return the fold name from a relative image path.

    Mirrors :func:`experiments.first_order_data.dcic_ensemble_pipeline.extract_fold_name`
    (file:line ``dcic_ensemble_pipeline.py:310``): the second path
    component of the image path is the DCIC fold name. E.g.
    ``"Plankton/fold1/img.png"`` -> ``"fold1"``.
    """
    parts = Path(image_path).parts
    return parts[1] if len(parts) > 1 else "unknown_fold"


def fold_indices(loader: _DeterministicDCIC) -> dict[str, list[int]]:
    """Group dataset indices by fold name from the image paths.

    The fold name is extracted from the second component of each
    image path (e.g. ``"Plankton/part1/img.png"`` -> ``"part1"``).
    Datasets without a fold component fall under ``"unknown_fold"``.
    """
    out: dict[str, list[int]] = {}
    for index, image_path in enumerate(loader.image_paths):
        fold = _extract_fold_name(image_path)
        out.setdefault(fold, []).append(index)
    return out


def split_test_train_indices(
    loader: _DeterministicDCIC,
    seed: int,
) -> tuple[str, list[int], list[int]]:
    """Pick the test fold per :func:`test_fold_for_seed` and partition indices.

    Returns ``(test_fold, train_indices, test_indices)``. The train
    indices include every image whose fold is not the test fold, in
    dataset order. ``train_indices`` is NOT split into train+val
    here; callers that need an early-stopping val set call
    :func:`split_train_val_indices` on the returned ``train_indices``.

    Raises:
        ValueError: If the loader has no folds at all (no image path
            has a second path component).
        KeyError: If the seed-derived test fold has no images, which
            would only happen if the dataset's fold metadata is
            corrupted.
    """
    indices_by_fold = fold_indices(loader)
    fold = pick_test_fold(seed, indices_by_fold.keys())
    test_indices = indices_by_fold.get(fold, [])
    if not test_indices:
        msg = (
            f"DCIC fold {fold!r} (derived from seed={seed}) is empty; "
            f"available folds with sizes: "
            f"{ {k: len(v) for k, v in indices_by_fold.items()}!r}."
        )
        raise KeyError(msg)
    train_indices = [
        idx
        for f, indices in indices_by_fold.items()
        if f != fold
        for idx in indices
    ]
    return fold, train_indices, test_indices


def split_train_val_indices(
    train_indices: Iterable[int],
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Carve a held-out validation slice from ``train_indices``.

    Mirrors ``experiments/first_order_data/dcic_ensemble_pipeline.py:338``'s
    ``split_train_validation_indices``: a seeded permutation, the
    last ``val_fraction`` of the permuted indices is the val set.
    Used by the basecls early-stopping flow.
    """
    indices = list(train_indices)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(indices))
    split = int(len(indices) * (1 - float(val_fraction)))
    shuffled = [indices[p] for p in perm]
    return shuffled[:split], shuffled[split:]


def build_transform(*, train: bool) -> T.Compose:
    """Build the locked DCIC image transform.

    Mirrors :func:`experiments.first_order_data.dcic_ensemble_pipeline.build_transform`:
    ``RandomHorizontalFlip`` (train only) -> ``Resize(224, 224)`` ->
    ``ToTensor`` -> ImageNet normalisation. The ``ToTensor +
    Normalize`` step happens at the end so PIL inputs flow through
    the pipeline correctly.
    """
    steps: list[Any] = []
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps.extend(
        [
            T.Resize((_RESNET18_INPUT_SIZE, _RESNET18_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    return T.Compose(steps)


@dataclass
class _DataLoaderProvider:
    """A FeatureProvider-shaped wrapper around a torch ``DataLoader``.

    Quacks as a :class:`FeatureProvider` (``__iter__``, ``__len__``,
    ``mode``, ``n_classes``, ``indices``, ``feature_dim``,
    ``n_samples``) but defers image loading to the DataLoader workers
    rather than preloading every batch into memory at fp32. The
    DCIC datasets at 224x224 would otherwise consume several GB of
    RAM per provider just to hold the input tensors.
    """

    dataloader: DataLoader[Any]
    n_classes: int
    indices: np.ndarray
    mode: Literal["full_network", "linear_probe"] = "full_network"
    feature_dim: int | None = None
    batches: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)

    @property
    def n_samples(self) -> int:
        return int(self.indices.shape[0])


def _make_provider(
    dataset_name: str,
    data_root: Path,
    indices: list[int],
    *,
    train: bool,
    augment: bool,
    batch_size: int,
    num_workers: int,
) -> FeatureProvider:
    """Build a FeatureProvider for a slice of a DCIC dataset.

    A fresh loader is instantiated with the appropriate transform
    (train-mode includes augmentation; test/val-mode does not), then
    Subset'd to ``indices`` and wrapped in a torch DataLoader. The
    returned object quacks as a FeatureProvider via :class:`_DataLoaderProvider`.
    """
    transform = build_transform(train=augment)
    loader = _build_loader(dataset_name, data_root, transform=transform)
    if not indices:
        msg = (
            f"empty index list for DCIC dataset {dataset_name!r}; "
            f"the loader has {len(loader)} total images but the seed-derived "
            f"slice was empty. Check the fold mapping and dataset on-disk layout."
        )
        raise ValueError(msg)
    subset = Subset(loader, list(indices))
    pin_memory = bool(torch.cuda.is_available())
    dataloader: DataLoader[Any] = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=bool(train and augment),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    provider = _DataLoaderProvider(
        dataloader=dataloader,
        n_classes=int(loader.num_classes),
        indices=np.asarray(indices, dtype=np.int64),
        mode="full_network",
        feature_dim=None,
    )
    return cast("FeatureProvider", provider)


def build_test_provider(
    dataset_config: dict[str, Any],
    seed: int,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
) -> FeatureProvider:
    """Build a test-time provider for the seed-derived test fold.

    No augmentation, no shuffling. Suitable for
    :mod:`extract_uncertainties` (test-time inference) and for
    p* sidecar prep.

    Args:
        dataset_config: Resolved DCIC dataset config (must declare
            ``loader_kwargs.root`` and ``loader.class``).
        seed: Run seed; the test fold is :func:`test_fold_for_seed`
            of this.
        batch_size: Per-DataLoader-batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        A :class:`FeatureProvider`-shaped object whose batches yield
        ``(images_3HW_float, soft_labels_K_float)`` tuples.
    """
    data_root = _data_root_from_config(dataset_config)
    dataset_name = _dataset_name_from_config(dataset_config)
    loader = _build_loader(dataset_name, data_root, transform=None)
    _, _, test_indices = split_test_train_indices(loader, seed)
    return _make_provider(
        dataset_name,
        data_root,
        test_indices,
        train=False,
        augment=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_full_train_provider(
    dataset_config: dict[str, Any],
    seed: int,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
) -> FeatureProvider:
    """Build a train-time provider over ALL non-test-fold images.

    No val split. Used by the from-scratch UQ training paths
    (ensemble per-member, evidential, ddu) which do not perform
    early stopping. Shuffling is enabled when ``augment=True``.
    """
    data_root = _data_root_from_config(dataset_config)
    dataset_name = _dataset_name_from_config(dataset_config)
    loader = _build_loader(dataset_name, data_root, transform=None)
    _, train_indices, _ = split_test_train_indices(loader, seed)
    return _make_provider(
        dataset_name,
        data_root,
        train_indices,
        train=True,
        augment=augment,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_train_val_providers(
    dataset_config: dict[str, Any],
    seed: int,
    *,
    val_fraction: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[FeatureProvider, FeatureProvider]:
    """Build (train, val) providers from the non-test fold images.

    Used by the basecls trainer: 4 folds -> 90/10 train/val split,
    early stopping watches val loss. The val provider is built with
    ``augment=False`` so val_loss is comparable across epochs.
    """
    data_root = _data_root_from_config(dataset_config)
    dataset_name = _dataset_name_from_config(dataset_config)
    loader = _build_loader(dataset_name, data_root, transform=None)
    _, train_indices, _ = split_test_train_indices(loader, seed)
    train_idx, val_idx = split_train_val_indices(
        train_indices, seed=seed, val_fraction=val_fraction
    )
    train_provider = _make_provider(
        dataset_name,
        data_root,
        train_idx,
        train=True,
        augment=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_provider = _make_provider(
        dataset_name,
        data_root,
        val_idx,
        train=False,
        augment=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_provider, val_provider


def make_resnet18_factory(
    num_classes: int,
    *,
    pretrained: bool = True,
) -> Callable[[], nn.Module]:
    """Build a callable that constructs a torchvision ResNet-18 with K-class head.

    The encoder is initialised from ``ResNet18_Weights.IMAGENET1K_V1``
    when ``pretrained=True`` (matches Oleg's reference recipe in
    :mod:`experiments.first_order_data.dcic_ensemble_pipeline`'s
    :class:`ImageEntmaxClassifier`); the final fully connected layer
    is replaced with a fresh ``nn.Linear(512, num_classes)``.

    Args:
        num_classes: Output dimension of the classifier head.
        pretrained: Whether to load ImageNet-pretrained weights for
            the encoder. ``False`` is provided mainly for tests; the
            production recipe is ``True``.

    Returns:
        A zero-argument callable returning a fresh
        :class:`torch.nn.Module`. The returned module's ``forward``
        emits raw logits of shape ``(B, num_classes)``.
    """

    def factory() -> nn.Module:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, int(num_classes))
        return model

    return factory


def _data_root_from_config(dataset_config: dict[str, Any]) -> Path:
    """Resolve the on-disk parent directory the loader expects.

    The DCIC loader classes expect ``root`` to be the parent dir of
    ``<DatasetName>/`` (e.g. pass ``data/`` for ``Plankton``). The
    dataset config's ``loader_kwargs.root`` follows that convention.
    """
    raw = dataset_config.get("loader_kwargs", {}).get("root")
    if not raw:
        msg = (
            "DCIC dataset config is missing the required "
            "'loader_kwargs.root' field (parent dir of <DatasetName>/)."
        )
        raise KeyError(msg)
    return Path(raw)


def _dataset_name_from_config(dataset_config: dict[str, Any]) -> str:
    """Read the DCIC class name from the dataset config.

    The pipeline's ``dataset_config['name']`` is the YAML-friendly
    lowercase name (e.g. ``"plankton"``); the upstream loader
    expects the canonical DCIC dataset name (e.g. ``"Plankton"``,
    ``"Treeversity#1"``). The config carries the canonical name in
    ``loader.dcic_name``.
    """
    loader_block = dataset_config.get("loader", {}) or {}
    dcic_name = loader_block.get("dcic_name")
    if not dcic_name:
        msg = (
            "DCIC dataset config is missing the required "
            "'loader.dcic_name' field (canonical DCIC name, e.g. 'Plankton', "
            "'Treeversity#1'). The pipeline's lower-case 'name' field is "
            "for run-id and config dispatch only."
        )
        raise KeyError(msg)
    if dcic_name not in DCIC_LOADERS:
        msg = (
            f"unknown DCIC dataset {dcic_name!r}; expected one of "
            f"{sorted(DCIC_LOADERS)!r}."
        )
        raise KeyError(msg)
    return str(dcic_name)


__all__ = [
    "DCIC_LOADERS",
    "DCIC_NUM_FOLDS",
    "build_full_train_provider",
    "build_test_provider",
    "build_train_val_providers",
    "build_transform",
    "fold_indices",
    "make_resnet18_factory",
    "pick_test_fold",
    "split_test_train_indices",
    "split_train_val_indices",
]
