"""Conformance tests for the project's first-order dataset loaders.

For each loader (CIFAR10H, ImageNetReaL, AppaReal, ImageNetReaLDropEmpty)
we check that, when the underlying data is available:

- The class is constructible.
- ``__getitem__(0)`` returns a 2-tuple.
- The second element is a 1-D non-negative float tensor.
- A complementary attribute (``classes`` for classification loaders,
  ``age_support`` for AppaReal) exposes a sequence whose length matches
  the second element's length.

Loaders are skipped individually when their data is not present on
disk, so this file is safe to run on any developer machine. The
ImageNetReaLDropEmpty case also defensively skips when the experiment
script is unavailable -- we do NOT add the experiments scripts dir to
sys.path from within ``tests/probly/`` to keep library tests decoupled
from experiment code.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from probly.datasets.appa_real import AppaReal
from probly.datasets.torch import CIFAR10H, ImageNetReaL

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"


def _check_first_order_sample(image: object, dist: object, support: object) -> None:
    """Shared assertions for first-order ``(image, dist)`` samples."""
    # ``image`` is opaque (PIL image, tensor, etc.); we only assert
    # tuple-shape via the caller. The dist must be a 1-D non-negative
    # float tensor whose length matches the support sequence length.
    assert isinstance(dist, torch.Tensor), f"dist is not a tensor: {type(dist)!r}"
    assert dist.dim() == 1, f"dist must be 1-D, got shape {tuple(dist.shape)}"
    assert torch.is_floating_point(dist), f"dist is not float, got dtype {dist.dtype}"
    assert (dist >= 0).all(), "dist has negative entries"
    assert hasattr(support, "__len__"), "support attribute is not sized"
    assert len(support) == dist.shape[0], (  # type: ignore[arg-type]
        f"support length {len(support)} does not match dist length {dist.shape[0]}"  # type: ignore[arg-type]
    )
    # We do not assert tuple-of-2 here so this helper stays single-purpose.
    del image  # unused; the caller already saw it


def _assert_two_tuple(sample: object) -> tuple[object, object]:
    """Assert that ``sample`` is a 2-tuple and return its elements."""
    assert isinstance(sample, tuple), f"sample is not a tuple: {type(sample)!r}"
    assert len(sample) == 2, f"sample is not length 2: len={len(sample)}"
    return sample[0], sample[1]


def test_cifar10h_compliance() -> None:
    cifar_root = _DATA_DIR / "cifar10h"
    if not (cifar_root / "cifar-10h-master" / "data" / "cifar10h-counts.npy").is_file():
        pytest.skip("CIFAR-10H counts not present; run download_cifar10h.py to fetch.")
    if not (cifar_root / "cifar-10-batches-py").is_dir():
        pytest.skip("CIFAR-10 test split not present at data/cifar10h/.")
    try:
        dataset = CIFAR10H(root=str(cifar_root), download=False)
    except RuntimeError as exc:
        # ``probly.datasets.torch.CIFAR10H`` inherits torchvision's
        # canonical-MD5 check. Locally-built pickles (e.g. those
        # produced by experiments/epistemic_eval/scripts/
        # build_canonical_cifar10_pickles.py) encode the same image
        # data but are not byte-identical to the official tarball,
        # so the MD5 check rejects them. Skip rather than fail; the
        # epistemic-eval pipeline uses CIFAR10NoMD5 to bypass this.
        pytest.skip(f"CIFAR10H rejected the on-disk data ({exc!s}); skipping compliance check.")
    image, dist = _assert_two_tuple(dataset[0])
    _check_first_order_sample(image, dist, dataset.classes)


def test_imagenet_real_compliance() -> None:
    imagenet_root = _DATA_DIR / "imagenet"
    if not (imagenet_root / "reassessed-imagenet-master" / "real.json").is_file():
        pytest.skip("ImageNet-ReaL real.json not present at data/imagenet/.")
    dataset = ImageNetReaL(root=imagenet_root)
    image, dist = _assert_two_tuple(dataset[0])
    _check_first_order_sample(image, dist, dataset.classes)


def test_appa_real_compliance() -> None:
    appa_root = _DATA_DIR / "appa_real"
    if not (appa_root / "gt_test.csv").is_file():
        pytest.skip("APPA-REAL data not present; see download_appa_real.py.")
    dataset = AppaReal(root=appa_root, split="test")
    image, dist = _assert_two_tuple(dataset[0])
    _check_first_order_sample(image, dist, dataset.age_support)


def test_imagenet_real_drop_empty_compliance() -> None:
    try:
        adapter_module = importlib.import_module("experiments.epistemic_eval.scripts.imagenet_real_adapter")
    except ImportError as exc:
        pytest.skip(
            f"experiments.epistemic_eval.scripts.imagenet_real_adapter is not importable from this layout: {exc}"
        )
    imagenet_root = _DATA_DIR / "imagenet"
    if not (imagenet_root / "reassessed-imagenet-master" / "real.json").is_file():
        pytest.skip("ImageNet-ReaL real.json not present at data/imagenet/.")
    dataset = adapter_module.ImageNetReaLDropEmpty(root=imagenet_root)
    image, dist = _assert_two_tuple(dataset[0])
    _check_first_order_sample(image, dist, dataset.classes)
