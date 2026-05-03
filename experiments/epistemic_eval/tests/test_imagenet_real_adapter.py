"""Tests for :mod:`imagenet_real_adapter`.

The wrapped :class:`probly.datasets.torch.ImageNetReaL` constructor
needs the real ImageNet val tarball at ``$root/val``, so we patch it
out and inject a small mock with hand-picked ``dists`` to make the
empty-detection logic deterministic.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

if TYPE_CHECKING:
    pass

# conftest.py prepends experiments/epistemic_eval/scripts/ to sys.path,
# so this resolves to that file.
import imagenet_real_adapter as adapter_module  # noqa: E402


_NUM_CLASSES = 4


class _FakeImageNetReaL:
    """Stand-in for :class:`probly.datasets.torch.ImageNetReaL`."""

    def __init__(self) -> None:
        self.classes = list(range(_NUM_CLASSES))
        # Build five distributions:
        #   0: one-hot at class 0
        #   1: uniform 1/C (empty-label)
        #   2: one-hot at class 2
        #   3: uniform 1/C (empty-label)
        #   4: one-hot at class 3
        self.dists = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.ones(_NUM_CLASSES) / _NUM_CLASSES,
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.ones(_NUM_CLASSES) / _NUM_CLASSES,
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        ]
        # Use a sentinel "image" object per index that's easy to compare.
        self._items = [(f"image-{i}", self.dists[i]) for i in range(5)]

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


def _make_adapter() -> adapter_module.ImageNetReaLDropEmpty:
    fake = _FakeImageNetReaL()
    with patch.object(adapter_module, "ImageNetReaL", return_value=fake):
        return adapter_module.ImageNetReaLDropEmpty(root="unused")


def test_adapter_drops_empty_label_images() -> None:
    adapter = _make_adapter()
    assert len(adapter) == 3
    assert adapter.num_dropped == 2
    assert adapter.valid_indices == [0, 2, 4]


def test_adapter_index_mapping_returns_non_empty_examples() -> None:
    adapter = _make_adapter()
    # adapter[0] -> wrapped[0] (one-hot at 0)
    image_0, dist_0 = adapter[0]
    assert image_0 == "image-0"
    assert torch.equal(dist_0, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # adapter[1] -> wrapped[2] (one-hot at 2), NOT wrapped[1] (uniform)
    image_1, dist_1 = adapter[1]
    assert image_1 == "image-2"
    assert torch.equal(dist_1, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    # adapter[2] -> wrapped[4] (one-hot at 3)
    image_2, dist_2 = adapter[2]
    assert image_2 == "image-4"
    assert torch.equal(dist_2, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_adapter_classes_property_forwards_to_wrapped() -> None:
    adapter = _make_adapter()
    assert list(adapter.classes) == list(range(_NUM_CLASSES))


def test_adapter_module_imported_via_conftest_path() -> None:
    # Sanity check: make sure conftest actually placed the scripts dir
    # on sys.path; otherwise the bare-name import above silently picks
    # up something unrelated.
    assert any("epistemic_eval/scripts" in p for p in sys.path)
