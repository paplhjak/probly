"""ImageNet-ReaL adapter that drops images with empty ReaL label sets.

probly's :class:`probly.datasets.torch.ImageNetReaL` returns a uniform
distribution over all classes for images with empty ReaL labels (see
``src/probly/datasets/torch.py:78``). Per ``decisions.md`` "p*(y | x)
construction" for ImageNet-ReaL, we drop these images instead. This
adapter is the project's intended-conformance layer: it exposes a
contiguous ``[0, len(self))`` index space mapped to the underlying
non-empty samples.

Index mapping is precomputed at construction (``valid_indices``);
``__getitem__(i)`` returns the i-th NON-EMPTY example, not the i-th
example of the wrapped dataset filtered post-hoc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.utils.data

from probly.datasets.torch import ImageNetReaL

if TYPE_CHECKING:
    from collections.abc import Sequence


class ImageNetReaLDropEmpty(torch.utils.data.Dataset):
    """Wrap :class:`probly.datasets.torch.ImageNetReaL`, dropping empty-label images.

    Attributes:
        valid_indices: Indices into the wrapped dataset that survive
            the empty-label filter, in ascending order.
        num_dropped: Number of images excluded due to empty ReaL labels.
    """

    valid_indices: list[int]
    num_dropped: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Construct the wrapped :class:`ImageNetReaL` and precompute the index map.

        All positional and keyword arguments are forwarded verbatim to
        :class:`probly.datasets.torch.ImageNetReaL`.
        """
        self._wrapped = ImageNetReaL(*args, **kwargs)
        # An image's ReaL label set was empty iff its dist is the
        # uniform tensor (probly assigns dist = ones(C) / C in that
        # branch). Detect by exact equality on float32 since probly
        # constructs the uniform vector with `ones(C)/C`, which is
        # bit-identical at the same dtype.
        #
        # Theoretical caveat: an image whose ReaL label set covered
        # every class would also normalise to `ones(C)/C` and be
        # falsely flagged as empty. No image in the released
        # ``real.json`` has all 1000 ImageNet labels assigned, so
        # this corner case does not arise in practice.
        num_classes = len(self._wrapped.classes)
        uniform_value = 1.0 / num_classes
        valid_indices: list[int] = []
        empty_count = 0
        for idx, dist in enumerate(self._wrapped.dists):
            if torch.all(dist == uniform_value):
                empty_count += 1
            else:
                valid_indices.append(idx)
        self.valid_indices = valid_indices
        self.num_dropped = empty_count

    def __len__(self) -> int:
        """Return the number of non-empty-label images."""
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        """Return the ``index``-th non-empty example.

        Args:
            index: Position in the contiguous ``[0, len(self))`` space.

        Returns:
            ``(image, dist)`` from the wrapped dataset at the
            corresponding ``valid_indices[index]``.
        """
        return self._wrapped[self.valid_indices[index]]

    @property
    def classes(self) -> Sequence[Any]:
        """Forward the wrapped dataset's class list."""
        return self._wrapped.classes
