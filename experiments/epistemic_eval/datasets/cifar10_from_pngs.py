"""CIFAR-10 loader that reads raw PNGs instead of the canonical pickled batches.

Drop-in replacement for ``torchvision.datasets.CIFAR10`` for the
training half of the epistemic-eval pipeline. Bypasses the
canonical tarball entirely: torchvision's MD5 integrity check
rejects re-encoded batches even when the underlying image data is
correct, and the Toronto host frequently 503s. We instead read
class-organised PNG directories that we already have on disk.

Expected on-disk layout::

    <root>/
        train/
            Airplane/*.png
            Automobile/*.png
            Bird/*.png
            Cat/*.png
            Deer/*.png
            Dog/*.png
            Frog/*.png
            Horse/*.png
            Ship/*.png
            Truck/*.png
        test/
            Airplane/*.png
            ...

Class indices follow the canonical CIFAR-10 ordering documented in
:mod:`torchvision.datasets.cifar`::

    0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
    5 dog, 6 frog, 7 horse, 8 ship, 9 truck.

The directory names use the title-case spellings the project ships
(``Airplane`` rather than ``airplane``); the class index for an item
is determined by the directory name's position in :attr:`CLASSES`.

Per-instance ordering is deterministic across runs: items are
indexed first by class (in :attr:`CLASSES` order) and then by
filename within a class (lexicographic sort).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class CIFAR10FromPNGs:
    """Mimics the public surface of ``torchvision.datasets.CIFAR10``.

    Constructor kwargs ``(root, train, transform)`` plus the dataset
    protocol (``__len__``, ``__getitem__``) returning
    ``(PIL.Image.Image, int)``. Arguments other than the documented
    three are accepted as keyword-only but ignored, so callers that
    pass ``download=True`` (a torchvision idiom) still work.

    Attributes:
        root: Root directory containing ``train/`` and ``test/``
            subdirectories.
        split: ``"train"`` or ``"test"`` -- the chosen split.
        transform: Optional torchvision-style transform applied to
            each loaded ``PIL.Image.Image`` before it is returned.
        classes: The 10 directory names (title-cased) used as class
            buckets, in canonical CIFAR-10 order.
    """

    CLASSES: tuple[str, ...] = (
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    )

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Any = None,
        **_unused: Any,
    ) -> None:
        self.root = Path(root)
        self.split = "train" if train else "test"
        self.transform = transform
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            msg = (
                f"expected CIFAR-10 PNG split directory at {split_dir} "
                f"(layout: <root>/{self.split}/<ClassName>/*.png)."
            )
            raise FileNotFoundError(msg)
        items: list[tuple[Path, int]] = []
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                msg = (
                    f"expected class directory at {class_dir}; "
                    f"all 10 class directories must exist for split {self.split!r}."
                )
                raise FileNotFoundError(msg)
            paths = sorted(class_dir.glob("*.png"))
            if not paths:
                msg = f"no PNG files found in {class_dir}."
                raise FileNotFoundError(msg)
            for path in paths:
                items.append((path, class_idx))
        self._items: list[tuple[Path, int]] = items
        # Mirror torchvision's `classes` attribute so downstream code
        # that introspects the loader still works.
        self.classes: tuple[str, ...] = self.CLASSES

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self._items[index]
        with Image.open(path) as img:
            # Force a load so the file handle can close before transform;
            # avoids "too many open files" on the long training loop.
            img.load()
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
