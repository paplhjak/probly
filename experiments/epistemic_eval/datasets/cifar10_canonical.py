"""Thin subclass of ``torchvision.datasets.CIFAR10`` that bypasses MD5 checks.

The project ships canonical-format CIFAR-10 pickles built locally
from raw PNGs (see
``experiments/epistemic_eval/scripts/build_canonical_cifar10_pickles.py``).
The image data is the same as the canonical Toronto tarball but
the pickle bytes are not byte-identical, so torchvision's
hardcoded MD5 list rejects the files even though they are
functionally correct.

This class overrides ``_check_integrity`` to verify only that the
expected files exist on disk; download is disabled (the canonical
tarball is unreliable from this network anyway). Use it everywhere
the project would otherwise call ``torchvision.datasets.CIFAR10``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torchvision


class CIFAR10NoMD5(torchvision.datasets.CIFAR10):
    """Drop-in for :class:`torchvision.datasets.CIFAR10` without MD5 verification.

    The pickled batches under ``<root>/cifar-10-batches-py/`` must
    follow the canonical layout (str-keyed dicts with ``"data"``,
    ``"labels"``, ``"filenames"``, ``"batch_label"``) -- see the
    docstring of
    ``experiments.epistemic_eval.scripts.build_canonical_cifar10_pickles``.

    Test-set ordering follows canonical CIFAR-10 (matching the
    indexing of ``cifar10h-counts.npy``); train-set ordering is
    deterministic but not canonical (training loops shuffle anyway).
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Any = None,
        target_transform: Any = None,
        download: bool = False,
    ) -> None:
        if download:
            msg = (
                "CIFAR10NoMD5 does not support download=True; the canonical "
                "tarball is unreliable. Run "
                "experiments/epistemic_eval/scripts/build_canonical_cifar10_pickles.py "
                "to materialise the pickles from raw PNGs first."
            )
            raise ValueError(msg)
        super().__init__(
            root=str(root),
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

    def _check_integrity(self) -> bool:
        # Verify only that the expected files exist; ignore canonical
        # MD5s since our pickles are bit-different from the tarball
        # while encoding the same image data.
        base = Path(self.root) / self.base_folder
        expected = [name for name, _md5 in (self.train_list + self.test_list)]
        expected.append(self.meta["filename"])
        return all((base / f).is_file() for f in expected)

    def _load_meta(self) -> None:
        # Same shape as the parent's :meth:`_load_meta` but without
        # ``check_integrity`` against the canonical meta MD5.
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not Path(path).is_file():
            msg = f"meta file missing at {path}; run build_canonical_cifar10_pickles.py."
            raise FileNotFoundError(msg)
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
