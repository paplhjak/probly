"""Smoke test for the CIFAR-10H training loop.

Exercises ``_train_cifar10h_run`` end-to-end on a small synthetic
CIFAR-shaped dataset by patching ``torchvision.datasets.CIFAR10``
with a stand-in that yields PIL images of the right shape and
``int`` labels in ``[0, 10)``. The real ResNet18, SGD, cosine
scheduler, augmentation pipeline, and val-split logic all run --
the only thing replaced is the data source, so we don't download
170 MB of CIFAR-10 in CI.

The user runs the actual 200-epoch training manually after this
test confirms the loop is structurally correct; per the brief,
"DO NOT add a real-data test that downloads CIFAR-10."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
pytest.importorskip("torchvision")

from experiments.epistemic_eval.scripts.train_classifier import (  # noqa: E402
    _train_cifar10h_run,
)


class _SyntheticCIFAR10:
    """Stand-in for ``torchvision.datasets.CIFAR10``.

    Mimics the public surface used by the training loop: constructor
    kwargs ``(root, train, transform, download)`` and the dataset
    protocol (``__len__``, ``__getitem__``) returning ``(PIL.Image,
    int)``. Deterministic per-instance via a hash of ``train``.
    """

    def __init__(
        self,
        root: str,  # noqa: ARG002
        train: bool,
        transform: Any = None,
        download: bool = False,  # noqa: ARG002
    ) -> None:
        self.transform = transform
        rng = np.random.default_rng(0 if train else 1)
        n = 64 if train else 16
        self._data = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
        self._targets = rng.integers(0, 10, size=(n,), dtype=np.int64).tolist()

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image  # local import; deferred to call time

        img = Image.fromarray(self._data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self._targets[index])


def _make_smoke_config(*, epochs: int = 2, batch_size: int = 8) -> dict[str, Any]:
    """Build a minimal CIFAR-10H-shaped config for the smoke test."""
    return {
        "name": "cifar10h_smoke",
        "loader_kwargs": {"root": "data/cifar10_smoke"},
        "classifier": {
            "architecture": "probly_benchmark.resnet.ResNet18",
            "num_classes": 10,
        },
        "training": {
            "optimizer": "sgd",
            "lr": 0.05,
            "momentum": 0.9,
            "weight_decay": 5.0e-4,
            "nesterov": False,
            "batch_size": batch_size,
            "epochs": epochs,
            "schedule": "cosine",
            "augmentation": {
                "random_crop": {"size": 32, "padding": 4},
                "horizontal_flip": True,
            },
            "normalization": {
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2023, 0.1994, 0.2010],
            },
        },
    }


def test_train_cifar10h_smoke(tmp_path: Path) -> None:
    """End-to-end: 2 epochs on synthetic CIFAR-shaped data; classifier.pth + log written."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _make_smoke_config(epochs=2, batch_size=8)

    with patch("torchvision.datasets.CIFAR10", _SyntheticCIFAR10):
        _train_cifar10h_run(config, run_dir, seed=0)

    classifier_path = run_dir / "classifier.pth"
    assert classifier_path.exists(), "classifier.pth not written"
    state = torch.load(classifier_path, map_location="cpu", weights_only=False)
    # probly_benchmark.resnet.ResNet18 ends with `linear` (a 10-class FC layer).
    assert "linear.weight" in state, f"unexpected state_dict keys: {list(state)[:5]}"
    assert state["linear.weight"].shape == (10, 512), (
        f"unexpected linear.weight shape: {state['linear.weight'].shape}"
    )

    log_path = run_dir / "training_log.csv"
    assert log_path.exists(), "training_log.csv not written"
    log_lines = log_path.read_text().strip().split("\n")
    # header + 2 epochs of rows.
    assert len(log_lines) == 3, f"expected 3 lines (header + 2 epochs), got {len(log_lines)}"
    assert log_lines[0] == "epoch,train_loss,val_loss,train_acc,val_acc"

    hash_path = run_dir / "classifier.config_hash"
    assert hash_path.exists(), "classifier.config_hash sidecar not written"


def test_train_cifar10h_idempotent(tmp_path: Path) -> None:
    """A second invocation with the same config skips retraining."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _make_smoke_config(epochs=2, batch_size=8)

    with patch("torchvision.datasets.CIFAR10", _SyntheticCIFAR10):
        _train_cifar10h_run(config, run_dir, seed=0)

    classifier_path = run_dir / "classifier.pth"
    mtime_first = classifier_path.stat().st_mtime_ns

    # Second invocation must be a no-op (same config, same seed).
    with patch("torchvision.datasets.CIFAR10", _SyntheticCIFAR10):
        _train_cifar10h_run(config, run_dir, seed=0)
    mtime_second = classifier_path.stat().st_mtime_ns

    assert mtime_first == mtime_second, (
        "second invocation rewrote classifier.pth; expected idempotent skip"
    )


def test_train_cifar10h_rejects_unknown_optimizer(tmp_path: Path) -> None:
    """The hyperparameter validator rejects unsupported optimizer values."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _make_smoke_config()
    config["training"]["optimizer"] = "adam"

    with patch("torchvision.datasets.CIFAR10", _SyntheticCIFAR10), pytest.raises(
        ValueError, match="unknown optimizer"
    ):
        _train_cifar10h_run(config, run_dir, seed=0)


def test_train_cifar10h_rejects_unknown_schedule(tmp_path: Path) -> None:
    """The hyperparameter validator rejects unsupported schedule values."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _make_smoke_config()
    config["training"]["schedule"] = "step"

    with patch("torchvision.datasets.CIFAR10", _SyntheticCIFAR10), pytest.raises(
        ValueError, match="unknown schedule"
    ):
        _train_cifar10h_run(config, run_dir, seed=0)


def test_train_cifar10h_rejects_wrong_num_classes(tmp_path: Path) -> None:
    """``ResNet18`` is hardcoded to 10 classes; mismatched config is rejected."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _make_smoke_config()
    config["classifier"]["num_classes"] = 100

    with patch("torchvision.datasets.CIFAR10", _SyntheticCIFAR10), pytest.raises(
        ValueError, match="hardcoded to 10 classes"
    ):
        _train_cifar10h_run(config, run_dir, seed=0)
