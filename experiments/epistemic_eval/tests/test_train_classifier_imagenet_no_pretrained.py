"""Tests for the ImageNet-ReaL no-pretrained failure path of train_classifier."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts" / "train_classifier.py"


def test_imagenet_no_pretrained_exits_one(tmp_path: Path) -> None:
    """Running with ``pretrained_classifier_path: null`` exits 1 and prints the message."""
    config = {
        "name": "imagenet_real",
        "loader": {
            "module": "experiments.epistemic_eval.scripts.imagenet_real_adapter",
            "class": "ImageNetReaLDropEmpty",
        },
        "loader_kwargs": {"root": "data/imagenet"},
        "supported_losses": ["cross_entropy"],
        "extraction_mode": "full_network",
        "classifier": {
            "architecture": "torchvision.models.resnet50",
            "pretrained_classifier_path": None,
            "num_classes": 1000,
        },
    }
    config_path = tmp_path / "imagenet_real.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--config",
            str(config_path),
            "--seed",
            "0",
            "--run-dir",
            str(run_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    combined = result.stderr + result.stdout
    assert "ImageNet-ReaL training from scratch is not supported" in combined
    assert "pretrained_classifier_path" in combined
    assert "decisions.md" in combined
