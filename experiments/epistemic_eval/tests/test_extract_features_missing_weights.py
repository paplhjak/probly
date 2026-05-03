"""Tests for the ``weights_path`` failure mode of extract_features."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts" / "extract_features.py"


def test_missing_weights_exit_one(tmp_path: Path) -> None:
    """Running with a non-existent ``weights_path`` exits 1 and prints the message."""
    weights_path = tmp_path / "does_not_exist.pth"
    config = {
        "name": "appa_real",
        "loader": {"module": "probly.datasets.appa_real", "class": "AppaReal"},
        "loader_kwargs": {"root": "data/appa_real", "split": "test"},
        "supported_losses": ["squared", "absolute"],
        "extraction_mode": "linear_probe",
        "backbone": {
            "architecture": "torchvision.models.resnet101",
            "weights_path": str(weights_path),
            "feature_dim": 2048,
        },
        "head": {"architecture": "probly.method.head.MlpHead", "hidden": 256},
    }
    config_path = tmp_path / "appa_real.yaml"
    config_path.write_text(yaml.safe_dump(config))

    cache_dir = tmp_path / "feature_cache"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--config",
            str(config_path),
            "--cache-dir",
            str(cache_dir),
            "--splits",
            "test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    combined = result.stderr + result.stdout
    assert "APPA-REAL backbone weights not found" in combined
    assert "decisions.md" in combined
