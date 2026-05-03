"""Structural test for :mod:`extract_features`.

This is a structural test, not a real-backbone test. It builds a tiny
backbone, saves its state-dict, points the script at it with a
synthetic dataset module, and asserts the cache file is created with
the right keys and shapes.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import yaml  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts" / "extract_features.py"


_FIXTURE_MODULE_TEMPLATE = """
\"\"\"Synthetic backbone fixture for extract_features structural tests.\"\"\"
from __future__ import annotations

from torch import nn


class TinyBackbone(nn.Module):
    \"\"\"Tiny CNN whose .fc gets replaced by Identity at use-time.\"\"\"

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 4)

    def forward(self, x):  # noqa: ANN001
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


def make_backbone():
    return TinyBackbone()


class SyntheticImageDataset:
    \"\"\"Fixed-size dataset of 6 random images.\"\"\"

    def __init__(self, *, root, split):  # noqa: ARG002
        import torch
        g = torch.Generator().manual_seed(0)
        self._items = [torch.randn(3, 8, 8, generator=g) for _ in range(6)]
        self.filenames = [f\"img_{i}.png\" for i in range(6)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):  # noqa: ANN204
        return (self._items[index], 0)
"""


def _write_fixture(tmp_path: Path) -> Path:
    """Write the fixture module under a tmp dir and prepend it to sys.path."""
    pkg_dir = tmp_path / "fixture_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "extract_features_fixture.py").write_text(_FIXTURE_MODULE_TEMPLATE)
    return pkg_dir


def test_extract_features_writes_cache(tmp_path: Path) -> None:
    """End-to-end: a tiny backbone produces a (6, 4) feature cache."""
    pkg_dir = _write_fixture(tmp_path)
    # Save the fixture state-dict at a known path.
    sys.path.insert(0, str(pkg_dir.parent))
    try:
        import importlib

        fixture = importlib.import_module("fixture_pkg.extract_features_fixture")
        backbone = fixture.make_backbone()
        weights_path = tmp_path / "weights.pth"
        torch.save(backbone.state_dict(), weights_path)
    finally:
        if str(pkg_dir.parent) in sys.path:
            sys.path.remove(str(pkg_dir.parent))

    config = {
        "name": "synthetic_for_extract_features",
        "loader": {
            "module": "fixture_pkg.extract_features_fixture",
            "class": "SyntheticImageDataset",
        },
        "loader_kwargs": {"root": str(tmp_path)},
        "supported_losses": ["squared"],
        "extraction_mode": "linear_probe",
        "backbone": {
            "architecture": "fixture_pkg.extract_features_fixture.make_backbone",
            "weights_path": str(weights_path),
            "feature_dim": 4,
        },
        "head": {"architecture": "probly.method.head.MlpHead", "hidden": 8},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    cache_dir = tmp_path / "cache"
    env = {**__import__("os").environ}
    # Ensure the fixture is importable in the subprocess.
    env["PYTHONPATH"] = str(pkg_dir.parent) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
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
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    cache_file = cache_dir / "test.npz"
    assert cache_file.exists()
    blob = np.load(cache_file, allow_pickle=True)
    assert "features" in blob.files
    assert "filenames" in blob.files
    features = blob["features"]
    assert features.shape == (6, 4)
    assert features.dtype == np.float32
    assert blob["filenames"].shape == (6,)
