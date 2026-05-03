"""End-to-end smoke test for the extraction layer.

Composes :mod:`train_classifier` (``--smoke-test-synthetic``),
:mod:`fit_uncertainty`, and :mod:`extract_uncertainties` via in-process
calls (the orchestrator path is exercised separately by
``test_run_grid_dry_run.py`` if/when it lands). This test exists so
the architecture can be probed by the reviewer's acid test:

    "swap the method config and adding a new method should require
     ZERO changes to the orchestrator scripts."

We exercise the linear-probe branch with a synthetic feature cache so
the smoke test runs in seconds without torchvision data.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import yaml  # noqa: E402

from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = _REPO_ROOT / "experiments" / "epistemic_eval" / "scripts"


def test_train_smoke_test_synthetic(tmp_path: Path) -> None:
    """``train_classifier.py --smoke-test-synthetic`` writes a classifier.pth."""
    run_dir = tmp_path / "run_train"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "train_classifier.py"),
            "--smoke-test-synthetic",
            "--run-dir",
            str(run_dir),
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (run_dir / "classifier.pth").exists()
    assert (run_dir / "training_log.csv").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "meta.json").exists()


def _make_feature_cache(cache_dir: Path, n: int, dim: int, seed: int = 0) -> None:
    """Write a synthetic features .npz at the cache path."""
    g = np.random.default_rng(seed)
    features = g.standard_normal((n, dim)).astype(np.float32)
    filenames = np.asarray([f"sample_{i}" for i in range(n)], dtype=object)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "train.npz", features=features, filenames=filenames)
    np.savez_compressed(cache_dir / "test.npz", features=features, filenames=filenames)


def test_fit_and_extract_linear_probe(tmp_path: Path) -> None:
    """Run fit_uncertainty.py + extract_uncertainties.py end-to-end."""
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=24, dim=16)
    dataset_cfg = {
        "name": "synthetic_appa",
        "loader": {"module": "probly.datasets.appa_real", "class": "AppaReal"},
        "loader_kwargs": {"root": "data/appa_real"},
        "supported_losses": ["squared"],
        "extraction_mode": "linear_probe",
        "backbone": {
            "architecture": "torchvision.models.resnet101",
            "weights_path": str(tmp_path / "ignored.pth"),
            "feature_dim": 16,
        },
        "head": {"architecture": "probly.method.head.MlpHead", "hidden": 8},
        "metadata": {"num_classes": 4},
    }
    dataset_config_path = tmp_path / "dataset.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_cfg))

    method_cfg = {
        "name": "mc_dropout",
        "method_module": "experiments.epistemic_eval.methods.mc_dropout",
        "n_samples": 3,
        "head_dropout_p": 0.5,
        "classifier_dropout_p": 0.1,
        "epochs": 1,
        "lr": 1.0e-2,
    }
    method_config_path = tmp_path / "method.yaml"
    method_config_path.write_text(yaml.safe_dump(method_cfg))

    run_dir = tmp_path / "run"
    fit_result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "fit_uncertainty.py"),
            "--method-config",
            str(method_config_path),
            "--dataset-config",
            str(dataset_config_path),
            "--seed",
            "1",
            "--run-dir",
            str(run_dir),
            "--feature-cache-dir",
            str(feature_cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fit_result.returncode == 0, fit_result.stderr + fit_result.stdout
    assert (run_dir / "method.pth").exists()
    assert (run_dir / "config.yaml").exists()

    extract_result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "extract_uncertainties.py"),
            "--run",
            str(run_dir),
            "--feature-cache-dir",
            str(feature_cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert extract_result.returncode == 0, extract_result.stderr + extract_result.stdout
    pred_path = run_dir / "predictions.npz"
    assert pred_path.exists()
    blob = np.load(pred_path)
    assert blob["logits"].shape == (24, 4, 3)
    assert blob["logits"].dtype == np.float32
    assert blob["indices"].shape == (24,)
    assert blob["indices"].dtype == np.int64


def test_swap_method_changes_no_other_files(tmp_path: Path) -> None:
    """Acid test: switching method config from MC-Dropout to Ensemble works
    without changing any of the scripts."""
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=16, dim=8)
    dataset_cfg = {
        "name": "synthetic_appa",
        "loader": {"module": "probly.datasets.appa_real", "class": "AppaReal"},
        "loader_kwargs": {"root": "data/appa_real"},
        "supported_losses": ["squared"],
        "extraction_mode": "linear_probe",
        "backbone": {
            "architecture": "torchvision.models.resnet101",
            "weights_path": str(tmp_path / "ignored.pth"),
            "feature_dim": 8,
        },
        "head": {"architecture": "probly.method.head.MlpHead", "hidden": 4},
        "metadata": {"num_classes": 3},
    }
    dataset_config_path = tmp_path / "dataset.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_cfg))

    method_cfg = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 2,
        "head_dropout_p": 0.1,
        "classifier_dropout_p": 0.0,
        "epochs": 1,
        "lr": 1.0e-2,
    }
    method_config_path = tmp_path / "method.yaml"
    method_config_path.write_text(yaml.safe_dump(method_cfg))

    run_dir = tmp_path / "run"
    fit_result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "fit_uncertainty.py"),
            "--method-config",
            str(method_config_path),
            "--dataset-config",
            str(dataset_config_path),
            "--seed",
            "0",
            "--run-dir",
            str(run_dir),
            "--feature-cache-dir",
            str(feature_cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fit_result.returncode == 0, fit_result.stderr + fit_result.stdout
    extract_result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "extract_uncertainties.py"),
            "--run",
            str(run_dir),
            "--feature-cache-dir",
            str(feature_cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert extract_result.returncode == 0, extract_result.stderr + extract_result.stdout
    blob = np.load(run_dir / "predictions.npz")
    # n_members=2, so S=2.
    assert blob["logits"].shape == (16, 3, 2)


def test_run_grid_dry_run(tmp_path: Path) -> None:
    """run_grid.py --dry-run prints the expected number of subprocess invocations."""
    grid = {
        "methods": ["mc_dropout", "ensemble"],
        "datasets": ["cifar10h"],
        "seeds": [0],
    }
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid))
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "run_grid.py"),
            "--config",
            str(grid_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # 2 methods x 1 dataset x 1 seed x 3 stages = 6 commands printed.
    nonempty = [line for line in result.stdout.strip().split("\n") if line.strip()]
    assert len(nonempty) == 6
