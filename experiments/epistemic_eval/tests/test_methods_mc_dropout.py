"""Tests for :mod:`experiments.epistemic_eval.methods.mc_dropout`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.methods import mc_dropout  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402
from probly.method.head import MlpHead  # noqa: E402


_NUM_FEATURES = 64
_NUM_CLASSES = 4
_NUM_SAMPLES = 32


def _make_provider(seed: int = 0) -> FeatureProvider:
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(_NUM_SAMPLES, _NUM_FEATURES, generator=g)
    labels = torch.zeros(_NUM_SAMPLES, _NUM_CLASSES)
    classes = torch.randint(0, _NUM_CLASSES, (_NUM_SAMPLES,), generator=g)
    labels[torch.arange(_NUM_SAMPLES), classes] = 1.0
    batches = [
        (features[i : i + 8], labels[i : i + 8]) for i in range(0, _NUM_SAMPLES, 8)
    ]
    return FeatureProvider(
        batches=batches,
        mode="linear_probe",
        feature_dim=_NUM_FEATURES,
        n_classes=_NUM_CLASSES,
        indices=np.arange(_NUM_SAMPLES, dtype=np.int64),
    )


def _factory() -> MlpHead:
    return MlpHead(
        in_features=_NUM_FEATURES,
        num_classes=_NUM_CLASSES,
        hidden=16,
        dropout_p=0.5,
    )


def test_mc_dropout_fit_and_extract_shape_and_variability() -> None:
    """fit + extract returns a (N, K, S) tensor that varies across S."""
    provider = _make_provider(seed=0)
    method_config = {
        "name": "mc_dropout",
        "method_module": "experiments.epistemic_eval.methods.mc_dropout",
        "n_samples": 4,
        "head_dropout_p": 0.5,
        "epochs": 2,
        "lr": 1.0e-2,
    }
    dataset_config = {
        "name": "synthetic",
        "extraction_mode": "linear_probe",
        "metadata": {"num_classes": _NUM_CLASSES},
    }
    handle = mc_dropout.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=_factory,
        seed=42,
    )
    assert isinstance(handle.state_dict, dict)
    assert handle.seed == 42

    out = mc_dropout.extract(
        handle,
        provider,
        n_samples=4,
        model_factory=_factory,
    )
    assert set(out.keys()) == {"logits", "indices"}
    assert out["logits"].shape == (_NUM_SAMPLES, _NUM_CLASSES, 4)
    assert out["logits"].dtype == np.float32
    assert out["indices"].shape == (_NUM_SAMPLES,)
    assert out["indices"].dtype == np.int64
    # Variability across S: with dropout_p=0.5 the per-sample logits
    # must differ across forward passes.
    assert not np.allclose(out["logits"][:, :, 0], out["logits"][:, :, 1])


def test_mc_dropout_save_and_load_roundtrip(tmp_path) -> None:
    """``save`` then ``load`` reproduces the handle."""
    provider = _make_provider(seed=0)
    method_config = {
        "name": "mc_dropout",
        "method_module": "experiments.epistemic_eval.methods.mc_dropout",
        "n_samples": 2,
        "head_dropout_p": 0.5,
        "epochs": 1,
        "lr": 1.0e-2,
    }
    dataset_config = {"name": "synthetic", "extraction_mode": "linear_probe"}
    handle = mc_dropout.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=_factory,
        seed=7,
    )
    path = tmp_path / "method.pth"
    mc_dropout.save(handle, path)
    reloaded = mc_dropout.load(path)
    assert reloaded.seed == 7
    assert set(reloaded.state_dict.keys()) == set(handle.state_dict.keys())
    for k in handle.state_dict:
        assert torch.equal(reloaded.state_dict[k], handle.state_dict[k])
