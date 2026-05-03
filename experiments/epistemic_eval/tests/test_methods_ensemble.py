"""Tests for :mod:`experiments.epistemic_eval.methods.ensemble`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.methods import ensemble as ens_module  # noqa: E402
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
        dropout_p=0.1,
    )


def test_ensemble_fit_and_extract_shape_and_member_variability() -> None:
    """fit trains 3 distinct members; extract returns ``(N, K, 3)``."""
    provider = _make_provider(seed=0)
    method_config = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 3,
        "seed_strategy": "derived",
        "head_dropout_p": 0.1,
        "epochs": 2,
        "lr": 1.0e-2,
    }
    dataset_config = {
        "name": "synthetic",
        "extraction_mode": "linear_probe",
        "metadata": {"num_classes": _NUM_CLASSES},
    }
    handle = ens_module.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=_factory,
        seed=0,
    )
    assert len(handle.state_dicts) == 3
    assert handle.member_seeds == [226078449, 1561542571, 430596021]

    out = ens_module.extract(
        handle,
        provider,
        n_samples=None,
        model_factory=_factory,
    )
    assert set(out.keys()) == {"logits", "indices"}
    assert out["logits"].shape == (_NUM_SAMPLES, _NUM_CLASSES, 3)
    assert out["logits"].dtype == np.float32
    # Variability across members: trained from different seeds, must differ.
    assert not np.allclose(out["logits"][:, :, 0], out["logits"][:, :, 1])
    assert not np.allclose(out["logits"][:, :, 1], out["logits"][:, :, 2])


def test_ensemble_pretrained_path_raises_when_too_few_paths(tmp_path) -> None:
    """``n_members > len(ensemble_classifier_paths)`` raises clearly."""
    provider = _make_provider(seed=0)
    # Save 2 fake state dicts.
    paths = []
    for i in range(2):
        m = _factory()
        p = tmp_path / f"ckpt_{i}.pth"
        torch.save(m.state_dict(), p)
        paths.append(str(p))

    method_config = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 3,
        "head_dropout_p": 0.1,
    }
    dataset_config = {
        "name": "imagenet_real_like",
        "extraction_mode": "full_network",
        "ensemble_classifier_paths": paths,
    }
    with pytest.raises(ValueError, match="ensemble_classifier_paths"):
        ens_module.fit(
            method_config=method_config,
            dataset_config=dataset_config,
            data_provider=provider,
            model_factory=_factory,
            seed=0,
        )


def test_ensemble_save_load_roundtrip(tmp_path) -> None:
    """save + load reconstructs the handle."""
    provider = _make_provider(seed=0)
    method_config = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 2,
        "epochs": 1,
        "lr": 1.0e-2,
    }
    dataset_config = {"name": "synthetic", "extraction_mode": "linear_probe"}
    handle = ens_module.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=_factory,
        seed=0,
    )
    path = tmp_path / "method.pth"
    ens_module.save(handle, path)
    reloaded = ens_module.load(path)
    assert reloaded.member_seeds == handle.member_seeds
    assert len(reloaded.state_dicts) == len(handle.state_dicts)
