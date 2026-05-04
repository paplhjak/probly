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


def test_make_member_subset_provider_drops_distinct_folds() -> None:
    """Members 0..N-1 each drop a different deterministic fold.

    Pin: per-member train rotation produces ``n_members`` subset
    providers whose held-out indices form a partition of the input
    rows, and each subset has size ~``(n_members - 1) / n_members``.
    """
    n_members = 5
    n_samples = 50  # divisible by n_members for a clean partition
    g = torch.Generator().manual_seed(0)
    features = torch.randn(n_samples, _NUM_FEATURES, generator=g)
    labels = torch.zeros(n_samples, _NUM_CLASSES)
    labels[:, 0] = 1.0
    batches = [(features[i : i + 8], labels[i : i + 8]) for i in range(0, n_samples, 8)]
    provider = FeatureProvider(
        batches=batches,
        mode="linear_probe",
        feature_dim=_NUM_FEATURES,
        n_classes=_NUM_CLASSES,
        indices=np.arange(n_samples, dtype=np.int64),
    )
    held_out_per_member: list[set[int]] = []
    kept_per_member: list[set[int]] = []
    for m in range(n_members):
        sub = ens_module._make_member_subset_provider(provider, m, n_members)
        kept = set(int(i) for i in sub.indices.tolist())
        kept_per_member.append(kept)
        held_out_per_member.append(set(range(n_samples)) - kept)
        # Each member trains on (n_members - 1)/n_members of the data
        # (exact for divisible cases).
        assert len(kept) == n_samples * (n_members - 1) // n_members, (m, len(kept))
    # Held-out folds partition the input rows.
    union = set().union(*held_out_per_member)
    assert union == set(range(n_samples))
    # No two members hold out the same fold.
    for i in range(n_members):
        for j in range(i + 1, n_members):
            assert held_out_per_member[i].isdisjoint(held_out_per_member[j]), (i, j)


def test_make_member_subset_provider_is_deterministic() -> None:
    """Two calls with the same args return the same kept indices."""
    provider = _make_provider(seed=0)
    sub_a = ens_module._make_member_subset_provider(provider, member_idx=2, n_members=5)
    sub_b = ens_module._make_member_subset_provider(provider, member_idx=2, n_members=5)
    np.testing.assert_array_equal(sub_a.indices, sub_b.indices)


def test_ensemble_fit_uses_rotation_when_flag_set() -> None:
    """``per_member_train_rotation: true`` makes members see different subsets.

    Pin: with rotation off, every member's recorded provider has the
    full N samples; with rotation on, members get distinct
    ``(n_members - 1)/n_members`` subsets that partition the input
    on the held-out folds.
    """
    provider = _make_provider(seed=0)
    seen_indices: list[np.ndarray] = []

    real_train = ens_module._train_member

    def recording_train_member(model, data_provider, method_config, seed, **kwargs):
        seen_indices.append(np.asarray(data_provider.indices))
        return real_train(model, data_provider, method_config, seed, **kwargs)

    method_config = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 5,
        "epochs": 1,
        "lr": 1.0e-2,
        "per_member_train_rotation": True,
    }
    dataset_config = {"name": "synthetic", "extraction_mode": "linear_probe"}

    import unittest.mock
    with unittest.mock.patch.object(ens_module, "_train_member", recording_train_member):
        ens_module.fit(
            method_config=method_config,
            dataset_config=dataset_config,
            data_provider=provider,
            model_factory=_factory,
            seed=0,
        )
    assert len(seen_indices) == 5
    # Each member saw a strict subset (rotation removes one fold).
    for arr in seen_indices:
        assert arr.shape[0] < _NUM_SAMPLES
    # Held-out indices across members partition the input.
    held_out = [set(range(_NUM_SAMPLES)) - set(arr.tolist()) for arr in seen_indices]
    union = set().union(*held_out)
    assert union == set(range(_NUM_SAMPLES))


def test_ensemble_fit_default_no_rotation() -> None:
    """Default behaviour (no rotation flag): every member sees the full provider.

    Pin back-compat with the CIFAR-10H/DCIC paths that don't opt in.
    """
    provider = _make_provider(seed=0)
    seen_lengths: list[int] = []

    real_train = ens_module._train_member

    def recording_train_member(model, data_provider, method_config, seed, **kwargs):
        seen_lengths.append(int(data_provider.indices.shape[0]))
        return real_train(model, data_provider, method_config, seed, **kwargs)

    method_config = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 3,
        "epochs": 1,
        "lr": 1.0e-2,
    }
    dataset_config = {"name": "synthetic", "extraction_mode": "linear_probe"}

    import unittest.mock
    with unittest.mock.patch.object(ens_module, "_train_member", recording_train_member):
        ens_module.fit(
            method_config=method_config,
            dataset_config=dataset_config,
            data_provider=provider,
            model_factory=_factory,
            seed=0,
        )
    assert seen_lengths == [_NUM_SAMPLES] * 3


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
