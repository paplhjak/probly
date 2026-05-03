"""Tests for :class:`experiments.epistemic_eval.methods._base.FeatureProvider`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402


def _build_full_network_provider() -> FeatureProvider:
    """Build a small ``full_network`` provider (8 images, 4 classes)."""
    images = torch.randn(8, 3, 8, 8)
    labels = torch.zeros(8, 4)
    labels[torch.arange(8), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])] = 1.0
    batches = [(images[:4], labels[:4]), (images[4:], labels[4:])]
    return FeatureProvider(
        batches=batches,
        mode="full_network",
        feature_dim=None,
        n_classes=4,
        indices=np.arange(8, dtype=np.int64),
    )


def _build_linear_probe_provider() -> FeatureProvider:
    """Build a small ``linear_probe`` provider (8 features of dim 16)."""
    features = torch.randn(8, 16)
    labels = torch.zeros(8, 3)
    labels[torch.arange(8), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])] = 1.0
    batches = [(features[:4], labels[:4]), (features[4:], labels[4:])]
    return FeatureProvider(
        batches=batches,
        mode="linear_probe",
        feature_dim=16,
        n_classes=3,
        indices=np.arange(8, dtype=np.int64),
    )


def test_full_network_provider_iterates() -> None:
    """Iteration yields image batches with the right shape."""
    provider = _build_full_network_provider()
    assert provider.mode == "full_network"
    assert provider.feature_dim is None
    assert provider.n_classes == 4
    assert len(provider) == 2
    assert provider.n_samples == 8
    seen = list(provider)
    assert seen[0][0].shape == (4, 3, 8, 8)
    assert seen[0][1].shape == (4, 4)
    assert seen[1][0].shape == (4, 3, 8, 8)


def test_linear_probe_provider_iterates() -> None:
    """Iteration yields feature batches with the right shape."""
    provider = _build_linear_probe_provider()
    assert provider.mode == "linear_probe"
    assert provider.feature_dim == 16
    assert provider.n_classes == 3
    assert len(provider) == 2
    assert provider.n_samples == 8
    for x, y in provider:
        assert x.shape[1] == 16
        assert y.shape[1] == 3


def test_provider_indices_are_traceable() -> None:
    """``indices`` is a per-sample numpy array of dataset indices."""
    provider = _build_linear_probe_provider()
    assert isinstance(provider.indices, np.ndarray)
    assert provider.indices.dtype == np.int64
    assert provider.indices.tolist() == list(range(8))
