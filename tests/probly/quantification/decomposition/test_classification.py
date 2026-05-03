"""Tests for the classification decompositions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.quantification.decomposition import (
    SecondOrderEntropyDecomposition,
    SecondOrderZeroOneDecomposition,
    cross_entropy_decomposition,
    zero_one_decomposition,
)
from probly.representation._helpers import compute_mean_probs
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)


SUPPORT = np.array([20, 30, 40, 50], dtype=np.float64)


def _canonical_logits() -> np.ndarray:
    """Same (N=2, K=4, S=3) probability fixture as test_regression.py."""
    probs = np.zeros((2, 4, 3), dtype=np.float64)
    probs[0, 0, :] = 1.0
    probs[1, 0, 0] = 0.5
    probs[1, 1, 0] = 0.5
    probs[1, 1, 1] = 0.5
    probs[1, 2, 1] = 0.5
    probs[1, 2, 2] = 0.5
    probs[1, 3, 2] = 0.5
    return np.log(probs + 1e-30).astype(np.float32)


def _direct_torch_decomposition(
    logits: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute via probly's torch primitives directly (the wrapper's truth)."""
    logits_t = torch.from_numpy(np.ascontiguousarray(logits))
    probs_NSK = torch.softmax(logits_t, dim=1).permute(0, 2, 1).contiguous()
    distribution = TorchCategoricalDistribution(unnormalized_probabilities=probs_NSK)
    sample = TorchCategoricalDistributionSample(tensor=distribution, sample_dim=1)
    ent = SecondOrderEntropyDecomposition(distribution=sample)
    zo = SecondOrderZeroOneDecomposition(distribution=sample)
    bma = compute_mean_probs(sample)
    return ent.aleatoric, ent.epistemic, zo.aleatoric, bma.probabilities


def test_cross_entropy_decomposition_matches_probly_primitive() -> None:
    """The wrapper produces the same A_hat / E_hat as the primitive call.

    This pins the wrapper's behaviour to probly's
    ``SecondOrderEntropyDecomposition`` so a future refactor of either
    side will be caught.
    """
    logits = _canonical_logits()
    ent_a, ent_e, _, bma_probs = _direct_torch_decomposition(logits)
    out = cross_entropy_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(
        out["A_hat"], ent_a.detach().cpu().numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        out["E_hat"], ent_e.detach().cpu().numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        out["H_hat"], bma_probs.detach().cpu().numpy(), atol=1e-6
    )
    assert out["H_hat"].shape == (2, 4)
    assert out["H_hat"].dtype == np.float32
    assert out["A_hat"].dtype == np.float32
    assert out["E_hat"].dtype == np.float32


def test_zero_one_decomposition_matches_probly_primitive() -> None:
    logits = _canonical_logits()
    _, _, zo_a, bma_probs = _direct_torch_decomposition(logits)
    out = zero_one_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(
        out["A_hat"], zo_a.detach().cpu().numpy(), atol=1e-6
    )
    expected_argmax = bma_probs.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    np.testing.assert_array_equal(out["H_hat"], expected_argmax)
    assert out["H_hat"].dtype == np.int64
    assert out["A_hat"].dtype == np.float32
    assert out["E_hat"].dtype == np.float32


def test_cross_entropy_decomposition_rejects_non_3d_logits() -> None:
    bad = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3-D"):
        cross_entropy_decomposition(bad, SUPPORT)


def test_zero_one_decomposition_rejects_non_3d_logits() -> None:
    bad = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3-D"):
        zero_one_decomposition(bad, SUPPORT)


def test_classification_decompositions_accept_no_support() -> None:
    """``support`` is optional for classification losses (it is ignored)."""
    logits = _canonical_logits()
    cross_entropy_decomposition(logits, None)
    zero_one_decomposition(logits, None)


def test_classification_decompositions_reject_support_mismatch() -> None:
    logits = _canonical_logits()
    bad_support = np.array([0, 1, 2], dtype=np.float64)
    with pytest.raises(ValueError, match="length 3"):
        cross_entropy_decomposition(logits, bad_support)
    with pytest.raises(ValueError, match="length 3"):
        zero_one_decomposition(logits, bad_support)
