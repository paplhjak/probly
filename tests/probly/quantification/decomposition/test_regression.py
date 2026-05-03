"""Tests for the squared and absolute regression decompositions."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.decomposition import (
    absolute_decomposition,
    squared_decomposition,
)


SUPPORT = np.array([20, 30, 40, 50], dtype=np.float64)


def _canonical_logits() -> np.ndarray:
    """Build the (N=2, K=4, S=3) fixture used across the regression tests.

    Probabilities:
        Point 0: all three samples have point mass at age 20.
        Point 1: s0=[0.5, 0.5, 0, 0], s1=[0, 0.5, 0.5, 0],
                 s2=[0, 0, 0.5, 0.5].

    Logits = log(probs + 1e-30) so softmax round-trips back to the
    intended probabilities (with sub-1e-3 numerical slack).
    """
    probs = np.zeros((2, 4, 3), dtype=np.float64)
    probs[0, 0, :] = 1.0
    probs[1, 0, 0] = 0.5
    probs[1, 1, 0] = 0.5
    probs[1, 1, 1] = 0.5
    probs[1, 2, 1] = 0.5
    probs[1, 2, 2] = 0.5
    probs[1, 3, 2] = 0.5
    return np.log(probs + 1e-30).astype(np.float32)


def test_squared_decomposition_canonical_fixture() -> None:
    """Hand-computed Point-0 and Point-1 expected values.

    Point 0: H=20, A=0, E=0.
    Point 1: H=35, A=25, E=200/3 ~= 66.6667.
    """
    logits = _canonical_logits()
    out = squared_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(out["H_hat"], [20.0, 35.0], atol=1e-3)
    np.testing.assert_allclose(out["A_hat"], [0.0, 25.0], atol=1e-3)
    np.testing.assert_allclose(out["E_hat"], [0.0, 200.0 / 3.0], atol=1e-3)
    assert out["H_hat"].dtype == np.float32
    assert out["A_hat"].dtype == np.float32
    assert out["E_hat"].dtype == np.float32


def test_squared_decomposition_law_of_total_variance() -> None:
    """A_hat + E_hat equals the marginal mixture variance (L2 only).

    This is the law of total variance applied to the empirical
    mixture over the S posterior samples; it provides a clean
    correctness check for the implementation. **It is NOT a test of
    the paper's T* = A* + E* identity** (which holds for any loss by
    Definition 1 of paper_tex/sections/preliminaries.tex). It is a
    test that the implementation correctly decomposes into
    within-sample variance and cross-sample mean dispersion.
    """
    logits = _canonical_logits()
    out = squared_decomposition(logits, SUPPORT)
    # Point 1 marginal mixture: {20: 1/6, 30: 2/6, 40: 2/6, 50: 1/6}.
    # Var = E[y^2] - E[y]^2 = 7900/6 - 35^2 = 1316.6667 - 1225 = 91.6667.
    t_via_mixture = 91.0 + 2.0 / 3.0
    a_plus_e = float(out["A_hat"][1] + out["E_hat"][1])
    assert abs(t_via_mixture - a_plus_e) < 1e-3


def test_absolute_decomposition_canonical_fixture() -> None:
    """Hand-computed Point-0 and Point-1 expected values.

    Point 0: H=20, A=0, E=0.
    Point 1: H=30, A=5, E=20/3 ~= 6.6667.

    The H_hat aggregation is the lower median over S; for S=3 sorted
    medians {20, 30, 40} the lower median is index (S-1)//2 = 1 -> 30.
    """
    logits = _canonical_logits()
    out = absolute_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(out["H_hat"], [20.0, 30.0], atol=1e-3)
    np.testing.assert_allclose(out["A_hat"], [0.0, 5.0], atol=1e-3)
    np.testing.assert_allclose(out["E_hat"], [0.0, 20.0 / 3.0], atol=1e-3)


def test_absolute_decomposition_no_law_of_total_deviation() -> None:
    """A_hat + E_hat does NOT equal the marginal mixture MAD (L1).

    This test asserts a significantly nonzero gap (in either direction
    -- the gap can be negative; we assert magnitude only). It reflects
    the absence of a law-of-total-deviation for L1. **It is NOT a
    violation of the paper's T* = A* + E* identity**, which holds for
    any loss by Definition 1 of
    paper_tex/sections/preliminaries.tex. The assertion exists so
    future code cannot accidentally enforce a bias-variance-style
    identity in the L1 case.
    """
    logits = _canonical_logits()
    out = absolute_decomposition(logits, SUPPORT)
    # Point 1: H_hat = 30. Marginal mixture MAD around 30:
    # (|20-30| + |30-30|*2 + |40-30|*2 + |50-30|) / 6 = (10 + 0 + 20 + 20) / 6 = 50 / 6.
    t_via_mixture = 50.0 / 6.0
    a_plus_e = float(out["A_hat"][1] + out["E_hat"][1])
    assert abs(t_via_mixture - a_plus_e) > 1.0


def test_squared_decomposition_rejects_non_3d_logits() -> None:
    bad = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3-D"):
        squared_decomposition(bad, SUPPORT)


def test_absolute_decomposition_rejects_non_3d_logits() -> None:
    bad = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3-D"):
        absolute_decomposition(bad, SUPPORT)


def test_regression_decompositions_reject_support_mismatch() -> None:
    logits = _canonical_logits()
    bad_support = np.array([20, 30, 40], dtype=np.float64)
    with pytest.raises(ValueError, match="length 3"):
        squared_decomposition(logits, bad_support)
    with pytest.raises(ValueError, match="length 3"):
        absolute_decomposition(logits, bad_support)


def test_regression_decompositions_reject_nan_logits() -> None:
    logits = _canonical_logits().copy()
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        squared_decomposition(logits, SUPPORT)
    with pytest.raises(ValueError, match="finite"):
        absolute_decomposition(logits, SUPPORT)


def test_regression_decompositions_reject_inf_logits() -> None:
    logits = _canonical_logits().copy()
    logits[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        squared_decomposition(logits, SUPPORT)
