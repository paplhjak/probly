"""Tests for the squared-error oracle.

E* is identically zero for the squared-error oracle on a first-order
dataset, per Definition 1 of paper_tex/sections/preliminaries.tex and
the "Frequentist Paradox" passage at lines 41-42 of that file: when
the conditional p*(y | x) is known, the Bayes-optimal predictor is
itself the oracle and the posterior over theta collapses, so the
expected regret is zero by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.oracle import compute_oracle, dense_from_sparse


SUPPORT = np.array([20, 30, 40, 50])


def test_squared_oracle_canonical_two_point_mass() -> None:
    """Hand-computed: p*=[0.5, 0.5, 0, 0], support=[20, 30, 40, 50].

    H_star = 20*0.5 + 30*0.5 = 25.
    A_star = (20-25)^2*0.5 + (30-25)^2*0.5 = 12.5 + 12.5 = 25.
    E_star = 0 (Frequentist Paradox).
    """
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    out = compute_oracle(p_star, "squared", SUPPORT)
    np.testing.assert_allclose(out["H_star"], [25.0], atol=1e-6)
    np.testing.assert_allclose(out["A_star"], [25.0], atol=1e-6)
    np.testing.assert_allclose(out["E_star"], [0.0], atol=0.0)
    assert out["H_star"].dtype == np.float32
    assert out["A_star"].dtype == np.float32
    assert out["E_star"].dtype == np.float32


def test_squared_oracle_uniform_distribution() -> None:
    """Uniform sanity: p*=[0.25, 0.25, 0.25, 0.25] over [20,30,40,50].

    H_star = 35.
    Var = E[y^2] - E[y]^2 = (400+900+1600+2500)/4 - 35^2 = 1350 - 1225 = 125.
    """
    p_star = np.array([[0.25, 0.25, 0.25, 0.25]])
    out = compute_oracle(p_star, "squared", SUPPORT)
    np.testing.assert_allclose(out["H_star"], [35.0], atol=1e-6)
    np.testing.assert_allclose(out["A_star"], [125.0], atol=1e-6)
    np.testing.assert_allclose(out["E_star"], [0.0], atol=0.0)


def test_squared_oracle_e_star_is_identically_zero() -> None:
    """E* is identically zero for any p* under squared loss.

    Per Definition 1 of preliminaries.tex and the Frequentist Paradox
    passage: with p*(y|x) known there is no posterior over theta, so
    the regret of the Bayes-optimal predictor against itself is zero.
    """
    rng = np.random.default_rng(0)
    p_star = rng.dirichlet(np.ones(4), size=10)
    out = compute_oracle(p_star, "squared", SUPPORT)
    assert np.all(out["E_star"] == 0.0)
    assert out["E_star"].shape == (10,)


def test_squared_oracle_handles_unnormalised_p_star() -> None:
    """Inputs are renormalised internally.

    Counts that happen to encode a [0.5, 0.5, 0, 0] mass should give
    the same answer as the explicit probability vector.
    """
    p_star = np.array([[2.0, 2.0, 0.0, 0.0]])
    out = compute_oracle(p_star, "squared", SUPPORT)
    np.testing.assert_allclose(out["H_star"], [25.0], atol=1e-6)
    np.testing.assert_allclose(out["A_star"], [25.0], atol=1e-6)


def test_squared_oracle_rejects_negative_p_star() -> None:
    p_star = np.array([[0.5, -0.1, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        compute_oracle(p_star, "squared", SUPPORT)


def test_squared_oracle_rejects_zero_row() -> None:
    p_star = np.array([[0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="strictly positive row sum"):
        compute_oracle(p_star, "squared", SUPPORT)


def test_squared_oracle_rejects_missing_support() -> None:
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="requires a `support`"):
        compute_oracle(p_star, "squared", None)


def test_squared_oracle_rejects_support_length_mismatch() -> None:
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="length 3"):
        compute_oracle(p_star, "squared", np.array([20, 30, 40]))


def test_dense_from_sparse_round_trip_with_oracle() -> None:
    """The sparse helper produces the same oracle result as the dense path."""
    sparse = [{20: 5, 30: 5}, {40: 1, 50: 3}]
    dense = dense_from_sparse(sparse, SUPPORT, normalize=True)
    out = compute_oracle(dense, "squared", SUPPORT)
    # Row 0: p* = [0.5, 0.5, 0, 0] -> mean 25, var 25.
    # Row 1: p* = [0, 0, 0.25, 0.75] -> mean 47.5, var = (40-47.5)^2*0.25 + (50-47.5)^2*0.75 = 14.0625 + 4.6875 = 18.75.
    np.testing.assert_allclose(out["H_star"], [25.0, 47.5], atol=1e-6)
    np.testing.assert_allclose(out["A_star"], [25.0, 18.75], atol=1e-6)
