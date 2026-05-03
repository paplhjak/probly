"""Tests for the classification oracles (cross-entropy, zero-one).

E* is identically zero for both classification oracles on a
first-order dataset, per Definition 1 of
paper_tex/sections/preliminaries.tex and the "Frequentist Paradox"
passage at lines 41-42 of that file: when the conditional p*(y | x)
is known, the Bayes-optimal predictor is itself the oracle and the
posterior over theta collapses, so the expected regret is zero by
construction.
"""

from __future__ import annotations

import math

import numpy as np

from probly.quantification.oracle import compute_oracle


def test_cross_entropy_oracle_h_star_is_p_star_itself() -> None:
    """H* under cross-entropy loss is the conditional p* itself.

    The Bayes-optimal soft predictor for log loss is the underlying
    distribution, so H_star == p_star.
    """
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    out = compute_oracle(p_star, "cross_entropy")
    np.testing.assert_allclose(out["H_star"], p_star, atol=1e-6)
    assert out["H_star"].dtype == np.float32
    np.testing.assert_allclose(out["A_star"], [math.log(2)], atol=1e-6)
    np.testing.assert_allclose(out["E_star"], [0.0], atol=0.0)


def test_cross_entropy_oracle_uniform_distribution() -> None:
    """Uniform over 4 classes -> A_star = log(4)."""
    p_star = np.array([[0.25, 0.25, 0.25, 0.25]])
    out = compute_oracle(p_star, "cross_entropy")
    np.testing.assert_allclose(out["A_star"], [math.log(4)], atol=1e-6)


def test_cross_entropy_oracle_e_star_is_identically_zero() -> None:
    rng = np.random.default_rng(2)
    p_star = rng.dirichlet(np.ones(5), size=12)
    out = compute_oracle(p_star, "cross_entropy")
    assert np.all(out["E_star"] == 0.0)


def test_zero_one_oracle_argmax_with_lowest_index_tiebreak() -> None:
    """H* under zero-one loss is the argmax (lowest-index tiebreak).

    For p*=[0.5, 0.5, 0, 0], numpy's argmax convention picks column 0.
    A_star = 1 - max_k p_star = 1 - 0.5 = 0.5.
    E_star = 0 (Frequentist Paradox).
    """
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    out = compute_oracle(p_star, "zero_one")
    assert int(out["H_star"][0]) == 0
    assert out["H_star"].dtype == np.int64
    np.testing.assert_allclose(out["A_star"], [0.5], atol=1e-6)
    np.testing.assert_allclose(out["E_star"], [0.0], atol=0.0)


def test_zero_one_oracle_clean_class() -> None:
    p_star = np.array([[0.0, 0.0, 1.0, 0.0]])
    out = compute_oracle(p_star, "zero_one")
    assert int(out["H_star"][0]) == 2
    np.testing.assert_allclose(out["A_star"], [0.0], atol=1e-6)


def test_zero_one_oracle_e_star_is_identically_zero() -> None:
    rng = np.random.default_rng(3)
    p_star = rng.dirichlet(np.ones(5), size=10)
    out = compute_oracle(p_star, "zero_one")
    assert np.all(out["E_star"] == 0.0)


def test_classification_oracles_ignore_support_when_provided() -> None:
    """Cross-entropy and zero-one ignore ``support`` but accept it."""
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    support = np.array([0, 1, 2, 3])
    out_ce = compute_oracle(p_star, "cross_entropy", support)
    out_zo = compute_oracle(p_star, "zero_one", support)
    # Recomputing without support should yield the same numbers.
    np.testing.assert_allclose(out_ce["H_star"], compute_oracle(p_star, "cross_entropy")["H_star"])
    np.testing.assert_allclose(out_zo["A_star"], compute_oracle(p_star, "zero_one")["A_star"])
