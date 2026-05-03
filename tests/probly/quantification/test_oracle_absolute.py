"""Tests for the absolute-error oracle.

E* is identically zero for the absolute-error oracle on a first-order
dataset, per Definition 1 of paper_tex/sections/preliminaries.tex and
the "Frequentist Paradox" passage at lines 41-42 of that file: when
the conditional p*(y | x) is known, the Bayes-optimal predictor is
itself the oracle and the posterior over theta collapses, so the
expected regret is zero by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.oracle import compute_oracle


SUPPORT = np.array([20, 30, 40, 50])
SUPPORT_FLOAT = np.array([20.0, 30.0, 40.0, 50.0])


def test_absolute_oracle_canonical_lower_median_tiebreak() -> None:
    """Hand-computed: p*=[0.5, 0.5, 0, 0], lower-median rule.

    CDF = [0.5, 1.0, 1.0, 1.0]; the smallest index whose CDF reaches
    0.5 is column 0, so H_star = support[0] = 20.
    A_star = |20 - 20|*0.5 + |30 - 20|*0.5 = 5.
    E_star = 0 (Frequentist Paradox).
    """
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    out = compute_oracle(p_star, "absolute", SUPPORT)
    assert int(out["H_star"][0]) == 20
    assert out["H_star"].dtype == np.int64
    np.testing.assert_allclose(out["A_star"], [5.0], atol=1e-6)
    np.testing.assert_allclose(out["E_star"], [0.0], atol=0.0)


def test_absolute_oracle_float_support_returns_float_h() -> None:
    """When ``support`` is float-valued, ``H_star`` is float32."""
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    out = compute_oracle(p_star, "absolute", SUPPORT_FLOAT)
    assert out["H_star"].dtype == np.float32
    np.testing.assert_allclose(out["H_star"], [20.0], atol=1e-6)


def test_absolute_oracle_uniform_distribution() -> None:
    """Uniform p*=[0.25, 0.25, 0.25, 0.25] over [20,30,40,50].

    CDF = [0.25, 0.5, 0.75, 1.0]; the smallest index reaching 0.5 is
    column 1, so H_star = 30.
    A_star = (|20-30| + |30-30| + |40-30| + |50-30|) * 0.25
            = (10 + 0 + 10 + 20) * 0.25 = 10.
    """
    p_star = np.array([[0.25, 0.25, 0.25, 0.25]])
    out = compute_oracle(p_star, "absolute", SUPPORT)
    assert int(out["H_star"][0]) == 30
    np.testing.assert_allclose(out["A_star"], [10.0], atol=1e-6)


def test_absolute_oracle_e_star_is_identically_zero() -> None:
    """E* is identically zero for any p* under absolute loss.

    Per Definition 1 of preliminaries.tex and the Frequentist Paradox
    passage.
    """
    rng = np.random.default_rng(1)
    p_star = rng.dirichlet(np.ones(4), size=10)
    out = compute_oracle(p_star, "absolute", SUPPORT)
    assert np.all(out["E_star"] == 0.0)


def test_absolute_oracle_rejects_support_length_mismatch() -> None:
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="length 3"):
        compute_oracle(p_star, "absolute", np.array([20, 30, 40]))


def test_absolute_oracle_rejects_missing_support() -> None:
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="requires a `support`"):
        compute_oracle(p_star, "absolute", None)


def test_absolute_oracle_lower_median_at_first_column() -> None:
    """When mass exceeds 0.5 at the first column, that's the median."""
    p_star = np.array([[0.6, 0.2, 0.1, 0.1]])
    out = compute_oracle(p_star, "absolute", SUPPORT)
    assert int(out["H_star"][0]) == 20


def test_absolute_oracle_rejects_unknown_loss() -> None:
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="unknown loss 'l1'"):
        compute_oracle(p_star, "l1", SUPPORT)
