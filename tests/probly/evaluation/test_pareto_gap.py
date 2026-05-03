"""Closed-form unit tests for IGD+ and the Pareto-gap diagnostic.

The IGD+ tests are exact: they cover identity, the direction of the
clipped distance ``d^+``, and a single-point translation. The
end-to-end :func:`pareto_gap` tests use small fixtures to verify
self-consistency (``A_hat = A_star``, ``E_hat = E_star``) and
positive-scale invariance (``A_hat = alpha A_star`` etc.) up to
grid spacing.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.pareto_gap import igd_plus, pareto_gap
from probly.evaluation.selectors import (
    sweep_empirical_surface,
    sweep_oracle_surface,
)


# ---------------------------------------------------------------------------
# IGD+ closed-form tests
# ---------------------------------------------------------------------------

def test_igd_plus_single_point_translation() -> None:
    # Test (i): one reference point, one achievable point shifted on
    # the first axis by 0.1. d+ = sqrt(0.1**2) = 0.1, IGD+ = 0.1.
    s_star = np.array([[0.0, 0.0, 0.0]])
    s = np.array([[0.1, 0.0, 0.0]])
    np.testing.assert_allclose(igd_plus(s_star, s), 0.1, rtol=1e-12, atol=1e-12)


def test_igd_plus_identity() -> None:
    # Test (ii).
    s = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_allclose(igd_plus(s, s), 0.0, rtol=1e-12, atol=1e-12)


def test_igd_plus_dominated_achievable_set_is_zero() -> None:
    # Test (iii) part one: achievable point dominates the reference
    # on every axis (lower is better), so d+ collapses to zero.
    s_star = np.array([[1.0, 1.0, 1.0]])
    s = np.array([[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(igd_plus(s_star, s), 0.0, rtol=1e-12, atol=1e-12)


def test_igd_plus_dominating_reference_gives_known_distance() -> None:
    # Test (iii) part two: achievable point is uniformly worse than
    # the reference by 0.5 on each axis, so d+ = sqrt(3 * 0.25).
    s_star = np.array([[0.0, 0.0, 0.0]])
    s = np.array([[0.5, 0.5, 0.5]])
    expected = float(np.sqrt(3.0 * 0.25))
    np.testing.assert_allclose(igd_plus(s_star, s), expected, rtol=1e-12, atol=1e-12)


def test_igd_plus_uses_min_over_achievable_set() -> None:
    # If at least one achievable point dominates a reference point,
    # d+ for that reference is zero; the per-reference contribution
    # to the mean must be exactly zero, not the average distance to
    # the rest of the set.
    s_star = np.array([[1.0, 1.0, 1.0]])
    s = np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(igd_plus(s_star, s), 0.0, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_igd_plus_empty_reference_raises() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        igd_plus(np.empty((0, 3)), np.array([[0.1, 0.2, 0.3]]))


def test_igd_plus_empty_achievable_raises() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        igd_plus(np.array([[0.1, 0.2, 0.3]]), np.empty((0, 3)))


def test_igd_plus_wrong_column_count_raises() -> None:
    with pytest.raises(ValueError, match="exactly 3 columns"):
        igd_plus(np.zeros((2, 4)), np.zeros((2, 3)))


def test_igd_plus_one_d_input_raises() -> None:
    with pytest.raises(ValueError, match=r"2-D"):
        igd_plus(np.zeros(3), np.zeros((1, 3)))


def test_igd_plus_nan_raises() -> None:
    bad = np.array([[float("nan"), 0.0, 0.0]])
    good = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        igd_plus(bad, good)


def test_pareto_gap_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        pareto_gap([], [], [], [])


def test_pareto_gap_nan_in_inputs_raises() -> None:
    a_hat = [0.1, 0.2, 0.3]
    e_hat = [0.4, 0.5, 0.6]
    a_star = [0.1, 0.2, 0.3]
    e_star = [0.4, float("nan"), 0.6]
    with pytest.raises(ValueError, match="finite"):
        pareto_gap(a_hat, e_hat, a_star, e_star)


def test_pareto_gap_too_small_grid_raises() -> None:
    a_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    e_star = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="n_lambda.*>= 2"):
        pareto_gap(a_star, e_star, a_star, e_star, n_lambda=1)
    with pytest.raises(ValueError, match="n_directions.*>= 2"):
        pareto_gap(a_star, e_star, a_star, e_star, n_directions=1)


# ---------------------------------------------------------------------------
# Pareto-gap end-to-end tests
# ---------------------------------------------------------------------------

# Anti-correlated 5-point fixture used by the self-consistency and
# scale-invariance tests.
_A_STAR = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
_E_STAR = np.array([0.5, 0.4, 0.3, 0.2, 0.1])


def test_pareto_gap_self_consistency() -> None:
    # Test (iv): when (A_hat, E_hat) == (A_star, E_star), the
    # empirical sweep over theta in [0, 2 pi) traces every line
    # orientation in (A_star, E_star) space; the resulting achievable
    # surface is therefore a superset of the oracle arc (which uses
    # (1 - lambda, lambda) for lambda in [1/2, 1], a quarter-circle's
    # worth of directions) modulo finite grid alignment. The IGD+ gap
    # is bounded by the angular grid spacing of
    # 2 pi / n_directions ~ 6.9e-2 radians at n_directions=91; we
    # allow up to 1e-3 here, well below that bound thanks to the
    # piecewise-linear smoothness of the achievable surface.
    gap = pareto_gap(_A_STAR, _E_STAR, _A_STAR, _E_STAR)
    assert gap < 1e-3


def test_pareto_gap_positive_scale_invariance() -> None:
    # Test (v): scaling the empirical components by positive
    # constants does not change the SET of selectors enumerated by
    # the empirical sweep, because as theta ranges over [0, 2 pi) the
    # effective weights (alpha * cos theta, beta * sin theta) trace
    # out a full ellipse that hits every line orientation in
    # (A_star, E_star) space (twice, once per half). The achievable
    # surface in operating-point space is therefore identical (up to
    # grid alignment) to the (A_hat = A_star, E_hat = E_star) case.
    alpha = 2.0
    beta = 3.0
    a_hat = alpha * _A_STAR
    e_hat = beta * _E_STAR
    gap = pareto_gap(a_hat, e_hat, _A_STAR, _E_STAR)
    assert gap < 1e-3


def test_pareto_gap_handles_sign_flipped_estimates() -> None:
    """Regression guard for the F1 fix to ``sweep_empirical_surface``.

    When ``E_hat = -E_star`` (estimates are perfect up to a sign on
    the epistemic axis), a method has all the right ranking
    information, just encoded with the opposite sign on E. The
    paper's selector definition
    ``S(A_hat, E_hat) = {1{w1 A_hat + w2 E_hat <= tau}}`` over
    ``(w1, w2, tau) in R^3`` recovers the correct ranking via
    ``(w1, w2) = (1, -1)``, i.e. direction
    ``theta = -pi/4 ~= 7 pi / 4``, which is in ``[0, 2 pi)``.

    Under the original (incorrect) half-circle sweep
    ``theta in [0, pi]``, that direction was unreachable, so the
    Pareto-gap was inflated. With the full-circle sweep, the gap is
    bounded by grid spacing, identical to the self-consistency case.
    """
    a_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    e_star = np.array([0.3, 0.1, 0.5, 0.2, 0.4])
    a_hat = a_star.copy()
    e_hat = -e_star
    gap = pareto_gap(a_hat, e_hat, a_star, e_star)
    assert gap < 1e-3


def test_pareto_gap_matches_explicit_pipeline() -> None:
    # Test (vi): pareto_gap deliberately does NOT dedupe or filter
    # the achievable surface. It must match an explicit pipeline of
    # sweep_oracle_surface -> sweep_empirical_surface -> coverage
    # flip -> igd_plus.
    rng = np.random.default_rng(1)
    n = 6
    a_star = rng.uniform(0.0, 1.0, size=n)
    e_star = rng.uniform(0.0, 1.0, size=n)
    a_hat = rng.uniform(0.0, 1.0, size=n)
    e_hat = rng.uniform(0.0, 1.0, size=n)

    n_lambda = 11
    n_directions = 9

    s_star = sweep_oracle_surface(a_star, e_star, n_lambda=n_lambda)
    s = sweep_empirical_surface(
        a_hat,
        e_hat,
        a_star,
        e_star,
        n_directions=n_directions,
    )
    s_star_flipped = s_star.copy()
    s_flipped = s.copy()
    s_star_flipped[:, 0] = 1.0 - s_star_flipped[:, 0]
    s_flipped[:, 0] = 1.0 - s_flipped[:, 0]
    expected = igd_plus(s_star_flipped, s_flipped)

    actual = pareto_gap(
        a_hat,
        e_hat,
        a_star,
        e_star,
        n_lambda=n_lambda,
        n_directions=n_directions,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_pareto_gap_returns_float() -> None:
    gap = pareto_gap(_A_STAR, _E_STAR, _A_STAR, _E_STAR)
    assert isinstance(gap, float)
