"""Closed-form unit tests for the Theorem 1 thresholded selectors.

The fixtures use a 5-point example chosen so that ranking by ``A``,
ranking by ``E``, and ranking by ``A + E`` all differ; this prevents
``lambda_ = 1/2`` and ``lambda_ = 1`` from accidentally agreeing on
small inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.selectors import (
    convex_selector,
    linear_selector,
    sweep_empirical_surface,
    sweep_oracle_surface,
    threshold_selector,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

A = [0.1, 0.5, 0.3, 0.8, 0.2]
E = [0.4, 0.1, 0.7, 0.2, 0.6]
# A and E are not rank-correlated:
#   A+E = [0.5, 0.6, 1.0, 1.0, 0.8]
#   sort by A : [0, 4, 2, 1, 3]
#   sort by E : [1, 3, 0, 4, 2]


# ---------------------------------------------------------------------------
# Endpoint identities
# ---------------------------------------------------------------------------


def test_convex_lambda_half_recovers_threshold_on_half_sum() -> None:
    # Sub-test (i): convex(lambda=0.5, tau=0.3) thresholds (A+E)/2
    # at 0.3, equivalent to threshold(A+E, 0.6).
    expected = np.array([True, True, False, False, False])
    np.testing.assert_array_equal(
        convex_selector(A, E, lambda_=0.5, tau=0.3),
        expected,
    )
    a_plus_e = np.asarray(A) + np.asarray(E)
    np.testing.assert_array_equal(
        threshold_selector(a_plus_e, tau=0.6),
        expected,
    )


def test_convex_lambda_one_recovers_threshold_on_E() -> None:
    # Sub-test (ii): convex(lambda=1.0, tau=0.5) == threshold(E, 0.5).
    expected = np.array([True, True, False, True, False])
    np.testing.assert_array_equal(
        convex_selector(A, E, lambda_=1.0, tau=0.5),
        expected,
    )
    np.testing.assert_array_equal(
        threshold_selector(E, tau=0.5),
        expected,
    )


def test_linear_recovers_threshold_on_sum() -> None:
    # Sub-test (iii): linear(w1=1, w2=1, tau=0.7) == threshold(A+E, 0.7).
    expected = np.array([True, True, False, False, False])
    np.testing.assert_array_equal(
        linear_selector(A, E, w1=1.0, w2=1.0, tau=0.7),
        expected,
    )
    np.testing.assert_array_equal(
        threshold_selector(np.asarray(A) + np.asarray(E), tau=0.7),
        expected,
    )


def test_lambda_half_and_lambda_one_disagree_on_unaligned_inputs() -> None:
    # Sub-test (iv): at tau=0.3, convex(0.5) thresholds (A+E)/2 <= 0.3
    # i.e. A+E <= 0.6 -> [T, T, F, F, F]; convex(1.0) thresholds
    # E <= 0.3 -> [F, T, F, T, F]. They differ, confirming the
    # fixture is non-degenerate.
    half = convex_selector(A, E, lambda_=0.5, tau=0.3)
    full = convex_selector(A, E, lambda_=1.0, tau=0.3)
    np.testing.assert_array_equal(half, np.array([True, True, False, False, False]))
    np.testing.assert_array_equal(full, np.array([False, True, False, True, False]))
    assert not np.array_equal(half, full)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_convex_lambda_below_zero_raises() -> None:
    with pytest.raises(ValueError, match=r"lambda_.*\[0, 1\]"):
        convex_selector(A, E, lambda_=-0.01, tau=0.5)


def test_convex_lambda_above_one_raises() -> None:
    with pytest.raises(ValueError, match=r"lambda_.*\[0, 1\]"):
        convex_selector(A, E, lambda_=1.01, tau=0.5)


def test_convex_lambda_endpoints_accepted() -> None:
    # No exception should be raised at the closed-interval endpoints.
    convex_selector(A, E, lambda_=0.0, tau=0.5)
    convex_selector(A, E, lambda_=1.0, tau=0.5)


def test_convex_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        convex_selector([0.1, 0.2], [0.3, 0.4, 0.5], lambda_=0.5, tau=0.5)


def test_convex_nan_in_inputs_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        convex_selector([0.1, float("nan"), 0.3], [0.4, 0.5, 0.6], lambda_=0.5, tau=0.5)


def test_convex_nan_in_lambda_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        convex_selector(A, E, lambda_=float("nan"), tau=0.5)


def test_threshold_empty_score_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        threshold_selector([], tau=0.5)


def test_threshold_inf_tau_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        threshold_selector([0.1, 0.2, 0.3], tau=float("inf"))


def test_linear_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        linear_selector([0.1, 0.2], [0.3, 0.4, 0.5], w1=1.0, w2=1.0, tau=0.5)


def test_linear_inf_weights_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        linear_selector(A, E, w1=float("inf"), w2=1.0, tau=0.5)


# ---------------------------------------------------------------------------
# Surface sweeps: shape contract + endpoints
# ---------------------------------------------------------------------------


def test_oracle_sweep_has_expected_shape() -> None:
    a_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    e_star = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    n_lambda = 11
    surface = sweep_oracle_surface(a_star, e_star, n_lambda=n_lambda)
    assert surface.shape == (n_lambda * (len(a_star) + 1), 3)
    # Every block starts at rho=0 and ends at rho=1.
    for k in range(n_lambda):
        block = surface[k * (len(a_star) + 1) : (k + 1) * (len(a_star) + 1)]
        assert block[0, 0] == 0.0
        assert block[-1, 0] == 1.0
        # Risk and regret start at zero.
        np.testing.assert_allclose(block[0, 1:], 0.0, atol=1e-12)
        # Endpoints reach the population means of (A+E) and E.
        np.testing.assert_allclose(block[-1, 1], a_star.mean() + e_star.mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(block[-1, 2], e_star.mean(), rtol=1e-12, atol=1e-12)


def test_empirical_sweep_has_expected_shape_and_endpoints() -> None:
    rng = np.random.default_rng(0)
    n = 7
    a_star = rng.uniform(0.0, 1.0, size=n)
    e_star = rng.uniform(0.0, 1.0, size=n)
    a_hat = rng.uniform(0.0, 1.0, size=n)
    e_hat = rng.uniform(0.0, 1.0, size=n)
    n_directions = 13
    surface = sweep_empirical_surface(a_hat, e_hat, a_star, e_star, n_directions=n_directions)
    assert surface.shape == (n_directions * (n + 1), 3)
    for k in range(n_directions):
        block = surface[k * (n + 1) : (k + 1) * (n + 1)]
        assert block[0, 0] == 0.0
        assert block[-1, 0] == 1.0
        np.testing.assert_allclose(block[-1, 1], a_star.mean() + e_star.mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(block[-1, 2], e_star.mean(), rtol=1e-12, atol=1e-12)


def test_oracle_sweep_rejects_too_small_grid() -> None:
    with pytest.raises(ValueError, match="n_lambda.*>= 2"):
        sweep_oracle_surface([0.1, 0.2], [0.3, 0.4], n_lambda=1)


def test_empirical_sweep_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        sweep_empirical_surface(
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
        )
