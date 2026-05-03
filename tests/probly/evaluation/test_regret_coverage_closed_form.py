"""Closed-form unit tests for AuReC, AuRC, and the underlying coverage curves.

The fixtures use small (5-point) hand-computable examples. The
expected AuReC values are exact rationals derived from the trapezoidal
rule on the ``(N + 1)``-point grid ``rho = [0, 1/N, ..., 1]``; we
compare with very tight tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.regret_coverage import (
    aurc,
    aurec,
    regret_coverage_curve,
    risk_coverage_curve,
)

# ---------------------------------------------------------------------------
# Closed-form value tests
# ---------------------------------------------------------------------------


def test_aurec_main_5_point_fixture() -> None:
    # Test A from the brief.
    score = [3, 1, 4, 1, 5]
    regret = [2, 5, 1, 3, 4]
    # Stable sort by score ascending picks indices [1, 3, 0, 2, 4]
    # so the accepted-regret order is [5, 3, 2, 1, 4]. The cumulative
    # sums (prefixed with 0) are [0, 5, 8, 10, 11, 15], which
    # divided by N=5 give Re = [0, 1.0, 1.6, 2.0, 2.2, 3.0]. With
    # delta-rho = 0.2, the trapezoidal rule yields 1.66.
    expected = 1.66
    np.testing.assert_allclose(aurec(score, regret), expected, rtol=1e-12, atol=1e-12)


def test_regret_coverage_curve_matches_hand_values() -> None:
    score = [3, 1, 4, 1, 5]
    regret = [2, 5, 1, 3, 4]
    rho, curve = regret_coverage_curve(score, regret)
    np.testing.assert_allclose(rho, np.linspace(0.0, 1.0, 6), rtol=1e-12, atol=1e-12)
    expected = np.array([0.0, 1.0, 1.6, 2.0, 2.2, 3.0])
    np.testing.assert_allclose(curve, expected, rtol=1e-12, atol=1e-12)


def test_aurec_ties_use_stable_sort() -> None:
    # Test B: identical scores; stable sort preserves input order.
    score = [1, 1, 1, 1, 1]
    regret = [5, 4, 3, 2, 1]
    # Re = [0, 1.0, 1.8, 2.4, 2.8, 3.0]; trapezoidal => 1.9.
    np.testing.assert_allclose(aurec(score, regret), 1.9, rtol=1e-12, atol=1e-12)


def test_aurec_all_zero_regret() -> None:
    # Test C.
    score = [0.5, 1.5, 2.5, 3.5, 4.5]
    regret = [0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(aurec(score, regret), 0.0, rtol=1e-12, atol=1e-12)


def test_aurec_monotone_triangular() -> None:
    # Test D.
    score = [1, 2, 3, 4, 5]
    regret = [1, 2, 3, 4, 5]
    # Re = [0, 0.2, 0.6, 1.2, 2.0, 3.0]; trapezoidal => 1.1.
    np.testing.assert_allclose(aurec(score, regret), 1.1, rtol=1e-12, atol=1e-12)


def test_aurec_single_point_nonzero() -> None:
    # Test E (positive).
    np.testing.assert_allclose(aurec([0.5], [3.0]), 1.5, rtol=1e-12, atol=1e-12)


def test_aurec_single_point_zero() -> None:
    # Test E (zero).
    np.testing.assert_allclose(aurec([0.0], [0.0]), 0.0, rtol=1e-12, atol=1e-12)


def test_aurc_parallel_to_aurec_on_triangular_fixture() -> None:
    # Test F: same shape as the AuReC monotone case but using `risk`.
    score = [1, 2, 3, 4, 5]
    risk = [1, 2, 3, 4, 5]
    np.testing.assert_allclose(aurc(score, risk), 1.1, rtol=1e-12, atol=1e-12)


def test_risk_coverage_curve_matches_regret_coverage_curve_when_inputs_are_equal() -> None:
    # AuRC and AuReC share the same plumbing; this guards against
    # accidental divergence.
    score = [3, 1, 4, 1, 5]
    values = [2, 5, 1, 3, 4]
    rho_re, curve_re = regret_coverage_curve(score, values)
    rho_ri, curve_ri = risk_coverage_curve(score, values)
    np.testing.assert_array_equal(rho_re, rho_ri)
    np.testing.assert_array_equal(curve_re, curve_ri)


# ---------------------------------------------------------------------------
# Boundary properties
# ---------------------------------------------------------------------------


def test_curve_endpoints() -> None:
    rng = np.random.default_rng(0)
    n = 17
    score = rng.normal(size=n)
    regret = rng.uniform(0.0, 5.0, size=n)
    rho, curve = regret_coverage_curve(score, regret)
    assert rho[0] == 0.0
    assert rho[-1] == 1.0
    np.testing.assert_allclose(curve[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(curve[-1], regret.mean(), rtol=1e-12, atol=1e-12)


def test_aurec_accepts_numpy_arrays_and_python_lists_identically() -> None:
    score_list = [3, 1, 4, 1, 5]
    regret_list = [2, 5, 1, 3, 4]
    score_arr = np.asarray(score_list, dtype=np.float64)
    regret_arr = np.asarray(regret_list, dtype=np.float64)
    a = aurec(score_list, regret_list)
    b = aurec(score_arr, regret_arr)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_score_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        aurec([], [])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        aurec([1, 2, 3], [1, 2])


def test_nan_in_score_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        aurec([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])


def test_inf_in_regret_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        aurec([1.0, 2.0, 3.0], [1.0, float("inf"), 3.0])


def test_two_d_score_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        aurec(np.zeros((2, 3)), np.zeros(6))


def test_complex_dtype_rejected() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        aurec(np.array([1.0 + 2.0j, 3.0, 4.0]), [1.0, 2.0, 3.0])


def test_risk_coverage_validation_parallels_regret_coverage() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        risk_coverage_curve([], [])
    with pytest.raises(ValueError, match="same length"):
        aurc([1, 2, 3], [1, 2])
