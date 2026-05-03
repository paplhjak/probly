"""Closed-form tests for the per-loss realized-regret functions.

Each loss has a hand-computed fixture covering several non-degenerate
rows; the helpers in
:mod:`probly.quantification.realized_regret` are asserted to match
the expected per-point regret to ``atol=1e-3``.

A pathological cross-entropy edge case asserts that clipping +
renormalising avoids ``+inf`` when ``h_hat`` is one-hot.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.realized_regret import compute_realized_regret


def test_squared_regret_matches_hand_computed_fixture() -> None:
    """Squared regret equals ``(H_hat_mean - H_star_mean)**2`` exactly.

    H_hat values in this fixture are constructed by choosing BMA rows
    whose mean equals each target H_hat; H_star analogous via p_star.
    """
    support = np.array([20, 30, 40, 50])
    # Each row's H_star_mean, derived as sum_k support[k] * p_star[i, k],
    # is 20, 30, 40, 50, 50 respectively (one-hot p*).
    p_star = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    # Each row's H_hat_mean is 25, 30, 35, 40, 45 respectively.
    bma = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    expected = np.array([25.0, 0.0, 25.0, 100.0, 25.0], dtype=np.float32)
    regret = compute_realized_regret(p_star, bma, "squared", support)
    np.testing.assert_allclose(regret, expected, atol=1e-3)


def test_absolute_regret_matches_hand_computed_fixture() -> None:
    """Absolute regret matches the lower-median formula exactly.

    H_hat values in this fixture are restricted to the support
    ``{20, 30, 40, 50}`` because they are derived as lower-medians of
    BMA distributions over that support.
    """
    support = np.array([20, 30, 40, 50])
    p_star = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.25, 0.5, 0.25, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    # Each row's lower-median: 50, 30, 40, 50, 20
    bma = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    expected = np.array([30.0, 0.0, 5.0, 5.0, 2.0], dtype=np.float32)
    regret = compute_realized_regret(p_star, bma, "absolute", support)
    np.testing.assert_allclose(regret, expected, atol=1e-3)


def test_zero_one_regret_matches_hand_computed_fixture() -> None:
    """Zero-one regret = ``max p_star - p_star[argmax(BMA)]``."""
    support = np.array([0, 1, 2, 3])
    p_star = np.array(
        [
            [0.5, 0.3, 0.2, 0.0],
            [0.5, 0.3, 0.2, 0.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.7, 0.2, 0.1, 0.0],
            [0.4, 0.4, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    # argmax(BMA) per row: 1, 0, 0, 2, 1
    bma = np.array(
        [
            [0.1, 0.5, 0.2, 0.2],
            [0.5, 0.3, 0.1, 0.1],
            [0.5, 0.2, 0.2, 0.1],
            [0.1, 0.2, 0.5, 0.2],
            [0.3, 0.4, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    expected = np.array([0.2, 0.0, 0.3, 0.6, 0.0], dtype=np.float32)
    regret = compute_realized_regret(p_star, bma, "zero_one", support)
    np.testing.assert_allclose(regret, expected, atol=1e-3)


def test_cross_entropy_regret_matches_hand_computed_fixture() -> None:
    """Cross-entropy regret = ``KL(p_star || h_hat)`` (natural log)."""
    support = np.array([0, 1, 2, 3])
    p_star = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    bma = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.3, 0.15, 0.05],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            np.log(2.5),
            0.0,
            0.0,
            np.log(2.0),
            0.4 * np.log(0.4 / 0.5)
            + 0.3 * np.log(1.0)
            + 0.2 * np.log(0.2 / 0.15)
            + 0.1 * np.log(0.1 / 0.05),
        ],
        dtype=np.float32,
    )
    regret = compute_realized_regret(p_star, bma, "cross_entropy", support)
    np.testing.assert_allclose(regret, expected, atol=1e-3)


def test_cross_entropy_regret_clip_one_hot_h_hat() -> None:
    """Pathological: H_hat puts all mass on one class, p* is split.

    After clip-and-renormalize with ``eps=1e-12``, the KL is finite
    and deterministic. With ``h_hat=[1, 0, 0, 0]`` clipped to
    ``[1, 1e-12, 1e-12, 1e-12]`` and renormalised by sum
    ``1 + 3e-12``, we get ``h_renorm[0] = 1 / (1 + 3e-12)`` and
    ``h_renorm[1] = 1e-12 / (1 + 3e-12)``. The KL evaluates to::

        0.5 * log(0.5 * (1 + 3e-12)) + 0.5 * log(0.5 * (1 + 3e-12) / 1e-12)
            ~= -0.347 + 13.469 ~= 13.122 nats.

    Finite (we don't return ``+inf``), large enough to flag the
    pathology, finite enough to aggregate downstream.
    """
    p_star = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
    h_hat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    support = np.array([0, 1, 2, 3], dtype=np.int64)
    regret = compute_realized_regret(p_star, h_hat, "cross_entropy", support)
    assert np.isfinite(regret).all()
    np.testing.assert_allclose(regret, 13.122, atol=5e-2)


def test_compute_realized_regret_dispatch_unknown() -> None:
    """Unknown loss strings raise ``ValueError``."""
    p_star = np.array([[1.0, 0.0]], dtype=np.float32)
    h_hat = np.array([[0.5, 0.5]], dtype=np.float32)
    support = np.array([0, 1])
    with pytest.raises(ValueError, match="unknown loss"):
        compute_realized_regret(p_star, h_hat, "kl_divergence", support)


def test_input_validation_nan_inf_and_shapes() -> None:
    """NaN, inf, wrong shapes, mismatched shapes -> ValueError or TypeError."""
    p_ok = np.array([[0.5, 0.5]], dtype=np.float32)
    h_ok = np.array([[0.5, 0.5]], dtype=np.float32)
    support = np.array([0, 1])

    # NaN p_star
    p_nan = np.array([[float("nan"), 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        compute_realized_regret(p_nan, h_ok, "cross_entropy", support)

    # inf h_hat
    h_inf = np.array([[float("inf"), 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        compute_realized_regret(p_ok, h_inf, "cross_entropy", support)

    # mismatched shapes
    h_wrong = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    support_wrong = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="matching shapes"):
        compute_realized_regret(p_ok, h_wrong, "cross_entropy", support_wrong)

    # 1-D p_star (wrong ndim)
    p_1d = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        compute_realized_regret(p_1d, h_ok, "cross_entropy", support)

    # Row-sum violation
    p_bad_sum = np.array([[0.3, 0.3]], dtype=np.float32)
    with pytest.raises(ValueError, match="rows must sum"):
        compute_realized_regret(p_bad_sum, h_ok, "cross_entropy", support)
