"""Tests for the loss-parametric ``decompose`` dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.decomposition import (
    _DECOMPOSITION_VERSION,
    absolute_decomposition,
    cross_entropy_decomposition,
    decompose,
    squared_decomposition,
    zero_one_decomposition,
)


SUPPORT = np.array([20, 30, 40, 50], dtype=np.float64)


def _canonical_logits() -> np.ndarray:
    probs = np.zeros((2, 4, 3), dtype=np.float64)
    probs[0, 0, :] = 1.0
    probs[1, 0, 0] = 0.5
    probs[1, 1, 0] = 0.5
    probs[1, 1, 1] = 0.5
    probs[1, 2, 1] = 0.5
    probs[1, 2, 2] = 0.5
    probs[1, 3, 2] = 0.5
    return np.log(probs + 1e-30).astype(np.float32)


def test_dispatcher_routes_cross_entropy() -> None:
    logits = _canonical_logits()
    a = decompose(logits, "cross_entropy", SUPPORT)
    b = cross_entropy_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(a["A_hat"], b["A_hat"])
    np.testing.assert_allclose(a["E_hat"], b["E_hat"])
    np.testing.assert_allclose(a["H_hat"], b["H_hat"])


def test_dispatcher_routes_zero_one() -> None:
    logits = _canonical_logits()
    a = decompose(logits, "zero_one", SUPPORT)
    b = zero_one_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(a["A_hat"], b["A_hat"])
    np.testing.assert_allclose(a["E_hat"], b["E_hat"])
    np.testing.assert_array_equal(a["H_hat"], b["H_hat"])


def test_dispatcher_routes_squared() -> None:
    logits = _canonical_logits()
    a = decompose(logits, "squared", SUPPORT)
    b = squared_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(a["A_hat"], b["A_hat"])
    np.testing.assert_allclose(a["E_hat"], b["E_hat"])
    np.testing.assert_allclose(a["H_hat"], b["H_hat"])


def test_dispatcher_routes_absolute() -> None:
    logits = _canonical_logits()
    a = decompose(logits, "absolute", SUPPORT)
    b = absolute_decomposition(logits, SUPPORT)
    np.testing.assert_allclose(a["A_hat"], b["A_hat"])
    np.testing.assert_allclose(a["E_hat"], b["E_hat"])
    np.testing.assert_allclose(a["H_hat"], b["H_hat"])


def test_dispatcher_per_loss_h_hat_shape() -> None:
    """Verify the documented heterogeneous H_hat contract."""
    logits = _canonical_logits()
    n, k = 2, 4
    out_ce = decompose(logits, "cross_entropy", SUPPORT)
    assert out_ce["H_hat"].shape == (n, k)
    assert out_ce["H_hat"].dtype == np.float32

    out_zo = decompose(logits, "zero_one", SUPPORT)
    assert out_zo["H_hat"].shape == (n,)
    assert out_zo["H_hat"].dtype == np.int64

    out_sq = decompose(logits, "squared", SUPPORT)
    assert out_sq["H_hat"].shape == (n,)
    assert out_sq["H_hat"].dtype == np.float32

    out_ab = decompose(logits, "absolute", SUPPORT)
    assert out_ab["H_hat"].shape == (n,)
    assert out_ab["H_hat"].dtype == np.float32


def test_dispatcher_rejects_unknown_loss_string() -> None:
    logits = _canonical_logits()
    with pytest.raises(ValueError, match="unknown loss 'l1'"):
        decompose(logits, "l1", SUPPORT)


def test_dispatcher_rejects_non_3d_logits() -> None:
    bad = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3-D"):
        decompose(bad, "squared", SUPPORT)


def test_dispatcher_rejects_nan_logits() -> None:
    logits = _canonical_logits().copy()
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        decompose(logits, "squared", SUPPORT)


def test_dispatcher_rejects_inf_logits() -> None:
    logits = _canonical_logits().copy()
    logits[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        decompose(logits, "squared", SUPPORT)


def test_dispatcher_rejects_support_length_mismatch() -> None:
    logits = _canonical_logits()
    bad_support = np.array([20, 30, 40], dtype=np.float64)
    with pytest.raises(ValueError, match="length 3"):
        decompose(logits, "squared", bad_support)


def test_dispatcher_coerces_logits_to_float32() -> None:
    """Integer or float64 logits flow through the float32 coercion."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(size=(2, 4, 3))  # float64
    out = decompose(logits, "squared", SUPPORT)
    # Only checking that the coercion works without error.
    assert out["A_hat"].dtype == np.float32


def test_decomposition_version_is_int() -> None:
    """The version constant gates cache invalidation; assert it exists."""
    assert isinstance(_DECOMPOSITION_VERSION, int)
    assert _DECOMPOSITION_VERSION >= 1
