"""Per-point realized-regret functions for the four supported losses.

Computes the per-point excess loss of an empirical Bayes predictor
(derived from a Bayesian model average distribution) against the
oracle predictor under ``p*(y | x)``. Used by
``experiments/epistemic_eval/scripts/compute_metrics.py`` to feed
:func:`probly.evaluation.regret_coverage.aurec`.

The four loss-specific formulae are documented in the per-function
docstrings, including the cleaned-up squared-loss derivation
(see :func:`_squared_regret`) and the clip+renormalize convention
for cross-entropy (see :func:`_cross_entropy_regret`).

Each call validates the input shapes / row-sums and returns a
non-negative ``(N,)`` ``float32`` array of per-point regrets.

The empirical Bayes predictor for each loss is derived internally
from the BMA distribution ``H_hat`` (e.g.
``softmax(logits, axis=1).mean(axis=2)``) so that a single function
signature ``compute_realized_regret(p_star, h_hat, loss, support)``
covers all four losses.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from probly.quantification._validation import (
    _check_1d_finite_real,
    _check_2d_finite_real,
    _check_loss_string,
    _check_support_matches_k,
)

LossName = Literal["cross_entropy", "zero_one", "squared", "absolute"]

_KL_EPSILON: float = 1e-12
"""Probability floor used in cross-entropy KL clipping. Below the
smallest probability softmax produces in practice (~1e-30 in
float64), so the clip rarely fires; when it does it bounds the KL
at ~28 nats rather than +inf, which preserves downstream
aggregability."""

_ROW_SUM_TOL: float = 1e-4
"""Per-row absolute tolerance for the unit-sum check on ``p_star``
and ``h_hat``."""


def _check_row_stochastic(arr: np.ndarray, *, name: str) -> None:
    """Raise ``ValueError`` if any row of ``arr`` deviates from sum 1.

    Args:
        arr: ``(N, K)`` non-negative array (already validated finite).
        name: Argument name to embed in error messages.

    Raises:
        ValueError: If any row sum is outside ``1 +/- _ROW_SUM_TOL``.
    """
    row_sums = arr.sum(axis=1)
    deviation = np.max(np.abs(row_sums - 1.0))
    if deviation > _ROW_SUM_TOL:
        msg = f"`{name}` rows must sum to 1.0 within {_ROW_SUM_TOL}; max deviation = {float(deviation)!r}."
        raise ValueError(msg)


def _lower_median_index(p: np.ndarray) -> np.ndarray:
    """Return the lower-median column index of every row of ``p``.

    For each row, the lower median is the smallest column index ``k``
    whose cumulative probability mass reaches 0.5. Implemented via
    ``(cdf >= 0.5).argmax(axis=1)``: on a ``bool`` array, ``argmax``
    returns the first ``True`` index, which is exactly the lower-median
    tiebreak we want.

    Args:
        p: ``(N, K)`` row-stochastic probability matrix.

    Returns:
        ``(N,)`` int64 array of column indices.
    """
    cdf = np.cumsum(p, axis=1)
    return np.asarray((cdf >= 0.5).argmax(axis=1), dtype=np.int64)


def _squared_regret(
    p_star: np.ndarray,
    h_hat: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Squared-loss realized regret: ``(H_hat_mean - H_star_mean) ** 2``.

    Derivation::

        (y_hat - y)^2 - (y_star - y)^2
            = (y_hat - y_star) * (y_hat + y_star - 2y)

    Taking the expectation over ``y ~ p*``::

        E[y_hat + y_star - 2y] = 2 * y_star - 2 * E[y]
                               = 0    when y_star = E[y]   (the L2-Bayes-optimal predictor under p*)

    So the cross-term vanishes and::

        E[(y_hat - y)^2 - (y_star - y)^2] = (y_hat - y_star)^2

    where ``y_hat = sum_k support[k] * h_hat[i, k]`` and
    ``y_star = sum_k support[k] * p_star[i, k]``.

    Args:
        p_star: ``(N, K)`` ground-truth row-stochastic conditional.
        h_hat: ``(N, K)`` empirical BMA, also row-stochastic.
        support: ``(K,)`` real-valued labels.

    Returns:
        ``(N,)`` float32 array of per-point regrets.
    """
    support_f = support.astype(np.float64, copy=False)
    h_star_mean = (p_star * support_f[None, :]).sum(axis=1)
    h_hat_mean = (h_hat * support_f[None, :]).sum(axis=1)
    return ((h_hat_mean - h_star_mean) ** 2).astype(np.float32, copy=False)


def _absolute_regret(
    p_star: np.ndarray,
    h_hat: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Absolute-loss realized regret.

    Per-point regret ::

        sum_k (|support[k] - H_hat_med| - |support[k] - H_star_med|) * p_star[i, k]

    where ``H_hat_med`` and ``H_star_med`` are the lower-medians (smallest
    ``support[k]`` whose cumulative probability mass reaches 0.5) of
    ``h_hat[i, :]`` and ``p_star[i, :]`` respectively.

    Because the Bayes-optimal predictor under L1 is the median of
    ``p*``, the regret is non-negative; we clip tiny negative values
    arising from floating-point arithmetic to zero.

    Args:
        p_star: ``(N, K)`` ground-truth row-stochastic conditional.
        h_hat: ``(N, K)`` empirical BMA, also row-stochastic.
        support: ``(K,)`` real-valued labels.

    Returns:
        ``(N,)`` float32 array of per-point regrets.
    """
    support_f = support.astype(np.float64, copy=False)
    h_star_idx = _lower_median_index(p_star)
    h_hat_idx = _lower_median_index(h_hat)
    h_star_med = support_f[h_star_idx]
    h_hat_med = support_f[h_hat_idx]
    diffs_hat = np.abs(support_f[None, :] - h_hat_med[:, None])
    diffs_star = np.abs(support_f[None, :] - h_star_med[:, None])
    regret = (p_star * (diffs_hat - diffs_star)).sum(axis=1)
    np.maximum(regret, 0.0, out=regret)
    return regret.astype(np.float32, copy=False)


def _zero_one_regret(
    p_star: np.ndarray,
    h_hat: np.ndarray,
) -> np.ndarray:
    """Zero-one realized regret: ``max_k p_star - p_star[H_hat_argmax]``.

    The Bayes-optimal predictor under 0/1 loss is the argmax of
    ``p*``; the empirical predictor is the argmax of ``h_hat``
    (numpy ``argmax`` with lowest-index tiebreak).

    Per-point regret::

        max_k p_star[i, k] - p_star[i, argmax_k h_hat[i, k]]

    Args:
        p_star: ``(N, K)`` ground-truth row-stochastic conditional.
        h_hat: ``(N, K)`` empirical BMA, also row-stochastic.

    Returns:
        ``(N,)`` float32 array of per-point regrets.
    """
    n = p_star.shape[0]
    h_hat_argmax = np.asarray(h_hat.argmax(axis=1), dtype=np.int64)
    p_at_pred = p_star[np.arange(n), h_hat_argmax]
    regret = p_star.max(axis=1) - p_at_pred
    np.maximum(regret, 0.0, out=regret)
    return regret.astype(np.float32, copy=False)


def _cross_entropy_regret(
    p_star: np.ndarray,
    h_hat: np.ndarray,
) -> np.ndarray:
    """Cross-entropy realized regret: ``KL(p_star || h_hat)``.

    We clip ``h_hat`` to ``[_KL_EPSILON, 1.0]`` BEFORE renormalizing,
    so the renormalized vector still sums to 1 with all entries
    strictly positive. The eps ``1e-12`` is below the smallest
    probability softmax produces in practice (~1e-30 in float64), so
    the clip rarely fires; when it does, the would-have-been infinite
    KL is finite (~13-28 nats) -- large enough to flag the pathology,
    finite enough to aggregate downstream.

    Per-point regret::

        KL(p_star || h_hat) = sum_k p_star[i, k] * log(p_star[i, k] / h_hat[i, k])

    with the convention ``0 * log(0 / x) = 0``.

    Args:
        p_star: ``(N, K)`` ground-truth row-stochastic conditional.
        h_hat: ``(N, K)`` empirical BMA, also row-stochastic.

    Returns:
        ``(N,)`` float32 array of per-point regrets.
    """
    h_hat_clipped = np.clip(h_hat, _KL_EPSILON, 1.0)
    h_hat_clipped = h_hat_clipped / h_hat_clipped.sum(axis=1, keepdims=True)
    log_ratio = np.where(
        p_star > 0,
        p_star * (np.log(np.where(p_star > 0, p_star, 1.0)) - np.log(h_hat_clipped)),
        0.0,
    )
    regret = log_ratio.sum(axis=1)
    np.maximum(regret, 0.0, out=regret)
    return regret.astype(np.float32, copy=False)


def compute_realized_regret(
    p_star: np.ndarray,
    h_hat: np.ndarray,
    loss: LossName | str,
    support: np.ndarray,
) -> np.ndarray:
    """Compute per-point realized regret under the chosen loss.

    The function dispatches to a per-loss helper. The empirical
    predictor for each loss is derived internally from the BMA
    distribution ``h_hat``; there is no need to pass a pre-derived
    ``H_hat`` (mean / median / argmax / soft predictor) separately.

    Args:
        p_star: ``(N, K)`` ground-truth conditional. Rows must sum to
            1.0 within ``1e-4``. Coerced to ``float64`` internally.
        h_hat: ``(N, K)`` empirical BMA (e.g.
            ``softmax(logits, axis=1).mean(axis=2)``). Rows must sum to
            1.0 within ``1e-4``. The cross-entropy branch clips to
            ``[_KL_EPSILON, 1.0]`` and renormalizes before computing
            KL.
        loss: One of ``"cross_entropy"``, ``"zero_one"``, ``"squared"``,
            ``"absolute"``.
        support: ``(K,)`` array of class labels (integer for regression,
            arbitrary for classification).

    Returns:
        ``(N,)`` ``float32`` array of per-point realized regrets, all
        ``>= 0``.

    Raises:
        TypeError: If ``loss`` is not a string or ``support``/``p_star``
            /``h_hat`` have a non-numeric dtype.
        ValueError: If ``loss`` is unknown, shapes are malformed,
            ``p_star``/``h_hat`` rows don't sum to ``1`` within
            tolerance, or any input contains non-finite entries.
    """
    loss_validated = _check_loss_string(loss)
    p_star_arr = _check_2d_finite_real(p_star, name="p_star").astype(np.float64, copy=True)
    h_hat_arr = _check_2d_finite_real(h_hat, name="h_hat").astype(np.float64, copy=True)
    support_arr = _check_1d_finite_real(support, name="support")

    if p_star_arr.shape != h_hat_arr.shape:
        msg = f"`p_star` and `h_hat` must have matching shapes; got {p_star_arr.shape} vs {h_hat_arr.shape}."
        raise ValueError(msg)
    k = int(p_star_arr.shape[1])
    _check_support_matches_k(support_arr, k)

    if np.any(p_star_arr < 0):
        msg = "`p_star` must have non-negative entries."
        raise ValueError(msg)
    if np.any(h_hat_arr < 0):
        msg = "`h_hat` must have non-negative entries."
        raise ValueError(msg)
    _check_row_stochastic(p_star_arr, name="p_star")
    _check_row_stochastic(h_hat_arr, name="h_hat")

    if loss_validated == "squared":
        return _squared_regret(p_star_arr, h_hat_arr, support_arr)
    if loss_validated == "absolute":
        return _absolute_regret(p_star_arr, h_hat_arr, support_arr)
    if loss_validated == "zero_one":
        return _zero_one_regret(p_star_arr, h_hat_arr)
    return _cross_entropy_regret(p_star_arr, h_hat_arr)


__all__ = [
    "LossName",
    "compute_realized_regret",
]
