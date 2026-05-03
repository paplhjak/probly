"""Regression decompositions for the loss-parametric ``decompose`` API.

Squared- and absolute-error decompositions of a method's posterior
samples expressed as a ``(N, K, S)`` tensor of raw class scores
(logits). The columns of the per-sample softmax are interpreted as a
probability vector over a real-valued ``support`` (e.g. integer ages
on APPA-REAL); the per-sample posterior is therefore a categorical
distribution on ``support``, and the per-sample Bayes-optimal
predictor is the mean (squared loss) or the lower median (absolute
loss) of that categorical.

The two functions in this module mirror the math in
``decisions.md`` -> "APPA-REAL deployment losses". They consume numpy
arrays and produce numpy arrays so callers do not need a torch
dependency; the parallel classification module wraps probly's
torch-flavoured primitives instead.
"""

from __future__ import annotations

import numpy as np

from probly.quantification._validation import (
    _check_1d_finite_real,
    _check_3d_finite_real,
    _check_support_matches_k,
)


def _softmax_nks(logits: np.ndarray) -> np.ndarray:
    """Stable softmax along axis 1 of an ``(N, K, S)`` tensor.

    Returns the same shape; rows along axis 1 sum to 1.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _lower_median_per_sample(probs_nks: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Lower median of every per-point per-sample categorical.

    Args:
        probs_nks: ``(N, K, S)`` row-stochastic along axis 1.
        support: ``(K,)`` real-valued labels.

    Returns:
        ``(N, S)`` array; entry ``[i, s]`` is the lower-median support
        value of ``probs_nks[i, :, s]``.
    """
    cdf = np.cumsum(probs_nks, axis=1)
    median_idx = np.asarray((cdf >= 0.5).argmax(axis=1), dtype=np.int64)  # (N, S)
    return support[median_idx]


def _validate_logits_and_support(
    logits: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``logits`` (``(N, K, S)``) and ``support`` (``(K,)``)."""
    logits_arr = _check_3d_finite_real(logits, name="logits")
    support_arr = _check_1d_finite_real(support, name="support")
    _check_support_matches_k(support_arr, int(logits_arr.shape[1]))
    return logits_arr.astype(np.float32, copy=False), support_arr.astype(np.float64, copy=False)


def squared_decomposition(
    logits: np.ndarray,
    support: np.ndarray,
) -> dict[str, np.ndarray]:
    """Squared-error decomposition of a ``(N, K, S)`` posterior tensor.

    For each test point ``i`` and posterior sample ``s``:

    * Per-sample posterior mean::

          mu_s(i) = sum_k support[k] * softmax(logits)[i, k, s].

    * Per-sample posterior variance::

          var_s(i) = sum_k (support[k] - mu_s(i))^2
                          * softmax(logits)[i, k, s].

    The decomposition then averages over the ``S`` posterior samples::

        H_hat[i] = (1 / S) sum_s mu_s(i)
        A_hat[i] = (1 / S) sum_s var_s(i)
        E_hat[i] = (1 / S) sum_s (mu_s(i) - H_hat[i])^2.

    Under squared loss, ``A_hat + E_hat`` equals the variance of the
    *marginal mixture* (over the ``S`` samples); this is the classical
    law of total variance and gives a clean correctness check on the
    implementation. It is **not** the paper's identity ``T* = A* + E*``
    (which holds for any loss by Definition 1); see the test suite for
    the framing.

    Args:
        logits: ``(N, K, S)`` raw pre-softmax class scores. Coerced
            to ``float32`` and validated for ``ndim == 3``, finite,
            real.
        support: ``(K,)`` real-valued labels in the same column order
            as ``logits``.

    Returns:
        Dict with ``"H_hat"``, ``"A_hat"``, ``"E_hat"``, each
        ``(N,)`` ``float32``.

    Raises:
        TypeError: If ``logits`` or ``support`` have a non-numeric
            dtype.
        ValueError: If shapes are wrong or values are non-finite.
    """
    logits_arr, support_f = _validate_logits_and_support(logits, support)
    probs = _softmax_nks(logits_arr.astype(np.float64))  # (N, K, S)
    # Per-sample mean: sum_k support[k] * probs[i, k, s] -> (N, S)
    mu = (probs * support_f[None, :, None]).sum(axis=1)
    # Per-sample variance: sum_k (support[k] - mu[i, s])^2 * probs[i, k, s]
    diffs = support_f[None, :, None] - mu[:, None, :]
    var = (probs * diffs * diffs).sum(axis=1)
    h_hat = mu.mean(axis=1)
    a_hat = var.mean(axis=1)
    e_hat = ((mu - h_hat[:, None]) ** 2).mean(axis=1)
    return {
        "H_hat": h_hat.astype(np.float32, copy=False),
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat.astype(np.float32, copy=False),
    }


def absolute_decomposition(
    logits: np.ndarray,
    support: np.ndarray,
) -> dict[str, np.ndarray]:
    """Absolute-error decomposition of a ``(N, K, S)`` posterior tensor.

    For each test point ``i`` and posterior sample ``s``:

    * Per-sample lower median::

          median_s(i) = lower_median over support of
                        softmax(logits)[i, :, s].

    * Per-sample mean absolute deviation around that median::

          mad_s(i) = sum_k |support[k] - median_s(i)|
                          * softmax(logits)[i, k, s].

    The decomposition then aggregates over ``S``::

        H_hat[i] = lower_median over s of {median_s(i, s)}
        A_hat[i] = (1 / S) sum_s mad_s(i)
        E_hat[i] = (1 / S) sum_s |median_s(i, s) - H_hat[i]|.

    Under absolute loss, ``A_hat + E_hat`` does **not** equal the
    marginal mixture MAD; there is no L1 analogue of the law of total
    variance. The test suite asserts this gap is significantly nonzero
    so future code cannot accidentally enforce a bias-variance-style
    identity in the L1 case.

    Args:
        logits: ``(N, K, S)`` raw pre-softmax class scores.
        support: ``(K,)`` real-valued labels.

    Returns:
        Dict with ``"H_hat"``, ``"A_hat"``, ``"E_hat"``, each
        ``(N,)`` ``float32``.

    Raises:
        TypeError, ValueError: Same as :func:`squared_decomposition`.
    """
    logits_arr, support_f = _validate_logits_and_support(logits, support)
    probs = _softmax_nks(logits_arr.astype(np.float64))  # (N, K, S)
    medians = _lower_median_per_sample(probs, support_f)  # (N, S)
    mad = (probs * np.abs(support_f[None, :, None] - medians[:, None, :])).sum(axis=1)  # (N, S)
    a_hat = mad.mean(axis=1)
    # Lower median over S of the per-sample medians: sort, take index (S - 1) // 2.
    s = medians.shape[1]
    lower_idx = (s - 1) // 2
    sorted_medians = np.sort(medians, axis=1)
    h_hat = sorted_medians[:, lower_idx]
    e_hat = np.abs(medians - h_hat[:, None]).mean(axis=1)
    return {
        "H_hat": h_hat.astype(np.float32, copy=False),
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat.astype(np.float32, copy=False),
    }


__all__ = ["absolute_decomposition", "squared_decomposition"]
