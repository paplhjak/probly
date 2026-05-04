"""Decomposition for DDU (Deep Deterministic Uncertainty) cached outputs.

Maps a method's cached ``(N, K)`` class probabilities and ``(N,)``
feature-space marginal log-densities to the ``(A_hat, E_hat, H_hat)``
triple consumed by the loss-parametric evaluation pipeline.

Decomposition (Mukhoti et al. 2023, "Deep Deterministic Uncertainty: A
New Simple Baseline"):

* ``probs[x] = softmax(classification_head(encoder(x)))``  -- the
  spectrally-normalised classifier's posterior. The encoder's
  spectral-norm bound on hidden Linear/Conv2d layers is what
  preserves feature-space density meaningfulness.
* ``density[x] = log q(z(x)) = log sum_c pi_c N(z; mu_c, Sigma_c)``
  -- the marginal log-density of the encoder feature ``z(x)``
  under a class-conditional GMM fit on the training set's features.
* ``E_hat(x) = -density[x]``  -- negate so HIGHER E_hat means MORE
  uncertainty (consistent direction with sampling-based methods'
  epistemic estimates). Far-from-train features have ``log q(z)``
  very negative, so ``-log q(z)`` is large.
* For ``cross_entropy`` deployment loss:
    - ``H_hat(x) = probs(x)``  -- the soft Bayes-optimal predictor.
    - ``A_hat(x) = -sum_k probs(k|x) log probs(k|x)``  -- categorical
      entropy of the classifier's softmax posterior, in nats.
* For ``zero_one`` deployment loss:
    - ``H_hat(x) = argmax_k probs(k|x)``.
    - ``A_hat(x) = 1 - max_k probs(k|x)``.
    - ``E_hat`` unchanged.

The aleatoric component reuses the classifier's softmax posterior
(NOT the GMM posterior); the epistemic component is the
GMM-density-based OOD score. This is how the original DDU paper
reports both signals.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from probly.quantification._validation import _check_loss_string

_DDULoss = Literal["cross_entropy", "zero_one", "squared", "absolute"]


def _check_support(support: np.ndarray | None, k: int) -> np.ndarray:
    """Coerce ``support`` to a 1-D float64 ``(K,)`` array.

    Required by the ``squared`` and ``absolute`` regression branches
    so we know what scalar value each ``probs`` column corresponds to.
    """
    if support is None:
        msg = (
            "`support` is required for the squared / absolute DDU "
            "decompositions; got None. Pass an integer ``(K,)`` array "
            "of class labels (e.g. ``np.arange(K)`` for an age support)."
        )
        raise ValueError(msg)
    arr = np.asarray(support)
    if arr.ndim != 1:
        msg = f"`support` must be 1-D with shape (K,), got ndim={arr.ndim}."
        raise ValueError(msg)
    if arr.shape[0] != k:
        msg = (
            f"`support` length {arr.shape[0]} does not match the K dimension "
            f"of `probs` ({k})."
        )
        raise ValueError(msg)
    return arr.astype(np.float64, copy=False)


def _check_probs(probs: np.ndarray) -> np.ndarray:
    """Coerce ``probs`` to a 2-D row-stochastic finite ``(N, K) float64`` array."""
    arr = np.asarray(probs)
    if arr.ndim != 2:
        msg = f"`probs` must be 2-D with shape (N, K), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        msg = f"`probs` must contain at least one sample and one class, got shape={arr.shape}."
        raise ValueError(msg)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`probs` must be a real numeric array, got dtype={arr.dtype!r}."
        raise TypeError(msg)
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        msg = "`probs` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    if np.any(arr < 0.0):
        msg = "`probs` must be non-negative."
        raise ValueError(msg)
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        msg = f"`probs` rows must sum to 1.0 (atol=1e-4); got sums in [{sums.min():.6f}, {sums.max():.6f}]."
        raise ValueError(msg)
    return arr


def _check_density(density: np.ndarray, n: int) -> np.ndarray:
    """Coerce ``density`` to a 1-D finite ``(N,) float64`` array."""
    arr = np.asarray(density)
    if arr.ndim != 1:
        msg = f"`density` must be 1-D with shape (N,), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] != n:
        msg = f"`density` length {arr.shape[0]} does not match `probs` length {n}."
        raise ValueError(msg)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`density` must be a real numeric array, got dtype={arr.dtype!r}."
        raise TypeError(msg)
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        msg = "`density` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    return arr


def ddu_decomposition(
    probs: np.ndarray,
    density: np.ndarray,
    loss: _DDULoss,
    support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Decompose DDU cached outputs.

    Args:
        probs: ``(N, K)`` row-stochastic categorical posterior from
            the spectrally-normalised classifier's softmax.
        density: ``(N,)`` marginal log-density ``log q(z)`` from the
            class-conditional GMM fit at training time.
        loss: One of ``"cross_entropy"``, ``"zero_one"``,
            ``"squared"``, or ``"absolute"``. The classification
            branches ignore ``support``; the regression branches
            require it.
        support: ``(K,)`` integer-valued class labels. Required for
            ``squared`` and ``absolute``; optional (and ignored) for
            ``cross_entropy`` and ``zero_one``.

    Returns:
        Dict with keys ``A_hat``, ``E_hat``, ``H_hat``. Per-loss
        ``H_hat`` shape contract matches the existing classification +
        regression decompositions:

            cross_entropy: H_hat is (N, K) float32 (soft predictor).
            zero_one:      H_hat is (N,)  int64   (argmax label).
            squared:       H_hat is (N,)  float32 (mean predictor).
            absolute:      H_hat is (N,)  float32 (median value in
                                                   support units;
                                                   matches the
                                                   ``logits_nks``
                                                   regression contract).

        ``A_hat`` and ``E_hat`` are always ``(N,) float32``.
        ``E_hat = -density`` for every loss so larger values denote
        more uncertainty (the DDU OOD signal is independent of the
        deployment loss; only ``A_hat`` / ``H_hat`` change).

    Raises:
        TypeError: If inputs have non-numeric dtypes or ``loss`` is
            not a string.
        ValueError: If shapes are wrong, ``probs`` is not row-
            stochastic, inputs contain non-finite entries, ``loss``
            is unknown, or ``support`` is missing for a
            regression-side loss.
    """
    loss_validated = _check_loss_string(loss)
    if loss_validated not in {"cross_entropy", "zero_one", "squared", "absolute"}:
        msg = (
            f"`loss` must be one of {{'cross_entropy', 'zero_one', "
            f"'squared', 'absolute'}} for the DDU decomposition, got "
            f"{loss_validated!r}."
        )
        raise ValueError(msg)
    probs_arr = _check_probs(probs)
    n, k = probs_arr.shape
    density_arr = _check_density(density, n)

    e_hat = (-density_arr).astype(np.float32, copy=False)

    if loss_validated == "cross_entropy":
        a_hat = -(probs_arr * np.log(np.clip(probs_arr, 1e-12, 1.0))).sum(axis=1)
        h_hat = probs_arr.astype(np.float32, copy=False)
        return {
            "A_hat": a_hat.astype(np.float32, copy=False),
            "E_hat": e_hat,
            "H_hat": h_hat,
        }
    if loss_validated == "zero_one":
        max_p = probs_arr.max(axis=1)
        a_hat = 1.0 - max_p
        h_hat = probs_arr.argmax(axis=1).astype(np.int64, copy=False)
        return {
            "A_hat": a_hat.astype(np.float32, copy=False),
            "E_hat": e_hat,
            "H_hat": h_hat,
        }
    # squared / absolute: regression-side branches.
    support_arr = _check_support(support, k)
    if loss_validated == "squared":
        # Mean predictor + per-row variance under the categorical posterior.
        h_hat_sq = (probs_arr * support_arr[None, :]).sum(axis=1)
        second_moment = (probs_arr * (support_arr[None, :] ** 2)).sum(axis=1)
        var_p = second_moment - h_hat_sq ** 2
        np.maximum(var_p, 0.0, out=var_p)
        return {
            "A_hat": var_p.astype(np.float32, copy=False),
            "E_hat": e_hat,
            "H_hat": h_hat_sq.astype(np.float32, copy=False),
        }
    # absolute
    cum = probs_arr.cumsum(axis=1)
    h_hat_idx = np.argmax(cum >= 0.5, axis=1).astype(np.int64)
    h_hat_value = support_arr[h_hat_idx]
    a_hat_abs = (np.abs(support_arr[None, :] - h_hat_value[:, None]) * probs_arr).sum(axis=1)
    return {
        "A_hat": a_hat_abs.astype(np.float32, copy=False),
        "E_hat": e_hat,
        "H_hat": h_hat_value.astype(np.float32, copy=False),
    }


__all__ = ["ddu_decomposition"]
