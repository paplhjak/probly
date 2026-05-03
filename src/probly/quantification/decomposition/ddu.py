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

_DDULoss = Literal["cross_entropy", "zero_one"]


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
) -> dict[str, np.ndarray]:
    """Decompose DDU cached outputs.

    Args:
        probs: ``(N, K)`` row-stochastic categorical posterior from
            the spectrally-normalised classifier's softmax.
        density: ``(N,)`` marginal log-density ``log q(z)`` from the
            class-conditional GMM fit at training time.
        loss: One of ``"cross_entropy"`` or ``"zero_one"``.

    Returns:
        Dict with keys ``A_hat``, ``E_hat``, ``H_hat``. Per-loss
        ``H_hat`` shape contract matches the existing classification
        decompositions:

            cross_entropy: H_hat is (N, K) float32 (soft predictor).
            zero_one:      H_hat is (N,)  int64   (argmax label).

        ``A_hat`` and ``E_hat`` are always ``(N,) float32``.
        ``E_hat = -density`` so larger values denote more uncertainty.

    Raises:
        TypeError: If inputs have non-numeric dtypes or ``loss`` is
            not a string.
        ValueError: If shapes are wrong, ``probs`` is not row-
            stochastic, inputs contain non-finite entries, or
            ``loss`` is unsupported.
    """
    loss_validated = _check_loss_string(loss)
    if loss_validated not in {"cross_entropy", "zero_one"}:
        msg = (
            f"`loss` must be one of {{'cross_entropy', 'zero_one'}} for the DDU decomposition, got {loss_validated!r}."
        )
        raise ValueError(msg)
    probs_arr = _check_probs(probs)
    n, _ = probs_arr.shape
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
    # zero_one
    max_p = probs_arr.max(axis=1)
    a_hat = 1.0 - max_p
    h_hat = probs_arr.argmax(axis=1).astype(np.int64, copy=False)
    return {
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat,
        "H_hat": h_hat,
    }


__all__ = ["ddu_decomposition"]
