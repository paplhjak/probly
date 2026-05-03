"""Decomposition for evidential classification outputs.

Maps a method's cached ``(N, K)`` Dirichlet concentration parameters
``alpha`` to the ``(A_hat, E_hat, H_hat)`` triple consumed by the
loss-parametric evaluation pipeline. The ``evidence`` scalar that
accompanies ``alpha`` in the cache is accepted for schema
completeness but does not enter the formulae below.

Decomposition (Sensoy et al. 2018, "Evidential Deep Learning to
Quantify Classification Uncertainty," NeurIPS 2018):

* ``alpha_0(x) = sum_k alpha[x, k]``
* Dirichlet mean ``p_hat(k|x) = alpha[x, k] / alpha_0(x)``
* Mass-based epistemic ``E_hat(x) = K / alpha_0(x)``  -- shrinks as
  the network sees more evidence (alpha_0 grows).
* For ``cross_entropy`` deployment loss:
    - ``H_hat(x) = p_hat(x)``  -- the soft Bayes-optimal predictor.
    - ``A_hat(x) = -sum_k p_hat(k|x) log p_hat(k|x)``  -- categorical
      entropy of the Dirichlet mean, in nats.
* For ``zero_one`` deployment loss:
    - ``H_hat(x) = argmax_k p_hat(k|x)``  -- hard label.
    - ``A_hat(x) = 1 - max_k p_hat(k|x)``  -- top-1 error of the
      Dirichlet mean.
    - ``E_hat`` unchanged.

The heterogeneous ``H_hat`` contract matches the existing
classification dispatcher (``probly.quantification.decomposition.
classification``): cross-entropy returns a soft ``(N, K)`` predictor;
zero-one returns a hard ``(N,)`` int64 label. ``A_hat`` and ``E_hat``
are always ``(N,) float32``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from probly.quantification._validation import _check_loss_string

_EvidentialLoss = Literal["cross_entropy", "zero_one"]


def _check_alpha(alpha: np.ndarray) -> np.ndarray:
    """Coerce ``alpha`` to a 2-D positive finite ``(N, K) float64`` array."""
    arr = np.asarray(alpha)
    if arr.ndim != 2:
        msg = f"`alpha` must be 2-D with shape (N, K), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        msg = f"`alpha` must contain at least one sample and one class, got shape={arr.shape}."
        raise ValueError(msg)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`alpha` must be a real numeric array, got dtype={arr.dtype!r}."
        raise TypeError(msg)
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        msg = "`alpha` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    if np.any(arr <= 0.0):
        msg = "`alpha` must be strictly positive (Dirichlet concentration parameters)."
        raise ValueError(msg)
    return arr


def _check_evidence(evidence: np.ndarray, n: int) -> np.ndarray:
    """Coerce ``evidence`` to a 1-D non-negative finite ``(N,) float64`` array."""
    arr = np.asarray(evidence)
    if arr.ndim != 1:
        msg = f"`evidence` must be 1-D with shape (N,), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] != n:
        msg = f"`evidence` length {arr.shape[0]} does not match `alpha` length {n}."
        raise ValueError(msg)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`evidence` must be a real numeric array, got dtype={arr.dtype!r}."
        raise TypeError(msg)
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        msg = "`evidence` must contain only finite values."
        raise ValueError(msg)
    return arr


def evidential_decomposition(
    alpha: np.ndarray,
    evidence: np.ndarray,
    loss: _EvidentialLoss,
) -> dict[str, np.ndarray]:
    """Decompose Evidential-Classification cached outputs.

    Args:
        alpha: ``(N, K)`` Dirichlet concentration parameters, all
            strictly positive (probly's evidential head produces
            ``softplus(logit) + 1`` so this holds by construction).
        evidence: ``(N,)`` per-point scalar evidence (the
            ``softplus(logit)`` sum, ``alpha_0 - K``). Cached but not
            used in the decomposition formulas; accepted for schema
            completeness.
        loss: One of ``"cross_entropy"`` or ``"zero_one"``. Other
            losses are not supported by this decomposition.

    Returns:
        Dict with keys ``A_hat``, ``E_hat``, ``H_hat``. Per-loss
        ``H_hat`` shape contract::

            cross_entropy: H_hat is (N, K) float32 (soft predictor).
            zero_one:      H_hat is (N,)  int64   (argmax label).

        ``A_hat`` and ``E_hat`` are always ``(N,) float32``.

    Raises:
        TypeError: If inputs have non-numeric dtypes or ``loss`` is
            not a string.
        ValueError: If shapes are wrong, ``alpha`` is non-positive,
            inputs contain non-finite entries, or ``loss`` is not one
            of the supported values.
    """
    loss_validated = _check_loss_string(loss)
    if loss_validated not in {"cross_entropy", "zero_one"}:
        msg = (
            f"`loss` must be one of {{'cross_entropy', 'zero_one'}} for the "
            f"evidential decomposition, got {loss_validated!r}."
        )
        raise ValueError(msg)
    alpha_arr = _check_alpha(alpha)
    n, k = alpha_arr.shape
    _check_evidence(evidence, n)  # validate; not used downstream

    alpha0 = alpha_arr.sum(axis=1)
    p_hat = alpha_arr / alpha0[:, None]
    e_hat = (k / alpha0).astype(np.float32, copy=False)

    if loss_validated == "cross_entropy":
        # Categorical entropy of p_hat in nats.
        a_hat = -(p_hat * np.log(np.clip(p_hat, 1e-12, 1.0))).sum(axis=1)
        h_hat = p_hat.astype(np.float32, copy=False)
        return {
            "A_hat": a_hat.astype(np.float32, copy=False),
            "E_hat": e_hat,
            "H_hat": h_hat,
        }
    # zero_one
    max_p = p_hat.max(axis=1)
    a_hat = 1.0 - max_p
    h_hat = p_hat.argmax(axis=1).astype(np.int64, copy=False)
    return {
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat,
        "H_hat": h_hat,
    }


__all__ = ["evidential_decomposition"]
