"""Uncertainty decomposition methods.

In addition to the class-based decompositions
(:class:`SecondOrderEntropyDecomposition`,
:class:`SecondOrderZeroOneDecomposition`,
:class:`CredalSetEntropyDecomposition`), this package exposes a thin
loss-parametric numpy wrapper :func:`decompose` that consumes a
method's cached ``(N, K, S)`` raw class scores and returns a dict of
``(A_hat, E_hat, H_hat)`` numpy arrays. The wrapper bridges the
torch-flavoured probly primitives to the numpy-only caching layer
used by the NeurIPS 2026 epistemic-eval pipeline (see
``decisions.md`` -> "Loss handling in the codebase").
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from probly.quantification._validation import (
    _check_1d_finite_real,
    _check_3d_finite_real,
    _check_loss_string,
    _check_support_matches_k,
)

from .classification import cross_entropy_decomposition, zero_one_decomposition
from .decomposition import (
    AdditiveDecomposition,
    AleatoricEpistemicDecomposition,
    AleatoricEpistemicTotalDecomposition,
    CachingDecomposition,
    Decomposition,
)
from .entropy import CredalSetEntropyDecomposition, SecondOrderEntropyDecomposition
from .regression import absolute_decomposition, squared_decomposition
from .zero_one import SecondOrderZeroOneDecomposition

#: Bumped on math changes to invalidate caches.
_DECOMPOSITION_VERSION = 1

LossName = Literal["cross_entropy", "zero_one", "squared", "absolute"]


def decompose(
    logits: np.ndarray,
    loss: LossName,
    support: np.ndarray,
) -> dict[str, np.ndarray]:
    """Dispatch to the per-loss decomposition function.

    Args:
        logits: ``(N, K, S)`` ``float32`` raw class scores. Coerced via
            ``np.asarray(..., dtype=np.float32)``. Must be 3-D, finite,
            and real.
        loss: One of ``"cross_entropy"``, ``"zero_one"``, ``"squared"``,
            ``"absolute"``.
        support: ``(K,)`` array of class labels. Required dtype and
            interpretation:

                cross_entropy: any (ignored).
                zero_one:      any (ignored).
                squared:       integer or float-valued; treated as
                               real-valued labels.
                absolute:      same.

    Returns:
        Dict with keys ``A_hat``, ``E_hat``, ``H_hat``. Per-loss shape
        contract::

            Loss          | H_hat shape | dtype
            cross_entropy | (N, K)      | float32 -- soft predictor
            zero_one      | (N,)        | int64   -- argmax label
            squared       | (N,)        | float32 -- mean predictor
            absolute      | (N,)        | float32 -- median predictor

        ``A_hat`` and ``E_hat`` are always ``(N,)`` ``float32``
        regardless of loss.

    Raises:
        TypeError: If ``loss`` is not a string or ``support``/``logits``
            have a non-numeric dtype.
        ValueError: If ``loss`` is unknown, ``logits`` is not 3-D,
            ``support`` length mismatches the ``K`` axis, or any input
            contains non-finite entries.
    """
    loss_validated = _check_loss_string(loss)
    logits_arr = np.asarray(logits, dtype=np.float32)
    logits_arr = _check_3d_finite_real(logits_arr, name="logits")
    support_arr = _check_1d_finite_real(support, name="support")
    _check_support_matches_k(support_arr, int(logits_arr.shape[1]))

    if loss_validated == "cross_entropy":
        return cross_entropy_decomposition(logits_arr, support_arr)
    if loss_validated == "zero_one":
        return zero_one_decomposition(logits_arr, support_arr)
    if loss_validated == "squared":
        return squared_decomposition(logits_arr, support_arr)
    return absolute_decomposition(logits_arr, support_arr)


__all__ = [
    "_DECOMPOSITION_VERSION",
    "AdditiveDecomposition",
    "AleatoricEpistemicDecomposition",
    "AleatoricEpistemicTotalDecomposition",
    "CachingDecomposition",
    "CredalSetEntropyDecomposition",
    "Decomposition",
    "LossName",
    "SecondOrderEntropyDecomposition",
    "SecondOrderZeroOneDecomposition",
    "absolute_decomposition",
    "cross_entropy_decomposition",
    "decompose",
    "squared_decomposition",
    "zero_one_decomposition",
]
