"""Classification decompositions for the loss-parametric ``decompose`` API.

Cross-entropy and zero-one decompositions of a method's posterior
samples expressed as a ``(N, K, S)`` numpy tensor of raw class scores
(logits). These functions wrap probly's existing torch-flavoured
:class:`SecondOrderEntropyDecomposition` and
:class:`SecondOrderZeroOneDecomposition` so the loss-parametric
dispatcher can stay numpy-only.

Layout: the input tensor is permuted from ``(N, K, S)`` to ``(N, S, K)``
so that classes occupy the last axis, as
:class:`TorchCategoricalDistribution` requires per its docstring at
``src/probly/representation/distribution/torch_categorical.py:32-36``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from probly.quantification._validation import (
    _check_1d_finite_real,
    _check_3d_finite_real,
    _check_support_matches_k,
)
from probly.quantification.decomposition.entropy._common import (
    SecondOrderEntropyDecomposition,
)
from probly.quantification.decomposition.zero_one._common import (
    SecondOrderZeroOneDecomposition,
)
from probly.representation._helpers import compute_mean_probs
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)


def _build_torch_sample_from_nks(logits_nks: np.ndarray) -> TorchCategoricalDistributionSample:
    """Build the ``(N, S, K)`` torch sample expected by probly's primitives.

    ``TorchCategoricalDistribution`` requires classes on the last axis;
    we permute ``(N, K, S) -> (N, S, K)`` and softmax along the *original*
    K axis (which becomes the last axis after the permute). The result
    is wrapped in a :class:`TorchCategoricalDistributionSample` with
    ``sample_dim=1`` so that probly's reductions over the sample axis
    work as expected.
    """
    logits_t = torch.from_numpy(np.ascontiguousarray(logits_nks))
    probs_nks = torch.softmax(logits_t, dim=1)
    probs_nsk = probs_nks.permute(0, 2, 1).contiguous()
    distribution = TorchCategoricalDistribution(unnormalized_probabilities=probs_nsk)
    return TorchCategoricalDistributionSample(tensor=distribution, sample_dim=1)


def _validate_classification_inputs(
    logits: np.ndarray,
    support: np.ndarray | None,
) -> np.ndarray:
    """Validate ``logits`` and (if provided) ``support``.

    Cross-entropy and zero-one ignore ``support``, but we still
    validate it so a malformed value raises early.
    """
    logits_arr = _check_3d_finite_real(logits, name="logits")
    if support is not None:
        support_arr = _check_1d_finite_real(support, name="support")
        _check_support_matches_k(support_arr, int(logits_arr.shape[1]))
    return logits_arr.astype(np.float32, copy=False)


def cross_entropy_decomposition(
    logits: np.ndarray,
    support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Cross-entropy decomposition of an ``(N, K, S)`` posterior tensor.

    Wraps probly's :class:`SecondOrderEntropyDecomposition`. ``A_hat``
    and ``E_hat`` are reductions over the ``K`` and ``S`` axes;
    ``H_hat`` is the soft Bayesian model average ``(N, K)`` produced
    by :func:`probly.representation._helpers.compute_mean_probs`.

    Args:
        logits: ``(N, K, S)`` raw pre-softmax class scores.
        support: Ignored; accepted for API parity with the regression
            losses. If provided, validated for shape compatibility.

    Returns:
        Dict with::

            "A_hat": (N,) float32.
            "E_hat": (N,) float32.
            "H_hat": (N, K) float32 -- the soft Bayes-optimal predictor.

    Raises:
        TypeError, ValueError: Per the validation helpers.
    """
    logits_arr = _validate_classification_inputs(logits, support)
    sample = _build_torch_sample_from_nks(logits_arr)
    # ty cannot prove that ``TorchCategoricalDistributionSample`` is a
    # structural subtype of ``SecondOrderDistributionLike`` (which is a
    # union of protocols); the wrapper at runtime works exactly as the
    # primitives expect, as exercised by the test_classification.py
    # parity tests against probly's own primitives.
    decomposition = SecondOrderEntropyDecomposition(distribution=cast("Any", sample))
    bma = compute_mean_probs(sample)
    return {
        "A_hat": decomposition.aleatoric.detach().cpu().numpy().astype(np.float32, copy=False),
        "E_hat": decomposition.epistemic.detach().cpu().numpy().astype(np.float32, copy=False),
        "H_hat": bma.probabilities.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def zero_one_decomposition(
    logits: np.ndarray,
    support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Zero-one decomposition of an ``(N, K, S)`` posterior tensor.

    Wraps probly's :class:`SecondOrderZeroOneDecomposition`. ``A_hat``
    and ``E_hat`` are reductions over the ``K`` and ``S`` axes;
    ``H_hat`` is the argmax of the Bayesian model average ``(N,)``
    int64.

    Args:
        logits: ``(N, K, S)`` raw pre-softmax class scores.
        support: Ignored; accepted for API parity. Validated if given.

    Returns:
        Dict with::

            "A_hat": (N,) float32.
            "E_hat": (N,) float32.
            "H_hat": (N,) int64 -- argmax of the Bayesian model average.

    Raises:
        TypeError, ValueError: Per the validation helpers.
    """
    logits_arr = _validate_classification_inputs(logits, support)
    sample = _build_torch_sample_from_nks(logits_arr)
    # See cast comment in cross_entropy_decomposition above.
    decomposition = SecondOrderZeroOneDecomposition(distribution=cast("Any", sample))
    bma = compute_mean_probs(sample)
    return {
        "A_hat": decomposition.aleatoric.detach().cpu().numpy().astype(np.float32, copy=False),
        "E_hat": decomposition.epistemic.detach().cpu().numpy().astype(np.float32, copy=False),
        "H_hat": bma.probabilities.argmax(dim=-1).detach().cpu().numpy().astype(np.int64, copy=False),
    }


__all__ = ["cross_entropy_decomposition", "zero_one_decomposition"]
