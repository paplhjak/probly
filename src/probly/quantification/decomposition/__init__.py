"""Uncertainty decomposition methods.

In addition to the class-based decompositions
(:class:`SecondOrderEntropyDecomposition`,
:class:`SecondOrderZeroOneDecomposition`,
:class:`CredalSetEntropyDecomposition`), this package exposes two
loss-parametric numpy wrappers:

* :func:`decompose` -- consumes a method's cached ``(N, K, S)`` raw
  class scores and returns a dict of ``(A_hat, E_hat, H_hat)`` numpy
  arrays. This is the original sampling-method dispatcher.
* :func:`decompose_from_schema` -- generalised dispatcher that
  consumes a dict of cached arrays plus an :data:`OutputSchema`
  string. The schema string identifies the cache-format dialect
  (logits-with-S-axis, evidential-Dirichlet-alpha,
  DDU-probs-and-density, ...). The schema is recorded in each
  method's config (``method_config['output_schema']``) so the
  cache + decomposition layer is method-agnostic: a new method only
  needs a new wrapper, a new config schema field, and a new
  per-schema decomposition function.

Both wrappers bridge probly's torch-flavoured primitives to the
numpy-only caching layer used by the NeurIPS 2026 epistemic-eval
pipeline (see ``decisions.md`` -> "Loss handling in the codebase").
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from probly.quantification._validation import (
    _check_1d_finite_real,
    _check_3d_finite_real,
    _check_loss_string,
    _check_support_matches_k,
)

from .classification import cross_entropy_decomposition, zero_one_decomposition
from .ddu import ddu_decomposition
from .decomposition import (
    AdditiveDecomposition,
    AleatoricEpistemicDecomposition,
    AleatoricEpistemicTotalDecomposition,
    CachingDecomposition,
    Decomposition,
)
from .entropy import CredalSetEntropyDecomposition, SecondOrderEntropyDecomposition
from .evidential import evidential_decomposition
from .regression import absolute_decomposition, squared_decomposition
from .zero_one import SecondOrderZeroOneDecomposition

#: Bumped on math changes to invalidate caches.
_DECOMPOSITION_VERSION = 1

LossName = Literal["cross_entropy", "zero_one", "squared", "absolute"]

#: Output-cache schema identifier. Each UQ-method wrapper declares one
#: of these in its method config (``method_config['output_schema']``);
#: ``decompose_from_schema`` dispatches on the string. New schemas
#: must be added here AND in :func:`decompose_from_schema`'s body.
OutputSchema = Literal[
    "logits_nks",  # (N, K, S) raw class scores; sampling-based methods.
    "evidential_alpha",  # (N, K) Dirichlet alphas + (N,) evidence scalar.
    "ddu_probs_density",  # (N, K) softmax probs + (N,) marginal log-density.
]


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


def decompose_from_schema(
    predictions: dict[str, Any],
    loss: LossName,
    support: np.ndarray,
    *,
    schema: OutputSchema,
) -> dict[str, np.ndarray]:
    """Schema-aware dispatcher to the per-schema decomposition function.

    The cache schema is the output dialect a method writes to
    ``predictions.npz`` (e.g. ``logits`` of shape ``(N, K, S)`` for
    sampling-based methods, ``alpha`` + ``evidence`` for evidential
    methods, ``probs`` + ``density`` for DDU). The schema is
    declared in each method's config and recorded in the cache hash;
    this dispatcher reads the right keys from ``predictions`` and
    routes to the matching decomposition function.

    Args:
        predictions: Dict of cached numpy arrays keyed by their name
            in ``predictions.npz``. The required keys depend on
            ``schema``:

                ``"logits_nks"``       -> ``predictions["logits"]`` of shape ``(N, K, S)``.
                ``"evidential_alpha"`` -> ``predictions["alpha"]`` of shape ``(N, K)`` and
                                          ``predictions["evidence"]`` of shape ``(N,)``.
                ``"ddu_probs_density"`` -> ``predictions["probs"]`` of shape ``(N, K)`` and
                                          ``predictions["density"]`` of shape ``(N,)``.

        loss: One of ``"cross_entropy"``, ``"zero_one"``,
            ``"squared"``, ``"absolute"``. Not all schemas accept all
            losses; see the per-schema function's documentation.
        support: ``(K,)`` array of class labels. Same contract as
            :func:`decompose`. Used by the ``logits_nks`` path; the
            evidential and DDU paths derive K from the prediction
            arrays directly and ignore ``support``.
        schema: Cache schema identifier. See :data:`OutputSchema`.

    Returns:
        Dict with keys ``A_hat``, ``E_hat``, ``H_hat``. Per-loss
        ``H_hat`` shape contract is documented in :func:`decompose`
        and is preserved by every schema-specific decomposition.

    Raises:
        TypeError: If ``loss`` or ``schema`` is not a string.
        ValueError: If ``schema`` is unknown, the schema-required
            keys are missing from ``predictions``, or a per-schema
            constraint is violated (e.g. evidential alphas must be
            positive, DDU probs must be row-stochastic). Also raised
            if a ``schema``/``loss`` pair is unsupported (the
            evidential and DDU decompositions only accept
            ``cross_entropy`` and ``zero_one``).
    """
    if not isinstance(schema, str):
        msg = f"`schema` must be a string, got {type(schema).__name__}."
        raise TypeError(msg)
    if schema == "logits_nks":
        if "logits" not in predictions:
            msg = (
                f"schema={schema!r} requires `predictions['logits']` of shape "
                f"(N, K, S); got keys {sorted(predictions)}."
            )
            raise ValueError(msg)
        return decompose(predictions["logits"], loss, support)
    if schema == "evidential_alpha":
        missing = {"alpha", "evidence"} - set(predictions)
        if missing:
            msg = (
                f"schema={schema!r} requires keys {{'alpha', 'evidence'}} in "
                f"`predictions`; missing {sorted(missing)} (got {sorted(predictions)})."
            )
            raise ValueError(msg)
        # The evidential decomposition only accepts cross_entropy / zero_one.
        # The runtime check inside `evidential_decomposition` raises ValueError
        # on squared / absolute; we silence ty's stricter Literal narrowing so
        # callers don't need to pre-validate.
        return evidential_decomposition(
            predictions["alpha"],
            predictions["evidence"],
            loss,  # ty: ignore[invalid-argument-type]
        )
    if schema == "ddu_probs_density":
        missing = {"probs", "density"} - set(predictions)
        if missing:
            msg = (
                f"schema={schema!r} requires keys {{'probs', 'density'}} in "
                f"`predictions`; missing {sorted(missing)} (got {sorted(predictions)})."
            )
            raise ValueError(msg)
        # Same Literal-narrowing comment as above.
        return ddu_decomposition(
            predictions["probs"],
            predictions["density"],
            loss,  # ty: ignore[invalid-argument-type]
        )
    msg = f"unknown output_schema: {schema!r}; expected one of {OutputSchema.__args__!r}."  # type: ignore[attr-defined]
    raise ValueError(msg)


__all__ = [
    "_DECOMPOSITION_VERSION",
    "AdditiveDecomposition",
    "AleatoricEpistemicDecomposition",
    "AleatoricEpistemicTotalDecomposition",
    "CachingDecomposition",
    "CredalSetEntropyDecomposition",
    "Decomposition",
    "LossName",
    "OutputSchema",
    "SecondOrderEntropyDecomposition",
    "SecondOrderZeroOneDecomposition",
    "absolute_decomposition",
    "cross_entropy_decomposition",
    "ddu_decomposition",
    "decompose",
    "decompose_from_schema",
    "evidential_decomposition",
    "squared_decomposition",
    "zero_one_decomposition",
]
