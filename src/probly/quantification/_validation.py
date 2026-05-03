"""Internal input-validation helpers for the quantification package.

This module is intentionally private (leading underscore). The public
oracle and decomposition modules in :mod:`probly.quantification` import
from here, but it must not be re-exported from the package
``__init__``.

The helpers here are stricter cousins of those in
:mod:`probly.evaluation._validation`: they cover the multi-axis
``(N, K)`` and ``(N, K, S)`` numpy arrays consumed by the oracle and
the loss-parametric decomposition dispatcher.
"""

from __future__ import annotations

import numpy as np


def _check_1d_finite_real(arr: object, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 1-D, finite, real-valued ``ndarray``.

    Args:
        arr: Object to coerce. Anything ``np.asarray`` accepts is
            allowed; the result must be 1-D, non-empty, real-valued
            (numeric dtype, neither complex nor object), and contain no
            NaN or infinite entries.
        name: Argument name to embed in error messages.

    Returns:
        The coerced 1-D ``ndarray``. Boolean and integer inputs are
        kept in their original dtype; the caller decides whether to
        promote them.

    Raises:
        TypeError: If the dtype is complex or object.
        ValueError: If the array is not 1-D, is empty, or contains
            non-finite entries.
    """
    coerced = np.asarray(arr)

    if coerced.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`{name}` must be a real numeric array, got dtype={coerced.dtype!r}."
        raise TypeError(msg)

    if coerced.ndim != 1:
        msg = f"`{name}` must be 1-D, got ndim={coerced.ndim} with shape={coerced.shape}."
        raise ValueError(msg)

    if coerced.size == 0:
        msg = f"`{name}` must be non-empty."
        raise ValueError(msg)

    finite_view = coerced if coerced.dtype.kind == "f" else coerced.astype(np.float64)
    if not np.all(np.isfinite(finite_view)):
        msg = f"`{name}` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)

    return coerced


def _check_2d_finite_real(arr: object, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 2-D, finite, real-valued ``ndarray``.

    Used by the oracle to validate ``p_star`` of shape ``(N, K)``.

    Args:
        arr: Object to coerce. Anything ``np.asarray`` accepts.
        name: Argument name to embed in error messages.

    Returns:
        The coerced 2-D ``ndarray``.

    Raises:
        TypeError: If the dtype is complex or object.
        ValueError: If the array is not 2-D, is empty along either
            axis, or contains non-finite entries.
    """
    coerced = np.asarray(arr)

    if coerced.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`{name}` must be a real numeric array, got dtype={coerced.dtype!r}."
        raise TypeError(msg)

    if coerced.ndim != 2:
        msg = f"`{name}` must be 2-D, got ndim={coerced.ndim} with shape={coerced.shape}."
        raise ValueError(msg)

    if coerced.size == 0:
        msg = f"`{name}` must be non-empty, got shape={coerced.shape}."
        raise ValueError(msg)

    finite_view = coerced if coerced.dtype.kind == "f" else coerced.astype(np.float64)
    if not np.all(np.isfinite(finite_view)):
        msg = f"`{name}` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)

    return coerced


def _check_3d_finite_real(arr: object, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 3-D, finite, real-valued ``ndarray``.

    Used by the loss-parametric decomposition dispatcher to validate
    ``logits`` of shape ``(N, K, S)``.

    Args:
        arr: Object to coerce. Anything ``np.asarray`` accepts.
        name: Argument name to embed in error messages.

    Returns:
        The coerced 3-D ``ndarray``.

    Raises:
        TypeError: If the dtype is complex or object.
        ValueError: If the array is not 3-D, is empty along any axis,
            or contains non-finite entries.
    """
    coerced = np.asarray(arr)

    if coerced.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`{name}` must be a real numeric array, got dtype={coerced.dtype!r}."
        raise TypeError(msg)

    if coerced.ndim != 3:
        msg = f"`{name}` must be 3-D, got ndim={coerced.ndim} with shape={coerced.shape}."
        raise ValueError(msg)

    if coerced.size == 0:
        msg = f"`{name}` must be non-empty, got shape={coerced.shape}."
        raise ValueError(msg)

    finite_view = coerced if coerced.dtype.kind == "f" else coerced.astype(np.float64)
    if not np.all(np.isfinite(finite_view)):
        msg = f"`{name}` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)

    return coerced


def _check_loss_string(loss: object, *, name: str = "loss") -> str:
    """Validate that ``loss`` is one of the supported loss strings.

    Args:
        loss: Candidate loss string.
        name: Argument name to embed in error messages.

    Returns:
        The validated loss string (unchanged).

    Raises:
        TypeError: If ``loss`` is not a ``str``.
        ValueError: If ``loss`` is not one of the four supported
            strings.
    """
    allowed = ("cross_entropy", "zero_one", "squared", "absolute")
    if not isinstance(loss, str):
        msg = f"`{name}` must be a string, got {type(loss).__name__}."
        raise TypeError(msg)
    if loss not in allowed:
        msg = f"unknown loss '{loss}'; expected one of {allowed}."
        raise ValueError(msg)
    return loss


def _check_support_matches_k(support: np.ndarray, k: int, *, name: str = "support") -> None:
    """Raise ``ValueError`` if ``len(support) != k``.

    Args:
        support: The (already 1-D-validated) support array.
        k: The expected length, typically the ``K`` axis of a
            probability or logits tensor.
        name: Argument name to embed in error messages.

    Raises:
        ValueError: If lengths disagree.
    """
    if int(support.shape[0]) != int(k):
        msg = f"`{name}` length {support.shape[0]} does not match the K axis ({k})."
        raise ValueError(msg)
