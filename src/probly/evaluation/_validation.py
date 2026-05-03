"""Internal input-validation helpers for the regret-coverage metrics.

This module is intentionally private (leading underscore). Public
modules in :mod:`probly.evaluation` import from here, but it must not
be re-exported from the package ``__init__``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _check_1d_finite_real(arr: Any, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 1-D, finite, real-valued ``ndarray``.

    The metrics in :mod:`probly.evaluation.regret_coverage`,
    :mod:`probly.evaluation.selectors`, and
    :mod:`probly.evaluation.pareto_gap` consume per-point uncertainty
    arrays. Inputs may be Python lists, tuples, or numpy arrays; this
    helper performs the common coercion and validation.

    Args:
        arr: Object to coerce. Anything ``np.asarray`` accepts is
            allowed; the result must be 1-D, non-empty, real-valued
            (numeric dtype, neither complex nor object), and contain no
            NaN or infinite entries.
        name: Argument name to embed in error messages.

    Returns:
        The coerced 1-D ``ndarray``.

    Raises:
        TypeError: If the dtype is complex or object (i.e. not a real
            numeric dtype).
        ValueError: If the array is not 1-D, is empty, or contains
            non-finite entries.

    """
    coerced = np.asarray(arr)

    # Reject complex / object / string dtypes outright.
    if coerced.dtype.kind not in {"b", "i", "u", "f"}:
        msg = (
            f"`{name}` must be a real numeric array, "
            f"got dtype={coerced.dtype!r}."
        )
        raise TypeError(msg)

    if coerced.ndim != 1:
        msg = (
            f"`{name}` must be 1-D, got ndim={coerced.ndim} "
            f"with shape={coerced.shape}."
        )
        raise ValueError(msg)

    if coerced.size == 0:
        msg = f"`{name}` must be non-empty."
        raise ValueError(msg)

    # Promote bools and integers to float so downstream arithmetic and
    # finiteness checks behave uniformly. ``np.isfinite`` does not
    # accept boolean arrays.
    if coerced.dtype.kind in {"b", "i", "u"}:
        coerced = coerced.astype(np.float64)

    if not np.all(np.isfinite(coerced)):
        msg = f"`{name}` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)

    return coerced


def _check_finite_scalar(value: Any, *, name: str) -> float:
    """Validate that ``value`` is a finite real scalar and return it as ``float``.

    Args:
        value: Candidate scalar. Must be a Python ``int``/``float`` or
            a 0-D numpy scalar; complex values are rejected.
        name: Argument name to embed in error messages.

    Returns:
        The scalar coerced to ``float``.

    Raises:
        TypeError: If ``value`` is complex or not a real scalar.
        ValueError: If ``value`` is NaN or infinite.

    """
    if isinstance(value, complex):
        msg = f"`{name}` must be a real scalar, got complex value {value!r}."
        raise TypeError(msg)

    # ``np.asarray(scalar)`` produces a 0-D array; reject anything that
    # is not 0-D so callers cannot sneak a vector in.
    arr = np.asarray(value)
    if arr.ndim != 0:
        msg = f"`{name}` must be a scalar, got shape={arr.shape}."
        raise ValueError(msg)

    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = (
            f"`{name}` must be a real numeric scalar, "
            f"got dtype={arr.dtype!r}."
        )
        raise TypeError(msg)

    out = float(arr)
    if not math.isfinite(out):
        msg = f"`{name}` must be finite, got {out}."
        raise ValueError(msg)
    return out


def _check_matching_lengths(a: np.ndarray, b: np.ndarray, *, name_a: str, name_b: str) -> None:
    """Raise ``ValueError`` if ``a`` and ``b`` have different lengths.

    Args:
        a: First array.
        b: Second array.
        name_a: Argument name of ``a`` for error messages.
        name_b: Argument name of ``b`` for error messages.

    Raises:
        ValueError: If ``len(a) != len(b)``.

    """
    if a.shape[0] != b.shape[0]:
        msg = (
            f"`{name_a}` and `{name_b}` must have the same length, "
            f"got len({name_a})={a.shape[0]} and len({name_b})={b.shape[0]}."
        )
        raise ValueError(msg)


def _check_int_at_least(value: Any, *, minimum: int, name: str) -> int:
    """Validate that ``value`` is an integer ``>= minimum``.

    Args:
        value: Candidate integer.
        minimum: Inclusive lower bound.
        name: Argument name for error messages.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < minimum``.

    """
    # Reject bools (which are technically ``int`` in Python) explicitly
    # because they are almost never the caller's intent for grid sizes.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"`{name}` must be an integer, got {type(value).__name__}."
        raise TypeError(msg)
    out = int(value)
    if out < minimum:
        msg = f"`{name}` must be >= {minimum}, got {out}."
        raise ValueError(msg)
    return out
