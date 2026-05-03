"""Pareto-gap diagnostic and the underlying IGD+ metric.

Implements the Pareto-gap diagnostic of the NeurIPS 2026 submission,
Section 6 (file: ``NeurIPS2026_paper_draft.pdf`` at the project root,
with TeX sources in the sister directory ``paper_tex/``). Pareto-gap
is a modified inverted generational distance (IGD+) between the
oracle Pareto-optimal surface ``S^{*}`` and a method's achievable
surface ``S(A_hat, E_hat)`` in the three-dimensional operating-point
space ``(rho, risk, regret)``.

References:
    Ishibuchi, H., Masuda, H., Tanigaki, Y., Nojima, Y. "Modified
    Distance Calculation in Generational Distance and Inverted
    Generational Distance." Evolutionary Multi-Criterion Optimization
    (EMO 2015).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from probly.evaluation._validation import _check_int_at_least
from probly.evaluation.selectors import (
    sweep_empirical_surface,
    sweep_oracle_surface,
)

_PARETO_GAP_VERSION: int = 1
"""Module-level version of the Pareto-gap (IGD+) implementation.
Bumped on math changes; read by ``compute_metrics.py`` and written
into ``metrics_<loss>.json`` as ``_pareto_gap_version`` for cache
invalidation."""


def _check_2d_finite_real_three_columns(arr: Any, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 2-D, finite, real-valued ``(M, 3)`` array.

    Used by :func:`igd_plus` to validate its two array arguments.

    Args:
        arr: Object to coerce. Must yield a 2-D real numeric array
            with shape ``(M, 3)`` and at least one row.
        name: Argument name to embed in error messages.

    Returns:
        The coerced 2-D ``ndarray``.

    Raises:
        TypeError: If the dtype is complex or non-numeric.
        ValueError: If the array is not 2-D, has the wrong number of
            columns, is empty, or contains non-finite entries.

    """
    coerced = np.asarray(arr)
    if coerced.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`{name}` must be a real numeric array, got dtype={coerced.dtype!r}."
        raise TypeError(msg)
    if coerced.ndim != 2:
        msg = f"`{name}` must be 2-D with shape (M, 3), got ndim={coerced.ndim} with shape={coerced.shape}."
        raise ValueError(msg)
    if coerced.shape[1] != 3:
        msg = f"`{name}` must have exactly 3 columns, got shape={coerced.shape}."
        raise ValueError(msg)
    if coerced.shape[0] == 0:
        msg = f"`{name}` must contain at least one row."
        raise ValueError(msg)
    if coerced.dtype.kind in {"b", "i", "u"}:
        coerced = coerced.astype(np.float64)
    if not np.all(np.isfinite(coerced)):
        msg = f"`{name}` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    return coerced


def igd_plus(reference_set: Any, achievable_set: Any) -> float:
    """Compute the modified Inverted Generational Distance (IGD+).

    Treats every coordinate as a *minimisation* axis. For a single
    reference point ``r`` and an achievable point ``s``, the
    one-sided distance is

    .. math::

        d^{+}(r, s) = \\sqrt{\\sum_i \\max(s_i - r_i, 0)^2}.

    For a reference set ``S^{*}`` and an achievable set ``S``,

    .. math::

        \\mathrm{IGD}^{+}(S^{*}, S)
        = \\frac{1}{|S^{*}|}
            \\sum_{r \\in S^{*}} \\min_{s \\in S} d^{+}(r, s).

    The function is a pure minimisation-over-three-axes routine:
    callers are responsible for any coordinate flips needed to make
    every axis a minimisation axis (see :func:`pareto_gap`, which
    flips the coverage axis).

    Reference: Ishibuchi, H., Masuda, H., Tanigaki, Y., Nojima, Y.
    "Modified Distance Calculation in Generational Distance and
    Inverted Generational Distance." EMO 2015.

    Args:
        reference_set: ``(M, 3)`` array of reference points
            ``S^{*}``.
        achievable_set: ``(K, 3)`` array of achievable points ``S``.

    Returns:
        The IGD+ value as a Python ``float``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If either argument has the wrong shape or
            contains non-finite entries.

    """
    ref = _check_2d_finite_real_three_columns(reference_set, name="reference_set")
    ach = _check_2d_finite_real_three_columns(achievable_set, name="achievable_set")

    # Broadcast difference of shape (M, K, 3) and clip negatives so
    # that achievable points dominating the reference contribute zero.
    diff = ach[np.newaxis, :, :] - ref[:, np.newaxis, :]
    np.maximum(diff, 0.0, out=diff)
    sq = np.sum(diff * diff, axis=2)
    d_plus = np.sqrt(np.min(sq, axis=1))
    return float(np.mean(d_plus))


def _flip_coverage(points: np.ndarray) -> np.ndarray:
    """Replace the coverage column ``rho`` with ``1 - rho``.

    The first column of an operating-point block is coverage in
    ``[0, 1]``, where larger is better. IGD+ assumes minimisation on
    every axis, so we flip the coverage column before computing
    IGD+. Risk and regret are already minimisation axes.

    Args:
        points: Operating-point array of shape ``(M, 3)``.

    Returns:
        A new array of the same shape with the first column flipped.

    """
    flipped = points.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return flipped


def pareto_gap(
    a_hat: Any,
    e_hat: Any,
    a_star: Any,
    e_star: Any,
    *,
    n_lambda: int = 51,
    n_directions: int = 91,
) -> float:
    """Compute the Pareto-gap between oracle and method achievable surfaces.

    Operating points live in ``(rho, risk, regret)`` space, where
    coverage ``rho`` is a *maximisation* axis while risk and regret
    are *minimisation* axes. Before invoking :func:`igd_plus` (which
    treats every axis as minimisation), this function flips the
    coverage column to ``1 - rho`` on both surfaces while leaving
    risk and regret unchanged. The flip is the responsibility of
    :func:`pareto_gap`, not :func:`igd_plus`.

    End-to-end pipeline (see the NeurIPS 2026 submission, Section 6):

    1. Build the oracle surface
       ``S^{*} = sweep_oracle_surface(a_star, e_star, n_lambda=...)``.
    2. Build the empirical surface
       ``S = sweep_empirical_surface(a_hat, e_hat, a_star, e_star,
       n_directions=...)``.
    3. Flip the coverage column ``rho -> 1 - rho`` on both surfaces
       so that all three axes are minimisation axes (risk and regret
       already are; coverage was a maximisation axis).
    4. Return ``igd_plus(S^{*}_flipped, S_flipped)``.

    The function does **not** internally deduplicate operating points
    or filter the achievable surface to its Pareto front; this is by
    design. The empirical surface is consumed as produced by the
    direction sweep. Lower values of Pareto-gap mean tighter
    representational capacity of ``(a_hat, e_hat)`` relative to
    ``(a_star, e_star)``.

    Args:
        a_hat: Estimated aleatoric component, shape ``(N,)``.
        e_hat: Estimated epistemic component, shape ``(N,)``.
        a_star: Ground-truth aleatoric component, shape ``(N,)``.
        e_star: Ground-truth epistemic component, shape ``(N,)``.
        n_lambda: Number of points on the oracle ``lambda``-grid in
            ``[1/2, 1]``; defaults to ``51``. Must be ``>= 2``.
        n_directions: Number of angles on the empirical
            ``theta``-grid in ``[0, 2 pi)``; defaults to ``91``. Must
            be ``>= 2``.

    Returns:
        The Pareto-gap as a Python ``float``.

    Raises:
        TypeError: If inputs have non-real dtypes or grid sizes are
            not integers.
        ValueError: If shapes are wrong, lengths mismatch, the arrays
            contain non-finite entries, or grid sizes are ``< 2``.

    """
    # Validate grid sizes here too (the sweep helpers also validate,
    # but we want a clean error-path documented for this entry point).
    n_lambda_int = _check_int_at_least(n_lambda, minimum=2, name="n_lambda")
    n_dir_int = _check_int_at_least(n_directions, minimum=2, name="n_directions")

    s_star = sweep_oracle_surface(a_star, e_star, n_lambda=n_lambda_int)
    s = sweep_empirical_surface(
        a_hat,
        e_hat,
        a_star,
        e_star,
        n_directions=n_dir_int,
    )
    s_star_flipped = _flip_coverage(s_star)
    s_flipped = _flip_coverage(s)
    return igd_plus(s_star_flipped, s_flipped)
