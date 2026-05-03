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
from paretoset import paretoset

from probly.evaluation._validation import _check_int_at_least
from probly.evaluation.selectors import (
    sweep_empirical_surface,
    sweep_oracle_surface,
)

_PARETO_GAP_VERSION: int = 2
"""Module-level version of the Pareto-gap (IGD+) implementation.
Bumped on math changes; read by ``compute_metrics.py`` and written
into ``metrics_<loss>.json`` as ``_pareto_gap_version`` for cache
invalidation.

History:
    v1: Initial implementation. ``pareto_gap`` ran ``igd_plus`` on
        the raw, fully-enumerated sweep surfaces. For ``N=10000``
        test points this required a 510k * 910k cross-product (~10
        TiB) and was made tractable only by random subsampling +
        chunking, producing approximate values.
    v2: ``pareto_gap`` now Pareto-filters the achievable surface
        ``S`` to its non-dominated subset before invoking
        ``igd_plus``. The filter is mathematically lossless:
        ``min_{s in S} d+(r, s)`` depends only on ``S``'s Pareto
        front, since any dominated ``q`` in ``S`` has a dominator
        ``s* <= q`` with ``d+(r, s*) <= d+(r, q)``. The reference
        surface ``S^{*}`` is passed through unchanged so v2 returns
        exactly the same value as the original "IGD+ on full sweep
        surfaces" definition, just computed efficiently. The
        Pareto-front step is delegated to :func:`paretoset.paretoset`,
        which handles the ~510k-910k row surfaces produced at
        ``N=10000`` in seconds. v2 also drops the v1 subsampling +
        random-seed knobs (``max_surface_points``,
        ``subsample_seed``); the metric is now deterministic."""


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

    # Chunked computation: for each chunk of ``ref`` points, compute
    # the broadcast distance against the entire ``ach`` set, take the
    # min along the achievable axis, and accumulate. The vectorised
    # form ``diff = ach[None, :, :] - ref[:, None, :]`` allocates an
    # ``(M, K, 3)`` array which can exceed memory for the ``N=10000``
    # surfaces produced on real datasets (10001 operating points per
    # angle/lambda direction; 51 * 10001 vs 91 * 10001 -> ~10 TiB
    # cross product). Cap the per-chunk allocation at ~256 MiB by
    # picking a chunk size whose product with ``len(ach)`` fits.
    chunk_bytes_target = 256 * 1024 * 1024  # 256 MiB
    bytes_per_pair = 3 * np.dtype(np.float64).itemsize
    chunk_size = max(1, chunk_bytes_target // (max(len(ach), 1) * bytes_per_pair))
    d_plus = np.empty(len(ref), dtype=np.float64)
    for start in range(0, len(ref), chunk_size):
        stop = min(start + chunk_size, len(ref))
        diff = ach[np.newaxis, :, :] - ref[start:stop, np.newaxis, :]
        np.maximum(diff, 0.0, out=diff)
        sq = np.sum(diff * diff, axis=2)
        d_plus[start:stop] = np.sqrt(np.min(sq, axis=1))
    return float(np.mean(d_plus))


def pareto_front(points: Any) -> np.ndarray:
    """Return the Pareto-non-dominated subset of ``points`` under min-on-all-axes.

    A point ``p`` is *dominated* if there exists another point ``q``
    in the set with ``q[i] <= p[i]`` for all ``i`` and
    ``q[j] < p[j]`` for at least one ``j``. The Pareto front is the
    set of points that are NOT dominated.

    Lossless for IGD+ on the *achievable* side: filtering ``S`` to
    its Pareto front does not change ``min_{s in S} d^{+}(r, s)`` for
    any reference point ``r`` (because the dominating ``q`` of any
    dominated ``s`` satisfies ``d^{+}(r, q) <= d^{+}(r, s)``). On the
    *reference* side, filtering ``S^{*}`` is a definitional choice
    (the resulting metric averages distance over a smaller set); v2
    of :func:`pareto_gap` adopts this choice so that the diagnostic
    is "IGD+ between Pareto fronts of the two surfaces."

    Implementation: delegates to :func:`paretoset.paretoset` from
    the ``paretoset`` PyPI package, a numba-JIT'd dominance filter
    that handles million-row 3-D inputs in well under a second. For
    the ~510k-row surfaces produced at ``N=10000`` this is the
    difference between minutes and seconds versus a hand-rolled
    sort-and-sweep, while still matching the brute-force baseline
    bit-for-bit.

    Args:
        points: ``(N, d)`` array (or array-like) of points. Coerced
            to a 2-D ``ndarray`` of finite real values.

    Returns:
        A ``(M, d)`` array with ``M <= N`` containing the
        non-dominated subset. Input ordering is preserved (the
        ``paretoset`` mask is applied directly to ``points``). The
        returned array is a copy.

    Raises:
        TypeError: If ``points`` has a non-real dtype.
        ValueError: If ``points`` is not 2-D or contains non-finite
            entries.
    """
    coerced = np.asarray(points)
    if coerced.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`points` must be a real numeric array, got dtype={coerced.dtype!r}."
        raise TypeError(msg)
    if coerced.ndim != 2:
        msg = f"`points` must be 2-D, got ndim={coerced.ndim} with shape={coerced.shape}."
        raise ValueError(msg)
    if coerced.dtype.kind in {"b", "i", "u"}:
        coerced = coerced.astype(np.float64)
    if coerced.size and not np.all(np.isfinite(coerced)):
        msg = "`points` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    n, d = coerced.shape
    if n == 0:
        return coerced.copy()
    if d == 0:
        # Degenerate: one "point" with no coordinates -> Pareto-front is just that one.
        return coerced[:1].copy()

    mask = paretoset(coerced, sense=["min"] * d)
    return coerced[mask].copy()


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
    4. Reduce the achievable surface ``S`` to its Pareto front via
       :func:`pareto_front`. This is mathematically lossless for
       IGD+: ``min_{s in S} d+(r, s)`` depends only on ``S``'s
       Pareto front. The oracle surface ``S^{*}`` is left
       unfiltered, both for exactness (filtering ``S^{*}`` would
       change the IGD+ value definitionally) and for tractability
       (the oracle sweep is empirically 98%+ on its own front, so
       filtering it would be both slow and a near no-op).
       Filtering ``S`` alone reduces the cross-product in
       :func:`igd_plus` from ``510k * 910k`` (~10 TiB at
       ``N=10000``) to ``510k * front-of-S``, where the empirical
       front is typically ~10-15% of the raw surface.
    5. Return ``igd_plus(S^{*}_pareto, S_pareto)``.

    The function does **not** internally deduplicate operating points
    beyond the Pareto-front reduction; the empirical surface is
    consumed as produced by the direction sweep. Lower values of
    Pareto-gap mean tighter representational capacity of
    ``(a_hat, e_hat)`` relative to ``(a_star, e_star)``.

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
    # Filter the achievable surface only. ``min_{s in S} d+(r, s)``
    # depends only on ``S``'s Pareto front, so this is lossless;
    # the value equals the brute-force IGD+ on the full surfaces.
    # The oracle surface is left unfiltered because (a) filtering
    # is unnecessary for exactness on the reference side and (b)
    # the oracle sweep typically has 98%+ of points on the front,
    # so filtering it is both expensive and a near no-op.
    s_pareto = pareto_front(s_flipped)
    return igd_plus(s_star_flipped, s_pareto)
