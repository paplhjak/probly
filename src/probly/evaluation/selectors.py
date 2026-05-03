"""Theorem 1 thresholded selectors and achievable-surface sweeps.

Implements the unified-selector decomposition from Theorem 1 of the
NeurIPS 2026 submission, Section 6 (file:
``NeurIPS2026_paper_draft.pdf`` at the project root, with TeX sources
in the sister directory ``paper_tex/``).

A point is accepted when a score derived from per-point aleatoric
``A`` and epistemic ``E`` components falls below a threshold.
Theorem 1 in ``framework.tex`` requires ``lambda_`` in
``[1/2, 1]`` for the ground-truth components ``(A*, E*)``: setting
``lambda_ = 1/2`` recovers selective classification (threshold on
``A + E``), and ``lambda_ = 1`` recovers the epistemic reject-option
(threshold on ``E`` alone). The convex selector here accepts
``lambda_ in [0, 1]`` and lets the caller enforce the
``[1/2, 1]`` constraint when applicable.

The two sweep helpers enumerate operating points of the achievable
surfaces ``S^{*}`` (oracle) and ``S(A_hat, E_hat)`` (empirical) from
the same paper section. The empirical sweep ranks points by the
*estimated* components but reports coordinates in terms of the
*ground-truth* ones; this keeps Pareto-gap an honest deployment-cost
diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from probly.evaluation._validation import (
    _check_1d_finite_real,
    _check_finite_scalar,
    _check_int_at_least,
    _check_matching_lengths,
)


def threshold_selector(score: Any, tau: Any) -> np.ndarray:
    """Build the boolean accept mask ``score <= tau``.

    Args:
        score: Per-point selector score, shape ``(N,)``.
        tau: Threshold; points with ``score <= tau`` are accepted.

    Returns:
        Boolean array of shape ``(N,)``: ``True`` for accepted points.

    Raises:
        TypeError: If ``score`` has a non-real dtype or ``tau`` is
            complex / non-numeric.
        ValueError: If ``score`` is not 1-D / non-empty / finite, or
            ``tau`` is non-finite.

    """
    score_arr = _check_1d_finite_real(score, name="score")
    tau_val = _check_finite_scalar(tau, name="tau")
    return score_arr <= tau_val


def convex_selector(
    a: Any,
    e: Any,
    lambda_: Any,
    tau: Any,
) -> np.ndarray:
    """Build the convex Theorem 1 selector ``(1 - lambda_) * A + lambda_ * E <= tau``.

    Endpoints reproduce the canonical reject-option families:

    * ``lambda_ = 1/2`` thresholds ``(A + E) / 2`` and recovers
      selective classification (the threshold ``tau`` is in the
      half-sum scale).
    * ``lambda_ = 1`` thresholds ``E`` alone and recovers the
      epistemic reject-option.

    See the NeurIPS 2026 submission, Section 6, Theorem 1 (the
    unified-selector decomposition).

    Args:
        a: Per-point aleatoric component, shape ``(N,)``.
        e: Per-point epistemic component, shape ``(N,)``.
        lambda_: Convex weight in ``[0, 1]``. Theorem 1 requires
            ``[1/2, 1]`` for the ground-truth components, but we
            accept the full closed interval and leave the ``[1/2, 1]``
            constraint to the caller.
        tau: Threshold.

    Returns:
        Boolean array of shape ``(N,)``: ``True`` for accepted points.

    Raises:
        TypeError: If any array has a non-real dtype, or if scalars
            are complex / non-numeric.
        ValueError: If shapes are wrong, lengths mismatch, the arrays
            contain non-finite entries, ``lambda_`` is outside
            ``[0, 1]``, or ``tau`` is non-finite.

    """
    a_arr = _check_1d_finite_real(a, name="a")
    e_arr = _check_1d_finite_real(e, name="e")
    _check_matching_lengths(a_arr, e_arr, name_a="a", name_b="e")
    lam = _check_finite_scalar(lambda_, name="lambda_")
    if lam < 0.0 or lam > 1.0:
        msg = f"`lambda_` must be in [0, 1], got {lam}."
        raise ValueError(msg)
    tau_val = _check_finite_scalar(tau, name="tau")
    score = (1.0 - lam) * a_arr + lam * e_arr
    return score <= tau_val


def linear_selector(
    a: Any,
    e: Any,
    w1: Any,
    w2: Any,
    tau: Any,
) -> np.ndarray:
    """Build the linear selector ``w1 * A + w2 * E <= tau`` over the full plane.

    No constraint is placed on ``w1`` or ``w2``; the caller may pass
    arbitrary real weights to enumerate any half-plane in the
    ``(A, E)`` plane.

    Args:
        a: Per-point first component, shape ``(N,)``.
        e: Per-point second component, shape ``(N,)``.
        w1: Coefficient on ``a``.
        w2: Coefficient on ``e``.
        tau: Threshold.

    Returns:
        Boolean array of shape ``(N,)``: ``True`` for accepted points.

    Raises:
        TypeError: If any array has a non-real dtype, or if scalars
            are complex / non-numeric.
        ValueError: If shapes are wrong, lengths mismatch, the arrays
            contain non-finite entries, or any scalar is non-finite.

    """
    a_arr = _check_1d_finite_real(a, name="a")
    e_arr = _check_1d_finite_real(e, name="e")
    _check_matching_lengths(a_arr, e_arr, name_a="a", name_b="e")
    w1_val = _check_finite_scalar(w1, name="w1")
    w2_val = _check_finite_scalar(w2, name="w2")
    tau_val = _check_finite_scalar(tau, name="tau")
    score = w1_val * a_arr + w2_val * e_arr
    return score <= tau_val


def _operating_points_from_score(
    score_for_ranking: np.ndarray,
    a_for_coords: np.ndarray,
    e_for_coords: np.ndarray,
) -> np.ndarray:
    """Build the ``(N + 1, 3)`` block of operating points for one sweep direction.

    Given a 1-D ``score_for_ranking`` and the coordinate components
    ``a_for_coords`` / ``e_for_coords``, sort by score ascending
    (stable) and emit, for every prefix length ``k = 0, 1, ..., N``,
    the operating-point coordinates

    .. math::

        \\rho(C) = \\frac{1}{N} \\sum_i C(x_i),
        \\qquad
        \\mathrm{Ri}(C) = \\frac{1}{N} \\sum_i C(x_i)
            \\bigl( a(x_i) + e(x_i) \\bigr),
        \\qquad
        \\mathrm{Re}(C) = \\frac{1}{N} \\sum_i C(x_i)\\, e(x_i),

    where ``C`` is the indicator of acceptance.

    Args:
        score_for_ranking: Score the selector ranks by, shape ``(N,)``.
        a_for_coords: Component used for the risk coordinate
            (paired with ``e_for_coords`` as ``a + e``), shape
            ``(N,)``.
        e_for_coords: Component used for the regret coordinate, shape
            ``(N,)``.

    Returns:
        Array of shape ``(N + 1, 3)`` whose columns are
        ``(rho, risk, regret)``.

    """
    n = score_for_ranking.shape[0]
    order = np.argsort(score_for_ranking, kind="stable")
    a_sorted = a_for_coords[order].astype(np.float64, copy=False)
    e_sorted = e_for_coords[order].astype(np.float64, copy=False)
    risk_sorted = a_sorted + e_sorted

    rho = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    risk_curve = np.empty(n + 1, dtype=np.float64)
    regret_curve = np.empty(n + 1, dtype=np.float64)
    risk_curve[0] = 0.0
    regret_curve[0] = 0.0
    np.cumsum(risk_sorted, out=risk_curve[1:])
    np.cumsum(e_sorted, out=regret_curve[1:])
    risk_curve /= float(n)
    regret_curve /= float(n)

    return np.column_stack((rho, risk_curve, regret_curve))


def sweep_oracle_surface(
    a_star: Any,
    e_star: Any,
    *,
    n_lambda: int = 51,
) -> np.ndarray:
    """Enumerate operating points of the Pareto-optimal oracle surface ``S^{*}``.

    For each ``lambda_`` on a uniform grid of ``n_lambda`` points in
    ``[1/2, 1]``, sort points by the convex score
    ``(1 - lambda_) * A* + lambda_ * E*`` ascending and emit
    ``N + 1`` operating points ``(rho, risk, regret)`` (one per
    threshold position). The full output is the row-wise stack of all
    blocks, i.e. shape ``(n_lambda * (N + 1), 3)``.

    The coordinate definitions match those in
    :func:`sweep_empirical_surface` so the two surfaces share a common
    coordinate system for IGD+ computation.

    Args:
        a_star: Ground-truth aleatoric component, shape ``(N,)``.
        e_star: Ground-truth epistemic component, shape ``(N,)``.
        n_lambda: Number of points in the uniform grid on ``[1/2, 1]``;
            defaults to ``51``. Must be ``>= 2``.

    Returns:
        Array of shape ``(n_lambda * (N + 1), 3)`` with columns
        ``(rho, risk, regret)``.

    Raises:
        TypeError: If inputs have non-real dtypes or ``n_lambda`` is
            not an integer.
        ValueError: If shapes are wrong, lengths mismatch, the arrays
            contain non-finite entries, or ``n_lambda < 2``.

    """
    a_arr = _check_1d_finite_real(a_star, name="a_star")
    e_arr = _check_1d_finite_real(e_star, name="e_star")
    _check_matching_lengths(a_arr, e_arr, name_a="a_star", name_b="e_star")
    n_lambda_int = _check_int_at_least(n_lambda, minimum=2, name="n_lambda")

    lambdas = np.linspace(0.5, 1.0, n_lambda_int, dtype=np.float64)
    blocks = []
    for lam in lambdas:
        score = (1.0 - lam) * a_arr + lam * e_arr
        blocks.append(_operating_points_from_score(score, a_arr, e_arr))
    return np.concatenate(blocks, axis=0)


def sweep_empirical_surface(
    a_hat: Any,
    e_hat: Any,
    a_star: Any,
    e_star: Any,
    *,
    n_directions: int = 91,
) -> np.ndarray:
    """Enumerate operating points of a method's achievable surface ``S(A_hat, E_hat)``.

    The selector ranks points by the *estimated* components, but the
    operating-point coordinates are reported in terms of the
    *ground-truth* components. This is what makes Pareto-gap an
    honest deployment-cost measure: the method's representation is
    judged by what its selectors actually achieve under the ground
    truth, not by self-evaluation.

    Sweeping ``theta in [0, 2 pi)`` (uniformly spaced,
    ``endpoint=False``) paired with ``tau in R`` enumerates the full
    set of selectors of the form
    ``1{w1 * A_hat + w2 * E_hat <= tau}`` for
    ``(w1, w2, tau) in R^3``. The choice
    ``(w1, w2) = (cos theta, sin theta)`` is canonical because positive
    scaling of ``(w1, w2)`` (with the same scaling on ``tau``) defines
    the same selector. For each direction, sweeping ``tau`` over the
    ``N + 1`` distinct sorted thresholds enumerates the prefix chain of
    points sorted by ``cos theta * A_hat + sin theta * E_hat``
    ascending. The opposite direction ``theta + pi`` produces the
    suffix (complement) chain of the same point order -- these are
    different selectors in general, so the full circle is required to
    match the paper's ``(w1, w2) in R^2`` definition. We use
    ``endpoint=False`` to avoid duplicating the
    ``theta = 0 ~= 2 pi`` direction.

    For each direction ``theta`` on a uniform grid of ``n_directions``
    points in ``[0, 2 pi)``, sort by
    ``cos(theta) * A_hat + sin(theta) * E_hat`` ascending and emit
    ``N + 1`` operating points ``(rho, risk, regret)``, where the
    coordinates use ``(A*, E*)``:

    .. math::

        \\rho(C)         &= \\frac{1}{N} \\sum_i C(x_i), \\\\
        \\mathrm{Ri}(C)  &= \\frac{1}{N} \\sum_i C(x_i)
            \\bigl( A^{*}(x_i) + E^{*}(x_i) \\bigr), \\\\
        \\mathrm{Re}(C)  &= \\frac{1}{N} \\sum_i C(x_i) \\, E^{*}(x_i).

    Args:
        a_hat: Estimated aleatoric component, shape ``(N,)``.
        e_hat: Estimated epistemic component, shape ``(N,)``.
        a_star: Ground-truth aleatoric component, shape ``(N,)``.
        e_star: Ground-truth epistemic component, shape ``(N,)``.
        n_directions: Number of angles on the half-open interval
            ``[0, 2 pi)``; defaults to ``91``. Must be ``>= 2``.

    Returns:
        Array of shape ``(n_directions * (N + 1), 3)`` with columns
        ``(rho, risk, regret)``.

    Raises:
        TypeError: If inputs have non-real dtypes or
            ``n_directions`` is not an integer.
        ValueError: If shapes are wrong, lengths mismatch, the arrays
            contain non-finite entries, or ``n_directions < 2``.

    """
    a_hat_arr = _check_1d_finite_real(a_hat, name="a_hat")
    e_hat_arr = _check_1d_finite_real(e_hat, name="e_hat")
    a_star_arr = _check_1d_finite_real(a_star, name="a_star")
    e_star_arr = _check_1d_finite_real(e_star, name="e_star")
    _check_matching_lengths(a_hat_arr, e_hat_arr, name_a="a_hat", name_b="e_hat")
    _check_matching_lengths(a_hat_arr, a_star_arr, name_a="a_hat", name_b="a_star")
    _check_matching_lengths(a_star_arr, e_star_arr, name_a="a_star", name_b="e_star")
    n_dir_int = _check_int_at_least(n_directions, minimum=2, name="n_directions")

    thetas = np.linspace(
        0.0,
        2.0 * np.pi,
        n_dir_int,
        endpoint=False,
        dtype=np.float64,
    )
    blocks = []
    for theta in thetas:
        score = np.cos(theta) * a_hat_arr + np.sin(theta) * e_hat_arr
        blocks.append(_operating_points_from_score(score, a_star_arr, e_star_arr))
    return np.concatenate(blocks, axis=0)
