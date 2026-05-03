"""Regret-coverage and risk-coverage curves with their area-under-curve summaries.

This module provides loss-agnostic implementations of the
regret-coverage curve, its area-under-curve (AuReC), and the
parallel risk-coverage curve (AuRC) from the NeurIPS 2026 submission,
Section 6 (file: ``NeurIPS2026_paper_draft.pdf`` at the project root,
with TeX sources in the sister directory ``paper_tex/``).

The functions consume per-point arrays only:

* ``score`` is an uncertainty score; lower scores are accepted first.
* ``regret`` is the per-point regret ``E*(x)`` (i.e. the deployment
  loss minus the irreducible Bayes loss).
* ``risk`` is the per-point risk ``T*(x)`` (i.e. the full deployment
  loss).

Crucially, the loss itself is not visible at this layer. Loss
parametricity is the responsibility of the caller / oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from probly.evaluation._validation import (
    _check_1d_finite_real,
    _check_matching_lengths,
)


def _coverage_curve(score: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the rejection / coverage curve underlying both AuReC and AuRC.

    Sort by ``score`` ascending (stable sort, so ties preserve input
    order) and return the cumulative average of ``value`` divided by
    ``N`` at each accepted prefix length ``k = 0, 1, ..., N``.

    Args:
        score: Uncertainty score, shape ``(N,)``. Lower means accept.
        value: Per-point quantity to integrate (regret or risk),
            shape ``(N,)``.

    Returns:
        A tuple ``(rho, curve)``:
            * ``rho`` of shape ``(N + 1,)`` equals ``[0, 1/N, ..., 1]``.
            * ``curve[k]`` equals ``(1/N) * sum_{i in accepted}
              value[i]``, where the accepted set is the ``k`` lowest-
              score points (by stable sort).

    """
    n = score.shape[0]
    order = np.argsort(score, kind="stable")
    sorted_value = value[order].astype(np.float64, copy=False)
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(sorted_value, out=cum[1:])
    cum /= float(n)
    rho = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    return rho, cum


def regret_coverage_curve(
    score: Any,
    regret: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the regret-coverage curve.

    Sorts points by ``score`` ascending (stable sort: ties keep input
    order, so callers can predict the curve for tied inputs) and
    accumulates regret over the accepted prefix.

    Concretely, with ``N = len(score)`` and ``rho_k = k / N``,

    .. math::

        \\mathrm{Re}(\\rho_k)
        = \\frac{1}{N} \\sum_{i \\in \\mathrm{accepted}_k} \\mathrm{regret}[i],

    where ``accepted_k`` is the set of the ``k`` lowest-score points.
    This matches the paper's definition

    .. math::

        \\mathrm{Re}(C) = \\mathbb{E}_{x \\sim p^{*}} \\bigl[ C(x) \\cdot E^{*}(x) \\bigr],

    i.e. a normalised average over the FULL test set, not over the
    accepted subset.

    Args:
        score: Uncertainty score, shape ``(N,)``. Larger means
            "reject"; the lowest-score points are accepted first.
        regret: Per-point regret ``E*(x)``, shape ``(N,)``.

    Returns:
        A tuple ``(rho, regret_curve)``:
            * ``rho``: array of shape ``(N + 1,)`` equal to
              ``[0, 1/N, ..., 1]``.
            * ``regret_curve``: array of shape ``(N + 1,)``;
              ``regret_curve[k]`` is the average regret at
              acceptance fraction ``rho[k]``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If shapes are wrong, lengths mismatch, or the
            arrays contain non-finite entries.

    """
    score_arr = _check_1d_finite_real(score, name="score")
    regret_arr = _check_1d_finite_real(regret, name="regret")
    _check_matching_lengths(score_arr, regret_arr, name_a="score", name_b="regret")
    return _coverage_curve(score_arr, regret_arr)


def aurec(score: Any, regret: Any) -> float:
    """Compute the area under the regret-coverage curve (AuReC).

    Lower is better. The integral is taken with the trapezoidal rule
    over ``rho in [0, 1]`` on the ``(N + 1)``-point grid produced by
    :func:`regret_coverage_curve`. By construction ``Re(rho=0) = 0``
    and ``Re(rho=1) = mean(regret)``.

    For a single point ``aurec([s], [r])`` reduces to the area under
    the line from ``(0, 0)`` to ``(1, r)``, i.e. ``r / 2``.

    See the NeurIPS 2026 submission, Section 6
    (``NeurIPS2026_paper_draft.pdf`` with TeX sources in
    ``paper_tex/``).

    Args:
        score: Uncertainty score, shape ``(N,)``.
        regret: Per-point regret ``E*(x)``, shape ``(N,)``.

    Returns:
        The trapezoidal AuReC as a Python ``float``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If shapes are wrong, lengths mismatch, or the
            arrays contain non-finite entries.

    """
    rho, curve = regret_coverage_curve(score, regret)
    return float(np.trapezoid(curve, rho))


def risk_coverage_curve(
    score: Any,
    risk: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the risk-coverage curve.

    The construction mirrors :func:`regret_coverage_curve` but with
    the per-point risk ``T*(x)`` instead of the per-point regret.

    Args:
        score: Uncertainty score, shape ``(N,)``.
        risk: Per-point risk ``T*(x)``, shape ``(N,)``.

    Returns:
        A tuple ``(rho, risk_curve)`` of shapes ``(N + 1,)``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If shapes are wrong, lengths mismatch, or the
            arrays contain non-finite entries.

    """
    score_arr = _check_1d_finite_real(score, name="score")
    risk_arr = _check_1d_finite_real(risk, name="risk")
    _check_matching_lengths(score_arr, risk_arr, name_a="score", name_b="risk")
    return _coverage_curve(score_arr, risk_arr)


def aurc(score: Any, risk: Any) -> float:
    """Compute the area under the risk-coverage curve (AuRC).

    Lower is better. Trapezoidal integration over ``rho in [0, 1]``
    on the ``(N + 1)``-point grid.

    Args:
        score: Uncertainty score, shape ``(N,)``.
        risk: Per-point risk ``T*(x)``, shape ``(N,)``.

    Returns:
        The trapezoidal AuRC as a Python ``float``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If shapes are wrong, lengths mismatch, or the
            arrays contain non-finite entries.

    """
    rho, curve = risk_coverage_curve(score, risk)
    return float(np.trapezoid(curve, rho))
