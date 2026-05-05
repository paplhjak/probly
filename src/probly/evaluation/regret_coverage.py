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

_AUREC_VERSION: int = 1
"""Module-level version of the AuReC implementation. Bumped on math
changes; read by ``compute_metrics.py`` and written into
``metrics_<loss>.json`` as ``_aurec_version`` for cache invalidation
across formula updates."""


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


def excess_aurec(score: Any, regret: Any) -> float:
    """Compute the excess area under the regret-coverage curve (E-AuReC).

    Subtracts the oracle AuReC (the theoretical minimum, achieved by
    sorting points ascending by their ground-truth ``regret``) from the
    measured AuReC. Lower is better; ``0.0`` indicates a theoretically
    perfect epistemic ranking.

    Excess AuReC has the **same units as the underlying loss**, so it
    is directly interpretable -- "the user is paying X units of
    avoidable per-point regret because the score isn't oracle-perfect."
    For cross-loss or cross-dataset comparisons use :func:`n_aurec`
    instead, which divides this excess by the random-score baseline
    and is unitless. The two metrics produce identical method
    orderings on a fixed ``regret`` vector
    (``excess_aurec / (0.5 * mean(regret) - oracle_aurec) == n_aurec``),
    so reporting both is redundant for ranking purposes; the value of
    ``excess_aurec`` is its loss-unit interpretability.

    Mirrors the "Excess AURC" construction from
    Geifman, Uziel & El-Yaniv (2018, "Bias-Reduced Uncertainty
    Estimation for Deep Neural Classifiers") but applied to the
    regret-coverage curve rather than the risk-coverage curve.

    Floating-point note. The return is non-negative *in exact
    arithmetic*. Floating-point underflow can produce values on the
    order of ``-1e-16`` when the score is the oracle ranking and
    ``regret`` has many ties; callers should not assert strict
    non-negativity.

    Args:
        score: Uncertainty score, shape ``(N,)``. Lower means accept
            first (same convention as :func:`aurec`).
        regret: Per-point ground-truth regret ``E*(x)``, shape
            ``(N,)``.

    Returns:
        The excess AuReC as a Python ``float``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If shapes are wrong, lengths mismatch, or the
            arrays contain non-finite entries.

    """
    measured = aurec(score, regret)
    # Sorting ``regret`` ascending against itself is the oracle order
    # because the (N + 1)-grid prefix sum of the sorted-ascending
    # regret minimises the trapezoidal area.
    oracle = aurec(score=regret, regret=regret)
    return float(measured - oracle)


def n_aurec(score: Any, regret: Any) -> float:
    """Compute the normalised area under the regret-coverage curve.

    Min-max normalisation of :func:`aurec` between the oracle and
    random-score baselines. The convention matches raw AuReC
    (lower-is-better):

    * ``0.0`` -- ranking is as good as the oracle (sorting by ground-
      truth ``regret``); the smallest AuReC achievable on this regret
      vector.
    * ``1.0`` -- ranking is no better than a uniformly-random score
      (in expectation).
    * ``> 1.0`` -- ranking is *worse* than random (e.g. an
      anti-correlated score systematically rejects the easy points).

    Why this exists. Raw ``aurec`` confounds two factors: how well the
    underlying predictor performs on the dataset (which sets the scale
    of ``regret``) and how well the uncertainty score ``ranks`` those
    regrets. A model with lower mean regret can win on raw AuReC even
    if its uncertainty scoring is the *worse* of two systems. The
    normalisation isolates ranking quality.

    The denominator uses the *expected* random-score AuReC,
    ``0.5 * mean(regret)`` (the area of the triangle from ``(0, 0)``
    to ``(1, mean(regret))``). This is the standard analytical
    baseline used in the Excess-AURC literature
    (Geifman, Uziel & El-Yaniv, 2018) and avoids Monte-Carlo noise
    from sampling random permutations.

    Relation to the AUGRC literature. The kernel
    :func:`_coverage_curve` divides cumulative regret by the *full* set
    size ``N`` rather than the number of accepted points ``k``; the
    resulting curve and area are the *generalised* (joint-expectation)
    risk-coverage quantities advocated by Jaeger, Traub et al.
    (2024) -- the formulation that sidesteps the monotonicity flaws of
    classical AURC. ``n_aurec`` adds a normalisation on top so the
    ranking-quality signal is comparable across datasets and
    predictors of differing capacity.

    Edge case. If every point has identical regret then
    ``oracle == random == measured`` and the denominator is zero. The
    ranking signal does not exist on such inputs (no ordering can do
    better than any other), and we return ``0.0`` -- the irreducible
    minimum is achieved trivially. Callers that need to distinguish
    "perfect" from "degenerate" should inspect the regret variance
    separately.

    Args:
        score: Uncertainty score, shape ``(N,)``. Lower means accept
            first (same convention as :func:`aurec`).
        regret: Per-point regret ``E*(x)``, shape ``(N,)``. Must be
            non-negative.

    Returns:
        The normalised AuReC as a Python ``float``.

    Raises:
        TypeError: If either argument has a non-real dtype.
        ValueError: If shapes are wrong, lengths mismatch, or the
            arrays contain non-finite entries.

    """
    score_arr = _check_1d_finite_real(score, name="score")
    regret_arr = _check_1d_finite_real(regret, name="regret")
    _check_matching_lengths(score_arr, regret_arr, name_a="score", name_b="regret")

    rho_meas, curve_meas = _coverage_curve(score_arr, regret_arr)
    measured = float(np.trapezoid(curve_meas, rho_meas))

    # Oracle: sorting by regret (same direction as score) minimises
    # every prefix sum, hence minimises the trapezoidal area.
    rho_oracle, curve_oracle = _coverage_curve(regret_arr, regret_arr)
    oracle = float(np.trapezoid(curve_oracle, rho_oracle))

    # Random baseline (expectation): area of the triangle from (0, 0)
    # to (1, mean(regret)).
    random_baseline = 0.5 * float(np.mean(regret_arr))

    denominator = random_baseline - oracle
    if denominator <= 0.0:
        return 0.0
    return float((measured - oracle) / denominator)


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
