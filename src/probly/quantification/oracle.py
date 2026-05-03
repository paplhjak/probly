"""Per-loss oracle from a first-order conditional ``p*(y | x)``.

Given a known conditional ``p*(y | x)`` (e.g. derived from dense human
annotations on a first-order dataset), this module computes the
ground-truth Bayes-optimal predictor ``H*``, the irreducible aleatoric
risk ``A*``, and the epistemic regret ``E*`` for one of four
deployment losses. The four losses match the project's locked
loss-parametric vocabulary (see ``decisions.md`` -> "Deployment loss"
and "Loss handling in the codebase"):

* ``"cross_entropy"`` -- log loss on a probability vector.
* ``"zero_one"`` -- 0/1 loss on a hard label.
* ``"squared"`` -- squared error on a real-valued prediction.
* ``"absolute"`` -- absolute error on a real-valued prediction.

The mathematical contract follows Definition 1 in
``paper_tex/sections/preliminaries.tex`` and the "Frequentist Paradox"
passage at lines 41-42 of the same file. Three load-bearing
properties:

1. **``E*`` is identically zero for all four losses on a first-order
   dataset.** Definition 1 expresses ``E*`` as the expected difference
   between the loss of the Bayesian predictor ``H(x, D)`` and the
   loss of the per-``theta`` Bayes predictor ``h(x, theta)``. When the
   conditional ``p*(y | x)`` is *known*, the Bayes-optimal predictor
   is itself the oracle; there is no posterior over ``theta`` to
   integrate out, so the regret is zero by construction. The
   "Frequentist Paradox" paragraph in the paper makes the same
   observation: "if the optimal predictor is known, estimating
   epistemic uncertainty to reject predictions is unnecessary; one
   would simply deploy the optimal predictor, reducing regret to
   zero." We therefore return ``E*`` as a vector of zeros for every
   loss; downstream code must not interpret these zeros as a bug.
2. **The shape of ``H*`` is loss-specific.** Cross-entropy returns
   the full ``(N, K)`` soft predictor (``p*`` itself); zero-one
   returns ``(N,)`` integer labels; squared and absolute return
   ``(N,)`` real-valued predictions. ``A*`` and ``E*`` are always
   ``(N,)`` float32 regardless of loss.
3. **The dispatch is a closed set.** Unknown loss strings raise
   ``ValueError``; there are no silent defaults.

The companion module
:mod:`probly.quantification.decomposition` exposes :func:`decompose`
which performs the analogous computation on a *learned* method's
``(N, K, S)`` posterior samples.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from probly.quantification._validation import (
    _check_1d_finite_real,
    _check_2d_finite_real,
    _check_loss_string,
    _check_support_matches_k,
)

#: Bumped on math changes to invalidate caches.
#:
#: History:
#:     v1 (Task 6): initial; per-loss ``oracle_<loss>.npz`` only.
#:     v2 (Task 7): ``compute_oracle.py`` also emits a loss-independent
#:         ``p_star.npz`` sidecar (``p_star``, ``support``, ``indices``);
#:         v1 caches lack it and are invalidated by this version bump.
# Bumped to 2 in Task 7: compute_oracle.py now emits p_star.npz sidecar; v1 caches lack it.
_ORACLE_VERSION = 2

LossName = Literal["cross_entropy", "zero_one", "squared", "absolute"]


def _normalise_p_star(p_star: np.ndarray) -> np.ndarray:
    """Coerce, validate and row-normalise ``p_star``.

    Rejects non-finite entries via :func:`_check_2d_finite_real` and
    rejects strictly negative probabilities, then divides each row by
    its row sum. Rows whose mass is zero raise ``ValueError`` -- a
    silent fallback would mask data-loading bugs.
    """
    arr = _check_2d_finite_real(p_star, name="p_star")
    arr = arr.astype(np.float64, copy=True)
    if np.any(arr < 0):
        msg = "`p_star` must have non-negative entries."
        raise ValueError(msg)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        msg = "`p_star` must have a strictly positive row sum for every row."
        raise ValueError(msg)
    return arr / row_sums


def _validate_support(support: np.ndarray | object, k: int) -> np.ndarray:
    """Validate ``support`` and confirm length agreement with ``K``."""
    arr = _check_1d_finite_real(support, name="support")
    _check_support_matches_k(arr, k)
    return arr


def _lower_median_index(p: np.ndarray) -> np.ndarray:
    """Return the lower-median column index of every row of ``p``.

    For each row, the lower median is the smallest column index ``k``
    whose cumulative probability mass reaches 0.5. Implemented via
    ``(cdf >= 0.5).argmax(axis=1)``: on a ``bool`` array, ``argmax``
    returns the first ``True`` index, which is exactly the lower-median
    tiebreak we want.

    Args:
        p: ``(N, K)`` row-stochastic probability matrix.

    Returns:
        ``(N,)`` int64 array of column indices.
    """
    cdf = np.cumsum(p, axis=1)
    return np.asarray((cdf >= 0.5).argmax(axis=1), dtype=np.int64)


def _oracle_cross_entropy(p_star: np.ndarray) -> dict[str, np.ndarray]:
    """Cross-entropy oracle.

    ``H*`` is the conditional itself (the optimal soft predictor);
    ``A*`` is its Shannon entropy under natural log;
    ``E*`` is identically zero.
    """
    safe = np.where(p_star > 0, p_star, 1.0)  # log(1.0) == 0 contributes nothing
    aleatoric = -np.sum(p_star * np.log(safe), axis=1)
    return {
        "H_star": p_star.astype(np.float32, copy=False),
        "A_star": aleatoric.astype(np.float32, copy=False),
        "E_star": np.zeros(p_star.shape[0], dtype=np.float32),
    }


def _oracle_zero_one(p_star: np.ndarray) -> dict[str, np.ndarray]:
    """Zero-one oracle.

    ``H*`` is the lowest-index argmax of every row (numpy's default
    tiebreak), ``A*`` is ``1 - max_k p_star[i, k]``, ``E*`` is zero.
    """
    h_star = np.asarray(p_star.argmax(axis=1), dtype=np.int64)
    aleatoric = (1.0 - p_star.max(axis=1)).astype(np.float32, copy=False)
    return {
        "H_star": h_star,
        "A_star": aleatoric,
        "E_star": np.zeros(p_star.shape[0], dtype=np.float32),
    }


def _oracle_squared(p_star: np.ndarray, support: np.ndarray) -> dict[str, np.ndarray]:
    """Squared-error oracle.

    ``H*`` is the conditional mean ``sum_k support[k] * p_star[i, k]``;
    ``A*`` is the conditional variance ``sum_k (support[k] - H*)^2 *
    p_star[i, k]``; ``E*`` is zero.
    """
    support_f = support.astype(np.float64, copy=False)
    h_star = (p_star * support_f[None, :]).sum(axis=1)
    diffs = support_f[None, :] - h_star[:, None]
    aleatoric = (p_star * diffs * diffs).sum(axis=1)
    return {
        "H_star": h_star.astype(np.float32, copy=False),
        "A_star": aleatoric.astype(np.float32, copy=False),
        "E_star": np.zeros(p_star.shape[0], dtype=np.float32),
    }


def _oracle_absolute(p_star: np.ndarray, support: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute-error oracle.

    ``H*`` is the lower median of ``p_star`` over ``support``,
    ``A*`` is the conditional mean absolute deviation around ``H*``,
    ``E*`` is zero.

    The output dtype of ``H*`` is ``int64`` if the input ``support`` is
    integer and ``float32`` otherwise. ``A*`` is always ``float32``.
    """
    support_f = support.astype(np.float64, copy=False)
    median_idx = _lower_median_index(p_star)
    h_star_values = support_f[median_idx]
    diffs = np.abs(support_f[None, :] - h_star_values[:, None])
    aleatoric = (p_star * diffs).sum(axis=1)
    if support.dtype.kind in {"i", "u"}:
        h_star_out: np.ndarray = support[median_idx].astype(np.int64, copy=False)
    else:
        h_star_out = h_star_values.astype(np.float32, copy=False)
    return {
        "H_star": h_star_out,
        "A_star": aleatoric.astype(np.float32, copy=False),
        "E_star": np.zeros(p_star.shape[0], dtype=np.float32),
    }


def compute_oracle(
    p_star: np.ndarray,
    loss: LossName,
    support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute ``(H*, A*, E*)`` per test point under the chosen loss.

    See module docstring for the full mathematical contract. In
    particular, **``E*`` is identically zero for all four losses** on
    a first-order dataset (where ``p*(y | x)`` is known); this is a
    direct consequence of Definition 1 in
    ``paper_tex/sections/preliminaries.tex`` and the "Frequentist
    Paradox" passage at lines 41-42 of that file.

    Args:
        p_star: Row-stochastic ``(N, K)`` array; rows are the known
            conditional probability vectors. Coerced to ``float64`` and
            normalised by row sum. Strictly negative entries or
            zero-mass rows raise ``ValueError``.
        loss: One of ``"cross_entropy"``, ``"zero_one"``, ``"squared"``,
            ``"absolute"``.
        support: Per-loss interpretation:

                cross_entropy: ignored (must be ``None`` or any 1-D
                               array of length ``K``).
                zero_one:      ignored (same).
                squared:       required ``(K,)`` real labels.
                absolute:      required ``(K,)`` real labels.

            For squared and absolute loss, the support carries the
            real-valued meaning of each column (e.g. integer ages on
            APPA-REAL).

    Returns:
        Dict with keys ``H_star``, ``A_star``, ``E_star``.

        ::

            Loss          | H_star shape | dtype
            cross_entropy | (N, K)       | float32 -- the soft predictor itself
            zero_one      | (N,)         | int64   -- the argmax label
            squared       | (N,)         | float32 -- the mean predictor
            absolute      | (N,)         | int64 if support is int; float32 otherwise

        ``A_star`` is always ``(N,)`` float32. ``E_star`` is always
        ``(N,)`` float32 and identically zero.

    Raises:
        TypeError: If ``loss`` is not a string or ``support`` has the
            wrong dtype.
        ValueError: If ``loss`` is unknown, ``p_star`` is malformed
            (wrong shape, non-finite, negative, zero-row), or
            ``support`` length mismatches ``K`` for losses that need it.
    """
    loss_validated = _check_loss_string(loss)
    p = _normalise_p_star(p_star)
    k = int(p.shape[1])

    if loss_validated == "cross_entropy":
        if support is not None:
            _validate_support(support, k)
        return _oracle_cross_entropy(p)
    if loss_validated == "zero_one":
        if support is not None:
            _validate_support(support, k)
        return _oracle_zero_one(p)
    if support is None:
        msg = f"loss '{loss_validated}' requires a `support` array of length {k}."
        raise ValueError(msg)
    support_arr = _validate_support(support, k)
    if loss_validated == "squared":
        return _oracle_squared(p, support_arr)
    return _oracle_absolute(p, support_arr)


def dense_from_sparse(
    p_star_sparse: list[dict[int, float]],
    support: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Convert a per-row sparse vote dict to a dense ``(N, K)`` matrix.

    Convenience for loaders that yield ``[{age: count}, ...]`` (for
    example, the APPA-REAL loader; see
    ``decisions.md`` -> "p*(y | x) construction"). The library oracle
    function ``compute_oracle`` always takes a dense ``(N, K)`` matrix;
    this helper bridges the two representations.

    Args:
        p_star_sparse: List of length ``N`` of dicts mapping a label
            (taken from ``support``) to a non-negative count or
            probability mass. Keys not in ``support`` are rejected to
            catch loader bugs.
        support: ``(K,)`` array of labels in the column order of the
            output. Must contain every key used by every row's dict.
        normalize: If ``True``, divide each row by its row sum to turn
            counts into a probability vector. Default ``True``.

    Returns:
        ``(N, K)`` ``float64`` array.

    Raises:
        TypeError: If ``p_star_sparse`` is not a list, an entry is not
            a dict, or a value is non-numeric.
        ValueError: If a row's key is not present in ``support``, a
            row's count is negative, or (with ``normalize=True``) a
            row's sum is zero.
    """
    if not isinstance(p_star_sparse, list):
        msg = f"`p_star_sparse` must be a list of dicts, got {type(p_star_sparse).__name__}."
        raise TypeError(msg)
    support_arr = _check_1d_finite_real(support, name="support")
    label_to_index: dict[float | int, int] = {
        (int(v) if support_arr.dtype.kind in {"i", "u"} else float(v)): i for i, v in enumerate(support_arr.tolist())
    }
    n = len(p_star_sparse)
    k = int(support_arr.shape[0])
    if n == 0:
        msg = "`p_star_sparse` must be non-empty."
        raise ValueError(msg)
    out = np.zeros((n, k), dtype=np.float64)
    for i, row in enumerate(p_star_sparse):
        if not isinstance(row, dict):
            msg = f"`p_star_sparse[{i}]` must be a dict mapping label -> count, got {type(row).__name__}."
            raise TypeError(msg)
        for label, count in row.items():
            key = int(label) if support_arr.dtype.kind in {"i", "u"} else float(label)
            if key not in label_to_index:
                msg = f"`p_star_sparse[{i}]` references label {label!r} which is not in `support`."
                raise ValueError(msg)
            value = float(count)
            if value < 0:
                msg = f"`p_star_sparse[{i}]` has a negative count for label {label!r}."
                raise ValueError(msg)
            out[i, label_to_index[key]] += value
    if normalize:
        row_sums = out.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            msg = "every row of `p_star_sparse` must have a strictly positive total mass."
            raise ValueError(msg)
        out = out / row_sums
    return out


__all__ = [
    "_ORACLE_VERSION",
    "LossName",
    "compute_oracle",
    "dense_from_sparse",
]
