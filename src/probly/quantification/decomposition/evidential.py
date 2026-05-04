"""Decomposition for evidential classification outputs.

Maps a method's cached ``(N, K)`` Dirichlet concentration parameters
``alpha`` to the ``(A_hat, E_hat, H_hat)`` triple consumed by the
loss-parametric evaluation pipeline. The ``evidence`` scalar that
accompanies ``alpha`` in the cache is accepted for schema
completeness; in the classification branches it does not enter the
formulae, while the regression branches use ``alpha_0 = K + evidence``
implicitly via the closed-form variance.

Classification decomposition (Sensoy et al. 2018, "Evidential Deep
Learning to Quantify Classification Uncertainty," NeurIPS 2018):

* ``alpha_0(x) = sum_k alpha[x, k]``
* Dirichlet mean ``p_hat(k|x) = alpha[x, k] / alpha_0(x)``
* Mass-based epistemic ``E_hat(x) = K / alpha_0(x)``  -- shrinks as
  the network sees more evidence (alpha_0 grows).
* For ``cross_entropy`` deployment loss:
    - ``H_hat(x) = p_hat(x)``  -- the soft Bayes-optimal predictor.
    - ``A_hat(x) = -sum_k p_hat(k|x) log p_hat(k|x)``  -- categorical
      entropy of the Dirichlet mean, in nats.
* For ``zero_one`` deployment loss:
    - ``H_hat(x) = argmax_k p_hat(k|x)``  -- hard label.
    - ``A_hat(x) = 1 - max_k p_hat(k|x)``  -- top-1 error of the
      Dirichlet mean.
    - ``E_hat`` unchanged.

Regression decomposition (extends Sensoy 2018 to integer-valued
``support`` via the ordinal-discrete trick used by APPA-REAL):

* For ``squared`` deployment loss the closed form is
    - ``H_hat(x) = E_{p ~ Dir(alpha)}[E_{y ~ Cat(p)}[y]] = sum_k support[k] * p_hat(k|x)``
    - ``A_hat(x) = Var_p[y] * alpha_0 / (alpha_0 + 1)``
    - ``E_hat(x) = Var_p[y] / (alpha_0 + 1)``
  where ``Var_p[y] = sum_k support[k]**2 * p_hat(k|x) - H_hat(x)**2``.
  ``A_hat + E_hat = Var_p[y]`` exactly (the squared decomposition's
  additivity is preserved by construction; see derivation in
  :func:`_squared_evidential_decomposition`).
* For ``absolute`` deployment loss the median-based functionals are
  non-smooth, so we sample ``S`` Dirichlet draws per row and compute
    - ``H_hat(x) = lower-median(p_hat(.|x))``
    - ``A_hat(x) = E_{y ~ p_hat}[|y - H_hat(x)|]``
    - ``E_hat(x) = std-of-medians_{p_s ~ Dir(alpha)}(median(p_s))``
  Default ``S = 20`` matches the ``logits_nks`` sampling-based estimators.

The heterogeneous ``H_hat`` contract matches the existing
classification dispatcher (``probly.quantification.decomposition.
classification``): cross-entropy returns a soft ``(N, K)`` predictor;
zero-one returns a hard ``(N,)`` int64 label; squared/absolute return
a ``(N,)`` scalar predictor (float32 for squared, int64 for absolute).
``A_hat`` and ``E_hat`` are always ``(N,) float32``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from probly.quantification._validation import _check_loss_string

_EvidentialLoss = Literal["cross_entropy", "zero_one", "squared", "absolute"]

#: Number of Dirichlet samples used for the absolute-loss decomposition.
#: Matches the ``logits_nks`` regression default; large enough to make
#: the per-row median estimate stable on the integer-age support.
_ABSOLUTE_NUM_SAMPLES: int = 20

#: Seed for the absolute-loss Dirichlet sampler. Independent of any
#: caller seed: the decomposition is a function of ``alpha`` only and
#: must be deterministic for cache hashing.
_ABSOLUTE_SAMPLE_SEED: int = 0


def _check_alpha(alpha: np.ndarray) -> np.ndarray:
    """Coerce ``alpha`` to a 2-D positive finite ``(N, K) float64`` array."""
    arr = np.asarray(alpha)
    if arr.ndim != 2:
        msg = f"`alpha` must be 2-D with shape (N, K), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        msg = f"`alpha` must contain at least one sample and one class, got shape={arr.shape}."
        raise ValueError(msg)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`alpha` must be a real numeric array, got dtype={arr.dtype!r}."
        raise TypeError(msg)
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        msg = "`alpha` must contain only finite values (no NaN or inf)."
        raise ValueError(msg)
    if np.any(arr <= 0.0):
        msg = "`alpha` must be strictly positive (Dirichlet concentration parameters)."
        raise ValueError(msg)
    return arr


def _check_evidence(evidence: np.ndarray, n: int) -> np.ndarray:
    """Coerce ``evidence`` to a 1-D non-negative finite ``(N,) float64`` array."""
    arr = np.asarray(evidence)
    if arr.ndim != 1:
        msg = f"`evidence` must be 1-D with shape (N,), got ndim={arr.ndim} shape={arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] != n:
        msg = f"`evidence` length {arr.shape[0]} does not match `alpha` length {n}."
        raise ValueError(msg)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        msg = f"`evidence` must be a real numeric array, got dtype={arr.dtype!r}."
        raise TypeError(msg)
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        msg = "`evidence` must contain only finite values."
        raise ValueError(msg)
    return arr


def _check_support(support: np.ndarray | None, k: int) -> np.ndarray:
    """Coerce ``support`` to a 1-D float64 ``(K,)`` array.

    Required by the ``squared`` and ``absolute`` regression branches
    so we know what scalar value each Dirichlet column corresponds to.
    """
    if support is None:
        msg = (
            "`support` is required for the squared / absolute evidential "
            "decompositions; got None. Pass an integer ``(K,)`` array of "
            "class labels (e.g. ``np.arange(K)`` for an age support)."
        )
        raise ValueError(msg)
    arr = np.asarray(support)
    if arr.ndim != 1:
        msg = f"`support` must be 1-D with shape (K,), got ndim={arr.ndim}."
        raise ValueError(msg)
    if arr.shape[0] != k:
        msg = (
            f"`support` length {arr.shape[0]} does not match the K dimension "
            f"of `alpha` ({k})."
        )
        raise ValueError(msg)
    return arr.astype(np.float64, copy=False)


def _squared_evidential_decomposition(
    alpha: np.ndarray, support: np.ndarray
) -> dict[str, np.ndarray]:
    """Closed-form squared-loss decomposition of a Dirichlet posterior.

    Given ``alpha`` of shape ``(N, K)`` and integer-valued ``support``
    of shape ``(K,)``, the decomposition is:

    * ``mu_k(x) = alpha[x, k] / alpha_0(x)``
    * ``H_hat(x) = sum_k support[k] * mu_k(x)`` (BMA mean predictor)
    * ``Var_p[y](x) = sum_k support[k]^2 * mu_k(x) - H_hat(x)^2``
    * ``A_hat(x) = Var_p[y](x) * alpha_0(x) / (alpha_0(x) + 1)``
    * ``E_hat(x) = Var_p[y](x) / (alpha_0(x) + 1)``

    Derivation: for ``theta ~ Dir(alpha)`` and ``h(theta) =
    sum_k support[k] * theta_k``, ``Var[h(theta)] = Var_p[y] /
    (alpha_0 + 1)`` (Dirichlet covariance). Aleatoric is
    ``E[Var_y|theta]`` which simplifies to ``Var_p[y] - E_hat``.
    Sum is ``Var_p[y]`` -- the additive contract the squared
    decomposition expects.
    """
    alpha0 = alpha.sum(axis=1)                                          # (N,)
    p_hat = alpha / alpha0[:, None]                                     # (N, K)
    h_hat = (p_hat * support[None, :]).sum(axis=1)                      # (N,)
    second_moment = (p_hat * (support[None, :] ** 2)).sum(axis=1)       # (N,)
    var_p = second_moment - h_hat ** 2                                   # (N,)
    np.maximum(var_p, 0.0, out=var_p)  # guard against tiny negatives
    inv_a0p1 = 1.0 / (alpha0 + 1.0)
    e_hat = var_p * inv_a0p1
    a_hat = var_p * (alpha0 * inv_a0p1)
    return {
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat.astype(np.float32, copy=False),
        "H_hat": h_hat.astype(np.float32, copy=False),
    }


def _absolute_evidential_decomposition(
    alpha: np.ndarray, support: np.ndarray
) -> dict[str, np.ndarray]:
    """Sample-based absolute-loss decomposition of a Dirichlet posterior.

    Median functionals are non-smooth, so we sample ``S = _ABSOLUTE_NUM_SAMPLES``
    Dirichlet draws per row using a fixed seed for cache-hashable
    determinism, and apply the same lower-median functional that
    :func:`probly.quantification.decomposition.regression.absolute_decomposition`
    uses on sampling-based methods.

    Per-row contracts:
    * ``H_hat(x) = lower-median(p_hat(.|x))``  -- BMA median.
    * ``A_hat(x) = E_{y ~ p_hat}[|support[y] - support[H_hat]|]``  --
      MAD around the BMA median.
    * ``E_hat(x) = E_{p_s ~ Dir(alpha)}[|support[median(p_s)] - support[H_hat]|]``
      -- mean per-sample-median deviation from the BMA median.

    These match the existing ``regression.absolute_decomposition``
    contract row-for-row; the only difference is that samples come
    from the Dirichlet posterior rather than from logits-NKS draws.
    """
    n, k = alpha.shape
    p_hat = alpha / alpha.sum(axis=1, keepdims=True)                     # (N, K)
    cum_p_hat = p_hat.cumsum(axis=1)
    h_hat_idx = np.argmax(cum_p_hat >= 0.5, axis=1).astype(np.int64)     # (N,)
    h_hat_value = support[h_hat_idx]                                     # (N,)
    a_hat = (np.abs(support[None, :] - h_hat_value[:, None]) * p_hat).sum(axis=1)

    # Sample S Dirichlet draws per row. ``np.random.Generator.dirichlet``
    # returns shape (S, K) per call so we batch-loop over rows.
    rng = np.random.default_rng(_ABSOLUTE_SAMPLE_SEED)
    sample_h_idx = np.zeros((n, _ABSOLUTE_NUM_SAMPLES), dtype=np.int64)
    for i in range(n):
        # ``rng.dirichlet`` returns shape (S, K).
        samples = rng.dirichlet(alpha[i], size=_ABSOLUTE_NUM_SAMPLES)
        cum = samples.cumsum(axis=1)
        sample_h_idx[i] = np.argmax(cum >= 0.5, axis=1)
    sample_h_value = support[sample_h_idx]                                # (N, S)
    e_hat = np.abs(sample_h_value - h_hat_value[:, None]).mean(axis=1)
    return {
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat.astype(np.float32, copy=False),
        "H_hat": h_hat_value.astype(np.float32, copy=False),
    }


def evidential_decomposition(
    alpha: np.ndarray,
    evidence: np.ndarray,
    loss: _EvidentialLoss,
    support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Decompose Evidential-Classification cached outputs.

    Args:
        alpha: ``(N, K)`` Dirichlet concentration parameters, all
            strictly positive (probly's evidential head produces
            ``softplus(logit) + 1`` so this holds by construction).
        evidence: ``(N,)`` per-point scalar evidence (the
            ``softplus(logit)`` sum, ``alpha_0 - K``). Cached but not
            used in the decomposition formulas; accepted for schema
            completeness.
        loss: One of ``"cross_entropy"``, ``"zero_one"``,
            ``"squared"``, ``"absolute"``. The classification branches
            (cross_entropy / zero_one) ignore ``support``; the
            regression branches (squared / absolute) require it.
        support: ``(K,)`` integer-valued class labels. Required for
            ``squared`` and ``absolute``; optional (and ignored) for
            ``cross_entropy`` and ``zero_one``.

    Returns:
        Dict with keys ``A_hat``, ``E_hat``, ``H_hat``. Per-loss
        ``H_hat`` shape contract::

            cross_entropy: H_hat is (N, K) float32 (soft predictor).
            zero_one:      H_hat is (N,)  int64   (argmax label).
            squared:       H_hat is (N,)  float32 (BMA mean).
            absolute:      H_hat is (N,)  float32 (BMA-median value
                                                   in support units;
                                                   matches the
                                                   ``logits_nks``
                                                   regression contract).

        ``A_hat`` and ``E_hat`` are always ``(N,) float32``.

    Raises:
        TypeError: If inputs have non-numeric dtypes or ``loss`` is
            not a string.
        ValueError: If shapes are wrong, ``alpha`` is non-positive,
            inputs contain non-finite entries, ``loss`` is unknown,
            or ``support`` is missing for a regression-side loss.
    """
    loss_validated = _check_loss_string(loss)
    if loss_validated not in {"cross_entropy", "zero_one", "squared", "absolute"}:
        msg = (
            f"`loss` must be one of {{'cross_entropy', 'zero_one', 'squared', "
            f"'absolute'}} for the evidential decomposition, got "
            f"{loss_validated!r}."
        )
        raise ValueError(msg)
    alpha_arr = _check_alpha(alpha)
    n, k = alpha_arr.shape
    _check_evidence(evidence, n)  # validate; not used downstream

    if loss_validated in {"squared", "absolute"}:
        support_arr = _check_support(support, k)
        if loss_validated == "squared":
            return _squared_evidential_decomposition(alpha_arr, support_arr)
        return _absolute_evidential_decomposition(alpha_arr, support_arr)

    alpha0 = alpha_arr.sum(axis=1)
    p_hat = alpha_arr / alpha0[:, None]
    e_hat = (k / alpha0).astype(np.float32, copy=False)

    if loss_validated == "cross_entropy":
        # Categorical entropy of p_hat in nats.
        a_hat = -(p_hat * np.log(np.clip(p_hat, 1e-12, 1.0))).sum(axis=1)
        h_hat = p_hat.astype(np.float32, copy=False)
        return {
            "A_hat": a_hat.astype(np.float32, copy=False),
            "E_hat": e_hat,
            "H_hat": h_hat,
        }
    # zero_one
    max_p = p_hat.max(axis=1)
    a_hat = 1.0 - max_p
    h_hat = p_hat.argmax(axis=1).astype(np.int64, copy=False)
    return {
        "A_hat": a_hat.astype(np.float32, copy=False),
        "E_hat": e_hat,
        "H_hat": h_hat,
    }


__all__ = ["evidential_decomposition"]
