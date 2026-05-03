# Methodological decisions

This file records locked choices for the **epistemic-eval** branch
(NeurIPS 2026 paper). **Claude must consult this before implementing
anything that touches data, losses, or the oracle.** If a decision is
`TBD`, surface the question to the user; do not guess.

Last updated: 2026-05-03 (re-issued after switch to in-probly architecture).

---

## Deployment loss

The framework in `NeurIPS2026_paper_draft.pdf` is parameterised by a
deployment loss `ℓ`. All of `T*, A*, E*, H*` and AuReC depend on it.
The codebase is loss-parametric: the loss is a top-level config field
with values in `{cross_entropy, zero_one, squared, absolute}`, and the
oracle and decomposition modules dispatch on it. AuReC, Pareto-gap, and
the Theorem 1 selector are loss-agnostic downstream.

| Dataset       | Loss(es) reported | Notes |
|---------------|-------------------|-------|
| CIFAR-10H     | cross-entropy     | Standard log-loss; matches the dominant convention in the UQ literature. |
| ImageNet-ReaL | cross-entropy     | Same. |
| APPA-REAL     | **squared error AND absolute error**, both reported as primary results | Loss-parametric APPA-REAL pipeline; see "APPA-REAL deployment losses" below. |

## APPA-REAL deployment losses

We report **both** squared error and absolute error as primary results
(not appendix). The pipeline is loss-parametric: each run carries a
loss string, the oracle dispatches on it, and the method-side
decomposition dispatches on it. AuReC and Pareto-gap consume the
resulting `(A_hat, E_hat)` arrays without knowing which loss produced
them.

APPA-REAL ages are **ordinal-discrete**; we treat them as real-valued
for both losses. This is acknowledged in paper §6.1.

### Squared loss  ℓ(ŷ, y) = (ŷ − y)²

- `H*(x) = E_{y ~ p*(·|x)}[y]`  (conditional mean of human votes)
- `A*(x) = Var_{y ~ p*(·|x)}[y]`  (irreducible noise / vote dispersion)
- `E*(x) = E_θ[(h(x, θ) − H*(x))²]`  (variance of per-θ posterior means)
- `T*    = A* + E*`  (exact, by orthogonality / law of total variance)
- Per-θ Bayes predictor: `h(x, θ) = E_{y ~ p(·|x, θ)}[y]`

### Absolute loss  ℓ(ŷ, y) = |ŷ − y|

- `H*(x) = median(p*(·|x))`  (lower-median tiebreak when ties occur)
- `A*(x) = E_{y ~ p*(·|x)}[|y − H*(x)|]`  (mean absolute deviation around the median)
- `E*(x) = E_θ[|h(x, θ) − H*(x)|]`  (per-θ-median deviation from H*)
- Per-θ Bayes predictor: `h(x, θ) = median(p(·|x, θ))`

### Identity vs. empirical-estimator additivity

Under both squared and absolute loss, the paper's identity
`T* = A* + E*` holds by definition (Definition 1).

What is loss-specific is the EMPIRICAL ESTIMATION of `A*` and `E*`
from finite-sample method outputs:

- For squared loss, the empirical estimators `A_hat` (mean of
  per-sample variances) and `E_hat` (variance of per-sample means)
  satisfy a closed-form identity with the marginal mixture
  variance via the law of total variance. This provides a clean
  correctness check for the implementation.

- For absolute loss, the empirical estimators `A_hat` (mean of
  per-sample MADs around per-sample medians) and `E_hat`
  (cross-sample dispersion of per-sample medians) do NOT enjoy a
  corresponding law-of-total-deviation. Their sum does not equal
  the marginal mixture MAD. This is a property of the L1 loss,
  not a violation of the paper's framework.

We report `A_hat` and `E_hat` under both losses as the
loss-specific empirical analogues of `A*` and `E*`. The
interpretation under squared loss has the additional comfort of
bias-variance exactness; under absolute loss it does not. We
acknowledge this in the paper's §6.1 limitations paragraph.

## p*(y | x) construction

| Dataset       | Construction of p*(y|x) |
|---------------|--------------------------|
| CIFAR-10H     | Empirical histogram over the 10 classes from the 50+ votes per image. Normalize to a probability vector. |
| ImageNet-ReaL | Filter to images with non-empty ReaL label sets. For single-label images, p* is one-hot. For multi-label images, use uniform mass `1/k` over the `k` labels. Drop empty-label images. **Pending**: the loader (Task 4) must report bucket fractions; if multi-label is the majority we revisit. |
| APPA-REAL     | Sparse vote histogram over the integer-age support (a `dict {age: count}` or sparse vector, NOT a dense one-hot). The support is the union of integer ages observed across the dataset; the loader documents this support in its docstring. Normalise to a probability vector at use-time. |

## H* — Bayes-optimal predictor

Definition: `H*(x) = arg min_y E_{y' ~ p*(y'|x)} [ℓ(y, y')]`. Per dataset:

- **CIFAR-10H**, cross-entropy: `H*(x) = p*(y | x)` itself.
- **ImageNet-ReaL**, cross-entropy: same — `H*(x) = p*(y | x)` on the
  kept (non-empty-label) images.
- **APPA-REAL**, squared error: `H*(x) = E_{y ~ p*}[y]`.
- **APPA-REAL**, absolute error: `H*(x) = median(p*(y | x))` with
  lower-median tiebreak.

## A* and E*

Per Definition 1 in the paper. The implementation must match the
formulae exactly. The oracle module computes these per test point given
`p*` and the chosen loss.

For Bayesian methods, `(θ, y) ~ p(θ, y | x, D)` involves the posterior
over θ. In our benchmark we treat the *learned* method's posterior as
the approximation; the **ground-truth** `A*` and `E*` are computed
from the human-annotated `p*` directly and do not depend on any
learned model. (This is the whole point of "first-order datasets".)

## APPA-REAL backbone strategy

- **Backbone:** user-provided pretrained age estimation model, large
  external corpus. **Path: TBD** (file lives at
  `experiments/epistemic_eval/assets/appa_real_backbone/` when
  available; weights gitignored, `backbone_meta.yaml` committed).
- **Head architecture (locked):** shallow MLP
  `Linear(in_features, hidden) → ReLU → Dropout(p) → Linear(hidden, K)`.
  Defaults: `hidden=256`, `p=0.1` (both config fields).
  - The Dropout layer participates in MC-Dropout at inference.
  - For ensemble members, dropout is disabled at inference (standard
    practice).
  - For LLLA, the Laplace approximation covers the second `Linear`
    layer's weights; dropout is disabled.
  - For evidential, the second `Linear` is replaced by the evidential
    head (Softplus + 1).
- **Adaptation on APPA-REAL train:** linear-probe / shallow-head
  training. Backbone features extracted once and cached; UQ training
  iterates over cached features, not raw images.

## CIFAR-10H training data

- Models train on CIFAR-10 train (50k images, no human annotations
  needed).
- Evaluation is on CIFAR-10 test (10k images), where human annotations
  give p*(y|x).
- Decision: reuse probly checkpoints if available **and** if probly's
  training set was CIFAR-10 train (not test).
- **Pending external confirmation**: user is sending one message to
  the probly group to confirm that the Zenodo CIFAR-10H ensemble
  checkpoints (https://zenodo.org/records/19712415, listed in
  `experiments/first_order_data/README.md`) were trained on CIFAR-10
  train and not on the CIFAR-10 test split. If the DCIC training fold
  contains CIFAR-10 test images, the checkpoints leak labels and are
  unusable. Non-blocking; if invalidated we fall back to a local
  CIFAR-10-train training run.

## ImageNet-ReaL training data

- Models train on ImageNet train.
- Evaluation is on ImageNet val with ReaL relabeling.
- Decision: reuse probly checkpoints if available, since probly's
  ImageNet models were almost certainly trained on ImageNet train.
  Confirmed via probly_audit.md §8 — wandb-fetch is the convention
  (`probly_benchmark/utils.py:153`, `load_model_from_wandb`).

## Methods to evaluate

The paper lists "MC-Dropout, Deep Ensembles, Last-Layer Laplace,
Evidential Networks". The pipeline is method-agnostic so any probly
method can be plugged in.

**Frozen list for the paper's main results table:** `mcd`, `ensemble`,
`llla`, `evidential`. Each gets a config in
`experiments/epistemic_eval/configs/methods/`.

- `mcd` → probly's `dropout` (`probly.method.dropout`).
- `ensemble` → probly's `ensemble` (`probly.method.ensemble`).
- `llla` → external `laplace-torch` (Daxberger et al.), pinned in
  `pyproject.toml` to `>=0.2.2,<0.3`. probly does not ship a Laplace
  approximation per `probly_audit.md` §1; the `bayesian` method is
  variational mean-field (Blundell), not Laplace. The adapter wraps
  `laplace.Laplace(...)` over the head only. Whether to lift the
  wrapper into a probly library module at `src/probly/method/laplace/`
  is a design decision deferred to Task 5.
- `evidential` → probly's `evidential_classification`; for APPA-REAL
  regression-style decompositions, `evidential_regression`.

**ImageNet-ReaL ensemble construction.** Deep Ensembles on ImageNet-ReaL
uses N independently pretrained classifiers (Lakshminarayanan et al.
2017's standard form), not a frozen-backbone head ensemble. The N
checkpoint paths are listed in `configs/datasets/imagenet_real.yaml`
under `ensemble_classifier_paths`. Acceptable proxies are torchvision
`ResNet50_Weights.IMAGENET1K_V1` / `V2` plus 3+ timm ResNet-50 variants
with documented training differences. The exact path list is locked
separately when checkpoints are assembled (a non-blocking item).

The pipeline accepts any probly method via config; this list is the
**frozen set for the paper**, not a code restriction.

## Seeds and replication

- **Number of seeds per `(method, dataset)` cell: 5.**
  - APPA-REAL linear-probe runs are cheap; 5 seeds essentially free.
  - CIFAR-10H and ImageNet-ReaL reuse probly checkpoints where
    possible; seeds in those cells correspond to UQ-side randomness
    (ensemble member init, dropout RNG, LLLA sampling), not to full
    backbone retraining.
- Seeds fixed and committed in the experiment configs.
- Deep Ensembles: ensemble size N is itself a hyperparameter; default
  N=5. The "seed" of an ensemble run is the seed of the seed-generator
  that produces the N member seeds.

## Compute budget

User-specified: a few H200 GPUs for 1-2 days. This implies:
- ImageNet runs must reuse checkpoints (training from scratch is out).
- CIFAR-10H runs can train from scratch if needed.
- APPA-REAL runs are cheap (linear probe on cached features).

## Loss handling in the codebase

- Loss is a **top-level config field** with values in
  `{cross_entropy, zero_one, squared, absolute}`.
- Each dataset config declares which losses it supports; experiment
  configs pick from that supported set.
- Library: probly ships `cross_entropy` and `zero_one` decompositions
  (`src/probly/quantification/decomposition/{entropy,zero_one}/`).
  Task 6.5 adds `squared_error` and `absolute_error` decompositions
  in the same package.
- Experiment: the oracle layer at
  `experiments/epistemic_eval/oracle/from_p_star.py` dispatches on the
  loss string and delegates to per-loss modules.
- AuReC, Pareto-gap, the Theorem 1 selector, and risk-coverage curves
  are loss-agnostic. They consume scalar `(A_hat, E_hat)` arrays plus
  per-point regret without knowing which loss produced them.

## Plot policy

**Metrics-only auto.** Every run automatically writes `metrics.json`
to its run directory. `aggregate_results.py` regenerates
`results_table.{md,csv}` from cached metrics. **Figures live behind
explicit scripts** under `experiments/epistemic_eval/scripts/make_figure_<N>.py`
that read cached metrics and emit `.pdf`/`.png` to a designated
figures directory. There is **no automatic figure regeneration**;
figures are recompiled when the user runs the figure scripts.

## Open questions for user

These remain after the Task 2 lock-in:

1. **APPA-REAL backbone path.** Still TBD. Required before the
   APPA-REAL run actually executes; not required to write the code.
2. **CIFAR-10H Zenodo checkpoint reuse** — pending external
   confirmation. Non-blocking; see "CIFAR-10H training data".
3. **ImageNet-ReaL bucket fractions.** Confirmation that the
   single-label fraction is large enough to keep the
   uniform-over-set + drop-empties policy. Reported by the loader in
   Task 4; revisit only if multi-label is the majority.
