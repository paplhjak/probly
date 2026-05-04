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
| DCIC (9 datasets) | cross-entropy + zero-one | Classification family from Schmarje et al. 2022; see "DCIC datasets" below for the per-dataset list and locked recipe. |

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
  - For evidential, the second `Linear` is replaced by the evidential
    head (Softplus + 1).
  - For DDU, the head Linear is preserved as `classification_head`
    (probly's `ddu(...)` swaps the head with `nn.Identity()` and
    keeps the head separately so the encoder is a pure feature
    extractor); spectral norm is applied to all hidden Linear and
    Conv2d layers, and the post-hoc GMM density is fit on the
    encoder's output features.
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

## DCIC datasets

The DCIC benchmark (:cite:`schmarjeIsOne2022`, Zenodo
`records/7180818`) provides 10 multi-rater image classification
datasets with per-image vote distributions — exactly the soft-label
shape this framework needs. We wire the **9** non-CIFAR variants
into the pipeline (CIFAR10HDCIC is excluded because the existing
CIFAR-10H wiring covers the same images via a different on-disk
layout):

`Benthic`, `MiceBone`, `Pig`, `Plankton`, `QualityMRI`, `Synthetic`,
`Treeversity#1`, `Treeversity#6`, `Turkey`.

### Test/train fold convention

DCIC ships predefined 5-fold splits embedded in image paths (the
second path component, e.g. `Plankton/part1/img.png`).

- **Locked decision**: seed `N` runs against test fold
  ``sorted(folds)[N % 5]``. Five seeds therefore cover all five
  folds — full 5-fold cross-validation by construction, with no
  expansion of the run grid (we already run 5 seeds per
  `(method, dataset)` cell).
- The four non-test folds are the train pool; basecls training
  carves a 10% val split from that pool for early stopping.
- **Divergence from Oleg's reference**: the
  `experiments/first_order_data` pipeline picks a single test fold
  per invocation and does not rotate. We deliberately rotate per
  seed so the seed grid exercises every fold.

### Backbone

- **Locked decision**: torchvision `resnet18`, ImageNet-pretrained
  (`ResNet18_Weights.IMAGENET1K_V1`), with the final `nn.Linear`
  replaced by a fresh K-class head.
- Inputs are resized to 224x224 (`Resize(224, 224) + ToTensor +
  ImageNet-Normalize`); train-time gets `RandomHorizontalFlip`.
- One backbone family for all 9 DCIC datasets, mirroring Oleg's
  reference (`experiments/first_order_data/dcic_ensemble_pipeline.py`).
  Trade-off: we forgo the smaller `probly_benchmark.resnet.ResNet18`
  used by CIFAR-10H but keep the cross-DCIC comparison
  recipe-uniform (and ImageNet-pretrained weights are necessary at
  all on the small datasets like QualityMRI).

### Training recipe

Two recipes apply, separated by which classifier is being trained.
Both were tuned from a QualityMRI smoke-test (310 images, 60 test,
25 val): the initial AdamW lr=1e-3 was too aggressive (val_loss
spiked at epoch 2), 20 epochs left the model under-converged, and
patience=4 fired on noisy val_loss from a tiny val set. The locked
values below absorb that noise and let cosine LR settle the
fine-tune.

1. **basecls** (used by mc_dropout): AdamW, lr=3e-4, weight_decay=1e-4,
   batch_size=32, 200 epochs, **cosine LR schedule** with
   ``T_max=epochs``, early stopping with patience 100 on a 10% val
   split. Patience is set to half the epoch budget so early stopping
   acts only as a safety net against catastrophic divergence rather
   than as the principal stopping criterion — best-val-loss
   checkpointing keeps whichever epoch generalised best regardless
   of how long the cosine tail runs. Diverges from Oleg's
   `experiments/first_order_data/run_dcic_ensemble.py:30-47`
   defaults (lr=1e-3, 20 epochs, patience 4) — the divergence is
   intentional and documented here.

2. **From-scratch UQ training** (ensemble per-member, evidential,
   ddu): the method wrappers are SGD-locked per the cross-method
   recipe-uniformity decision (see "Methods to evaluate"). The DCIC
   dataset config provides `training.method_overrides` with tuned
   hyperparameters (lr=3e-3, momentum=0.9, weight_decay=1e-4, 200
   epochs, cosine via the wrapper's ``CosineAnnealingLR``) so SGD
   fine-tunes from ImageNet weights without destroying them. The
   SGD lr is 10x the AdamW basecls lr (3e-3 vs. 3e-4) because SGD
   typically wants a higher lr than AdamW; both are still 30x lower
   than the CIFAR-10H from-scratch lr=0.1. The optimizer choice
   differs from basecls (SGD vs. AdamW) by design: cross-method
   comparison within DCIC stays uniform (all four UQ methods use
   the same training recipe on the same dataset).

### Class label ordering

`probly.datasets.torch.DCICDataset` builds `label_mappings` from a
`set()` whose iteration order depends on `PYTHONHASHSEED`. To keep
column ordering of `targets` stable across processes (train vs.
extract vs. p* prep), the DCIC adapter wraps every loader instance
with `_DeterministicDCIC` which re-keys `label_mappings` by
`str(label)` order and rebuilds `targets` accordingly.

### p\* sidecar

Per-seed because the test fold rotates: the p* prep script
(`scripts/prepare_dcic_p_star.py`) writes
`data/<DatasetName>/p_star_seed<N>.npz`. The orchestrator passes the
right path to `compute_oracle.py --p-star-path` per run.

### Out of scope

- Multi-fold averaging or cross-validation aggregation. Each
  `(method, dataset, seed)` cell is one run with one test fold.
- Hyperparameter tuning per dataset. The locked recipe is shared
  across all 9 DCIC datasets; per-dataset deviations need an
  explicit decision update here.

## Methods to evaluate

The paper compares four UQ methods spanning the major UQ families:
variational sampling, multi-model uncertainty, single-pass
Dirichlet, and post-hoc density-based. Last-Layer Laplace
Approximation (LLLA) was considered and dropped — see "LLLA
dropped" subsection below.

**Frozen list for the paper's main results table:** `mcd`,
`ensemble`, `evidential`, `ddu`. Each gets a config in
`experiments/epistemic_eval/configs/methods/`.

- `mcd` → probly's `dropout` (`probly.method.dropout`). MC-Dropout
  (Gal & Ghahramani 2016) — variational sampling via Bernoulli
  dropout at inference.
- `ensemble` → probly's `ensemble` (`probly.method.ensemble`). Deep
  Ensembles (Lakshminarayanan et al. 2017) — multi-model uncertainty
  via independently-trained members.
- `evidential` → probly's `evidential_classification`
  (`probly.method.evidential.classification`). Evidential Deep
  Learning (Sensoy et al. 2018) — Dirichlet-output approach
  producing concentration parameters directly from a single forward
  pass. For APPA-REAL regression, `evidential_regression` (Amini
  et al. 2020) is the analog.
- `ddu` → probly's `ddu` (`probly.method.ddu`). Deep Deterministic
  Uncertainty (Mukhoti et al. 2023) — deterministic single-forward-pass
  UQ via post-hoc Gaussian Mixture Model density estimation in the
  spectrally-normalised feature space. Two-phase pipeline: train a
  classifier with spectral-norm-bounded hidden layers, then fit a
  class-conditional GMM on penultimate features.

### Why DDU specifically

DDU is included to test the paper's central empirical claim: good
OOD-detection performance does not imply good regret-ranking in the
first-order setting. We expect DDU to underperform the
sampling-based methods on AuReC despite being competitive on
traditional OOD-detection benchmarks. Either outcome is informative
for the paper.

### LLLA dropped

Last-Layer Laplace Approximation (LLLA) was considered and dropped:

1. probly's `epistemic-eval` branch does not contain a Laplace
   approximation module. An implementation exists on `main` but the
   paper-stage decision is not to merge `main` into `epistemic-eval`.
2. Adding LLLA via the external `laplace-torch` library was
   considered but rejected: LLLA, MC-Dropout, and mean-field BNN
   all live in the same "Gaussian posterior over weights" UQ
   family. Adding LLLA over the existing MC-Dropout would add an
   instance, not a family. The paper compares four UQ
   *families*; spending one slot on a second member of the
   variational-sampling family would not broaden the comparison.

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
    (ensemble member init, dropout RNG, evidential weight init,
    DDU GMM-fit perturbations), not to full backbone retraining.
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
