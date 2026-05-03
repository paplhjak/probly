# probly audit (Task 1, re-issue)

Branch audited: `epistemic-eval` (cut from `first-order-data-integration`,
which is 6 commits ahead of `origin/main` — no library code added on
that branch beyond what the audit already covers).

All paths are relative to the probly repo root. Nothing under `probly/`
or `src/` was modified. Tests under `tests/probly/...` and an
out-of-scope branch `experiments/first_order_data/` exist but were not
edited.

A note on terminology before the questions: probly does **not** export
`A_hat` / `E_hat` as named outputs. The pipeline is

```
predictor  →  Sampler / representer  →  Sample  →  quantify(...)  →  Decomposition.{aleatoric, epistemic, total}
```

What we will write into `predictions.npz` as `(A_hat, E_hat)` is the
output of `quantify(sample).aleatoric / .epistemic` for an appropriate
loss-specific `Decomposition`. Mapping probly's primitives to the
paper's `(A_hat, E_hat)` is the central job of the per-method adapters
in `experiments/epistemic_eval/scripts/`.

**Re-verification vs. the prior audit.** The headline findings still
hold on this branch. No Laplace under `src/probly/method/`; no AuReC /
Pareto / IGD / regret anywhere; no APPA-REAL loader; only `entropy/`
and `zero_one/` decompositions; `subensemble/torch.py:55` still freezes
the backbone; first-order classes still at the same lines in
`src/probly/datasets/torch.py`.

---

## 1. Method inventory

probly exposes UQ methods as `@predictor_transformation` factories
declared in `src/probly/method/__init__.py:1` and lifted to `probly.method.<name>`.

Deep on the four working methods for the paper:

- **`dropout`** (`src/probly/method/dropout/_common.py:47`).
  Inserts `nn.Dropout` before each `nn.Linear` (`dropout/torch.py:25`),
  with `is_first_layer` skip (`dropout/_common.py:41`). Train: standard
  CE on the dropout-augmented net. Inference: wrap in
  `Sampler(model, num_samples=...)` (`representer/sampler/_common.py:101`).
- **`ensemble`** (`src/probly/method/ensemble/_common.py:82`).
  Duplicates the model into an `nn.ModuleList` of `num_members`;
  `reset_params=True` re-initialises each member (`ensemble/torch.py:29`).
  Train: each member trained independently (caller's loop). Inference:
  `predict_module_list` stacks (`ensemble/torch.py:40`); pass through
  `representer(...)`.
- **`evidential_classification`** (`src/probly/method/evidential/classification/common.py:42`).
  Appends `Softplus` + `+1` so the head outputs Dirichlet `α`
  (`classification/torch.py:18`). Train: one of `evidential_log_loss` /
  `evidential_ce_loss` / `evidential_mse_loss` plus an optional KL
  regulariser (`train/evidential/torch.py:307,332,357,388`). Inference:
  `predict()` returns `DirichletDistribution`.
- **`evidential_regression`** (`src/probly/method/evidential/regression/`).
  NIG head. Train: `der_loss` (`train/evidential/torch.py:843`).
  Inference: direct.

Remaining methods (one-line entries):

- `subensemble` (`subensemble/_common.py:37`) — frozen backbone +
  ensemble of heads (`subensemble/torch.py:55`).
- `bayesian` (`bayesian/_common.py:46`) — variational mean-field
  (Blundell), **NOT Laplace**.
- `dropconnect` (`dropconnect/_common.py:47`) — weight-drop dropout.
- `batchensemble` (`batchensemble/_common.py:67`) — rank-1 perturbations.
- `ddu` (`ddu/_common.py:34`) — spectral norm + density head (Mukhoti 2023).
- `posterior_network`, `natural_posterior_network`, `prior_network` —
  Dirichlet-output evidential variants (`posterior_network/_common.py:32`,
  `natural_posterior_network/_common.py`, `prior_network/_common.py`).
- `credal_bnn`, `credal_wrapper`, `credal_ensembling`,
  `credal_relative_likelihood`, `efficient_credal_prediction`, `dare` —
  credal-set variants.

**Critical gap.** No Laplace approximation in probly. `grep -rn -i
"laplace\|llla\|last.layer.gauss" src/probly/method/` returns no method
implementation. The `bayesian` method is variational mean-field
(`bayesian/_common.py:53` cites Blundell 2015), not Laplace.
**Implication for the paper:** `llla` is wired via the external
`laplace-torch` package (Daxberger et al.), pinned in
`pyproject.toml`. Whether to lift the wrapper into a probly library
module at `src/probly/method/laplace/` is a design decision deferred
to Task 5.

## 2. Common interface

Uniform for inference, **heterogeneous for training**.

**Inference (uniform).** Every method produces a `Predictor`. The path
is `representer(model, ...)` →
`Sampler` / `IterableSampler`
(`src/probly/representer/_representer.py:64`,
`src/probly/representer/sampler/_common.py:64,101`) →
`Sample` (`representation/sample/`) → `quantify(sample)` returning a
`Decomposition` (`src/probly/quantification/_quantification.py:24`)
with `.aleatoric`, `.epistemic`, `.total`
(`decomposition/decomposition.py:107`).

The selective-prediction script
`src/probly_benchmark/selective_prediction.py:52-60` is the canonical
end-to-end inference example. The minimal one-liner pattern is in
`src/probly_benchmark/method_scripts/dropout.py:14-28`.

**Training (heterogeneous).** No `.fit()`. Each method has its own
training requirements:

- `dropout`, `dropconnect`, `ensemble`, `subensemble`, `batchensemble`,
  `ddu`, `credal_wrapper`: standard CE / MSE; no probly-specific loss.
- `bayesian`, `credal_bnn`: ELBO with KL collected via
  `collect_kl_divergence` (`src/probly/train/bayesian/torch.py:31`).
- `evidential_*`, `posterior_network`, `natural_posterior_network`,
  `prior_network`, `dare`: per-method losses in
  `src/probly/train/evidential/torch.py` (300–1095). `prior_network` /
  `natural_posterior_network` additionally need an OOD-batch loader.

Implication: each adapter under `experiments/epistemic_eval/scripts/`
owns its training loop. A common `extract(handle, data) ->
ExtractionResult` is feasible and uniform; a common `train(...)` is not.

**Per-class predictive distributions** are exposed via `predict()`:
categorical for classifiers (`src/probly/predictor/_common.py:187`),
Dirichlet for evidential
(`src/probly/method/evidential/classification/common.py:54`), `Sample`
for ensembles. `compute_mean_probs` in `representation/_helpers/`
gives the BMA mean used in `selective_prediction.py:63`.

## 3. Loss handling — which decompositions exist

Decompositions live in `src/probly/quantification/decomposition/`
(`__init__.py:1`):

| Decomposition | Loss it corresponds to | File:line | A | E | T |
|---|---|---|---|---|---|
| `SecondOrderEntropyDecomposition` | log loss / cross-entropy | `decomposition/entropy/_common.py:25` | `conditional_entropy(Q)` | `mutual_information(Q)` (BALD) | `entropy_of_expected_predictive_distribution(Q)` |
| `SecondOrderZeroOneDecomposition` | 0/1 loss | `decomposition/zero_one/_common.py:20` | `expected_max_probability_complement` | `max_disagreement` | `max_probability_complement_of_expected` |
| `CredalSetEntropyDecomposition` | log loss on credal sets | `decomposition/entropy/_common.py:55` | `lower_entropy(C)` | `upper_entropy(C) − lower_entropy(C)` | `upper_entropy(C)` |
| `quantify_sample_variance` | (regression, sample variance) | `quantification/measure/sample.py:11` | n/a | n/a | sample variance |

flexdispatch primitives: `quantification/measure/distribution/_common.py:18-68`;
torch specialisations: `quantification/measure/distribution/torch.py:27-130`.

**Coupling for the three datasets.**

- **CIFAR-10H** (cross-entropy): `SecondOrderEntropyDecomposition`
  directly. `A_hat = conditional_entropy`, `E_hat = mutual_information`.
- **ImageNet-ReaL** (cross-entropy): same as CIFAR-10H. probly's
  `ImageNetReaL` (`src/probly/datasets/torch.py:46-95`) fills empty
  ReaL label sets with a uniform vector (line 78); we override that in
  our wrapper to drop empties per `decisions.md`.
- **APPA-REAL** (squared error AND absolute error). probly does not
  ship a Bregman / squared-error / CRPS / absolute-error decomposition.
  `quantify_sample_variance` returns variance only and treats it as
  `T_hat`, with no aleatoric / epistemic split. **Library work for
  Task 6.5:** add
  `src/probly/quantification/decomposition/{squared_error,absolute_error}.py`
  with the formulae in `decisions.md` → "APPA-REAL deployment losses".

In all cases the paper's `(A*, E*, H*)` are defined *given a
deployment loss*. probly does not expose a single "deployment loss"
parameter; the loss is implicit in the choice of decomposition class.
The oracle layer in `experiments/epistemic_eval/oracle/` mirrors this
and dispatches on a `loss` config string.

## 4. Dataset interface

There is **no formal `FirstOrderDataset` base class** in probly. The
de-facto contract is:

- subclass `torch.utils.data.Dataset` (or an existing torchvision
  dataset);
- override `__getitem__` to return `(image_tensor, dist)` where `dist`
  is a `(num_classes,)` `torch.Tensor` summing to 1
  (`datasets/torch.py:81`, `datasets/torch.py:181`);
- store the per-image `targets` as a tensor of distributions
  (`datasets/torch.py:28-43`, `122`).

`CIFAR10H` (`datasets/torch.py:19`) consumes
`cifar-10h-master/data/cifar10h-counts.npy` and normalises votes →
soft labels (line 43). `ImageNetReaL` (`datasets/torch.py:46`) reads
`reassessed-imagenet-master/real.json` and turns the multi-label list
into a uniform distribution over the labels, falling back to
uniform-over-classes for empty label sets (line 78). `DCICDataset`
(line 98) parses `annotations.json` of per-image votes into a
normalised vote-histogram (line 165).

**No APPA-REAL loader.** `grep -rni appa src/probly/` returns only
unrelated `kappa, alpha, beta` matches in
`src/probly/train/evidential/torch.py`. **Library work for Task 4:**
add `src/probly/datasets/appa_real.py` (or extend
`src/probly/datasets/torch.py`) yielding `p*` as a sparse
representation over the integer-age support per `decisions.md`.

Top-level dataset registration is empty: `src/probly/datasets/__init__.py:1`
is just a docstring. Imports are `from probly.datasets.torch import ...`.

The branch `first-order-data-integration` adds these dataset classes
plus `experiments/first_order_data/` (a self-contained DCIC ensemble
pipeline; +1851 LOC). That pipeline is **out of scope** for our work
per `.claude/settings.json` (deny on `Edit(experiments/first_order_data/**)`).

## 5. Existing evaluation utilities

**probly does not ship AuReC, regret-coverage, IGD+, Pareto-gap, or
the Theorem-1 selector.** Inventory:

- `src/probly/evaluation/__init__.py:1` — only re-exports
  `active_learning`.
- `src/probly/evaluation/tasks.py:8` — `selective_prediction(criterion,
  losses, n_bins)` sorts losses by `−criterion`, bins, and computes a
  trapezoid integral. The variable name `auroc` (line 25) is a
  misnomer; it is *not* the paper's AuReC. Differences:
  - bins coverage rather than per-instance sweep;
  - integrates risk (sorted loss), not regret = `risk − H*(p*)`;
  - no `H*` subtraction anywhere in probly;
  - no Pareto / 3-D `(coverage, risk, regret)` machinery.
- `src/probly/evaluation/ood.py` — OOD AUROC / AUPR / FPR@TPR. Useful
  if we add an OOD ablation; not for AuReC.
- `src/probly/evaluation/active_learning/loop.py` — irrelevant here.

**Library work for Task 3.** New modules under `src/probly/evaluation/`:

```
src/probly/evaluation/regret_coverage.py   # AuReC, risk-coverage curves
src/probly/evaluation/pareto_gap.py        # IGD+ between achievable and oracle surfaces
src/probly/evaluation/selectors.py         # Theorem-1 thresholded selectors + sweep helpers
```

Tests under `tests/probly/evaluation/`. The Rossellini sandbox is
out of scope; tests use closed-form 5-point hand-computed examples.

## 6. Determinism and seeding

No single seed entry point inside probly. Seeding is a caller
responsibility, with two re-usable helpers:

- `src/probly_benchmark/utils.py:27` — `set_seed(seed)` calls
  `random.seed`, `np.random.seed`, `np.random.default_rng(seed)`,
  `torch.manual_seed`, `torch.mps.manual_seed`,
  `torch.cuda.manual_seed{,_all}`. Uses the deprecated
  `np.random.seed` with `# noqa: NPY002`; we should not call it
  directly because of that and the wandb assumptions.
- `experiments/first_order_data/dcic_ensemble_pipeline.py:451` —
  per-pipeline `set_seed`, called per ensemble member with `seed =
  config.seed + member_index` (line 730).

Sampler-side RNG is not threaded through: `Sampler` does not take a
seed argument (`representer/sampler/_common.py:107-126`). Method-level
randomness:

- `dropout`, `dropconnect`: `rngs: int = 1` (Flax-only); torch dropout
  uses the global torch RNG.
- `bayesian`: variational draws use the global torch RNG.

**Implication.** Our extraction script in
`experiments/epistemic_eval/scripts/` must set up a single seeded
`numpy.random.Generator` and `torch.Generator` per process per the
project conventions in CLAUDE.md.

## 7. Linear-probe / frozen-backbone support

Critical for APPA-REAL. Per planned method:

- **Deep ensembles of heads.** Two paths:
  1. `probly.method.subensemble(base, num_heads, head_layer=1)`
     (`subensemble/_common.py:37`). Torch backend
     (`subensemble/torch.py:54-66`) freezes
     `backbone.parameters()` (line 55) and creates an `nn.ModuleList`
     of `(backbone, head_i)` compositions. Caveat: `head_layer`
     slicing only works for `nn.Sequential` models (line 36); for
     non-Sequential heads, pass `head=...` explicitly.
  2. Manual freeze + `ensemble`. Reference:
     `experiments/first_order_data/dcic_ensemble_pipeline.py:170-203`
     replaces the encoder's classification head with `nn.Identity`,
     builds a fresh head, freezes encoder params, then calls
     `ensemble(base_model, num_members=...)`. This is the cleanest
     template for the APPA-REAL MLP head.
- **MC-Dropout in head only.** `dropout` traverses all submodules and
  prepends `nn.Dropout` before every `nn.Linear`
  (`dropout/_common.py:64`), with `skip_if=is_first_layer` (line 41).
  Applied to a single-`Linear` head, the first-layer skip will skip
  the only layer. Workaround: apply `dropout` to a
  `nn.Sequential(Linear, Linear)` head, **or** add `nn.Dropout`
  manually in the head architecture (which is already the locked
  shape per `decisions.md`).
- **Last-Layer Laplace.** Not in probly. Wired via `laplace-torch`
  in the adapter at `experiments/epistemic_eval/scripts/method_llla.py`
  (or library at `src/probly/method/laplace/` — design TBD per Task 5).
- **Evidential head.** `evidential_classification` only requires the
  base to be a `LogitClassifier`
  (`evidential/classification/common.py:40`). Apply to the head, freeze
  the backbone separately, train with the chosen evidential loss.

**Bottom line.** Three of four methods (`mcd`, `ensemble`, `evidential`)
admit head-only application without modifying probly. `llla` requires
an external dependency.

## 8. Existing checkpoints

probly itself ships no checkpoints. No `from_pretrained`, no URLs in
source. Two indirect sources:

1. **wandb.** `probly_benchmark` is the canonical training/eval
   harness. `train.py:933-955` saves `model_state_dict + config` as a
   `.pt` artifact; `utils.py:153` `load_model_from_wandb` downloads
   and reconstructs. Naming (`utils.py:145`):
   `{method}_{base_model}_{dataset}_{seed}`. Recipes in
   `src/probly_benchmark/configs/recipe/` cover only `resnet18_cifar10`
   and `resnet50_imagenet`. **No APPA-REAL recipe.**
2. **Zenodo (DCIC).** First-order branch advertises pretrained
   ResNet-18 ensembles at <https://zenodo.org/records/19712415>
   (`experiments/first_order_data/README.md:303-321`). These use an
   ImageNet-pretrained backbone with frozen-backbone training of the
   head on DCIC CIFAR10H folds. Whether these are usable for our
   evaluation is **pending external confirmation** from the probly
   group per `decisions.md` → "CIFAR-10H training data".

For ImageNet-ReaL we need ImageNet-train-trained backbones. wandb if
the user has them; otherwise out of scope per `decisions.md` →
"Compute budget". For APPA-REAL the backbone is user-provided and
lives at `experiments/epistemic_eval/assets/appa_real_backbone/` per
the new layout.

## 9. Minimal working examples

The shortest end-to-end example for one method on one dataset is
`src/probly_benchmark/method_scripts/dropout.py:14-28`:

```python
from probly.method.dropout import dropout
from probly.quantification import quantify
from probly.representer import Sampler
from probly_benchmark.models import LeNet
import torch

model = LeNet(n_classes=5)
cep = dropout(model, p=0.5, predictor_type="probabilistic_classifier")
sampler = Sampler(cep, num_samples=10)
sample = sampler.predict(torch.randn(3, 1, 28, 28))
decomp = quantify(sample)
T_hat, A_hat, E_hat = decomp.total, decomp.aleatoric, decomp.epistemic
```

For ensembles (`method_scripts/deep_ensemble.py:14-23`):

```python
from probly.method.ensemble import ensemble
from probly.representer import representer

model = LeNet(n_classes=5)
cep = ensemble(model, num_members=10)
# train each member ...
rep = representer(cep)
sample = rep.predict(x)
decomp = quantify(sample)
```

For frozen-backbone + head ensemble (the APPA-REAL pattern), follow
`experiments/first_order_data/dcic_ensemble_pipeline.py:170-203` and
`run_dataset_experiment` (lines 644-808). Skeleton:

```python
base = ImageMlpHeadClassifier(encoder, n_classes, hidden=256, dropout_p=0.1,
                              pretrained=True, freeze_encoder=True)
members = ensemble(base, num_members=N, reset_params=False)
for i, member in enumerate(members):
    train_single_model(model=member, ..., seed=base_seed + i)
sample = ArrayCategoricalDistributionSample(
    array=ArrayCategoricalDistribution(np.stack(member_probs, axis=0)),
    sample_axis=0,
)
decomp = quantify(sample)
```

The full CIFAR-10H end-to-end inference path (with wandb checkpoint
load) is `src/probly_benchmark/selective_prediction.py:30-67`.

---

## Adapter signature sketches (where probly's interface needs an adapter)

### Last-Layer Laplace (external `laplace-torch`)

```python
# experiments/epistemic_eval/scripts/method_llla.py (or src/probly/method/laplace/)
from laplace import Laplace

def fit_llla(model, train_loader, head_module, *, hessian_structure="kron"):
    """Fit a last-layer Laplace approximation over the head only."""
    la = Laplace(model, "classification",
                 subset_of_weights="last_layer",
                 hessian_structure=hessian_structure,
                 last_layer_name="head")
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")
    return la

def llla_extract(la, x, n_samples: int) -> ExtractionResult:
    # Sample logits from the LLLA posterior, build a probly Sample, quantify
    ...
```

### Squared-error decomposition (library, Task 6.5)

```python
# src/probly/quantification/decomposition/squared_error/_common.py
@dataclass(frozen=True, slots=True)
class SquaredErrorDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Decomposition for the squared-error loss.

    Total = Var of mixture (= variance of conditional means + mean of
    conditional variances). Aleatoric = E_θ[Var_{y|θ}[y]]. Epistemic =
    Var_θ[E_{y|θ}[y]]. Additive (T = A + E) by the law of total variance.
    """
    distribution: SecondOrderDistributionLike
    ...
```

### Absolute-error decomposition (library, Task 6.5)

```python
# src/probly/quantification/decomposition/absolute_error/_common.py
@dataclass(frozen=True, slots=True)
class AbsoluteErrorDecomposition[T](Decomposition):  # NOT AdditiveDecomposition
    """Decomposition for the absolute-error loss.

    Aleatoric = E_{y~p*}[|y - median(p*)|]. Epistemic = E_θ[|median(p_θ)
    − median(p*)|]. T = A + E is NOT exact under L1; expose A and E
    separately, no additive total.
    """
    distribution: SecondOrderDistributionLike
    ...
```

These do NOT subclass `AdditiveDecomposition` under L1; the additive
property in `decomposition.py:146-171` does not hold for absolute error.

---

## Recommendations summary

1. Drop the implicit assumption that `T = A + E` in any loss-agnostic
   downstream code. AuReC and Pareto-gap consume `(A, E)` and per-point
   regret directly; they should not assume `T = A + E`.
2. Library work belongs in `src/probly/{evaluation,quantification/decomposition,datasets}/`.
   Paper-specific work (oracles on first-order datasets, configs, paper
   figures) belongs in `experiments/epistemic_eval/`.
3. Method adapters: thin shims under `experiments/epistemic_eval/scripts/`
   are sufficient for the four working methods. Lift `laplace-torch`
   to `src/probly/method/laplace/` only if the wrapper turns out to
   be generically useful — design choice in Task 5.
