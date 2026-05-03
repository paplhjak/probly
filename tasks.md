# Task list

Tasks are ordered. Do not start task N+1 until task N is reviewed and
accepted by the user. Each task has an explicit acceptance criterion.

For every task:
1. Read `CLAUDE.md`/`AGENTS.md`, `decisions.md`, and the relevant
   section of `NeurIPS2026_paper_draft.pdf` / `paper_tex/sections/*.tex`
   first.
2. Write a plan and have the user approve it before executing.
3. Run `pytest tests/` (and `uv run prek run --all-files` if editing
   library code) before declaring the task done.
4. Write a short summary in the task entry below.

**Code organisation rule.** Each piece of work is tagged
**[library]** if it lives in `src/probly/...` (upstream-quality,
loss-parametric, dataset-agnostic — eventually PR-able to probly), or
**[experiment]** if it lives in `experiments/epistemic_eval/...`
(paper-specific configs, oracle wrappers, extraction scripts, paper
figures — stays here).

---

## Task 1 — Audit probly

**Status.** Done (2026-05-03). See `probly_audit.md`. Re-issued after
the architecture switch to in-probly work; key findings unchanged
(no Laplace; no AuReC/Pareto/IGD/regret; no APPA-REAL loader; only
`entropy/` and `zero_one/` decompositions; subensemble freezes the
backbone; first-order classes already present in
`src/probly/datasets/torch.py`).

---

## Task 2 — Lock decisions and freeze method list

**Status.** Done (2026-05-03). `decisions.md` locks: APPA-REAL reports
both squared and absolute error as primary results; methods list
frozen to `{mcd, ensemble, llla, evidential}` with `llla` via external
`laplace-torch>=0.2.2,<0.3`; ImageNet-ReaL p* via uniform-over-set +
drop-empties (revisitable after Task 4 reports bucket fractions);
5 seeds per `(method, dataset)`; loss is a top-level config field with
oracle and decomposition modules dispatching on it; APPA-REAL head
locked to shallow MLP `Linear→ReLU→Dropout→Linear` (hidden=256, p=0.1
defaults); plot policy is metrics-only auto, figures gated behind
explicit scripts. Remaining TBDs (APPA-REAL backbone path, CIFAR-10H
checkpoint reuse pending external confirmation, ImageNet-ReaL bucket
fractions awaiting Task 4) are non-blocking for tasks 3–6.

---

## Task 3 — AuReC, selectors, and Pareto-gap as probly extensions  [library]

**Goal.** Extend probly with the metrics needed to benchmark expected
regret on first-order datasets.

**Library code** under `src/probly/evaluation/`:

- `regret_coverage.py` — given per-point arrays of `(uncertainty
  score, per-point regret)`, compute the regret-coverage curve and
  AuReC. Also expose risk-coverage for completeness.
  - Public API sketch:
    `aurec(scores, regrets, *, n_grid=None) -> float`,
    `regret_coverage_curve(scores, regrets, *, n_grid=None) -> (coverage, regret_array)`,
    `risk_coverage_curve(scores, losses, *, n_grid=None)`.
  - Per-instance sweep by default (set `n_grid` to bin into a fixed
    number of coverage points). Trapezoid integral over `[0, 1]`
    coverage.
- `pareto_gap.py` — given two sets of points in `(coverage, risk,
  regret)` space (the achievable surface `S(Â, Ê)` and the oracle
  surface `S*`), compute the modified IGD+ distance per Ishibuchi
  et al. 2015.
  - Public API sketch:
    `igd_plus(achievable, reference) -> float`,
    `pareto_gap(achievable, reference) -> float` (alias).
  - Both inputs are `(N, 3)` arrays in `(coverage, risk, regret)` order;
    minimisation is on risk and regret, maximisation on coverage,
    handled by an explicit sign convention.
- `selectors.py` — Theorem 1 thresholded selectors and sweep helpers.
  - `make_selector(A, E, lambda_, tau) -> np.ndarray` returns a
    boolean acceptance mask.
  - `sweep_lambda_tau(A, E, *, lambdas=None, taus=None) -> np.ndarray`
    enumerates `(λ ∈ [1/2, 1], τ ∈ ℝ)` to construct the **oracle**
    achievable surface for the paper's ground-truth `(A*, E*)`.
  - `sweep_w_tau(A, E, *, ...)` enumerates `(w₁, w₂, τ) ∈ ℝ³`
    (unconstrained) for the **method** achievable surface
    `S(Â, Ê)` per the paper §6.1.

Integration with probly's existing layout:
`src/probly/evaluation/__init__.py:1` currently re-exports only
`active_learning`. Extend it to re-export the new modules.
`src/probly/evaluation/tasks.py:8` `selective_prediction` is left
untouched; the new `regret_coverage.py` is the paper's AuReC, distinct
from the existing risk-only utility.

**Tests** under `tests/probly/evaluation/` (per probly's pytest
convention with `pytest.importorskip` for backend-specific bits):

- `test_aurec_closed_form.py` — fixed 5-point hand-computed example.
  The user-known AuReC value asserted to floating-point tolerance.
  No synthetic data generator, no Bayesian regression — just a fixed
  array of `(score, regret)` pairs and the known answer.
- `test_selectors_theorem1.py` — fixed toy `A`, `E` arrays. Verify:
  - `λ = 1/2` produces the same acceptance mask as thresholding
    `T = A + E`.
  - `λ = 1` produces the same acceptance mask as thresholding `E`.
  - For random `(λ, τ)`, the mask matches the indicator
    `(1−λ)·A + λ·E ≤ τ`.
- `test_pareto_gap.py` — fixed two surfaces with a known IGD+
  distance (worked out by hand on a small 4-point example); verify
  match.

**The synthetic Rossellini setup is OUT OF SCOPE for Task 3 and for
the project.** We use real datasets only.

**Acceptance.**

1. `pytest tests/probly/evaluation/` passes.
2. `pytest tests/` (the full probly suite) still passes.
3. The new modules are importable as
   `probly.evaluation.regret_coverage`, `probly.evaluation.pareto_gap`,
   `probly.evaluation.selectors`.
4. Each module has Google-style docstrings with formula and citation
   per probly's AGENTS.md conventions.
5. `uv run prek run --all-files` is clean.

**Status.** Not started.

---

## Task 4 — Dataset loaders  [library + experiment]

**Goal.** Ensure all three first-order datasets yield `(image, p*)`
under a uniform interface and write the per-dataset configs.

**Library work** in `src/probly/datasets/`:

- `CIFAR10H` and `ImageNetReaL` — already present in
  `src/probly/datasets/torch.py:19,46` per the audit. **Do not
  duplicate.** Override `ImageNetReaL.__getitem__` behaviour for
  empty-label sets at the experiment-config level (drop empties)
  rather than mutating the library class. Document the
  bucket-fraction reporting requirement (see below) in the loader
  docstring.
- `APPA-REAL` loader — **new library contribution**. Add as
  `src/probly/datasets/appa_real.py` (or extend
  `src/probly/datasets/torch.py` if probly's convention prefers
  consolidation). Yield `p*` as a sparse representation over integer
  ages (a `dict {age: count}` or sparse vector indexed by the chosen
  integer support). **Do NOT** return a dense one-hot. The integer
  support is the union of ages observed across the dataset; compute
  it once at construction time, store as `self.age_support: list[int]`,
  and document in the docstring.

**Experiment work** in `experiments/epistemic_eval/configs/datasets/`:

- `cifar10h.yaml`, `imagenet_real.yaml`, `appa_real.yaml` — paths,
  splits, supported losses (CE for the first two; squared and
  absolute for APPA-REAL), `p*` construction parameters.
- The ImageNet-ReaL config drives a **bucket-fraction report**:
  on first construction the loader-driver script writes a sidecar
  JSON with single-label / multi-label / dropped counts. If
  multi-label is the majority, surface this and revisit
  `decisions.md` before Task 5.

**Tests** under `tests/probly/datasets/`:

- `test_appa_real.py` — shapes; sparse-not-dense; `p*` sums to 1
  after normalisation; integer support documented; range of labels.
- Existing `test_torch.py` covers `CIFAR10H` and `ImageNetReaL`; do
  not duplicate.

**Acceptance.**

1. Loaders run end-to-end and yield non-empty p* tensors / sparse
   structures.
2. Tests pass.
3. The ImageNet-ReaL bucket-fraction report exists for the user to
   inspect.

**Status.** Not started.

---

## Task 5 — Method wrapper layer  [experiment, with one library design TBD]

**Goal.** A method-agnostic interface to probly such that adding a
method = adding a config file. Use the linear-probe pattern for
APPA-REAL.

**Experiment work** in `experiments/epistemic_eval/scripts/`:

- `method_registry.py` — registry mapping method name → factory.
- `method_{mcd,ensemble,llla,evidential}.py` — thin adapters calling
  probly (or `laplace-torch` for `llla`). Each exposes:
  - `train(config, data, seed) -> trained_method_handle`.
  - `extract(handle, data) -> ExtractionResult` carrying `logits`,
    `A_hat`, `E_hat`, `indices`, **and** posterior samples or
    method-specific sufficient statistics needed by the regression
    decompositions.
    - Sample-based methods (`mcd`, `ensemble`, `llla` via posterior
      sampling): `samples` of shape `(N_test, K_samples, ...)`.
    - Evidential methods that don't expose posterior samples: cache
      analytic sufficient statistics under a documented schema
      (Dirichlet `alpha` of shape `(N_test, C)` for classification,
      NIG `(γ, ν, α, β)` of shape `(N_test, 4)` for regression) and
      document the closed-form `(A_hat, E_hat)` expression used.
- `linear_probe.py` — feature extraction from a frozen backbone, then
  dispatch to the per-method head training. APPA-REAL only.

**Library design TBD.** Whether to lift the `laplace-torch` wrapper
into a probly library module at `src/probly/method/laplace/` is
deferred. Decision criterion: if the wrapper turns out to be
generically useful (i.e. callers other than this paper would want
it), promote; otherwise keep the experiment-side adapter.

**Configs** at `experiments/epistemic_eval/configs/methods/`:

- `mcd.yaml`, `ensemble.yaml`, `llla.yaml`, `evidential.yaml`.

**Acceptance.**

1. `experiments/epistemic_eval/scripts/extract_uncertainties.py
   --method-config X --dataset-config Y --seed Z` produces a
   `predictions.npz` for at least one (method, dataset) pair end-to-end.
2. Swapping the method config to a different method produces a valid
   `predictions.npz` with no other code change.

**Status.** Not started.

---

## Task 6 — Oracle layer  [experiment]

**Goal.** Compute and cache `(A*, E*, H*)` per dataset from the human
annotations and the dataset's deployment loss.

**Experiment work** in `experiments/epistemic_eval/oracle/`:

- `from_p_star.py` — generic
  `compute_oracle(p_star, loss: Literal["cross_entropy", "zero_one", "squared", "absolute"])`
  that **dispatches on the loss string** and delegates:
  - `cross_entropy` → probly's `SecondOrderEntropyDecomposition`
    formulae, applied to `p*` rather than to a method's posterior.
  - `zero_one` → probly's `SecondOrderZeroOneDecomposition` formulae.
  - `squared` → probly's `SquaredErrorDecomposition` (Task 6.5).
  - `absolute` → probly's `AbsoluteErrorDecomposition` (Task 6.5).
- `compute_oracle.py` (script) — runs the oracle on a dataset config,
  caches `oracle.npz`. Reads the loss from the experiment config and
  passes it to `compute_oracle`.

**Tests** at `experiments/epistemic_eval/tests/`:

- `test_oracle_classification.py` — for hand-computable cases (e.g.
  3-class p* on 5 points with cross-entropy), verify the numbers
  against probly's library decomposition output.

**Acceptance.**

1. Oracle outputs are deterministic and cached.
2. Tests pass.

**Status.** Not started.

---

## Task 6.5 — Regression decompositions for APPA-REAL  [library]

**Goal.** Implement and test the squared-error and absolute-error
decompositions used by the APPA-REAL oracle and methods. These are
**library work** because they generalise to any regression UQ user.

**Library work** in `src/probly/quantification/decomposition/`:

- `squared_error/` — new submodule paralleling `entropy/` and
  `zero_one/`. Contains:
  - `_common.py` with `SquaredErrorDecomposition`, subclassing
    `AdditiveDecomposition` (because `T = A + E` is exact under L2 by
    the law of total variance).
  - flexdispatch primitives in
    `src/probly/quantification/measure/distribution/_common.py`
    extended with `mean_of_means`, `var_of_means`, `mean_of_vars` if
    not already present.
- `absolute_error/` — same shape, but `AbsoluteErrorDecomposition`
  subclasses `Decomposition` (NOT `AdditiveDecomposition`) because
  `T = A + E` is NOT exact under L1. Expose `aleatoric` and
  `epistemic` as properties; do NOT expose an additive `total`.

Formulae per `decisions.md` → "APPA-REAL deployment losses".

**Tests** under `tests/probly/quantification/decomposition/`:

- `test_squared_error.py`:
  - Hand-compute `(H*, A*, E*)` for 5-point toy distributions over a
    small integer support (e.g. ages `{20, 21, 22, 23}` with hand-made
    vote histograms). Verify bit-exact match.
  - Verify additivity: `T == A + E` to floating-point tolerance on a
    non-degenerate case.
- `test_absolute_error.py`:
  - Same hand-computed `(H*, A*, E*)` check.
  - Verify **non-additivity**: pick a non-degenerate case where the
    cross-term is nonzero, assert `|T − (A + E)|` exceeds
    floating-point tolerance.

**Acceptance.**

1. `pytest tests/probly/quantification/decomposition/` passes.
2. Modules importable as
   `probly.quantification.decomposition.SquaredErrorDecomposition` and
   `probly.quantification.decomposition.AbsoluteErrorDecomposition`,
   re-exported from `decomposition/__init__.py`.
3. `uv run prek run --all-files` is clean.

**Status.** Not started.

---

## Task 7 — End-to-end evaluation  [experiment]

**Goal.** Plug method outputs and oracle outputs into the metrics
layer to produce AuReC and Pareto-gap per `(method, dataset, seed)`.

**Experiment work** in `experiments/epistemic_eval/scripts/`:

- `compute_aurec.py` — reads `predictions.npz` and `oracle.npz`,
  emits AuReC via `probly.evaluation.regret_coverage`.
- `compute_pareto_gap.py` — reads both, sweeps `(w₁, w₂, τ)` on the
  test set (oracle evaluation per the paper's §6.1) using
  `probly.evaluation.selectors.sweep_w_tau`, emits Pareto-gap via
  `probly.evaluation.pareto_gap`.
- `aggregate_results.py` — collects `metrics.json` across all runs;
  produces `results_table.{md,csv}`.

**Tests** at `experiments/epistemic_eval/tests/`:

- `test_pipeline_smoke.py` — end-to-end on a tiny synthetic dataset
  (10 points, 3 classes), all methods, single seed, in <30s.

**Acceptance.**

1. Smoke test passes.
2. Aggregated results table renders for a partial run grid.

**Status.** Not started.

---

## Task 8 — SLURM templates and launcher  [experiment]

**Goal.** A skeleton for the user to fill in cluster specifics.

**Experiment work** at `experiments/epistemic_eval/`:

- `slurm/extract.sbatch`, `slurm/evaluate.sbatch` — templates with
  `<TODO: USER>` placeholders.
- `scripts/launch_grid.py` — fans out `(method × dataset × seed)`
  over `sbatch` calls. Grid defined by an experiment config.
- `configs/experiments/main.yaml` — the main paper experiment grid.

**Acceptance.**

1. Templates work locally with `bash slurm/extract.sbatch ...`.
2. `launch_grid.py --dry-run` prints the full set of sbatch commands.

**Status.** Not started.

---

## Task 9 — Reproduce paper Figure 1 from a real run  [experiment]

User-driven verification task. Run the pipeline end-to-end on
CIFAR-10H with two methods (e.g. ensemble and MC-Dropout) and one
seed. Confirm numbers are sensible.

**Status.** Not started.

---

## Task 10 — Full grid  [experiment]

User launches the full experiment grid on the cluster. Claude
assists with any failures. **No new code unless explicitly requested.**

**Status.** Not started.
