# AGENTS.md
## Directives for Agents:
- If you are creating new files add them to git via `git add <file>`. If you forget to do this, your changes will not be committed and may be lost. Do not add bloat files like `__pycache__` or `*.pyc` files or auto-generated files to git.
- Always run pre-commit checks with `uv run prek run --all-files` before committing to ensure that your code adheres to the project's style and quality standards.
- When adding public-facing features add docstrings in Google-Style format and American english.
## Useful Commands:
### Build Docs (only use this command verbatim from the project root)
```bash
rm -rf docs/source/api && uv run sphinx-build -b html docs/source docs/build/html
```
Running docs building with errors on warning (to see if CI passes):
```bash
rm -rf docs/source/api && uv run sphinx-build -b html docs/source docs/build/html -W
```
### Run Pre-commit (takes only 1s)
```bash
uv run prek run --all-files
```
### Only Run type checking
```bash
ty check <path-to-file-or-directory>
```
## Example files to look at:
Examples of how to use probly lives in the examples directory. You can find tutorials on how to use the pytraverser flexdispatch and so forth.
## How Dispatch Works (short version):
Probly uses three dispatch layers to stay backend-agnostic:
1. **`flexdispatch`** (from `flextype`) -- type-based function dispatch, like `functools.singledispatch` but more flexible. Core operations (`predict`, `calibrate`, `ensemble_generator`) use this to route to the right backend implementation based on the predictor's type.
2. **`flexdispatch_traverser`** (from `pytraverse`) -- walks a neural network's layer tree and dispatches per-layer transformations by type. Methods like `dropout` and `batchensemble` use a traverser to apply backend-specific transforms to each layer.
3. **`switchdispatch`** (`probly/utils/switchdispatch.py`) -- dispatches on value equality instead of type. Used for the predictor registry to map name strings to predictor classes.
**Typical flow** (e.g. `dropout(torch_model)`):
```
user calls dropout(model)
  -> @predictor_transformation validates & infers backend
  -> traverse(model, dropout_traverser)
     -> traverser visits each layer, dispatches by type
     -> torch handler (method/dropout/torch.py) wraps the layer
```
**Lazy backend loading**: backends register via `delayed_register` in `__init__.py` files using fully-qualified type strings from `probly/lazy_types.py` (e.g. `TORCH_MODULE = "torch.nn.modules.module.Module"`). This means torch/flax/sklearn are only imported when actually needed.
## Common Mistakes to do right:
- `ty` may still fail to treat `np.ndarray` as a structural subtype of our `ArrayLike` protocol even after relaxing method requirements. Keep bounds as `ArrayLike | np.ndarray` where needed and use local `cast("Any", ...)` when dispatching ndarray-specific dunder methods.
- Do not use special unicode characters where it is not necessary (comments, docstrings, variable names)
- Tests are split by backend. Put backend-agnostic checks in `test_common.py`, and backend-specific checks in files like `test_array.py`, `test_torch.py`, or `test_jax.py`. In backend-specific test files, call `pytest.importorskip("<backend>")` at the top and avoid per-test skip decorators for missing optional deps.
- Pickle default-state behavior is subtle: `object.__getstate__()` may return `None` even when an instance has a populated `__dict__`. Do not use `super().__getstate__()` as a drop-in replacement for pickle's default state extraction when implementing cooperative `__getstate__` wrappers.

---

# Project: Epistemic Uncertainty Evaluation (NeurIPS 2026)

The sections below cover the paper-specific work happening on branch
`epistemic-eval`. Code in `experiments/epistemic_eval/` and changes
to `src/probly/{evaluation,quantification,datasets,method}/` for this
paper follow these conventions IN ADDITION to the probly-wide
conventions above.

## What this project is

Implementation of the empirical benchmark from
`NeurIPS2026_paper_draft.pdf` (Section 6). We compute **AuReC** (Area
under Regret-Coverage) and the **Pareto-gap** diagnostic for a suite
of UQ methods on three datasets with dense human annotations:

- **CIFAR-10H** — human soft labels on the CIFAR-10 test set
- **ImageNet-ReaL** — relabeled ImageNet validation set
- **APPA-REAL** — apparent age, integer human votes treated as `p*(y | x)`

The framework, definitions of `T*, A*, E*`, and Theorem 1 (the
unified selector) are in the paper. Always check
`NeurIPS2026_paper_draft.pdf` and `paper_tex/sections/*.tex` for
mathematical definitions before guessing.

## Code organisation

This is a single repository (probly). Two work surfaces co-exist:

- **Library code** in `src/probly/...` is upstream-quality:
  loss-parametric, dataset-agnostic, well-tested, Google-style
  docstrings, passes `prek`, `ty`, and `pytest`. Eventually PR-able to
  the probly maintainers. Specifically, this paper's library
  contributions are:
  - `src/probly/evaluation/` — AuReC, risk-coverage, regret-coverage
    curves, Pareto-gap (IGD+), Theorem 1 thresholded selectors.
    (Task 3.)
  - `src/probly/quantification/decomposition/` — squared-error and
    absolute-error decompositions, paralleling the existing `entropy/`
    and `zero_one/` submodules. (Task 6.5.)
  - `src/probly/datasets/` — APPA-REAL loader yielding `p*` over
    integer-age support. (Task 4.)
  - `src/probly/method/laplace/` — laplace-torch wrapper. **Pending
    design TBD in Task 5**: lift to library only if generically useful;
    otherwise the wrapper stays at experiment scope.
- **Experiment code** in `experiments/epistemic_eval/...` is
  paper-specific: configs, oracle wrappers (oracles only make sense
  for first-order datasets), extraction scripts, results aggregation,
  paper figures. Stays here, not upstreamed.

## Architecture: strict three-layer separation

Inside `experiments/epistemic_eval/`, the runtime pipeline is layered:

1. **Training / extraction layer**
   (`experiments/epistemic_eval/scripts/extract_uncertainties.py`).
   Inputs: `(method, dataset, seed)` config. Outputs: cached numpy
   arrays of `(A_hat, E_hat, logits, indices)` plus posterior samples
   or sufficient statistics, written to
   `experiments/epistemic_eval/runs/<run_id>/`. The only layer that
   touches GPUs.
2. **Oracle layer**
   (`experiments/epistemic_eval/scripts/compute_oracle.py`). Inputs:
   dataset config. Outputs: cached `(A_star, E_star, H_star, p_star)`
   per test point. Depends only on the dataset and the loss, not on
   any method.
3. **Evaluation layer**
   (`experiments/epistemic_eval/scripts/{compute_aurec,compute_pareto_gap}.py`).
   Inputs: cached arrays from layers 1 and 2. Outputs: scalar metrics,
   plots, tables. Fast and re-runnable in seconds. Calls into the
   `probly.evaluation` library.

Adding a new UQ method should require **a new method config file
only**, no code changes outside `experiments/epistemic_eval/scripts/`.

## Loss-parametricity

The framework is parameterised by a deployment loss `ℓ`. The loss is
a top-level config field with values in
`{cross_entropy, zero_one, squared, absolute}`. Each dataset config
declares which losses it supports; experiment configs pick from that
set. Dispatch:

```
config.loss (string)
   │
   ├─→ experiments/epistemic_eval/oracle/from_p_star.py
   │      dispatches on the string and delegates to:
   │        cross_entropy → probly.quantification.decomposition.SecondOrderEntropyDecomposition
   │        zero_one      → probly.quantification.decomposition.SecondOrderZeroOneDecomposition
   │        squared       → probly.quantification.decomposition.SquaredErrorDecomposition  (Task 6.5)
   │        absolute      → probly.quantification.decomposition.AbsoluteErrorDecomposition (Task 6.5)
   │      → emits (H*, A*, E*)
   │
   └─→ experiments/epistemic_eval/scripts/method_*.py
          each adapter constructs the same per-loss decomposition
          on the method's posterior to compute (Â, Ê).
```

Everything downstream — AuReC, Pareto-gap, the Theorem 1 selector,
risk-coverage curves — is **loss-agnostic**. It consumes scalar
`(A_hat, E_hat)` arrays and per-point regret without knowing which
loss produced them.

## Datasets

| Dataset       | Task            | `p*(y \| x)` from              | Loss(es) reported       | Backbone strategy         |
|---------------|-----------------|--------------------------------|-------------------------|---------------------------|
| CIFAR-10H     | classification  | 50+ human votes per image       | cross-entropy           | train-from-scratch or probly checkpoint |
| ImageNet-ReaL | classification  | multi-label ReaL annotations    | cross-entropy           | probly checkpoint if compatible |
| APPA-REAL     | age (integer ages, treated as real-valued) | per-image integer votes (sparse over the observed integer support) | **squared error AND absolute error, both as primary results** | **pretrained backbone + linear probe** |

The APPA-REAL backbone is provided externally by the user and lives
at `experiments/epistemic_eval/assets/appa_real_backbone/` (weights
gitignored, `backbone_meta.yaml` committed). The head is a shallow
MLP `Linear(in_features, hidden) → ReLU → Dropout(p) → Linear(hidden, K)`
with `hidden=256`, `p=0.1` defaults per `decisions.md`.

## Methodological decisions

`decisions.md` is the source of truth for locked choices (loss
functions, discretization, p* construction, H* computation per
dataset, head architecture, plot policy). When in doubt, read
`decisions.md`. If `decisions.md` does not specify something, **ask
the user** — do not guess.

## Coding conventions (additional to probly's above)

- All scripts take a YAML config and a `--seed`. No hardcoded paths
  or hyperparameters in `experiments/epistemic_eval/` or library
  code added for this paper.
- All run outputs go to
  `experiments/epistemic_eval/runs/<YYYYMMDD>_<exp>_<method>_<dataset>_seed<N>/`.
- The full resolved config is copied into the run dir at start of run.
- Fail loudly on missing config keys. No silent defaults. Use a
  schema (e.g. pydantic) and raise on unknown or missing fields.
- No `try/except` blocks that swallow errors. If something is wrong,
  the pipeline must crash with a clear message.
- All randomness goes through a single seeded `numpy.random.Generator`
  and `torch.Generator` per process. No bare `np.random.seed`.
- File I/O uses `pathlib.Path`, not string concatenation.
- Type hints on all public functions.
- One run = one `(method, dataset, seed)` tuple. Don't bundle.
- **Loss-parametric throughout.** Never hardcode a loss in `src/` or
  `experiments/epistemic_eval/scripts/`. The loss flows from config
  → oracle dispatch → decomposition dispatch.

## Caching contract

Cached arrays in `experiments/epistemic_eval/runs/<run_id>/`:

- `config.yaml` — resolved config
- `predictions.npz` — keys: `logits`, `A_hat`, `E_hat`, `indices`,
  plus method-specific posterior samples or sufficient statistics
  (documented schema per Task 5)
- `oracle.npz` (oracle layer only) — keys: `A_star`, `E_star`,
  `H_star`, `p_star`
- `metrics.json` (evaluation layer only) — AuReC, Pareto-gap, etc.
- `meta.json` — git commit, probly commit, timestamp, hostname

If a cached file exists with a matching config hash, reuse it unless
`--force-recompute` is passed.

## Plot policy

**Metrics-only auto.** Every run automatically writes `metrics.json`.
`aggregate_results.py` regenerates `results_table.{md,csv}` from
cached metrics. Figures live behind explicit
`experiments/epistemic_eval/scripts/make_figure_<N>.py` scripts that
read cached metrics and emit `.pdf`/`.png`. **No automatic figure
regeneration.**

## Verification

- `pytest tests/probly/evaluation/` covers the AuReC / selector /
  Pareto-gap library work (Task 3) with closed-form 5-point
  hand-computed examples.
- `pytest tests/probly/quantification/decomposition/` covers the
  squared- and absolute-error decompositions (Task 6.5), including
  additivity (squared) and non-additivity (absolute) tests.
- `pytest tests/` (the full probly suite) must always stay green —
  library extensions don't break existing tests.
- `uv run prek run --all-files` must pass before committing any
  library change.

## Out of scope (do not work on these unless asked)

- Training new ImageNet backbones from scratch.
- New UQ methods beyond what probly provides (the four frozen for
  the paper are `mcd`, `ensemble`, `llla`, `evidential`).
- Datasets beyond CIFAR-10H, ImageNet-ReaL, APPA-REAL.
- The proofs in the appendix (the user's job).
- Cluster-specific SLURM details (partition names, account codes,
  mounts).
- `experiments/first_order_data/` — that's another contributor's
  branch work; gated by `.claude/settings.json` deny rule.

## Workflow

1. Read this file, `decisions.md`, and the relevant section of
   `NeurIPS2026_paper_draft.pdf` / `paper_tex/sections/*.tex` before
   starting any task.
2. For any non-trivial task, write a plan first and have the user
   approve it before executing.
3. Implement, then run `pytest tests/`. If editing library code,
   also run `uv run prek run --all-files`. If tests don't exist for
   the thing you just changed, write them.
4. Cache outputs, write a one-paragraph summary of what was run and
   the key numbers, and stop. Do not chain into the next task.

## Push policy

Pushes are only allowed to repositories owned by
`github.com/paplhjak/*`. This is enforced by a pre-push hook
committed to the repo at `scripts/git-hooks/pre-push`.

After cloning this branch, install the hook locally:

```bash
ln -sf ../../scripts/git-hooks/pre-push .git/hooks/pre-push
```

If `ln -sf` fails (e.g. on filesystems that restrict symlinks in
`.git/hooks/`), copy instead:

```bash
cp scripts/git-hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

The hook then needs to be re-copied if
`scripts/git-hooks/pre-push` is updated.

Claude must **never** use `--no-verify` to bypass the hook;
`.claude/settings.json` denies the `--no-verify` variants of `git
push`.
