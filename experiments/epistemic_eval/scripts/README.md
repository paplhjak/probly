# Scripts

Runnable entry points for the three-layer pipeline (see AGENTS.md →
"Architecture: strict three-layer separation").

## Layer 1 — extraction (touches GPUs)

- `extract_uncertainties.py` — given `(method, dataset, seed)` config,
  produces `predictions.npz`.
- `method_registry.py`, `method_{mcd,ensemble,llla,evidential}.py` —
  thin adapters wrapping probly (or `laplace-torch` for `llla`). Each
  exposes `train(...)` and `extract(...) -> ExtractionResult`.
- `linear_probe.py` — frozen-backbone feature extraction for APPA-REAL.

## Layer 2 — oracle (no method)

- `compute_oracle.py` — drives `oracle/from_p_star.py` over a dataset
  config; produces `oracle.npz`.

## Layer 3 — evaluation (fast, re-runnable)

- `compute_aurec.py` — calls `probly.evaluation.regret_coverage.aurec`.
- `compute_pareto_gap.py` — calls
  `probly.evaluation.selectors.sweep_w_tau` and
  `probly.evaluation.pareto_gap.pareto_gap`.
- `aggregate_results.py` — collects `metrics.json` across runs;
  writes `results_table.{md,csv}`.

## Figures (explicit; metrics-only auto)

`make_figure_<N>.py` per paper figure, reading cached metrics.
No automatic figure regeneration per `decisions.md` → "Plot policy".

## Grid launcher

- `launch_grid.py` — fans out `(method × dataset × seed)` over `sbatch`
  calls. Pairs with `slurm/extract.sbatch` and `slurm/evaluate.sbatch`
  (Task 8).
