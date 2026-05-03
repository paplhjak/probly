# Epistemic Uncertainty Evaluation (NeurIPS 2026)

This directory holds the paper-specific work for the
`epistemic-eval` branch: configs, oracle wrappers, extraction and
evaluation scripts, paper figures, and SLURM templates. Library
extensions to probly itself live under
`src/probly/{evaluation,quantification,datasets,method}/`.

For the project-wide context, conventions, datasets, methods,
loss-parametricity, and push policy, see the "Project: Epistemic
Uncertainty Evaluation (NeurIPS 2026)" section appended to
`AGENTS.md` (linked from `CLAUDE.md`).

For locked methodological choices (loss functions, p* construction,
H*/A*/E* formulae, head architecture, plot policy), see
`decisions.md` at the repo root.

For the ordered task list, see `tasks.md` at the repo root. Code in
this directory is "experiment" — see the "Code organisation" section
of AGENTS.md for the library/experiment split.

## Layout

- `configs/datasets/`   — per-dataset YAML (paths, splits, supported losses).
- `configs/methods/`    — per-method YAML (one of `mcd`, `ensemble`, `llla`, `evidential`).
- `configs/experiments/` — full grids over (method × dataset × seed).
- `oracle/`             — oracle wrappers that compute `(A*, E*, H*)` from `p*` and the loss.
- `scripts/`            — runnable entry points: `extract_uncertainties.py`, `compute_oracle.py`, `compute_aurec.py`, `compute_pareto_gap.py`, `aggregate_results.py`, `make_figure_<N>.py`.
- `tests/`              — paper-specific tests (oracle on real datasets, end-to-end smoke).
- `assets/appa_real_backbone/` — user-provided pretrained backbone (weights gitignored, metadata committed).
- `notebooks/`          — exploration only, not part of pipeline.

Run outputs are cached under `runs/<YYYYMMDD>_<exp>_<method>_<dataset>_seed<N>/` (gitignored) per the caching contract in AGENTS.md.
