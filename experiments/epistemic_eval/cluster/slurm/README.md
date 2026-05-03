# SLURM scaffolding for the production CIFAR-10H grid

`run_grid.sh` is a SLURM array job that runs the locked four-method
× three-seed grid on the RCI `h200` partition. All twelve tasks
share no checkpoints; each method retrains per seed at 200 epochs
(matching `cifar10h.yaml`'s training recipe).

## Prerequisites (one-time, on the cluster)

Run these once before the first `sbatch` invocation. They produce
artifacts the array job depends on:

```bash
cd ~/neurips2026/probly

# (1) p_star sidecar from cifar10h-counts.npy.
.venv/bin/python -m experiments.epistemic_eval.scripts.prepare_cifar10h_p_star

# (2) Oracle (loss/seed-independent; the grid reads only the seed=0 oracle).
.venv/bin/python -m experiments.epistemic_eval.scripts.compute_oracle \
    --dataset-config experiments/epistemic_eval/configs/datasets/cifar10h.yaml \
    --seed 0
```

## Launch

```bash
cd ~/neurips2026/probly
git pull
mkdir -p logs
sbatch experiments/epistemic_eval/cluster/slurm/run_grid.sh
```

`sbatch` returns a job ID; the 12 array tasks dispatch independently.

## Monitor

```bash
squeue -u $USER
tail -f logs/neurips2026_<JOBID>-*.log         # all 12
tail -f logs/neurips2026_<JOBID>-3.log         # specific task
```

## Task-id mapping

Task ID maps to `(method_idx = TASK_ID / 3, seed = TASK_ID % 3)`:

| TASK_ID | METHOD       | SEED |
|---------|--------------|------|
|   0     | mc_dropout   |  0   |
|   1     | mc_dropout   |  1   |
|   2     | mc_dropout   |  2   |
|   3     | evidential   |  0   |
|   4     | evidential   |  1   |
|   5     | evidential   |  2   |
|   6     | ddu          |  0   |
|   7     | ddu          |  1   |
|   8     | ddu          |  2   |
|   9     | ensemble     |  0   |
|  10     | ensemble     |  1   |
|  11     | ensemble     |  2   |

The mapping is locked in `run_grid.sh` lines under
`# Task-id mapping.` and pinned by `tests/test_run_grid_mapping.py`.

## Per-task wall-time expectations (h200, 1 GPU)

| Method      | Wall time         | Notes |
|-------------|-------------------|-------|
| mc_dropout  | ~50 min if basecls cached for that seed; ~4 h cold | Cold path runs basecls-on-demand (200 epochs ResNet-18 ≈ 3 h on h200) before the no-train wrap + 20-pass extract. |
| evidential  | ~50 min            | 200-epoch evidential CE training + single-pass extract. |
| ddu         | ~1 h               | 200-epoch SN-ResNet-18 training + GMM density-head fit on 50k features + single-pass extract. |
| ensemble    | ~3.5 h             | 5 members × ~40 min/member of from-scratch CIFAR-10 training. |

Total compute (sum of wall-times): **~14 GPU-hours**. Total real
wall-time on h200 with 12 tasks running in parallel:
**≈ longest task** = the ensembles, so ~3.5 h for the cold cohort
and another ~1 h for stragglers. Allow 4–5 h of clock time for the
whole grid to drain.

## Re-running a single failed task

```bash
# Task 7 (ddu, seed 1) failed mid-extract:
sbatch --array=7 experiments/epistemic_eval/cluster/slurm/run_grid.sh
```

This is **safe** because every pipeline stage is idempotent:

* `train_classifier.py`: skips if `classifier.config_hash` matches.
* `fit_uncertainty.py`: writes `method.pth` + `config.yaml`; a re-run
  overwrites in place.
* `extract_uncertainties.py`: writes `predictions.npz`; a re-run
  overwrites.
* `compute_decomposition.py`: per-loss sidecar hash check.
* `compute_metrics.py`: per-loss sidecar hash check.

The basecls existence check is a glob (`*_main_basecls_cifar10h_seed${SEED}`),
not a path equality, so a basecls trained on a different date in a
prior run is picked up automatically.

## Inspecting intermediate state from local

The cluster's `~/neurips2026/probly/` is mounted read-only at
`cluster_probly/probly/` locally. Read files through the mount as
the run progresses:

```bash
# Watch a single task's log live:
tail -f cluster_probly/probly/logs/neurips2026_<JOBID>-3.log

# Read a finished task's metrics JSON:
cat cluster_probly/probly/experiments/epistemic_eval/runs/20260503_main_evidential_cifar10h_seed0/metrics_cross_entropy.json
```

The mount is read-only; do not attempt to write to `cluster_probly/`.
The grid writes only to `~/neurips2026/probly/...` on the cluster.

## After the grid finishes

```bash
ssh paplhjak@login.rci.cvut.cz
cd ~/neurips2026/probly
.venv/bin/python -m experiments.epistemic_eval.scripts.aggregate_results
cat experiments/epistemic_eval/results/results_table.md
```

The aggregator walks every `runs/*/metrics_*.json` and emits a
single CSV + Markdown table grouped by `(dataset, loss)`.
