# SLURM scaffolding for the production DCIC grid

`run_grid_dcic.sh` is a SLURM array job that runs the locked
four-method × nine-dataset × five-seed grid on the RCI `h200`
partition. All 180 tasks share no checkpoints; each method retrains
per (dataset, seed) cell at 200 epochs (matching the locked AdamW /
SGD recipes in `decisions.md` -> "DCIC datasets").

The DCIC grid differs from `run_grid.sh` (CIFAR-10H) in two ways:

- **Per-seed test fold** — each seed runs against a different DCIC
  fold (per `dcic.pick_test_fold`), so `prepare_dcic_p_star` and
  `compute_oracle` produce per-seed artifacts. The script inlines
  both as on-demand prereqs gated on existence checks; cache-hash
  matching (the recently-fixed `_p_star_fingerprint` in
  `compute_oracle.py`) makes concurrent re-runs from sibling tasks
  for the same `(dataset, seed)` correct, just slightly wasteful.
- **No shared oracle** — CIFAR-10H reads a single seed=0 oracle for
  all twelve grid tasks. DCIC reads `_main_oracle_<dataset>_seed<N>`
  per task; the oracle directory carries the seed.

## Prerequisites (one-time, on the cluster)

```bash
cd ~/neurips2026/probly

# (1) Install the 9 DCIC archives from Zenodo records/7180818 into data/.
.venv/bin/python experiments/first_order_data/install_dcic_datasets.py \
    --dest data
```

The `prepare_dcic_p_star` and `compute_oracle` prereqs run inline
inside the array job (idempotent), so no additional one-time setup
is required after `install_dcic_datasets.py` succeeds. If you'd
rather build all 45 `(dataset, seed)` p_star + oracle pairs
up-front to keep the array job pure-GPU work:

```bash
DATASETS=(benthic mice_bone pig plankton quality_mri \
          dcic_synthetic treeversity_1 treeversity_6 turkey)
for D in "${DATASETS[@]}"; do
    for SEED in 0 1 2 3 4; do
        .venv/bin/python -m experiments.epistemic_eval.scripts.prepare_dcic_p_star \
            --dataset-config "experiments/epistemic_eval/configs/datasets/${D}.yaml" \
            --seed "${SEED}"
    done
done
```

## Launch

```bash
cd ~/neurips2026/probly
git pull
mkdir -p logs
sbatch experiments/epistemic_eval/cluster/slurm/run_grid_dcic.sh
```

`sbatch` returns a job ID; the 180 array tasks dispatch
independently. The cluster's queue policy may cap concurrent jobs,
so expect a long-tail drain.

## Monitor

```bash
squeue -u $USER
tail -f logs/neurips2026_dcic_<JOBID>-*.log         # all 180
tail -f logs/neurips2026_dcic_<JOBID>-3.log         # specific task
```

## Task-id mapping

```
METHOD_IDX  = TASK_ID / 45
DATASET_IDX = (TASK_ID % 45) / 5
SEED        = TASK_ID % 5

METHODS  = (mc_dropout, evidential, ddu, ensemble)
DATASETS = (benthic, mice_bone, pig, plankton, quality_mri,
             dcic_synthetic, treeversity_1, treeversity_6, turkey)
```

| TASK_ID range | METHOD       |
|---------------|--------------|
|   0..44       | mc_dropout   |
|  45..89       | evidential   |
|  90..134      | ddu          |
| 135..179      | ensemble     |

Within each method's 45-task block, the 9 datasets each get 5
contiguous seeds.

## Per-task wall-time expectations (h200, 1 GPU)

DCIC datasets are smaller than CIFAR-10 (a few thousand to ~15k
images each), and the recipe is 200 epochs of fine-tuning a
ResNet-18 from ImageNet weights — much cheaper than CIFAR-10H's
from-scratch ResNet-18.

| Method      | Wall time per task | Notes |
|-------------|--------------------|-------|
| mc_dropout  | ~30 min – 2 h      | basecls fine-tune (200ep AdamW) + 20-pass extract. The 2h upper bound is on Plankton / Treeversity#6 (largest train sets). |
| evidential  | ~30 min – 2 h      | One classifier fine-tune at 200 epochs SGD-cosine. |
| ddu         | ~30 min – 2 h      | Phase A classifier + Phase B GMM density-head fit on encoder features. |
| ensemble    | ~2 h – 8 h         | 5 members × ~25-90 min/member of fine-tuning. |

Total compute (sum of wall-times): **~250 GPU-hours upper bound**.
With the grid running concurrently subject to queue limits, real
wall-clock is bounded by the longest task. QualityMRI (310 images)
is the cheapest cell; Plankton / Treeversity#6 are the most
expensive. A subset run (e.g. quality_mri + plankton only, 40
tasks) is a sensible smoke before launching the full 180.

## Re-running a single failed task

```bash
# Task 67 (evidential, quality_mri, seed 2) failed mid-extract:
sbatch --array=67 experiments/epistemic_eval/cluster/slurm/run_grid_dcic.sh
```

Safe because every pipeline stage is idempotent:

* `prepare_dcic_p_star.py`: skips on `_matches_existing` value-match.
* `compute_oracle.py`: per-loss sidecar hash includes the
  `p_star_fingerprint`, so a fresh `p_star_seed<N>.npz` invalidates
  cleanly.
* `train_classifier.py`: skips if `classifier.config_hash` matches.
* `fit_uncertainty.py`: writes `method.pth` + `config.yaml`; a re-run
  overwrites in place.
* `extract_uncertainties.py`: writes `predictions.npz`; a re-run
  overwrites.
* `compute_decomposition.py`: per-loss sidecar hash includes the
  `predictions_fingerprint` (bug-3 fix from earlier in the project).
* `compute_metrics.py`: per-loss sidecar hash includes the
  `predictions_fingerprint`.

The basecls existence check is a glob (`*_main_basecls_<dataset>_seed${SEED}`),
not a path equality, so a basecls trained on a different date in a
prior run is picked up automatically.

## Subsetting the grid for a smoke run

To launch only one dataset (e.g. plankton, all 4 methods × 5 seeds
= 20 tasks):

```bash
# plankton's index in DATASETS is 3, so the per-method offsets are:
#   mc_dropout : 3*5 .. 3*5+4   = 15..19
#   evidential : 45+15..45+19   = 60..64
#   ddu        : 90+15..90+19   = 105..109
#   ensemble   : 135+15..135+19 = 150..154
sbatch --array=15-19,60-64,105-109,150-154 \
    experiments/epistemic_eval/cluster/slurm/run_grid_dcic.sh
```

Or to launch a single (method, dataset) cell (e.g. ddu × plankton,
all 5 seeds):

```bash
sbatch --array=105-109 experiments/epistemic_eval/cluster/slurm/run_grid_dcic.sh
```

## After the grid finishes

```bash
ssh paplhjak@login.rci.cvut.cz
cd ~/neurips2026/probly
.venv/bin/python -m experiments.epistemic_eval.scripts.aggregate_results
cat experiments/epistemic_eval/results/results_table.md
```

The aggregator walks every `runs/*/metrics_*.json` and emits a
single CSV + Markdown table grouped by `(dataset, loss)`. CIFAR-10H
and DCIC results coexist in the same table.
