# Cluster scaffolding

Helpers for running the epistemic-eval pipeline on the RCI cluster
(`paplhjak@login.rci.cvut.cz`). The user runs everything here
manually after `git pull` on the cluster -- no automated dispatch
from the local repo.

## Dry runs

Single-cell verification of the post-classifier pipeline on real
CIFAR-10H data, end-to-end, before the SLURM grid is launched. One
short run per method so any pipeline regression surfaces before
production wall time is committed.

### What a dry run verifies

Each invocation runs the full chain:

```
fit_uncertainty -> extract_uncertainties -> compute_decomposition
                -> compute_metrics (cross_entropy + zero_one)
```

against the locked dataset config (`cifar10h.yaml`), the
pre-trained CIFAR-10H base classifier (basecls run dir on the
cluster), and the pre-computed oracle (oracle run dir on the
cluster). Output lands in
`experiments/epistemic_eval/runs/<run_id>/` with the standard
`predictions.npz` / `decomposition_<loss>.npz` /
`metrics_<loss>.json` layout.

The dry-run configs reduce the wall-time knobs (`epochs: 1` for the
methods that train; `n_members: 2` for ensemble) but otherwise
match the production configs key-for-key. A pipeline regression in
the dry run is, by construction, also a regression in the
paper-grid run -- the only difference is how long it takes to
surface.

Expected wall time per method on V100 (training + extract + decomp
+ metrics, CIFAR-10H 10k test, K=10):

  * `evidential`:   ~3 min (1 epoch training + single-pass extract).
  * `ddu`:          ~3 min (1 epoch training + single-pass extract + GMM fit).
  * `ensemble`:     ~3 min (2 members x 1 epoch + 2 forward passes per test point).
  * `mc_dropout`:   ~5 min (no training; dropout-wrap + 20 forward passes per test point).

### How to invoke

On the cluster, after a fresh `git pull`:

```bash
cd ~/neurips2026/probly
bash experiments/epistemic_eval/cluster/run_method_dryrun.sh evidential
bash experiments/epistemic_eval/cluster/run_method_dryrun.sh ddu
bash experiments/epistemic_eval/cluster/run_method_dryrun.sh ensemble
bash experiments/epistemic_eval/cluster/run_method_dryrun.sh mc_dropout
```

Each invocation is independent and writes its own per-method run
dir (named via `make_run_id()`: `<YYYYMMDD>_main_<method>_cifar10h_seed0`).
Pre-existing basecls and oracle runs are picked up automatically
(most-recent date wins among matching glob entries).

The script accepts an optional second positional argument for the
seed override: `bash run_method_dryrun.sh evidential 42`. Default
seed is 0.

### What sane outputs look like

For each method, after the dry run completes there should be a
`metrics_cross_entropy.json` and a `metrics_zero_one.json` in the
run dir. Each contains `aurec`, `aurc`, `pareto_gap`, plus
versioning + provenance fields. A sane dry-run output has:

  * `aurec` finite and in `[0, 5]` (the dry-run-trained models are
    barely-trained; expect aurec values one or two orders of
    magnitude worse than the fully-trained MC-Dropout baseline of
    ~0.10 cross-entropy / ~0.015 zero-one). Single-digit-magnitude
    values that are positive and bounded are the smoke check, not
    the paper-quality numbers.
  * `aurc` finite, no NaN, no Inf.
  * `pareto_gap` finite, non-negative.
  * `_pareto_gap_version` = 2; `_aurec_version` = 1;
    `_metrics_version` = 1 (i.e. matches the locked library
    constants -- a mismatch means a stale wheel got pulled in).
  * `config_hash` is a 16-hex string AND it differs across the four
    methods within the same dataset (each method sees the same
    dataset+loss inputs but a different method config -> hash).

Any NaN, any "ran but produced nothing" (predictions.npz exists
with the wrong shape), or any non-zero exit code from the bash
wrapper is a regression and should block the SLURM-grid launch.

### Production grids vs. dry-run

The dry-run configs at
`experiments/epistemic_eval/configs/methods/*_dryrun.yaml` are
**verification only**. They MUST NOT be used for the paper grid:

  * Single-epoch training is well below the convergence point for
    ResNet18 on CIFAR-10; aurec in the paper table requires the
    full 30+ epoch recipe.
  * 2-member ensembles are too small; production ensembles use 5
    members.

The paper grid uses the un-suffixed configs
(`evidential.yaml`, `ddu.yaml`, `ensemble.yaml`, `mc_dropout.yaml`)
exclusively. A future task adds SLURM templates that pin those
configs in the job body.

### One-time prep: building `p_star.npz`

The CIFAR-10H pipeline depends on `data/cifar10h/p_star.npz`
(row-stochastic conditional `p*(y | x)` derived from
`cifar10h-counts.npy`). It is built once via:

```bash
.venv/bin/python -m experiments.epistemic_eval.scripts.prepare_cifar10h_p_star
```

The script is idempotent (skips if the file already exists with
matching content); pass `--force` to rebuild unconditionally. Run
this once after the first `git pull` on the cluster, before any
dry run.
