#!/bin/bash
# Idempotent recompute of metrics_<loss>.json across every run directory.
#
# Run this on the cluster (or locally) after a math change in
# probly.evaluation.regret_coverage or compute_metrics that bumps
# _METRICS_VERSION / _AUREC_VERSION / _PARETO_GAP_VERSION. Uses
# --force-recompute so cached metrics are regenerated even when only
# the JSON schema changed.
#
# Per-run cost is fast (~1s, no GPU): the heavy work
# (predictions / decompositions / oracle / p_star sidecars) is cached.
#
# Usage:
#   bash experiments/epistemic_eval/cluster/slurm/recompute_metrics.sh
#
# Skips:
#   - basecls and oracle runs (no metrics_*.json by design)
#   - runs missing decomposition_<loss>.npz (the upstream stage hasn't
#     completed for that run yet -- skip without failing)

set -euo pipefail
cd "$(dirname "$0")/../../../.."

PY="${PY:-.venv/bin/python}"
RUNS_ROOT="experiments/epistemic_eval/runs"

# Match runs/2026<YYYYMMDD>_<exp>_<method>_<dataset>_seed<N>/ for every
# UQ method we evaluate. Skip basecls (classifier-only) and oracle
# (no metrics).
shopt -s nullglob
ALL_RUNS=("$RUNS_ROOT"/2026*_main_{mc_dropout,evidential,ddu,ensemble}_*_seed*)
shopt -u nullglob

if [[ ${#ALL_RUNS[@]} -eq 0 ]]; then
    echo "no method runs found under $RUNS_ROOT/" >&2
    exit 1
fi

echo "found ${#ALL_RUNS[@]} method run directories."

n_done=0
n_skipped=0
n_failed=0

for run_dir in "${ALL_RUNS[@]}"; do
    if [[ ! -f "$run_dir/config.yaml" ]]; then
        echo "skip $run_dir (no config.yaml)" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Pull dataset name from the resolved config to find the matching
    # oracle run. Datasets keep one oracle run (seed 0) shared across
    # method seeds; pick it via glob.
    dataset_name=$(
        $PY - "$run_dir/config.yaml" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print((cfg.get("dataset") or {}).get("name", ""))
PYEOF
    )
    if [[ -z "$dataset_name" ]]; then
        echo "skip $run_dir (no dataset.name in config)" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi

    shopt -s nullglob
    ORACLE_CANDIDATES=("$RUNS_ROOT"/*_main_oracle_${dataset_name}_seed0)
    shopt -u nullglob
    if [[ ${#ORACLE_CANDIDATES[@]} -eq 0 ]]; then
        echo "skip $run_dir (no oracle run for dataset=$dataset_name)" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi
    oracle_run="${ORACLE_CANDIDATES[-1]}"

    # Recompute every loss for which a decomposition exists. Each
    # dataset declares its own supported losses, so the decomposition
    # files present in the run dir are the source of truth.
    shopt -s nullglob
    DECOMPS=("$run_dir"/decomposition_*.npz)
    shopt -u nullglob
    if [[ ${#DECOMPS[@]} -eq 0 ]]; then
        # Upstream pipeline didn't reach decomposition; not a failure.
        n_skipped=$((n_skipped + 1))
        continue
    fi

    for decomp in "${DECOMPS[@]}"; do
        # Strip prefix and suffix to get the loss name.
        loss=$(basename "$decomp" .npz)
        loss=${loss#decomposition_}
        if $PY -m experiments.epistemic_eval.scripts.compute_metrics \
            --run "$run_dir" \
            --loss "$loss" \
            --oracle-run "$oracle_run" \
            --force-recompute >/dev/null; then
            n_done=$((n_done + 1))
        else
            echo "FAILED: $run_dir loss=$loss" >&2
            n_failed=$((n_failed + 1))
        fi
    done
done

echo
echo "============================================================"
echo " recompute_metrics.sh complete"
echo "   metrics regenerated: $n_done"
echo "   runs skipped:        $n_skipped"
echo "   failures:            $n_failed"
echo "============================================================"

if [[ $n_failed -gt 0 ]]; then
    exit 1
fi
