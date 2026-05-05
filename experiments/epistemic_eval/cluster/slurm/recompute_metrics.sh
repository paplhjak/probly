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

total_runs=${#ALL_RUNS[@]}
echo "found $total_runs method run directories."
echo "(progress prints one line per run; '.' per loss, 'F' on failure, 'S' on skip)"
echo

n_done=0
n_skipped=0
n_failed=0
i=0
declare -a FAILED_LOG=()

for run_dir in "${ALL_RUNS[@]}"; do
    i=$((i + 1))
    printf "[%3d/%d] %s " "$i" "$total_runs" "$(basename "$run_dir")"
    if [[ ! -f "$run_dir/config.yaml" ]]; then
        printf "S (no config.yaml)\n"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Pull dataset name and seed from the resolved config to find the
    # matching oracle run. DCIC's test set varies per seed (different
    # seeds yield different subsets after annotator filtering), so
    # we pick the same-seed oracle when one exists; first-order
    # datasets like CIFAR-10H share a single seed=0 oracle and we
    # fall back to it when a per-seed oracle is absent.
    read -r dataset_name run_seed < <(
        $PY - "$run_dir/config.yaml" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
ds = (cfg.get("dataset") or {}).get("name", "")
sd = cfg.get("seed", "")
print(f"{ds} {sd}")
PYEOF
    )
    if [[ -z "$dataset_name" || -z "$run_seed" ]]; then
        printf "S (config missing dataset.name or seed)\n"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    shopt -s nullglob
    ORACLE_MATCHING=("$RUNS_ROOT"/*_main_oracle_${dataset_name}_seed${run_seed})
    ORACLE_FALLBACK=("$RUNS_ROOT"/*_main_oracle_${dataset_name}_seed0)
    shopt -u nullglob
    if [[ ${#ORACLE_MATCHING[@]} -gt 0 ]]; then
        oracle_run="${ORACLE_MATCHING[-1]}"
    elif [[ ${#ORACLE_FALLBACK[@]} -gt 0 ]]; then
        oracle_run="${ORACLE_FALLBACK[-1]}"
    else
        printf "S (no oracle run for dataset=%s)\n" "$dataset_name"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Recompute every loss for which a decomposition exists. Each
    # dataset declares its own supported losses, so the decomposition
    # files present in the run dir are the source of truth.
    shopt -s nullglob
    DECOMPS=("$run_dir"/decomposition_*.npz)
    shopt -u nullglob
    if [[ ${#DECOMPS[@]} -eq 0 ]]; then
        # Upstream pipeline didn't reach decomposition; not a failure.
        printf "S (no decompositions yet)\n"
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
            --force-recompute >/dev/null 2>&1; then
            printf "."
            n_done=$((n_done + 1))
        else
            printf "F(%s)" "$loss"
            FAILED_LOG+=("$(basename "$run_dir") loss=$loss")
            n_failed=$((n_failed + 1))
        fi
    done
    printf "\n"
done

echo
echo "============================================================"
echo " recompute_metrics.sh complete"
echo "   metrics regenerated: $n_done"
echo "   runs skipped:        $n_skipped"
echo "   failures:            $n_failed"
echo "============================================================"

if [[ $n_failed -gt 0 ]]; then
    echo
    echo "Failed (loss):"
    for f in "${FAILED_LOG[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
