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
# Calls are independent, so we parallelise via ``xargs -P``.
#
# Usage:
#   bash experiments/epistemic_eval/cluster/slurm/recompute_metrics.sh
#
# Tunables (env vars):
#   N_PARALLEL  worker processes (default: $(nproc), capped at 16)
#   PY          python interpreter (default: .venv/bin/python)
#
# Skips:
#   - basecls and oracle runs (no metrics_*.json by design)
#   - runs missing config.yaml or decomposition_<loss>.npz (the
#     upstream pipeline hasn't reached that stage yet -- skip
#     without failing)
#
# Oracle resolution:
#   For DCIC the test set varies per seed (annotator-availability
#   filtering produces different per-seed subsets), so we pick the
#   same-seed oracle when one exists; first-order datasets like
#   CIFAR-10H share a single seed=0 oracle and we fall back to it
#   when a per-seed oracle is absent.

set -euo pipefail
cd "$(dirname "$0")/../../../.."

PY="${PY:-.venv/bin/python}"
RUNS_ROOT="experiments/epistemic_eval/runs"

if [[ -z "${N_PARALLEL:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        N_PARALLEL=$(nproc)
    else
        N_PARALLEL=8
    fi
    [[ $N_PARALLEL -gt 16 ]] && N_PARALLEL=16
fi

# ---------------------------------------------------------------------
# Phase 1: enumerate work units (run_dir | loss | oracle_run).
#
# Doing this serially up-front lets us:
#   - count the total so progress can show k/N
#   - skip cleanly when oracle / config / decomposition is missing
#   - keep the parallel worker dead-simple (just call compute_metrics)
# ---------------------------------------------------------------------
shopt -s nullglob
ALL_RUNS=("$RUNS_ROOT"/2026*_main_{mc_dropout,evidential,ddu,ensemble}_*_seed*)
shopt -u nullglob

if [[ ${#ALL_RUNS[@]} -eq 0 ]]; then
    echo "no method runs found under $RUNS_ROOT/" >&2
    exit 1
fi

echo "found ${#ALL_RUNS[@]} method run directories. enumerating work..."

JOBS_FILE=$(mktemp)
SKIP_LOG=$(mktemp)
trap 'rm -f "$JOBS_FILE" "$SKIP_LOG"' EXIT

n_skipped=0

for run_dir in "${ALL_RUNS[@]}"; do
    if [[ ! -f "$run_dir/config.yaml" ]]; then
        echo "$(basename "$run_dir"): no config.yaml" >> "$SKIP_LOG"
        n_skipped=$((n_skipped + 1))
        continue
    fi

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
        echo "$(basename "$run_dir"): config missing dataset.name or seed" >> "$SKIP_LOG"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    shopt -s nullglob
    ORACLE_MATCHING=("$RUNS_ROOT"/*_main_oracle_${dataset_name}_seed${run_seed})
    ORACLE_FALLBACK=("$RUNS_ROOT"/*_main_oracle_${dataset_name}_seed0)
    DECOMPS=("$run_dir"/decomposition_*.npz)
    shopt -u nullglob

    if [[ ${#ORACLE_MATCHING[@]} -gt 0 ]]; then
        oracle_run="${ORACLE_MATCHING[-1]}"
    elif [[ ${#ORACLE_FALLBACK[@]} -gt 0 ]]; then
        oracle_run="${ORACLE_FALLBACK[-1]}"
    else
        echo "$(basename "$run_dir"): no oracle for dataset=$dataset_name" >> "$SKIP_LOG"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    if [[ ${#DECOMPS[@]} -eq 0 ]]; then
        echo "$(basename "$run_dir"): no decompositions yet" >> "$SKIP_LOG"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    for decomp in "${DECOMPS[@]}"; do
        loss=$(basename "$decomp" .npz)
        loss="${loss#decomposition_}"
        # Tab-separated; xargs splits on tabs cleanly.
        printf '%s\t%s\t%s\n' "$run_dir" "$loss" "$oracle_run" >> "$JOBS_FILE"
    done
done

total=$(wc -l < "$JOBS_FILE" | tr -d ' ')
echo "queued $total recompute jobs (skipped $n_skipped runs); running $N_PARALLEL in parallel."
echo

# ---------------------------------------------------------------------
# Phase 2: dispatch. Each xargs worker runs compute_metrics on one
# (run, loss, oracle) triple. Workers print a single line on
# completion; the awk pipe attaches a sequential [k/total] counter.
# ---------------------------------------------------------------------
RESULTS_FILE=$(mktemp)
trap 'rm -f "$JOBS_FILE" "$SKIP_LOG" "$RESULTS_FILE"' EXIT

export PY

# shellcheck disable=SC2016
< "$JOBS_FILE" xargs -P "$N_PARALLEL" -I {} -d '\n' bash -c '
    IFS=$'\t' read -r run_dir loss oracle_run <<< "$0"
    name=$(basename "$run_dir")
    if "$PY" -m experiments.epistemic_eval.scripts.compute_metrics \
        --run "$run_dir" --loss "$loss" \
        --oracle-run "$oracle_run" --force-recompute >/dev/null 2>&1; then
        printf "OK   %s loss=%s\n" "$name" "$loss"
    else
        printf "FAIL %s loss=%s\n" "$name" "$loss"
    fi
' {} \
    | tee "$RESULTS_FILE" \
    | awk -v t="$total" 'BEGIN{n=0} {n++; printf "[%4d/%d] %s\n", n, t, $0; fflush()}'

n_done=$(grep -c '^OK ' "$RESULTS_FILE" || true)
n_failed=$(grep -c '^FAIL ' "$RESULTS_FILE" || true)

echo
echo "============================================================"
echo " recompute_metrics.sh complete"
echo "   metrics regenerated: $n_done"
echo "   runs skipped:        $n_skipped"
echo "   failures:            $n_failed"
echo "============================================================"

if [[ $n_skipped -gt 0 ]]; then
    echo
    echo "Skipped run details:"
    sed 's/^/  - /' "$SKIP_LOG"
fi

if [[ $n_failed -gt 0 ]]; then
    echo
    echo "Failed jobs:"
    grep '^FAIL ' "$RESULTS_FILE" | sed 's/^FAIL /  - /'
    exit 1
fi
