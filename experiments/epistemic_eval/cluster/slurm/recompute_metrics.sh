#!/bin/bash
# Idempotent recompute of metrics_<loss>.json across every run directory.
#
# Run this on the cluster (or locally) after a math change in
# probly.evaluation.regret_coverage or compute_metrics that bumps
# _METRICS_VERSION / _AUREC_VERSION / _PARETO_GAP_VERSION.
#
# Per-call cost is fast (~1s, no GPU). Calls are independent so we
# parallelise via a Python ProcessPoolExecutor (workers reuse imports;
# much faster than spawning a fresh subprocess per call).
#
# Usage:
#   bash experiments/epistemic_eval/cluster/slurm/recompute_metrics.sh
#
# Tunables (env vars):
#   N_PARALLEL  worker processes (default: $(nproc), capped at 16)
#   PY          python interpreter (default: .venv/bin/python)
#
# This script is a thin wrapper that just resolves PY and exec's the
# Python helper at experiments/epistemic_eval/scripts/recompute_metrics.py.

set -euo pipefail
cd "$(dirname "$0")/../../../.."

PY="${PY:-.venv/bin/python}"

if [[ -z "${N_PARALLEL:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        N_PARALLEL=$(nproc)
    else
        N_PARALLEL=8
    fi
    [[ $N_PARALLEL -gt 16 ]] && N_PARALLEL=16
fi

exec "$PY" -m experiments.epistemic_eval.scripts.recompute_metrics \
    --runs-root experiments/epistemic_eval/runs \
    --workers "$N_PARALLEL"
