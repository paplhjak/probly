#!/usr/bin/env bash
# Run the post-classifier dry-run pipeline for a single UQ method on
# CIFAR-10H, end-to-end:
#
#   fit_uncertainty -> extract_uncertainties -> compute_decomposition
#                   -> compute_metrics (cross_entropy + zero_one)
#
# Reads the pre-trained CIFAR-10H base classifier from the basecls
# run directory and the pre-computed oracle from the oracle run
# directory. Writes a fresh per-method run directory under
# experiments/epistemic_eval/runs/.
#
# Usage:
#     bash run_method_dryrun.sh <method> [<seed>]
#
#   <method>  One of: evidential, ddu, ensemble, mc_dropout.
#   <seed>    Optional integer seed; defaults to 0.
#
# The dry-run configs (one_epoch / two_member variants) are picked
# automatically for {evidential, ddu, ensemble}; mc_dropout uses
# the production config because it does no training (epochs = 0 in
# the load-pretrained branch of fit_uncertainty.py).
#
# This script is for cluster-side manual verification BEFORE the
# SLURM grid is launched. It does NOT submit jobs; the user runs it
# directly on a node (or through srun) after `git pull`. See
# README.md "Dry runs" for context.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: bash $0 <method> [<seed>]" >&2
    echo "  <method> in {evidential, ddu, ensemble, mc_dropout}" >&2
    exit 2
fi

METHOD="$1"
SEED="${2:-0}"

# Project root: this script lives at experiments/epistemic_eval/cluster/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Pick the python interpreter from the project's .venv, falling back
# to PATH. On the cluster, the venv is at ~/neurips2026/probly/.venv;
# prefer it when present so we don't accidentally pick up a different
# torch / CUDA build.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY="${REPO_ROOT}/.venv/bin/python"
else
    PY="$(command -v python)"
fi

# Resolve method config path. {evidential, ddu, ensemble} -> _dryrun
# variant; mc_dropout -> the production config (no-train path).
case "${METHOD}" in
    evidential|ddu|ensemble)
        METHOD_CONFIG="experiments/epistemic_eval/configs/methods/${METHOD}_dryrun.yaml"
        ;;
    mc_dropout)
        METHOD_CONFIG="experiments/epistemic_eval/configs/methods/mc_dropout.yaml"
        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'; expected one of {evidential, ddu, ensemble, mc_dropout}." >&2
        exit 2
        ;;
esac

DATASET_CONFIG="experiments/epistemic_eval/configs/datasets/cifar10h.yaml"
DATASET_NAME="cifar10h"

if [[ ! -f "${METHOD_CONFIG}" ]]; then
    echo "ERROR: method config not found at ${METHOD_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "ERROR: dataset config not found at ${DATASET_CONFIG}" >&2
    exit 1
fi

# Derive the per-stage run_id deterministically from the date +
# method + dataset + seed, matching make_run_id() in
# experiments.epistemic_eval.methods._base.
TODAY="$(date -u +%Y%m%d)"
RUN_ID="${TODAY}_main_${METHOD}_${DATASET_NAME}_seed${SEED}"
RUN_DIR="experiments/epistemic_eval/runs/${RUN_ID}"

# The basecls + oracle run_ids: also follow the make_run_id format,
# but we accept any prior date by globbing -- the user may have
# trained the basecls on an earlier day. Pick the most recent match.
shopt -s nullglob
BASECLS_CANDIDATES=(experiments/epistemic_eval/runs/*_main_basecls_${DATASET_NAME}_seed${SEED})
ORACLE_CANDIDATES=(experiments/epistemic_eval/runs/*_main_oracle_${DATASET_NAME}_seed${SEED})
shopt -u nullglob

if [[ ${#BASECLS_CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: no basecls run found at experiments/epistemic_eval/runs/*_main_basecls_${DATASET_NAME}_seed${SEED}." >&2
    echo "       Train the base classifier first (train_classifier.py)." >&2
    exit 1
fi
if [[ ${#ORACLE_CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: no oracle run found at experiments/epistemic_eval/runs/*_main_oracle_${DATASET_NAME}_seed${SEED}." >&2
    echo "       Compute the oracle first (compute_oracle.py)." >&2
    exit 1
fi
# Most-recent by lexical order on the date prefix (sufficient since
# the format is YYYYMMDD).
BASECLS_RUN="${BASECLS_CANDIDATES[-1]}"
ORACLE_RUN="${ORACLE_CANDIDATES[-1]}"

CLASSIFIER_PATH="${BASECLS_RUN}/classifier.pth"
if [[ ! -f "${CLASSIFIER_PATH}" ]]; then
    echo "ERROR: classifier.pth missing at ${CLASSIFIER_PATH}." >&2
    exit 1
fi

echo "============================================================"
echo " run_method_dryrun.sh"
echo "   method:           ${METHOD}"
echo "   seed:             ${SEED}"
echo "   method_config:    ${METHOD_CONFIG}"
echo "   dataset_config:   ${DATASET_CONFIG}"
echo "   basecls_run:      ${BASECLS_RUN}"
echo "   oracle_run:       ${ORACLE_RUN}"
echo "   target run_dir:   ${RUN_DIR}"
echo "============================================================"

echo
echo "===== fit_uncertainty ====="
"${PY}" -m experiments.epistemic_eval.scripts.fit_uncertainty \
    --method-config "${METHOD_CONFIG}" \
    --dataset-config "${DATASET_CONFIG}" \
    --seed "${SEED}" \
    --classifier-state-dict-path "${CLASSIFIER_PATH}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: expected run_dir not found at ${RUN_DIR}; fit_uncertainty.py did not write where derived." >&2
    exit 1
fi

echo
echo "===== extract_uncertainties ====="
"${PY}" -m experiments.epistemic_eval.scripts.extract_uncertainties \
    --run "${RUN_DIR}"

echo
echo "===== compute_decomposition ====="
"${PY}" -m experiments.epistemic_eval.scripts.compute_decomposition \
    --run "${RUN_DIR}"

for LOSS in cross_entropy zero_one; do
    echo
    echo "===== compute_metrics (loss=${LOSS}) ====="
    "${PY}" -m experiments.epistemic_eval.scripts.compute_metrics \
        --run "${RUN_DIR}" \
        --loss "${LOSS}" \
        --oracle-run "${ORACLE_RUN}"
done

echo
echo "============================================================"
echo " run_method_dryrun.sh: completed for method=${METHOD}"
echo "   metrics written under: ${RUN_DIR}"
echo "============================================================"
