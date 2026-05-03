#!/bin/bash
#SBATCH --job-name=neurips2026
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paplhjak@fel.cvut.cz
#SBATCH --mem=256gb
#SBATCH --output=logs/%x_%A-%a.log
#SBATCH --error=logs/%x_%A-%a.log
#SBATCH --partition=h200
#SBATCH --cpus-per-task=17
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#
# Production CIFAR-10H grid: 4 methods x 3 seeds = 12 array tasks.
# All four methods retrain per seed (no shared classifiers) using
# the locked production configs.
#
# Task-id layout:
#     METHOD_IDX = SLURM_ARRAY_TASK_ID / 3
#     SEED       = SLURM_ARRAY_TASK_ID % 3
#     METHODS = (mc_dropout, evidential, ddu, ensemble)
# So:
#     0..2  -> mc_dropout, seeds 0..2
#     3..5  -> evidential, seeds 0..2
#     6..8  -> ddu,        seeds 0..2
#     9..11 -> ensemble,   seeds 0..2
#
# Per-task pipeline:
#     (mc_dropout only) basecls-on-demand for this seed, then
#     fit_uncertainty -> extract_uncertainties -> compute_decomposition
#                     -> compute_metrics x {cross_entropy, zero_one}
#
# Re-running a single failed task is safe: every stage is idempotent
# under sidecar-hash matching, and basecls existence is a glob.
# Re-run a single task with: sbatch --array=K run_grid.sh
#
# See README.md alongside this script for launch + monitoring + the
# full task table.

set -euo pipefail
cd /home/paplhjak/neurips2026/probly

# ---------------------------------------------------------------------
# Per-task ephemeral venv (paplhjak's pattern; see also other paplhjak
# SLURM jobs that follow the same template).
# ---------------------------------------------------------------------
VENV_DIR="${SLURM_TMPDIR:-/tmp}/.venv-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
trap 'rm -rf "$VENV_DIR"' EXIT
uv venv --python=3.13 "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active

# ---------------------------------------------------------------------
# Task-id mapping.
# ---------------------------------------------------------------------
METHODS=("mc_dropout" "evidential" "ddu" "ensemble")
METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
METHOD="${METHODS[$METHOD_IDX]}"

DATASET_NAME="cifar10h"
DATASET_CONFIG="experiments/epistemic_eval/configs/datasets/cifar10h.yaml"
METHOD_CONFIG="experiments/epistemic_eval/configs/methods/${METHOD}.yaml"

if [[ ! -f "${METHOD_CONFIG}" ]]; then
    echo "ERROR: locked method config not found at ${METHOD_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "ERROR: dataset config not found at ${DATASET_CONFIG}" >&2
    exit 1
fi

PY="python"  # the per-task venv is now active.

TODAY="$(date -u +%Y%m%d)"
RUN_ID="${TODAY}_main_${METHOD}_${DATASET_NAME}_seed${SEED}"
RUN_DIR="experiments/epistemic_eval/runs/${RUN_ID}"

# ---------------------------------------------------------------------
# Oracle: loss/seed-independent. We always read from the seed=0
# oracle run; any seed would do but seed=0 is conventional and was
# the one already computed during dryrun. The pareto-gap and AuReC
# math depend only on (p_star, support) which the oracle layer
# encodes once.
# ---------------------------------------------------------------------
shopt -s nullglob
ORACLE_CANDIDATES=(experiments/epistemic_eval/runs/*_main_oracle_${DATASET_NAME}_seed0)
shopt -u nullglob
if [[ ${#ORACLE_CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: no oracle run found at experiments/epistemic_eval/runs/*_main_oracle_${DATASET_NAME}_seed0." >&2
    echo "       Run experiments.epistemic_eval.scripts.compute_oracle once before launching the grid." >&2
    exit 1
fi
ORACLE_RUN="${ORACLE_CANDIDATES[-1]}"

# ---------------------------------------------------------------------
# basecls-on-demand for the mc_dropout branch.
#
# mc_dropout's full-network flow (per fit_uncertainty.py:415-416)
# loads a pretrained classifier and applies probly.method.dropout
# with epochs=0; it does NOT train from scratch. So the mc_dropout
# task needs a basecls run for its seed; train one on the fly if
# missing.
#
# evidential / ddu / ensemble are in
# fit_uncertainty._FROM_SCRATCH_FULL_NETWORK_METHODS (file:line
# experiments/epistemic_eval/scripts/fit_uncertainty.py:51-71); they
# train from scratch inside fit_uncertainty.py and do NOT consume
# the basecls classifier.pth. They get a CLASSIFIER_PATH placeholder
# below only because fit_uncertainty.py's argparse signature accepts
# the optional --classifier-state-dict-path; the Path-2 branch in
# main() ignores it.
# ---------------------------------------------------------------------
shopt -s nullglob
BASECLS_CANDIDATES=(experiments/epistemic_eval/runs/*_main_basecls_${DATASET_NAME}_seed${SEED})
shopt -u nullglob

if [[ "${METHOD}" == "mc_dropout" ]]; then
    if [[ ${#BASECLS_CANDIDATES[@]} -eq 0 ]]; then
        echo
        echo "===== train_classifier (basecls on-demand for seed=${SEED}) ====="
        # train_classifier.py auto-derives the run dir as
        # runs/<YYYYMMDD>_main_basecls_cifar10h_seed${SEED}/ via
        # _resolve_dataset_run_id at
        # experiments/epistemic_eval/scripts/train_classifier.py:186-188.
        ${PY} -m experiments.epistemic_eval.scripts.train_classifier \
            --config "${DATASET_CONFIG}" \
            --seed "${SEED}"
        # Re-glob now that the run exists.
        shopt -s nullglob
        BASECLS_CANDIDATES=(experiments/epistemic_eval/runs/*_main_basecls_${DATASET_NAME}_seed${SEED})
        shopt -u nullglob
    fi
    if [[ ${#BASECLS_CANDIDATES[@]} -eq 0 ]]; then
        echo "ERROR: basecls training did not produce a run dir for seed=${SEED}." >&2
        exit 1
    fi
    BASECLS_RUN="${BASECLS_CANDIDATES[-1]}"
    CLASSIFIER_PATH="${BASECLS_RUN}/classifier.pth"
    if [[ ! -f "${CLASSIFIER_PATH}" ]]; then
        echo "ERROR: classifier.pth missing at ${CLASSIFIER_PATH}." >&2
        exit 1
    fi
else
    # The from-scratch methods don't need a basecls; pass /dev/null
    # so fit_uncertainty.py's existence check is harmless (the Path-2
    # branch never opens this path).
    BASECLS_RUN="(unused; method retrains from scratch)"
    CLASSIFIER_PATH="/dev/null"
fi

echo "============================================================"
echo " run_grid.sh  (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"
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
if [[ "${METHOD}" == "mc_dropout" ]]; then
    ${PY} -m experiments.epistemic_eval.scripts.fit_uncertainty \
        --method-config "${METHOD_CONFIG}" \
        --dataset-config "${DATASET_CONFIG}" \
        --seed "${SEED}" \
        --classifier-state-dict-path "${CLASSIFIER_PATH}"
else
    ${PY} -m experiments.epistemic_eval.scripts.fit_uncertainty \
        --method-config "${METHOD_CONFIG}" \
        --dataset-config "${DATASET_CONFIG}" \
        --seed "${SEED}"
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: expected run_dir not found at ${RUN_DIR}; fit_uncertainty.py did not write where derived." >&2
    exit 1
fi

echo
echo "===== extract_uncertainties ====="
${PY} -m experiments.epistemic_eval.scripts.extract_uncertainties \
    --run "${RUN_DIR}"

echo
echo "===== compute_decomposition ====="
${PY} -m experiments.epistemic_eval.scripts.compute_decomposition \
    --run "${RUN_DIR}"

for LOSS in cross_entropy zero_one; do
    echo
    echo "===== compute_metrics (loss=${LOSS}) ====="
    ${PY} -m experiments.epistemic_eval.scripts.compute_metrics \
        --run "${RUN_DIR}" \
        --loss "${LOSS}" \
        --oracle-run "${ORACLE_RUN}"
done

echo
echo "============================================================"
echo " run_grid.sh: completed for method=${METHOD} seed=${SEED}"
echo "   metrics written under: ${RUN_DIR}"
echo "============================================================"

# Cat the metrics JSONs for visibility in the SLURM log; small
# enough to render inline (~16 lines each).
for LOSS in cross_entropy zero_one; do
    METRICS_JSON="${RUN_DIR}/metrics_${LOSS}.json"
    if [[ -f "${METRICS_JSON}" ]]; then
        echo
        echo "----- ${METRICS_JSON} -----"
        cat "${METRICS_JSON}"
    fi
done
