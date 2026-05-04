#!/bin/bash
#SBATCH --job-name=neurips2026_dcic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paplhjak@fel.cvut.cz
#SBATCH --mem=32gb
#SBATCH --output=logs/%x_%A-%a.log
#SBATCH --error=logs/%x_%A-%a.log
#SBATCH --partition=gpufast
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --array=0-179
#
# Production DCIC grid: 4 methods x 9 datasets x 5 seeds = 180 array tasks.
# All four methods retrain per (dataset, seed) cell using the locked
# DCIC recipe (AdamW lr=3e-4, cosine, 200 epochs, patience 100;
# from-scratch UQ training uses SGD lr=3e-3 with the same schedule).
#
# Task-id layout:
#     METHOD_IDX  = SLURM_ARRAY_TASK_ID / (9 * 5)        -> 0..3
#     DATASET_IDX = (SLURM_ARRAY_TASK_ID % 45) / 5       -> 0..8
#     SEED        = SLURM_ARRAY_TASK_ID % 5              -> 0..4
#
#     METHODS  = (mc_dropout, evidential, ddu, ensemble)
#     DATASETS = (benthic, mice_bone, pig, plankton, quality_mri,
#                 dcic_synthetic, treeversity_1, treeversity_6, turkey)
# So:
#       0..44   -> mc_dropout x 9 datasets x 5 seeds
#      45..89   -> evidential x 9 datasets x 5 seeds
#      90..134  -> ddu        x 9 datasets x 5 seeds
#     135..179  -> ensemble   x 9 datasets x 5 seeds
#
# Per-task pipeline:
#   (1) prepare_dcic_p_star  -- builds data/<DCICName>/p_star_seed<N>.npz
#                                (idempotent; per-seed because the test fold
#                                 rotates per :func:`dcic.pick_test_fold`).
#   (2) compute_oracle       -- builds the per-seed oracle run dir.
#   (3) (mc_dropout only) basecls-on-demand for this (dataset, seed).
#   (4) fit_uncertainty -> extract_uncertainties -> compute_decomposition
#                       -> compute_metrics x {cross_entropy, zero_one}
#
# Steps (1)-(2) race-safely re-run if multiple tasks share the same
# (dataset, seed) cell; both scripts are hash-cache idempotent. The
# extra wall-clock cost is a few seconds of redundant io per cell.
#
# Re-running a single failed task is safe: every stage is idempotent
# under sidecar-hash matching, and basecls existence is a glob.
# Re-run a single task with: sbatch --array=K run_grid_dcic.sh
#
# See README_dcic.md alongside this script for launch + monitoring.

set -euo pipefail
cd /home/paplhjak/neurips2026/probly

# ---------------------------------------------------------------------
# Per-task ephemeral venv (paplhjak's pattern; matches run_grid.sh).
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
DATASETS=(
    "benthic"
    "mice_bone"
    "pig"
    "plankton"
    "quality_mri"
    "dcic_synthetic"
    "treeversity_1"
    "treeversity_6"
    "turkey"
)
# Canonical DCIC names (case- and #-sensitive); aligned by index with
# DATASETS above. The p_star sidecar lives at
# data/<DCIC_NAMES[i]>/p_star_seed<N>.npz.
DCIC_NAMES=(
    "Benthic"
    "MiceBone"
    "Pig"
    "Plankton"
    "QualityMRI"
    "Synthetic"
    "Treeversity#1"
    "Treeversity#6"
    "Turkey"
)

NUM_DATASETS=${#DATASETS[@]}     # 9
NUM_SEEDS=5
TASKS_PER_METHOD=$(( NUM_DATASETS * NUM_SEEDS ))   # 45

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_METHOD ))
LOCAL_IDX=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_METHOD ))
DATASET_IDX=$(( LOCAL_IDX / NUM_SEEDS ))
SEED=$(( LOCAL_IDX % NUM_SEEDS ))

METHOD="${METHODS[$METHOD_IDX]}"
DATASET_NAME="${DATASETS[$DATASET_IDX]}"
DCIC_NAME="${DCIC_NAMES[$DATASET_IDX]}"

DATASET_CONFIG="experiments/epistemic_eval/configs/datasets/${DATASET_NAME}.yaml"
METHOD_CONFIG="experiments/epistemic_eval/configs/methods/${METHOD}.yaml"

if [[ ! -f "${METHOD_CONFIG}" ]]; then
    echo "ERROR: locked method config not found at ${METHOD_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "ERROR: dataset config not found at ${DATASET_CONFIG}" >&2
    exit 1
fi

PY="python"   # the per-task venv is now active.

TODAY="$(date -u +%Y%m%d)"
RUN_ID="${TODAY}_main_${METHOD}_${DATASET_NAME}_seed${SEED}"
RUN_DIR="experiments/epistemic_eval/runs/${RUN_ID}"

# ---------------------------------------------------------------------
# Prereq (1): per-seed p_star sidecar.
#
# The DCIC test fold rotates per seed (see decisions.md "DCIC datasets"
# -> "Test/train fold convention"), so p_star is a per-seed artifact:
# different seeds have different test images. The prep script is
# idempotent via :func:`prepare_dcic_p_star._matches_existing`.
# ---------------------------------------------------------------------
P_STAR_PATH="data/${DCIC_NAME}/p_star_seed${SEED}.npz"
if [[ ! -f "${P_STAR_PATH}" ]]; then
    echo
    echo "===== prepare_dcic_p_star (dataset=${DATASET_NAME} seed=${SEED}) ====="
    ${PY} -m experiments.epistemic_eval.scripts.prepare_dcic_p_star \
        --dataset-config "${DATASET_CONFIG}" \
        --seed "${SEED}"
fi
if [[ ! -f "${P_STAR_PATH}" ]]; then
    echo "ERROR: p_star sidecar not produced at ${P_STAR_PATH}." >&2
    exit 1
fi

# ---------------------------------------------------------------------
# Prereq (2): per-seed oracle.
#
# CIFAR-10H reads a single seed=0 oracle for every grid task because
# its test set is fixed; for DCIC the oracle is per-seed because the
# test fold rotates. Each (dataset, seed) gets its own oracle run dir;
# all four methods on that (dataset, seed) share it.
#
# compute_oracle.py reads --p-star-path explicitly when the dataset
# layout doesn't follow the cifar10h default. We also pass --seed so
# the oracle run dir name carries the seed.
# ---------------------------------------------------------------------
shopt -s nullglob
ORACLE_CANDIDATES=(experiments/epistemic_eval/runs/*_main_oracle_${DATASET_NAME}_seed${SEED})
shopt -u nullglob
if [[ ${#ORACLE_CANDIDATES[@]} -eq 0 ]]; then
    echo
    echo "===== compute_oracle (dataset=${DATASET_NAME} seed=${SEED}) ====="
    ${PY} -m experiments.epistemic_eval.scripts.compute_oracle \
        --dataset-config "${DATASET_CONFIG}" \
        --seed "${SEED}" \
        --p-star-path "${P_STAR_PATH}"
    shopt -s nullglob
    ORACLE_CANDIDATES=(experiments/epistemic_eval/runs/*_main_oracle_${DATASET_NAME}_seed${SEED})
    shopt -u nullglob
fi
if [[ ${#ORACLE_CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: oracle run dir not produced for ${DATASET_NAME} seed=${SEED}." >&2
    exit 1
fi
ORACLE_RUN="${ORACLE_CANDIDATES[-1]}"

# ---------------------------------------------------------------------
# Prereq (3): basecls-on-demand for the mc_dropout branch.
#
# mc_dropout's full-network flow loads a pretrained classifier and
# applies probly.method.dropout with epochs=0; it does NOT train from
# scratch. So the mc_dropout task needs a basecls run for THIS
# (dataset, seed); train one on the fly if missing.
#
# evidential / ddu / ensemble are in
# fit_uncertainty._FROM_SCRATCH_FULL_NETWORK_METHODS; they retrain
# from scratch inside fit_uncertainty.py and ignore CLASSIFIER_PATH.
# ---------------------------------------------------------------------
shopt -s nullglob
BASECLS_CANDIDATES=(experiments/epistemic_eval/runs/*_main_basecls_${DATASET_NAME}_seed${SEED})
shopt -u nullglob

if [[ "${METHOD}" == "mc_dropout" ]]; then
    if [[ ${#BASECLS_CANDIDATES[@]} -eq 0 ]]; then
        echo
        echo "===== train_classifier (basecls on-demand for ${DATASET_NAME} seed=${SEED}) ====="
        ${PY} -m experiments.epistemic_eval.scripts.train_classifier \
            --config "${DATASET_CONFIG}" \
            --seed "${SEED}"
        shopt -s nullglob
        BASECLS_CANDIDATES=(experiments/epistemic_eval/runs/*_main_basecls_${DATASET_NAME}_seed${SEED})
        shopt -u nullglob
    fi
    if [[ ${#BASECLS_CANDIDATES[@]} -eq 0 ]]; then
        echo "ERROR: basecls did not produce a run dir for ${DATASET_NAME} seed=${SEED}." >&2
        exit 1
    fi
    BASECLS_RUN="${BASECLS_CANDIDATES[-1]}"
    CLASSIFIER_PATH="${BASECLS_RUN}/classifier.pth"
    if [[ ! -f "${CLASSIFIER_PATH}" ]]; then
        echo "ERROR: classifier.pth missing at ${CLASSIFIER_PATH}." >&2
        exit 1
    fi
else
    BASECLS_RUN="(unused; method retrains from scratch)"
    CLASSIFIER_PATH="/dev/null"
fi

echo "============================================================"
echo " run_grid_dcic.sh  (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"
echo "   method:           ${METHOD}"
echo "   dataset:          ${DATASET_NAME}  (DCIC=${DCIC_NAME})"
echo "   seed:             ${SEED}"
echo "   method_config:    ${METHOD_CONFIG}"
echo "   dataset_config:   ${DATASET_CONFIG}"
echo "   p_star:           ${P_STAR_PATH}"
echo "   oracle_run:       ${ORACLE_RUN}"
echo "   basecls_run:      ${BASECLS_RUN}"
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
        --oracle-run "${ORACLE_RUN}" \
        --p-star-path "${P_STAR_PATH}"
done

echo
echo "============================================================"
echo " run_grid_dcic.sh: completed for method=${METHOD} dataset=${DATASET_NAME} seed=${SEED}"
echo "   metrics written under: ${RUN_DIR}"
echo "============================================================"

for LOSS in cross_entropy zero_one; do
    METRICS_JSON="${RUN_DIR}/metrics_${LOSS}.json"
    if [[ -f "${METRICS_JSON}" ]]; then
        echo
        echo "----- ${METRICS_JSON} -----"
        cat "${METRICS_JSON}"
    fi
done
