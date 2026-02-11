#!/usr/bin/env bash
set -euo pipefail

# Example: meta-train MUSTANG, then fine-tune on one target dataset.
# Usage:
#   bash scripts/examples/run_mustang_then_finetune.sh [target_dataset]
# Example:
#   bash scripts/examples/run_mustang_then_finetune.sh ssc

TARGET_DATASET="${1:-ssc}"
CONFIG_NAME="base.yaml"
DEVICE="cuda:0"
SEED=1
SEQ_LEN=16
TRAIN_MISSING="point"
TEST_MISSING="point"
TEST_MISSING_RATIO=0.2
FINETUNE_EPOCHS=100
META_OUTER_STEPS=500
VAL_LEN=0.3
TEST_LEN=0.0
ADJ_PATH="original_data/Nitrate/flow_direction_12stations.csv"

# Set to 1 to enable graph-aware model, 0 for non-graph model.
USE_GRAPH="${USE_GRAPH:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
META_SAVE_DIR="./save/meta_examples/meta_to_${TARGET_DATASET}_${TIMESTAMP}"
FT_SAVE_DIR="./save/finetune_examples/${TARGET_DATASET}_${TIMESTAMP}"
mkdir -p "${META_SAVE_DIR}" "${FT_SAVE_DIR}"

GRAPH_FLAG=()
if [[ "${USE_GRAPH}" == "1" ]]; then
  GRAPH_FLAG=(--use_graph --adj_path "${ADJ_PATH}")
fi

echo "[1/2] Meta-training MUSTANG (held-out=${TARGET_DATASET})"
python train_mustang.py \
  --config "${CONFIG_NAME}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --held_out "${TARGET_DATASET}" \
  --sequence_length "${SEQ_LEN}" \
  --training_missing "${TRAIN_MISSING}" \
  --val_len "${VAL_LEN}" \
  --test_len "${TEST_LEN}" \
  --num_outer_steps "${META_OUTER_STEPS}" \
  --save_folder "${META_SAVE_DIR}" \
  --use_wandb false \
  "${GRAPH_FLAG[@]}"

META_MODEL_PATH="${META_SAVE_DIR}/meta_model.pth"
if [[ ! -f "${META_MODEL_PATH}" ]]; then
  echo "ERROR: meta model not found at ${META_MODEL_PATH}" >&2
  exit 1
fi

echo "[2/2] Fine-tuning on target dataset (${TARGET_DATASET})"
python main_conditional_diffusion.py \
  --config "${CONFIG_NAME}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --dataset "${TARGET_DATASET}" \
  --sequence_length "${SEQ_LEN}" \
  --training_missing "${TRAIN_MISSING}" \
  --test_missing "${TEST_MISSING}" \
  --testmissingratio "${TEST_MISSING_RATIO}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --pretrained_model "${META_MODEL_PATH}" \
  --use_wandb false \
  "${GRAPH_FLAG[@]}"

echo "Done."
echo "Meta model: ${META_MODEL_PATH}"
echo "Fine-tune outputs: ./save/${TARGET_DATASET}/"
