#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Starting training at $(date)"
echo "[INFO] Workdir: $(pwd)"

# Install project dependencies inside the run container.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Default: single-GPU debug run.
# Multi-GPU example: ACCEL_NUM_PROCESSES=8 bash scripts/run_train.sh
ACCEL_NUM_PROCESSES="${ACCEL_NUM_PROCESSES:-1}"
LR="${LR:-2e-4}"
WD="${WD:-0.0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# Accounts to train on. "all" = every instagram-* folder under data_dir.
# Single-account timing test example: INSTAGRAM_IDS=5082hjb
INSTAGRAM_IDS="${INSTAGRAM_IDS:-all}"

# Keep output path filename-safe.
LR_TAG="${LR//./p}"
WD_TAG="${WD//./p}"
OUTPUT_DIR="log/lr_${LR_TAG}_wd_${WD_TAG}_ep_${EPOCHS}_bs_${BATCH_SIZE}_ga_${GRAD_ACCUM_STEPS}_loraR_${LORA_R}_loraA_${LORA_ALPHA}"
mkdir -p log

echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] instagram_ids=${INSTAGRAM_IDS}"

accelerate launch --num_processes "${ACCEL_NUM_PROCESSES}" train.py \
  --data_dir ./data \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WD}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --instagram_ids ${INSTAGRAM_IDS} \
  --logging_steps 1

echo "[INFO] Training finished at $(date)"
