#!/usr/bin/env bash
set -euo pipefail

# Hyperparameters (override per run with env vars).
LR="${LR:-2e-4}"
WD="${WD:-0.0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# VESSL resource options.
CLUSTER="${CLUSTER:-snu-eng-dgx}"
PRESET="${PRESET:-a100-1}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.05-py3}"
KNOWN_PRESETS="${KNOWN_PRESETS:-cpu-only,a100-1,a100-2,a100-4,a100-8}"
SKIP_PRESET_CHECK="${SKIP_PRESET_CHECK:-0}"

# VESSL run name (UI). Keep generic; experiment details go to log/ folder only.
RUN_NAME="${RUN_NAME:-test}"
DESCRIPTION="${DESCRIPTION:-LoRA training via scripts/run_train.sh}"

# Pre-flight checks
if ! command -v vessl >/dev/null 2>&1; then
  echo "[ERROR] 'vessl' command not found. Activate your VESSL environment first."
  exit 1
fi

if ! vessl whoami >/dev/null 2>&1; then
  echo "[ERROR] VESSL auth is not ready. Run: vessl configure"
  exit 1
fi

if ! vessl cluster list | grep -q "[[:space:]]${CLUSTER}[[:space:]]"; then
  echo "[ERROR] Cluster '${CLUSTER}' not found in current organization."
  echo "[INFO] Available clusters:"
  vessl cluster list
  exit 1
fi

if [[ "${SKIP_PRESET_CHECK}" != "1" ]]; then
  if [[ ",${KNOWN_PRESETS}," != *",${PRESET},"* ]]; then
    echo "[ERROR] PRESET='${PRESET}' is not in KNOWN_PRESETS='${KNOWN_PRESETS}'."
    echo "[INFO] Either set PRESET to one of KNOWN_PRESETS, or bypass with SKIP_PRESET_CHECK=1."
    exit 1
  fi
fi

TMP_BASE="$(mktemp -t vessl-run)"
TMP_YAML="${TMP_BASE}.yaml"
mv "${TMP_BASE}" "${TMP_YAML}"
trap 'rm -f "${TMP_YAML}"' EXIT
cat > "${TMP_YAML}" <<EOF
name: ${RUN_NAME}
description: ${DESCRIPTION}
image: ${IMAGE}

resources:
  cluster: ${CLUSTER}
  preset: ${PRESET}

run:
  - workdir: /input
    command: |
      LR=${LR} WD=${WD} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} LORA_R=${LORA_R} LORA_ALPHA=${LORA_ALPHA} bash scripts/run_train.sh
EOF

LR_TAG="${LR//./p}"
WD_TAG="${WD//./p}"
LOG_DIR="log/lr_${LR_TAG}_wd_${WD_TAG}_ep_${EPOCHS}_bs_${BATCH_SIZE}_ga_${GRAD_ACCUM_STEPS}_loraR_${LORA_R}_loraA_${LORA_ALPHA}"

echo "[INFO] submitting run: ${RUN_NAME}"
echo "[INFO] cluster=${CLUSTER}, preset=${PRESET}"
echo "[INFO] log output (in container): ${LOG_DIR} (+ timestamp suffix from train.py)"
echo "[INFO] yaml=${TMP_YAML}"

vessl run create -f "${TMP_YAML}"
