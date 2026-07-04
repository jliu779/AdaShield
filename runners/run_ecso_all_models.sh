#!/usr/bin/env bash
# Run full ECSO pipeline for all configured LLaVA models sequentially.
#
# Usage:
#   bash runners/run_ecso_all_models.sh
#   VLM=llava15 LIMIT=5 bash runners/run_ecso_all_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_ecso_full.sh"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

log() { echo "[$(date +%T)] $*"; }

declare -a JOBS=(
  "llava15:/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"

  "qwen25vl:/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
  "qwen3vl:/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
  "phi4:/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct"
  "glm41v:/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking"
  "internvl3:/hub/huggingface/models/OpenGVLab/InternVL3-8B"
  "internvl35:/hub/huggingface/models/OpenGVLab/InternVL3_5-8B"
)

run_one() {
  local vlm="$1"
  local model_path="$2"
  log "======== START $vlm (ECSO) ========"
  VLM="$vlm" MODEL_PATH="$model_path" bash "$RUNNER"
  log "======== DONE  $vlm (ECSO) ========"
}

if [[ -n "${VLM:-}" ]]; then
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    path="${job#*:}"
    if [[ "$name" == "$VLM" ]]; then
      run_one "$name" "$path"
      exit 0
    fi
  done
  echo "ERROR: unknown VLM='$VLM'" >&2
  exit 1
fi

failed=()
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  path="${job#*:}"
  if ! run_one "$name" "$path"; then
    failed+=("$name")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "ERROR: $name failed; set CONTINUE_ON_ERROR=1 to continue" >&2
      exit 1
    fi
    log "WARN: $name failed, continuing..."
  fi
done

log "ALL ECSO RUNS FINISHED"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed models: ${failed[*]}" >&2
  exit 1
fi
