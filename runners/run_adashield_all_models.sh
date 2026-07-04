#!/usr/bin/env bash
# AdaShield 全模型 × 全 benchmark 统一评估
#   6 模型 × 12 benchmark = 72 组推理
#
# 用法:
#   bash runners/run_adashield_all_models.sh
#
#   # 只跑单个模型
#   TARGET_MODEL=qwen2.5-vl bash runners/run_adashield_all_models.sh
#
#   # 限制样本数快速验证
#   LIMIT=10 bash runners/run_adashield_all_models.sh
#
#   # 使用防御池
#   DEFENSE_POOL="figstep_wandb/cogvlm" bash runners/run_adashield_all_models.sh
#
#   # 单模型 + 防御 + 快速测试
#   TARGET_MODEL=glm-4.1v LIMIT=20 NO_DEFENSE=0 DEFENSE_POOL="wandb/llava" bash runners/run_adashield_all_models.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_adashield_full.sh"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

log() { echo "[$(date +%T)] $*"; }

# ===== 6 个目标 MLLM =====
declare -a JOBS=(
  "llava-1.5:/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"
  "qwen2.5-vl:/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
  "internvl3.5:/hub/huggingface/models/OpenGVLab/InternVL3_5-8B"
  "internvl3:/hub/huggingface/models/OpenGVLab/InternVL3-8B"
  "qwen3-vl:/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
  "glm-4.1v:/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking"
)

run_one_model() {
  local model="$1"
  local model_path="$2"

  log ""
  log "############################################################"
  log "##  MODEL: $model"
  log "##  PATH:  $model_path"
  log "############################################################"

  TARGET_MODEL="$model" \
    MODEL_PATH="$model_path" \
    DEFENSE_POOL="${DEFENSE_POOL:-}" \
    NO_DEFENSE="${NO_DEFENSE:-1}" \
    RETRIVAL_TYPE="${RETRIVAL_TYPE:-sample-wise}" \
    BENIGN_BENCHMARKS="${BENIGN_BENCHMARKS:-0}" \
    BETA="${BETA:-0.7}" \
    LIMIT="${LIMIT:-}" \
    SKIP_SAFETY="${SKIP_SAFETY:-0}" \
    SKIP_BENIGN="${SKIP_BENIGN:-0}" \
    OUT_DIR="${OUT_DIR:-$ROOT/results}" \
    CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}" \
    bash "$RUNNER"

  log "##  DONE  $model"
}

# ---- 单模型模式 ----
if [[ -n "${TARGET_MODEL:-}" ]]; then
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    path="${job#*:}"
    if [[ "$name" == "$TARGET_MODEL" ]]; then
      run_one_model "$name" "$path"
      exit 0
    fi
  done
  echo "ERROR: unknown model '$TARGET_MODEL'" >&2
  echo "Available: ${JOBS[*]}" >&2
  exit 1
fi

# ---- 全模型模式 ----
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/results}"

log "============================================"
log "AdaShield ALL MODELS Pipeline"
log "  Models:      ${JOBS[*]}"
log "  NO_DEFENSE:  ${NO_DEFENSE:-1}"
log "  DEFENSE_POOL: ${DEFENSE_POOL:-(none)}"
log "  LIMIT:       ${LIMIT:-(all)}"
log "  OUT_DIR:     $OUT_DIR"
log "============================================"

total=${#JOBS[@]}
current=0
failed_models=()

for job in "${JOBS[@]}"; do
  current=$((current + 1))
  name="${job%%:*}"
  path="${job#*:}"

  log ">>> [$current/$total] $name"

  if run_one_model "$name" "$path"; then
    log "<<< [$current/$total] $name OK"
  else
    log "<<< [$current/$total] $name FAILED"
    failed_models+=("$name")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      log "FATAL: 中止（设置 CONTINUE_ON_ERROR=1 可继续）"
      exit 1
    fi
  fi
done

log ""
log "============================================"
if [[ ${#failed_models[@]} -gt 0 ]]; then
  log "ALL DONE — 失败 ${#failed_models[@]}/$total: ${failed_models[*]}"
  exit 1
else
  log "ALL $total MODELS PASSED"
fi
log "输出目录: $OUT_DIR"
