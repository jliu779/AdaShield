#!/usr/bin/env bash
# ============================================================
# AdaShield 全模型主控脚本
# 6 模型 × (13 QR 训练 + 12 Benchmark 评测)
#
# 用法:
#   bash master_run_all_models.sh
#   START_FROM=3 bash master_run_all_models.sh  # 从第 3 个模型开始
# ============================================================
set -euo pipefail

source /home/jingliu/miniconda3/etc/profile.d/conda.sh
conda activate venv_neurostrike

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR" && pwd)"
cd "$ROOT"

START_FROM="${START_FROM:-1}"

# ===== 6 个模型 =====
declare -a MODELS=(
  "llava-1.5"
  "qwen2.5-vl"
  "internvl3.5"
  "internvl3"
  "qwen3-vl"
  "glm-4.1v"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

total=${#MODELS[@]}
current=0
failed_models=()
passed_models=()

for model in "${MODELS[@]}"; do
  current=$((current + 1))

  # 支持从指定模型开始
  if [[ $current -lt $START_FROM ]]; then
    log "SKIP [$current/$total] $model (START_FROM=$START_FROM)"
    continue
  fi

  log ""
  log "############################################################"
  log "##  [$current/$total] MODEL: $model"
  log "##  START: $(date '+%Y-%m-%d %H:%M:%S')"
  log "############################################################"

  START_TIME=$(date +%s)

  if CUDA_VISIBLE_DEVICES=1 \
     TARGET_MODEL="$model" \
     DEFENSE_MODEL="llama-3" \
     SKIP_FIGSTEP=1 \
     CONTINUE_ON_ERROR=1 \
     bash runners/run_adashield_train_and_eval.sh; then

    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    log "##  [$current/$total] $model PASSED  (耗时 ${ELAPSED} min)"
    passed_models+=("$model")
  else
    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    log "##  [$current/$total] $model FAILED  (耗时 ${ELAPSED} min)"
    failed_models+=("$model")
    # 继续跑下一个，不中断
  fi
done

# ===== 汇总 =====
log ""
log "============================================================"
log "ALL MODELS DONE"
log "  Passed: ${#passed_models[@]}/$total  ${passed_models[*]:-}"
log "  Failed: ${#failed_models[@]}/$total  ${failed_models[*]:-}"
log "  Results:"
log "    Training: wandb/ + figstep_wandb/"
log "    Evaluation: results/"
log "============================================================"

if [[ ${#failed_models[@]} -gt 0 ]]; then
  exit 1
fi
