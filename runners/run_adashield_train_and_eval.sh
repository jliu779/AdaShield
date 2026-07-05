#!/usr/bin/env bash
# AdaShield 完整 pipeline：训练防御提示词 → 测评 12 个 benchmark
#
# 用法:
#   bash runners/run_adashield_train_and_eval.sh
#   TARGET_MODEL=llava-1.5 DEFENSE_MODEL=llama-3 bash runners/run_adashield_train_and_eval.sh
#   TARGET_MODEL=qwen2.5-vl SKIP_TRAIN=1 bash runners/run_adashield_train_and_eval.sh   # 只测评
#   TARGET_MODEL=glm-4.1v SKIP_EVAL=1 bash runners/run_adashield_train_and_eval.sh      # 只训练
#   TARGET_MODEL=llava-1.5 TRAIN_LIMIT=3 LIMIT=50 bash runners/run_adashield_train_and_eval.sh  # 少训多测
#
# 环境变量:
#   TARGET_MODEL       目标 VLM（默认 llava-1.5）
#   DEFENSE_MODEL      防御 LLM（默认 llama-3）
#   INIT_PROMPT        初始防御提示词路径（默认 prompts/static_defense_prompt.txt）
#   SKIP_TRAIN         设为 1 跳过训练
#   SKIP_EVAL          设为 1 跳过测评
#   SKIP_FIGSTEP       设为 1 跳过 FigStep 训练
#   SKIP_QR            设为 1 跳过 QueryRelated 训练
#   TRAIN_LIMIT        训练时限制样本数（默认空=全部，设数字限制每场景样本数）
#   LIMIT              测评时限制样本数
#   CONTINUE_ON_ERROR  设为 1 遇错继续
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ===== 配置 =====
TARGET_MODEL="${TARGET_MODEL:-llava-1.5}"
DEFENSE_MODEL="${DEFENSE_MODEL:-llama-3}"
INIT_PROMPT="${INIT_PROMPT:-prompts/static_defense_prompt.txt}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_FIGSTEP="${SKIP_FIGSTEP:-0}"
SKIP_QR="${SKIP_QR:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
LIMIT="${LIMIT:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
PROMPT_NAME=$(basename "$INIT_PROMPT")

# 训练输出目录
FIGSTEP_POOL="figstep_wandb/${PROMPT_NAME}/${TARGET_MODEL}"
QR_POOL="wandb/${PROMPT_NAME}/${TARGET_MODEL}"

# ===== 场景列表 =====
FIGSTEP_SCENARIOS=(
    "01-Illegal_Activity" "02-HateSpeech" "03-Malware_Generation"
    "04-Physical_Harm" "05-Fraud" "06-Pornography" "07-Privacy_Violence"
    "08-Legal_Opinion" "09-Financial_Advice" "10-Health_Consultation"
)

QR_SCENARIOS=(
    "01-Illegal_Activitiy" "02-HateSpeech" "03-Malware_Generation"
    "04-Physical_Harm" "05-EconomicHarm" "06-Fraud" "07-Sex"
    "08-Political_Lobbying" "09-Privacy_Violence" "10-Legal_Opinion"
    "11-Financial_Advice" "12-Health_Consultation" "13-Gov_Decision"
)

# ===== 工具函数 =====
log() { echo "[$(date +%T)] $*"; }

run_training() {
    local script="$1"
    local scene="$2"
    local label="$3"
    local skip_flag="${4:-0}"

    if [[ "$skip_flag" == "1" ]]; then
        log "SKIP $label/$scene ($skip_flag=1)"
        return 0
    fi

    local extra_args=()
    [[ -n "$TRAIN_LIMIT" ]] && extra_args+=(--n-iterations 1)

    log "TRAIN $label | $scene | model=$TARGET_MODEL | defender=$DEFENSE_MODEL"
    python "$script" \
        --target-model "$TARGET_MODEL" \
        --defense-model "$DEFENSE_MODEL" \
        --scenario "$scene" \
        --init_defense_prompt_path "$INIT_PROMPT" \
        "${extra_args[@]}"
}

# ===== Phase 1: 训练 =====
if [[ "$SKIP_TRAIN" != "1" ]]; then
    log ""
    log "============================================"
    log "PHASE 1: 训练防御提示词"
    log "  Model:     $TARGET_MODEL"
    log "  Defender:  $DEFENSE_MODEL"
    log "============================================"

    # --- FigStep (10 场景) ---
    if [[ "$SKIP_FIGSTEP" != "1" ]]; then
        log ">>> FigStep 训练 (${#FIGSTEP_SCENARIOS[@]} 场景)"
        for scene in "${FIGSTEP_SCENARIOS[@]}"; do
            if run_training "main_figstep.py" "$scene" "FigStep" 0; then
                log "  OK  FigStep/$scene"
            else
                log "  FAIL FigStep/$scene"
                [[ "$CONTINUE_ON_ERROR" != "1" ]] && exit 1
            fi
        done
    fi

    # --- QueryRelated (13 场景) ---
    if [[ "$SKIP_QR" != "1" ]]; then
        log ">>> QueryRelated 训练 (${#QR_SCENARIOS[@]} 场景)"
        for scene in "${QR_SCENARIOS[@]}"; do
            if run_training "main_qureyrelated.py" "$scene" "QR" 0; then
                log "  OK  QR/$scene"
            else
                log "  FAIL QR/$scene"
                [[ "$CONTINUE_ON_ERROR" != "1" ]] && exit 1
            fi
        done
    fi

    log "PHASE 1 DONE — 防御池位于:"
    log "  FigStep: $FIGSTEP_POOL"
    log "  QR:      $QR_POOL"
else
    log "SKIP Phase 1 (SKIP_TRAIN=1)"
fi

# ===== Phase 2: 测评 =====
if [[ "$SKIP_EVAL" != "1" ]]; then
    log ""
    log "============================================"
    log "PHASE 2: Benchmark 测评"
    log "  Model:       $TARGET_MODEL"
    log "  Defense pool: $FIGSTEP_POOL + $QR_POOL"
    log "============================================"

    # 构建防御池参数（空格分隔）
    DEFENSE_POOLS=""
    [[ -d "$FIGSTEP_POOL" ]] && DEFENSE_POOLS="$FIGSTEP_POOL"
    [[ -d "$QR_POOL" ]] && DEFENSE_POOLS="$DEFENSE_POOLS $QR_POOL"
    DEFENSE_POOLS="${DEFENSE_POOLS## }"  # 去掉前导空格

    if [[ -z "$DEFENSE_POOLS" ]]; then
        log "[WARN] 无防御池，使用 --no-defense 模式"
        TARGET_MODEL="$TARGET_MODEL" \
            NO_DEFENSE=1 \
            LIMIT="${LIMIT:-}" \
            OUT_DIR="$ROOT/results/${TARGET_MODEL}_defended" \
            CONTINUE_ON_ERROR=1 \
            bash "$SCRIPT_DIR/run_adashield_full.sh"
    else
        log "[INFO] 防御池: $DEFENSE_POOLS"
        TARGET_MODEL="$TARGET_MODEL" \
            DEFENSE_POOL="$DEFENSE_POOLS" \
            NO_DEFENSE=0 \
            LIMIT="${LIMIT:-}" \
            OUT_DIR="$ROOT/results/${TARGET_MODEL}_defended" \
            CONTINUE_ON_ERROR=1 \
            bash "$SCRIPT_DIR/run_adashield_full.sh"
    fi

    log "PHASE 2 DONE"
else
    log "SKIP Phase 2 (SKIP_EVAL=1)"
fi

log ""
log "============================================"
log "PIPELINE COMPLETE"
log "  Model:  $TARGET_MODEL"
log "  Train:  $FIGSTEP_POOL  /  $QR_POOL"
log "  Eval:   results/${TARGET_MODEL}_defended/"
log "============================================"
