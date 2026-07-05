#!/usr/bin/env bash
# ============================================================
# AdaShield 全模型主控脚本 v2 — 带断点续跑
#
# checkpoint 文件: runs/latest/.checkpoint
# 每完成一个 scenario/模型 立即写入，重启自动跳过
#
# 用法:
#   bash master_run_all_models.sh              # 全新开始
#   bash master_run_all_models.sh --resume     # 从断点续跑
# ============================================================
set -euo pipefail

source /home/jingliu/miniconda3/etc/profile.d/conda.sh
conda activate venv_neurostrike

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR" && pwd)"
cd "$ROOT"

# ===== 参数 =====
RESUME="${RESUME:-0}"
[[ "${1:-}" == "--resume" ]] && RESUME=1

# ===== 运行目录 =====
if [[ "$RESUME" == "1" ]] && [[ -L "$ROOT/runs/latest" ]]; then
    RUN_DIR="$(readlink -f "$ROOT/runs/latest")"
    RUN_NAME="$(basename "$RUN_DIR")"
    log ">>> 续跑模式: $RUN_NAME"
else
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$ROOT/runs/$RUN_NAME"
    mkdir -p "$RUN_DIR/models"
    rm -f "$ROOT/runs/latest"
    ln -sf "$RUN_DIR" "$ROOT/runs/latest"
fi

MASTER_LOG="$RUN_DIR/master.log"
CHECKPOINT="$RUN_DIR/.checkpoint"

# ===== 日志 =====
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

# ===== checkpoint: 检查某步是否已完成 =====
is_done() {
    local key="$1"
    grep -qFx "$key" "$CHECKPOINT" 2>/dev/null
}

# ===== checkpoint: 标记某步完成 (原子写入) =====
mark_done() {
    local key="$1"
    echo "$key" >> "$CHECKPOINT"
    log "  ✅ CHECKPOINT: $key"
}

# ===== 场景列表 =====
QR_SCENARIOS=(
    "01-Illegal_Activitiy" "02-HateSpeech" "03-Malware_Generation"
    "04-Physical_Harm" "05-EconomicHarm" "06-Fraud" "07-Sex"
    "08-Political_Lobbying" "09-Privacy_Violence" "10-Legal_Opinion"
    "11-Financial_Advice" "12-Health_Consultation" "13-Gov_Decision"
)

# ===== 模型列表 =====
declare -A MODEL_PATHS
MODEL_PATHS=(
  ["llava-1.5"]="/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"
  ["qwen2.5-vl"]="/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
  ["internvl3.5"]="/hub/huggingface/models/OpenGVLab/InternVL3_5-8B"
  ["internvl3"]="/hub/huggingface/models/OpenGVLab/InternVL3-8B"
  ["qwen3-vl"]="/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
  ["glm-4.1v"]="/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking"
)

MODEL_ORDER=("llava-1.5" "qwen2.5-vl" "internvl3.5" "internvl3" "qwen3-vl" "glm-4.1v")

# ===== 统计 =====
total_models=${#MODEL_ORDER[@]}
total_scenarios=${#QR_SCENARIOS[@]}
START_TIME=$(date +%s)
skipped=0
done_count=0

log "============================================================"
log "AdaShield 全模型 Pipeline v2 (断点续跑)"
log "  Run:       $RUN_NAME"
log "  Models:    $total_models (${MODEL_ORDER[*]})"
log "  Scenarios: $total_scenarios per model"
log "  Checkpoint: $CHECKPOINT"
log "  GPU:       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}"
log "============================================================"

for model in "${MODEL_ORDER[@]}"; do
    MODEL_KEY="MODEL:$model"
    TRAIN_DONE_KEY="$MODEL_KEY|TRAIN_DONE"
    EVAL_DONE_KEY="$MODEL_KEY|EVAL_DONE"

    # ========== Phase 1: 训练 (逐场景 checkpoint) ==========

    if is_done "$TRAIN_DONE_KEY"; then
        log ""
        log "SKIP [$model] 训练 — 全部场景已完成"
        skipped=$((skipped + total_scenarios))
        done_count=$((done_count + 1))
    else
        log ""
        log "############################################################"
        log "##  [$model] 开始训练 (13 场景)"
        log "############################################################"

        trained=0
        for scene in "${QR_SCENARIOS[@]}"; do
            SCENE_KEY="$MODEL_KEY|SCENE:$scene"

            if is_done "$SCENE_KEY"; then
                log "  SKIP $scene (已完成)"
                skipped=$((skipped + 1))
                trained=$((trained + 1))
                continue
            fi

            log "  TRAIN [$model] $scene ($((trained + 1))/13)"

            python main_qureyrelated.py \
                --target-model "$model" \
                --defense-model llama-3 \
                --scenario "$scene" \
                --init_defense_prompt_path prompts/static_defense_prompt.txt 2>&1 | sed 's/^/    /'
            # 到此说明成功（失败会直接退出脚本）

            mark_done "$SCENE_KEY"
            python "$SCRIPT_DIR/extract_results.py" "$model" "$scene" >> "$RUN_DIR/results.jsonl" 2>/dev/null || true
            log "  ✅ [$model] $scene 完成并存档"

            trained=$((trained + 1))
            done_count=$((done_count + 1))
        done

        # 全部场景训练完成 → 标记训练阶段完成
        mark_done "$TRAIN_DONE_KEY"
    fi

    # ========== Phase 2: 评测 ==========

    if is_done "$EVAL_DONE_KEY"; then
        log "SKIP [$model] 评测 — 已完成"
    else
        log ""
        log "##  [$model] 开始评测 (12 Benchmark)"

        EVAL_LOG="$RUN_DIR/models/$model/eval.log"
        mkdir -p "$RUN_DIR/models/$model"

        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
            TARGET_MODEL="$model" \
            SKIP_TRAIN=1 \
            SKIP_FIGSTEP=1 \
            bash runners/run_adashield_train_and_eval.sh 2>&1 | tee "$EVAL_LOG"
        # 到此说明成功

        mark_done "$EVAL_DONE_KEY"
        log "  ✅ [$model] 评测完成"
    fi
done

# ===== 汇总 =====
TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))
H=$(( TOTAL_ELAPSED / 60 ))
M=$(( TOTAL_ELAPSED % 60 ))

{
    echo ""
    echo "============================================================"
    echo "AdaShield Pipeline 完成"
    echo "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  总耗时:   ${H}h${M}m"
    echo "  已完成:   $(wc -l < "$CHECKPOINT") 步"
    echo "  输出:"
    echo "    训练: wandb/ + figstep_wandb/"
    echo "    评测: results/"
    echo "============================================================"
} | tee -a "$MASTER_LOG" "$RUN_DIR/SUMMARY.txt"
