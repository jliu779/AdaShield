#!/usr/bin/env bash
# ============================================================
# AdaShield 进度查看脚本 v2
# 用法: ssh zju_5880_haichao_direct "bash /home/jingliu/workspece/AdaShield/check_progress.sh"
# ============================================================

RUN_DIR="/home/jingliu/workspece/AdaShield/runs/latest"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "=== 没有找到运行记录 ==="
    ls -la /home/jingliu/workspece/AdaShield/runs/ 2>/dev/null
    exit 1
fi

RUN_NAME=$(basename "$(readlink -f "$RUN_DIR")")
CHECKPOINT="$RUN_DIR/.checkpoint"

echo "============================================================"
echo "  AdaShield 运行状态"
echo "  Run:  $RUN_NAME"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# checkpoint 统计
if [[ -f "$CHECKPOINT" ]]; then
    TOTAL_DONE=$(wc -l < "$CHECKPOINT")
    SCENES_DONE=$(grep -c "SCENE:" "$CHECKPOINT" 2>/dev/null || echo 0)
    TRAINS_DONE=$(grep -c "TRAIN_DONE" "$CHECKPOINT" 2>/dev/null || echo 0)
    EVALS_DONE=$(grep -c "EVAL_DONE" "$CHECKPOINT" 2>/dev/null || echo 0)
    echo ""
    echo "--- Checkpoint 进度 ---"
    echo "  已完成: $TOTAL_DONE 步 (场景:$SCENES_DONE | 训练完成:$TRAINS_DONE | 评测完成:$EVALS_DONE)"
    echo ""
    echo "--- 各模型状态 ---"
    for model in llava-1.5 qwen2.5-vl internvl3.5 internvl3 qwen3-vl glm-4.1v; do
        scenes=$(grep -c "MODEL:$model|SCENE:" "$CHECKPOINT" 2>/dev/null || echo 0)
        train_done=$(grep -qFx "MODEL:$model|TRAIN_DONE" "$CHECKPOINT" 2>/dev/null && echo "✅" || echo "🔄")
        eval_done=$(grep -qFx "MODEL:$model|EVAL_DONE" "$CHECKPOINT" 2>/dev/null && echo "✅" || echo "⏳")
        printf "  %-15s 训练:%s(%s/13) 评测:%s\n" "$model" "$train_done" "$scenes" "$eval_done"
    done
else
    echo ""
    echo "--- Checkpoint ---"
    echo "  (尚无 checkpoint，首个场景运行中...)"
fi

# 当前进度行
if [[ -f "$RUN_DIR/.progress" ]]; then
    echo ""
    echo "--- 当前任务 ---"
    cat "$RUN_DIR/.progress"
fi

# 最近日志
echo ""
echo "--- 最近 15 行日志 ---"
tail -15 "$RUN_DIR/master.log" 2>/dev/null

# tmux
echo ""
echo "--- tmux ---"
tmux ls 2>/dev/null || echo "  无 tmux 会话"

echo ""
echo "============================================================"
echo "  操作:"
echo "  tmux attach -t adashield    实时输出"
echo "  tail -f $RUN_DIR/master.log 跟踪日志"
echo "  bash check_progress.sh      刷新状态"
echo "============================================================"
