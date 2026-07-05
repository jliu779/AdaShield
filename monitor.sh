#!/usr/bin/env bash
# ============================================================
# AdaShield 远程监控脚本 — 每 30 分钟检查一次运行状态
# ============================================================
REMOTE="zju_5880_haichao_direct"
PROJECT_DIR="/home/jingliu/workspece/AdaShield"
RUN_DIR="$PROJECT_DIR/runs/latest"

echo "============================================================"
echo "AdaShield 监控 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 1. 检查进程是否存活
RUNNING=$(ssh "$REMOTE" "ps aux | grep -E 'main_qureyrelated|master_run_all_models' | grep -v grep | wc -l" 2>/dev/null || echo "0")
RUNNING=$(echo "$RUNNING" | tr -d ' ')

if [[ "$RUNNING" -gt 0 ]]; then
    echo "✅ 进程存活 ($RUNNING 个训练进程运行中)"
else
    echo "❌ 进程已停止！"
fi

# 2. 当前场景
echo ""
echo "--- 当前进度 ---"
ssh "$REMOTE" "cat $RUN_DIR/.progress 2>/dev/null || echo '(无进度文件)'" 2>/dev/null

# 3. Checkpoint 统计
echo ""
echo "--- Checkpoint ---"
ssh "$REMOTE" "
if [[ -f '$RUN_DIR/.checkpoint' ]]; then
    total=\$(wc -l < '$RUN_DIR/.checkpoint')
    scenes=\$(grep -c 'SCENE:' '$RUN_DIR/.checkpoint' 2>/dev/null || echo 0)
    trains=\$(grep -c 'TRAIN_DONE' '$RUN_DIR/.checkpoint' 2>/dev/null || echo 0)
    evals=\$(grep -c 'EVAL_DONE' '$RUN_DIR/.checkpoint' 2>/dev/null || echo 0)
    echo \"  总步数: \$total | 场景: \$scenes | 训练完成: \$trains | 评测完成: \$evals\"
else
    echo '  (尚无 checkpoint)'
fi" 2>/dev/null

# 4. 最近日志 (最后 8 行)
echo ""
echo "--- 最近日志 ---"
ssh "$REMOTE" "tail -8 $RUN_DIR/master.log 2>/dev/null || echo '(无日志)'" 2>/dev/null

# 5. GPU 状态
echo ""
echo "--- GPU ---"
ssh "$REMOTE" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null" 2>/dev/null

# 6. 结果文件
echo ""
echo "--- 结果存档 ---"
ssh "$REMOTE" "[[ -f '$RUN_DIR/results.jsonl' ]] && wc -l < '$RUN_DIR/results.jsonl' && echo '条结果已存档' || echo '(尚无结果)'" 2>/dev/null

echo ""
echo "============================================================"
