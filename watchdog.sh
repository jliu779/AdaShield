#!/usr/bin/env bash
# AdaShield 看门狗 — 每30分钟检查一次，挂了就标记
set -euo pipefail
PROJECT="/home/jingliu/workspece/AdaShield"
LOG="$PROJECT/runs/latest/watchdog.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

while true; do
    sleep 1800
    ALIVE=$(ps aux | grep -c '[m]ain_qureyrelated\|[m]aster_run_all_models' || true)
    CKPT=$(wc -l < "$PROJECT/runs/latest/.checkpoint" 2>/dev/null || echo 0)
    if [[ "$ALIVE" -eq 0 ]]; then
        log "DOWN — 进程已停，ckpt=$CKPT，需手动续跑: cd $PROJECT && bash master_run_all_models.sh --resume"
    else
        log "OK   — ckpt=$CKPT"
    fi
done
