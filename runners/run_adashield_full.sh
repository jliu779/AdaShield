#!/usr/bin/env bash
# AdaShield 单模型全 benchmark 推理 pipeline
#   infer_benchmark.py 生成 → 输出 JSONL
#
# 用法:
#   bash runners/run_adashield_full.sh
#   TARGET_MODEL=qwen2.5-vl bash runners/run_adashield_full.sh
#   TARGET_MODEL=glm-4.1v DEFENSE_POOL="figstep_wandb/cogvlm" bash runners/run_adashield_full.sh
#   TARGET_MODEL=llava-1.5 NO_DEFENSE=1 LIMIT=50 bash runners/run_adashield_full.sh
#
# 环境变量:
#   TARGET_MODEL      目标 VLM 名称（默认 llava-1.5）
#   DEFENSE_POOL      防御池目录路径（与 --no-defense 互斥）
#   NO_DEFENSE        设为 1 跳过防御（baseline 模式，默认 1）
#   RETRIVAL_TYPE     检索方式: sample-wise（默认）| random
#   BENIGN_BENCHMARKS 设为 1 对良性 benchmark 启用相似度门控
#   BETA              良性模式相似度阈值（默认 0.7）
#   LIMIT             限制每个 benchmark 的样本数（用于快速测试）
#   SKIP_SAFETY       跳过安全 benchmark（默认 0）
#   SKIP_BENIGN       跳过良性 benchmark（默认 0）
#   OUT_DIR           输出目录（默认 results）
#   CONTINUE_ON_ERROR 设为 1 则在单个 benchmark 失败后继续
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFER_SCRIPT="$ROOT/infer_benchmark.py"
MANIFEST_DIR="$ROOT/data/manifests"

# ===== 配置 =====
TARGET_MODEL="${TARGET_MODEL:-llava-1.5}"
DEFENSE_POOL="${DEFENSE_POOL:-}"
NO_DEFENSE="${NO_DEFENSE:-1}"
RETRIVAL_TYPE="${RETRIVAL_TYPE:-sample-wise}"
BENIGN_BENCHMARKS="${BENIGN_BENCHMARKS:-0}"
BETA="${BETA:-0.7}"
LIMIT="${LIMIT:-}"
SKIP_SAFETY="${SKIP_SAFETY:-0}"
SKIP_BENIGN="${SKIP_BENIGN:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# ===== Benchmark 清单 =====
# 安全/越狱 benchmark → max_new_tokens
declare -A SAFETY_BENCHMARKS=(
  ["vlsafe_examine_eval"]=256
  ["spa_vl_test_530"]=192
  ["mmsb_vision_risk_sdtypo"]=192
  ["mm_safetybench_300"]=192
  ["siuo_167"]=192
  ["mssbench_unsafe_full"]=192
  ["vlsafe_alignment_anchor_seed42"]=192
)

# 良性/通用能力 benchmark
declare -A BENIGN_BENCHMARK_LIST=(
  ["scienceqa_imgval_full"]=192
  ["mmstar"]=192
  ["mme_realworld"]=192
  ["mathvista"]=256
  ["colorbench"]=192
)

# ===== 工具函数 =====
log() { echo "[$(date +%T)] $*"; }

build_extra_args() {
  local args=()
  if [[ -n "$LIMIT" ]]; then
    args+=(--max-samples "$LIMIT")
  fi
  if [[ "$NO_DEFENSE" == "1" ]]; then
    args+=(--no-defense)
  elif [[ -n "${DEFENSE_POOL:-}" ]]; then
    # 支持多个防御池路径（空格分隔）
    for pool in $DEFENSE_POOL; do
      [[ -n "$pool" ]] && args+=(--defense-pool "$pool")
    done
  fi
  args+=(--retrival-type "$RETRIVAL_TYPE")
  echo "${args[@]}"
}

run_one_benchmark() {
  local name="$1"
  local max_tokens="$2"
  local benign_flag="${3:-0}"
  local manifest="$MANIFEST_DIR/${name}.jsonl"
  local output="$OUT_DIR/${name}_${TARGET_MODEL}.jsonl"

  if [[ ! -f "$manifest" ]]; then
    log "SKIP $name (manifest not found: $manifest)"
    return 0
  fi

  if [[ -f "$output" && -s "$output" ]]; then
    log "SKIP $name (output exists: $output)"
    return 0
  fi

  mkdir -p "$(dirname "$output")"

  local extra_args=()
  read -ra extra_args <<< "$(build_extra_args)"
  if [[ "$benign_flag" == "1" ]]; then
    extra_args+=(--benign --beta "$BETA")
  fi

  log "RUN  $name (target=$TARGET_MODEL, max_tokens=$max_tokens)"
  python "$INFER_SCRIPT" \
    --manifest "$manifest" \
    --target-model "$TARGET_MODEL" \
    --output "$output" \
    "${extra_args[@]}"
}

# ===== 主流程 =====
log "============================================"
log "AdaShield Full Pipeline"
log "  MODEL:       $TARGET_MODEL"
log "  OUT_DIR:     $OUT_DIR"
log "  NO_DEFENSE:  $NO_DEFENSE"
log "  DEFENSE_POOL: ${DEFENSE_POOL:-(none)}"
log "  RETRIVAL:    $RETRIVAL_TYPE"
log "  LIMIT:       ${LIMIT:-(all)}"
log "============================================"

failed=()

# ---- 安全 benchmark ----
if [[ "$SKIP_SAFETY" != "1" ]]; then
  log ">>> Phase 1: 安全/越狱 Benchmark (7个)"
  for name in "${!SAFETY_BENCHMARKS[@]}"; do
    max_tokens="${SAFETY_BENCHMARKS[$name]}"
    if run_one_benchmark "$name" "$max_tokens" 0; then
      log "  OK  $name"
    else
      log "  FAIL $name"
      failed+=("$name")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        log "FATAL: $name 失败，中止（设置 CONTINUE_ON_ERROR=1 可继续）"
        exit 1
      fi
    fi
  done
else
  log "SKIP 安全 benchmark (SKIP_SAFETY=1)"
fi

# ---- 良性 benchmark ----
if [[ "$SKIP_BENIGN" != "1" ]]; then
  log ">>> Phase 2: 良性/通用能力 Benchmark (5个)"
  # 良性 benchmark 自动启用相似度门控
  local benign_flag=0
  if [[ "$BENIGN_BENCHMARKS" == "1" ]]; then benign_flag=1; fi

  for name in "${!BENIGN_BENCHMARK_LIST[@]}"; do
    max_tokens="${BENIGN_BENCHMARK_LIST[$name]}"
    if run_one_benchmark "$name" "$max_tokens" "$benign_flag"; then
      log "  OK  $name"
    else
      log "  FAIL $name"
      failed+=("$name")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        log "FATAL: $name 失败，中止（设置 CONTINUE_ON_ERROR=1 可继续）"
        exit 1
      fi
    fi
  done
else
  log "SKIP 良性 benchmark (SKIP_BENIGN=1)"
fi

log "============================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  log "DONE (有 ${#failed[@]} 个失败): ${failed[*]}"
  exit 1
else
  log "ALL DONE — 全部成功"
fi
log "输出目录: $OUT_DIR"
