#!/usr/bin/env bash
# Run full ECSO pipeline for one VLM on all manifest benchmarks:
#   ECSO generate -> judge/score -> summary markdown
#
# Default: dual-GPU pipeline (DUAL_GPU=1)
#   GPU 0 (CUDA_VISIBLE_DEVICES): VLM ECSO generation
#   GPU 1 (JUDGE_GPU): optional Llama judge (background while next gen proceeds)
#
# Usage:
#   bash runners/run_ecso_full.sh
#   VLM=qwen25vl MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct bash runners/run_ecso_full.sh
#   VLM=phi4    MODEL_PATH=/path/to/Phi-4-multimodal-instruct bash runners/run_ecso_full.sh
#   VLM=glm41v  MODEL_PATH=/path/to/GLM-4.1V-9B-Thinking bash runners/run_ecso_full.sh
#   VLM=internvl3 MODEL_PATH=/path/to/InternVL3-8B bash runners/run_ecso_full.sh
#   LIMIT=5 bash runners/run_ecso_full.sh
#   SKIP_GEN=1 / SKIP_JUDGE=1
set -euo pipefail

# ===== CONFIG =====
VLM="${VLM:-llava15}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
DUAL_GPU="${DUAL_GPU:-1}"
VENV="${VENV:-python3}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"

# Default model paths — override with MODEL_PATH=... if needed
if [[ -z "${MODEL_PATH:-}" ]]; then
  case "$VLM" in
    llava15)    MODEL_PATH="/hub/huggingface/models/llava-hf/llava-1.5-7b-hf" ;;
    qwen25vl)   MODEL_PATH="/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct" ;;
    qwen3vl)    MODEL_PATH="/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct" ;;
    phi4)       MODEL_PATH="/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct" ;;
    glm41v)     MODEL_PATH="/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking" ;;
    internvl3)  MODEL_PATH="/hub/huggingface/models/OpenGVLab/InternVL3-8B" ;;
    internvl35) MODEL_PATH="/hub/huggingface/models/OpenGVLab/InternVL3_5-8B" ;;
  esac
fi
JUDGE_CFG="${JUDGE_CFG:-}"
LIMIT="${LIMIT:-}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
PHI4_ATTN_IMPLEMENTATION="${PHI4_ATTN_IMPLEMENTATION:-$ATTN_IMPLEMENTATION}"
PHI4_DISABLE_CACHE="${PHI4_DISABLE_CACHE:-0}"
GLM41V_ATTN_IMPLEMENTATION="${GLM41V_ATTN_IMPLEMENTATION:-$ATTN_IMPLEMENTATION}"
INTERNVL3_USE_FLASH_ATTN="${INTERNVL3_USE_FLASH_ATTN:-0}"
# ==================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EVAL="$ROOT/eval"
MANIFESTS="$ROOT/manifests"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"
OUT_DIR="$OUT_BASE/${VLM}_ecso"
GENERATE_SCRIPT="$EVAL/ecso/generate.py"

JUDGE_PIDS=()

if [[ -z "$JUDGE_CFG" ]]; then
  JUDGE_CFG="$EVAL/configs/judge_default.yaml"
fi

if [[ ! -f "$JUDGE_CFG" ]]; then
  echo "ERROR: judge config not found: $JUDGE_CFG" >&2
  exit 1
fi

if [[ ! -f "$GENERATE_SCRIPT" ]]; then
  echo "ERROR: ECSO generate script not found: $GENERATE_SCRIPT" >&2
  exit 1
fi

case "$VLM" in
  llava15|qwen25vl|qwen3vl|phi4|glm41v|internvl3|internvl35)
    ;;
  *)
    echo "ERROR: unsupported VLM='$VLM' (supported: llava15 qwen25vl qwen3vl phi4 glm41v internvl3)" >&2
    exit 1
    ;;
esac

if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: set MODEL_PATH to the subject VLM checkpoint" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

log() { echo "[$(date +%T)] $*"; }

judge_gpu() {
  if [[ "$DUAL_GPU" == "1" ]]; then
    echo "$JUDGE_GPU"
  else
    echo "$CUDA_VISIBLE_DEVICES"
  fi
}

check_judges() {
  local still_running=()
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      if ! wait "$pid"; then
        log "ERROR: background judge PID $pid failed"
        for p in "${JUDGE_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
        exit 1
      fi
    else
      still_running+=("$pid")
    fi
  done
  JUDGE_PIDS=("${still_running[@]+"${still_running[@]}"}")
}

wait_judges() {
  if [[ ${#JUDGE_PIDS[@]} -eq 0 ]]; then return 0; fi
  log "waiting for ${#JUDGE_PIDS[@]} background judge job(s)..."
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || { log "ERROR: background judge PID $pid failed"; exit 1; }
  done
  JUDGE_PIDS=()
}

_run_gen_on_gpu() {
  local gpu="$1"
  local manifest="$2"
  local out="$3"
  local max_tokens="$4"
  local tmp="${out}.tmp"

  local extra=()
  if [[ -n "$LIMIT" ]]; then extra+=(--limit "$LIMIT"); fi
  case "$VLM" in
    phi4)
      extra+=(--attn_implementation "$PHI4_ATTN_IMPLEMENTATION")
      if [[ "$PHI4_DISABLE_CACHE" == "1" ]]; then
        extra+=(--disable_cache)
      fi
      ;;
    glm41v)
      extra+=(--attn_implementation "$GLM41V_ATTN_IMPLEMENTATION")
      ;;
    internvl3)
      if [[ "$INTERNVL3_USE_FLASH_ATTN" == "1" ]]; then
        extra+=(--use_flash_attn)
      fi
      ;;
  esac

  ( export CUDA_VISIBLE_DEVICES="$gpu"
    "$VENV" "$GENERATE_SCRIPT" \
      --manifest "$manifest" \
      --out "$tmp" \
      --vlm "$VLM" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --max_new_tokens "$max_tokens" \
      "${extra[@]}"
  )
  mv "$tmp" "$out"
}

run_gen() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  local out="$OUT_DIR/${stem}.jsonl"

  check_judges

  if [[ -f "$out" && -s "$out" ]]; then
    log "SKIP gen $stem (exists)"
    return 0
  fi

  log "RUN ECSO gen $stem on GPU $CUDA_VISIBLE_DEVICES (max_new_tokens=$max_tokens)"
  _run_gen_on_gpu "$CUDA_VISIBLE_DEVICES" "$manifest" "$out" "$max_tokens"
}

run_actionable_judge() {
  local stem="$1"
  local bg="${2:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  [[ -f "$gen" ]] || { log "SKIP judge $stem (missing generation)"; return 0; }
  [[ -f "$judged" && -s "$judged" ]] && { log "SKIP actionable judge $stem (exists)"; return 0; }

  _run() {
    local _tmp="${judged}.tmp"
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      "$VENV" "$EVAL/judge/judge_actionable_safety.py" \
        --config "$JUDGE_CFG" \
        --judge_style actionable \
        --generations "$gen" \
        --out "$_tmp"
    )
    mv "$_tmp" "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE actionable judge $stem on GPU $gpu (background)"
    _run & JUDGE_PIDS+=("$!")
  else
    log "RUN actionable judge $stem on GPU $gpu"
    _run
  fi
}

run_context_judge() {
  local stem="$1"
  local manifest="$2"
  local bg="${3:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  [[ -f "$gen" ]] || { log "SKIP judge $stem (missing generation)"; return 0; }
  [[ -f "$judged" && -s "$judged" ]] && { log "SKIP context judge $stem (exists)"; return 0; }

  _run() {
    local _tmp="${judged}.tmp"
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      "$VENV" "$EVAL/judge/judge_context_aware.py" \
        --config "$JUDGE_CFG" \
        --generations "$gen" \
        --manifest "$manifest" \
        --out "$_tmp"
    )
    mv "$_tmp" "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE context judge $stem on GPU $gpu (background)"
    _run & JUDGE_PIDS+=("$!")
  else
    log "RUN context judge $stem on GPU $gpu"
    _run
  fi
}

run_over_refusal_judge() {
  local stem="$1"
  local bg="${2:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  [[ -f "$gen" ]] || { log "SKIP judge $stem (missing generation)"; return 0; }
  [[ -f "$judged" && -s "$judged" ]] && { log "SKIP over-refusal judge $stem (exists)"; return 0; }

  _run() {
    local _tmp="${judged}.tmp"
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      "$VENV" "$EVAL/judge/judge_over_refusal.py" \
        --config "$JUDGE_CFG" \
        --generations "$gen" \
        --out "$_tmp"
    )
    mv "$_tmp" "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE over-refusal judge $stem on GPU $gpu (background)"
    _run & JUDGE_PIDS+=("$!")
  else
    log "RUN over-refusal judge $stem on GPU $gpu"
    _run
  fi
}

schedule_gpu_judges_after_gen() {
  local stem="$1"
  local manifest="${2:-}"
  local bg=0
  [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]] && bg=1

  case "$stem" in
    vlsafe_examine_eval|spa_vl_test_530|mmsb_vision_risk_sdtypo|mm_safetybench_300)
      run_actionable_judge "$stem" "$bg" ;;
    siuo_167|mssbench_unsafe_full)
      run_context_judge "$stem" "$manifest" "$bg" ;;
    mossbench|xstest_safe)
      run_over_refusal_judge "$stem" "$bg" ;;
  esac
}

run_gen_and_maybe_judge() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  run_gen "$stem" "$manifest" "$max_tokens"
  schedule_gpu_judges_after_gen "$stem" "$manifest"
}

run_mcq_score() {
  local stem="$1"
  local manifest="$2"
  local csv_name="${3:-${stem}_score.csv}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/${csv_name}"
  [[ -f "$gen" ]] || { log "SKIP mcq score $stem (missing generation)"; return 0; }
  [[ -f "$csv" && -s "$csv" ]] && { log "SKIP mcq score $stem (exists)"; return 0; }
  log "RUN mcq score $stem (CPU)"
  "$VENV" "$EVAL/score/score_scienceqa.py" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$csv"
}

run_mathvista_score() {
  local stem="$1"
  local manifest="$2"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/${stem}_score.csv"
  [[ -f "$gen" ]] || { log "SKIP $stem score (missing generation)"; return 0; }
  [[ -f "$csv" && -s "$csv" ]] && { log "SKIP $stem score (exists)"; return 0; }
  log "RUN $stem score (CPU)"
  "$VENV" "$EVAL/score/score_mathvista.py" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$csv"
}

run_cpu_scores() {
  run_mcq_score "scienceqa_imgval_full" "$MANIFESTS/scienceqa_imgval_full.jsonl" "sciqa_full_score.csv"
  run_mcq_score "mmstar" "$MANIFESTS/mmstar.jsonl" "mmstar_score.csv"
  run_mcq_score "mme_realworld" "$MANIFESTS/mme_realworld.jsonl" "mme_realworld_score.csv"
  run_mathvista_score "mathvista" "$MANIFESTS/mathvista.jsonl"
  run_mathvista_score "colorbench" "$MANIFESTS/colorbench.jsonl"
}

run_all_gpu_judges_sequential() {
  run_actionable_judge "vlsafe_examine_eval"
  run_actionable_judge "spa_vl_test_530"
  run_actionable_judge "mmsb_vision_risk_sdtypo"
  run_actionable_judge "mm_safetybench_300"
  run_context_judge "siuo_167" "$MANIFESTS/siuo_167.jsonl"
  run_context_judge "mssbench_unsafe_full" "$MANIFESTS/mssbench_unsafe_full.jsonl"
  run_over_refusal_judge "mossbench"
  run_over_refusal_judge "xstest_safe"
}

log "ROOT=$ROOT"
log "VLM=$VLM METHOD=ECSO OUT_DIR=$OUT_DIR"
log "MANIFESTS=$MANIFESTS"
log "GEN_GPU=$CUDA_VISIBLE_DEVICES JUDGE_GPU=$(judge_gpu) DUAL_GPU=$DUAL_GPU"

if [[ "$SKIP_GEN" != "1" ]]; then
  if [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]]; then
    log "pipeline mode: ECSO gen on GPU $CUDA_VISIBLE_DEVICES, judge on GPU $(judge_gpu)"
    run_gen_and_maybe_judge "vlsafe_examine_eval"      "$MANIFESTS/vlsafe_examine_eval.jsonl"      256
    run_gen_and_maybe_judge "spa_vl_test_530"          "$MANIFESTS/spa_vl_test_530.jsonl"          192
    run_gen_and_maybe_judge "mmsb_vision_risk_sdtypo"  "$MANIFESTS/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen_and_maybe_judge "mm_safetybench_300"       "$MANIFESTS/mm_safetybench_300.jsonl"       192
    run_gen_and_maybe_judge "siuo_167"                 "$MANIFESTS/siuo_167.jsonl"                 192
    run_gen_and_maybe_judge "mssbench_unsafe_full"     "$MANIFESTS/mssbench_unsafe_full.jsonl"     192
    run_gen_and_maybe_judge "mossbench"                "$MANIFESTS/mossbench.jsonl"                192
    run_gen_and_maybe_judge "xstest_safe"              "$MANIFESTS/xstest_safe.jsonl"              192
    run_gen "scienceqa_imgval_full"    "$MANIFESTS/scienceqa_imgval_full.jsonl"    192
    run_gen "mmstar"                   "$MANIFESTS/mmstar.jsonl"                   192
    run_gen "mme_realworld"            "$MANIFESTS/mme_realworld.jsonl"            192
    run_gen "mathvista"                "$MANIFESTS/mathvista.jsonl"                256
    run_gen "colorbench"               "$MANIFESTS/colorbench.jsonl"               192
    wait_judges
  else
    run_gen "vlsafe_examine_eval"      "$MANIFESTS/vlsafe_examine_eval.jsonl"      256
    run_gen "spa_vl_test_530"          "$MANIFESTS/spa_vl_test_530.jsonl"          192
    run_gen "mmsb_vision_risk_sdtypo"  "$MANIFESTS/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen "mm_safetybench_300"       "$MANIFESTS/mm_safetybench_300.jsonl"       192
    run_gen "siuo_167"                 "$MANIFESTS/siuo_167.jsonl"                 192
    run_gen "mssbench_unsafe_full"     "$MANIFESTS/mssbench_unsafe_full.jsonl"     192
    run_gen "mossbench"                "$MANIFESTS/mossbench.jsonl"                192
    run_gen "xstest_safe"              "$MANIFESTS/xstest_safe.jsonl"              192
    run_gen "scienceqa_imgval_full"    "$MANIFESTS/scienceqa_imgval_full.jsonl"    192
    run_gen "mmstar"                   "$MANIFESTS/mmstar.jsonl"                   192
    run_gen "mme_realworld"            "$MANIFESTS/mme_realworld.jsonl"            192
    run_gen "mathvista"                "$MANIFESTS/mathvista.jsonl"                256
    run_gen "colorbench"               "$MANIFESTS/colorbench.jsonl"               192
  fi
else
  log "SKIP_GEN=1, generation phase skipped"
fi

if [[ "$SKIP_JUDGE" != "1" ]]; then
  if [[ "$SKIP_GEN" == "1" || "$DUAL_GPU" != "1" ]]; then
    run_all_gpu_judges_sequential
  fi
  run_cpu_scores

  SUMMARY_MD="$OUT_DIR/ecso_summary.md"
  "$VENV" "$EVAL/aggregate/summarize_ecso_metrics.py" \
    --out_dir "$OUT_DIR" \
    --method "ecso" \
    --out_md "$SUMMARY_MD"
  log "Summary written: $SUMMARY_MD"
else
  log "SKIP_JUDGE=1, judging/scoring phase skipped"
fi

log "DONE ECSO pipeline for $VLM"
log "Outputs: $OUT_DIR"
