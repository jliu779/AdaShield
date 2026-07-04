#!/usr/bin/env python
"""
统一 Benchmark 推理脚本 —— AdaShield 防御评估

支持所有 12 个 benchmark manifest 和 9 种目标 MLLM。
自动构建防御提示词池，通过 CLIP 检索匹配最优 defense prompt。

用法:
  # 安全 benchmark 推理
  python infer_benchmark.py \
      --manifest data/manifests/mm_safetybench_300.jsonl \
      --target-model qwen2.5-vl \
      --defense-pool "figstep_wandb/cogvlm" \
      --output results/mm_safetybench_300_qwen25vl.jsonl

  # 良性 benchmark 推理（相似度门控，避免过度防御）
  python infer_benchmark.py \
      --manifest data/manifests/mmstar.jsonl \
      --target-model llava-1.5 \
      --defense-pool "wandb/llava" \
      --benign --beta 0.7 \
      --output results/mmstar_llava15.jsonl

  # 无防御 baseline
  python infer_benchmark.py \
      --manifest data/manifests/colorbench.jsonl \
      --target-model internvl3.5 \
      --no-defense \
      --output results/colorbench_baseline.jsonl
"""

import argparse
import json
import os
import sys
import glob
import random
import torch
import clip
import pandas as pd
import shortuuid
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import (LLAVA15_PATH, QWEN25VL_PATH, INTERNVL35_PATH,
                    INTERNVL3_PATH, QWEN3VL_PATH, GLM41V_PATH)

# ============================================================
# Benchmark 名称 → manifest 文件映射
# ============================================================
BENCHMARK_MANIFESTS = {
    # 安全/越狱攻击
    "mm_safetybench_300":       "data/manifests/mm_safetybench_300.jsonl",
    "mmsb_vision_risk_sdtypo":  "data/manifests/mmsb_vision_risk_sdtypo.jsonl",
    "mssbench_unsafe_full":     "data/manifests/mssbench_unsafe_full.jsonl",
    "siuo_167":                 "data/manifests/siuo_167.jsonl",
    "spa_vl_test_530":          "data/manifests/spa_vl_test_530.jsonl",
    "vlsafe_examine_eval":      "data/manifests/vlsafe_examine_eval.jsonl",
    "vlsafe_alignment_anchor":  "data/manifests/vlsafe_alignment_anchor_seed42.jsonl",
    # 良性/通用能力
    "colorbench":               "data/manifests/colorbench.jsonl",
    "mathvista":                "data/manifests/mathvista.jsonl",
    "mme_realworld":            "data/manifests/mme_realworld.jsonl",
    "mmstar":                   "data/manifests/mmstar.jsonl",
    "scienceqa_imgval":         "data/manifests/scienceqa_imgval_full.jsonl",
}

# 目标模型路径映射
MODEL_PATHS = {
    "llava":        "modellib/llava-v1.5-13b",
    "minigptv2":    None,  # 需要特殊 cfg 加载，暂不支持
    "cogvlm":       "modellib/cogvlm-chat-hf",
    "llava-1.5":    LLAVA15_PATH,
    "qwen2.5-vl":   QWEN25VL_PATH,
    "internvl3.5":  INTERNVL35_PATH,
    "internvl3":    INTERNVL3_PATH,
    "qwen3-vl":     QWEN3VL_PATH,
    "glm-4.1v":     GLM41V_PATH,
}

MODEL_CONFIGS = {
    "llava":        {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 3000},
    "cogvlm":       {"temperature": 0.8, "top_p": 0.4, "max_new_tokens": 512},
    "llava-1.5":    {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 2048},
    "qwen2.5-vl":   {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 2048},
    "internvl3.5":  {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 2048},
    "internvl3":    {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 2048},
    "qwen3-vl":     {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 2048},
    "glm-4.1v":     {"temperature": 0.2, "top_p": 0.7, "max_new_tokens": 2048},
}


# ============================================================
# CLIP 嵌入
# ============================================================
@torch.no_grad()
def get_clip_embedding(clip_model, clip_preprocess, image_path, query):
    """计算图片+文本的拼接 CLIP 嵌入 (1024维)"""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception:
        return None
    image_tensor = clip_preprocess(image).unsqueeze(0).cuda()
    text_tokens = clip.tokenize(query).cuda()
    image_feat = clip_model.encode_image(image_tensor)
    text_feat = clip_model.encode_text(text_tokens)
    embedding = torch.cat((image_feat, text_feat), dim=-1)  # (1, 1024)
    return torch.nn.functional.normalize(embedding, dim=-1)


# ============================================================
# 防御提示词池
# ============================================================
def build_defense_pool(pool_dir_or_glob, clip_model, clip_preprocess):
    """
    从训练结果的 CSV 文件中构建防御提示词池。

    pool_dir_or_glob: 目录路径或 glob 模式
        例: "figstep_wandb/cogvlm" → 自动搜索其下所有 final_table.csv
        例: "figstep_wandb/cogvlm/*/wandb/latest/files/final_table.csv" → 精确匹配
    """
    if os.path.isdir(pool_dir_or_glob):
        pattern = os.path.join(pool_dir_or_glob, "**/final_table.csv")
    else:
        pattern = pool_dir_or_glob

    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        # 尝试在目录下搜索
        csv_files = glob.glob(os.path.join(pool_dir_or_glob, "**/final_table.csv"), recursive=True)

    if not csv_files:
        print(f"[WARNING] 未找到任何 defense pool CSV 文件: {pattern}")
        return [], None, []

    print(f"[INFO] 找到 {len(csv_files)} 个 defense pool CSV 文件")

    embedding_pool = None
    image_pool = []
    defense_pool = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [SKIP] 无法读取 {csv_path}: {e}")
            continue

        # 筛选成功防御的样本
        if "final_judge_scores" in df.columns:
            df = df[df["final_judge_scores"] == 1]

        if df.empty:
            continue

        images = df["image"].tolist()
        defenses = df["defense_prompt_list"].tolist()
        queries = df["query"].tolist()

        for img_path, defense, query in zip(images, defenses, queries):
            if not os.path.exists(img_path):
                continue  # 防御池中的图片不存在，跳过
            embedding = get_clip_embedding(clip_model, clip_preprocess, img_path, query)
            if embedding is None:
                continue
            if embedding_pool is None:
                embedding_pool = torch.zeros(0, embedding.shape[1]).to(embedding.device)
            embedding_pool = torch.cat((embedding_pool, embedding), dim=0)
            image_pool.append(img_path)
            defense_pool.append(defense)

    print(f"[INFO] 防御池共 {len(defense_pool)} 条成功样本")
    return defense_pool, embedding_pool, image_pool


def retrieve_defense(defense_pool, embedding_pool, sample_embedding,
                     retrival_type="sample-wise"):
    """检索最优 defense prompt"""
    if retrival_type == "random":
        idx = random.randint(0, len(defense_pool) - 1)
        return defense_pool[idx], 0.0
    elif retrival_type == "sample-wise":
        similarity = (sample_embedding.float() @ embedding_pool.float().t()).squeeze()
        best_indices = torch.nonzero(similarity == similarity.max())
        best_idx = random.choice(best_indices).item()
        return defense_pool[best_idx], similarity[best_idx].item()
    else:
        raise ValueError(f"未知 retrival_type: {retrival_type}")


# ============================================================
# VLM 模型加载（统一 HF 模式）
# ============================================================
def load_hf_vlm(model_path):
    """统一加载 HuggingFace VLM"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, processor


def hf_vlm_generate(model, processor, query, image_path,
                    max_new_tokens=2048, temperature=0.2, top_p=0.7):
    """统一 HuggingFace VLM 推理"""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception:
        return None

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": query}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text, images=image, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return response


# ============================================================
# 主推理流程
# ============================================================
def run_benchmark(args):
    # ---- 1. 加载 manifest ----
    if args.manifest:
        manifest_path = args.manifest
    elif args.benchmark:
        if args.benchmark not in BENCHMARK_MANIFESTS:
            available = "\n  ".join(BENCHMARK_MANIFESTS.keys())
            print(f"[ERROR] 未知 benchmark: {args.benchmark}")
            print(f"可用: \n  {available}")
            sys.exit(1)
        manifest_path = os.path.join(PROJECT_ROOT, BENCHMARK_MANIFESTS[args.benchmark])
    else:
        print("[ERROR] 必须指定 --manifest 或 --benchmark")
        sys.exit(1)

    print(f"[INFO] 加载 manifest: {manifest_path}")
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[INFO] 共 {len(samples)} 条样本")

    # ---- 2. 加载 CLIP ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 加载 CLIP ViT-B/32 ...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # ---- 3. 构建防御池 ----
    defense_pool = []
    embedding_pool = None
    if not args.no_defense and args.defense_pool:
        print(f"[INFO] 构建防御池: {args.defense_pool}")
        defense_pool, embedding_pool, _ = build_defense_pool(
            args.defense_pool, clip_model, clip_preprocess
        )
        if not defense_pool:
            print("[WARNING] 防御池为空，将使用无防御模式")
    elif args.no_defense:
        print("[INFO] 无防御模式（--no-defense）")
    else:
        print("[WARNING] 未指定 --defense-pool，将使用无防御模式")

    # ---- 4. 加载 VLM ----
    model_path = MODEL_PATHS.get(args.target_model)
    if model_path is None:
        print(f"[ERROR] 不支持的模型: {args.target_model}")
        print(f"可用: {list(MODEL_PATHS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS.get(args.target_model, {})
    temperature = config.get("temperature", 0.2)
    top_p = config.get("top_p", 0.7)
    max_new_tokens = config.get("max_new_tokens", 2048)

    print(f"[INFO] 加载 VLM: {args.target_model} ({model_path})")
    model, processor = load_hf_vlm(model_path)

    # ---- 5. 输出文件 ----
    if args.output:
        output_path = args.output
    else:
        bench_name = args.benchmark or os.path.splitext(os.path.basename(manifest_path))[0]
        output_path = f"results/{bench_name}_{args.target_model}.jsonl"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_file = open(output_path, "w", encoding='utf-8')

    # ---- 6. 逐样本推理 ----
    success_count = 0
    fail_count = 0

    with tqdm(total=len(samples), desc=args.target_model) as pbar:
        for sample in samples:
            sample_id = sample["id"]
            query = sample["query"]
            image_path = sample["image_path"]

            # 检查图片是否存在
            if not os.path.exists(image_path):
                result = {
                    "id": sample_id,
                    "query": query,
                    "image_path": image_path,
                    "response": None,
                    "defense_prompt": None,
                    "best_similarity": None,
                    "error": "image_not_found",
                    "model_id": args.target_model,
                }
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                fail_count += 1
                pbar.update(1)
                continue

            # CLIP 检索 defense prompt
            defense_prompt = ""
            best_similarity = None

            if defense_pool and embedding_pool is not None:
                sample_emb = get_clip_embedding(
                    clip_model, clip_preprocess, image_path, query
                )
                if sample_emb is not None:
                    defense_prompt, best_similarity = retrieve_defense(
                        defense_pool, embedding_pool, sample_emb,
                        retrival_type=args.retrival_type
                    )

                    # 良性模式：相似度低于阈值时不注入
                    if args.benign and best_similarity <= args.beta:
                        defense_prompt = ""

            # 构建输入 query
            if defense_prompt:
                full_query = query + defense_prompt + query
            else:
                full_query = query

            # VLM 推理
            response = hf_vlm_generate(
                model, processor, full_query, image_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            if response is not None:
                success_count += 1
            else:
                fail_count += 1

            result = {
                "id": sample_id,
                "query": query,
                "image_path": image_path,
                "response": response,
                "defense_prompt": defense_prompt,
                "best_similarity": best_similarity,
                "answer_id": shortuuid.uuid(),
                "model_id": args.target_model,
                "metadata": sample.get("metadata", {}),
            }
            # 保留原始 answer_letter（如果有）
            if "answer_letter" in sample:
                result["answer_letter"] = sample["answer_letter"]

            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            pbar.update(1)

    out_file.close()
    print(f"\n[DONE] 成功: {success_count}, 失败: {fail_count}")
    print(f"[OUTPUT] {output_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AdaShield 统一 Benchmark 推理"
    )

    # Benchmark 选择（二选一）
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--benchmark", type=str,
                     help=f"Benchmark 名称，可用: {', '.join(BENCHMARK_MANIFESTS.keys())}")
    grp.add_argument("--manifest", type=str,
                     help="manifest JSONL 文件路径")

    # 模型
    parser.add_argument("--target-model", type=str, required=True,
                        choices=list(MODEL_PATHS.keys()),
                        help="目标 VLM 模型")

    # 防御
    parser.add_argument("--defense-pool", type=str, default=None,
                        help="防御池目录或 glob 模式（训练结果 CSV 所在目录）")
    parser.add_argument("--no-defense", action="store_true",
                        help="不使用防御提示词（baseline）")
    parser.add_argument("--retrival-type", type=str,
                        default="sample-wise", choices=["sample-wise", "random"],
                        help="CLIP 检索方式")

    # 良性评估
    parser.add_argument("--benign", action="store_true",
                        help="良性评估模式（相似度低于阈值时不注入 defense prompt）")
    parser.add_argument("--beta", type=float, default=0.7,
                        help="良性模式的相似度阈值（默认 0.7）")

    # 输出
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSONL 文件路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大测试样本数（用于快速测试）")

    args = parser.parse_args()
    run_benchmark(args)
