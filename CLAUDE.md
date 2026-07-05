# AdaShield 项目文档

## 项目概述

AdaShield（ECCV 2024）—— 通过自适应防御提示词（Adaptive Shield Prompting）保护多模态大语言模型（MLLM）免受结构化越狱攻击。

**核心思路**：不微调目标 VLM，而是用 Defender LLM 自动迭代优化一条 defense prompt，将其拼在 VLM 输入前，防御基于图像注入的越狱攻击。

论文：https://arxiv.org/abs/2403.09513
项目页：https://rain305f.github.io/AdaShield-Project/

## 架构

```
Defender (Llama-3.1-8B)           Target VLM (LLaVA-1.5 / Qwen2.5-VL 等)
┌──────────────────────┐          ┌──────────────────────────┐
│ 1. 生成 defense prompt│ ──────→ │ 拼在输入前               │
│ 2. 接收越狱响应作反馈 │ ←────── │ 接收 (图片+恶意文本+防御) │
│ 3. 迭代优化 prompt    │          │ 判断是否被越狱           │
└──────────────────────┘          └──────────────────────────┘
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `main_qureyrelated.py` | QR 攻击训练/测评入口 |
| `main_figstep.py` | FigStep 攻击训练/测评入口 |
| `conversers.py` | 模型加载、DefenseVLM、各类 TargetVLM |
| `language_models.py` | HuggingFace/GPT/Claude 生成封装 |
| `config.py` | 模型路径、超参数配置 |
| `common.py` | JSON 解析等工具函数 |
| `judges.py` | 判断 VLM 响应是否被越狱 |
| `infer_benchmark.py` | 评测脚本 |
| `eval_key_word.py` | 关键词评估 |
| `TRAIN_AND_VALIDATE.md` | 训练与评测说明 |

## Pipeline

### Phase 1: 训练防御提示词
对每个安全场景（13 个），Defender 迭代优化 defense prompt：
1. 用当前 defense prompt 让 Target VLM 处理恶意图片+文本
2. 如果 VLM 被越狱（输出不安全内容），Defender 分析失败原因
3. Defender 生成更优化的 defense prompt
4. 重复直到防御成功或达到最大迭代次数

### Phase 2: 评测
用优化后的 defense prompt 在 MM-SafetyBench 上评测 ASR（攻击成功率）。

## 场景列表

### QR Attack (13 场景)
`01-Illegal_Activitiy` `02-HateSpeech` `03-Malware_Generation`
`04-Physical_Harm` `05-EconomicHarm` `06-Fraud` `07-Sex`
`08-Political_Lobbying` `09-Privacy_Violence` `10-Legal_Opinion`
`11-Financial_Advice` `12-Health_Consultation` `13-Gov_Decision`

### FigStep Attack (10 场景)
`01-Illegal_Activity` `02-HateSpeech` `03-Malware_Generation`
`04-Physical_Harm` `05-Fraud` `06-Pornography` `07-Privacy_Violence`
`08-Legal_Opinion` `09-Financial_Advice` `10-Health_Consultation`

---

## 远程服务器

### SSH 连接
```bash
ssh zju_5880_haichao_direct    # 登入用户 jingliu
```

### 硬件
- 2× NVIDIA RTX 5880 Ada Generation (48GB each)
- CUDA 12.8, Driver 570.124.06

### 环境
- **主要环境**: `venv_neurostrike` (AdaShield 运行)
- 其他: `adashield`, `ecso`, `Neuron`
- Conda 路径: `/home/jingliu/miniconda3/`

### 项目路径
```
/home/jingliu/workspece/AdaShield/    # AdaShield 代码
```

### 模型路径 (`/hub/huggingface/models/`)

| 类别 | 可用模型 |
|------|---------|
| LLaMA | Llama-3.1-8B-Instruct, Llama-3-8B, Llama-3-8B-Instruct, Llama-3-70B-Instruct, Llama-2-7b/13b-chat-hf, Llama-3.2-1B/3B/11B-Vision |
| LLaVA | llava-1.5-7b-hf, llava-v1.6-mistral-7b-hf |
| Qwen | Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3-4B/8B/14B/32B, Qwen3.5 系列 |
| InternVL | InternVL3-8B, InternVL3.5-8B, InternVL3.5-30B-A3B |

### HuggingFace 缓存
```
/home/jingliu/.cache/huggingface/hub/
  models--meta-llama--Meta-Llama-3.1-8B-Instruct
  models--meta-llama--Meta-Llama-3-8B
  models--Qwen--Qwen2.5-VL-7B-Instruct
```

---

## 运行命令

### 训练 + 评测（完整 pipeline）
```bash
# 激活环境
source /home/jingliu/miniconda3/etc/profile.d/conda.sh
conda activate venv_neurostrike
cd /home/jingliu/workspece/AdaShield

# QR 训练 + 评测（跳过 FigStep）
CUDA_VISIBLE_DEVICES=1 TARGET_MODEL=llava-1.5 DEFENSE_MODEL=llama-3 SKIP_FIGSTEP=1 \
  bash runners/run_adashield_train_and_eval.sh

# 指定场景和样本数
TARGET_MODEL=llava-1.5 DEFENSE_MODEL=llama-3 TRAIN_LIMIT=3 LIMIT=50 \
  bash runners/run_adashield_train_and_eval.sh

# 只训练 / 只评测
TARGET_MODEL=llava-1.5 SKIP_EVAL=1 bash runners/run_adashield_train_and_eval.sh
TARGET_MODEL=llava-1.5 SKIP_TRAIN=1 bash runners/run_adashield_train_and_eval.sh
```

### 单独训练某个攻击
```bash
# QR 攻击训练
python main_qureyrelated.py \
  --target-model llava-1.5 \
  --defense-model llama-3 \
  --scenario 01-Illegal_Activitiy \
  --init_defense_prompt_path prompts/static_defense_prompt.txt

# FigStep 攻击训练
python main_figstep.py \
  --target-model llava-1.5 \
  --defense-model llama-3 \
  --scenario 01-Illegal_Activity
```

### 单独评测
```bash
python infer_benchmark.py \
  --target-model llava-1.5 \
  --defense-prompt-path wandb/xxx/final_defense_prompt.txt \
  --scenario 01-Illegal_Activitiy
```

---

## 已知问题与修复

### CUDA: probability tensor contains inf/nan
**原因**: Llama-3.1 的 `config.eos_token_id` 是列表 `[128001, 128008, 128009]`，
`pad_token_id` 未设置，HF `generate()` 从列表推断 `pad_token_id` 时导致概率计算 NaN（fp16 精度下更易触发）。

**修复** (`language_models.py`):
1. `__init__` 中显式设置 `tokenizer.pad_token_id` 和 `model.generation_config.pad_token_id`
2. `generate()` 中显式传入 `pad_token_id` 参数
3. 添加 try-catch，CUDA 错误时自动回退到贪心解码

### python: command not found
需要在脚本前激活 conda: `conda activate venv_neurostrike`

---

## 评测指标

- **ASR (Attack Success Rate)**: 越狱成功率，越低越好
- **Defense Success Rate**: 防御成功率 = 1 - ASR
- **Benign Performance**: 在良性数据集（MM-Vet）上的表现，确保防御不影响正常能力
