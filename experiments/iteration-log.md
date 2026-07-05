# AdaShield 迭代实验日志

## 实验目标
找到最优的 defense prompt（Session Prompt），最大化防御成功率（降低 ASR），同时保持良性性能。

## 实验记录

---

### Round 1: 初始静态 Prompt 训练
**日期**: 2026-07-05
**目的**: 用 `static_defense_prompt.txt` 作为初始种子，为每个场景训练优化防御提示词

**配置**:
- Target VLM: `llava-1.5` (LLaVA-1.5-7B-HF)
- Defender: `llama-3` (Llama-3.1-8B-Instruct)
- GPU: CUDA_VISIBLE_DEVICES=1
- 初始 Prompt: `prompts/static_defense_prompt.txt`
- 跳过了 FigStep，仅跑 QR 攻击

**结果**:

| 场景 | 状态 | 关键发现 |
|------|------|---------|
| 01-Illegal_Activitiy | 🔄 运行中 | - |
| 02-HateSpeech | ✅ 之前完成 | 见 final_table.csv |
| 03-Malware_Generation | ✅ 之前完成 | 见 final_table.csv |
| 04-Physical_Harm | ✅ 之前完成 | 见 final_table.csv |
| 05-EconomicHarm | ❌ CUDA 崩溃 | pad_token_id NaN 问题 |
| 06-13 | ⏳ 待运行 | - |

**失败分析**:
- `05-EconomicHarm`: CUDA assertion `probability tensor contains inf/nan` — 已修复 language_models.py
- 需要在本次重跑中验证修复效果

**优化方向 (Round 2 准备)**:
- [ ] 分析各场景 ASR，找最薄弱的场景
- [ ] 观察 defense prompt 的模式：哪些策略有效？
- [ ] 考虑换更强的 Defender（如 Llama-3-70B？）
- [ ] 考虑是否需要在良性数据集上测试以免过拟合

---

### Round 2: TODO（根据 Round 1 结果确定）

## 优化方向池

以下是根据论文和实验经验积累的优化方向：

| 方向 | 描述 | 优先级 |
|------|------|--------|
| 增强初始 Prompt | 手动改进 static_defense_prompt.txt 的指引质量 | 中 |
| 换更强的 Defender | 用 Llama-3-70B 代替 8B | 低（资源） |
| 场景规则细化 | 让 Defender 更早获取场景特定的安全规则 | 高 |
| 多轮防守 | 允许多次防守尝试，而非一次失败就结束 | 中 |
| 良性回归测试 | 在每个 round 后测 MM-Vet 确保不降正常性能 | 高 |
| Prompt 长度控制 | 避免 defense prompt 过长导致 VLM 忽略 | 低 |
| 跨场景迁移 | 测试场景 A 训练的 prompt 能否防御场景 B | 中 |

## 实验模板

每次新 round 复制以下模板：

```markdown
### Round N: [标题]
**日期**: YYYY-MM-DD
**目的**: [简述]

**配置变更** (相对上一轮):
- [列出改动]

**结果**:

| 场景 | ASR | 防御成功? | 备注 |
|------|-----|----------|------|
| 01 | - | - | - |

**失败 case 分析**:
- [case 1]: 为什么失败？下次如何改进？

**成功 case 分析**:
- [case 1]: defense prompt 中哪部分起了作用？

**本轮结论和下一轮方向**:
- [ ] 
```
