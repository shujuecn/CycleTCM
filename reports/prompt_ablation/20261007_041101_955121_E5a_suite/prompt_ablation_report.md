# CycleTCM E5a 提示词替换实验结果

生成时间：2026-10-07T09:59:59+08:00。本报告完成 A0 seed=42；A1 seed=42；A2 seed=42；P0 复用既有对应种子。

## 协议

固定 Qwen3-VL-4B-Instruct 本地权重、BF16 单次前向、最后一层全序列 masked mean、2560 维特征、224×224 原有图像与受试者划分。所有全量特征含 5109 张图；full 模型继续使用 FP32、Adam、batch 32、最多 200 epochs、patience 50 和验证集任务平均 Acc 选模；测试为全部 895 位受试者，固定阈值 >0.5。训练 seeds=42，除提示词产生的特征外，模型和训练配置相同。

队列最大并行任务数为 1，每个任务仍为同一 GPU 上的独立训练进程。续训来源、起始 epoch 和 checkpoint 哈希保存在 source_manifest.json；续训使用完整优化器、调度器和随机状态。并行任务的训练耗时包含资源争用，不用于独立推理效率比较。

A0/A1/A2 的 system/user 文本逐字取自验证方案附录 B。图片均在 user 消息内、位于文本之前，与原提取器顺序一致。没有生成 JSON 或执行显式反思，因此本实验衡量的是提示词对隐状态特征与下游分类的影响，不能代表 TongueBench 的生成判读指标。

归档 A0 含显式标签清单，A1 以标签定义代替清单且新增目标句；A1/A2 的开头目标句也不同。保留原文使其符合方案的归档复用要求，但存在这些伴随变化；A1−A0 应理解为归档知识提示版本效应，A2−A1 为归档反思提示版本效应，不能完全排除措辞或长度效应。P0−A0 同时变化语言、粒度、格式和提示内容，仅作整体比较。脏腑标签未出现在 A 组提示词，脏腑变化仅为次级读出。

## 单种子结果

| 提示词 | 证候 Acc (%) | 证候 F1 (%) | 脏腑 Acc (%) | 脏腑 F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| P0 | 83.24 | 68.15 | 76.80 | 79.70 |
| A0 | 82.40 | 66.67 | 76.13 | 79.07 |
| A1 | 82.42 | 65.67 | 74.70 | 77.52 |
| A2 | 82.56 | 66.58 | 75.58 | 79.44 |

结果只使用固定 training seed=42；表中没有训练种子间标准差，不能据此估计训练随机性的总体不确定性。

![提示词 F1](figures/prompt_f1.png)

## 配对差异

按受试者配对重采样 10,000 次；该 CI 衡量固定 seed=42 已训练模型的测试样本不确定性，不包含重新训练的种子总体不确定性。逐种子区间保存在 paired_bootstrap.json。以下差值均为后一配置减前一配置；CI 跨零时不宣称有效提升。多项比较未进行校正，显著结果按探索性证据解读。

| 比较 | 证候 F1 Δ (pp) [95% CI] | 脏腑 F1 Δ (pp) [95% CI] |
| --- | ---: | ---: |
| A1_minus_A0 | -1.00 [-2.61, +0.59] | -1.56 [-2.67, -0.45] |
| A2_minus_A1 | +0.91 [-0.84, +2.66] | +1.93 [+0.84, +3.00] |
| A0_minus_P0 | -1.48 [-3.40, +0.45] | -0.63 [-1.85, +0.60] |
| A2_minus_A0 | -0.10 [-1.84, +1.63] | +0.37 [-0.68, +1.43] |

## 逐类变化与解释

A1−A0 的证候变化最大的三个标签：Spot -6.81 pp；TipSideRed -5.86 pp；Toothmark +4.15 pp。逐类 Acc、F1、支持度和混淆矩阵见 per_class_results.csv。
区间跨零，当前结果不足以宣称该归档提示版本有效提高证候 F1，也不足以证明二者等效。

A2−A1 的证候变化最大的三个标签：Spot +4.16 pp；Ecchymosis +3.03 pp；Toothmark -2.11 pp。逐类 Acc、F1、支持度和混淆矩阵见 per_class_results.csv。
区间跨零，当前结果不足以宣称该归档提示版本有效提高证候 F1，也不足以证明二者等效。

Spleen 的测试支持度为阳性 889、阴性 6，严重不平衡；其 F1 不应独立用来说明提示词的临床知识价值。

## 复现与产物

完整 feature JSON、逐图预测和权重保留在本地 data/ 与 outputs/，不推送受试者级资料。此目录保存完整汇总、逐类结果、prompt 原文与哈希、训练来源清单、配对区间以及图表。

```bash
for variant in A0 A1 A2; do
  uv run --no-sync python scripts/extract_prompt_features.py \
    --variant "$variant" --model-dir /path/to/Qwen3-VL-4B-Instruct \
    --images-dir data/processed/CycleTCM/images \
    --output-dir "data/features/prompt_20261007_$variant" --resume
done
uv run --no-sync python scripts/run_prompt_ablation.py --seeds 42 --jobs 1 \
  --features data/features/prompt_20261007_A0/all_features.json \
             data/features/prompt_20261007_A1/all_features.json \
             data/features/prompt_20261007_A2/all_features.json
uv run --no-sync python scripts/report_prompt_ablation.py --suite outputs/prompt_ablation/20261007_041101_955121_E5a_suite
```
