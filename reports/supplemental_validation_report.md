# CycleTCM 补充实验结果报告

日期：2026-10-07（Asia/Shanghai）。本报告记录依据 `docs/CycleTCM-质疑查证与消融验证方案.md` 已实际完成的 E1 控制特征、E2 MLLM-only、E3 效率、E5a 提示词替换和 E6 多种子验证。训练沿用 `code_compat`：受试者级固定划分、895 张测试图、FP32、batch 32、200 epoch 上限、patience 50、验证集选模和正类频率加权 BCE。

## 结果

### E1/E2：特征来源控制与 MLLM-only

所有配置均使用相同的 MLP 分类器和测试集。`Qwen-4B` 的 seed 42 来自既有正式复现实验，seed 43/44 为本次补跑；常量和随机向量是 2560 维、冻结、逐图固定的控制特征。

| 特征来源 | seed | 证候 Acc (%) | 证候 F1 (%) | 脏腑 Acc (%) | 脏腑 F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-4B | 42 | 74.32 | 62.09 | 69.68 | 77.20 |
| Qwen3-VL-4B | 43 | 74.50 | 64.35 | 70.50 | 77.95 |
| Qwen3-VL-4B | 44 | 75.34 | 62.56 | 66.35 | 77.86 |
| 常量向量 | 42 | 64.16 | 50.78 | 65.65 | 77.84 |
| 随机向量 | 42 | 71.06 | 44.52 | 62.68 | 67.78 |

Qwen 的三种子平均为证候 F1 **63.00±1.18**、脏腑 F1 **77.67±0.40**（样本标准差）。常量控制在证候 F1 上低 12.22 pp，随机控制再低 6.26 pp；脏腑任务的常量结果接近 Qwen，反映该任务受多数类和标签不平衡影响，不能单独用 Acc 或脏腑 F1 证明语义信息有效。结果支持“Qwen 特征包含超出固定全局偏置的样本变化”，但尚未证明这些变化全部来自临床语义，也没有完成 full 模型的常量／随机替换训练，因此 E1 的端到端结论仍是部分完成。

### E3：效率

在 NVIDIA L20 48 GiB、锁定 Transformers 4.57.6 环境上测量，Qwen 特征单图前向取 8 次（首个 warm-up 样本也记录），视觉和 full 各取 8 次推理前向并跳过 2 次 warm-up：

| 项目 | 均值 | 峰值显存 |
| --- | ---: | ---: |
| Qwen3-VL-4B 特征前向，batch 1 | 124.93 ms/图 | 8.40 GiB |
| visual 视觉分支前向 | 8.39 ms/图 | 10.29 GiB |
| full 视觉分支 + Adapter 前向 | 8.29 ms/图 | 10.33 GiB |

Qwen 特征前向约为视觉分支的 **14.89 倍**。这里的 full 测量只包含已经缓存的特征输入；端到端在线流程还应加上 Qwen 前向，因此部署延迟不能用 full 的 8.29 ms 代表。测量原始值保存在 `reports/efficiency_measurement.json`。

### E6：多种子与配对区间

已有正式运行的 visual/full 三种子结果为：visual 证候/脏腑 F1 **70.84±0.56 / 81.78±0.51**，full **67.15±1.14 / 79.68±0.40**。full−visual 的平均变化为 **−3.68 / −2.10 pp**。895 位受试者配对 bootstrap（10,000 次）在 seed 42 上给出证候 F1 95% CI **[−5.25, −1.35] pp**、脏腑 F1 **[−3.52, −1.23] pp**；seed 43 和 44 的两个区间也均在 0 以下。因而本次实现中，多模态拼接没有重现论文声称的增益。

### E5a：提示词替换

A0、A1、A2 均完成 5109 张图像的 Qwen3-VL-4B 特征抽取，并在 full 模型上完成 seed 42 训练；P0 使用既有正式复现的 seed 42。测试集均为 895 位受试者：

| 提示词 | 证候 F1 (%) | 脏腑 F1 (%) |
| --- | ---: | ---: |
| P0 | 68.15 | 79.70 |
| A0 | 66.67 | 79.07 |
| A1 | 65.67 | 77.52 |
| A2 | 66.58 | 79.44 |

配对受试者 bootstrap（10,000 次）显示：A1−A0 的证候 F1 变化为 **−1.00 pp**，95% CI **[−2.61, +0.59]**；A2−A1 为 **+0.91 pp**，95% CI **[−0.84, +2.66]**。脏腑 F1 分别为 **−1.56 pp [−2.67, −0.45]** 和 **+1.93 pp [+0.84, +3.00]**。本次按用户要求只使用 seed 42，表中没有训练种子间标准差，不能据此估计训练随机性的总体不确定性。提示词归档还伴随开头措辞和标签清单形式变化，因此差异按归档提示版本效应解释。完整产物见 [E5a 报告](prompt_ablation/20261007_041101_955121_E5a_suite/prompt_ablation_report.md)。

## 结论和边界

现有证据形成三点：Qwen MLLM-only 特征比常量和随机控制更有判别力；其离线前向成本显著高于视觉分支；在当前训练实现和缓存版本中，full 融合反而稳定损害 visual 的 macro positive-class F1。常量／随机控制目前只完成 MLLM-only 路径，不能替代完整 E1-F1/F2 端到端矩阵。

E4 池化、E5b 提示词生成模式、E7 BCE 与 Lovász 的完整三种子训练没有纳入本次结果；E5a 仅完成上述 seed42 单种子版本。训练器已加入可选 `loss=bce_lovasz` 和任意 MLLM 输入维度，以便后续按同一协议继续；本报告不把未运行配置写成实验结论。

## 可追溯文件

- 实验方案：[docs/CycleTCM-质疑查证与消融验证方案.md](../docs/CycleTCM-质疑查证与消融验证方案.md)
- 效率原始记录：[reports/efficiency_measurement.json](efficiency_measurement.json)
- 控制特征生成器：[scripts/build_synthetic_feature_controls.py](../scripts/build_synthetic_feature_controls.py)
- 效率测量器：[scripts/measure_efficiency.py](../scripts/measure_efficiency.py)
- Lovász 损失实现：[src/train/losses.py](../src/train/losses.py)
- E5a 提示词替换报告：[reports/prompt_ablation/20261007_041101_955121_E5a_suite/prompt_ablation_report.md](prompt_ablation/20261007_041101_955121_E5a_suite/prompt_ablation_report.md)

控制特征按图像文件名的 SHA-256 派生随机种子生成；其完整 5109×2560 JSON 保留在本机实验目录，未纳入 Git。
