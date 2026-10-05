# CycleTCM 复现执行报告

只汇总实际完成且覆盖全部 895 测试图的运行；短跑为工程检查，不计入论文复现结果。
论文值是参考列。`B` 使用三分支工作假设；原仓库未公开消融 baseline 的明确实现。
固定 fold1、Adam、physical batch 32、FP32、最多 200 epoch、patience 50；按 validation task-average macro Acc 选 checkpoint。
code_compat 保留独立七图增强、无 ImageNet normalization、global-only 预训练、inverse-positive-ratio 权重、validation 自己的权重及 train_loss scheduler。
本次 Qwen 特征在锁定的 Transformers 4.57.6 下重新提取；旧缓存来自不同数值行为，另行保留。
缓存权重、prompt、输入、processor token 和逐图特征 provenance 在 data/features 的时间戳目录中。

| 模型 | seed | best epoch (0-based) | 证候 Acc / F1 (%) | 脏腑 Acc / F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| global | 42 | 104 | 85.32 / 69.88 | 78.79 / 81.18 |
| mllm | 42 | 30 | 74.32 / 62.09 | 69.68 / 77.20 |
| visual | 42 | 49 | 84.76 / 71.47 | 79.53 / 82.09 |
| full | 42 | 96 | 83.24 / 68.15 | 76.80 / 79.70 |
| B | 42 | 108 | 85.31 / 70.88 | 78.66 / 81.01 |
| BA | 42 | 81 | 85.01 / 70.78 | 78.86 / 81.47 |

六组消融和主模型多 seed 的运行状态见 `run_status.json`；未完成时不宣称整个复现完成。
TVMoE 等外部实现/同 split 逐图预测、论文 Sankey 8×5 公式仍缺失，不能复现对应比较和图。
bootstrap.json 仅包含实际存在的本地模型配对；不把本地比较区间说成论文 TVMoE 比较区间。
Spleen 等类别高度不均衡；逐类 support、阴性数、TP/TN/FP/FN 与 AUC 定义状态保存在各运行 metrics 中。
区域 montage 中 heart_lung 位于图像下方舌尖、kidney 位于上方，liver 取图像右侧；以实际代码和图像坐标为准。

数据审计与工程检查见 docs/reproduction_progress.md。
