# 复现执行记录

执行日期：2026-10-05，Asia/Shanghai。分支 `shujuecn`，推送到 `fork`（`shujuecn/CycleTCM`）。第一笔提交 `92c3bce` 保留执行前的环境、路径改动和复现计划。

所有新执行都创建 `YYYYMMDD_HHMMSS_microseconds_用途` 目录。显式 `--resume` 继续同一个特征版本；训练 resume 创建新时间戳目录，并保留来源 checkpoint。已有 legacy 输出不移动、不覆盖。

P0/P1 已完成的数据审计：`outputs/audit/20261005_040401_123463_data_audit/`。3371/843/895 张图、3004/751/895 位受试者，各 split subject 无交叉；原始 CSV、现有 manifest、split JSON 标签一致。35,763 张七视图全部成功解码且为 RGB 224×224，原始 segmented 图与 `pp` 一并解码和记录哈希。5109 个旧缓存向量维度 2560 且 finite。`files.csv` 保存全量产物与源图哈希，`data_manifest.json` 保存 fingerprint、参数与检查结果。区域 montage 与 prevalence 图保留在本地该目录。

区域方向核查：heart_lung 在图像下方舌尖，kidney 在图像上方，liver 在图像右侧，spleen 在中央。源码的 `top/upper` 使用较大的图像 y 值，与普通图像坐标命名相反；报告使用实际区域图和坐标，不凭变量名字判断方位。当前产物与源码对应，不交换已有区域。

旧 Qwen 缓存不能直接归属于当前环境。此前同图对照记录在 `outputs/environment/feature_cache_comparison.json`：Transformers 4.57.6 与旧向量 cosine=0.69205，5.17.0 重提与旧向量完全一致（仅一个样本，不能视为全量历史 provenance）。本次新环境 10 图检查、随后全量重建保存在 `data/features/20261005_041211_284070_qwen_features/`。所有 5109 图均完成，当前/旧向量 cosine 范围 0.68801–0.70916，均值 0.69951；同一输入 `2_1.png` 的当前环境重复抽取与之前当前环境向量逐元素相同。差异远大于已观测到的重复运行波动，因此新旧缓存分版本保留，不做近似等价复用。

新特征 provenance：ModelScope snapshot/master 的全部本地模型文件哈希、固定 prompt/hash、Torch/Transformers/Pillow 版本、BF16、masked mean pooling、逐图输入图哈希、token IDs hash、processor tensor shapes、hidden shape。`records/` 逐图原子落盘，`--resume` 核对权重、配置、已完成输入哈希。最终 `all_features.json` SHA256 为 `e9fb55ce38532e73ac58c86e1f3b14e420693a44e74f9ea044496496a6bdd545`。`master` 不是不可变发布 revision；内容哈希固定的是本地实际权重。

P2 使用 `src/train/reproduce.py` 一个训练/评价实现，原三个入口为薄封装。visual/full 共享前向；以执行前提交中的原模型和相同 state_dict 对照，两者 eval logits 都逐元素完全一致。六组开关真实构建/移除模块；B 保留三分支，A 关闭时使用原始全局/局部向量，U 关闭时直接池化局部向量。该 B 定义是计划中的工作假设，原仓库没有明确的 Table 3 baseline 实现。

数值验证覆盖：实际 AGLFF 门控重建到分类头输入、confidence 对称性/常量/零范数、Top-K 权重归一化与手算 expert 加权、双向残差、Adapter 形状、共享增强的像素相等、两个任务的加权 BCE 之和、task macro positive F1、undefined AUC=null。缺图和越界路径都抛错；无黑图占位或 batch 异常跳过。

checkpoint 保存 model、Adam、scheduler、scaler、epoch、选中指标、early-stop counter、Python/NumPy/Torch/CUDA RNG、配置、数据/特征哈希。best/last 原子写入磁盘，不保存额外 GPU best 副本。保存的实际 best epoch 与历史 val_loss 最低点分别记录。小样本 MLLM 检查中，独立加载后 64 图 logits 全部完全一致；2 epoch + resume 到第 3 epoch 与连续 3 epoch 的全部模型参数、Adam 状态、scheduler 状态完全一致。这些短跑不进入正式结果表。

正式配置已固定在 `configs/reproduction/code_compat.json`：fold1、Adam lr=2e-4 / decay=1e-4、physical batch=32、FP32、200 epoch 上限、patience=50 / min_delta=.001、val task-average Acc 选模型、train_loss ReduceLROnPlateau factor=.3 / patience=5。保留独立七图增强、ToTensor 无 ImageNet normalization、global-only ImageNet V1 初始化、inverse-positive-ratio train/val 各自权重、test 无权重 loss。新的 Qwen 4.57.6 缓存是本次明确记录的版本变化，不宣称与旧提取环境完全相同。未用 test 选择缓存、参数或模型协议。

本机 NVIDIA L20 48 GiB；新训练器 visual FP32 batch32 的真实 optimizer step + validation 峰值 allocated 为 11.568 GiB（工程短跑）。epoch benchmark 以正式运行 history 的 compute_seconds / seconds_with_checkpoint 计量；200 epoch 上限估算需要乘实测完整 epoch，不能由单 step 推断正式训练耗时。

正式执行命令：

```bash
uv run --no-sync python scripts/run_reproduction_suite.py \
  --features data/features/20261005_041211_284070_qwen_features/all_features.json \
  --publish
```

固定队列为 global42 → mllm42 → visual42 → full42 → B42 → BA42 → BU42 → BM42 → B43 → visual43 → full43 → B44 → visual44 → full44。每个模型按既定 early-stop 或 200 epoch 结束后，评价全部 895 图，生成量化图、表格、逐类论文差值，提交并推送汇总结果；任一运行失败则停止队列并保留日志，不跳过继续报成功。

训练数据、模型权重、逐图预测和区域原图保留本地 `data/outputs`；Git 记录代码、配置、aggregate 表格与量化图。`outputs/reproduction/<时间戳>_suite/suite_status.json` 为队列实时状态；`reports/reproduction/<同时间戳>_suite/` 在每次模型结束后更新。报告只收录实际完成的正式实验，不把短跑当作论文结果。

P5 的外部 SOTA 实现和同 split 逐图预测、Sankey 的 8×5 提取公式目前尚未取得；本地 B/visual/full 的配对 subject bootstrap 使用 10,000 次与固定 seed=20261005。不能把这些区间声称为论文 TVMoE 对比。GradCAM 在主要模型训练完成后使用固定样本补充。上述内容仍为待完成项。
