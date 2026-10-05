# 复现执行记录

执行日期：2026-10-05，Asia/Shanghai。分支 `shujuecn`，推送到 `fork`（`shujuecn/CycleTCM`）。第一笔提交 `92c3bce` 保留执行前的环境、路径改动和复现计划。

所有新执行都创建 `YYYYMMDD_HHMMSS_microseconds_用途` 目录。显式 `--resume` 继续同一个特征版本；训练 resume 创建新时间戳目录，并保留来源 checkpoint。已有 legacy 输出不移动、不覆盖。

P0/P1 已完成的数据审计：`outputs/audit/20261005_040401_123463_data_audit/`。3371/843/895 张图、3004/751/895 位受试者，各 split subject 无交叉；原始 CSV、现有 manifest、split JSON 标签一致。35,763 张七视图全部成功解码且为 RGB 224×224，原始 segmented 图与 `pp` 一并解码和记录哈希。5109 个旧缓存向量维度 2560 且 finite。`files.csv` 保存全量产物与源图哈希，`data_manifest.json` 保存 fingerprint、参数与检查结果。区域 montage 与 prevalence 图保留在本地该目录。

区域方向核查：heart_lung 在图像下方舌尖，kidney 在图像上方，liver 在图像右侧，spleen 在中央。源码的 `top/upper` 使用较大的图像 y 值，与普通图像坐标命名相反；报告使用实际区域图和坐标，不凭变量名字判断方位。当前产物与源码对应，不交换已有区域。

旧 Qwen 缓存不能直接归属于当前环境。此前同图对照记录在 `outputs/environment/feature_cache_comparison.json`：Transformers 4.57.6 与旧向量 cosine=0.69205，5.17.0 重提与旧向量完全一致（仅一个样本，不能视为全量历史 provenance）。本次新环境 10 图检查、随后全量重建保存在 `data/features/20261005_041211_284070_qwen_features/`。所有 5109 图均完成，当前/旧向量 cosine 范围 0.68801–0.70916，均值 0.69951；同一输入 `2_1.png` 的当前环境重复抽取与之前当前环境向量逐元素相同。差异远大于已观测到的重复运行波动，因此新旧缓存分版本保留，不做近似等价复用。

新特征 provenance：ModelScope snapshot/master 的全部本地模型文件哈希、固定 prompt/hash、Torch/Transformers/Pillow 版本、BF16、masked mean pooling、逐图输入图哈希、token IDs hash、processor tensor shapes、hidden shape。`records/` 逐图原子落盘，`--resume` 核对权重、配置、已完成输入哈希。最终 `all_features.json` SHA256 为 `e9fb55ce38532e73ac58c86e1f3b14e420693a44e74f9ea044496496a6bdd545`。`master` 不是不可变发布 revision；内容哈希固定的是本地实际权重。

P2 使用 `src/train/reproduce.py` 一个训练/评价实现，原三个入口为薄封装。visual/full 共享前向；以执行前提交中的原模型和相同 state_dict 对照，两者 eval logits 都逐元素完全一致。六组开关真实构建/移除模块；B 保留三分支，A 关闭时使用原始全局/局部向量，U 关闭时直接池化局部向量。该 B 定义是计划中的工作假设，原仓库没有明确的 Table 3 baseline 实现。

数值验证覆盖：实际 AGLFF 门控重建到分类头输入、confidence 对称性/常量/零范数、Top-K 权重归一化与手算 expert 加权、双向残差、Adapter 形状、共享增强的像素相等、两个任务的加权 BCE 之和、task macro positive F1、undefined AUC=null。缺图和越界路径都抛错；无黑图占位或 batch 异常跳过。

训练中的 `last.pt` 保存 model、Adam、scheduler、scaler、epoch、选中指标、early-stop counter、Python/NumPy/Torch/CUDA RNG、配置、数据/特征哈希，用于 resume。`best.pt` 现只保存模型参数与评价元信息；完成后删除 `last.pt`，仅保留可独立评价的最优权重。best/last 原子写入磁盘，不保存额外 GPU best 副本。保存的实际 best epoch 与历史 val_loss 最低点分别记录。此前小样本 MLLM 检查中，独立加载后 64 图 logits 全部完全一致；2 epoch + resume 到第 3 epoch 与连续 3 epoch 的全部模型参数、Adam 状态、scheduler 状态完全一致。这些短跑不进入正式结果表。

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

## 2026-10-05 下午续跑记录

原固定 suite 父进程在 `visual seed=43` 第 71 轮后退出；该运行的 `last.pt` 可读且状态完整，没有使用测试集选择模型。已从该 checkpoint 续跑至原定 patience=50，续跑目录为 `outputs/reproduction/20261005_042629_774918_suite/20261005_154605_393374_code_compat_visual_seed43/`，best epoch=81，测试集 895 张完整评估：证候 Acc/F1=84.86%/70.60%，脏腑 Acc/F1=79.55%/82.07%。旧中断目录 `20261005_132817_358044_code_compat_visual_seed43` 的日志、配置和 best checkpoint 保留为 provenance；续跑完成后删除其过时的 `last.pt`，不计入报告。该目录已于当日晚间被误删，损失与留存范围见下文「中断目录误删与提交信息更正」。

新增 `scripts/continue_reproduction_suite.py` 用于按 suite 状态恢复：已完成目录复用，未完成项顺序执行，每项结束更新报告，成功后删除 `last.pt` 以保留 best checkpoint。该恢复队列当前已启动 `full seed=43`；报告和 `suite_status.json` 会随队列推进更新。当前报告已收录 10 个实际完成运行（global、mllm、visual/full seed=42、B/BA/BU/BM seed=42、B/visual seed=43），未将 full seed=43 及 seed=44 运行提前计入。

## 2026-10-05 晚间跟踪记录

19:12 核查时已完成 12/14 个正式运行：full/43 与 B/44 已完整评估，visual/44 正常训练，full/44 待执行。12 个完成运行均有 895 个唯一测试样本，metrics/summary/逐图预测中的 checkpoint 哈希一致，且各完成目录仅保留 best.pt。续跑控制器 PID=1069139，visual/44 训练 PID=1518002；实时状态以 suite_status.json 为准。

full/43 证候 Acc/F1=82.84%/67.39%，脏腑 Acc/F1=77.12%/80.07%；B 的 seeds 42/43/44 已齐全，证候 F1=70.29%±0.94%，脏腑 F1=81.00%±0.32%（样本标准差）。visual 与 full 仍缺 seed44，暂不把两种子统计当作最终三种子结果。full 在两个已完成种子上均低于 visual，按固定协议报告该偏差，不根据测试结果调参。

权重压缩清单为 `outputs/cleanup/20261005_185917_808971_completed_model_only_checkpoints/cleanup.json`；清理时验证模型 tensor 字节一致，并同步更换所有引用的文件哈希。独立 CPU 加载压缩后的 mllm/42 权重、评价全部 895 测试图，阈值分类与原结果完全一致，最大概率差 4.7132e-7；验证文件为 `outputs/verification/20261005_191303_675867_code_compat_mllm_seed42_eval/verification.json`。

新增 full/43 固定规则 GradCAM 位于 `outputs/qualitative/20261005_191359_522751_gradcam_full_seed43/`；metadata 记录样本、标签、概率、目标层与权重哈希。现有 GradCAM 展示正确样本，错误样本分析及外部 SOTA/Sankey 材料仍属于 P5 待办。

续跑脚本本身没有自动发布；`scripts/track_reproduction_suite.py` 跟踪现有控制器，待报告与完成队列一致后将报告单独提交并推送 fork/shujuecn，最终标记跟踪完成。跟踪记录使用新的时间戳目录 `outputs/tracking/`；失败时记录明确错误，不将未完成运行提前计入。

## 2026-10-05 晚间中断目录误删与提交信息更正

### 续跑事实的确认与此前推断的更正

`visual seed=43` 的续跑是真实发生的续跑，不是从零重训。证据：`20261005_154605_393374_code_compat_visual_seed43/command.json` 记录了 `["src/train/reproduce.py", "--resume", ".../20261005_132817_358044_code_compat_visual_seed43/checkpoints/last.pt", "--output-dir", "..."]`；其 `history.csv` 为 epoch 0–131 连续无缺口，其中 epoch 71 行的 `best_epoch=55`、`early_stop_counter=16` 与被中断运行 `status.json` 记录的状态完全一致，epoch 72 行 `best_epoch=72`、`early_stop_counter=0` 表明早停计数自续跑点重新开始。因此中断前 72 轮的训练轨迹被完整继承，没有被丢弃或重复计算。此前基于时间线推断的「71 轮白跑」结论不成立，特此更正。

同时确认：`--resume` 当时是手动从 shell 调用的，全部脚本中均无调用点。因此「中断的 run 不会被自动续训」这一控制器缺陷仍然成立，只是它并未导致 visual/43 的损失。

### 中断目录误删

在上述更正之前，依据「该 run 与续跑 run 的 `config.json` 完全一致、无进程占用、且状态停留在 training」判定 `20261005_132817_358044_code_compat_visual_seed43` 为无价值残留并执行了删除。该判定不成立：该目录是上节声明保留的 provenance。删除造成的实际损失与留存如下。

已留存（无实质损失）：epoch 0–131 全部逐轮指标（train/val loss、两任务 acc/f1、lr、耗时）完整保存于续跑目录的 `history.csv`；续跑来源路径记录于其 `command.json`；最终最优权重、`metrics/` 与逐图 `predictions/` 均在续跑目录内；数据与特征一致性由 `src/train/reproduce.py` 的 `data_manifest` 严格比对保证，续跑能通过即证明两目录的 manifest 相同；`config.json` 删除前已比对为一致。

已丢失：中断目录的 `train.log`（其中 epoch 0–71 的控制台记录与 `history.csv` 指标重复，仅启动阶段的初始化日志不再留存）、`environment.json`（该次启动时的 git revision 与依赖版本；续跑目录记录的是 14:05 提交 `531bed7`，被中断运行对应的是 13:28 提交 `ca54175`）、`command.json`、`status.json`（其内容已由 `history.csv` epoch 71 行等价覆盖）、`data_manifest.json` 与 `config.json` 副本。

结论：科学结论与全部量化结果不受影响，损失限于中断时段的启动日志与环境记录。审计链上以本节作为断点说明，不再尝试重建已删除目录。

### 两处控制器缺陷及其修复

`src/train/reproduce.py` 的 `--resume` 从未被任何脚本调用，控制器重启后会把无 `summary.json` 的中断 run 一律判为 pending 并从零重训。同时 `scripts/continue_reproduction_suite.py` 以「新增目录」判定本次启动产生的 run，一旦该 cell 存在任何无 `summary.json` 的历史目录，运行成功后即因目录数不为 1 而抛 `RuntimeError` 并终止整个队列。二者均为潜在缺陷，未在本次 visual/43 上触发。

修复内容：`--resume` 改为原地复用被中断的 run 目录（校验 checkpoint 位于 `--output-dir` 内、目录名后缀与 `profile/model/seed` 一致），`train.log` 改为追加模式，`checkpoints` 目录创建改为幂等，并在恢复后释放不再需要的 checkpoint 副本；控制器启动扫描会定位最新可续训目录并传入 `--resume`，run 判定改为「该 cell 下具有 `summary.json` 的目录」，孤儿目录不再静默忽略而是写入队列状态并告警。

### 提交信息与内容不符

`571ac62`（`results: record twelve completed runs and track remaining suite`）同时包含 `src/train/reproduce.py` 的存盘语义变更；`d43acda`（`fix: track completed reports while next training runs`）同时包含 `scripts/continue_reproduction_suite.py` 的续训修复。两处代码内容均已验证正确并推送至 `fork/shujuecn`，但提交标题未反映其中的 checkpoint 语义变更（`best.pt` 改为仅保存可独立评价的权重、`--resume` 改为原地续训）。因历史已发布，不做重写推送，以本节作为对照说明。
