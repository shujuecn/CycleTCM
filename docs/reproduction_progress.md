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

原固定 suite 父进程在 `visual seed=43` 第 71 轮后退出；该运行的 `last.pt` 可读且状态完整，没有使用测试集选择模型。已从该 checkpoint 续跑至原定 patience=50，续跑目录为 `outputs/reproduction/20261005_042629_774918_suite/20261005_154605_393374_code_compat_visual_seed43/`，best epoch=81，测试集 895 张完整评估：证候 Acc/F1=84.86%/70.60%，脏腑 Acc/F1=79.55%/82.07%。旧中断目录 `20261005_132817_358044_code_compat_visual_seed43` 原本保留作 provenance，随后于当日晚间被误删；损失与留存范围见下文「并发改动核查与修正」。

新增 `scripts/continue_reproduction_suite.py` 用于按 suite 状态恢复：已完成目录复用，未完成项顺序执行，每项结束更新报告，成功后删除 `last.pt` 以保留 best checkpoint。该恢复队列当前已启动 `full seed=43`；报告和 `suite_status.json` 会随队列推进更新。当前报告已收录 10 个实际完成运行（global、mllm、visual/full seed=42、B/BA/BU/BM seed=42、B/visual seed=43），未将 full seed=43 及 seed=44 运行提前计入。

## 2026-10-05 晚间跟踪记录

19:12 核查时已完成 12/14 个正式运行：full/43 与 B/44 已完整评估，visual/44 正常训练，full/44 待执行。12 个完成运行均有 895 个唯一测试样本，metrics/summary/逐图预测中的 checkpoint 哈希一致，且各完成目录仅保留 best.pt。续跑控制器 PID=1069139，visual/44 训练 PID=1518002；实时状态以 suite_status.json 为准。

full/43 证候 Acc/F1=82.84%/67.39%，脏腑 Acc/F1=77.12%/80.07%；B 的 seeds 42/43/44 已齐全，证候 F1=70.29%±0.94%，脏腑 F1=81.00%±0.32%（样本标准差）。visual 与 full 仍缺 seed44，暂不把两种子统计当作最终三种子结果。full 在两个已完成种子上均低于 visual，按固定协议报告该偏差，不根据测试结果调参。

权重压缩清单为 `outputs/cleanup/20261005_185917_808971_completed_model_only_checkpoints/cleanup.json`；清理时验证模型 tensor 字节一致，并同步更换所有引用的文件哈希。独立 CPU 加载压缩后的 mllm/42 权重、评价全部 895 测试图，阈值分类与原结果完全一致，最大概率差 4.7132e-7；验证文件为 `outputs/verification/20261005_191303_675867_code_compat_mllm_seed42_eval/verification.json`。

新增 full/43 固定规则 GradCAM 位于 `outputs/qualitative/20261005_191359_522751_gradcam_full_seed43/`；metadata 记录样本、标签、概率、目标层与权重哈希。当时的 GradCAM 展示正确样本，错误样本分析及外部 SOTA/Sankey 材料尚待补充；该早期图组后来由 seed42 固定正确/错误样本对照取代，整理时已清理。

续跑脚本本身没有自动发布；`scripts/track_reproduction_suite.py` 跟踪现有控制器，待报告与完成队列一致后将报告单独提交并推送 fork/shujuecn，最终标记跟踪完成。跟踪记录使用新的时间戳目录 `outputs/tracking/`；失败时记录明确错误，不将未完成运行提前计入。

## 2026-10-05 并发改动核查与修正

20:05 核查依据为 OpenCode 会话 `docs/session-ses_ef46.md`、实际 Git diff、留存的命令、日志和训练历史。会话中的推断不直接作为执行事实；以下结论由本地文件核验。

### visual/43 的真实续训与误删范围

`20261005_154605_393374_code_compat_visual_seed43/command.json` 记录了从 `20261005_132817_358044_code_compat_visual_seed43/checkpoints/last.pt` 手动续训。其 `history.csv` 为 epoch 0–131 连续无缺口。套件根目录的 `visual_seed43.log` 仍保存原始启动 RUN/PARAMETERS 记录及 epoch 0–71 的全部控制台输出；本次把这 72 轮的每个数值字段与续训历史比较，全部相同。

epoch 71 的 best_epoch=55、early_stop_counter=16；epoch 72 的验证选择分数超过此前最优分数与 min_delta=.001 之和，因此 best_epoch 更新为 72，counter 才归零。计数归零由验证改善触发，不能解释为恢复时自动重置。中断前 72 轮确已继承，「从零重训」「71 轮白跑」和「原始启动日志全部丢失」均不成立。

OpenCode 清理时删除了原中断目录。最终 best、全部 132 轮指标、895 图测试指标与逐图预测仍在续训目录内；原始控制台日志仍在套件根目录。实际缺失的是原目录的 environment、command、status、data_manifest、config 等元数据副本及旧 checkpoint。原始 environment.json 已删除，不能把根据提交时间推测的 revision 写成已记录事实。续训执行时的代码要求 checkpoint 的 data_manifest 与当前输入一致，但如今无法重新比较已删除的原始文件。保留这一 provenance 缺口，不伪造原目录。

本次核验和原始 stdout 的字节相同副本保存在 [audit.json](../outputs/audit/20261005_200511_332975_concurrent_edit_reconciliation/audit.json)；比较结果没有改变已报告的量化指标。

### 统一续训与失败行为

并发改动曾把训练 resume 改为原地覆盖目录，启动时会重写 config/environment/command 等文件，违反每次执行创建时间戳目录的要求。本次恢复为每次续训新建目录，写入 resume.json 记录来源与哈希，继承模型、优化器、调度器、scaler、RNG、history、best 与早停计数；原目录保持不变。此前 best 复制到新目录，即使续训没有新的改善，最终结果仍有自己的可独立评价权重。best.pt 继续只存模型及评价元信息，last.pt 才支持续训。

继续队列启动时自动寻找最新未完成且有 last.pt 的目录，核对特征缓存与固定配置。运行归属以启动前后新增的目录判断，成功同时要求 summary.json 与 status=complete；历史孤儿目录只记录并告警，不再影响新增目录计数。非零退出、无新目录或缺失完成记录均写入 suite/report 的 failed 状态再退出。套件 stdout 追加保留，训练自身的日志位于新的执行目录。

实际 CPU 验证命令：

```bash
uv run --no-sync python scripts/verify_resume.py \
  --features data/features/20261005_041211_284070_qwen_features/all_features.json
```

验证结果见 [verification.json](../outputs/verification/20261005_200827_948865_resume_reconciliation/verification.json)：2 轮训练后续到第 3 轮与连续 3 轮的 model/Adam/scheduler/scaler/选优和早停状态完全一致；来源全部文件哈希不变；新增时间戳目录且无新改善时 best 仍独立保留。恢复时若已达到 epoch 上限，仍保存继承的 history 并完成评价。控制器成功续训发现新目录、历史孤儿目录并存、失败时保存 failed 状态均通过。这些小样本工程验证不进入正式结果表。

### Git 记录与持续跟踪

`571ac62` 同时收录了仅存模型的 best checkpoint 修改与 OpenCode 的原地续训修改；`d43acda` 同时收录了报告跟踪修改与已暂存的 OpenCode 控制器修改。问题分别涉及共享文件并发编辑，以及普通 git commit 包含整个暂存区；仅指定 git add 文件仍不足以隔离提交。会话中的「使用 git add -A」是推断，本次没有证据支持。

保留这些历史 revision，新增明确说明续训、状态与 provenance 修正的提交，避免改变已有 environment.json 中引用的 revision。所有发布入口均列出具体文件路径，并用 git commit --only 提交指定文件；本次也按此执行。原始 OpenCode 会话与无关 HTML 不纳入修复提交。

现有 GPU visual/44 与控制器继续运行，不为代码核查重启。20:04 已到 epoch 76，已完成正式运行仍为 12/14，之后按固定队列启动 full/44。运行中的进程保持启动时的代码；后续新进程读取修正后的源码。报告跟踪器只在报告与完成队列一致后提交并推送明确列出的报告文件。完成后的权重继续只保留 best，日志、配置和元数据保留。

## 2026-10-05 论文式报告与定量、定性分析准备

22:03 的阶段性分析基于实际完成的 13/14 个正式运行，visual/44 已完成；full/44 尚在训练，不提前计入。阶段报告原存于 `reports/reproduction/20261005_220342_355189_analysis/paper_report.md`（整理时已由最终版替代，Git 提交 `e38d3a6` 可查阅），包含摘要、方法、结果、讨论、结论、参考材料及运行附录。报告区分论文引用值、本地单种子消融、多种子均值±样本标准差和同种子的配对 bootstrap 区间，不将 full 的两种子均值视为最终三种子统计。

`scripts/analyze_reproduction.py` 从完成队列及其保存的 895 图预测生成独立时间戳报告。全部 13 个运行的逐类指标及主要 macro 指标均由概率精确重算并核对，样本、标签、subject、权重哈希引用对齐。新增五类图：多种子分布、论文/本地消融对照、配对效应森林图、类别构成及条件错误率、逐类 F1 差值；连同原有三类图共八张定量图，新增图同时输出 PNG 与 PDF。来源文件哈希、完整环境快照、逐类统计及训练资源另存 source_manifest.json、per_class_analysis.csv、resources.csv。报告检查见 `outputs/verification/20261005_220558_589655_scientific_report/verification.json`。

定性选择使用 visual/42 为参考，固定 TonguePale、Crack、Heart、Kidney 四个标签，每个 TP/TN/FP/FN cell 选择词典序首个样本，在 visual/42 与 full/42 的同一样本上对照。共 16 个标签—样本组合、32 张真实 GradCAM、四张三列组合图。源 metadata 位于 `outputs/qualitative/20261005_214746_808479_gradcam_visual_seed42_confusion/` 与 `20261005_215317_840103_gradcam_full_seed42_confusion/`；组合图路径由报告中的相对链接给出。

GradCAM 使用 CPU，不占训练 GPU；针对正类 logit、全局 layer4[-1]、每图独立归一化。CPU 单图与正式 GPU batch 概率最大差为 0.000675828，所选 32 个模型—样本组合均未改变阈值分类。metadata 同时保留原正式概率和解释前向概率。已修正开发时过严的 1e-4 概率断言并记录失败原因（后续整理已清理失败目录，原因保存在清理清单），不把不同硬件的近似前向声称为逐元素相同。报告使用 C01–C16 代号，原始图像、CAM、样本映射及逐图预测文件保留本地。

目前已完成的 visual/full 配对 seeds 42/43 中，完整模型证候/脏腑 F1 平均变化为 -3.26/-2.19 pp，对应区间均低于 0；不作未经控制的差异来源归因。Spleen 的六个阴性样本在 seed42 两种模型中均被误判为阳性，FurThick 也存在高 F1 与较高 FPR 并存的现象。固定定性集合里 full 修正三个 visual 错例，同时新增三个错误；这一人为分层集合不能估计总体增益。

跟踪器新增 `--analysis-qualitative VISUAL_METADATA FULL_METADATA`，只在 suite 和报告均完成后自动生成新的时间戳最终报告并推送；不重启训练控制器。完成触发一次、传递固定样本 metadata、传递 publish、分析失败时保存 failed 状态均已验证，见 `outputs/verification/20261005_220225_075009_analysis_tracker/verification.json`。新报告提交依然使用明确文件列表和 git commit --only；开发阶段的渲染草稿当时保留于 outputs/analysis_drafts，最终产物核验后已在仓库整理时删除。

## 2026-10-05 固定队列完成与最终报告

22:23 固定队列已完成 14/14 个正式运行，全部覆盖 895 位测试受试者；B、visual、full 的 seeds 42/43/44 均齐全。最终论文式报告为 [paper_report.md](../reports/reproduction/20261005_222333_753567_analysis/paper_report.md)，阶段版后来在仓库整理时删除，历史内容可从 Git 提交 `e38d3a6` 查阅。最终报告与定量图已由跟踪器自动生成，提交 `7065b37` 并推送；固定队列汇总提交为 `374330b`。

| 模型（三种子） | 证候 Acc / F1 (%)，均值±样本 SD | 脏腑 Acc / F1 (%)，均值±样本 SD |
| --- | --- | --- |
| B | 84.93±0.58 / 70.29±0.94 | 78.69±0.09 / 81.00±0.32 |
| visual | 84.54±0.47 / 70.84±0.56 | 79.12±0.73 / 81.78±0.51 |
| full | 82.97±0.24 / 67.15±1.14 | 76.95±0.16 / 79.68±0.40 |

完整模型相对纯视觉的配对三种子 F1 平均变化为 -3.68/-2.10 pp；六组本地配对 bootstrap 的定义、样本和哈希均保留。此结论限定于本次固定协议，不宣称复现了论文的多模态增益，也不对差异来源作未经控制的因果归因。最终检查见 `outputs/verification/20261005_222831_471708_final_scientific_report/verification.json`：14 个运行、三种子统计、六个配对区间、报告链接及图形均通过检查。

收尾时发现原控制器、full/44 训练和旧跟踪器均已退出，套件状态仍为 running，最后训练日志停在 22:09:41 的 epoch124。日志无 Python 错误堆栈，所检查的内核日志也无对应记录，退出原因无法确认。22:18 启动新控制器和报告跟踪器，记录于 `outputs/tracking/20261005_221810_503750_suite_resume_with_analysis/launch.json`；使用保存的 last.pt 从 epoch125 续训，继承 best_epoch=80、counter=44，至 epoch130 按 patience=50 结束。正式完成目录为 `20261005_221822_932720_code_compat_full_seed44`，继承的 history 连续为 epoch0–130，resume.json 保存来源 checkpoint 与 best 的哈希。

完成评价和最终报告后，核对旧 full/44 来源的 last/best 文件哈希及新目录最优权重哈希，删除旧目录的过时 checkpoint，释放 7.58 GiB；来源环境、配置、日志、历史和命令记录均保留，不删除目录。清单为 `outputs/cleanup/20261005_222835_404787_full44_resume_source_weights/cleanup.json`。14 个正式完成目录均只有 best.pt；训练控制器及报告跟踪器已结束，跟踪状态为 complete。

主要结果、六组消融及预设三种子训练已经完成。本文定性对照覆盖四个标签的固定正确/错误样本，并非论文 13 类解释图的全量精确复现；外部 SOTA 的同 split 实现/预测与论文 Sankey 提取公式仍缺失，整个 P5 不标记为全部完成。

## 2026-10-05 仓库与输出整理

按用户要求清理已被最终结果取代的阶段产物。清单见 [cleanup.json](../reports/maintenance/20261005_230055_671161_repository_cleanup/cleanup.json)，本地副本为 `outputs/cleanup/20261005_230055_671161_repository_cleanup/cleanup.json`。本次删除 35 项、124 个文件，共释放 2.84 GiB：工程短跑及续训回归测试权重、两套论文分析草稿、早期/失败的 GradCAM 与重复组合图、13/14 阶段报告，以及运行时绘图库缓存。工程验证的配置、日志、指标和验证结论继续保留；失败图组原因已写入清单。阶段报告历史仍在 Git 中，没有重写实验所引用的提交。

保留 14 个正式运行的 best.pt、逐图预测、评价、历史、配置和环境；full/44 的中断来源元数据保留。最终论文式报告和两个模型的原始 GradCAM、最终组合图均保留。清理前后的 218 个受保护记录文件逐字节哈希相同，14 个最优权重的大小及修改时间未变。README 重写为中文复现入口，列出实际结果、锁定环境、可执行命令、产物位置及 P5 未完成项。无关工作区文件及他人改动不纳入提交。
