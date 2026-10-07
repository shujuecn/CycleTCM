# CycleTCM 实证验证方案

> 对象：MICCAI 2026 论文 *MLLM-Enhanced Region-Aware Bidirectional Evidence-Based Model for Tongue Diagnosis*（编号 0903）；代码：`src/CycleTCM/`（官方仓库克隆）。
> 前提：论文实验已复现——visual-only（`train_model_visual.py`）与完整模型（`train_model_multimodal.py`）的结果、以及 MLLM 特征文件 `all_features.json` 均已拿到，直接用作下述实验的对照与输入。
> 本文只记录**需要新跑的验证实验**；已通过论文与代码静态核实、无需重跑的结论压缩为 §1 事实表（作为实验动机），不再展开论证。本文自明，不依赖其他文档。
> 日期：2026-10-07

---

## 1. 背景速览

**任务与报告数字**：TongueDx 数据集（5,109 图 / 4,650 受试者，手机/笔记本拍摄），8 证候 + 5 脏腑多标签分类，受试者级划分 3,371 / 843 / 895。论文报告证候平均 Acc 86.05 / F1 72.75，脏腑平均 Acc 80.09 / F1 82.67（"mean%"，未定义运行次数）。架构：三分支 ResNet50 编码器（whole / edge+body / 4 脏腑区，共 7 视图）→ AGLFF 全局-局部融合 → UWBMoE 双向 MoE（4 专家 top-2）→ 双分类头；冻结 Qwen3-VL-4B-Instruct 特征经适配器拼接进两个分支。训练：H20、224×224、batch 32、Adam 2e-4 + ReduceLROnPlateau×0.3、WD 1e-4、200 epochs 早停 patience 50。

**已确认事实**（论文与代码逐条核实，出处见 §5 代码地图）：

| 编号 | 事实 |
|---|---|
| F-A | MLLM 用法：冻结 Qwen3-VL-4B **单次前向、从不生成**（`add_generation_prompt=True` 后直接取 `hidden_states[-1]`），对**全序列**（图像 token + 固定文本 + 模板）mean pooling 成 2560 维向量，离线预计算存 `all_features.json`；部署时每个新图必须先过 4B 前向（BF16 ≈ 8 GB）。特征经适配器后**同一向量硬拼接**进证候与脏腑两分支（各占 6144 维输入的 1/3，无交互）。论文与代码均无延迟/显存/吞吐数据 |
| F-B | MLLM 的"临床先验"实为 8 句英文同义反复（如 "TonguePale indicates the tongue is pale."），无视觉判据、区域映射、程度分级、鉴别规则；全文见附录 A |
| F-C | MLLM 模块消融贡献为三个模块中最小之一：单独加在 backbone 上 +0.62 / +1.09 Acc（证候/脏腑），完整模型中约 +0.87 / +0.63 |
| F-D | 仓库自带 MLLM-only 基线（`model_mllm.py`：MLP 直接吃 2560 维特征）与预测后融合脚本（`late_fusion.py`），论文均未报告结果 |
| F-E | 训练损失代码实际只有按正类频率加权的 BCE；论文声称 `WCE + 0.7·Lovász-Softmax` |
| F-F | 类别不平衡严重：Spleen 为退化标签（所有方法 99.33 / 99.66 完全打平），Pale/Ecchymosis 的 Acc≈82–89 而 F1 仅 29.7–45.4，MIRnet 在两个标签上 F1=0.00（塌缩多数类）。模型间差异应以 F1 为准，Acc 参考价值有限 |
| F-G | 固定文本（系统提示 + 先验 + 用户句）对每张图完全相同，全序列 mean pooling 使 2560 维特征中很大比例为跨样本近似常量成分，样本间信号主要来自图像 token 且被稀释 |

---

## 2. 验证实验

> 约定：评测协议与论文一致（受试者级划分、per-label Acc/F1 + 平均）；除特别说明，每个配置 ≥3 个随机种子（见 E6）。

### E1（核心）特征来源替换矩阵：4B 是必要的吗

**动机**（F-A、F-C）：MLLM 特征可能只起"全局偏置"作用（常量成分 + 硬拼接），或任意随样本变化的特征即可替代，且 4B 规模可能不必要——这直接决定该模块的存废与部署成本。

**设计**：固定视觉流水线与训练超参，仅替换进入适配器的特征向量：

| 配置 | 特征来源 | 参数量 | 检验什么 |
|---|---|---|---|
| F0 | 无（visual-only） | 0 | 对照基线（复用复现结果） |
| F1 | 常量向量（全 1 或固定高斯向量 × 可学习缩放） | ~0 | 特征是否只起"全局偏置"作用 |
| F2 | 随机冻结编码器（随机初始化 ResNet50/ViT，冻结，全局池化） | ~25–86M | 任意变化的特征是否就能拿到增益（增益是否与语义无关） |
| F3 | 对比式图文模型：SigLIP（~0.4B）或 Chinese-CLIP ViT-B/16（~0.2B，支持中文文本输入），与原版相同协议（图像 + 先验文本 → 池化特征） | 0.2–0.4B | 同等信息量的视觉-语言先验，1/10 规模是否足够 |
| F4 | 同家族缩放对照：Qwen2-VL-2B-Instruct，提示词与提取协议与原版完全一致 | 2B | 隔离"规模"单一变量 |
| F5 | 原版 Qwen3-VL-4B-Instruct | 4B | 锚点（复用复现结果） |

**要点**：F3/F4 的提取脚本由 `mllm_feature_extract.py` 改造（换模型类与 processor，保留 chat 模板与池化逻辑）；F1/F2 为纯张量操作；所有来源统一经相同的 `MLLM_Adapter`（input_dim 改为实际维度）进入分类器。

**判定与解读**：
- F1 ≈ F5：特征 = 全局偏置，"MLLM 增强"的核心叙事失去支撑；
- F2 ≈ F5：增益与语义无关；
- F3 ≈ F5：4B 不必要，可低成本替换（部署结论的直接证据）；
- F4 显著劣于 F5：规模确实重要，问题弱化为"成本收益比"；
- 全部 ≈ F0：MLLM 模块整体无效。

### E2 MLLM-only 基线：Qwen 特征本身值多少钱

**动机**（F-D）：检验特征是否携带任务判别信息。该实验代码已有、论文缺席，无论结果如何都值得写进复现报告。

**做法**：跑仓库自带 `train_model_mllm.py`（MLP 直接分类 2560 维特征，8+5 两头），报告 per-label Acc/F1。

**解读**：接近 chance（尤其稀有标签 F1≈0）→ 特征几乎不含判别信息，multimodal 增益来自偏置/正则化效应，与 E1-F1 互证；接近 visual-only → 特征有信息但与视觉分支冗余编码同样内容，融合增益自然边际。

### E3 效率剖析：把"4B 负担"变成数字

**动机**（F-A）：论文无任何效率数据，而其远程筛查立论与 4B 在线依赖直接冲突；需实测量化。

**测量项**（分设备记录：GPU 型号 / CPU / 如有可能加移动端）：
1. 特征提取延迟：Qwen3-VL-4B 单图前向（batch 1 / batch 32），BF16 与 INT4（如可量化）分别记录；
2. 端到端推理延迟：(a) 视觉分支单独（3×ResNet50 + 融合 + MoE）；(b) 完整流水线（含 Qwen 特征提取）；报告比值；
3. 显存/内存峰值（两者分别记录）；
4. 训练时长摊销（一次性提取成本 + multimodal 训练）。

**判定**：(b)/(a) 延迟比 ≥5× 或显存比 ≥10×，且 E1 显示 F3 可替代 → "以 <5% 延迟换同等精度"成立。纯测量，无训练成本。

### E4 池化策略消融：稀释假说

**动机**（F-A、F-G）：若换读出方式即可显著提升特征质量，则原模块的问题不是"4B 没用"而是"读出方式丢信息"——修复比替换模型便宜。

**做法**：改造提取脚本，单次前向同时输出 4 种池化向量（避免重复计算）：
- (a) 全序列 masked mean（原版）；
- (b) 仅图像 token 的 mean（图像 token 位置由 processor 的 image token id 在 `input_ids` 中定位）；
- (c) 最后一个位置（assistant 生成起点）的隐状态；
- (d) 各层 image-token mean 的对比（`hidden_states` 已全量返回，顺带看浅层 vs 末层）。

**判定**：(b) 或 (c) 显著优于 (a) → 稀释假说成立。

### E5 提示词内容消融（TongueBench 归档提示词 A0/A1/A2）

**动机**（F-B）：原版先验是 8 句同义反复；利用外部基准 TongueBench 的归档提示词，把"先验内容"拆成两个可单因子检验的变量。全部使用归档原文，不引入自设计提示词。

**提示词来源**：TongueBench 补充材料（`materials/TongueBench_Supplementary_Material.PDF`，归档路径 `Tongue_Prompt_20260108/`）保留的三组提示词，逐字复用（完整原文见附录 B）：
- **A0_Control_Direct**（其结果表中称 DP）：仅 22 标签清单（固定顺序）+ JSON 输出示例，无定义知识；
- **A1_Knowledge_Direct**（KGP）：A0 + 四域（舌体形态/舌体颜色/舌苔质地/舌苔颜色）22 条标签定义；
- **A2_Knowledge_Reflection_Strict**（KGRP）：A1 + 反思流程（互斥冲突审查 7 条规则 + 保守回退）。

三组共享同一 SYSTEM_PROMPT（舌诊识别系统角色、仅舌部区域判断、JSON-only 输出、hard_case 保守模式、confidence 三级）。其协议的消融变量定义：A0→A1 仅增加"标签定义知识"；A1→A2 仅增加"反思流程模块"。

**配置表**（固定模型与提取协议，仅换提示词）：

| 配置 | 提示词内容 | 检验什么 |
|---|---|---|
| P0 | CycleTCM 原版 8 句同义反复（附录 A） | 锚点（复用复现的特征 JSON 与结果） |
| A0 | TongueBench 控制组（中文标签清单 + JSON schema） | 链内对照；vs P0 为语言/粒度/格式的综合效应 |
| A1 | A0 + 22 条标签定义 | 定义知识的净效应（单因子 A1−A0） |
| A2 | A1 + 反思流程/互斥/保守回退 | 流程指令的净效应（单因子 A2−A1） |

消融的内部对照由 A0→A1→A2 链条承担（A0 即其设计中的无知识对照组）。

**归因注意**：P0↔A0 之间同时差了语言（英→中）、标签粒度（8→22）、输出格式三个因子，只可做综合比较、不做单因子归因；严格的单因子结论只能取自 A0→A1→A2 链条。

**拼装方式**：P0 维持原版拼装（system = SYSTEM_PROMPT + TCM_PRIOR，user = 中性句 + 图像）；A0/A1/A2 按 TongueBench 归档结构拼装（system = 附录 B.1，user = 附录 B.2 / B.3 / B.4 + 图像）。

**两种执行模式**：
- **E5a 特征提取模式（主实验，与 CycleTCM 协议完全兼容）**：照旧单次前向 + 全序列 mean pooling → 特征 → 训练 multimodal 模型。注意：A2 的反思流程是"要求模型在生成中执行"的指令，在不生成的提取模式下只能经隐状态间接影响特征，效应预期被压缩。脏腑分支共享同一特征、会随特征变化，但 A 组提示词完全未提脏腑（P0 仅一句），脏腑任务的变化只能作"特征整体质量"的次级读出，不作单因子归因。
- **E5b 生成判读模式（可选对照，TongueBench 原生协议）**：让 Qwen3-VL-4B 逐图真正生成 JSON（推理超参固定、每组 ≥3 次生成、多数投票），按 TongueBench 强制指标评分（Parse Success Rate、Micro/Macro-F1、per-label P/R/F1 含支持度、Hamming Loss、Subset Accuracy、hard_case 子集性能与 Q-hard/D-hard 来源比例）。评分限定在 22 标签中可映射到 TongueDx 真值的 8 个标签子集（下表）；其余 14 个标签无真值，仅报告预测率不评分。

**标签映射（TongueBench 22 → TongueDx 8 证候）**：

| TongueDx | TongueBench 最近标签 | 映射质量 |
|---|---|---|
| TonguePale | 舌色淡白 | 直接 |
| TipSideRed | 舌色红绛 | **部分**：TongueBench 标签无位置信息（舌尖/舌边），"红绛"的判据也与"局部红"不同，存在系统性偏差，须在报告显式声明 |
| Spot | 舌上点刺 | 直接 |
| Ecchymosis | 舌上瘀斑 | 直接 |
| Crack | 舌面裂纹 | 直接 |
| ToothMark | 舌边齿痕 | 直接 |
| FurThick | 苔厚 | 直接 |
| FurYellow | 苔色黄 | 直接 |

**预期效应量**（TongueBench 已发表参考值，用于设定统计功效）：其 2,694 图基准、九个模型均值下 DP→KGP→KGRP 的 Micro-F1 = 47.57% → 47.68% → 48.51%；KGRP−DP = +0.94pp（图像 bootstrap 95% CI [+0.77, +1.10]）；KGP−DP 仅 Macro-F1 +0.81pp；且效应强模型依赖（GPT-5.3、Qwen3-Omni 的 KGRP−DP Micro-F1 为负）。即 **A1 vs A0 预期 ≈ 0~微弱，A2 vs A1 预期小幅为正但不保证**。要在 TongueDx 上检出 ~1pp 级效应，3 种子可能不足：提示词消融建议 5 种子，或以图像级 bootstrap 为主。

**判定规则**（沿用 TongueBench 协议原文）：每组 ≥3 次独立运行并报告均值±std；对 A1 vs A0、A2 vs A1 做配对显著性检验（bootstrap 95% CI）；**CI 跨零不得宣称"有效提升"**。固定项照其协议：标签集合与顺序、JSON schema、推理超参（temperature/top_p/max_tokens/seed）、数据划分、评测与后处理脚本、重试策略与 JSON 解析器全部固定；记录 prompt 版本号与时间戳。

**成本**：E5a = 新特征提取 3 套（A0/A1/A2，各 5,109 次前向；P0 复用复现时的 `all_features.json`）+ 3 配置 × ≥3 种子训练；E5b ≈ 5,109 × 3 提示词 × 3 次生成 ≈ 4.6 万次调用，成本高，建议先用 270–895 图子集跑通再决定是否全量。

**结果解读树**：
- A0 ≈ A1 ≈ A2 ≈ P0（E5a）：提示词家族与内容对特征质量整体无净效应 → 与 E1-F1/F2 互证；
- A2 > A0 显著（E5a）：流程指令在不生成时也有效，值得单独分析（可能经 JSON schema 提升了特征的结构度）；
- E5b 中 A2 > A0 显著而 E5a 中不可分：差异来自"生成 vs 不生成"——直接证明 CycleTCM 的特征提取用法丢掉了 MLLM 的核心能力（生成式判读），与 F-A 互证。

### E6 统计严谨性复测

**动机**（F-F）：论文"mean%"未定义运行次数与方差，且类别不平衡使 Acc 虚高；±1% 的单次波动足以吞掉声称增益的一半。

**做法**：
- 同一最优配置跑 ≥3（建议 5）个种子，报告 mean±std，并明确定义"mean%"；
- 输出 per-label F1 完整表（Spleen 等退化标签单独标注）；
- 受试者级 bootstrap 置信区间（沿用论文自身的 bootstrap 实践）比较 visual-only vs multimodal vs E1-F3；
- 显著性：McNemar 检验或 bootstrap 差值 CI——若 multimodal 对 visual-only 的增益 CI 跨零，则 MLLM 部分的主贡献在统计上不成立。

### E7 损失声明验证

**动机**（F-E）：论文损失声明与开源代码不一致，需确定论文报告数字对应的代码状态。

**做法**：把 Lovász-Softmax（多标签设定下对每个标签的 IoU 代理取均值，用官方实现）按 α=0.7 加回 weighted BCE，与 BCE-only 各跑 3 种子。

**解读**：两者不可分 → 论文的损失描述属于无害的写作瑕疵；可分且方向明确 → 论文数字对应的代码状态与开源版不一致，复现报告需显式注明以哪个为准。

---

## 3. 优先级与建议顺序

| 优先级 | 实验 | 额外训练成本 | 额外提取成本 | 结论价值 |
|---|---|---|---|---|
| P0 | E1-F1/F2（常量/随机对照） | 2 配置 × 3 种子 | 无（纯张量） | 高：直接检验"MLLM=偏置"假说 |
| P0 | E2 MLLM-only | 1 次 MLP 训练 | 无（JSON 已有） | 高：论文缺失的关键消融 |
| P0 | E3 效率剖析 | 无 | 少量前向即可 | 高：把定性判断变成数字 |
| P1 | E1-F3/F4（轻量 VLM 替换） | 2 配置 × 3 种子 | 5,109 前向 × 2 模型 | 高：核心可替换性证据 |
| P1 | E5 提示词消融（A0/A1/A2） | 3 配置 × ≥3 种子（P0 复用复现） | 5,109 前向 × 3 | 中高：先验内容价值，A0→A1→A2 单因子归因 |
| P2 | E4 池化消融 | 3–4 配置 | 1 次前向可得全部（建议合并进 E1 提取） | 中：修复方案 |
| P2 | E6 多种子复测 | 已有配置 × 5 种子 | 无 | 中：统计效力 |
| P3 | E7 损失验证 | 1 配置 × 3 种子 | 无 | 中：写作规范性 |

建议顺序：E2 → E1-F1/F2 → E3 → E1-F3/F4 → E5 → E4 → E6 → E7。前四步即可形成"MLLM 模块价值"的完整证据链。

---

## 4. 结果记录表模板

**E1 特征来源矩阵**（每个格子：证候 Acc / F1 与脏腑 Acc / F1，3 种子均值±std）

| 配置 | 来源 | 参数量 | 证候 Acc | 证候 F1 | 脏腑 Acc | 脏腑 F1 | Δ vs F5 |
|---|---|---|---|---|---|---|---|
| F0 | 无 | 0 | | | | | |
| F1 | 常量向量 | ~0 | | | | | |
| F2 | 随机冻结编码器 | ~25M | | | | | |
| F3 | SigLIP / Chinese-CLIP | 0.2–0.4B | | | | | |
| F4 | Qwen2-VL-2B | 2B | | | | | |
| F5 | Qwen3-VL-4B（原版） | 4B | | | | | — |

**E3 效率**：特征提取 ms/图（GPU/CPU、BF16/INT4）、端到端 ms/图（含/不含 Qwen）、峰值显存 GB、延迟比 (b)/(a)。

**E5 提示词**：E5a——P0 / A0 / A1 / A2 × {证候(8), 脏腑(5)} 平均 Acc/F1，重点报告单因子对比 A1−A0、A2−A1 的配对 bootstrap 95% CI 与 per-label F1 变化最大的标签；E5b（若做）——A0/A1/A2 在 8 个可映射标签上的 Micro/Macro-F1、Parse Success Rate、hard_case 比例及 Q-hard/D-hard 来源。

---

## 5. 代码地图

| 内容 | 位置 |
|---|---|
| 原版提示词三段（SYSTEM_PROMPT / TCM_PRIOR / USER_PROMPT） | `src/utils/mllm_feature_extract.py:14-32`；逐字转录见附录 A |
| 前向提取 + 全序列 mean pooling + JSON 落盘 | `mllm_feature_extract.py:126-179`（拼装逻辑 126-135） |
| MLLM 适配器（2560→2048） | `src/models/adapter.py`；`model_multimodal.py` 内同构副本 |
| MLLM 特征拼接进两分支 | `src/models/model_multimodal.py`（`CycleTCM.forward` 末段：`torch.cat([x_syndrome_all, mllm_feature], dim=1)`） |
| 骨干与区域混合（ResNet50、6ch/12ch→3ch） | `model_multimodal.py`（`resnet50_backbone_6ch/12ch`） |
| 双向 MoE（4 专家、top-2） | `model_multimodal.py`（`ConditionalTopKMoE` / `BidirectionalMoE`） |
| MLLM-only 基线（E2） | `src/models/model_mllm.py` + `src/train/train_model_mllm.py` |
| 后融合脚本（论文未报告） | `src/utils/late_fusion.py` |
| 训练损失（weighted BCE，E7 对照） | `src/train/train_model_multimodal.py` |
| 七视图预处理（body/edge 腐蚀环带、脏腑弧线分区） | `src/data_preprocessed/data_segment_regions.py` / `data_segment_organs.py` |

---

## 附录 A：CycleTCM 原版提示词（P0 配置用，逐字）

```text
[SYSTEM_PROMPT]
You are an experienced Traditional Chinese Medicine tongue-diagnosis expert. Based on the input tongue image, please determine the presence of the following tongue attributes: TonguePale, TipSideRed, Spot, Ecchymosis, Crack, ToothMark, FurThick, and FurYellow.
Based on these tongue manifestations, further assess whether the five organs, Heart, Lung, Spleen, Liver, and Kidney, may show abnormal tendencies.

[TCM_PRIOR]
TonguePale indicates the tongue is pale.
TipSideRed reflects the tip or sides of the tongue are red.
Spot denotes the presence of spots on the tongue.
Ecchymosis shows there is ecchymosis on the tongue.
Crack indicates there are cracks on the tongue.
ToothMark reflects the presence of tooth marks on the tongue.
FurThick describes the thickness of the tongue fur.
FurYellow indicates the tongue fur is yellow.

[USER_PROMPT]
Please analyze this tongue image according to your expertise and the instructions above.
```

拼装方式（`mllm_feature_extract.py:126-135`）：system 消息 = SYSTEM_PROMPT + "\n" + TCM_PRIOR；user 消息 = 图像 + USER_PROMPT。

## 附录 B：TongueBench 归档提示词原文（A0/A1/A2，逐字复用）

> 来源：`materials/TongueBench_Supplementary_Material.PDF`（*TongueBench: Performance of Prompt-Engineered Large Multimodal Models in Traditional Chinese Medicine Tongue Diagnosis* 补充材料第 1–6 页），归档路径 `Tongue_Prompt_20260108/`。原文声明"reproduced without changing its substantive wording"，本文同样逐字转录，仅还原 PDF 换行；协议文本描述的是设计意图，不代表历史调用参数均经验证（补充材料原注）。
> B.2–B.4 每个代码块即该组 user prompt 的**完整内容**（标签定义、反思流程、JSON 示例均已写入块内，无跨节引用），可整块复制直接使用；B.1 SYSTEM_PROMPT 是归档中独立的 system 消息文件，按 E5 的拼装方式（system = B.1，user = 对应块 + 图像）组合使用。

### B.1 SYSTEM_PROMPT（三组共享）

```text
你是一个舌诊图像识别系统，能够通过图像对中医舌诊做出精确识别。
任务：基于舌象图像中的可见视觉特征，对预定义 22 个标签进行判断。
全局约束：
1. 仅依据图像可见特征判断，只按格式输出结果，不得进行任何解释和说明。
2. 只允许舌部区域作为判断目标；不得依据嘴唇、牙齿、口腔、文字水印、边框、背景等作判断。
3. 输出必须且只能是一个合法 JSON 对象，不得输出额外文本、注释或 Markdown。
4. 顶层键顺序固定为：`labels`、`hard_case`、`confidence`。
5. `labels` 的值只允许 `0` 或 `1`；`hard_case` 只允许 `true` 或 `false`；`confidence` 只允许 `"high"`、`"medium"`、`"low"`。
6. 必须输出全部 22 个标签键，每个键仅出现一次。
7. `hard_case` 定义为"高难样本"，包含两类：图像质量困难 或 标签判别争议；不等价于"仅低质量图像"。
8. 证据不足、图像质量不佳或标签不稳定时，对应标签必须输出 `0`，不得猜测。
9. `confidence` 规则为：`"high"`：无明显质量问题，核心维度均可稳定判断。`"medium"`：存在轻度质量问题或局部不稳定，但整体可判。`"low"`：存在明显质量问题或多个维度不稳定。
10. `hard_case` 规则：若舌体主体可见面积明显不足（约低于 60%）或存在严重失焦/严重曝光异常/严重色偏导致核心维度无法判断，则输出进入保守模式：`labels` 全部置 `0`、`hard_case=true`、`confidence="low"`。
11. 示例仅用于格式演示，不可复用示例中的标签取值。
```

### B.2 A0_Control_Direct（完整原文，整块直接可用）

```text
标签集合（固定顺序）：
["舌嫩","舌老","舌胖","舌瘦","舌上点刺","舌面裂纹","舌边齿痕","舌上瘀斑","舌色淡白","舌色淡红","舌色红绛","舌色青紫","苔无","苔薄","苔厚","苔滑润","苔燥","苔腐腻","苔剥落","苔色白","苔色黄","苔色褐黑"]
输出 JSON 示例：
{
  "labels": {
    "舌嫩": 0,
    "舌老": 0,
    "舌胖": 1,
    "舌瘦": 0,
    "舌上点刺": 0,
    "舌面裂纹": 0,
    "舌边齿痕": 1,
    "舌上瘀斑": 0,
    "舌色淡白": 0,
    "舌色淡红": 1,
    "舌色红绛": 0,
    "舌色青紫": 0,
    "苔无": 0,
    "苔薄": 1,
    "苔厚": 0,
    "苔滑润": 0,
    "苔燥": 0,
    "苔腐腻": 0,
    "苔剥落": 0,
    "苔色白": 1,
    "苔色黄": 0,
    "苔色褐黑": 0
  },
  "hard_case": false,
  "confidence": "medium"
}
```

### B.3 A1_Knowledge_Direct（完整原文，整块直接可用）

```text
目标：增加标签定义知识。
## 标签定义
### 舌体形态
- 舌嫩：舌体较柔嫩，纹理细腻，表面相对光滑。
- 舌老：舌体纹理较粗，质地偏老，皱缩或粗糙感较明显。
- 舌胖：舌体宽大肥厚，边缘膨隆。
- 舌瘦：舌体瘦薄狭长，整体偏小。
- 舌上点刺：舌面可见点状或刺点样突起。
- 舌面裂纹：舌面可见裂沟或裂纹。
- 舌边齿痕：舌边可见牙齿压痕，呈波浪或锯齿样凹陷。
- 舌上瘀斑：局部紫暗色斑点或斑片，与周围区域有明显对比。
### 舌体颜色
- 舌色淡白：整体偏白偏淡，红色成分不足。
- 舌色淡红：淡红或粉红，色泽相对均匀。
- 舌色红绛：整体呈较明显红色至深红色。
- 舌色青紫：整体或明显区域呈青紫、紫暗色调。
### 舌苔质地
- 苔无：舌面基本无可辨苔层，以舌质外露为主。
- 苔薄：苔层薄，舌质可透见。
- 苔厚：苔层较厚，较大范围遮蔽舌质。
- 苔滑润：苔面津润，光泽或湿润感明显。
- 苔燥：苔面干燥少津，欠光泽，可伴粗糙或干裂感。
- 苔腐腻：苔质黏腻或腐松，呈腻滞或豆渣样外观。
- 苔剥落：苔层局部缺失，露出下方舌质。
### 舌苔颜色
- 苔色白：苔色白或乳白。
- 苔色黄：苔色黄或黄褐。
- 苔色褐黑：苔色褐至灰黑。
输出 JSON 示例：
{
  "labels": {
    "舌嫩": 0,
    "舌老": 0,
    "舌胖": 1,
    "舌瘦": 0,
    "舌上点刺": 0,
    "舌面裂纹": 0,
    "舌边齿痕": 1,
    "舌上瘀斑": 0,
    "舌色淡白": 0,
    "舌色淡红": 1,
    "舌色红绛": 0,
    "舌色青紫": 0,
    "苔无": 0,
    "苔薄": 1,
    "苔厚": 0,
    "苔滑润": 0,
    "苔燥": 0,
    "苔腐腻": 0,
    "苔剥落": 0,
    "苔色白": 1,
    "苔色黄": 0,
    "苔色褐黑": 0
  },
  "hard_case": false,
  "confidence": "medium"
}
```

### B.4 A2_Knowledge_Reflection_Strict（完整原文，整块直接可用）

> 注：归档文件名为 A2_Knowledge_Reflection.md，文档内部标题为 A2_Knowledge_Reflection_Strict，转录保持原样。

```text
目标：引入严格反思流程（初判 -> 反证审查 -> 冲突审查 -> 保守回退 -> 难例判定）。
## 标签定义
### 舌体形态
- 舌嫩：舌体较柔嫩，纹理细腻，表面相对光滑。
- 舌老：舌体纹理较粗，质地偏老，皱缩或粗糙感较明显。
- 舌胖：舌体宽大肥厚，边缘膨隆。
- 舌瘦：舌体瘦薄狭长，整体偏小。
- 舌上点刺：舌面可见点状或刺点样突起。
- 舌面裂纹：舌面可见裂沟或裂纹。
- 舌边齿痕：舌边可见牙齿压痕，呈波浪或锯齿样凹陷。
- 舌上瘀斑：局部紫暗色斑点或斑片，与周围区域有明显对比。
### 舌体颜色
- 舌色淡白：整体偏白偏淡，红色成分不足。
- 舌色淡红：淡红或粉红，色泽相对均匀。
- 舌色红绛：整体呈较明显红色至深红色。
- 舌色青紫：整体或明显区域呈青紫、紫暗色调。
### 舌苔质地
- 苔无：舌面基本无可辨苔层，以舌质外露为主。
- 苔薄：苔层薄，舌质可透见。
- 苔厚：苔层较厚，较大范围遮蔽舌质。
- 苔滑润：苔面津润，光泽或湿润感明显。
- 苔燥：苔面干燥少津，欠光泽，可伴粗糙或干裂感。
- 苔腐腻：苔质黏腻或腐松，呈腻滞或豆渣样外观。
- 苔剥落：苔层局部缺失，露出下方舌质。
### 舌苔颜色
- 苔色白：苔色白或乳白。
- 苔色黄：苔色黄或黄褐。
- 苔色褐黑：苔色褐至灰黑。
## 反思流程：
1. 冲突审查：执行互斥一致性检查。
   - `舌嫩` 与 `舌老` 不可同时为 `1`。
   - `舌胖` 与 `舌瘦` 不可同时为 `1`。
   - `苔薄` 与 `苔厚` 不可同时为 `1`。
   - `苔滑润` 与 `苔燥` 不可同时为 `1`。
   - `苔无` 不可与 `苔薄`、`苔厚`、`苔滑润`、`苔燥`、`苔腐腻`、`苔剥落`、`苔色白`、`苔色黄`、`苔色褐黑` 同时为 `1`。
   - `舌色淡白`、`舌色淡红`、`舌色红绛` 不可同时为1。
   - `舌色青紫` 可与 `舌色淡白`、`舌色淡红`、`舌色红绛`三者之一共存，也可单独出现。
2. 保守回退：
   - 冲突时仅保留证据最明确者；无法稳定排序时，冲突标签全部置 `0`。
   - 仅由单一弱证据支撑的阳性标签置 `0`。
输出 JSON 示例：
{
  "labels": {
    "舌嫩": 0,
    "舌老": 0,
    "舌胖": 1,
    "舌瘦": 0,
    "舌上点刺": 0,
    "舌面裂纹": 0,
    "舌边齿痕": 1,
    "舌上瘀斑": 0,
    "舌色淡白": 0,
    "舌色淡红": 1,
    "舌色红绛": 0,
    "舌色青紫": 0,
    "苔无": 0,
    "苔薄": 1,
    "苔厚": 0,
    "苔滑润": 0,
    "苔燥": 0,
    "苔腐腻": 0,
    "苔剥落": 0,
    "苔色白": 1,
    "苔色黄": 0,
    "苔色褐黑": 0
  },
  "hard_case": false,
  "confidence": "medium"
}
```

### B.5 协议要点（User_Prompt/Benchmark_Protocol.md，实验执行时对照）

- 消融变量定义：A0→A1 仅增加"标签定义知识"；A1→A2 仅增加"反思流程模块"（反证审查 + 互斥冲突审查 + 保守回退 + hard_case 判定）。
- 固定项：22 标签集合与顺序、JSON schema（`labels`/`hard_case`/`confidence`）、推理超参（temperature、top_p、max_tokens、seed）、数据划分、评测脚本与后处理、重试策略与 JSON 解析器、hard_case 并集定义。
- 强制报告指标：Parse Success Rate / Invalid Output Rate、Micro-F1、Macro-F1、每标签 P/R/F1（含支持度）、Hamming Loss、Subset Accuracy、hard_case 子集 Micro/Macro-F1。
- 统计要求：每组 ≥3 次独立运行报均值±std；A1 vs A0、A2 vs A1 配对显著性检验（推荐 bootstrap 95% CI）；提升不显著（CI 跨 0）不得宣称"有效提升"。
- 结果记录：错例按"假阳性来源/假阴性来源/冲突回退样例"分类；A2 额外报告 hard_case 比例、hard_case 与非 hard_case 性能差、触发来源比例（Q-hard/D-hard/both）、冲突回退触发比例；全部实验记录 prompt 版本号与时间戳。
