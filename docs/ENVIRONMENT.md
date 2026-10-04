# 环境配置与使用方法

## 1. 环境概览

用 uv 管理的虚拟环境，位于项目根目录 `.venv/`。**必须通过 `uv run` 执行脚本**，系统 `python3` 未安装依赖。

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.14 | `requires-python = ">=3.10,<3.13"` |
| torch | 2.6.0+cu124 | 匹配本机驱动 550.90.07（CUDA 12.4） |
| torchvision | 0.21.0+cu124 | ResNet-50 骨干 |
| transformers | 4.57.6 | 提供 `Qwen3VLForConditionalGeneration` |
| accelerate | 1.15.0 | `device_map="auto"` 必需 |
| modelscope | 1.40.1 | 下载 Qwen3-VL 权重 |
| opencv-python-headless | 5.0.0 | 数据分割用 cv2（headless 免 libGL 依赖） |
| numpy / scikit-learn | 2.5.3 / 1.9.1 | 指标计算 |
| matplotlib / pillow / tqdm | 3.11.2 / 12.3.0 / 4.70.1 | |

torch 与 torchvision 来自 `https://download.pytorch.org/whl/cu124`（在 `pyproject.toml` 的 `[[tool.uv.index]]` 中声明）。依赖版本由 `uv.lock` 锁定。

## 2. 安装 / 校验

```bash
uv sync --locked        # 按 uv.lock 创建/更新 .venv
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

首次安装约 5.8 G（`.venv`）。已验证：L20（sm_89）前向通过，三个模型 visual / multimodal / mllm-only 输出均为 `[B,8]` 与 `[B,5]`。

## 3. MLLM 权重

**VL-4B 权重已下载完成**，来源为 ModelScope（不是 HuggingFace）。`/tmp/qwen4b_download.log` 证据：

```text
INFO | modelscope_hub.download | Downloading 15 files from Qwen/Qwen3-VL-4B-Instruct@master
Downloading: 100%|##########| 15/15 [12:36<00:00, 50.43s/file]
/home/ps/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master
```

即权重位于（8.3 G）：

```text
~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master/
```

旧特征缓存已存在，但本次检查发现 Transformers 4.57.6 与旧提取版本 5.17.0 的输出明显不同，不能直接认定数值一致。当前锁定环境已全量重提 5109 图，新文件为：

```text
data/features/20261005_041211_284070_qwen_features/all_features.json
```

该时间戳目录保存权重、prompt、逐图输入/token 哈希和版本；旧缓存未覆盖。详情见 [执行记录](reproduction_progress.md)。提取脚本已使用正确的 ModelScope 默认目录。

权重缺失时用 `uv run python src/utils/download_qwen_vl.py` 重新下载（约 8 G）。

> 特征抽取与分类训练顺序运行；模型、processor、prompt 或依赖版本发生变化时，先做小样本数值核查。

## 4. 常用命令

均在项目根目录执行。

```bash
# 1) 生成七视图 + 划分清单
uv run --no-sync python scripts/prepare_data.py

# 2) 抽取 MLLM 特征（多模态训练前置步骤）
uv run --no-sync python src/utils/mllm_feature_extract.py

# 3) 训练（三个入口，默认 200 epoch）
uv run python src/train/train_model_visual.py        # AGLFF + UWBMoE
uv run python src/train/train_model_multimodal.py   # + MLLM 融合
uv run python src/train/train_model_mllm.py         # MLLM-only 基线

# 4) 后融合
uv run python src/utils/late_fusion.py \
    --llm-json [LLM_PREDICTIONS] --tcm-json [TCM_PREDICTIONS] --truth-json [GROUND_TRUTH]
```

三个入口共享 `src/train/reproduce.py`：支持 `--config`、`--seed`、`--epochs`、`--batch-size`、`--precision`、`--resume`、`--evaluate` 与数据路径参数。`--output-dir` 为父目录，每次执行自动创建时间戳子目录。metrics 为结构化 JSON/CSV，不再使用旧 `--output-log`。multimodal / mllm 训练请显式传入经过核查的新 `--mllm-features-file`。

## 5. 路径默认值

定义在 `src/utils/paths.py`，相对仓库根解析，故从仓库根、`src/` 或其他目录运行均可。

| 变量 | 指向 |
|---|---|
| `PROCESSED_DATA_DIR` | `data/processed/CycleTCM/` |
| `FEATURE_FILE` | `data/processed/CycleTCM/feature_all_encoded.json` |
| `MLLM_FEATURES_FILE` | `data/features/all_features.json` |

## 6. 常用环境变量

| 变量 | 作用 |
|---|---|
| `QWEN3_VL_MODEL_DIR` | 覆盖 MLLM 权重目录（见第 3 节） |
| `MODELSCOPE_CACHE` | 覆盖 ModelScope 缓存根目录（modelscope ≥1.38 生效；旧的 `MS_CACHE_HOME` 已失效） |
| `HF_HOME` | 覆盖 HuggingFace 缓存根目录（本项目 MLLM 权重不走 HF） |

## 7. 复现环境

```bash
uv sync --locked                                  # 原样还原
uv lock --upgrade-package transformers             # 单独升级某包
uv pip list                                       # 查看实际安装版本
```