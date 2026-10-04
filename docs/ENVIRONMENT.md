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

对应的特征抽取也已完成（`/tmp/mllm_extract.log`）：5109 张图全部处理，`Wrote 5109 records -> all_features.json`，
`last_hidden_states[-1]` 形状 `(1, 276, 2560)`，池化 2560 维。所以**多模态训练无需重新抽特征**。

### 路径不匹配的坑

`src/utils/mllm_feature_extract.py` 的默认路径写的是旧版 ModelScope 布局
`~/.cache/modelscope/hub/Qwen/Qwen3-VL-4B-Instruct`，**该目录不存在**。原因是 modelscope 1.40.1
改用了新布局：

```text
{$MODELSCOPE_CACHE 或 ~/.cache/modelscope}/{repo_type}s/{owner}--{name}/snapshots/{revision}
```

`src/utils/download_qwen_vl.py` 里传的 `cache_dir=~/.cache/modelscope/hub` 在 1.40.1 下已不再决定最终落盘位置，
故权重落到了 `~/.cache/modelscope/models/`。直接运行抽取脚本会报 `FileNotFoundError`，二选一绕过：

```bash
export QWEN3_VL_MODEL_DIR=~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master
# 或
uv run python src/utils/mllm_feature_extract.py --model-dir ~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master
```

权重缺失时用 `uv run python src/utils/download_qwen_vl.py` 重新下载（约 8 G）。

> 注：因特征已抽取完毕，第 4 节命令 2 仅在换权重或换图像预处理后才需重跑。

## 4. 常用命令

均在项目根目录执行。

```bash
# 1) 生成七视图 + 划分清单
uv run python prepare_data.py

# 2) 抽取 MLLM 特征（多模态训练前置步骤）
mkdir -p outputs/feature_extraction
uv run python src/utils/mllm_feature_extract.py > outputs/feature_extraction/extract.log 2>&1

# 3) 训练（三个入口，默认 200 epoch）
uv run python src/train/train_model_visual.py        # AGLFF + UWBMoE
uv run python src/train/train_model_multimodal.py   # + MLLM 融合
uv run python src/train/train_model_mllm.py         # MLLM-only 基线

# 4) 后融合
uv run python src/utils/late_fusion.py \
    --llm-json [LLM_PREDICTIONS] --tcm-json [TCM_PREDICTIONS] --truth-json [GROUND_TRUTH]
```

三个训练脚本通用参数：`--data-dir`、`--feature-file`、`--label-dir`、`--output-dir`、`--output-log`
（多模态与 mllm-only 另有 `--mllm-features-file`）。日志与 per-class 指标写入各自 `--output-dir`。

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