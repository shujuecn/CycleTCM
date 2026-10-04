# MLLM-Enhanced Region-Aware Bidirectional Evidence-Based Model for Tongue Diagnosis

📌 **MLLM-Enhanced Region-Aware Bidirectional Evidence-Based Model for Tongue Diagnosis (CycleTCM)**

Tongue diagnosis, a convenient and noninvasive traditional diagnostic method in Traditional Chinese Medicine (TCM), provides an important tool for early health screening. Tongue images not only reveal TCM syndrome patterns but also allow a preliminary assessment of relevant organ health. However, most existing methods face three major limitations: (i) syndrome patterns and organ states prediction are often treated as independent tasks, ignoring their coupled mechanisms; (ii) tongue region-dependent signs are insufficiently integrated with global tongue appearance, resulting in inadequate attention to salient local cues; and (iii) clinical priors and TCM knowledge are underutilized, constraining clinically grounded reasoning and interpretability.

To address these limitations, we propose **CycleTCM**, an MLLM-Enhanced Region-Aware Bidirectional Evidence-Based Model for tongue diagnosis. Specifically, first, an **Augmented Global-Local Feature Fusion (AGLFF)** module is introduced to reconcile holistic tongue context with regional cues by mutually refining global and local representations, strengthening region-sensitive evidence extraction. Second, an **Uncertainty-Weighted Bidirectional Mixture-of-Experts (UWBMoE)** module is designed to propagate syndrome-level and organ-level information, thereby stabilizing multi-task learning and cross-level reasoning. Moreover, a multimodal large language model (MLLM) is incorporated to enrich semantic representations and improve alignment between visual evidence and clinically meaningful concepts. Experiments demonstrate that the proposed approach outperforms state-of-the-art baselines on both syndrome patterns and organ states prediction tasks.

<p align="center"><img src="figures/framework.png" width="800"/></p>

## 📰News

**[NOTE]** The paper is accepted by **MICCAI 2026**.

## 💡Key Features

- A **region-aware multi-branch visual encoder** that jointly processes seven tongue views (whole, body, edge, and four organ-associated regions) for fine-grained evidence extraction.
- An **AGLFF module** that mutually refines global and local representations via cross-attention and gated fusion, strengthening region-sensitive sign detection.
- An **UWBMoE module** that performs bidirectional syndrome↔organ information propagation with uncertainty weighting, stabilizing coupled multi-task learning.
- **MLLM-enhanced multimodal fusion** using [Qwen3-VL-4B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) to align visual evidence with TCM clinical priors.
- Joint prediction of **8 syndrome attributes** and **5 organ states** with weighted BCE loss and comprehensive evaluation.

## 🛠Setup

**Tips A**: We test the framework using PyTorch ≥ 2.0 with CUDA support. A GPU with sufficient memory is recommended for MLLM feature extraction and multimodal training.

**Tips B**: Download the Qwen3-VL backbone before MLLM feature extraction.

The paper analysis, implementation audit, experiment targets, and staged execution plan are documented in [the reproduction plan](docs/reproduction_plan.md).

## 📚Data Preparation

Raw data defaults to `~/Documents/NuanWorkSpace/Datasets/TongueDx2/release`, containing `list/` CSV splits and `seg/` tongue images. Generated data and experiment outputs live outside `src/`:

```text
data/
  processed/CycleTCM/
    pp/                         # Segmented images before resizing
    images/                     # Whole tongue images
    images_body/, images_edge/
    images_heart_lung/, images_kidney/, images_liver/, images_spleen/
    feature_all_encoded.json    # Labels and relative paths to the seven views
    labels/json/                # train_dataset.json, val_dataset.json, test.json
  features/all_features.json    # Pooled Qwen vectors
outputs/
  visual/, multimodal/, mllm/   # Separate training run directories
    train.log                   # Training and evaluation log
    results.log                 # Per-class test metrics
    checkpoints/                # Reserved checkpoint directory
  feature_extraction/           # Feature extraction logs
```

The defaults are defined in `src/utils/paths.py` relative to the repository, so scripts can run from the repository root, `src/`, or another working directory. `data/` and `outputs/` are excluded from Git.

**Step 1 — Prepare all seven views and the split manifests.** Run from the repository root:

```bash
python3 prepare_data.py
# Optional: --raw-data-dir /path/to/TongueDx2/release --output-dir /path/to/processed --workers 8
```

This applies body/edge segmentation and organ-associated segmentation, then resizes the seven training views to 224×224. Image paths inside the manifest stay relative to the processed dataset directory.

**Step 2 — Extract MLLM features (for multimodal training).**

```bash
mkdir -p outputs/feature_extraction
python3 src/utils/mllm_feature_extract.py > outputs/feature_extraction/extract.log 2>&1
# Optional: --images-dir /path/to/processed/images --output /path/to/all_features.json --model-dir /path/to/Qwen
```

Existing generated images, split labels, and feature JSONs have been moved into these locations without regenerating them. Previous visual training logs are archived under `outputs/visual/legacy/`; the previous extraction log is under `outputs/feature_extraction/legacy/`. The original `/tmp` logs are also retained.


## ⏳Training the Model

The examples below run from the repository root. Use `--data-dir` to select another processed dataset, `--feature-file` / `--label-dir` to override individual inputs, and `--output-dir` for a separate experiment. Multimodal and MLLM-only training also accept `--mllm-features-file`.

Each execution creates a new `YYYYMMDD_HHMMSS_microseconds_...` directory under `--output-dir` (Asia/Shanghai). The shared trainer saves `best.pt`, `last.pt`, configuration, environment, history, per-image validation/test predictions and per-class metrics. See [the execution record](docs/reproduction_progress.md) for the fixed protocol and current progress.

### Visual Model (AGLFF + UWBMoE)

Train the visual-only CycleTCM using seven regional tongue images:

```bash
uv run --no-sync python src/train/train_model_visual.py --config configs/reproduction/code_compat.json
# Engineering smoke only: --epochs 1 --limit 32 --init none
```

### Multimodal Model (AGLFF + UWBMoE + MLLM)

Train the full CycleTCM with Qwen3-VL features fused at the representation level:

```bash
uv run --no-sync python src/train/train_model_multimodal.py --config configs/reproduction/code_compat.json --mllm-features-file [VERIFIED_FEATURE_JSON]
```

### MLLM-Only Baseline

Train a lightweight MLP classifier on Qwen3-VL features alone:

```bash
uv run --no-sync python src/train/train_model_mllm.py --config configs/reproduction/code_compat.json --mllm-features-file [VERIFIED_FEATURE_JSON]
```

## 🎇Late Fusion Strategy

For late-fusion strategy of MLLM and visual model predictions, please refer to:

```bash
python3 src/utils/late_fusion.py \
    --llm-json [LLM_PREDICTIONS] \
    --tcm-json [TCM_PREDICTIONS] \
    --truth-json [GROUND_TRUTH]
```
