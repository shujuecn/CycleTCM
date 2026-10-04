"""Default artifact locations, resolved independently of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = Path.home() / "Documents/NuanWorkSpace/Datasets/TongueDx2/release"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed/CycleTCM"
FEATURE_FILE = PROCESSED_DATA_DIR / "feature_all_encoded.json"
LABEL_DIR = PROCESSED_DATA_DIR / "labels/json"
MLLM_FEATURES_FILE = PROJECT_ROOT / "data/features/all_features.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
