"""
Late fusion: fit linear regression on LLM and TCM predictions, then compare with ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import LABEL_DIR, OUTPUT_DIR

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, roc_auc_score

SYNDROME_LABELS = [
    "TonguePale",
    "TipSideRed",
    "Spot",
    "Ecchymosis",
    "Crack",
    "Toothmark",
    "FurThick",
    "FurYellow",
]
ORGAN_LABELS = ["Heart", "Lung", "Spleen", "Liver", "Kidney"]


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def align_by_id(
    llm_data: list[dict],
    tcm_data: list[dict],
    truth_data: list[dict],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    llm_map = {item["id"]: item for item in llm_data}
    tcm_map = {item["id"]: item for item in tcm_data}
    truth_map = {item["id"]: item for item in truth_data}

    common_ids = sorted(set(llm_map) & set(tcm_map) & set(truth_map))
    if not common_ids:
        raise ValueError("No overlapping sample ids across the three JSON files.")

    llm_preds = np.array([[llm_map[sid][label] for label in labels] for sid in common_ids], dtype=np.float64)
    tcm_preds = np.array([[tcm_map[sid][label] for label in labels] for sid in common_ids], dtype=np.float64)
    truth = np.array([[truth_map[sid][label] for label in labels] for sid in common_ids], dtype=np.float64)
    return llm_preds, tcm_preds, truth, common_ids


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    pred_binary = (predictions > threshold).astype(int)
    labels_binary = labels.astype(int)
    num_classes = labels_binary.shape[1]

    per_class_accuracy = []
    per_class_sensitivity = []
    per_class_specificity = []
    per_class_precision = []

    for c in range(num_classes):
        tp = np.sum((pred_binary[:, c] == 1) & (labels_binary[:, c] == 1))
        tn = np.sum((pred_binary[:, c] == 0) & (labels_binary[:, c] == 0))
        fp = np.sum((pred_binary[:, c] == 1) & (labels_binary[:, c] == 0))
        fn = np.sum((pred_binary[:, c] == 0) & (labels_binary[:, c] == 1))

        total = tp + tn + fp + fn
        per_class_accuracy.append((tp + tn) / total if total > 0 else 0.0)
        per_class_sensitivity.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        per_class_specificity.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
        per_class_precision.append(tp / (tp + fp) if tp + fp > 0 else 0.0)

    per_class_accuracy = np.array(per_class_accuracy)
    per_class_sensitivity = np.array(per_class_sensitivity)
    per_class_specificity = np.array(per_class_specificity)
    per_class_precision = np.array(per_class_precision)

    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels_binary,
        pred_binary,
        average=None,
        zero_division=0,
    )

    per_class_mcc = []
    for c in range(num_classes):
        try:
            per_class_mcc.append(matthews_corrcoef(labels_binary[:, c], pred_binary[:, c]))
        except ValueError:
            per_class_mcc.append(0.0)
    per_class_mcc = np.array(per_class_mcc)

    try:
        auc = roc_auc_score(labels_binary, predictions, average="macro")
    except ValueError:
        auc = 0.0

    return {
        "per_class_acc_mean": float(np.mean(per_class_accuracy)),
        "per_class_accuracy": per_class_accuracy,
        "per_class_f1": f1_per_class,
        "per_class_mcc": per_class_mcc,
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "per_class_precision_calc": per_class_precision,
        "f1": float(np.mean(f1_per_class)),
        "auc": float(auc),
        "mcc": float(np.mean(per_class_mcc)),
        "sensitivity_mean": float(np.mean(per_class_sensitivity)),
        "specificity_mean": float(np.mean(per_class_specificity)),
        "precision_calc_mean": float(np.mean(per_class_precision)),
    }


def fit_late_fusion(
    llm_preds: np.ndarray,
    tcm_preds: np.ndarray,
    truth: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, list[dict]]:
    num_samples, num_labels = truth.shape
    fused_preds = np.zeros((num_samples, num_labels), dtype=np.float64)
    coefficients = []

    for idx, label in enumerate(labels):
        x = np.column_stack([llm_preds[:, idx], tcm_preds[:, idx]])
        y = truth[:, idx]
        model = LinearRegression()
        model.fit(x, y)
        fused_preds[:, idx] = np.clip(model.predict(x), 0.0, 1.0)
        coefficients.append(
            {
                "label": label,
                "intercept": float(model.intercept_),
                "llm_weight": float(model.coef_[0]),
                "tcm_weight": float(model.coef_[1]),
            }
        )

    return fused_preds, coefficients


def print_metrics(title: str, metrics: dict, class_names: list[str]) -> None:
    print(f"\n{title}")
    print(
        f"  mean acc: {metrics['per_class_acc_mean']:.4f}, "
        f"mean F1: {metrics['f1']:.4f}, "
        f"AUC: {metrics['auc']:.4f}, "
        f"MCC: {metrics['mcc']:.4f}, "
        f"mean SEN: {metrics['sensitivity_mean']:.4f}, "
        f"mean SPE: {metrics['specificity_mean']:.4f}, "
        f"mean PRE: {metrics['precision_calc_mean']:.4f}"
    )
    for name, acc, f1, mcc, sen, spe, pre in zip(
        class_names,
        metrics["per_class_accuracy"],
        metrics["per_class_f1"],
        metrics["per_class_mcc"],
        metrics["per_class_sensitivity"],
        metrics["per_class_specificity"],
        metrics["per_class_precision_calc"],
    ):
        print(
            f"    {name:12s} acc={acc:.4f} f1={f1:.4f} mcc={mcc:.4f} "
            f"sen={sen:.4f} spe={spe:.4f} pre={pre:.4f}"
        )


def print_coefficients(title: str, coefficients: list[dict]) -> None:
    print(f"\n{title} linear regression coefficients")
    for item in coefficients:
        print(
            f"  {item['label']:12s} "
            f"intercept={item['intercept']:+.4f} "
            f"llm={item['llm_weight']:+.4f} "
            f"tcm={item['tcm_weight']:+.4f}"
        )


def evaluate_group(
    group_name: str,
    labels: list[str],
    llm_preds: np.ndarray,
    tcm_preds: np.ndarray,
    truth: np.ndarray,
) -> dict:
    fused_preds, coefficients = fit_late_fusion(llm_preds, tcm_preds, truth, labels)

    llm_metrics = calculate_metrics(llm_preds, truth)
    tcm_metrics = calculate_metrics(tcm_preds, truth)
    fused_metrics = calculate_metrics(fused_preds, truth)

    print(f"\n{'=' * 72}")
    print(f"{group_name} ({len(labels)} labels, {truth.shape[0]} samples)")
    print_metrics("LLM only", llm_metrics, labels)
    print_metrics("TCM only", tcm_metrics, labels)
    print_metrics("Late fusion (linear regression)", fused_metrics, labels)
    print_coefficients(group_name, coefficients)

    return {
        "labels": labels,
        "coefficients": coefficients,
        "llm": llm_metrics,
        "tcm": tcm_metrics,
        "fusion": fused_metrics,
        "fused_predictions": fused_preds,
    }


def run_late_fusion(
    llm_path: Path,
    tcm_path: Path,
    truth_path: Path,
) -> dict:
    llm_data = load_json(llm_path)
    tcm_data = load_json(tcm_path)
    truth_data = load_json(truth_path)

    llm_syndrome, tcm_syndrome, truth_syndrome, sample_ids = align_by_id(
        llm_data, tcm_data, truth_data, SYNDROME_LABELS
    )
    llm_organ, tcm_organ, truth_organ, organ_ids = align_by_id(
        llm_data, tcm_data, truth_data, ORGAN_LABELS
    )
    if sample_ids != organ_ids:
        raise ValueError("Syndrome and organ sample alignment mismatch.")

    print(f"Loaded {len(sample_ids)} aligned samples.")
    print(f"  LLM:   {llm_path}")
    print(f"  TCM:   {tcm_path}")
    print(f"  Truth: {truth_path}")

    syndrome_result = evaluate_group(
        "Syndrome",
        SYNDROME_LABELS,
        llm_syndrome,
        tcm_syndrome,
        truth_syndrome,
    )
    organ_result = evaluate_group(
        "Organ",
        ORGAN_LABELS,
        llm_organ,
        tcm_organ,
        truth_organ,
    )

    return {
        "sample_ids": sample_ids,
        "syndrome": syndrome_result,
        "organ": organ_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late fusion with linear regression for TCM + LLM predictions.")
    parser.add_argument(
        "--llm-json",
        type=Path,
        default=OUTPUT_DIR / "mllm/test_llm.json",
        help="Path to LLM prediction JSON.",
    )
    parser.add_argument(
        "--tcm-json",
        type=Path,
        default=OUTPUT_DIR / "visual/test_tcm.json",
        help="Path to TCM prediction JSON.",
    )
    parser.add_argument(
        "--truth-json",
        type=Path,
        default=LABEL_DIR / "test.json",
        help="Path to ground-truth JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_late_fusion(args.llm_json, args.tcm_json, args.truth_json)


if __name__ == "__main__":
    main()
