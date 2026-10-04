"""Per-label binary metrics and task macro averages (positive-class F1)."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from train.data import LABELS
from utils.experiment import write_json

PAPER_ACC = [87.93, 74.64, 81.23, 90.06, 86.15, 77.77, 97.54, 93.07, 73.41, 70.61, 99.33, 77.88, 79.22]
PAPER_F1 = [41.94, 70.48, 81.12, 35.04, 91.81, 82.74, 98.75, 80.13, 69.64, 76.45, 99.66, 82.84, 84.75]


def metrics(truth, probabilities):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities)
    assert truth.shape == probabilities.shape and truth.ndim == 2 and truth.shape[1] == 13
    assert len(truth) and np.isfinite(probabilities).all()
    assert np.isin(truth, [0, 1]).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    prediction = probabilities > .5
    per_class = []
    for column, label in enumerate(LABELS):
        actual, predicted = truth[:, column].astype(bool), prediction[:, column]
        tp, tn = int((actual & predicted).sum()), int((~actual & ~predicted).sum())
        fp, fn = int((~actual & predicted).sum()), int((actual & ~predicted).sum())
        divide = lambda a, b: a / b if b else 0.
        acc = (tp + tn) / len(truth)
        f1 = divide(2 * tp, 2 * tp + fp + fn)
        auc = float(roc_auc_score(actual, probabilities[:, column])) if actual.any() and (~actual).any() else None
        mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        per_class.append({'label': label, 'task': 'syndrome' if column < 8 else 'organ',
                          'acc': acc, 'f1': f1, 'sen': divide(tp, tp + fn),
                          'pre': divide(tp, tp + fp), 'spe': divide(tn, tn + fp),
                          'mcc': divide(tp * tn - fp * fn, mcc_denominator), 'auc': auc,
                          'auc_reason': None if auc is not None else 'Only one ground-truth class',
                          'positive': tp + fn, 'negative': tn + fp,
                          'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                          'paper_full_acc_percent': PAPER_ACC[column], 'paper_full_f1_percent': PAPER_F1[column],
                          'delta_acc_pp': acc * 100 - PAPER_ACC[column], 'delta_f1_pp': f1 * 100 - PAPER_F1[column]})
    result = {'samples': len(truth), 'threshold': '>0.5', 'per_class': per_class}
    for task, rows in [('syndrome', per_class[:8]), ('organ', per_class[8:])]:
        result[task] = {key: float(np.mean([row[key] for row in rows]))
                        for key in ['acc', 'f1', 'sen', 'pre', 'spe', 'mcc']}
        aucs = [row['auc'] for row in rows if row['auc'] is not None]
        result[task]['auc'] = float(np.mean(aucs)) if aucs else None
        result[task]['auc_defined_classes'] = len(aucs)
    result['selection_acc'] = (result['syndrome']['acc'] + result['organ']['acc']) / 2
    return result


def save_evaluation(output, split, records, logits, checkpoint_hash):
    output = Path(output)
    logits = np.asarray(logits, dtype=np.float32)
    assert len(records) == len(logits)
    assert len({row['image_file'] for row in records}) == len(records)
    probabilities = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -700, 700)))
    truth = [[row[label] for label in LABELS] for row in records]
    result = metrics(truth, probabilities)
    result['checkpoint_sha256'] = checkpoint_hash
    predictions = output / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    with (predictions / f'{split}.jsonl').open('w') as handle:
        for row, scores, probs, labels in zip(records, logits, probabilities, truth):
            handle.write(json.dumps({'subject_id': row['id'], 'image_file': row['image_file'],
                                     'split': split, 'labels': labels, 'label_order': LABELS,
                                     'logits': scores.tolist(), 'probabilities': probs.tolist(),
                                     'checkpoint_sha256': checkpoint_hash}, allow_nan=False) + '\n')
    write_json(output / 'metrics' / f'{split}.json', result)
    with (output / 'metrics' / f'{split}_per_class.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=result['per_class'][0].keys())
        writer.writeheader()
        writer.writerows(result['per_class'])
    return result
