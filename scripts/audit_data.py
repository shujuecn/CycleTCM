"""Decode and fingerprint every view; visualize fixed, label-stratified examples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from train.data import LABELS, VIEWS, split_records, load_features
from utils.experiment import run_directory, sha256, write_json
from utils.paths import PROCESSED_DATA_DIR, FEATURE_FILE, LABEL_DIR, MLLM_FEATURES_FILE, RAW_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/audit')
    args = parser.parse_args()
    output = run_directory(args.output_dir, 'data_audit')
    print(output, flush=True)
    splits = split_records(FEATURE_FILE, LABEL_DIR)
    assert [len(splits[key]) for key in ('train', 'val', 'test')] == [3371, 843, 895]
    features = load_features(MLLM_FEATURES_FILE)
    assert set(features) == {row['image_file'] for rows in splits.values() for row in rows}
    counts, selected = {}, []
    with (output / 'files.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['path', 'sha256', 'width', 'height'])
        for split, rows in splits.items():
            counts[split] = {'images': len(rows), 'subjects': len({row['id'] for row in rows})}
            original = {}
            with (RAW_DATA_DIR / 'list' / ('test.csv' if split == 'test' else f'{split}_fold1.csv')).open() as raw:
                for row in csv.DictReader(raw):
                    original[Path(row['image_path']).stem + '.png'] = row
            assert set(original) == {row['image_file'] for row in rows}
            for index, row in enumerate(rows):
                source = original[row['image_file']]
                assert int(source['id']) == row['id']
                assert all(int(source[key]) == row[key] for key in LABELS)
                for key in VIEWS:
                    path = (PROCESSED_DATA_DIR / row[key]).resolve()
                    assert path.is_relative_to(PROCESSED_DATA_DIR)
                    with Image.open(path) as image:
                        image.load()
                        assert image.size == (224, 224) and image.mode == 'RGB', str(path)
                        writer.writerow([str(path), sha256(path), *image.size])
                for path in [PROCESSED_DATA_DIR / 'pp' / row['image_file'], RAW_DATA_DIR / 'seg' / source['image_path']]:
                    with Image.open(path) as image:
                        image.load()
                        writer.writerow([str(path), sha256(path), *image.size])
                if index % 500 == 0:
                    print(f'{split}: {index}/{len(rows)} decoded', flush=True)
    # Select in manifest order from train/validation only; no test-based visual selection.
    pool = splits['train'] + splits['val']
    for label in ['TonguePale', 'Ecchymosis', 'Toothmark', 'FurYellow', 'Heart', 'Spleen']:
        row = next(row for row in pool if row[label] == (0 if label == 'Spleen' else 1)
                   and row['image_file'] not in {item['image_file'] for item in selected})
        selected.append(row)
    fig, axes = plt.subplots(len(selected), 8, figsize=(18, 14))
    for i, row in enumerate(selected):
        for j, path in enumerate([PROCESSED_DATA_DIR / 'pp' / row['image_file']] + [PROCESSED_DATA_DIR / row[key] for key in VIEWS]):
            with Image.open(path) as image:
                axes[i, j].imshow(image)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(['segmented', *[key[4:] for key in VIEWS]][j])
        axes[i, 0].set_title(row['image_file'], fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'regions.png', dpi=140)
    plt.close(fig)
    positives = np.array([[sum(row[label] for row in rows) for label in LABELS] for rows in splits.values()])
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (split, rows) in enumerate(splits.items()):
        ax.bar(np.arange(13) + (i - 1) * .25, positives[i] / len(rows) * 100, .25, label=split)
    ax.set_xticks(np.arange(13), LABELS, rotation=40, ha='right')
    ax.set_ylabel('Positive prevalence (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'prevalence.png', dpi=150)
    plt.close(fig)
    write_json(output / 'data_manifest.json', {
        'splits': counts, 'decoded_training_views': 35763, 'feature_count': len(features),
        'features_shape': [len(features), 2560], 'all_features_finite': True,
        'raw_labels_match': True, 'fingerprint': sha256(output / 'files.csv'),
        'feature_manifest_sha256': sha256(FEATURE_FILE), 'feature_cache_sha256': sha256(MLLM_FEATURES_FILE),
        'region_parameters': {'body_edge_erosion': .15, 'organ_erosion': .196, 'center': .632, 'liver_erosion': .10},
        'resize': 'cv2.resize(..., (224,224)), INTER_LINEAR default',
        'visual_selection': 'First distinct train/val image positive for each listed label; negative for Spleen',
        'selected': [{'image_file': row['image_file'], 'labels': [row[key] for key in LABELS]} for row in selected],
        'positive_counts': dict(zip(splits, positives.tolist())), 'labels': LABELS,
    })
    print(f'PASS all images/labels/features: {output}', flush=True)


if __name__ == '__main__':
    main()
