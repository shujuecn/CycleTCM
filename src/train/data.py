"""Strict seven-view loading, preserving image identity within subject splits."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABELS = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark',
          'FurThick', 'FurYellow', 'Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
VIEWS = ['img_whole', 'img_edge', 'img_body', 'img_heart_lung', 'img_spleen',
         'img_liver', 'img_kidney']
SPLITS = {'train': 'train_dataset.json', 'val': 'val_dataset.json', 'test': 'test.json'}


def split_records(feature_file, label_dir):
    records = json.loads(Path(feature_file).read_text())
    assert len({row['image_file'] for row in records}) == len(records), 'Duplicate image identity'
    result, seen = {}, set()
    for split, filename in SPLITS.items():
        labels = json.loads((Path(label_dir) / filename).read_text())
        ids = {row['id'] for row in labels}
        assert not ids & seen, 'Subjects overlap between splits'
        seen |= ids
        by_subject = {}
        for row in labels:
            values = [row[key] for key in LABELS]
            assert all(value in (0, 1) for value in values)
            assert row['id'] not in by_subject or by_subject[row['id']] == values
            by_subject[row['id']] = values
        result[split] = [row for row in records if row['id'] in ids]
        assert len(result[split]) == len(labels), f'{split}: image count mismatch'
        for row in result[split]:
            assert [row[key] for key in LABELS] == by_subject[row['id']], 'Label mismatch'
    assert sum(map(len, result.values())) == len(records), 'Unassigned images'
    return result


def augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])


def transform_views(images, transform, synchronized):
    # Replay the same sampled geometry/erasing while consuming RNG once per sample.
    before = torch.get_rng_state()
    output = []
    for image in images:
        if synchronized:
            torch.set_rng_state(before)
        output.append(transform(image))
    return output


class TongueDataset(Dataset):
    def __init__(self, records, data_dir, model, features=None, training=False,
                 profile='code_compat', normalize=False):
        self.records = records
        self.data_dir = Path(data_dir).resolve()
        self.views = [] if model == 'mllm' else VIEWS[:1] if model == 'global' else VIEWS
        self.features = features
        self.transform = augmentation() if training else transforms.ToTensor()
        self.synchronized = training and profile == 'reviewed'
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalize else None
        if features is not None:
            for row in records:
                vector = features[row['image_file']]
                assert vector.shape == (2560,) and torch.isfinite(vector).all()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        images = []
        for key in self.views:
            path = (self.data_dir / row[key]).resolve()
            assert path.is_relative_to(self.data_dir), f'View outside data directory: {path}'
            with Image.open(path) as image:
                assert image.size == (224, 224), f'Invalid image size: {path}'
                images.append(image.convert('RGB'))
        tensors = transform_views(images, self.transform, self.synchronized)
        if self.normalize:
            tensors = [self.normalize(tensor) for tensor in tensors]
        batch = dict(zip(self.views, tensors))
        batch.update(subject_id=row['id'], image_file=row['image_file'],
                     labels=torch.tensor([row[key] for key in LABELS], dtype=torch.float32))
        if self.features is not None:
            batch['mllm_feature'] = self.features[row['image_file']]
        return batch


def load_features(path):
    rows = json.loads(Path(path).read_text())
    features = {row['image_file']: torch.tensor(row['qwen_feature'], dtype=torch.float32) for row in rows}
    assert len(features) == len(rows), 'Duplicate feature identity'
    return features


def positive_weights(records):
    ratios = np.array([[row[key] for key in LABELS] for row in records]).mean(axis=0)
    return torch.tensor(1 / np.maximum(ratios, 1e-6), dtype=torch.float32)
