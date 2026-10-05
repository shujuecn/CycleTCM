"""Create fixed-rule GradCAM overlays from a saved visual/full checkpoint."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from train.data import LABELS, split_records, TongueDataset, load_features
from train.reproduce import build_model, model_inputs
from utils.experiment import run_directory, sha256, write_json


def choose_rows(predictions, labels):
    """Choose the first lexical correct positive/negative examples for two labels."""
    indexed = {row['image_file']: row for row in predictions}
    choices = []
    for label, expected in [('Crack', 1), ('TonguePale', 0)]:
        index = LABELS.index(label)
        candidates = []
        for name in sorted(indexed):
            row = indexed[name]
            actual = int(labels[name][index])
            predicted = int(row['probabilities'][index] > 0.5)
            if actual == expected and predicted == expected:
                candidates.append((name, label, actual, predicted))
        if not candidates:
            raise RuntimeError(f'No fixed-rule example found for {label}={expected}')
        choices.append(candidates[0])
    return choices


def overlay_cam(image, cam, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(cam, cmap='magma', alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed visual/full run directory')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/qualitative')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / 'config.json').read_text())
    if config['model'] not in {'visual', 'full'}:
        raise ValueError('GradCAM requires a visual or full run')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = build_model(config, initialize=False).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    predictions = [json.loads(line) for line in (run / 'predictions/test.jsonl').read_text().splitlines()]
    records = split_records(config['feature_file'], config['label_dir'])['test']
    by_name = {row['image_file']: row for row in records}
    labels = {row['image_file']: row['labels'] for row in predictions}
    selected = choose_rows(predictions, labels)
    features = load_features(config['mllm_features_file']) if config['model'] == 'full' else None
    dataset = TongueDataset(records, config['data_dir'], config['model'], features, training=False,
                            profile=config['profile'], normalize=config['normalize'])
    dataset_by_name = {row['image_file']: i for i, row in enumerate(records)}
    output = run_directory(args.output_dir, f'gradcam_{config["model"]}_seed{config["seed"]}')
    target_layer = model.backbone_whole.layer4[-1]
    activations, gradients = {}, {}

    def save_activation(_, __, value):
        activations['value'] = value

    def save_gradient(_, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    hooks = [target_layer.register_forward_hook(save_activation),
             target_layer.register_full_backward_hook(save_gradient)]
    metadata = {'run': str(run), 'checkpoint_sha256': sha256(run / 'checkpoints/best.pt'),
                'target_layer': 'backbone_whole.layer4[-1]',
                'selection_rule': 'lexically first correct Crack positive and TonguePale negative',
                'device': str(device), 'selected': []}
    try:
        for name, label, actual, predicted in selected:
            sample = dataset[dataset_by_name[name]]
            batch = {key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim > 0 else value
                     for key, value in sample.items()}
            batch['labels'] = batch['labels'].to(device)
            inputs = {key: value.to(device) for key, value in batch.items()
                      if key in {'img_whole', 'img_edge', 'img_body', 'img_heart_lung', 'img_spleen',
                                 'img_liver', 'img_kidney', 'mllm_feature'}}
            activations.clear(); gradients.clear(); model.zero_grad(set_to_none=True)
            syn, org = model(*model_inputs(inputs, config, device))
            index = LABELS.index(label)
            logit = (syn if index < 8 else org)[0, index if index < 8 else index - 8]
            logit.backward()
            activation = activations['value']
            gradient = gradients['value']
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            image = sample['img_whole'].permute(1, 2, 0).numpy().clip(0, 1)
            probability = float(torch.sigmoid((syn if index < 8 else org)[0, index if index < 8 else index - 8]).detach().cpu())
            filename = f'{name.replace("/", "_")}_{label}.png'
            overlay_cam(image, cam, f'{name} | {label} p={probability:.3f}', output / filename)
            metadata['selected'].append({'image_file': name, 'label': label, 'actual': actual,
                                         'predicted': predicted, 'probability': probability,
                                         'output': filename})
    finally:
        for hook in hooks:
            hook.remove()
    write_json(output / 'metadata.json', metadata)
    print(f'QUALITATIVE {output}')


if __name__ == '__main__':
    main()
