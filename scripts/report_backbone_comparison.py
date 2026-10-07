"""Verify fixed-seed visual/Qwen/MedGemma fits and report paired test effects."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from report_reproduction import predictions, paired_bootstrap
from train.evaluation import metrics
from utils.experiment import sha256, write_json

KEYS = ['syndrome_acc', 'syndrome_f1', 'organ_acc', 'organ_f1']


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-run', type=Path, required=True)
    parser.add_argument('--medgemma-run', type=Path, required=True)
    parser.add_argument('--visual-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {'visual': args.visual_run.resolve(), 'qwen_p0': args.qwen_run.resolve(),
             'medgemma_p0': args.medgemma_run.resolve()}
    summaries, provenance, rows, classes = {}, {}, [], []
    protocol = identity = data_reference = None
    for name, path in paths.items():
        summary = json.loads((path / 'summary.json').read_text())
        config = summary['config']
        assert json.loads((path / 'status.json').read_text())['status'] == 'complete'
        assert not summary['engineering_only'] and config['limit'] == 0 and config['seed'] == 42
        assert config['model'] == ('visual' if name == 'visual' else 'full')
        scientific = {k: config[k] for k in ('profile', 'seed', 'epochs', 'batch_size', 'precision',
                      'learning_rate', 'weight_decay', 'patience', 'min_delta', 'init', 'normalize',
                      'scheduler_monitor', 'limit', 'data_dir', 'feature_file', 'label_dir')}
        scientific['loss'] = config.get('loss', 'bce')
        if protocol is None:
            protocol = scientific
        assert scientific == protocol
        data = json.loads((path / 'data_manifest.json').read_text())
        shared_data = {k: v for k, v in data.items() if k != 'mllm_features_sha256'}
        if data_reference is None:
            data_reference = shared_data
        assert shared_data == data_reference
        indexed = predictions(path / 'predictions/test.jsonl')
        actual_identity = {k: (v['subject_id'], v['labels']) for k, v in indexed.items()}
        if identity is None:
            identity = actual_identity
        assert actual_identity == identity and len({v[0] for v in identity.values()}) == 895
        assert sha256(path / 'checkpoints/best.pt') == summary['checkpoint_sha256']
        assert all(r['checkpoint_sha256'] == summary['checkpoint_sha256'] for r in indexed.values())
        measured = metrics([r['labels'] for r in indexed.values()], [r['probabilities'] for r in indexed.values()])
        saved = json.loads((path / 'metrics/test.json').read_text())
        assert measured['per_class'] == saved['per_class']
        row = {'model': name, 'seed': 42, 'best_epoch': summary['best_epoch'], 'test_images': 895}
        for key in KEYS:
            task, metric = key.split('_')
            assert measured[task][metric] == saved[task][metric] == summary['test'][task][metric]
            row[key + '_percent'] = measured[task][metric] * 100
        rows.append(row)
        classes.extend({'model': name, **{k: v for k, v in c.items()
                       if not k.startswith(('paper_', 'delta_'))}} for c in measured['per_class'])
        provenance[name] = {'run': str(path), 'config': config,
                            'checkpoint_sha256': summary['checkpoint_sha256'],
                            'files_sha256': {f: sha256(path / f) for f in ('summary.json', 'config.json',
                                'data_manifest.json', 'metrics/test.json', 'predictions/test.jsonl',
                                'history.csv', 'environment.json')}}
        if name != 'visual':
            feature = Path(config['mllm_features_file'])
            assert sha256(feature) == data['mllm_features_sha256']
            metadata = json.loads((feature.parent / 'metadata.json').read_text())
            provenance[name]['feature_metadata'] = metadata
        summaries[name] = summary
    a = provenance['qwen_p0']['feature_metadata']
    b = provenance['medgemma_p0']['feature_metadata']
    for key in ('system_prompt', 'tcm_prior', 'user_prompt', 'prompt_sha256', 'images_dir',
                'pooling', 'add_generation_prompt', 'use_cache', 'dtype', 'seed', 'versions'):
        assert a[key] == b[key], f'P0 extraction protocol differs: {key}'
    comparisons = {}
    for first, second in [('qwen_p0', 'medgemma_p0'), ('visual', 'medgemma_p0')]:
        effect = paired_bootstrap(paths[first] / 'predictions/test.jsonl',
                                  paths[second] / 'predictions/test.jsonl', seed=20261007)
        effect['effect_pp'] = {key: (summaries[second]['test'][key.split('_')[0]][key.split('_')[1]] -
                                   summaries[first]['test'][key.split('_')[0]][key.split('_')[1]]) * 100
                               for key in KEYS}
        comparisons[second + '_minus_' + first] = effect
    write_csv(output / 'main_results.csv', rows)
    write_csv(output / 'per_class_results.csv', classes)
    write_json(output / 'paired_bootstrap.json', comparisons)
    write_json(output / 'source_manifest.json', provenance)
    (output / 'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, task in zip(axes, ('syndrome', 'organ')):
        values = [r[task + '_f1_percent'] for r in rows]
        ax.bar(['Visual', 'Qwen P0', 'MedGemma P0'], values)
        ax.set(title=task.capitalize() + '; seed 42', ylabel='Macro positive-class F1 (%)', ylim=(0, 100))
        for index, value in enumerate(values):
            ax.text(index, value + 1, f'{value:.2f}', ha='center')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'figures/backbone_f1.{suffix}', dpi=180)
    plt.close(fig)
    print(json.dumps({'output': str(output), 'results': rows, 'effects': comparisons}, ensure_ascii=False))


if __name__ == '__main__':
    main()
