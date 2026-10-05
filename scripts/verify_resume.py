"""Verify real CPU resume behavior and suite failure handling."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from utils.experiment import run_directory, sha256, write_json
import continue_reproduction_suite as controller


def equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    else:
        assert first == second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    args = parser.parse_args()
    output = run_directory(ROOT / 'outputs/verification', 'resume_reconciliation')
    features = args.features.resolve()
    common = [sys.executable, str(ROOT / 'src/train/reproduce.py'), '--model', 'mllm',
              '--device', 'cpu', '--workers', '0', '--batch-size', '32', '--limit', '64',
              '--init', 'none', '--seed', '73', '--min-delta', '1',
              '--mllm-features-file', str(features), '--output-dir', str(output)]

    def train(name, extra):
        before = set(output.iterdir())
        with (output / f'{name}.log').open('w') as log:
            subprocess.run(common + extra, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        created = [p for p in set(output.iterdir()) - before if p.is_dir()]
        assert len(created) == 1
        return created[0]

    first = train('initial', ['--epochs', '2'])
    originals = {p: sha256(p) for p in first.rglob('*') if p.is_file()}
    resumed = train('resumed', ['--resume', str(first / 'checkpoints/last.pt'), '--epochs', '3'])
    assert resumed != first and all(sha256(p) == digest for p, digest in originals.items())
    continuous = train('continuous', ['--epochs', '3'])
    a = torch.load(resumed / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    b = torch.load(continuous / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    for key in ('model', 'optimizer', 'scheduler', 'scaler', 'epoch', 'best_epoch', 'best_score', 'counter'):
        equal(a[key], b[key])
    assert a['best_epoch'] == 0  # Resume must carry the old best even without new improvement.
    assert sha256(first / 'checkpoints/best.pt') == sha256(resumed / 'checkpoints/best.pt')
    assert Path(a['best_path']) == resumed / 'checkpoints/best.pt'
    assert [row['epoch'] for row in a['history']] == [0, 1, 2]
    best = torch.load(resumed / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    assert 'optimizer' not in best
    results = {'source_files_unchanged': True, 'new_timestamp_directory': True,
               'resumed_matches_continuous_exactly': True, 'best_without_new_improvement': True}
    no_epoch = train('no_remaining_epochs', ['--resume', str(first / 'checkpoints/last.pt'), '--epochs', '2'])
    with (no_epoch / 'history.csv').open(newline='') as handle:
        assert [int(row['epoch']) for row in csv.DictReader(handle)] == [0, 1]
    assert sha256(no_epoch / 'checkpoints/best.pt') == sha256(first / 'checkpoints/best.pt')
    assert json.loads((no_epoch / 'status.json').read_text())['status'] == 'complete'
    results['resume_without_remaining_epochs_keeps_history'] = True

    for fail in (False, True):
        suite = run_directory(output, 'controller_failure' if fail else 'controller_resume')
        write_json(suite / 'fixed_config.json', {'mllm_features_file': str(features)})
        source = suite / first.name
        (source / 'checkpoints').mkdir(parents=True)
        (source / 'checkpoints/last.pt').write_bytes(b'stub')
        orphan = run_directory(suite, 'code_compat_mllm_seed73')
        write_json(suite / 'suite_status.json', {'status': 'running', 'queue': [
            {'model': 'mllm', 'seed': 73, 'status': 'pending'}]})
        launched = []

        def launch(command, **kwargs):
            launched.append(command)
            assert command[command.index('--resume') + 1] == str(source / 'checkpoints/last.pt')
            child = run_directory(suite, 'code_compat_mllm_seed73')
            write_json(child / 'status.json', {'status': 'failed' if fail else 'complete'})
            if not fail:
                write_json(child / 'summary.json', {})
            return subprocess.CompletedProcess(command, 7 if fail else 0)

        with patch.object(controller, 'ROOT', output), patch.object(controller, 'build_report'), \
             patch.object(controller.subprocess, 'run', side_effect=launch), \
             patch.object(sys, 'argv', ['verify', '--suite', str(suite), '--features', str(features)]):
            try:
                controller.main()
                assert not fail
            except RuntimeError:
                assert fail
        state = json.loads((suite / 'suite_status.json').read_text())
        assert len(launched) == 1 and state['status'] == ('failed' if fail else 'complete')
        assert (source / 'checkpoints/last.pt').is_file()
        assert str(orphan) in state['queue'][0]['orphaned']
        results['controller_failure_recorded' if fail else 'controller_resume_found_new_run'] = True
    write_json(output / 'verification.json', results)
    print(f'VERIFIED {output}')


if __name__ == '__main__':
    main()
