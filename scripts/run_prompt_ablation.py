"""Run E5a's nine full-model fits with the existing formal training protocol."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.experiment import run_directory, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', nargs=3, type=Path, required=True, metavar='A0_A1_A2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/prompt_ablation')
    parser.add_argument('--resume-suite', type=Path)
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=1, help='Concurrent fits on the same GPU; 2 requires enough VRAM')
    args = parser.parse_args()
    features = dict(zip(['A0', 'A1', 'A2'], [p.resolve() for p in args.features]))
    for variant, path in features.items():
        rows = json.loads(path.read_text())
        assert len(rows) == len({r['image_file'] for r in rows}) == 5109
        assert json.loads((path.parent / 'metadata.json').read_text())['variant'] == variant
    suite = args.resume_suite.resolve() if args.resume_suite else run_directory(args.output_dir, 'E5a_suite')
    state_path = suite / 'suite_status.json'
    if args.resume_suite:
        state = json.loads(state_path.read_text())
        assert state['features_sha256'] == {k: sha256(v) for k, v in features.items()}
    else:
        state = {'status': 'running', 'features': {k: str(v) for k, v in features.items()},
                 'features_sha256': {k: sha256(v) for k, v in features.items()},
                 'queue': [{'variant': v, 'seed': s, 'status': 'pending'}
                           for s in (42, 43, 44) for v in features]}
        write_json(suite / 'fixed_config.json', json.loads((ROOT / 'configs/reproduction/code_compat.json').read_text()))
    state['status'], state['jobs'] = 'running', args.jobs
    pending = []
    for entry in state['queue']:
        if entry['status'] == 'complete':
            assert json.loads((Path(entry['run']) / 'status.json').read_text())['status'] == 'complete'
            continue
        if entry['status'] == 'running' and entry.get('pid'):
            try:
                os.kill(entry['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Existing trainer is still live: pid={entry["pid"]}')
        pending.append(entry)
    print(f'SUITE {suite}', flush=True)

    def start(entry):
        variant, seed = entry['variant'], entry['seed']
        parent = suite / f'{variant}_seed{seed}'
        parent.mkdir(exist_ok=True)
        previous = sorted(parent.glob('*/checkpoints/last.pt'))
        if previous:
            command = [sys.executable, str(ROOT / 'src/train/reproduce.py'), '--resume', str(previous[-1]),
                       '--output-dir', str(parent)]
        else:
            command = [sys.executable, str(ROOT / 'src/train/reproduce.py'), '--config', str(suite / 'fixed_config.json'),
                       '--model', 'full', '--seed', str(seed), '--mllm-features-file', str(features[variant]),
                       '--output-dir', str(parent)]
        before = set(parent.iterdir())
        entry['status'] = 'running'
        write_json(state_path, state)
        print(f'START {variant} seed={seed}', flush=True)
        with (suite / f'{variant}_seed{seed}.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry['pid'] = process.pid
        write_json(state_path, state)
        return entry, process, parent, before, time.monotonic()

    def finish(job):
        entry, process, parent, before, started = job
        variant, seed, returncode = entry['variant'], entry['seed'], process.returncode
        created = [p for p in set(parent.iterdir()) - before if p.is_dir() and (p / 'config.json').exists()]
        if len(created) != 1:
            raise RuntimeError(f'Expected one training run: {created}; returncode={returncode}')
        entry.update(run=str(created[0]), returncode=returncode, seconds=time.monotonic()-started,
                     status='complete' if returncode == 0 else 'failed')
        state['status'] = 'running' if returncode == 0 else 'failed'
        write_json(state_path, state)
        if returncode:
            raise RuntimeError(f'{variant} seed={seed} failed; see {suite / (variant + "_seed" + str(seed) + ".log")}')
        # Remove only completed resumable states, as in the original reproduction suite.
        last = created[0] / 'checkpoints/last.pt'
        if last.exists():
            last.unlink()
        print(f'COMPLETE {variant} seed={seed} seconds={entry["seconds"]:.1f}', flush=True)

    active = []
    try:
        while pending or active:
            for job in active.copy():
                if job[1].poll() is not None:
                    active.remove(job)
                    finish(job)
            while pending and len(active) < args.jobs:
                active.append(start(pending.pop(0)))
            if active:
                time.sleep(1)
    except BaseException as error:
        for job in active:
            if job[1].poll() is None:
                job[1].send_signal(signal.SIGINT)
        for job in active:
            job[1].wait()
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        write_json(state_path, state)
        raise
    state['status'] = 'complete'
    write_json(state_path, state)
    print(f'COMPLETE SUITE {suite}', flush=True)


if __name__ == '__main__':
    main()
