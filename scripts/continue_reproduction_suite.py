"""Continue an interrupted fixed reproduction suite from its saved state."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from report_reproduction import build_report
from utils.experiment import sha256, write_json


def run_dirs(suite, model, seed):
    return sorted(suite.glob(f'*_code_compat_{model}_seed{seed}'))


def completed_run(path):
    return ((path / 'summary.json').is_file() and (path / 'status.json').is_file()
            and json.loads((path / 'status.json').read_text()).get('status') == 'complete')


def resumable_run(suite, model, seed):
    """Newest unfinished run for this cell with a resumable last.pt."""
    interrupted = [path for path in run_dirs(suite, model, seed)
                   if not completed_run(path) and (path / 'checkpoints/last.pt').is_file()]
    return interrupted[-1] if interrupted else None


def orphaned_runs(suite, model, seed, known):
    return [str(path) for path in run_dirs(suite, model, seed) if path not in known]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    fixed_config = suite / 'fixed_config.json'
    config = json.loads(fixed_config.read_text())
    if sha256(args.features.expanduser()) != sha256(config['mllm_features_file']):
        parser.error('--features differs from the fixed suite feature cache')
    report = ROOT / 'reports/reproduction' / suite.name
    report.mkdir(parents=True, exist_ok=True)
    state = json.loads((suite / 'suite_status.json').read_text())
    state['status'] = 'running'
    state.pop('pid', None)
    run_paths = []
    orphans = []
    for item in state['queue']:
        paths = run_dirs(suite, item['model'], item['seed'])
        complete = [path for path in paths if completed_run(path)]
        if complete:
            chosen = Path(item['run']) if item.get('run') and Path(item['run']) in complete else complete[-1]
            item.update(status='complete', run=str(chosen), returncode=0)
            run_paths.append(chosen)
            item['orphaned'] = orphaned_runs(suite, item['model'], item['seed'], run_paths)
            orphans += item['orphaned']
        else:
            item.update(status='pending')
            resume = resumable_run(suite, item['model'], item['seed'])
            item.update(resume=str(resume / 'checkpoints/last.pt') if resume else None,
                        orphaned=orphaned_runs(suite, item['model'], item['seed'], run_paths))
            orphans += item['orphaned']
    if orphans:
        print(f'WARNING {len(orphans)} orphaned run directories need review: {orphans}', flush=True)
    write_json(suite / 'suite_status.json', state)
    write_json(report / 'suite_status.json', state)
    build_report(run_paths, report)

    for item in state['queue']:
        if item['status'] == 'complete':
            continue
        if shutil.disk_usage(ROOT).free < 20 * 2**30:
            raise OSError('Less than 20 GiB free; cannot safely write full training checkpoints')
        command = [sys.executable, str(ROOT / 'src/train/reproduce.py'),
                   '--config', str(fixed_config), '--model', item['model'],
                   '--seed', str(item['seed']), '--output-dir', str(suite)]
        if item.get('resume'):
            command += ['--resume', item['resume']]
            print(f'RESUME {item["model"]} seed={item["seed"]} from {item["resume"]}', flush=True)
        item['status'] = 'running'
        before = set(run_dirs(suite, item['model'], item['seed']))
        log_path = suite / f'{item["model"]}_seed{item["seed"]}.log'
        started = time.monotonic()
        write_json(suite / 'suite_status.json', state)
        with log_path.open('a') as log:
            returncode = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
        paths = run_dirs(suite, item['model'], item['seed'])
        created = [path for path in paths if path not in before]
        if returncode or len(created) != 1 or not completed_run(created[0]):
            item.update(status='failed', returncode=returncode,
                        seconds=time.monotonic() - started,
                        error=f'Expected one completed new run; created={list(map(str, created))}')
            if len(created) == 1:
                item['run'] = str(created[0])
            state['status'] = 'failed'
            write_json(suite / 'suite_status.json', state)
            write_json(report / 'suite_status.json', state)
            raise RuntimeError(f'Failed run {item["model"]}/{item["seed"]}: {item["error"]}; returncode={returncode}')
        path = created[0]
        run_paths.append(path)
        item.update(status='complete',
                    returncode=returncode, seconds=time.monotonic() - started,
                    run=str(path), resume=None,
                    orphaned=orphaned_runs(suite, item['model'], item['seed'], run_paths))
        if (path / 'checkpoints/last.pt').exists():
            (path / 'checkpoints/last.pt').unlink()
        write_json(suite / 'suite_status.json', state)
        build_report(run_paths, report)
        write_json(report / 'suite_status.json', state)
    state['status'] = 'complete'
    write_json(suite / 'suite_status.json', state)
    write_json(report / 'suite_status.json', state)
    build_report(run_paths, report)
    print(f'COMPLETE SUITE {report}')


if __name__ == '__main__':
    main()
