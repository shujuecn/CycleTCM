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
from utils.experiment import write_json

QUEUE = [('global', 42), ('mllm', 42), ('visual', 42), ('full', 42),
         ('B', 42), ('BA', 42), ('BU', 42), ('BM', 42), ('B', 43),
         ('visual', 43), ('full', 43), ('B', 44), ('visual', 44), ('full', 44)]


def run_dirs(suite, model, seed):
    return sorted(suite.glob(f'*_code_compat_{model}_seed{seed}'))


def resumable_run(suite, model, seed):
    """Newest run for this cell that stopped without summary.json but kept a resumable last.pt."""
    interrupted = [path for path in run_dirs(suite, model, seed)
                   if not (path / 'summary.json').is_file() and (path / 'checkpoints/last.pt').is_file()]
    return interrupted[-1] if interrupted else None


def orphaned_runs(suite, model, seed, known):
    return [str(path) for path in run_dirs(suite, model, seed) if path not in known]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    report = ROOT / 'reports/reproduction' / suite.name
    report.mkdir(parents=True, exist_ok=True)
    state = json.loads((suite / 'suite_status.json').read_text())
    state['status'] = 'running'
    state.pop('pid', None)
    run_paths = []
    orphans = []
    for item in state['queue']:
        paths = run_dirs(suite, item['model'], item['seed'])
        complete = [path for path in paths if (path / 'summary.json').is_file()]
        if complete:
            chosen = complete[-1]
            item.update(status='complete', run=str(chosen), returncode=0)
            run_paths.append(chosen)
            orphans += orphaned_runs(suite, item['model'], item['seed'], run_paths)
        else:
            item.update(status='pending')
            resume = resumable_run(suite, item['model'], item['seed'])
            item.update(resume=str(resume / 'checkpoints/last.pt') if resume else None,
                        orphaned=orphaned_runs(suite, item['model'], item['seed'], run_paths))
    if orphans:
        print(f'WARNING {len(orphans)} orphaned run directories need review: {orphans}', flush=True)
    write_json(suite / 'suite_status.json', state)
    write_json(report / 'suite_status.json', state)
    build_report(run_paths, report)

    fixed_config = suite / 'fixed_config.json'
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
        log_path = suite / f'{item["model"]}_seed{item["seed"]}.log'
        started = time.monotonic()
        write_json(suite / 'suite_status.json', state)
        with log_path.open('w') as log:
            returncode = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
        paths = run_dirs(suite, item['model'], item['seed'])
        resumed = Path(item['resume']).parent.parent if item.get('resume') else None
        done = [path for path in paths if (path / 'summary.json').is_file()
                and (path not in run_paths or path == resumed)]
        if len(done) != 1:
            raise RuntimeError(f'Expected one completed run for {item["model"]}/{item["seed"]}: {done}')
        path = done[0]
        run_paths.append(path)
        item.update(status='complete' if returncode == 0 else 'failed',
                    returncode=returncode, seconds=time.monotonic() - started,
                    run=str(path), resume=None, orphaned=[])
        if returncode == 0 and (path / 'checkpoints/last.pt').exists():
            (path / 'checkpoints/last.pt').unlink()
        write_json(suite / 'suite_status.json', state)
        build_report(run_paths, report)
        write_json(report / 'suite_status.json', state)
        if returncode:
            state['status'] = 'failed'
            write_json(suite / 'suite_status.json', state)
            write_json(report / 'suite_status.json', state)
            raise RuntimeError(f'Failed run: {path}')
    state['status'] = 'complete'
    write_json(suite / 'suite_status.json', state)
    write_json(report / 'suite_status.json', state)
    build_report(run_paths, report)
    print(f'COMPLETE SUITE {report}')


if __name__ == '__main__':
    main()
