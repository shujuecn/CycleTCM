"""Track an existing suite and publish its completed reports."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.experiment import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    suite = args.suite.resolve()
    report = ROOT / 'reports/reproduction' / suite.name
    output = run_directory(ROOT / 'outputs/tracking', 'suite_tracker')
    state = {'pid': os.getpid(), 'controller_pid': args.controller_pid,
             'suite': str(suite), 'status': 'tracking'}
    published = set()
    print(f'TRACKER {output}', flush=True)
    try:
        if args.publish:
            branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
            if branch != 'shujuecn':
                raise RuntimeError('Publishing requires shujuecn branch')
        while True:
            queue_state = json.loads((suite / 'suite_status.json').read_text())
            completed = {(item['model'], item['seed']) for item in queue_state['queue']
                         if item['status'] == 'complete'}
            with (report / 'main_results.csv').open(newline='') as handle:
                reported = {(row['model'], int(row['seed'])) for row in csv.DictReader(handle)}
            report_state = json.loads((report / 'suite_status.json').read_text())
            report_completed = {(item['model'], item['seed']) for item in report_state['queue']
                                if item['status'] == 'complete'}
            ready = reported == completed == report_completed
            if queue_state['status'] == 'complete':
                ready = ready and report_state['status'] == 'complete'
            if ready and (completed != published or queue_state['status'] == 'complete'):
                if args.publish:
                    relative = str(report.relative_to(ROOT))
                    subprocess.run(['git', 'add', '--', relative], cwd=ROOT, check=True)
                    changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', relative], cwd=ROOT)
                    if changed.returncode == 1:
                        message = f'results: reproduction suite {len(completed)}/{len(queue_state["queue"])} {queue_state["status"]}'
                        subprocess.run(['git', 'commit', '--only', '-m', message, '--', relative], cwd=ROOT, check=True)
                    elif changed.returncode != 0:
                        raise RuntimeError('Unable to inspect staged report')
                    subprocess.run(['git', 'push', 'fork', 'shujuecn'], cwd=ROOT, check=True)
                published = completed
                state['completed'] = len(completed)
                state['last_report_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
                print(f'REPORTED {len(completed)}/{len(queue_state["queue"])} {queue_state["status"]}', flush=True)
            if queue_state['status'] == 'failed':
                raise RuntimeError('Training controller reported failure; inspect suite logs')
            if ready and queue_state['status'] == 'complete':
                state['status'] = 'complete'
                break
            os.kill(args.controller_pid, 0)
            write_json(output / 'status.json', state)
            time.sleep(15)
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        write_json(output / 'status.json', state)


if __name__ == '__main__':
    main()
