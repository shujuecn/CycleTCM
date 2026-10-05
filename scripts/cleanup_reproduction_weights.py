"""Remove resumable checkpoints from completed reproduction runs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from utils.experiment import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True,
                        help='Suite directory containing timestamped run directories')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/cleanup')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    entries = []
    for status_path in sorted(args.run_root.glob('*_code_compat_*/status.json')):
        status = json.loads(status_path.read_text())
        if status.get('status') != 'complete':
            continue
        last_path = status_path.parent / 'checkpoints/last.pt'
        if not last_path.exists():
            continue
        size = last_path.stat().st_size
        entry = {'run': str(status_path.parent), 'path': str(last_path), 'bytes': size,
                 'action': 'would_remove' if args.dry_run else 'removed'}
        if not args.dry_run:
            last_path.unlink()
        entries.append(entry)

    output = run_directory(args.output_dir, 'completed_last_checkpoints')
    write_json(output / 'cleanup.json', {'run_root': str(args.run_root.resolve()),
                                         'dry_run': args.dry_run, 'entries': entries,
                                         'removed_bytes': sum(e['bytes'] for e in entries)})
    print(f'CLEANUP {output} entries={len(entries)} bytes={sum(e["bytes"] for e in entries)}')


if __name__ == '__main__':
    main()
