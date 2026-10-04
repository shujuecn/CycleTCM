"""Small helpers shared by reproduction commands."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def run_directory(parent, name):
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    path = Path(parent).expanduser().resolve() / f'{stamp}_{name}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
