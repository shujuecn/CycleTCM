"""Create deterministic feature controls for E1 without touching source data."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.source.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    constant = np.ones(2560, dtype=np.float32)
    controls = {'constant': [], 'random': []}
    for row in rows:
        name = row['image_file']
        seed = int.from_bytes(hashlib.sha256(('random-control:' + name).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        controls['constant'].append({'image_file': name, 'qwen_feature': constant.tolist()})
        controls['random'].append({'image_file': name, 'qwen_feature': rng.standard_normal(2560).astype(np.float32).tolist()})
    for name, values in controls.items():
        out = args.output_dir / f'{name}_features.json'
        out.write_text(json.dumps(values, ensure_ascii=False))
        metadata = {'source': str(args.source.resolve()), 'type': name, 'seed_rule': 'sha256(image_file)', 'count': len(values), 'dimension': 2560}
        (args.output_dir / f'{name}_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(out)


if __name__ == '__main__':
    main()
