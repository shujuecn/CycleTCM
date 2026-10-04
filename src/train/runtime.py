"""Shared path arguments and logging for training entry points."""

import logging
from pathlib import Path

from utils.paths import MLLM_FEATURES_FILE, OUTPUT_DIR, PROCESSED_DATA_DIR


def add_path_arguments(parser, model_name, *, mllm=False):
    parser.add_argument('--data-dir', type=Path, default=PROCESSED_DATA_DIR,
                        help='Processed dataset directory (image root)')
    parser.add_argument('--feature-file', type=Path,
                        help='Feature manifest; defaults to DATA_DIR/feature_all_encoded.json')
    parser.add_argument('--label-dir', type=Path,
                        help='Split labels; defaults to DATA_DIR/labels/json')
    if mllm:
        parser.add_argument('--mllm-features-file', type=Path, default=MLLM_FEATURES_FILE,
                            help='Qwen feature JSON')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / model_name,
                        help='Run directory for logs and checkpoints')
    parser.add_argument('--output-log', '--output_log', dest='output_log', type=Path,
                        help='Per-class metrics file; relative paths are under OUTPUT_DIR '
                             '(default: results.log)')


def training_path_config(args):
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    feature_file = (args.feature_file or data_dir / 'feature_all_encoded.json').expanduser().resolve()
    label_dir = (args.label_dir or data_dir / 'labels/json').expanduser().resolve()
    output_log = (args.output_log or Path('results.log')).expanduser()
    if not output_log.is_absolute():
        output_log = output_dir / output_log
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    config = {
        'feature_file': str(feature_file),
        'base_dir': str(data_dir),
        'label_dir': str(label_dir),
        'checkpoint_dir': str(checkpoint_dir),
        'output_dir': str(output_dir),
        'output_log': str(output_log.resolve()),
    }
    if hasattr(args, 'mllm_features_file'):
        config['mllm_features_file'] = str(args.mllm_features_file.expanduser().resolve())
    return config


def configure_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(output_dir) / 'train.log', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
