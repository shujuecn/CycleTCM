"""One recorded training/evaluation path for diagnostics and CycleTCM ablations."""

import argparse
from contextlib import nullcontext
import csv
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from models.model_mllm import MLLM_Model
from models.model_visual import CycleTCM, resnet50_backbone
from train.data import LABELS, VIEWS, TongueDataset, split_records, load_features, positive_weights
from train.evaluation import metrics, save_evaluation
from utils.experiment import run_directory, sha256, write_json
from utils.paths import PROCESSED_DATA_DIR, FEATURE_FILE, LABEL_DIR, MLLM_FEATURES_FILE

MODULES = {'B': (False, False, False), 'BA': (True, False, False),
           'BU': (False, True, False), 'BM': (False, False, True),
           'visual': (True, True, False), 'full': (True, True, True)}
DEFAULTS = dict(model='visual', profile='code_compat', seed=42, epochs=200, batch_size=32,
                device='cuda:0', precision='fp32', workers=4, learning_rate=2e-4,
                weight_decay=1e-4, patience=50, min_delta=.001, init='global_only',
                normalize=False, scheduler_monitor='train_loss', limit=0,
                data_dir=str(PROCESSED_DATA_DIR), feature_file=str(FEATURE_FILE),
                label_dir=str(LABEL_DIR), mllm_features_file=str(MLLM_FEATURES_FILE))
ACTIVE_RUN = None


def write_history(output, history):
    if history:
        with (output / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in history for key in row)))
            writer.writeheader()
            writer.writerows(history)


class GlobalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_whole = resnet50_backbone()
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(.3)
        self.classifier1, self.classifier2 = nn.Linear(2048, 8), nn.Linear(2048, 5)

    def forward(self, image):
        x = self.backbone_whole(image).mean(dim=(2, 3))
        x = self.dropout(self.bn(x))
        return self.classifier1(x), self.classifier2(x)


def build_model(config, initialize=False):
    name = config['model']
    model = MLLM_Model() if name == 'mllm' else GlobalBackbone() if name == 'global' else CycleTCM(
        aglff=MODULES[name][0], uwbmoe=MODULES[name][1], mllm=MODULES[name][2])
    if initialize and name != 'mllm' and config['init'] != 'none':
        # Never silently replace missing pretrained weights with random initialization.
        weight_file = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-0676ba61.pth'
        state = torch.load(weight_file, map_location='cpu', weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith('fc.')}
        model.backbone_whole.load_state_dict(state, strict=True)
        if config['init'] == 'all_branches' and name != 'global':
            model.backbone_syndrome.resnet.load_state_dict(state, strict=True)
            model.backbone_organ.resnet.load_state_dict(state, strict=True)
    return model


def model_inputs(batch, config, device):
    name = config['model']
    keys = [] if name == 'mllm' else VIEWS[:1] if name == 'global' else VIEWS
    if name == 'mllm' or (name in MODULES and MODULES[name][2]):
        keys = [*keys, 'mllm_feature']
    return [batch[key].to(device, non_blocking=True) for key in keys]


def autocast(config):
    if config['precision'] == 'fp32':
        return nullcontext()
    return torch.autocast(torch.device(config['device']).type,
                          dtype=torch.bfloat16 if config['precision'] == 'bf16' else torch.float16)


def epoch_pass(model, loader, config, weights, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    losses, logits, labels, names = [], [], [], []
    device = torch.device(config['device'])
    weights = weights.to(device)
    with torch.set_grad_enabled(training):
        for batch in loader:
            targets = batch['labels'].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(config):
                syn, org = model(*model_inputs(batch, config, device))
                loss = (nn.functional.binary_cross_entropy_with_logits(syn, targets[:, :8], pos_weight=weights[:8])
                        + nn.functional.binary_cross_entropy_with_logits(org, targets[:, 8:], pos_weight=weights[8:]))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {batch["image_file"]}')
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(loss.item())
            logits.append(torch.cat([syn, org], dim=1).detach().float().cpu().numpy())
            labels.append(targets.cpu().numpy())
            names.extend(batch['image_file'])
    assert names == [row['image_file'] for row in loader.dataset.records] if not training else len(names) == len(loader.dataset)
    assert len(set(names)) == len(loader.dataset), 'Missing or repeated evaluation/training images'
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    probability = torch.tensor(logits).sigmoid().numpy()
    return float(np.mean(losses)), metrics(labels, probability), logits


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def environment(config):
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    result = {'python': sys.version, 'git_revision': git('rev-parse', 'HEAD'),
              'git_status': git('status', '--short'), 'lock_sha256': sha256(ROOT / 'uv.lock'),
              'versions': {name: importlib.metadata.version(name) for name in
                           ['torch', 'torchvision', 'transformers', 'numpy', 'scikit-learn', 'Pillow']},
              'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'deterministic_cudnn': torch.backends.cudnn.deterministic,
              'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_tf32': torch.backends.cudnn.allow_tf32,
              'precision': config['precision'], 'physical_batch': config['batch_size'],
              'effective_batch': config['batch_size'], 'labels': LABELS,
              'table3_B_definition': 'Three branches; raw global/local pooled vectors; A/U/M removed when disabled',
              'table3_B_evidence': 'Working assumption; upstream provides only a separate single-global diagnostic model'}
    if torch.device(config['device']).type == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(torch.device(config['device']))
        result['gpu_memory_gib'] = torch.cuda.get_device_properties(torch.device(config['device'])).total_memory / 2**30
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                                      '--format=csv,noheader'], text=True).strip()
    if config['init'] != 'none' and config['model'] != 'mllm':
        result['imagenet'] = {'weights': 'ResNet50_Weights.IMAGENET1K_V1',
                              'sha256': sha256(Path(torch.hub.get_dir()) / 'checkpoints/resnet50-0676ba61.pth')}
    return result


def loaders(config):
    splits = split_records(config['feature_file'], config['label_dir'])
    uses_features = config['model'] == 'mllm' or (config['model'] in MODULES and MODULES[config['model']][2])
    features = load_features(config['mllm_features_file']) if uses_features else None
    if features is not None:
        assert {row['image_file'] for rows in splits.values() for row in rows} == set(features)
    weights = {key: positive_weights(rows) for key, rows in splits.items()}
    if config['profile'] == 'reviewed':
        weights['val'] = weights['train']
    weights['test'] = torch.ones(13)
    output = {}
    for split, rows in splits.items():
        if config['limit']:
            rows = rows[:config['limit']]
        dataset = TongueDataset(rows, config['data_dir'], config['model'], features,
                                training=split == 'train', profile=config['profile'], normalize=config['normalize'])
        output[split] = DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
                                   num_workers=config['workers'], pin_memory=config['device'].startswith('cuda'),
                                   generator=torch.Generator())
    return output, weights


def run(default_model=None):
    global ACTIVE_RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reproduction', help='Parent for a new timestamp-prefixed run')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evaluate', type=Path, help='Independently load and evaluate a checkpoint')
    parser.add_argument('--split', choices=['val', 'test', 'both'], default='both')
    for key, choices in [('model', ['global', 'mllm', *MODULES]), ('profile', ['code_compat', 'reviewed']),
                         ('precision', ['fp32', 'bf16', 'fp16']), ('init', ['none', 'global_only', 'all_branches']),
                         ('scheduler_monitor', ['train_loss', 'val_loss'])]:
        parser.add_argument('--' + key.replace('_', '-'), choices=choices)
    for key in ['seed', 'epochs', 'batch_size', 'workers', 'patience', 'limit']:
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ['learning_rate', 'weight_decay', 'min_delta']:
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    for key in ['device', 'data_dir', 'feature_file', 'label_dir', 'mllm_features_file']:
        parser.add_argument('--' + key.replace('_', '-'))
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.resume and args.evaluate:
        parser.error('Choose resume or evaluate')
    checkpoint_path = args.resume or args.evaluate
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path else None
    config = DEFAULTS.copy()
    if default_model:
        config['model'] = default_model
    if checkpoint:
        config.update(checkpoint['config'])
    if args.config:
        supplied = json.loads(args.config.read_text())
        if set(supplied) - set(DEFAULTS):
            parser.error(f'Unknown configuration keys: {set(supplied) - set(DEFAULTS)}')
        config.update(supplied)
    config.update({key: getattr(args, key) for key in DEFAULTS if getattr(args, key, None) is not None})
    if args.data_dir:
        if not args.feature_file:
            config['feature_file'] = str(Path(args.data_dir) / 'feature_all_encoded.json')
        if not args.label_dir:
            config['label_dir'] = str(Path(args.data_dir) / 'labels/json')
    for key in ['data_dir', 'feature_file', 'label_dir', 'mllm_features_file']:
        config[key] = str(Path(config[key]).expanduser().resolve())
    if config['batch_size'] < 2 or config['epochs'] < 1 or config['workers'] < 0 or config['limit'] < 0 or config['limit'] == 1:
        parser.error('Require batch-size >=2, epochs >=1, workers/limit >=0, limit !=1')
    if config['profile'] == 'code_compat' and (config['normalize'] or config['init'] == 'all_branches' or config['scheduler_monitor'] != 'train_loss'):
        parser.error('Protocol changes require profile=reviewed')
    if config['precision'] == 'fp16' and not config['device'].startswith('cuda'):
        parser.error('fp16 requires CUDA')
    if config['limit'] and config['limit'] % config['batch_size'] == 1:
        parser.error('Training tail batch cannot contain one sample (BatchNorm)')
    if args.resume:
        if not all(key in checkpoint for key in ('optimizer', 'scheduler', 'scaler', 'rng', 'history')):
            parser.error('Resume requires a full training checkpoint (last.pt); best.pt contains evaluation weights only')
        allowed = {'epochs', 'device', 'workers'}
        changed = {key for key in DEFAULTS if config[key] != checkpoint['config'][key]}
        if changed - allowed:
            parser.error(f'Resume cannot change scientific config: {changed - allowed}')
        source_best = Path(checkpoint['best_path']).resolve()
        if not source_best.is_file():
            parser.error(f'Resume requires the selected best checkpoint: {source_best}')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    label = f'{config["profile"]}_{config["model"]}_seed{config["seed"]}'
    output = run_directory(args.output_dir, label + ('_eval' if args.evaluate else ''))
    ACTIVE_RUN = output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output / 'train.log'),
                                  logging.StreamHandler()])
    logging.info('RUN %s', output)
    write_json(output / 'status.json', {'status': 'initializing'})
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment(config))
    (output / 'source.patch').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True))
    (output / 'command.json').write_text(json.dumps(sys.argv) + '\n')
    if args.resume:
        write_json(output / 'resume.json', {
            'checkpoint': str(args.resume.resolve()), 'checkpoint_sha256': sha256(args.resume),
            'best_checkpoint': str(source_best), 'best_checkpoint_sha256': sha256(source_best),
            'start_epoch': checkpoint['epoch'] + 1, 'best_epoch': checkpoint['best_epoch'],
            'early_stop_counter': checkpoint['counter']})
    data_loaders, weights = loaders(config)
    data_manifest = {'feature_manifest_sha256': sha256(config['feature_file']),
                     'splits': {split: {'images': len(loader.dataset),
                                        'subjects': len({row['id'] for row in loader.dataset.records})}
                                for split, loader in data_loaders.items()},
                     'loss_weights': {key: value.tolist() for key, value in weights.items()}}
    if config['model'] == 'mllm' or (config['model'] in MODULES and MODULES[config['model']][2]):
        data_manifest['mllm_features_sha256'] = sha256(config['mllm_features_file'])
    write_json(output / 'data_manifest.json', data_manifest)
    if checkpoint and data_manifest != checkpoint['data_manifest']:
        raise ValueError('Checkpoint data/weights differ from current inputs')
    model = build_model(config, initialize=not checkpoint).to(config['device'])
    logging.info('PARAMETERS %d', sum(p.numel() for p in model.parameters()))
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    history = checkpoint['history'].copy() if args.resume else []
    write_history(output, history)
    if not args.evaluate:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.3, patience=5)
        scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16')
        best_score, counter, best_epoch = -1., 0, None
        best_path = output / 'checkpoints/best.pt'
        best_path.parent.mkdir()
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            best_score, counter, best_epoch = checkpoint['best_score'], checkpoint['counter'], checkpoint['best_epoch']
            shutil.copyfile(source_best, best_path)
            restore_rng(checkpoint['rng'])
            start = checkpoint['epoch'] + 1
            del checkpoint
        else:
            start = 0
        state = None
        for epoch in range(start, config['epochs']):
            if counter >= config['patience']:
                break
            epoch_started = time.monotonic()
            if config['device'].startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            for split, loader in data_loaders.items():
                loader.generator.manual_seed(config['seed'] + epoch * 3 + ['train', 'val', 'test'].index(split))
            train_loss, train_metrics, _ = epoch_pass(model, data_loaders['train'], config, weights['train'], optimizer, scaler)
            val_loss, val_metrics, _ = epoch_pass(model, data_loaders['val'], config, weights['val'])
            scheduler.step(train_loss if config['scheduler_monitor'] == 'train_loss' else val_loss)
            score = val_metrics['selection_acc']
            improved = score > best_score + config['min_delta']
            if improved:
                best_score, counter, best_epoch = score, 0, epoch
                best_path = output / 'checkpoints/best.pt'
            else:
                counter += 1
            row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'selection_acc': score,
                   'best_epoch': best_epoch, 'early_stop_counter': counter, 'lr': optimizer.param_groups[0]['lr'],
                   'compute_seconds': time.monotonic() - epoch_started,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if config['device'].startswith('cuda') else None}
            for split, result in [('train', train_metrics), ('val', val_metrics)]:
                for task in ['syndrome', 'organ']:
                    for metric in ['acc', 'f1']:
                        row[f'{split}_{task}_{metric}'] = result[task][metric]
            history.append(row)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, best_epoch=best_epoch, best_score=best_score,
                         counter=counter, best_path=str(best_path), config=config, data_manifest=data_manifest,
                         selected_metrics=val_metrics, history=history, rng=rng_state())
            if improved:
                save_checkpoint(best_path, {key: state[key] for key in
                                            ('model', 'epoch', 'best_epoch', 'config', 'data_manifest', 'selected_metrics')})
            save_checkpoint(output / 'checkpoints/last.pt', state)
            row['seconds_with_checkpoint'] = time.monotonic() - epoch_started
            write_history(output, history)
            write_json(output / 'status.json', {'status': 'training', 'epoch': epoch, 'best_epoch': best_epoch,
                                               'early_stop_counter': counter, 'output_dir': str(output)})
            logging.info('EPOCH %s', json.dumps(row))
        if not best_path.is_file():
            raise RuntimeError('No best checkpoint was saved')
        del optimizer, scheduler, scaler, state
        checkpoint_path = best_path
        selected = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'], strict=True)
        del selected
    selected_hash = sha256(checkpoint_path)
    summary = {'checkpoint': str(checkpoint_path.resolve()), 'checkpoint_sha256': selected_hash,
               'config': config, 'engineering_only': bool(config['limit']),
               'best_epoch': checkpoint['epoch'] if args.evaluate else best_epoch,
               'stop_reason': 'independent_evaluation' if args.evaluate else 'patience' if counter >= config['patience'] else 'epoch_limit'}
    for split in ['val', 'test'] if args.split == 'both' else [args.split]:
        if split == 'test' and not config['limit']:
            assert len(data_loaders[split].dataset) == 895, 'Formal test must include all 895 images'
        _, _, logits = epoch_pass(model, data_loaders[split], config, weights[split])
        result = save_evaluation(output, split, data_loaders[split].dataset.records, logits, selected_hash)
        summary[split] = {task: result[task] for task in ['syndrome', 'organ']}
        logging.info('%s samples=%d syndrome Acc/F1=%.4f/%.4f organ Acc/F1=%.4f/%.4f',
                     split, result['samples'], result['syndrome']['acc'], result['syndrome']['f1'], result['organ']['acc'], result['organ']['f1'])
    write_json(output / 'summary.json', summary)
    write_json(output / 'status.json', {'status': 'complete', 'engineering_only': bool(config['limit']),
                                       'best_epoch': summary['best_epoch'], 'stop_reason': summary['stop_reason']})
    logging.info('COMPLETE %s', output)


def main(default_model=None):
    try:
        run(default_model)
    except BaseException as error:
        if ACTIVE_RUN is not None:
            write_json(ACTIVE_RUN / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                                   'error': str(error), 'error_type': type(error).__name__})
            logging.exception('Run failed')
        raise


if __name__ == '__main__':
    main()
