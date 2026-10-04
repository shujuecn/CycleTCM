"""Check the reproduction environment without starting an experiment or downloading weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/environment')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--model-dir', type=Path, default=Path(os.environ.get(
        'QWEN3_VL_MODEL_DIR',
        str(Path.home() / '.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master'),
    )))
    parser.add_argument('--skip-qwen', action='store_true', help='Skip local Qwen loading and extraction')
    args = parser.parse_args()
    if args.batch_size < 2 or args.workers < 0:
        parser.error('batch-size must be >=2 (BatchNorm); workers must be >=0')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
    report = {
        'timestamp': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'checks': [],
        'file_hashes': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ['pyproject.toml', 'uv.lock']
        },
    }

    def check(name, function):
        print(f'CHECK {name}', flush=True)
        started = time.monotonic()
        try:
            result = function()
            entry = {'name': name, 'status': 'passed', 'details': result}
        except Exception as error:
            entry = {'name': name, 'status': 'failed', 'error': str(error),
                     'traceback': traceback.format_exc()}
        entry['seconds'] = round(time.monotonic() - started, 3)
        report['checks'].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        (output / 'environment_check.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
        )
        return entry['status'] == 'passed'

    def dependencies():
        versions = {}
        for name in ['torch', 'torchvision', 'transformers', 'accelerate', 'safetensors',
                     'modelscope', 'cv2', 'numpy', 'sklearn', 'matplotlib', 'PIL', 'tqdm']:
            module = importlib.import_module(name)
            versions[name] = getattr(module, '__version__', 'unknown')
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        assert AutoProcessor is not None and Qwen3VLForConditionalGeneration is not None
        return versions

    if not check('dependencies', dependencies):
        return 1
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from utils.paths import FEATURE_FILE, LABEL_DIR, MLLM_FEATURES_FILE, PROCESSED_DATA_DIR
    torch.manual_seed(42)
    torch.set_num_threads(8)

    def cuda():
        assert torch.cuda.is_available(), 'CUDA is not accessible from this process'
        probe = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
             '--format=csv,noheader'], capture_output=True, text=True, check=True,
        )
        x = torch.randn(64, 64, device='cuda', requires_grad=True)
        (x @ x.T).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        from torchvision.ops import nms
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device='cuda')
        indices = nms(boxes, torch.tensor([0.9, 0.8], device='cuda'), 0.5)
        assert indices.tolist() == [0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = x.detach() @ x.detach().T
        assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
        torch.cuda.synchronize()
        return {'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
                'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'bf16': torch.cuda.is_bf16_supported(), 'nvidia_smi': probe.stdout.strip()}

    gpu_ok = check('cuda_kernels_backward_and_bf16', cuda)
    records = json.loads(FEATURE_FILE.read_text())
    label_keys = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark',
                  'FurThick', 'FurYellow', 'Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
    image_keys = [key for key in records[0] if key.startswith('img_')]
    assert len(image_keys) == 7
    cached_sample = {}

    def data():
        assert len(records) == 5109
        splits = []
        counts = {}
        for split, filename in [('train', 'train_dataset.json'), ('val', 'val_dataset.json'),
                                ('test', 'test.json')]:
            rows = json.loads((LABEL_DIR / filename).read_text())
            ids = {row['id'] for row in rows}
            splits.append(ids)
            counts[split] = {'images': len(rows), 'subjects': len(ids)}
        assert not splits[0] & splits[1] and not splits[0] & splits[2] and not splits[1] & splits[2]
        assert set.union(*splits) == {row['id'] for row in records}
        for row in records:
            for key in image_keys:
                assert (PROCESSED_DATA_DIR / row[key]).is_file()
            assert all(row[key] in (0, 1) for key in label_keys)
        features = json.loads(MLLM_FEATURES_FILE.read_text())
        names = [row['image_file'] for row in features]
        assert len(features) == len(records) and len(set(names)) == len(names)
        assert set(names) == {row['image_file'] for row in records}
        vectors = np.asarray([row['qwen_feature'] for row in features], dtype=np.float32)
        assert vectors.shape == (5109, 2560) and np.isfinite(vectors).all()
        selected = next(row for row in features if row['image_file'] == records[0]['image_file'])
        cached_sample.update(selected)
        return {'splits': counts, 'image_references': len(records) * len(image_keys),
                'qwen_shape': list(vectors.shape), 'all_vectors_finite': True}

    check('data_and_cached_features', data)

    def vision_utilities():
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from data_preprocessed import data_segment_organs as organs, data_segment_regions as regions
        source = PROCESSED_DATA_DIR / 'pp' / records[0]['image_file']
        directories = {name: str(output / 'region_smoke' / folder) for name, folder in {
            'top_edge': 'images_heart_lung', 'bottom_edge': 'images_kidney',
            'right_edge': 'images_liver', 'center_rect': 'images_spleen',
        }.items()}
        for directory in directories.values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        regions.process_image(str(source), str(output / 'region_smoke/images_body'),
                              str(output / 'region_smoke/images_edge'), 0.15)
        assert organs.process_single_image(str(source), directories, r=0.196, r2=0.632, r_liver=0.10)
        generated = sorted((output / 'region_smoke').rglob('*.png'))
        assert len(generated) == 6
        assert all(cv2.imread(str(path)) is not None for path in generated)
        for key in image_keys:
            with Image.open(PROCESSED_DATA_DIR / records[0][key]) as image:
                image.load()
                assert image.size == (224, 224)
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.savefig(output / 'plot_smoke.png')
        plt.close()
        return {'regional_outputs': 6, 'pillow_decode': True, 'matplotlib_save': True}

    check('opencv_regions_pillow_and_plotting', vision_utilities)

    def pretrained():
        from models.model_visual import resnet50_backbone
        checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-0676ba61.pth'
        assert checkpoint.is_file(), f'ImageNet weights missing: {checkpoint}; no download attempted'
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = resnet50_backbone(pretrained=True)
        assert torch.equal(model.conv1.weight.detach(), weights['conv1.weight'])
        assert torch.equal(model.layer4[2].conv3.weight.detach(), weights['layer4.2.conv3.weight'])
        return {'cache': str(checkpoint), 'global_backbone_weights_verified': True,
                'note': 'Local backbones currently ignore pretrained; tracked in the reproduction plan.'}

    check('local_imagenet_weights', pretrained)

    def model_step(name):
        module = importlib.import_module(f'train.train_model_{name}')
        transform = transforms.ToTensor()
        selected_ids = [records[0]['id'], records[1]['id']]
        if name == 'mllm':
            dataset = module.TongueMLLMDataset(str(FEATURE_FILE), str(MLLM_FEATURES_FILE), selected_ids)
            model = module.MLLM_Model()
        else:
            kwargs = {'base_dir': str(PROCESSED_DATA_DIR), 'transform': transform, 'id_list': selected_ids}
            if name == 'multimodal':
                kwargs['mllm_features_file'] = str(MLLM_FEATURES_FILE)
            dataset = module.TongueImageDataset(str(FEATURE_FILE), **kwargs)
            model = module.CycleTCM(pretrained=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            num_workers=args.workers, shuffle=False)
        batch = next(iter(loader))
        # Repeat a small real-data batch if a larger resource smoke test was requested.
        size = batch['syndrome_labels'].shape[0]
        batch = {key: value.repeat((args.batch_size + size - 1) // size,
                                    *([1] * (value.dim() - 1)))[:args.batch_size].cuda()
                 for key, value in batch.items() if isinstance(value, torch.Tensor)}
        model = model.cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
        before = model.classifier1.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        if name == 'mllm':
            inputs = [batch['mllm_feature']]
        else:
            inputs = [batch[key] for key in ['img_whole', 'img_edge', 'img_body', 'img_heart_lung',
                                             'img_spleen', 'img_liver', 'img_kidney']]
            if name == 'multimodal':
                inputs.append(batch['mllm_feature'])
        syndrome, organ = model(*inputs)
        assert syndrome.shape == (args.batch_size, 8) and organ.shape == (args.batch_size, 5)
        assert torch.isfinite(syndrome).all() and torch.isfinite(organ).all()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(syndrome, batch['syndrome_labels'])
                + torch.nn.functional.binary_cross_entropy_with_logits(organ, batch['organ_labels']))
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                   if parameter.grad is not None)
        optimizer.step()
        assert not torch.equal(before, model.classifier1.weight)
        torch.cuda.synchronize()
        return {'parameters': sum(parameter.numel() for parameter in model.parameters()),
                'output_shapes': [list(syndrome.shape), list(organ.shape)], 'loss': loss.item(),
                'precision': 'fp32', 'real_data_workers': args.workers,
                'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
                'peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 2**30, 3),
                'optimizer_step': True, 'weights_changed': True, 'pretrained_for_smoke': False}

    if gpu_ok:
        for name in ['mllm', 'visual', 'multimodal']:
            check(f'{name}_real_data_cuda_training_step', lambda name=name: model_step(name))
            gc.collect()
            torch.cuda.empty_cache()

    def metrics():
        from train.train_model_visual import calculate_metrics
        truth = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        probabilities = np.array([[0.9, 0.2], [0.6, 0.8], [0.7, 0.4], [0.2, 0.3]])
        result = calculate_metrics(probabilities, truth)
        assert np.isclose(result['per_class_acc_mean'], 0.75)
        assert np.isclose(result['f1'], (0.8 + 2 / 3) / 2)
        return {'macro_accuracy': float(result['per_class_acc_mean']), 'macro_f1': float(result['f1'])}

    check('sklearn_metrics', metrics)

    def qwen():
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from utils.mllm_feature_extract import SYSTEM_PROMPT, TCM_PRIOR, USER_PROMPT, _last_hidden_pooled
        model_dir = args.model_dir.expanduser().resolve()
        assert model_dir.is_dir(), f'Model directory not found: {model_dir}'
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True,
        ).eval()
        with Image.open(PROCESSED_DATA_DIR / records[0]['img_whole']) as source:
            image = source.convert('RGB')
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT + '\n' + TCM_PRIOR}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                       {'type': 'text', 'text': USER_PROMPT}]},
        ]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt')
        inputs.pop('token_type_ids', None)
        inputs = {key: value.cuda() if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            result = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[-1]
            pooled = _last_hidden_pooled(hidden, inputs.get('attention_mask')).float()
        assert pooled.shape == (1, 2560) and torch.isfinite(pooled).all()
        old = torch.tensor(cached_sample['qwen_feature'], device='cuda').unsqueeze(0)
        details = {'model_dir': str(model_dir), 'image': records[0]['image_file'],
                   'hidden_shape': list(hidden.shape), 'pooled_shape': list(pooled.shape),
                   'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   'cached_cosine_similarity': torch.nn.functional.cosine_similarity(pooled, old).item(),
                   'cached_max_absolute_difference': (pooled - old).abs().max().item(),
                   'note': 'One sample only; full feature provenance remains a separate reproduction task.'}
        (output / 'qwen_sample.json').write_text(json.dumps({
            'image_file': records[0]['image_file'], 'qwen_feature': pooled[0].cpu().tolist(),
        }) + '\n', encoding='utf-8')
        return details

    if gpu_ok and not args.skip_qwen:
        check('local_qwen_image_feature_extraction', qwen)
        gc.collect()
        torch.cuda.empty_cache()
    elif args.skip_qwen:
        report['checks'].append({'name': 'local_qwen_image_feature_extraction', 'status': 'skipped'})
    report['passed'] = all(entry['status'] != 'failed' for entry in report['checks'])
    (output / 'environment_check.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
    )
    print(f"REPORT {output / 'environment_check.json'}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
