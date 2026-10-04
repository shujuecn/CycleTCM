"""
Extract features from processed whole-tongue images using Qwen3-VL-4B-Instruct.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import time
import importlib.metadata
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import MLLM_FEATURES_FILE, PROCESSED_DATA_DIR
from utils.experiment import run_directory, sha256, write_json

import torch
from PIL import Image

SYSTEM_PROMPT = """
You are an experienced Traditional Chinese Medicine tongue-diagnosis expert. Based on the input tongue image, please determine the presence of the following tongue attributes: TonguePale, TipSideRed, Spot, Ecchymosis, Crack, ToothMark, FurThick, and FurYellow.
Based on these tongue manifestations, further assess whether the five organs, Heart, Lung, Spleen, Liver, and Kidney, may show abnormal tendencies.
""".strip()

TCM_PRIOR="""
TonguePale indicates the tongue is pale.
TipSideRed reflects the tip or sides of the tongue are red.
Spot denotes the presence of spots on the tongue.
Ecchymosis shows there is ecchymosis on the tongue.
Crack indicates there are cracks on the tongue.
ToothMark reflects the presence of tooth marks on the tongue.
FurThick describes the thickness of the tongue fur.
FurYellow indicates the tongue fur is yellow.
""".strip()

USER_PROMPT = (
    "Please analyze this tongue image according to your expertise and the instructions above."
)


def _default_model_dir() -> str:
    return os.environ.get(
        "QWEN3_VL_MODEL_DIR",
        str(Path.home() / ".cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"),
    )


def _collect_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def _inputs_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _last_hidden_pooled(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """[B, S, H] -> [B, H]: masked arithmetic mean over S (same as mean(dim=1) when mask is all ones)."""
    if attention_mask is None:
        return last_hidden.mean(dim=1)
    m = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
    summed = (last_hidden * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return summed / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(_default_model_dir()))
    parser.add_argument('--images-dir', type=Path, default=PROCESSED_DATA_DIR / 'images')
    parser.add_argument('--output-dir', type=Path, default=MLLM_FEATURES_FILE.parent)
    parser.add_argument('--output', type=Path, help='Filename under a new timestamp-prefixed directory')
    parser.add_argument('--resume', type=Path, help='Continue an existing extraction directory')
    parser.add_argument('--max-images', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    if args.max_images < 0:
        parser.error('max-images must be >=0')
    model_dir, images_dir = args.model_dir.expanduser().resolve(), args.images_dir.expanduser().resolve()
    if not model_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(f'Missing model/images directory: {model_dir}, {images_dir}')
    if args.output and args.resume:
        parser.error('output and resume cannot be combined')
    output = args.resume.expanduser().resolve() if args.resume else run_directory(
        args.output.parent if args.output else args.output_dir, 'qwen_features')
    if args.resume and not output.is_dir():
        raise FileNotFoundError(output)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output / 'extract.log'), logging.StreamHandler()])
    logging.info('RUN %s', output)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    metadata = {'model_dir': str(model_dir), 'weight_revision': 'Local ModelScope snapshot/master; content pinned by hashes',
                'model_files': {str(path.relative_to(model_dir)): sha256(path) for path in sorted(model_dir.rglob('*')) if path.is_file()},
                'images_dir': str(images_dir), 'dtype': str(dtype), 'pooling': 'Last hidden state; attention-masked sequence mean',
                'system_prompt': SYSTEM_PROMPT, 'tcm_prior': TCM_PRIOR, 'user_prompt': USER_PROMPT,
                'prompt_sha256': hashlib.sha256((SYSTEM_PROMPT+'\n'+TCM_PRIOR+'\n'+USER_PROMPT).encode()).hexdigest(),
                'versions': {name: importlib.metadata.version(name) for name in ['torch','transformers','Pillow']},
                'add_generation_prompt': True, 'use_cache': False, 'seed': 42}
    if args.resume:
        if metadata != json.loads((output / 'metadata.json').read_text()):
            raise ValueError('Extraction provenance changed; start a new feature version')
    else:
        write_json(output / 'metadata.json', metadata)
    records_dir = output / 'records'
    records_dir.mkdir(exist_ok=True)
    paths = _collect_images(images_dir)
    if args.max_images:
        # Evenly spaced names cover ten distinct images across the manifest range.
        indices = torch.linspace(0, len(paths)-1, min(args.max_images,len(paths))).long().tolist()
        paths = [paths[index] for index in indices]
    pending = []
    for path in paths:
        saved = records_dir / (path.name + '.json')
        if saved.is_file():
            record = json.loads(saved.read_text())
            if record['image_sha256'] != sha256(path):
                raise ValueError(f'Input changed: {path}')
        else:
            pending.append(path)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if pending:
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype, device_map={'': 0} if torch.cuda.is_available() else None,
            local_files_only=True).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        device = next(model.parameters()).device
        start = time.monotonic()
        for index, img_path in enumerate(pending):
            with Image.open(img_path) as source:
                image = source.convert('RGB')
            messages = [
                {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT+'\n'+TCM_PRIOR}]},
                {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': USER_PROMPT}]},
            ]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors='pt')
            inputs.pop('token_type_ids', None)
            input_ids_hash = hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()
            input_shapes = {key: list(value.shape) for key,value in inputs.items() if isinstance(value,torch.Tensor)}
            inputs = _inputs_to_device(inputs, device)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[-1]
                pooled = _last_hidden_pooled(hidden, inputs.get('attention_mask')).float().cpu()[0]
            if pooled.shape != (2560,) or not torch.isfinite(pooled).all():
                raise ValueError(f'Invalid feature: {img_path.name}')
            write_json(records_dir / (img_path.name+'.json'), {
                'image_file': img_path.name, 'image_sha256': sha256(img_path),
                'qwen_feature': pooled.tolist(), 'input_ids_sha256': input_ids_hash,
                'input_shapes': input_shapes, 'hidden_shape': list(hidden.shape)})
            if index % 25 == 0 or index == len(pending)-1:
                elapsed = time.monotonic()-start
                write_json(output / 'status.json', {'status': 'extracting', 'completed':len(paths)-len(pending)+index+1,
                                                   'target':len(paths), 'seconds_per_image':elapsed/(index+1)})
                logging.info('EXTRACT %d/%d %.3fs/image %s hidden=%s', len(paths)-len(pending)+index+1,
                             len(paths),elapsed/(index+1),img_path.name,list(hidden.shape))
            del outputs, hidden, inputs
    records = [json.loads((records_dir/(path.name+'.json')).read_text()) for path in paths]
    feature_file = output / (args.output.name if args.output else 'all_features.json')
    write_json(feature_file,[{'image_file':row['image_file'],'qwen_feature':row['qwen_feature']} for row in records])
    old = {row['image_file']:row['qwen_feature'] for row in json.loads(MLLM_FEATURES_FILE.read_text())}
    comparisons=[]
    for row in records:
        if row['image_file'] not in old:
            continue
        new, legacy = torch.tensor(row['qwen_feature']), torch.tensor(old[row['image_file']])
        comparisons.append({'image_file':row['image_file'],'sequence_length':row['hidden_shape'][1],
                            'cosine_similarity':torch.nn.functional.cosine_similarity(new,legacy,dim=0).item(),
                            'max_absolute_difference':(new-legacy).abs().max().item(),
                            'max_relative_difference':((new-legacy).abs()/legacy.abs().clamp_min(1e-6)).max().item()})
    write_json(output/'cache_comparison.json', {'comparisons':comparisons,
               'decision':'New version stored separately; legacy cache preserved. Versions have different numerical behavior.'})
    write_json(output/'status.json', {'status':'complete','count':len(records), 'feature_sha256':sha256(feature_file),
                                     'full_dataset':len(records)==5109})
    logging.info('COMPLETE %d records -> %s',len(records),feature_file)


if __name__ == '__main__':
    main()
