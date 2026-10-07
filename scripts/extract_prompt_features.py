"""Extract Qwen3-VL or MedGemma features for A0/A1/A2 and unrelated U0 prompts."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, logging, sys, time
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.experiment import sha256, write_json
from utils.mllm_backbone import backbone_info, load_backbone, image_inputs, model_inputs

def archived_prompts(path: Path):
    text = path.read_text()
    names = {
        'system': 'B.1 SYSTEM_PROMPT（三组共享）',
        'A0': 'B.2 A0_Control_Direct（完整原文，整块直接可用）',
        'A1': 'B.3 A1_Knowledge_Direct（完整原文，整块直接可用）',
        'A2': 'B.4 A2_Knowledge_Reflection_Strict（完整原文，整块直接可用）',
    }
    result = {}
    for key, heading in names.items():
        start = text.index(heading)
        block = text[start:].split('```text\n', 1)[1].split('\n```', 1)[0]
        result[key] = block
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['A0', 'A1', 'A2', 'U0'], required=True)
    p.add_argument('--prompt-doc', type=Path, default=ROOT / 'reports/validation/20261007/plan.md')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--images-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-images', type=int, default=0)
    args = p.parse_args()
    if args.max_images < 0:
        p.error('max-images must be >= 0')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    model_type, feature_dim = backbone_info(args.model_dir.resolve())
    if args.variant == 'U0':
        prompts = {'system': '喜羊羊 美羊羊 懒羊羊 沸羊羊 慢羊羊 软绵绵 红太狼 灰太狼',
                   'U0': '别看我只是一只羊 羊儿的聪明难以想象'}
    else:
        prompts = archived_prompts(args.prompt_doc.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / 'records'; records_dir.mkdir(exist_ok=True)
    paths = sorted([x for x in args.images_dir.resolve().iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}], key=lambda x:x.name)
    if args.max_images:
        indices = torch.linspace(0, len(paths)-1, min(args.max_images, len(paths))).long().tolist()
        paths = [paths[i] for i in indices]
    prompt_sha = hashlib.sha256((prompts['system']+'\n'+prompts[args.variant]).encode()).hexdigest()
    prompt_source = Path(__file__) if args.variant == 'U0' else args.prompt_doc.resolve()
    metadata = {'variant': args.variant, 'prompt_doc': str(prompt_source), 'prompt_sha256': prompt_sha,
                'system_prompt': prompts['system'], 'user_prompt': prompts[args.variant],
                'model_dir': str(args.model_dir.resolve()), 'images_dir': str(args.images_dir.resolve()),
                'pooling': 'last hidden state; attention-masked sequence mean', 'add_generation_prompt': True,
                'dtype': 'torch.bfloat16' if torch.cuda.is_available() else 'torch.float32', 'seed': 42,
                'target': len(paths)}
    if model_type == 'gemma3':
        metadata.update(model_type=model_type, feature_dim=feature_dim,
                        input_protocol='native chat template; preserve image token_type_ids; pixels cast to model dtype')
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        if not args.resume:
            p.error('Output already exists; use --resume or a new output directory')
        previous_metadata = json.loads(metadata_path.read_text())
        # A moved archive may resume if its bytes and all extraction settings still match.
        if previous_metadata['prompt_doc'] != metadata['prompt_doc'] and args.variant != 'U0':
            provenance = json.loads((output / 'extraction_provenance.json').read_text())
            source_hashes = provenance['source_files_sha256']
            prompt_hash = source_hashes.get('prompt_doc') or source_hashes.get('docs/CycleTCM-质疑查证与消融验证方案.md')
            if prompt_hash == sha256(prompt_source):
                metadata['prompt_doc'] = previous_metadata['prompt_doc']
        if previous_metadata != metadata:
            raise ValueError('Prompt or input metadata changed; use a new output directory')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    model_files = {p.name: sha256(p) for p in sorted(args.model_dir.resolve().iterdir())
                   if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors', '.jinja', '.model'}}
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'Pillow']}
    provenance_path = output / 'extraction_provenance.json'
    if args.resume and provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if any(previous['model_files'].get(k) != v for k,v in model_files.items()) or previous['versions'] != versions:
            raise ValueError('Weights, processor or library versions changed; use a new output directory')
    else:
        write_json(provenance_path, {'created_at':datetime.now().astimezone().isoformat(), 'model_files':model_files, 'versions':versions,
                   'source_files_sha256':{'extractor':sha256(__file__), 'backbone':sha256(ROOT/'src/utils/mllm_backbone.py'), 'prompt_doc':sha256(prompt_source), 'uv.lock':sha256(ROOT/'uv.lock')},
                   'execution_protocol':{'model_eval':True, 'frozen_parameters':True, 'inference_mode':True,
                                         'use_cache':False, 'add_generation_prompt':True, 'image_before_text':True,
                                         'torch_manual_seed':42, 'torch_num_threads':8}})
    pending = []
    for image_path in paths:
        record_path = records_dir / (image_path.name + '.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['image_sha256'] != hashlib.sha256(image_path.read_bytes()).hexdigest():
                raise ValueError(f'Input changed: {image_path}')
        else:
            pending.append(image_path)
    if pending:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        processor, model = load_backbone(args.model_dir.resolve(), dtype, 'cuda:0' if torch.cuda.is_available() else 'cpu')
        started = time.monotonic()
        logging.info('EXTRACT variant=%s pending=%d target=%d', args.variant, len(pending), len(paths))
        for index, image_path in enumerate(pending):
            with Image.open(image_path) as source: image = source.convert('RGB')
            inputs = image_inputs(processor, model_type, prompts['system'], prompts[args.variant], image)
            input_hash = hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()
            input_shapes = {k:list(v.shape) for k,v in inputs.items() if isinstance(v,torch.Tensor)}
            inputs = model_inputs(inputs, model)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[-1]
                mask = inputs.get('attention_mask')
                if mask is None: pooled = hidden.mean(dim=1)
                else:
                    mask = mask.to(hidden.dtype).unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = pooled.float().cpu()[0]
            if pooled.shape != (feature_dim,) or not torch.isfinite(pooled).all(): raise ValueError(f'Invalid feature {image_path}')
            record = {'image_file':image_path.name, 'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                      'qwen_feature':pooled.tolist(), 'input_ids_sha256':input_hash,
                      'input_shapes':input_shapes, 'hidden_shape':list(hidden.shape)}
            (records_dir / (image_path.name+'.json')).write_text(json.dumps(record, ensure_ascii=False))
            if index % 25 == 0 or index == len(pending)-1:
                elapsed = time.monotonic()-started
                (output/'status.json').write_text(json.dumps({'status':'extracting','variant':args.variant,'completed':len(paths)-len(pending)+index+1,'target':len(paths),'seconds_per_image':elapsed/(index+1)}, indent=2)+'\n')
                logging.info('EXTRACT %d/%d %.3fs/image', len(paths)-len(pending)+index+1, len(paths), elapsed/(index+1))
            del outputs, hidden, inputs
    records = [json.loads((records_dir/(x.name+'.json')).read_text()) for x in paths]
    feature_file = output / 'all_features.json'
    feature_file.write_text(json.dumps([{'image_file':r['image_file'],'qwen_feature':r['qwen_feature']} for r in records], ensure_ascii=False))
    digest = hashlib.sha256(feature_file.read_bytes()).hexdigest()
    (output/'status.json').write_text(json.dumps({'status':'complete','variant':args.variant,'count':len(records),'full_dataset':len(records)==5109,'feature_sha256':digest}, indent=2)+'\n')
    logging.info('COMPLETE %d records -> %s', len(records), feature_file)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
