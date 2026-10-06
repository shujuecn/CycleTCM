"""Extract Qwen features for the archived TongueBench A0/A1/A2 prompts."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, logging, sys, time
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.experiment import sha256, write_json

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
    p.add_argument('--variant', choices=['A0', 'A1', 'A2'], required=True)
    p.add_argument('--prompt-doc', type=Path, default=ROOT / 'docs/CycleTCM-质疑查证与消融验证方案.md')
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
    prompts = archived_prompts(args.prompt_doc.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / 'records'; records_dir.mkdir(exist_ok=True)
    paths = sorted([x for x in args.images_dir.resolve().iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}], key=lambda x:x.name)
    if args.max_images:
        indices = torch.linspace(0, len(paths)-1, min(args.max_images, len(paths))).long().tolist()
        paths = [paths[i] for i in indices]
    prompt_sha = hashlib.sha256((prompts['system']+'\n'+prompts[args.variant]).encode()).hexdigest()
    metadata = {'variant': args.variant, 'prompt_doc': str(args.prompt_doc.resolve()), 'prompt_sha256': prompt_sha,
                'system_prompt': prompts['system'], 'user_prompt': prompts[args.variant],
                'model_dir': str(args.model_dir.resolve()), 'images_dir': str(args.images_dir.resolve()),
                'pooling': 'last hidden state; attention-masked sequence mean', 'add_generation_prompt': True,
                'dtype': 'torch.bfloat16' if torch.cuda.is_available() else 'torch.float32', 'seed': 42,
                'target': len(paths)}
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        if not args.resume:
            p.error('Output already exists; use --resume or a new output directory')
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError('Prompt or input metadata changed; use a new output directory')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    model_files = {p.name: sha256(p) for p in sorted(args.model_dir.resolve().iterdir())
                   if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors'}}
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'Pillow']}
    provenance_path = output / 'extraction_provenance.json'
    if args.resume and provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if any(previous['model_files'].get(k) != v for k,v in model_files.items()) or previous['versions'] != versions:
            raise ValueError('Weights, processor or library versions changed; use a new output directory')
    else:
        write_json(provenance_path, {'created_at':datetime.now().astimezone().isoformat(), 'model_files':model_files, 'versions':versions,
                   'source_files_sha256':{'extractor':sha256(__file__), 'prompt_doc':sha256(args.prompt_doc), 'uv.lock':sha256(ROOT/'uv.lock')},
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
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(str(args.model_dir.resolve()), local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model_dir.resolve()), dtype=dtype,
            device_map={'': 0} if torch.cuda.is_available() else None, local_files_only=True).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        device = next(model.parameters()).device
        started = time.monotonic()
        logging.info('EXTRACT variant=%s pending=%d target=%d', args.variant, len(pending), len(paths))
        for index, image_path in enumerate(pending):
            with Image.open(image_path) as source: image = source.convert('RGB')
            messages = [{'role':'system','content':[{'type':'text','text':prompts['system']}]},
                        {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompts[args.variant]}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
            inputs.pop('token_type_ids', None)
            input_hash = hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()
            input_shapes = {k:list(v.shape) for k,v in inputs.items() if isinstance(v,torch.Tensor)}
            inputs = {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[-1]
                mask = inputs.get('attention_mask')
                if mask is None: pooled = hidden.mean(dim=1)
                else:
                    mask = mask.to(hidden.dtype).unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = pooled.float().cpu()[0]
            if pooled.shape != (2560,) or not torch.isfinite(pooled).all(): raise ValueError(f'Invalid feature {image_path}')
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
