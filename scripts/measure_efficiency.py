"""Measure Qwen feature extraction and visual/full forward latency on this host."""
import argparse, json, time
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--images-dir', type=Path, required=True)
    p.add_argument('--visual-checkpoint', type=Path, required=True)
    p.add_argument('--full-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iterations', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(8)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model_dir), dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        device_map={'': 0} if device.type == 'cuda' else None, local_files_only=True).eval()
    image_paths = sorted([p for p in args.images_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}])[:args.iterations]
    import sys; sys.path.insert(0,str(ROOT/'src'))
    from utils.mllm_feature_extract import SYSTEM_PROMPT, TCM_PRIOR, USER_PROMPT, _inputs_to_device
    qwen_times=[]
    for i, path in enumerate(image_paths):
        with Image.open(path) as source: image=source.convert('RGB')
        messages=[{'role':'system','content':[{'type':'text','text':SYSTEM_PROMPT+'\n'+TCM_PRIOR}]},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':USER_PROMPT}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt'); inputs.pop('token_type_ids',None)
        inputs=_inputs_to_device(inputs,next(qwen.parameters()).device)
        with torch.inference_mode():
            if device.type == 'cuda': torch.cuda.synchronize()
            start=time.perf_counter(); qwen(**inputs,output_hidden_states=True,use_cache=False)
            if device.type == 'cuda': torch.cuda.synchronize()
            qwen_times.append((time.perf_counter()-start)*1000)
    from train.reproduce import build_model
    def visual_time(checkpoint, model_name):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False); cfg=state['config']; cfg['model']=model_name
        model=build_model(cfg).to(device).eval(); model.load_state_dict(state['model']);
        x=[torch.rand(1,3,224,224,device=device) for _ in range(7)]
        feature=torch.rand(1,2560,device=device)
        values=[]
        for _ in range(args.warmup):
            with torch.inference_mode(): model(*x,feature) if model_name=='full' else model(*x)
        for _ in range(args.iterations):
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.inference_mode(): model(*x,feature) if model_name=='full' else model(*x)
            if device.type=='cuda': torch.cuda.synchronize()
            values.append((time.perf_counter()-start)*1000)
        return {'mean_ms':sum(values)/len(values),'values_ms':values,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None}
    result={'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','qwen_feature_batch1':{'mean_ms':sum(qwen_times)/len(qwen_times),'values_ms':qwen_times,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None},'visual_forward':visual_time(args.visual_checkpoint,'visual'),'full_visual_plus_adapter_forward':visual_time(args.full_checkpoint,'full')}
    result['full_to_visual_latency_ratio']=result['full_visual_plus_adapter_forward']['mean_ms']/result['visual_forward']['mean_ms']
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
