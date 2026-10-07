"""Local Qwen3-VL / MedGemma loading and model-specific image inputs."""

import json
from pathlib import Path

import torch


def backbone_info(model_dir):
    config = json.loads((Path(model_dir) / 'config.json').read_text())
    model_type = config['model_type']
    if model_type not in ('qwen3_vl', 'gemma3'):
        raise ValueError(f'Unsupported MLLM model_type: {model_type}')
    return model_type, config['text_config']['hidden_size']


def load_backbone(model_dir, dtype, device):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration

    model_type, _ = backbone_info(model_dir)
    model_class = Gemma3ForConditionalGeneration if model_type == 'gemma3' else Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = model_class.from_pretrained(
        str(model_dir), dtype=dtype, device_map={'': str(device)}, local_files_only=True).eval()
    model.requires_grad_(False)
    return processor, model


def image_inputs(processor, model_type, system_prompt, user_prompt, image):
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
        {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                     {'type': 'text', 'text': user_prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    # Gemma's image tokens need their bidirectional attention mask.
    if model_type == 'qwen3_vl':
        inputs.pop('token_type_ids', None)
    return inputs


def model_inputs(inputs, model):
    parameter = next(model.parameters())
    return {key: value.to(device=parameter.device,
                          dtype=parameter.dtype if model.config.model_type == 'gemma3' and value.is_floating_point() else value.dtype)
            if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
