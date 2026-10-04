"""
Extract features from processed whole-tongue images using Qwen3-VL-4B-Instruct.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import MLLM_FEATURES_FILE, PROCESSED_DATA_DIR

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
        str(Path.home() / ".cache/modelscope/hub/Qwen/Qwen3-VL-4B-Instruct"),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=_default_model_dir(), help="Local model directory")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR / "images"),
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MLLM_FEATURES_FILE),
        help="Output JSON path",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Process only the first N images; 0 means all")
    parser.add_argument(
        "--save-full-sequence",
        action="store_true",
        help="Write full [seq, hidden] to JSON (very large; for small-scale debugging only)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Download weights first, or set QWEN3_VL_MODEL_DIR to your Qwen3-VL-4B-Instruct folder."
        )

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)

    device = next(model.parameters()).device
    paths = _collect_images(images_dir)
    if args.max_images > 0:
        paths = paths[: args.max_images]

    records: list[dict] = []
    for img_path in paths:
        image = Image.open(img_path).convert("RGB")
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT + "\n" + TCM_PRIOR}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = _inputs_to_device(inputs, device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        if not outputs.hidden_states:
            raise RuntimeError(
                "Model returned no hidden_states; upgrade transformers or check the Qwen3-VL implementation."
            )

        last_hidden = outputs.hidden_states[-1]
        print(f"{img_path.name}\tlast_hidden_states[-1] shape: {tuple(last_hidden.shape)}")

        attn = inputs.get("attention_mask")
        if args.save_full_sequence:
            # [S, H] — can still be very large
            vec = last_hidden[0].float().cpu()
            qwen_feature = vec.tolist()
        else:
            # [1,S,H] -> mean over S -> [1,H]; JSON stores a length-H list
            pooled_bh = _last_hidden_pooled(last_hidden, attn)
            print(f"{img_path.name}\tpooled (mean over seq) shape: {tuple(pooled_bh.shape)}")
            qwen_feature = pooled_bh[0].float().cpu().tolist()

        records.append(
            {
                "image_file": img_path.name,
                "qwen_feature": qwen_feature,
            }
        )

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
