"""
Qwen2.5-VL-7B zero-shot inference on test_students.csv → test_predictions.csv

Features:
  - Checkpoint every 100 images (auto-resume if job killed)
  - GPU split: --rank 0 processes first half, --rank 1 second half
  - Merge: --merge combines both halves into final submission

Usage (single GPU):
    python src/predict_qwen_test.py --rank 0 --world-size 1

Usage (2 GPU):
    CUDA_VISIBLE_DEVICES=0 python src/predict_qwen_test.py --rank 0 --world-size 2
    CUDA_VISIBLE_DEVICES=1 python src/predict_qwen_test.py --rank 1 --world-size 2

Merge:
    python src/predict_qwen_test.py --merge
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info  # type: ignore
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

TEST_CSV     = Path("data/raw/test_students.csv")
IMAGE_BASE   = Path("data/raw")
LORA_PATH    = Path("outputs/lora_adapters")
OUTPUT_DIR   = Path("results/qwen_test_inference")
FINAL_OUTPUT = Path("test_predictions.csv")

PROMPT = (
    "You are estimating face occlusion on a cropped 224×224 face image.\n\n"
    "Occlusion = the FRACTION OF THE FACE SURFACE that is hidden, from 0.00 "
    "(fully visible) to 1.00 (fully hidden). What counts as hidden: anything "
    "covering the face surface — a hand, mask, glasses, scarf, object, heavy "
    "blur, deep shadow, and hair where it covers part of the face area.\n\n"
    "Estimate what fraction of the face surface is occluded.\n\n"
    'Respond ONLY as JSON:\n'
    '{"observation": "1-2 sentences", "percentage": <float 0.00-1.00>}'
)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse(raw: str) -> float:
    raw = raw.strip()
    try:
        d = json.loads(raw)
        if "percentage" in d:
            return float(np.clip(float(d["percentage"]), 0.0, 1.0))
    except Exception:
        pass
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if "percentage" in d:
                return float(np.clip(float(d["percentage"]), 0.0, 1.0))
        except Exception:
            pass
    for tok in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", raw):
        try:
            v = float(tok.group())
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return 0.10


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    lora_exists = LORA_PATH.exists() and (LORA_PATH / "adapter_config.json").exists()

    if lora_exists:
        print(f"Loading Qwen2.5-VL-7B + LoRA from {LORA_PATH} ...")
        from peft import PeftModel  # type: ignore
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(base, str(LORA_PATH))
    else:
        print("No LoRA adapter — loading Qwen2.5-VL-7B zero-shot.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )

    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model loaded.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_one(model, processor, filename: str) -> float:
    img_path = IMAGE_BASE / filename
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  ERROR opening {filename}: {e}")
        return 0.10

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    processor_kwargs: dict = dict(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    if video_inputs:
        processor_kwargs["videos"] = video_inputs

    inputs = processor(**processor_kwargs).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    return _parse(raw)


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference(rank: int, world_size: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"checkpoint_rank{rank}.csv"

    df_test = pd.read_csv(TEST_CSV)
    filenames = df_test["filename"].tolist()

    chunk_size = len(filenames) // world_size
    start = rank * chunk_size
    end   = start + chunk_size if rank < world_size - 1 else len(filenames)
    my_filenames = filenames[start:end]

    print(f"Rank {rank}/{world_size}: images {start}–{end} ({len(my_filenames)} total)")

    done: dict[str, float] = {}
    if checkpoint_path.exists():
        df_cp = pd.read_csv(checkpoint_path)
        done = dict(zip(df_cp["filename"], df_cp["pred"]))
        print(f"  Resumed: {len(done)} already done")

    remaining = [f for f in my_filenames if f not in done]
    if not remaining:
        print("  All done already.")
        return

    model, processor = load_model()

    t0 = time.time()
    for i, filename in enumerate(tqdm(remaining, desc=f"Rank {rank}")):
        done[filename] = predict_one(model, processor, filename)

        if (i + 1) % 100 == 0 or i == len(remaining) - 1:
            pd.DataFrame(
                [{"filename": f, "pred": p} for f, p in done.items()]
            ).to_csv(checkpoint_path, index=False)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(remaining) - i - 1)
            print(f"  [{i+1}/{len(remaining)}] checkpoint saved  ETA={eta/60:.0f}min")

    print(f"Rank {rank} done. Results in {checkpoint_path}")


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_and_finalize() -> None:
    df_test = pd.read_csv(TEST_CSV)
    filenames = df_test["filename"].tolist()

    all_preds: dict[str, float] = {}
    for cp in sorted(OUTPUT_DIR.glob("checkpoint_rank*.csv")):
        df_cp = pd.read_csv(cp)
        for _, row in df_cp.iterrows():
            all_preds[row["filename"]] = float(row["pred"])
        print(f"  Loaded {cp.name}: {len(df_cp)} preds")

    missing = [f for f in filenames if f not in all_preds]
    if missing:
        print(f"WARNING: {len(missing)} images missing — filling with 0.10")
        for f in missing:
            all_preds[f] = 0.10

    df_out = pd.DataFrame({
        "filename":      filenames,
        "FaceOcclusion": [all_preds[f] for f in filenames],
        "gender":        ["x"] * len(filenames),
    })
    df_out.to_csv(FINAL_OUTPUT, index=False)
    print(f"\nSubmission saved → {FINAL_OUTPUT}  ({len(df_out)} rows)")
    print(
        f"Pred stats: min={df_out['FaceOcclusion'].min():.3f}  "
        f"max={df_out['FaceOcclusion'].max():.3f}  "
        f"mean={df_out['FaceOcclusion'].mean():.3f}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global LORA_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",       type=int, default=0)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--merge",      action="store_true")
    ap.add_argument("--lora-path",  default=str(LORA_PATH))
    args = ap.parse_args()

    LORA_PATH = Path(args.lora_path)

    if args.merge:
        merge_and_finalize()
    else:
        run_inference(args.rank, args.world_size)


if __name__ == "__main__":
    main()
