"""
Inférence post-fine-tuning : chargement des adapteurs LoRA sauvegardés
+ calcul des signaux à la volée sur le test set + prédictions.

VRAM : Qwen 7B 4-bit ~8.5GB + Segformer ~0.4GB = ~8.9GB / 16.3GB.
SAM2 non utilisé ici — pas de conflit GPU.

Usage:
    uv run python src/qwen_inference.py config/qwen_inference.yaml
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.qwen_finetune import build_prompt

IMAGE_BASE   = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)
SIGNALS_CSV  = Path("data/signals_precomputed.csv")
OUTPUT_DIR   = Path("results")
SEGFORMER_REPO = "jonathandinu/face-parsing"
HAIR_CLS       = 13
FALLBACK_PRED  = 0.05

def laplacian_blur(img: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_segformer(device: str):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc  = SegformerImageProcessor.from_pretrained(SEGFORMER_REPO)
    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(SEGFORMER_REPO)
        .eval()
        .to(device)
    )
    return proc, model


def hair_pct_single(img: Image.Image, seg_proc, seg_model, device: str) -> float:
    inputs = seg_proc(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    seg = F.interpolate(logits, size=(224, 224), mode="bilinear",
                        align_corners=False).argmax(1).cpu().numpy()
    return float((seg[0] == HAIR_CLS).sum() / (224 * 224))


def mediapipe_signals(img: Image.Image) -> tuple[float, float, float, bool, str]:
    from src.inference.zero_shot_pipeline import analyze_face_pose
    pose = analyze_face_pose(np.array(img.convert("RGB")))
    if not pose.detected:
        return 0.0, 0.0, 0.0, False, "unknown"
    return pose.yaw_deg, pose.pitch_deg, pose.pose_occlusion, True, pose.orientation


def compute_signals_live(
    img: Image.Image,
    seg_proc,
    seg_model,
    device: str,
    use_segformer: bool = True,
) -> Dict[str, Any]:
    blur = laplacian_blur(img)
    yaw, _, pose_occ, _, orient = mediapipe_signals(img)
    hair = hair_pct_single(img, seg_proc, seg_model, device) if use_segformer else -1.0
    return {
        "blur_score":  round(blur, 2),
        "hair_pct":    round(hair, 4),
        "yaw_deg":     round(yaw, 1),
        "pose_occ":    round(pose_occ, 4),
        "orientation": orient,
    }

def load_finetuned_model(lora_path: str, device: str = "auto"):
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenCls  # type: ignore

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(lora_path)
    adapter_cfg = Path(lora_path) / "adapter_config.json"
    base_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base_repo = cfg.get("base_model_name_or_path", base_repo)

    print(f"Loading base model {base_repo} in 4-bit …")
    base = QwenCls.from_pretrained(
        base_repo,
        quantization_config=bnb_config,
        device_map=device,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    print(f"Applying LoRA adapters from {lora_path} …")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return model, processor


# ── Inférence image ───────────────────────────────────────────────────────────

def _parse_percentage(raw: str) -> Optional[float]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if "percentage" in data:
            return float(np.clip(float(data["percentage"]), 0.0, 1.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    depth, start = 0, -1
    for i, c in enumerate(raw):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    data = json.loads(raw[start : i + 1])
                    if "percentage" in data:
                        return float(np.clip(float(data["percentage"]), 0.0, 1.0))
                except (json.JSONDecodeError, ValueError, TypeError):
                    start = -1
    for m in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", raw):
        try:
            v = float(m.group())
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return None


def run_single(
    model,
    processor,
    img: Image.Image,
    signals: Optional[Dict[str, Any]],
    cot: bool,
    max_new_tokens: int,
) -> tuple[float, str]:
    prompt = build_prompt(signals, cot=cot)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info  # type: ignore
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

    n_input = inputs["input_ids"].shape[1]
    raw = processor.decode(out[0][n_input:], skip_special_tokens=True)
    pct = _parse_percentage(raw)
    return (pct if pct is not None else FALLBACK_PRED), raw

def report_stats(preds: np.ndarray, gt: Optional[np.ndarray], gender: Optional[np.ndarray]) -> None:
    print("\n  Distribution des prédictions :")
    print(f"    min={preds.min():.3f}  max={preds.max():.3f}"
          f"  mean={preds.mean():.3f}  median={float(np.median(preds)):.3f}"
          f"  std={preds.std():.3f}")

    mode_counts = pd.Series(np.round(preds, 1)).value_counts()
    top_mode, top_count = mode_counts.index[0], mode_counts.iloc[0]
    if top_count / len(preds) > 0.30:
        print(f"    ANCRAGE DETECTE : {top_count}/{len(preds)} preds ~ {top_mode:.1f}")

    if gt is not None and gender is not None:
        from src.utils.metrics import compute_score
        from src.zero_shot_eval import OCC_BINS
        m = compute_score(preds, gt, gender.astype(float))
        print(f"\n  challenge_score : {m['challenge_score']:.5f}  (lower = better)")
        print(f"  MAE             : {m['mae']:.5f}")
        print(f"  Err_F / Err_M   : {m['err_F']:.5f} / {m['err_M']:.5f}")
        print(f"  biais signé     : {(preds - gt).mean():+.5f}")
        print(f"\n  Par bin :")
        for level, lo, hi in OCC_BINS:
            mask = (gt >= lo) & (gt < hi)
            if mask.sum() == 0:
                continue
            p_b, g_b = preds[mask], gt[mask]
            print(f"    {level:<8} n={mask.sum():3d}  MAE={np.abs(p_b-g_b).mean():.4f}"
                  f"  bias={(p_b-g_b).mean():+.4f}")

def _get(cfg: dict, *keys: str, default=None):
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val if val is not None else default


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("config", nargs="?", default="config/qwen_inference.yaml")
    ap.add_argument("--chunk-id",  type=int, default=None)
    ap.add_argument("--n-chunks",  type=int, default=None)
    cli, _ = ap.parse_known_args()

    print(f"Config : {cli.config}")
    cfg = load_config(cli.config)

    # ── Paramètres modèle / inférence ────────────────────────────────────────
    lora_path    = str(_get(cfg, "model", "lora_path",             default="outputs/lora_adapters"))
    no_segformer = bool(_get(cfg, "model", "no_segformer",         default=False))
    cot          = bool(_get(cfg, "inference", "cot",              default=False))
    max_new_tokens_cfg = _get(cfg, "inference", "max_new_tokens",  default=None)

    # CLI écrase YAML pour chunk_id / n_chunks
    chunk_id = cli.chunk_id if cli.chunk_id is not None else int(_get(cfg, "inference", "chunk_id", default=0))
    n_chunks = cli.n_chunks if cli.n_chunks is not None else int(_get(cfg, "inference", "n_chunks", default=1))

    # ── Paramètres données ────────────────────────────────────────────────────
    gt_col       = str(_get(cfg, "data", "gt_col",             default=""))
    gender_col   = str(_get(cfg, "data", "gender_col",         default="gender"))
    test_csv     = Path(str(_get(cfg, "data", "test_csv",      default="data/test_students.csv")))
    image_base   = Path(str(_get(cfg, "data", "image_base",    default=str(IMAGE_BASE))))
    signals_csv  = Path(str(_get(cfg, "data", "signals_csv",   default=str(SIGNALS_CSV))))
    output       = Path(str(_get(cfg, "data", "output",        default="results/predictions_lora.csv")))
    failures_log = Path(str(_get(cfg, "data", "failures_log",  default="results/parse_failures_lora.csv")))
    n_limit      = int(_get(cfg, "data", "n",    default=0))
    seed         = int(_get(cfg, "data", "seed", default=42))

    # Chemins chunk-spécifiques si n_chunks > 1
    if n_chunks > 1:
        output      = output.with_name(output.stem + f"_chunk{chunk_id}" + output.suffix)
        failures_log = failures_log.with_name(failures_log.stem + f"_chunk{chunk_id}" + failures_log.suffix)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = int(max_new_tokens_cfg) if max_new_tokens_cfg is not None else (256 if cot else 64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Chargement CSV + découpage chunk ─────────────────────────────────────
    print(f"Loading {test_csv} ...")
    df = pd.read_csv(test_csv)
    if n_limit > 0:
        df = df.sample(n=min(n_limit, len(df)), random_state=seed).reset_index(drop=True)

    if n_chunks > 1:
        # Round-robin : chaque chunk couvre tous les bins d'occlusion uniformément
        df = df.iloc[chunk_id::n_chunks].reset_index(drop=True)
        print(f"  Chunk {chunk_id}/{n_chunks} : {len(df)} images → {output}")

    has_gt     = bool(gt_col) and gt_col in df.columns
    has_gender = gender_col in df.columns
    print(f"  {len(df)} images  |  GT={has_gt}  |  CoT={cot}  |  max_new_tokens={max_new_tokens}")

    done_files: set = set()
    existing_rows: list = []
    if output.exists():
        existing = pd.read_csv(output)
        done_files    = set(existing["filename"].tolist())
        existing_rows = existing.to_dict("records")
        print(f"  Reprise : {len(done_files)} images déjà traitées, {len(df) - len(done_files)} restantes")
    df = df[~df["filename"].isin(done_files)].reset_index(drop=True)

    if len(df) == 0:
        print("Toutes les images sont déjà traitées.")
        return

    signals_cache: dict = {}
    if signals_csv.exists():
        sig_df = pd.read_csv(signals_csv)
        for _, row in sig_df.iterrows():
            signals_cache[row["filename"]] = {
                "blur_score":  float(row.get("blur_score",  -1)),
                "hair_pct":    float(row.get("hair_pct",    -1)),
                "yaw_deg":     float(row.get("yaw_deg",      0)),
                "pose_occ":    float(row.get("pose_occ",     0)),
                "orientation": str(row.get("orientation", "unknown")),
            }
        cached = sum(1 for f in df["filename"] if f in signals_cache)
        print(f"  Signaux précalculés : {len(signals_cache)} entrées ({cached}/{len(df)} images couvertes)")
        print(f"  Signaux à calculer à la volée : {len(df) - cached}")
    else:
        print(f"  {signals_csv} absent — tous les signaux calculés à la volée")

    seg_proc = seg_model = None
    if not no_segformer:
        print(f"Loading Segformer on {device} (~400MB) ...")
        seg_proc, seg_model = load_segformer(device)
        print("Segformer ready.")

    print(f"\nLoading fine-tuned model from {lora_path} ...")
    model, processor = load_finetuned_model(lora_path)

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f"VRAM apres chargement : {info.used/1024**3:.1f} GB / {info.total/1024**3:.1f} GB")
    except Exception:
        pass

    failures: list = []
    new_rows: list = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    DEBUG_N = 3  

    with progress:
        task = progress.add_task("Inference LoRA", total=len(df))
        for idx, row in enumerate(df.itertuples(index=False)):
            img_path = image_base / row.filename
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                failures.append({"filename": row.filename, "error": f"image load: {e}", "raw": ""})
                new_rows.append({
                    "filename":   row.filename,
                    "prediction": FALLBACK_PRED,
                    **({"gender": int(getattr(row, gender_col, 0))} if has_gender else {}),
                    **({"gt":     float(getattr(row, gt_col, 0))}   if has_gt     else {}),
                })
                progress.advance(task)
                continue

            if row.filename in signals_cache:
                signals = signals_cache[row.filename]
            else:
                signals = compute_signals_live(
                    img, seg_proc, seg_model, device,
                    use_segformer=(not no_segformer),
                )

            if idx == 0:
                prompt_shown = build_prompt(signals, cot=cot)
                print(f"\n{'='*60}")
                print(f"DEBUG — PROMPT image 0 (cot={cot}) :")
                print(prompt_shown)
                print(f"{'='*60}\n")

            try:
                pred, raw = run_single(model, processor, img, signals, cot, max_new_tokens)
                if idx < DEBUG_N:
                    print(f"\n{'='*60}")
                    print(f"DEBUG — RAW OUTPUT image {idx} :")
                    print(repr(raw))
                    print(f"  → parsed: {_parse_percentage(raw)}  pred={pred}")
                    print(f"{'='*60}\n")
                if _parse_percentage(raw) is None:
                    failures.append({"filename": row.filename, "error": "parse_fail", "raw": raw[:300]})
            except Exception as e:
                failures.append({"filename": row.filename, "error": str(e), "raw": ""})
                pred = FALLBACK_PRED

            rec: dict = {"filename": row.filename, "prediction": round(pred, 4)}
            if has_gender:
                rec["gender"] = int(getattr(row, gender_col, 0))
            if has_gt:
                rec["gt"] = float(getattr(row, gt_col, 0))

            new_rows.append(rec)

            if len(new_rows) % 50 == 0:
                pd.DataFrame(existing_rows + new_rows).to_csv(output, index=False)

            progress.advance(task)

    df_out = pd.DataFrame(existing_rows + new_rows)
    df_out.to_csv(output, index=False)
    print(f"\nPredictions -> {output}  ({len(df_out)} lignes)")

    if failures:
        pd.DataFrame(failures).to_csv(failures_log, index=False)
        print(f"Parse failures ({len(failures)}) -> {failures_log}")

    preds  = df_out["prediction"].values.astype(float)
    gt_arr = df_out["gt"].values.astype(float)     if has_gt     else None
    gend   = df_out["gender"].values.astype(float) if has_gender else None
    report_stats(preds, gt_arr, gend)


if __name__ == "__main__":
    main()
