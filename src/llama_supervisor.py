"""
Test superviseur : Llama 3.2 Vision 11B (4-bit) + Qwen 7B décrivent 150 images.
Mesure : coverage_amount corrèle-t-il avec err_D3 mieux que r=0.17 (plafond luminosité) ?

Usage:
    uv run python src/llama_supervisor.py --model llama   # Llama seul
    uv run python src/llama_supervisor.py --model qwen    # Qwen seul
    uv run python src/llama_supervisor.py --model both    # séquentiel (défaut)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.zero_shot_eval import OCC_BINS

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)

LLAMA_REPO = "unsloth/Llama-3.2-11B-Vision-Instruct"
QWEN_REPO  = "Qwen/Qwen2.5-VL-7B-Instruct"

SUPERVISOR_PROMPT = (
    "Describe this face photo for an automatic analysis system. Answer:\n"
    "- coverage: is something covering part of the face (hat, text/overlay, object, hand, hair)? what?\n"
    "- coverage_amount: none | little | half | most\n"
    "- pose: frontal | turned\n"
    'Respond ONLY as JSON: {"coverage_type":"...","coverage_amount":"none|little|half|most","pose":"..."}'
)

AMOUNT_MAP = {"none": 0.0, "little": 0.33, "half": 0.66, "most": 1.0}
VALID_AMOUNTS = set(AMOUNT_MAP.keys())


# ── Parsers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try: return json.loads(text[start:i+1])
                except json.JSONDecodeError: start = -1
    return None


def parse_supervisor(raw: str) -> tuple[str, float, str]:
    """Returns (coverage_type, coverage_amount_float, pose)."""
    data = _extract_json(raw)
    if data:
        amt_str = str(data.get("coverage_amount", "none")).lower().strip()
        if amt_str not in VALID_AMOUNTS:
            for v in VALID_AMOUNTS:
                if v in amt_str:
                    amt_str = v; break
            else:
                amt_str = "none"
        ctype = str(data.get("coverage_type", "none")).lower().strip()[:50]
        pose  = str(data.get("pose", "frontal")).lower().strip()
        return ctype, AMOUNT_MAP[amt_str], pose
    # fallback: scan text
    text = raw.lower()
    amt = "none"
    for v in ("most", "half", "little", "none"):
        if v in text:
            amt = v; break
    return "unknown", AMOUNT_MAP[amt], "frontal"


# ── Sample ────────────────────────────────────────────────────────────────────

def build_testlike(seed=7, n_per_bin=50) -> pd.DataFrame:
    df = pd.read_csv("data/train.csv").dropna(
        subset=["filename", "FaceOcclusion", "gender"]
    )
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(0.5)
    df = df[df["gender"].isin([0, 1])]
    rng = np.random.default_rng(seed)
    parts = []
    for _, lo, hi in OCC_BINS:
        pool = df[(df["FaceOcclusion"] >= lo) & (df["FaceOcclusion"] < hi)]
        k = min(n_per_bin, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Llama runner ──────────────────────────────────────────────────────────────

def run_llama(sample: pd.DataFrame, d3_csv: Path) -> pd.DataFrame:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    print(f"Loading Llama 3.2 Vision 11B (4-bit) from {LLAMA_REPO} …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(LLAMA_REPO)
    model = MllamaForConditionalGeneration.from_pretrained(
        LLAMA_REPO,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    print(f"Llama loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} Go\n")

    records = []
    n = len(sample)
    t0 = time.time()
    for i, row in enumerate(sample.itertuples(index=False), 1):
        if i == 1 or i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Llama {i:3d}/{n}  elapsed={elapsed:.0f}s  ETA={(elapsed/i*(n-i)):.0f}s",
                  flush=True)
        img = Image.open(IMAGE_BASE / row.filename).convert("RGB")
        # Llama 3.2 Vision: image in content, NO system role
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": SUPERVISOR_PROMPT},
        ]}]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ctype, camount, pose = parse_supervisor(raw)
        records.append({
            "filename":   row.filename,
            "llama_type": ctype,
            "llama_amt":  camount,
            "llama_pose": pose,
            "llama_raw":  raw[:100],
        })

    del model; torch.cuda.empty_cache()
    return pd.DataFrame(records)


# ── Qwen runner ───────────────────────────────────────────────────────────────

def run_qwen(sample: pd.DataFrame) -> pd.DataFrame:
    from src.inference.zero_shot_pipeline import QwenEstimator

    print(f"Loading Qwen 2.5-VL-7B …")
    qwen = QwenEstimator(QWEN_REPO, device="auto")
    print("Qwen loaded.\n")

    records = []
    n = len(sample)
    t0 = time.time()
    for i, row in enumerate(sample.itertuples(index=False), 1):
        if i == 1 or i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Qwen {i:3d}/{n}  elapsed={elapsed:.0f}s  ETA={(elapsed/i*(n-i)):.0f}s",
                  flush=True)
        img = Image.open(IMAGE_BASE / row.filename).convert("RGB")
        raw = qwen._run_qwen(img, SUPERVISOR_PROMPT, max_new_tokens=80)
        ctype, camount, pose = parse_supervisor(raw)
        records.append({
            "filename":  row.filename,
            "qwen_type": ctype,
            "qwen_amt":  camount,
            "qwen_pose": pose,
            "qwen_raw":  raw[:100],
        })

    del qwen; torch.cuda.empty_cache()
    return pd.DataFrame(records)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, prefix: str) -> None:
    amt_col  = f"{prefix}_amt"
    type_col = f"{prefix}_type"
    pose_col = f"{prefix}_pose"
    err_col  = "err_d3"
    gt_col   = "gt"

    SEP  = "=" * 68
    LINE = "─" * 68
    print(f"\n{SEP}")
    print(f"  ANALYSE — {prefix.upper()}")
    print(SEP)

    # Distribution
    print(f"  coverage_amount :")
    for v in ["none","little","half","most"]:
        mask = df[amt_col] == AMOUNT_MAP[v]
        n_v = mask.sum()
        print(f"    {v:<8} n={n_v:3d}  ({100*n_v/len(df):.1f}%)")

    # Corrélations
    print(f"\n  Corrélations :")
    for target, label in [(err_col, "err_D3"), (gt_col, "gt")]:
        r = float(np.corrcoef(df[amt_col].fillna(0), df[target].fillna(0))[0,1])
        print(f"    {prefix}_amt ↔ {label:<8} : r = {r:+.3f}")

    # err_D3 par coverage_type
    print(f"\n{LINE}")
    print(f"  err_D3 moyen par coverage_type (top 8) :")
    type_stats = (df.groupby(type_col)[err_col]
                    .agg(["count","mean","median"])
                    .sort_values("count", ascending=False)
                    .head(8))
    for ct, row in type_stats.iterrows():
        print(f"    {ct:<20} n={int(row['count']):3d}  err_mean={row['mean']:.4f}  median={row['median']:.4f}")


def compare(df: pd.DataFrame) -> None:
    SEP  = "=" * 68
    LINE = "─" * 68
    if "llama_amt" not in df.columns or "qwen_amt" not in df.columns:
        return

    print(f"\n{SEP}")
    print("  COMPARAISON Llama vs Qwen")
    print(SEP)

    # Accord sur coverage_amount (quantized to 4 classes)
    def to_class(v):
        for cls, val in AMOUNT_MAP.items():
            if abs(v - val) < 0.01:
                return cls
        return "unknown"

    df["l_cls"] = df["llama_amt"].apply(to_class)
    df["q_cls"] = df["qwen_amt"].apply(to_class)
    accord = (df["l_cls"] == df["q_cls"]).mean()
    print(f"  Accord coverage_amount Llama↔Qwen : {100*accord:.1f}%")
    print(f"  Accord pose Llama↔Qwen            : {100*(df['llama_pose']==df['qwen_pose']).mean():.1f}%")

    # Divergences les plus grandes (|llama_amt - qwen_amt| max)
    df["div_amt"] = (df["llama_amt"] - df["qwen_amt"]).abs()
    worst_div = df.nlargest(10, "div_amt")[
        ["filename","gt","err_d3","llama_amt","qwen_amt","llama_type","qwen_type","div_amt"]
    ]
    print(f"\n{LINE}")
    print("  Top divergences Llama≠Qwen :")
    print(f"  {'filename':<35}  {'gt':>5}  {'err':>5}  {'L_amt':>6}  {'Q_amt':>6}  L_type → Q_type")
    for _, r in worst_div.iterrows():
        print(f"  {Path(r['filename']).name:<35}  {r['gt']:>5.3f}  {r['err_d3']:>5.3f}"
              f"  {r['llama_amt']:>6.2f}  {r['qwen_amt']:>6.2f}  {r['llama_type'][:15]} → {r['qwen_type'][:15]}")

    # Verdict
    r_llama_err = float(np.corrcoef(df["llama_amt"], df["err_d3"])[0,1])
    r_qwen_err  = float(np.corrcoef(df["qwen_amt"],  df["err_d3"])[0,1])
    r_llama_gt  = float(np.corrcoef(df["llama_amt"], df["gt"])[0,1])
    r_qwen_gt   = float(np.corrcoef(df["qwen_amt"],  df["gt"])[0,1])
    print(f"\n{SEP}")
    print("  VERDICT FINAL")
    print(SEP)
    print(f"  coverage_amt ↔ err_D3 :  Llama r={r_llama_err:+.3f}   Qwen r={r_qwen_err:+.3f}")
    print(f"  coverage_amt ↔ gt      :  Llama r={r_llama_gt:+.3f}    Qwen r={r_qwen_gt:+.3f}")
    print(f"  Plafond luminosité       : r = +0.164")
    best_err = max(abs(r_llama_err), abs(r_qwen_err))
    if best_err > 0.164:
        best = "Llama" if abs(r_llama_err) > abs(r_qwen_err) else "Qwen"
        print(f"  ✓ {best} dépasse le plafond luminosité sur err_D3 (r={best_err:+.3f} > 0.164)")
    else:
        print(f"  ✗ Ni Llama ni Qwen ne dépassent le plafond (max r={best_err:.3f})")
        print(f"    → Le plafond de D3 N'EST PAS expliqué sémantiquement par ces VLMs.")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama","qwen","both"], default="both")
    args = ap.parse_args()

    out_dir = Path("results/zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load err_D3
    d3_csv = out_dir / "face_area_testlike.csv"
    d3 = pd.read_csv(d3_csv)[["filename","gt","pred_d3","gender"]]
    d3["err_d3"] = (d3["gt"] - d3["pred_d3"]).abs()

    sample = build_testlike()
    base = d3.merge(sample[["filename"]], on="filename")

    results_list = [base]

    if args.model in ("llama", "both"):
        df_llama = run_llama(sample, d3_csv)
        df_llama.to_csv(out_dir / "supervisor_llama.csv", index=False)
        print(f"Saved → {out_dir}/supervisor_llama.csv")
        results_list.append(df_llama)
        df_merged = pd.concat([r.set_index("filename") for r in results_list], axis=1).reset_index()
        analyze(df_merged, "llama")

    if args.model in ("qwen", "both"):
        df_qwen = run_qwen(sample)
        df_qwen.to_csv(out_dir / "supervisor_qwen.csv", index=False)
        print(f"Saved → {out_dir}/supervisor_qwen.csv")
        results_list.append(df_qwen)
        df_merged = pd.concat([r.set_index("filename") for r in results_list], axis=1).reset_index()
        analyze(df_merged, "qwen")

    if args.model == "both":
        compare(df_merged)


if __name__ == "__main__":
    main()
