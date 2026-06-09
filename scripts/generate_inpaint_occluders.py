"""Generate a *photorealistic* synthetic occluder dataset via diffusion inpainting.

Motivation
    The geometric generator (`generate_synthetic_occluders.py`) paints flat
    black rectangles / ellipses and solid-colour "masks". Inspecting the real
    high-occlusion images of this dataset shows the actual occluders are
    almost never censor bars or medical masks — they are:

        - HAIR strands / fringe falling across the forehead and cheeks
        - HANDS / fingers over the face
        - HEADWEAR (caps, hats, surgical caps, head-scarves) over the forehead
        - (image-quality degradation — handled separately, not here)

    A model trained on flat black blobs learns "detect black", not "recognise
    a strand of hair". This script instead inpaints a *realistic* occluder of
    the requested class into a controlled mask region.

Why inpainting fits a REGRESSION task
    The label is `FaceOcclusion` = fraction of the face surface occluded. With
    free-form generation the label is unknown. With inpainting WE define the
    mask, so:

        new_label = original_label + area(mask) / area(image)      (capped 1.0)

    is exact (same image≈face approximation as the aligned 224px crops use).
    The diffusion model only fills colour/texture INSIDE the mask; the masked
    fraction — hence the label — is fully under our control. Un-masked pixels
    are re-composited from the original so the visible face is pixel-identical.

Bias toward the high-occlusion tail
    The official metric weights high occlusion (`w = 1/30 + y`) and there is a
    severe train→test shift (train median ≈ 0.05; only a handful of train
    images above 0.6). So the area is sampled mostly in [--area-min, --area-max]
    (default 0.30–0.70) to manufacture the rare high-occlusion examples the
    train set lacks.

USAGE
    # Defaults: 10k samples, hair/hand/headwear, area 0.30-0.70, SD2-inpaint.
    python scripts/generate_inpaint_occluders.py \\
        --train-csv ../occlusion_datasets/train.csv \\
        --image-dir data/raw/Crop_224_5fp_100K \\
        --output-dir data/synthetic/occluder_inpaint

    # Heavier / different mix:
    python scripts/generate_inpaint_occluders.py \\
        --n-samples 20000 --types hair hand --area-min 0.40 --area-max 0.75

OUTPUTS  (identical schema to generate_synthetic_occluders.py → drop-in)
    <output_dir>/
        images/syn_000000.webp ...
        synthetic.csv   columns:
            filename, FaceOcclusion, gender, source_filename,
            occluder_type, actual_area, original_label

INTEGRATION
    Point a training YAML at the CSV (v19 concatenates it automatically):
        data:
          data_csv: data/raw/train.csv
          extra_train_csv: data/synthetic/occluder_inpaint/synthetic.csv

NOTES
    - Runs on GPU (cluster). CPU works but is very slow — use the sbatch
      script `scripts/generate_inpaint_occluders_1xA100.sh`.
    - Determinism: seeds python/numpy/torch with --seed; per-image torch
      generators are derived from it. Same seed → same dataset.
    - The CSV is checkpointed every --checkpoint-every images so a crash does
      not lose finished work.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

WORK = 512  # diffusion working resolution (SD2-inpaint native)
OUT = 224   # dataset crop resolution


# ---------------------------------------------------------------------------
# Mask generators — each returns a binary PIL "L" mask at WORK×WORK (255 = the
# region to occlude / inpaint) shaped plausibly for its occluder class, sized
# so its covered area ≈ target_area_frac of the image. The CALLER measures the
# actual covered fraction and uses THAT as the label.
# ---------------------------------------------------------------------------

def _smooth_noise(n: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """1-D zero-mean smooth noise in roughly [-1, 1], length n."""
    out = np.zeros(n)
    for o in range(octaves):
        freq = 2 ** o
        phase = rng.uniform(0, 2 * np.pi)
        amp = 1.0 / (o + 1)
        out += amp * np.sin(np.linspace(0, freq * np.pi, n) + phase)
    out -= out.mean()
    m = np.abs(out).max() or 1.0
    return out / m


def mask_hair(rng: np.random.Generator, target: float) -> Image.Image:
    """Hair: a curtain from the top and/or a side, with a ragged lower edge
    and a few thin strands drooping further down."""
    H = W = WORK
    grid_y = np.arange(H)[:, None]
    mask = np.zeros((H, W), dtype=bool)

    style = rng.choice(["top", "side", "diagonal"], p=[0.5, 0.25, 0.25])
    noise = _smooth_noise(W, rng) * (H * 0.12)

    if style == "top":
        base = target * H
        boundary = base + noise                       # per-column lower edge
        mask |= grid_y < boundary[None, :]
    elif style == "side":
        # vertical curtain on one side; boundary is a per-row x edge
        ynoise = _smooth_noise(H, rng) * (W * 0.12)
        width = target * W
        if rng.random() < 0.5:
            edge = width + ynoise
            mask |= np.arange(W)[None, :] < edge[:, None]
        else:
            edge = (W - width) + ynoise
            mask |= np.arange(W)[None, :] > edge[:, None]
    else:  # diagonal sweep across the face
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        ang = rng.uniform(-0.6, 0.6)
        d = yy - ang * xx
        thr = np.quantile(d, target) + _smooth_noise(W, rng)[None, :] * (H * 0.05)
        mask |= d < thr

    # a few drooping strands below the main mass
    for _ in range(rng.integers(2, 6)):
        x0 = rng.integers(0, W)
        thick = rng.integers(4, 14)
        length = rng.integers(int(H * 0.15), int(H * 0.45))
        start = int(np.argmin(mask[:, min(x0, W - 1)][::-1]))  # near current edge
        start = max(0, H - start)
        wob = (_smooth_noise(length, rng) * thick * 1.5).astype(int)
        for k in range(length):
            y = start + k
            if y >= H:
                break
            xa = np.clip(x0 + wob[k] - thick // 2, 0, W - 1)
            xb = np.clip(x0 + wob[k] + thick // 2, 0, W - 1)
            mask[y, xa:xb + 1] = True

    return _to_pil(mask, feather=2)


def mask_headwear(rng: np.random.Generator, target: float) -> Image.Image:
    """Cap / hat / surgical cap: a band from the top with a gently curved
    (brim-like) lower edge — cleaner than hair."""
    H = W = WORK
    x = np.linspace(-1, 1, W)
    curve = (x ** 2 - x.mean()) * (H * rng.uniform(0.04, 0.12))  # smile/frown brim
    curve *= rng.choice([-1, 1])
    base = target * H
    boundary = base + curve + _smooth_noise(W, rng) * (H * 0.03)
    grid_y = np.arange(H)[:, None]
    mask = grid_y < boundary[None, :]
    return _to_pil(mask, feather=2)


def mask_scarf(rng: np.random.Generator, target: float) -> Image.Image:
    """Scarf / lower-face covering: a band rising from the bottom over the
    chin / mouth / nose, with a wavy upper edge."""
    H = W = WORK
    top = (1.0 - target) * H
    boundary = top + _smooth_noise(W, rng) * (H * 0.06)
    grid_y = np.arange(H)[:, None]
    mask = grid_y > boundary[None, :]
    # occasionally wrap up the sides a little
    if rng.random() < 0.4:
        side = int(W * rng.uniform(0.08, 0.16))
        up = int(H * rng.uniform(0.1, 0.25))
        mask[int(top) - up:, :side] = True
        mask[int(top) - up:, W - side:] = True
    return _to_pil(mask, feather=2)


def mask_hand(rng: np.random.Generator, target: float) -> Image.Image:
    """Hand: a palm ellipse plus 4 finger capsules and a thumb, rotated and
    placed over a random face region, scaled to hit the target area."""
    H = W = WORK
    target_px = target * H * W

    # build the hand in a local canvas, then scale to area, rotate, place
    canvas = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(canvas)
    cx, cy = W // 2, int(W * 0.62)
    palm_w, palm_h = int(W * 0.34), int(W * 0.30)
    d.ellipse((cx - palm_w // 2, cy - palm_h // 2, cx + palm_w // 2, cy + palm_h // 2), fill=255)
    n_fingers = 4
    spread = rng.uniform(0.10, 0.22)
    for i in range(n_fingers):
        off = (i - (n_fingers - 1) / 2) * spread * W
        fx = int(cx + off)
        flen = int(W * rng.uniform(0.26, 0.36))
        fw = int(W * rng.uniform(0.055, 0.075))
        ftop = cy - palm_h // 2 - flen
        d.rounded_rectangle((fx - fw // 2, ftop, fx + fw // 2, cy - palm_h // 4),
                            radius=fw // 2, fill=255)
    # thumb
    tw = int(W * 0.07)
    d.rounded_rectangle((cx - palm_w // 2 - tw, cy - tw, cx - palm_w // 4, cy + int(W * 0.12)),
                        radius=tw // 2, fill=255)

    arr = np.asarray(canvas) > 0
    cur = arr.sum()
    if cur == 0:
        return Image.new("L", (W, W), 0)
    scale = float(np.sqrt(target_px / cur))
    new = max(8, min(int(W * scale), int(W * 1.6)))
    hand = canvas.resize((new, new), Image.BILINEAR)
    hand = hand.rotate(rng.uniform(-50, 50), expand=True, resample=Image.BILINEAR)

    out = Image.new("L", (W, W), 0)
    # bias placement toward the centre / lower face
    px = int(rng.uniform(0.15, 0.85) * W - hand.width / 2)
    py = int(rng.uniform(0.25, 0.75) * H - hand.height / 2)
    out.paste(hand, (px, py), hand)
    arr = np.asarray(out) > 64
    return _to_pil(arr, feather=2)


def _to_pil(mask_bool: np.ndarray, feather: int = 0) -> Image.Image:
    img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    if feather:
        img = img.filter(ImageFilter.GaussianBlur(feather))
    return img


MASK_FUNCS = {
    "hair": mask_hair,
    "hand": mask_hand,
    "headwear": mask_headwear,
    "scarf": mask_scarf,
}

# Prompt bank per class (a variant is sampled per image for diversity).
PROMPTS = {
    "hair": [
        "long hair strands falling across the face, natural hair covering part of the face, photorealistic, detailed hair texture",
        "messy fringe of hair over the forehead and cheeks, realistic hair, sharp focus",
        "wavy hair draped across the face, photorealistic portrait, natural lighting",
    ],
    "hand": [
        "a hand covering part of the face, fingers over the face, realistic skin texture, photorealistic",
        "fingers pressed against the face, hand partially blocking the face, natural skin, sharp focus",
        "an open hand in front of the face, realistic palm and fingers, photorealistic",
    ],
    "headwear": [
        "a cap pulled down over the forehead, hat brim shadowing the upper face, photorealistic",
        "a person wearing a surgical cap covering the hair and forehead, realistic fabric, sharp focus",
        "a knitted beanie low over the brow, realistic wool texture, photorealistic",
    ],
    "scarf": [
        "a scarf wrapped over the lower face, fabric covering the mouth and chin, photorealistic, detailed cloth",
        "a person with a head-scarf covering the lower part of the face, realistic fabric folds, sharp focus",
    ],
}
NEG_PROMPT = ("deformed, distorted, disfigured, extra fingers, mutated hands, "
              "blurry, low quality, lowres, text, watermark, signature, frame, cartoon, painting")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_pipeline(model: str, device: str, dtype):
    """Load an inpainting pipeline. Auto-selects the SDXL class for *-xl-* models."""
    is_xl = "xl" in model.lower()
    if is_xl:
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, torch_dtype=dtype)
    else:
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # Memory savings so SDXL@1024 fits a 24GB 3090 (no a100 on this cluster).
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-csv", default="../occlusion_datasets/train.csv")
    ap.add_argument("--image-dir", default="data/raw/Crop_224_5fp_100K",
                    help="Directory the train.csv `filename` column resolves against.")
    ap.add_argument("--output-dir", default="data/synthetic/occluder_inpaint")
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--clean-label-max", type=float, default=0.05,
                    help="Only use source images with FaceOcclusion strictly below this.")
    ap.add_argument("--area-min", type=float, default=0.30,
                    help="Lower bound on the occluder area fraction (biased high to fill the tail).")
    ap.add_argument("--area-max", type=float, default=0.70)
    # Hands are kept in the mix but are the hardest type: SD1.5/dreamshaper
    # render them unreliably (no-hand / black-bar failures). Two safety nets
    # make them viable — (1) SDXL handles hand anatomy far better, and (2) the
    # quality guard below rejects + retries the failures. Validate hands in the
    # cluster smoke test before the full run.
    ap.add_argument("--types", nargs="+", default=["hair", "headwear", "scarf", "hand"],
                    choices=list(MASK_FUNCS.keys()))
    ap.add_argument("--model", default="Lykon/dreamshaper-8-inpainting",
                    help="Ungated inpainting model. Default: dreamshaper-8 (realistic, fast, 512px). "
                         "Best quality: diffusers/stable-diffusion-xl-1.0-inpainting-0.1 "
                         "(set --work-size 1024). stabilityai/* repos are gated (need an HF token).")
    ap.add_argument("--work-size", type=int, default=512,
                    help="Diffusion working resolution. Use 512 for SD1.5/dreamshaper, 1024 for SDXL.")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=90, help="WebP quality.")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    # --- quality guard: reject failed inpaints before they reach the dataset ---
    ap.add_argument("--no-quality-guard", action="store_true",
                    help="Disable the automatic rejection of black-bar / no-occluder failures.")
    ap.add_argument("--guard-dark", type=float, default=0.12,
                    help="Reject if the masked region mean brightness is below this AND it is flat.")
    ap.add_argument("--guard-flat", type=float, default=0.06,
                    help="Std-dev threshold below which a dark masked region is deemed a black-bar fill.")
    ap.add_argument("--guard-change", type=float, default=0.045,
                    help="Reject if the masked region barely changed vs the original (no occluder drawn).")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Re-attempt a rejected sample this many times (fresh mask/seed) before dropping it.")
    args = ap.parse_args()

    import torch

    global WORK
    WORK = args.work_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[init] device={device} dtype={dtype} model={args.model}")
    if device == "cpu":
        print("[warn] running on CPU — this is very slow; prefer the cluster sbatch.")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    clean = df[df["FaceOcclusion"] < args.clean_label_max].copy()
    print(f"[load] {len(df)} train rows, {len(clean)} clean (FaceOcclusion < {args.clean_label_max})")
    if len(clean) == 0:
        sys.exit("ERROR: no clean source samples — relax --clean-label-max")

    replace = len(clean) < args.n_samples
    sampled = clean.sample(n=args.n_samples, replace=replace, random_state=args.seed).reset_index(drop=True)
    if replace:
        print(f"[sample] sampling with replacement ({args.n_samples} > {len(clean)} clean sources)")

    pipe = build_pipeline(args.model, device, dtype)

    rows: list[dict] = []
    n_skipped = n_retried = n_dropped = 0

    def flush_csv():
        pd.DataFrame(rows).to_csv(output_dir / "synthetic.csv", index=False)

    def build_job(idx, init, orig, gender, source, otype, retry=0):
        target = float(rng.uniform(args.area_min, args.area_max))
        mask = MASK_FUNCS[otype](rng, target)
        area = float((np.asarray(mask) > 127).mean())   # actual covered fraction
        return {
            "idx": idx, "init": init, "mask": mask, "type": otype, "retry": retry,
            "prompt": random.choice(PROMPTS[otype]), "area": area, "orig": orig,
            "new_label": min(orig + area, 1.0), "gender": gender, "source": source,
        }

    def passes_guard(gen_w: Image.Image, init: Image.Image, mask: Image.Image) -> bool:
        """Reject black-bar fills and 'no occluder drawn' results inside the mask."""
        if args.no_quality_guard:
            return True
        mb = np.asarray(mask.resize((WORK, WORK)).convert("L")) > 127
        if mb.sum() == 0:
            return False
        g = np.asarray(gen_w.convert("L"), dtype=np.float32) / 255.0
        o = np.asarray(init.convert("L"), dtype=np.float32) / 255.0
        reg, oreg = g[mb], o[mb]
        # black-bar: masked region is both very dark AND very flat
        if reg.mean() < args.guard_dark and reg.std() < args.guard_flat:
            return False
        # no-occluder: masked region barely changed vs the original face
        if np.abs(reg - oreg).mean() < args.guard_change:
            return False
        return True

    def run_batch(b: list[dict]) -> list[dict]:
        """Run one inpainting batch; write passes, return the failed jobs."""
        if not b:
            return []
        gens = [torch.Generator(device=device).manual_seed(args.seed * 1009 + j["idx"] * 17 + j["retry"])
                for j in b]
        result = pipe(
            prompt=[j["prompt"] for j in b],
            negative_prompt=[NEG_PROMPT] * len(b),
            image=[j["init"] for j in b],
            mask_image=[j["mask"] for j in b],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gens,
            height=WORK, width=WORK,
        ).images
        failed = []
        for j, gen_img in zip(b, result):
            gen_w = gen_img.resize((WORK, WORK))
            if not passes_guard(gen_w, j["init"], j["mask"]):
                failed.append(j)
                continue
            # re-composite: keep original pixels outside the mask exactly
            m = j["mask"].resize((WORK, WORK)).convert("L")
            comp = Image.composite(gen_w, j["init"], m).resize((OUT, OUT), Image.LANCZOS)
            fname = f"syn_{j['idx']:06d}.webp"
            comp.save(images_dir / fname, format="WEBP", quality=args.quality)
            rows.append({
                "filename": (output_dir / "images" / fname).as_posix(),
                "FaceOcclusion": round(j["new_label"], 6),
                "gender": j["gender"],
                "source_filename": j["source"],
                "occluder_type": j["type"],
                "actual_area": round(j["area"], 6),
                "original_label": round(j["orig"], 6),
            })
        return failed

    # ---- batched generation with a retry queue for guard-rejected samples ----
    src_iter = iter(sampled.itertuples(index=False))
    retry_q: list[dict] = []
    next_idx = 0
    exhausted = False
    last_ckpt = 0

    while True:
        batch: list[dict] = []
        while retry_q and len(batch) < args.batch_size:
            batch.append(retry_q.pop())
        while len(batch) < args.batch_size and not exhausted:
            try:
                row = next(src_iter)
            except StopIteration:
                exhausted = True
                break
            src = Path(args.image_dir) / row.filename
            try:
                init = Image.open(src).convert("RGB").resize((WORK, WORK), Image.LANCZOS)
            except Exception as e:
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"[skip] {src}: {e}")
                continue
            otype = random.choice(args.types)
            batch.append(build_job(next_idx, init, float(row.FaceOcclusion), row.gender, row.filename, otype))
            next_idx += 1

        if not batch:
            break

        for j in run_batch(batch):
            if j["retry"] < args.max_retries:
                # retry with a fresh mask/seed, keeping the same source + type
                retry_q.append(build_job(j["idx"], j["init"], j["orig"], j["gender"],
                                         j["source"], j["type"], retry=j["retry"] + 1))
                n_retried += 1
            else:
                n_dropped += 1

        if len(rows) - last_ckpt >= args.checkpoint_every:
            last_ckpt = len(rows)
            flush_csv()
            print(f"[gen]  {len(rows)} written  (retried={n_retried} dropped={n_dropped})")

    flush_csv()

    out_df = pd.DataFrame(rows)
    print("\n[done]")
    print(f"  generated   : {len(out_df)} rows")
    print(f"  skipped(io) : {n_skipped}")
    print(f"  guard       : {n_retried} retried, {n_dropped} dropped after {args.max_retries} retries")
    print(f"  images dir  : {images_dir}")
    print(f"  csv path    : {output_dir / 'synthetic.csv'}")
    if len(out_df):
        print(f"  label range : min={out_df.FaceOcclusion.min():.3f}  "
              f"mean={out_df.FaceOcclusion.mean():.3f}  max={out_df.FaceOcclusion.max():.3f}")
        print(f"  per-type    : {out_df.occluder_type.value_counts().to_dict()}")
        print(f"  per-gender  : {out_df.gender.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
