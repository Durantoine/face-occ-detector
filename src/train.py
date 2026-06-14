from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.data.dataset import make_loaders, FaceOccDataset
from src.models.face_occ_regressor import FaceOccModel
from src.utils.distribution import stratified_is_score
from src.utils.losses import ChallengeLoss
from src.utils.metrics import challenge_score


def dist_info() -> tuple[bool, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world > 1, rank, world, local_rank


def _device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bcast(values: list[float], device: torch.device, is_dist: bool) -> list[float]:
    if not is_dist:
        return values
    import torch.distributed as dist
    t = torch.tensor(values, dtype=torch.float32, device=device)
    dist.broadcast(t, src=0)
    return t.tolist()


def _param_groups(model, base_lr: float, wd: float, layer_decay: float):
    if layer_decay >= 1.0:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return [{"params": decay, "weight_decay": wd, "lr": base_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": base_lr}]
    depths = [int(m.group(1)) for n, p in model.named_parameters()
              for m in [re.search(r"(?:stages|blocks|layers)\.(\d+)\.", n)] if m]
    n_layers = (max(depths) + 1) if depths else 1
    groups: dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"(?:stages|blocks|layers)\.(\d+)\.", n)
        # head AND pool are fresh (randomly-init) modules -> top-layer depth = full lr. Backbone
        # params that don't match the block regex (patch_embed/pos_embed/cls) stay at depth 0.
        # Anchor on (^|.)head./pool. so a DDP "module." prefix still resolves to full lr (and a
        # backbone "conv_head." is NOT caught -> it stays a backbone param at depth 0).
        is_top = re.search(r"(?:^|\.)(?:head|pool)\.", n) is not None
        depth = int(m.group(1)) if m else (n_layers if is_top else 0)
        scale = layer_decay ** (n_layers - depth)
        wd_i = 0.0 if p.ndim <= 1 else wd
        key = (depth, wd_i)
        groups.setdefault(key, {"params": [], "weight_decay": wd_i, "lr": base_lr * scale})
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def evaluate(raw_model, loader, device, val_mode: str, weight_fn=None, mem_fmt=torch.contiguous_format):
    raw_model.eval()
    preds, gts, gs = [], [], []
    use_amp = device.type == "cuda"
    for b in loader:
        x = b["x"].to(device, non_blocking=(device.type == "cuda"), memory_format=mem_fmt)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = raw_model(x)
        else:
            p = raw_model(x)
        preds.append(p.float().cpu().numpy())
        gts.append(b["y"].numpy())
        gs.append(b["g"].numpy())
    preds, gts, gs = np.concatenate(preds), np.concatenate(gts), np.concatenate(gs)
    
    # 1. Raw Metrics (Actual distribution of the loader)
    m_raw = challenge_score(preds, gts, gs)
    
    # 2. Stratified Metrics (Weighted to target distribution)
    # Note: weight_fn should be the 'selection' mode function
    m_strat = stratified_is_score(preds, gts, gs, weight_fn=weight_fn)
    
    # Choose 'primary' for checkpointing and return all for logging
    primary = m_strat if weight_fn is not None else m_raw
    
    # Combined metrics dict for MLflow and logs
    metrics = {
        "eval_challenge_score": float(primary["challenge_score"]),
        "eval_err_F": float(primary["err_F"]),
        "eval_err_M": float(primary["err_M"]),
        "eval_err_diff": float(primary["err_diff"]),
        "eval_err_gap": float(primary.get("err_gap", 0.0)),
        "eval_mae_pct": float(m_raw["mae_pct"]),
        "eval_r2": float(m_raw["r2"]),
        # Full breakdown
        "m_raw": m_raw,
        "m_strat": m_strat,
        "primary_is_strat": weight_fn is not None
    }
    return metrics, preds, gts, gs


def _occ_gender_report(preds, gts, gens, tag: str) -> None:
    """v35 diagnostic: weighted err (w=1/30+y, the challenge weighting) per occ-bin x gender.
    Surfaces WHERE the F/M gap is born and lets us compare natural-train vs val (overfit check)."""
    preds = np.asarray(preds).ravel(); gts = np.asarray(gts).ravel(); g = np.asarray(gens).ravel() >= 0.5
    w = 1.0 / 30.0 + gts; e2 = (preds - gts) ** 2
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 1.01]
    print(f"  [per-bin {tag:9}] occ      |   F:   n   werr     |   M:   n   werr")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gts >= lo) & (gts < hi)
        def we(sub):
            mm = m & sub; sw = float(w[mm].sum())
            return int(mm.sum()), (float((w[mm] * e2[mm]).sum() / sw) if sw > 0 else 0.0)
        nF, eF = we(~g); nM, eM = we(g)
        print(f"    {lo:.1f}-{min(hi, 1.0):.1f}            | {nF:6d}  {eF:.5f}   | {nM:6d}  {eM:.5f}")


def _save_qualitative(out_dir: Path, val_df, cfg: Config, preds, gt, gender, k: int, weight_fn=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    qual = out_dir / "qualitative"
    abs_err = np.abs(preds - gt)
    
    # v26: Use the same distance-based weight function as the rest of the pipeline
    is_w = weight_fn(gt, gender) if weight_fn is not None else np.ones_like(gt)
    w = (1.0 / 30.0 + gt) * is_w
    werr = w * (preds - gt) ** 2
    order = np.argsort(-werr)

    base = Path(cfg.image_dir)
    files = val_df[cfg.image_col].astype(str).values
    for label, idx in [("worst", order[:k]), ("best", order[::-1][:k])]:
        sub = qual / label
        img_dir = sub / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, i in enumerate(idx):
            rows.append({"rank": rank, "gt": float(gt[i]), "pred": float(preds[i]),
                         "abs_err": float(abs_err[i]), "gender": int(gender[i] >= 0.5),
                         "filename": str(files[i])})
            src = base / files[i]
            if src.exists():
                Image.open(src).convert("RGB").save(img_dir / f"{rank:03d}.png")
        pd.DataFrame(rows).to_csv(sub / f"{label}.csv", index=False)

    diag = qual / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    from src.utils.distribution import BIN_WIDTH, N_BINS
    b = np.clip((gt / BIN_WIDTH).astype(int), 0, N_BINS - 1)
    centers = (np.arange(N_BINS) + 0.5) * BIN_WIDTH * 100
    f, m = gender < 0.5, gender >= 0.5

    def per_bin(mask, arr, agg):
        o = np.full(N_BINS, np.nan)
        for bi in range(N_BINS):
            s = mask & (b == bi)
            if s.any():
                o[bi] = agg(arr[s])
        return o

    mae_f = per_bin(f, abs_err, lambda a: a.mean())
    mae_m = per_bin(m, abs_err, lambda a: a.mean())
    contrib_f = per_bin(f, werr, lambda a: a.sum()) / max(w[f].sum(), 1e-9)
    contrib_m = per_bin(m, werr, lambda a: a.sum()) / max(w[m].sum(), 1e-9)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(centers, mae_f * 100, "-o", color="tab:red", label="F", ms=3)
    ax[0, 0].plot(centers, mae_m * 100, "-^", color="tab:blue", label="M", ms=3)
    ax[0, 0].set_title("A — MAE per Y bin × gender (%)"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    ax[0, 1].bar(centers - 0.5, np.nan_to_num(contrib_f) * 1e3, 1.0, color="tab:red", alpha=.8, label="F")
    ax[0, 1].bar(centers + 0.5, np.nan_to_num(contrib_m) * 1e3, 1.0, color="tab:blue", alpha=.8, label="M")
    ax[0, 1].set_title("B — contribution to err_g per bin (×1e-3)"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3, axis="y")
    ax[1, 0].hist([gt[f], gt[m]], bins=26, color=["tab:red", "tab:blue"], label=["F", "M"])
    ax[1, 0].set_title("C — sample density per Y bin × gender"); ax[1, 0].legend()
    ax[1, 1].hist([abs_err[f], abs_err[m]], bins=40, density=True, color=["tab:red", "tab:blue"], alpha=.6, label=["F", "M"])
    ax[1, 1].set_title("D — |pred−gt| distribution by gender"); ax[1, 1].legend()
    for a in ax.ravel():
        a.set_xlabel("occlusion Y (%)")
    fig.tight_layout()
    fig.savefig(diag / "error_vs_occlusion_and_density.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def train(cfg: Config, optuna_trial=None, experiment: str = "faceocc",
          tracking_uri: str = "sqlite:///mlflow.db") -> dict:
    import mlflow
    from src.utils.mlflow_utils import log_metrics as ml_metrics
    from src.utils.mlflow_utils import log_params as ml_params

    is_dist, rank, world, local_rank = dist_info()
    is_main = rank == 0
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = _device(local_rank)
    out = Path(cfg.out_dir) / cfg.name
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_df, val_df, sampler = make_loaders(cfg, rank, world, pin_memory=(device.type == "cuda"))

    # v35 diagnostic: natural-train eval loader (no sampler) to compare per-bin gender error vs val.
    train_eval_loader = None
    if is_main:
        # Diagnostic train-eval: include the ENTIRE sparse tail (occ>=0.3, the bins that drive the
        # fairness gap) so its per-bin werr is stable, + a random sample of the bulk to keep it cheap.
        _tail = train_df[train_df[cfg.label_col] >= 0.3]
        _bulk = train_df[train_df[cfg.label_col] < 0.3].sample(
            min((train_df[cfg.label_col] < 0.3).sum(), 16000), random_state=cfg.seed)
        _te = pd.concat([_tail, _bulk]).reset_index(drop=True)
        train_eval_loader = DataLoader(FaceOccDataset(_te, cfg, train=False),
                                       batch_size=cfg.batch_size * 2, shuffle=False,
                                       num_workers=cfg.num_workers, persistent_workers=False)

    from src.utils.distribution import get_joint_weight_fn as get_weight_fn
    # Loss weights are precomputed per-sample in make_loaders by the same canonical weight
    # function and carried in each batch as b["w"].

    # Single weight function for scoring + monitoring. alpha=cfg.eval_alpha (default 0.5) targets the
    # INTERMEDIATE distribution P_target=(1-a)P_train+a*P_test (the interim leaderboard sits between
    # train and test), with the same alpha used to reweight training -> selection and training are
    # coherent. The W_HI_FLOOR keeps high-occlusion weight >= 1 regardless of alpha.
    weight_fn_score = get_weight_fn(train_df[cfg.label_col].values,
                                    train_df[cfg.gender_col].values,
                                    target=cfg.is_target, alpha=cfg.eval_alpha, lam=cfg.is_lambda)
    weight_fn_mon = weight_fn_score
    # Full-P_test (alpha=1) honest score, LOGGED FOR INFO ONLY (never used for selection): proxies the
    # FINAL metric (test-only) vs eval_alpha which proxies the interim -> shows the interim->final gap.
    weight_fn_full = get_weight_fn(train_df[cfg.label_col].values, train_df[cfg.gender_col].values,
                                   target=cfg.is_target, alpha=1.0, lam=cfg.is_lambda)

    # pretrained_source -> init_backbone_from resolved here so HPO and direct runs behave identically
    # (the translation used to live only in optimize.py, so a direct train run ignored it).
    if cfg.pretrained_source and not cfg.init_backbone_from:
        _src = str(cfg.pretrained_source)
        if _src.startswith("ibot:"):
            cfg.init_backbone_from = _src[len("ibot:"):]
        elif _src not in ("sapiens_default", "lvd", "default"):
            raise ValueError(f"Unknown pretrained_source: {cfg.pretrained_source}")

    target_mean = float(train_df[cfg.label_col].mean())
    if is_main: print(f"[debug] Initializing model {cfg.backbone}...", flush=True)
    model = FaceOccModel(cfg.backbone, drop_path=cfg.drop_path, 
                         pooling_dropout=cfg.pooling_dropout,
                         head_dropout=cfg.head_dropout,
                         pooling_type=cfg.pooling_type, grid_size=cfg.grid_size,
                         attn_queries=cfg.attn_queries, target_mean=target_mean,
                         init_backbone_from=cfg.init_backbone_from,
                         head_mlp_ratio=cfg.head_mlp_ratio).to(device)
    if is_main: print(f"[debug] Model initialized and moved to {device}.", flush=True)
    raw_model = model
    # --- perf: channels_last for CNN backbones ONLY (ViT/transformer gets no benefit), + torch.compile ---
    _cnn_prefixes = ("convnext", "resnet", "resnext", "efficientnet", "regnet", "mobilenet",
                     "dla", "hrnet", "densenet", "cspnet", "rexnet", "nfnet", "repvgg")
    _is_cnn_backbone = any(str(cfg.backbone).lower().startswith(p) for p in _cnn_prefixes)
    _mem_fmt = torch.contiguous_format
    if _is_cnn_backbone and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)  # in-place; raw_model shares the module
        _mem_fmt = torch.channels_last
        if is_main: print("[perf] channels_last enabled (CNN backbone)", flush=True)
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if getattr(cfg, "compile_model", True) and device.type == "cuda":
        try:
            model = torch.compile(model)  # raw_model stays the un-compiled module (clean state_dict)
            if is_main: print("[perf] torch.compile enabled", flush=True)
        except Exception as _ce:
            if is_main: print(f"[perf] torch.compile unavailable ({_ce}); continuing uncompiled", flush=True)

    opt = AdamW(_param_groups(model, cfg.lr, cfg.weight_decay, cfg.layer_decay), betas=(0.9, 0.999))
    steps = len(train_loader) * cfg.epochs
    warmup = int(steps * cfg.warmup_ratio)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    base_lrs = [g["lr"] for g in opt.param_groups]
    loss_fn = ChallengeLoss(lambda_gap=cfg.lambda_gap,
                            asymmetric=getattr(cfg, "gap_asymmetric", False))
    use_amp = device.type == "cuda" and cfg.bf16
    best = {"eval_challenge_score": float("inf")}; bad = 0; gstep = 0


    if is_main:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        run_ctx = mlflow.start_run(run_name=cfg.name, nested=optuna_trial is not None)
    else:
        run_ctx = contextlib.nullcontext()

    with run_ctx:
        if is_main:
            ml_params(None, None, {k: v for k, v in cfg.__dict__.items() if v is not None})
            bb = raw_model.backbone
            if getattr(bb, "init_pretrain_run_id", None):
                ml_params(None, None, {"init_backbone_pretrain_run_id": bb.init_pretrain_run_id,
                                       "init_backbone_missing_keys": bb.init_missing_keys,
                                       "init_backbone_unexpected_keys": bb.init_unexpected_keys})

        for epoch in range(cfg.epochs):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            model.train(); opt.zero_grad()
            for i, b in enumerate(train_loader):
                is_cuda = device.type == "cuda"
                x = b["x"].to(device, non_blocking=is_cuda, memory_format=_mem_fmt)
                y = b["y"].to(device, non_blocking=is_cuda)
                g = b["g"].to(device, non_blocking=is_cuda)
                w = b["w"].to(device, non_blocking=is_cuda)

                # Direct step on M3 Max (Metal/Metal Performance)
                sync_ctx = contextlib.nullcontext()
                
                with sync_ctx:
                    # ChallengeLoss handles the fairness gap internally; its asymmetry tilt is set
                    # once per epoch from the validation gap (loss_fn.update_tilt below), not per batch.
                    if use_amp:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = loss_fn(model(x), y, g, w)
                    else:
                        loss = loss_fn(model(x), y, g, w)
                    loss.backward()

                scale = lr_at(gstep)
                for grp, lr0 in zip(opt.param_groups, base_lrs):
                    grp["lr"] = lr0 * scale
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); gstep += 1
                
                if is_main and i % 25 == 0:
                    # v34: Final transparent monitoring
                    with torch.no_grad():
                        pred_raw = model(x).view(-1)
                        err_sq = (pred_raw - y) ** 2

                        f_m, m_m = g < 0.5, g >= 0.5
                        w_met = (1.0/30.0 + y)
                        
                        def get_m(weights):
                            def w_avg(mask):
                                if not mask.any(): return None # v34: return None instead of 0.0 to detect empty segments
                                return float((weights[mask] * err_sq[mask]).sum() / weights[mask].sum().clamp(min=1e-8))
                            ef, em = w_avg(f_m), w_avg(m_m)
                            
                            # v34: Handle cases where one gender is missing from the small scarcity tail in the batch
                            if ef is None or em is None:
                                return None
                            
                            gap = ef - em
                            return {"sc": (ef + em) / 2.0 + abs(gap), "gap": gap, "F": ef, "M": em}

                        # 1. [Optimizer] - What drives the gradient (Damped by Alpha)
                        m_opt = get_m(w_met * w)
                        
                        # 2. [Challenge] - Honest projection on P_test (Alpha=1.0)
                        is_w_h = torch.as_tensor(weight_fn_mon(y.cpu().numpy(), g.cpu().numpy()), device=device)
                        m_chal = get_m(w_met * is_w_h)
                        
                        # 3. [Batch] - Performance on raw batch distribution
                        m_batch = get_m(w_met)
                        
                    # v34: Only log and print if we have a valid estimate for both genders
                    if m_opt and m_chal and m_batch:
                        ml_metrics(None, None, {"train_loss": loss.item(), 
                                               "train_score_opt": m_opt["sc"],
                                               "train_score_chal": m_chal["sc"],
                                               "train_gap_chal": m_chal["gap"],
                                               "train_gap_batch": m_batch["gap"]}, step=gstep)
                        
                        print(f"\nepoch {epoch:2d} | batch {i:4d}/{len(train_loader)} | lam: {loss_fn.lam:.2f} | Alpha: {cfg.eval_alpha:.2f}")
                        print(f"  [Optimizer] loss: {loss.item():.4f} | score: {m_opt['sc']:.5f} | gap: {m_opt['gap']:+.5f} (F:{m_opt['F']:.4f}, M:{m_opt['M']:.4f})")
                        print(f"  [Challenge] score: {m_chal['sc']:.5f} | gap: {m_chal['gap']:+.5f} (F:{m_chal['F']:.4f}, M:{m_chal['M']:.4f})")
                        print(f"  [Batch    ] score: {m_batch['sc']:.5f} | gap: {m_batch['gap']:+.5f} (F:{m_batch['F']:.4f}, M:{m_batch['M']:.4f})")


            stop_flag = False
            new_lam = loss_fn.lam
            if is_main:
                m, vp, vgt, vg = evaluate(raw_model, val_loader, device, cfg.val_mode, weight_fn=weight_fn_score, mem_fmt=_mem_fmt)
                # only scalar entries are loggable; m_raw/m_strat are nested dicts (float(dict) would
                # blow up the whole log_metrics call -> previously no val curves appeared in MLflow)
                ml_metrics(None, None, {k: v for k, v in m.items() if isinstance(v, (int, float))}, step=epoch)
                tilt = loss_fn.update_tilt(m["eval_err_M"], m["eval_err_F"])  # v38: lean penalty to val-worse gender
                
                print(f"\nepoch {epoch:2d} | VALIDATION (mode: {cfg.val_mode}) | lam: {new_lam:.2f} | tilt: {tilt:+.2f}")
                mh, mr = m["m_strat"], m["m_raw"]
                print(f"  [Honest] score: {mh['challenge_score']:.5f} | gap: {mh['err_gap']:+.5f} (F:{mh['err_F']:.4f}, M:{mh['err_M']:.4f}) | mae: {m['eval_mae_pct']:.2f}%")
                print(f"  [Raw   ] score: {mr['challenge_score']:.5f} | gap: {mr['err_gap']:+.5f} (F:{mr['err_F']:.4f}, M:{mr['err_M']:.4f}) | r2: {m['eval_r2']:.3f}")
                m_full = stratified_is_score(vp, vgt, vg, weight_fn=weight_fn_full)
                ml_metrics(None, None, {"eval_honest_ptest": float(m_full["challenge_score"]),
                                        "eval_honest_ptest_gap": float(m_full["err_gap"])}, step=epoch)
                print(f"  [Full-Pt] score: {m_full['challenge_score']:.5f} | gap: {m_full['err_gap']:+.5f} (F:{m_full['err_F']:.4f}, M:{m_full['err_M']:.4f})  (alpha=1, proxy FINAL — info only)")

                # v35 diagnostic: where is the gap born, and does male error jump train->val (overfit)?
                _occ_gender_report(vp, vgt, vg, "VAL")
                if train_eval_loader is not None:
                    _, tp, tgt, tgg = evaluate(raw_model, train_eval_loader, device, cfg.val_mode, weight_fn=weight_fn_score, mem_fmt=_mem_fmt)
                    _occ_gender_report(tp, tgt, tgg, "TRAIN-nat")

                ml_metrics(None, None, {"lambda_adapt": new_lam}, step=epoch)

                full_data = getattr(cfg, "full_data", False)  # 100%-data retrain: no held-out val -> save EVERY epoch (best.pt = final), never early-stop
                if full_data or m["eval_challenge_score"] < best["eval_challenge_score"]:
                    best = {**m, "epoch": epoch}
                    torch.save(raw_model.state_dict(), out / "best.pt"); bad = 0
                    pd.DataFrame({"gt": np.asarray(vgt).ravel(), "pred": np.asarray(vp).ravel(),
                                  "gender": (np.asarray(vg).ravel() >= 0.5).astype(int)}).to_csv(out / "val_predictions.csv", index=False)
                    if cfg.save_qualitative_k > 0 and not full_data:
                        _save_qualitative(out, val_df, cfg, vp, vgt, vg, cfg.save_qualitative_k, weight_fn=weight_fn_score)
                else:
                    bad += 1
                    stop_flag = bad >= cfg.early_stop_patience

            stop_flag, new_lam = _bcast([float(stop_flag), new_lam], device, is_dist)
            loss_fn.lam = float(new_lam)
            if stop_flag > 0.5:
                if is_main:
                    print(f"early stop (no improvement in {bad} epochs)", flush=True)
                break

        if is_main:
            # v35: log the FULL breakdown of the best epoch (not just best_score) so Optuna and
            # the Streamlit viewer report best-epoch metrics, never the last (possibly overfit) epoch.
            ml_metrics(None, None, {
                "best_score": best["eval_challenge_score"],
                "best_err_F": best.get("eval_err_F", 0.0),
                "best_err_M": best.get("eval_err_M", 0.0),
                "best_err_diff": best.get("eval_err_diff", 0.0),
                "best_err_gap": best.get("eval_err_gap", 0.0),
                "best_mae_pct": best.get("eval_mae_pct", 0.0),
                "best_r2": best.get("eval_r2", 0.0),
                "best_epoch": float(best.get("epoch", -1)),
            })
            if cfg.save_qualitative_k > 0 and (out / "qualitative").exists():
                mlflow.log_artifacts(str(out / "qualitative"), "qualitative")
            if (out / "val_predictions.csv").exists():
                mlflow.log_artifact(str(out / "val_predictions.csv"), "val_predictions")
            (out / "metrics.json").write_text(json.dumps({**best, "config": cfg.__dict__}, indent=2))
            print(f"BEST epoch={best.get('epoch')} score={best['eval_challenge_score']:.5f} → {out/'best.pt'}", flush=True)

            # Test-set prediction distribution for the Streamlit viewer (best epoch, real inference).
            # OPTIMIZED: only for NEW-BEST trials (avoids a 30k-image pass every trial), reusing the
            # in-memory model (no re-instantiation) and with no CSV write→read round-trip.
            _do_testpred = bool(getattr(cfg, "log_test_pred", True))
            if _do_testpred and optuna_trial is not None:
                try:
                    _do_testpred = best["eval_challenge_score"] < optuna_trial.study.best_value
                except ValueError:
                    _do_testpred = True   # first completed trial -> it is the best so far
            if _do_testpred:
                try:
                    raw_model.load_state_dict(torch.load(out / "best.pt", map_location=device))
                    raw_model.eval()
                    _tdf = pd.read_csv(cfg.test_csv).dropna(subset=[cfg.image_col])
                    _tdf[cfg.label_col] = 0.0
                    _tloader = DataLoader(FaceOccDataset(_tdf, cfg, train=False),
                                          batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers,
                                          pin_memory=(device.type == "cuda"))
                    _amp = device.type == "cuda"
                    _preds = []
                    with torch.no_grad():
                        for _b in _tloader:
                            _x = _b["x"].to(device, non_blocking=_amp, memory_format=_mem_fmt)
                            if _amp:
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    _preds.append(raw_model(_x).float().cpu().numpy())
                            else:
                                _preds.append(raw_model(_x).float().cpu().numpy())
                    _tp = np.clip(np.concatenate(_preds), 0.0, 1.0).ravel()
                    _csv = out / "test_predictions.csv"
                    pd.DataFrame({cfg.image_col: _tdf[cfg.image_col].values, cfg.label_col: _tp}).to_csv(_csv, index=False)
                    mlflow.log_artifact(str(_csv), "test_predictions")
                    ml_metrics(None, None, {
                        "test_pred_mean": float(_tp.mean()), "test_pred_std": float(_tp.std()),
                        "test_pred_median": float(np.median(_tp)), "test_pred_frac_gt_0_3": float((_tp > 0.3).mean()),
                    })
                    print(f"[test-pred] {len(_tp):,} predictions logged (new-best trial, mean={_tp.mean():.3f})", flush=True)
                except Exception as _e:
                    print(f"[test-pred] skipped: {_e}", flush=True)
    return best


if __name__ == "__main__":
    import argparse
    is_dist, rank, world, local_rank = dist_info()
    if is_dist:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="default", help="Name or path of the architecture config")
    parser.add_argument("--optimize", action="store_true", help="Ignored in train.py (use src/optimize.py for HPO sweeps)")
    parser.add_argument("--fold", type=int, default=None, help="K-fold: index of the validation fold (0-based)")
    parser.add_argument("--n-folds", type=int, default=None, help="K-fold: total number of folds")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed (model init/training, NOT the fold partition)")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    args, unknown = parser.parse_known_args()

    if unknown and rank == 0:
        print(f"Warning: unknown arguments ignored: {unknown}")

    cfg, _ = load_config(args.config)
    if args.n_folds is not None:
        cfg.n_folds = args.n_folds
    if args.fold is not None:
        cfg.fold_idx = args.fold
        cfg.name = f"{cfg.name}_fold{args.fold}"
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg, experiment=f"faceocc-{cfg.name}", tracking_uri=args.tracking_uri)
    
    if is_dist:
        import torch.distributed as dist
        dist.destroy_process_group()
