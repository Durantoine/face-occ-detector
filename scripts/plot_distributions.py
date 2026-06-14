from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.distribution import (
    CENTERS,
    DEFAULT_LAMBDA,
    N_BINS,
    P_TEST_PMF,
    W_CEIL,
    Y_LOW,
    Y_SUPPORT,
    _p_test_density,
    _pr_grid_gender,
    _pr_grid_global,
    get_joint_weight_fn,
    get_p_test_spline,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/train.csv")
    ap.add_argument("--target", default="joint", choices=["hc", "joint"])
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--xmax", type=float, default=0.7)
    ap.add_argument("--out", default="docs/assets/ratio_regularized.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df["FaceOcclusion"].values.astype(np.float64)
    g = (df["gender"].values >= 0.5).astype(np.float64)

    grid = np.linspace(0.0, 1.0, 1001)
    p_f, p_m = float((g < 0.5).mean()), float((g >= 0.5).mean())
    spline = get_p_test_spline(s=1e-4)

    # P_test density: clamped at 0 and forced to 0 beyond the support (no spline ringing).
    p_test = _p_test_density(grid, spline)
    p_train = _pr_grid_global(y.tobytes(), 0.018)  # scored on the same 1001-pt grid

    wf_hc = get_joint_weight_fn(y, g, target="hc", alpha=1.0, lam=args.lam)
    wf_joint = get_joint_weight_fn(y, g, target="joint", alpha=1.0, lam=args.lam)
    prg_f, prg_m, pf2, pm2 = _pr_grid_gender(y.tobytes(), g.tobytes(), 0.018)
    raw_glob = p_test / (p_train + args.lam)
    raw_F = 0.5 * p_test / (prg_f * pf2 + args.lam)
    raw_M = 0.5 * p_test / (prg_m * pm2 + args.lam)
    zf, om = np.zeros_like(grid), np.ones_like(grid)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))

    a = ax[0]
    a.bar(CENTERS, P_TEST_PMF * N_BINS, width=1.0 / N_BINS, color="tab:red", alpha=0.2, label="P_test PMF")
    a.plot(grid, p_test, color="tab:red", lw=2.2, label="P_test (bornée au support)")
    a.plot(grid, p_train, color="tab:blue", lw=2.2, label="P_train (KDE)")
    a.fill_between(grid, p_test, p_train, where=p_train > p_test, color="tab:blue", alpha=0.12, label="surplus")
    a.fill_between(grid, p_test, p_train, where=p_test > p_train, color="tab:red", alpha=0.15, label="pénurie")
    a.axvline(Y_SUPPORT, color="k", ls="--", lw=1, alpha=0.5)
    a.set_title("A — Densités (P_test = 0 au-delà du support)")
    a.set_xlabel("occlusion Y"); a.set_ylabel("densité"); a.legend(fontsize=8); a.grid(alpha=0.3)

    b = ax[1]
    b.axhline(1.0, color="gray", lw=0.8, ls=":")
    b.axvline(Y_LOW, color="k", ls="--", lw=1, alpha=0.5, label=f"Y_LOW={Y_LOW:.2f}")
    for raw, wf_g, col, lab in [(raw_glob, wf_hc(grid, zf), "tab:green", "global"),
                                (raw_F, wf_joint(grid, zf), "tab:red", "F"),
                                (raw_M, wf_joint(grid, om), "tab:blue", "M")]:
        b.plot(grid, np.clip(raw, 0, W_CEIL), color=col, lw=1.0, ls="--", alpha=0.45)
        b.plot(grid, wf_g, color=col, lw=2.2, label=lab)
    b.set_title(f"B — Ratios global/F/M (plein=floor≥1 passé Y_LOW, tireté=brut, λ={args.lam:.2f})")
    b.set_xlabel("occlusion Y"); b.set_ylabel("poids w(y)"); b.legend(fontsize=8); b.grid(alpha=0.3)

    c = ax[2]
    p_test_bad = np.clip(spline(grid), 1e-12, None); p_test_bad /= np.trapezoid(p_test_bad, grid)
    p_train_n = p_train / np.trapezoid(p_train, grid)
    c.plot(grid, np.clip(p_test_bad / p_train_n, 0, 12), color="tab:red", lw=1.5, ls=":", label="AVANT (spline ringing)")
    c.plot(grid, wf_hc(grid, zf), color="tab:green", lw=2.4, label=f"APRÈS (floor≥1 passé Y_LOW, λ={args.lam:.2f})")
    c.set_xlim(0.30, 0.70); c.set_ylim(0, 8)
    c.set_title("C — Zoom queue : artefact creux+spike corrigé")
    c.set_xlabel("occlusion Y"); c.set_ylabel("poids w(y)"); c.legend(fontsize=8); c.grid(alpha=0.3)

    for a_ in (ax[0], ax[1]):
        a_.set_xlim(0, args.xmax)
    fig.suptitle(f"Reweighting P_test/(P_train+λ) — λ={args.lam:.2f}, P_test bornée au support {Y_SUPPORT:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.out} | p_f={p_f:.4f} p_m={p_m:.4f} | mean Y={y.mean():.4f}")


if __name__ == "__main__":
    main()
