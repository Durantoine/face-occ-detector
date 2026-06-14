"""v36: near-duplicate detection by pixel template matching (NCC), scope-aware.

Perceptual hashes fail on aligned face crops (shared eyes/nose/mouth geometry -> distinct people
collide). We use Normalized Cross-Correlation on the pixels instead: NCC(a,b) on mean-subtracted
L2-normalized templates == cosine. Distinct aligned faces score ~0.6-0.9, true near-duplicates of
the SAME shot ~0.96-0.99.

A single global threshold cannot be both strict (no cross-identity false merge) and permissive
(catch intra-identity dups under recompression), so we split by SCOPE:
  - INTRA-MID: pairwise within each identity, permissive (--thr-intra, 0.93). No cross-identity
    risk, so we recover slightly-recompressed dups. This is where label conflicts live.
  - CROSS-MID: global, strict (--thr-cross, 0.985). Genuine same-photo-under-two-identities leakage
    only; on this dataset it is essentially empty (cross-MID "dups" at lower thr are variant chains).
Each reported cluster carries its cohesion = min pairwise NCC at --verify-res, so any union-find
chaining is visible. Detection only: writes a report + CSV, nothing is relabelled or dropped.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_MID = r"(m\.[0-9a-zA-Z_]+)"


def _template(base: Path, rel: str, res: int):
    try:
        a = np.asarray(Image.open(base / rel).convert("L").resize((res, res)), dtype=np.float32).ravel()
    except Exception:
        return None
    a -= a.mean(); nrm = np.linalg.norm(a)
    return a / nrm if nrm > 1e-6 else None


class _UF:
    def __init__(self, ids): self.p = {i: i for i in ids}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def _clusters_from_edges(ids, edges):
    uf = _UF(ids)
    for a, b in edges:
        uf.union(a, b)
    groups = defaultdict(list)
    for i in ids:
        groups[uf.find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/train.csv")
    ap.add_argument("--image-dir", default="data/raw")
    ap.add_argument("--out-html", default="docs/duplicates_report.html")
    ap.add_argument("--out-csv", default="results/duplicate_clusters.csv")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--verify-res", type=int, default=128, help="cohesion (min pairwise NCC) resolution")
    ap.add_argument("--thr-intra", type=float, default=0.93)
    ap.add_argument("--thr-cross", type=float, default=0.985)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--outlier-dev", type=float, default=0.10, help="flag a cluster member as outlier if |y - cluster median| exceeds this")
    ap.add_argument("--min-cluster", type=int, default=3, help="min cluster size to auto-flag an outlier (need a consensus)")
    args = ap.parse_args()

    base = Path(args.image_dir)
    df = pd.read_csv(args.csv).reset_index(drop=True)
    df["mid"] = df["filename"].str.extract(_MID)[0]
    n = len(df)

    dim = args.res * args.res
    V = np.zeros((n, dim), dtype=np.float32); ok = np.zeros(n, dtype=bool)
    for i, rel in enumerate(df["filename"]):
        t = _template(base, rel, args.res)
        if t is not None:
            V[i] = t; ok[i] = True
        if i % 10000 == 0:
            print(f"  templated {i}/{n}", flush=True)

    # ---- INTRA-MID pass (cheap: small per-identity groups) ----
    intra_clusters = []
    for mid, grp in df[ok].groupby("mid"):
        ids = grp.index.tolist()
        if len(ids) < 2:
            continue
        edges = [(a, b) for a, b in combinations(ids, 2) if float(V[a] @ V[b]) >= args.thr_intra]
        for cl in _clusters_from_edges(ids, edges):
            intra_clusters.append(cl)
    print(f"  intra-MID clusters: {len(intra_clusters)}", flush=True)

    # ---- CROSS-MID pass (global, strict) ----
    idx_ok = np.where(ok)[0]; Vok = V[idx_ok]
    cross_edges = []
    for s in range(0, len(idx_ok), args.block):
        sims = Vok[s:s + args.block] @ Vok.T
        rs, cs = np.where(sims >= args.thr_cross)
        for r, c in zip(rs, cs):
            gi, gj = int(idx_ok[s + r]), int(idx_ok[c])
            if gj > gi and df.at[gi, "mid"] != df.at[gj, "mid"]:
                cross_edges.append((gi, gj))
    cross_clusters = _clusters_from_edges(sorted({i for e in cross_edges for i in e}), cross_edges)
    print(f"  cross-MID clusters (strict {args.thr_cross}): {len(cross_clusters)}", flush=True)

    # ---- cohesion (min pairwise NCC at verify-res) + rows ----
    vcache: dict[int, np.ndarray] = {}
    def vtpl(j):
        if j not in vcache:
            vcache[j] = _template(base, df.at[j, "filename"], args.verify_res)
        return vcache[j]
    def cohesion(members):
        ts = [vtpl(j) for j in members]; mn = 1.0
        for a, b in combinations(range(len(ts)), 2):
            if ts[a] is not None and ts[b] is not None:
                mn = min(mn, float(ts[a] @ ts[b]))
        return mn

    # We do NOT relabel: the FaceOcclusion labels are the (likely model-generated) ground truth we
    # must imitate. We only FLAG outliers for removal -- a member of a tight near-dup cluster whose
    # label deviates far from the cluster median is a teacher glitch (e.g. m.01c56w/50-FaceId-54).
    rows = []; html_groups = []; cid = 0
    for scope, clusters in [("intra", intra_clusters), ("cross", cross_clusters)]:
        for members in clusters:
            sub = df.iloc[members]; labs = sub["FaceOcclusion"].to_numpy()
            spread = float(labs.max() - labs.min()); coh = cohesion(members)
            med = float(np.median(labs))
            can_flag = (scope == "intra" and len(members) >= args.min_cluster)
            outliers = set()
            if can_flag:
                outliers = {members[k] for k in range(len(members))
                            if abs(labs[k] - med) > args.outlier_dev}
                # never flag everyone (would mean no consensus)
                if len(outliers) >= len(members) - 1:
                    outliers = set()
            for k, j in enumerate(members):
                rows.append({"cluster_id": cid, "scope": scope, "filename": df.at[j, "filename"],
                             "mid": df.at[j, "mid"], "FaceOcclusion": df.at[j, "FaceOcclusion"],
                             "cluster_size": len(members), "n_mids": int(sub["mid"].nunique()),
                             "label_spread": round(spread, 4), "cohesion": round(coh, 4),
                             "cluster_median": round(med, 4), "outlier": j in outliers})
            html_groups.append((scope, members, spread, coh, int(sub["mid"].nunique()), outliers))
            cid += 1
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    rm = out[out.outlier]
    rm[["filename", "mid", "FaceOcclusion", "cluster_id", "cluster_median", "cohesion"]].to_csv(
        "results/outliers_to_remove.csv", index=False)
    print(f"  outliers flagged for removal: {len(rm)} (in {rm.cluster_id.nunique()} clusters) "
          f"-> results/outliers_to_remove.csv", flush=True)

    n_intra = len(intra_clusters); n_cross = len(cross_clusters)
    intra_conf = sum(1 for g in html_groups if g[0] == "intra" and g[2] >= 0.10)
    n_out = int(out.outlier.sum())
    css = ("body{background:#111;color:#ddd;font-family:monospace}"
           ".cl{border-bottom:1px solid #333;padding:8px;display:flex;align-items:flex-start;flex-wrap:wrap}"
           ".hdr{width:170px;font-size:12px;color:#ff0}.x{color:#f6a}.im{display:inline-block;margin:4px;text-align:center;font-size:11px}"
           ".im img{height:130px;border:3px solid #444;display:block}.out img{border-color:#e33}"
           ".fn{color:#9cf;font-size:11px;word-break:break-all;max-width:140px}"
           ".lb{color:#fff;font-weight:bold}h3{color:#fb0;margin-top:20px}")
    P = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
         f"<h2>Doublons par NCC &mdash; intra-MID&ge;{args.thr_intra} ({n_intra}, dont {intra_conf} conflits &Delta;&ge;0.10) | "
         f"cross-MID&ge;{args.thr_cross} ({n_cross}) | <span class=x>outliers &agrave; retirer (bord rouge) : {n_out}</span></h2>"]
    # intra conflicts first (sorted by spread), then the rest, then cross
    for label, key in [("INTRA-MID — conflits de label (Δ≥0.10), outlier en rouge", lambda g: g[0] == "intra" and g[2] >= 0.10),
                       ("INTRA-MID — dups sans conflit", lambda g: g[0] == "intra" and g[2] < 0.10),
                       ("CROSS-MID — strict (fuite éventuelle)", lambda g: g[0] == "cross")]:
        sel = sorted([g for g in html_groups if key(g)], key=lambda g: -g[2])
        P.append(f"<h3>{label} — {len(sel)}</h3>")
        for scope, members, spread, coh, nm, outliers in sel[:300]:
            P.append("<div class='cl'>")
            P.append(f"<div class='hdr'>{'<span class=x>CROSS</span><br>' if scope=='cross' else ''}n={len(members)}<br>"
                     f"&Delta;label={spread:.3f}<br>cohésion={coh:.3f}{'<br>MIDs='+str(nm) if scope=='cross' else ''}</div>")
            for j in members:
                cls = "out" if j in outliers else ""
                P.append(f"<div class='im {cls}'><img src='../{base / df.at[j,'filename']}'>"
                         f"<div class='lb'>{df.at[j,'FaceOcclusion']:.3f}</div>"
                         f"<div class='fn'>{df.at[j,'mid']}</div></div>")
            P.append("</div>")
    P.append("</body></html>")
    Path(args.out_html).write_text("\n".join(P))
    print(f"\nintra-MID={n_intra} (conflits Δ≥0.10: {intra_conf}) | cross-MID={n_cross} | images={len(out)}")
    print(f"saved -> {args.out_csv} , {args.out_html}")


if __name__ == "__main__":
    main()
