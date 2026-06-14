"""v36: select label outliers to REMOVE from the near-duplicate clusters (no relabel -- the labels
are the teacher ground truth we imitate; we only excise glitches).

Reads results/duplicate_clusters.csv (from detect_duplicates.py) + train.csv. Cheap and tunable:
re-run with different thresholds without recomputing the NCC clusters. Two consensus sources:
  - cluster size >= --min-cluster: consensus = cluster median; flag members far from it (never all).
  - cluster size == 2 (a pair, no internal majority): consensus = the IDENTITY median, when that MID
    has >= --min-idn frames (stable). Flag the single pair member that deviates, only if exactly one
    does (else ambiguous -> deferred to the model-residual pass).
Both gated by cohesion >= --min-cohesion so we only excise genuine near-dups. Output:
results/outliers_to_remove.csv + docs/outliers_report.html. Removal stays manual (user-validated).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_MID = r"(m\.[0-9a-zA-Z_]+)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="results/duplicate_clusters.csv")
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--image-dir", default="data/raw")
    ap.add_argument("--out-csv", default="results/outliers_to_remove.csv")
    ap.add_argument("--out-html", default="docs/outliers_report.html")
    ap.add_argument("--dev", type=float, default=0.10, help="min |y - consensus| to flag an outlier")
    ap.add_argument("--min-cluster", type=int, default=3, help="cluster size that uses the cluster median")
    ap.add_argument("--min-idn", type=int, default=4, help="min identity frames to arbitrate a pair")
    ap.add_argument("--min-cohesion", type=float, default=0.90, help="min cluster cohesion (genuine near-dup)")
    args = ap.parse_args()

    d = pd.read_csv(args.clusters)
    tr = pd.read_csv(args.train)
    tr["mid"] = tr["filename"].str.extract(_MID)[0]
    id_med = tr.groupby("mid")["FaceOcclusion"].median()
    id_n = tr.groupby("mid").size()

    flagged = []  # (filename, mid, y, consensus, source, cohesion, cluster_id)
    for cid, c in d[d.scope == "intra"].groupby("cluster_id"):
        if c["cohesion"].iloc[0] < args.min_cohesion:
            continue
        labs = c["FaceOcclusion"].to_numpy()
        if len(c) >= args.min_cluster:
            cons = float(np.median(labs)); src = "cluster-median"
            dev = np.abs(labs - cons)
            out = c[dev > args.dev]
            if len(out) >= len(c) - 1:   # need a consensus majority
                continue
        elif len(c) == 2:
            mid = c["mid"].iloc[0]
            if pd.isna(mid) or id_n.get(mid, 0) < args.min_idn:
                continue
            cons = float(id_med[mid]); src = "identity-median"
            dev = np.abs(labs - cons)
            out = c[dev > args.dev]
            if len(out) != 1:            # exactly one member must be the outlier
                continue
        else:
            continue
        for _, r in out.iterrows():
            flagged.append((r["filename"], r["mid"], r["FaceOcclusion"], round(cons, 4), src,
                            r["cohesion"], cid))

    cols = ["filename", "mid", "FaceOcclusion", "consensus", "source", "cohesion", "cluster_id"]
    rm = pd.DataFrame(flagged, columns=cols)
    rm["deviation"] = (rm["FaceOcclusion"] - rm["consensus"]).abs().round(4)
    rm = rm.sort_values("deviation", ascending=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rm.to_csv(args.out_csv, index=False)

    base = Path(args.image_dir)
    cl = d.set_index("filename")
    css = ("body{background:#111;color:#ddd;font-family:monospace}.cl{border-bottom:1px solid #333;padding:8px;"
           "display:flex;align-items:center;flex-wrap:wrap}.hdr{width:230px;font-size:12px;color:#ff0}"
           ".im{display:inline-block;margin:4px;text-align:center;font-size:11px}.im img{height:140px;border:3px solid #444;display:block}"
           ".out img{border-color:#e33}.lb{color:#fff;font-weight:bold}.fn{color:#9cf;font-size:10px;max-width:150px;word-break:break-all}")
    P = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
         f"<h2>{len(rm)} outliers à retirer (bord rouge) — {rm.source.value_counts().to_dict()}</h2>",
         "<p>chaque ligne : l'outlier (rouge) + ses co-membres du cluster near-dup (consensus)</p>"]
    for _, r in rm.iterrows():
        cid = r["cluster_id"]; mates = d[d.cluster_id == cid]
        P.append("<div class='cl'>")
        P.append(f"<div class='hdr'>{r['mid']}<br>outlier={r['FaceOcclusion']:.3f}<br>"
                 f"consensus={r['consensus']:.3f} ({r['source']})<br>dev={r['deviation']:.3f} coh={r['cohesion']:.2f}</div>")
        for _, m in mates.iterrows():
            is_out = m["filename"] == r["filename"]
            P.append(f"<div class='im {'out' if is_out else ''}'><img src='../{base / m['filename']}'>"
                     f"<div class='lb'>{m['FaceOcclusion']:.3f}</div>"
                     f"<div class='fn'>{m['filename'].split('/')[-1]}</div></div>")
        P.append("</div>")
    P.append("</body></html>")
    Path(args.out_html).write_text("\n".join(P))
    print(f"flagged {len(rm)} outliers ({rm.source.value_counts().to_dict()}) -> {args.out_csv} , {args.out_html}")


if __name__ == "__main__":
    main()
