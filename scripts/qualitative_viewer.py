"""Streamlit viewer for the qualitative best/worst-K examples logged per Optuna trial.

Reads the MLflow tracking DB (sqlite:///mlflow.db) to list experiments + runs, then
downloads the `qualitative/` artifact subtree on demand and renders the images with
their metadata (gt, pred, abs_err, gender) in a grid.

Launch :
    uvx --python 3.12 --with mlflow --with pandas --with pillow \
        --from streamlit streamlit run scripts/qualitative_viewer.py \
        --server.port 8501 --server.address 0.0.0.0

By design this script avoids any project-internal import so it can run in a tiny
ephemeral venv (CPU partition, no torch).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Qualitative viewer", layout="wide")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


@st.cache_resource(show_spinner=False)
def _get_client(tracking_uri: str):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=tracking_uri)


@st.cache_data(ttl=30, show_spinner=False)
def _list_experiments(tracking_uri: str) -> List[Tuple[str, str]]:
    client = _get_client(tracking_uri)
    exps = client.search_experiments()
    return sorted([(e.experiment_id, e.name) for e in exps], key=lambda x: x[1])


@st.cache_data(ttl=30, show_spinner=False)
def _list_runs(tracking_uri: str, experiment_id: str) -> List[Dict[str, Any]]:
    client = _get_client(tracking_uri)
    runs = client.search_runs(
        [experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=500,
    )
    out: List[Dict[str, Any]] = []
    for r in runs:
        out.append({
            "run_id": r.info.run_id,
            "name": r.info.run_name or r.info.run_id[:8],
            "status": r.info.status,
            "eval_score": r.data.metrics.get("eval_score") or r.data.metrics.get("val_score"),
            "eval_score_raw": r.data.metrics.get("eval_score_raw"),
            "err_F": r.data.metrics.get("eval_err_F") or r.data.metrics.get("val_err_F"),
            "err_M": r.data.metrics.get("eval_err_M") or r.data.metrics.get("val_err_M"),
            "err_diff": r.data.metrics.get("eval_err_diff") or r.data.metrics.get("val_err_diff"),
        })
    return out


@st.cache_data(ttl=300, show_spinner="Downloading artifacts...")
def _download_qualitative(tracking_uri: str, run_id: str) -> Optional[str]:
    """Download the `qualitative/` artifact subtree for the given run.

    Returns the local path to the downloaded directory, or None if no such artifact.
    """
    client = _get_client(tracking_uri)
    try:
        artifacts = client.list_artifacts(run_id, "qualitative")
    except Exception as e:
        st.error(f"Could not list artifacts for run {run_id}: {e}")
        return None
    if not artifacts:
        return None
    tmp_root = Path(tempfile.gettempdir()) / "face_occ_qualitative_cache" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        local = client.download_artifacts(run_id, "qualitative", dst_path=str(tmp_root))
        return local
    except Exception as e:
        st.error(f"Download failed for run {run_id}: {e}")
        return None


def _load_csv(folder: Path, name: str) -> Optional[pd.DataFrame]:
    csv = folder / f"{name}.csv"
    if not csv.exists():
        return None
    try:
        return pd.read_csv(csv)
    except Exception as e:
        st.warning(f"CSV {csv} unreadable: {e}")
        return None


def _render_grid(folder: Path, name: str, df: pd.DataFrame, cols_per_row: int = 5) -> None:
    img_dir = folder / "images"
    if not img_dir.exists():
        st.info(f"No images/ subfolder under {folder}")
        return
    imgs = sorted(img_dir.iterdir(), key=lambda p: p.name)
    # Build a {rank -> image_path} map by parsing filenames like "001_gt0.317_..."
    by_rank: Dict[int, Path] = {}
    for p in imgs:
        try:
            rank = int(p.name.split("_", 1)[0])
            by_rank[rank] = p
        except ValueError:
            continue

    for chunk_start in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            i = chunk_start + j
            if i >= len(df):
                break
            row = df.iloc[i]
            with cols[j]:
                rank = int(row["rank"])
                img_path = by_rank.get(rank)
                if img_path and img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption(f"(no image rank={rank})")
                gt = float(row["gt"])
                pred = float(row["pred"])
                abs_err = float(row["abs_err"])
                gender = "F" if int(row["gender"]) == 0 else "M"
                st.caption(
                    f"**#{rank}** · {gender} · gt=`{gt:.3f}` · pred=`{pred:.3f}` · "
                    f"|err|=`{abs_err:.3f}`"
                )


# === UI ===

st.title("Qualitative best/worst examples — per Optuna trial")
st.caption(f"Reading from `{TRACKING_URI}`")

with st.sidebar:
    st.header("Run selector")
    experiments = _list_experiments(TRACKING_URI)
    if not experiments:
        st.error("No MLflow experiments found.")
        st.stop()

    exp_label_to_id = {f"{name} ({eid})": eid for eid, name in experiments}
    selected_exp_label = st.selectbox("Experiment", list(exp_label_to_id.keys()))
    selected_exp_id = exp_label_to_id[selected_exp_label]

    runs = _list_runs(TRACKING_URI, selected_exp_id)
    if not runs:
        st.warning("No runs in this experiment.")
        st.stop()

    def _fmt(r: Dict[str, Any]) -> str:
        s = r["eval_score"]
        score_str = f"{s:.5f}" if isinstance(s, (int, float)) else "n/a"
        return f"{r['name']}  ·  score={score_str}  ·  [{r['status']}]"

    selected_run = st.selectbox("Run", runs, format_func=_fmt)
    show_metrics = st.checkbox("Show metric panel", value=True)

# === Main panel ===

if show_metrics:
    cols = st.columns(5)
    cols[0].metric("eval_score (reweighted)", f"{selected_run['eval_score']:.5f}"
                    if selected_run["eval_score"] is not None else "n/a")
    cols[1].metric("eval_score_raw", f"{selected_run['eval_score_raw']:.5f}"
                    if selected_run["eval_score_raw"] is not None else "n/a")
    cols[2].metric("err_F", f"{selected_run['err_F']:.5f}"
                    if selected_run["err_F"] is not None else "n/a")
    cols[3].metric("err_M", f"{selected_run['err_M']:.5f}"
                    if selected_run["err_M"] is not None else "n/a")
    cols[4].metric("err_diff", f"{selected_run['err_diff']:.5f}"
                    if selected_run["err_diff"] is not None else "n/a")

qual_dir = _download_qualitative(TRACKING_URI, selected_run["run_id"])
if qual_dir is None:
    st.info("No qualitative/ artifacts logged for this run "
            "(needs `save_qualitative_k > 0` and a non-skipped trial).")
    st.stop()

qual_path = Path(qual_dir)
worst_df = _load_csv(qual_path / "worst", "worst")
best_df = _load_csv(qual_path / "best", "best")

tab_worst, tab_best, tab_csv = st.tabs(["Worst-K (highest weighted_err)", "Best-K (lowest weighted_err)", "Raw CSVs"])

with tab_worst:
    if worst_df is None or worst_df.empty:
        st.info("No worst.csv found")
    else:
        st.caption(f"{len(worst_df)} samples")
        _render_grid(qual_path / "worst", "worst", worst_df)

with tab_best:
    if best_df is None or best_df.empty:
        st.info("No best.csv found")
    else:
        st.caption(f"{len(best_df)} samples")
        _render_grid(qual_path / "best", "best", best_df)

with tab_csv:
    if worst_df is not None:
        st.subheader("Worst-K")
        st.dataframe(worst_df, use_container_width=True)
    if best_df is not None:
        st.subheader("Best-K")
        st.dataframe(best_df, use_container_width=True)
