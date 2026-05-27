"""Streamlit viewer — two modes:

1. **Trials comparison (live)** — overlay metric curves across active Optuna sweeps,
   hover for trial params. Default view, auto-refresh.
2. **Qualitative viewer (per-run)** — best/worst-K image gallery for a selected run.

Reads the MLflow tracking DB (sqlite:///mlflow.db).

Launch :
    uvx --python 3.12 --with mlflow --with pandas --with pillow --with plotly \
        --from streamlit streamlit run scripts/qualitative_viewer.py \
        --server.port 8501 --server.address 0.0.0.0

By design this script avoids any project-internal import so it can run in a tiny
ephemeral venv (CPU partition, no torch).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Diagnostic preamble — printed to stdout BEFORE Streamlit takes over, so it ends
# up in the launcher's log file. Catches "package missing" / "wrong version" /
# "tracking DB unreadable" issues without needing to attach to the running process.
print(f"[viewer] Python = {sys.version.split()[0]} at {sys.executable}", flush=True)
print(f"[viewer] cwd = {os.getcwd()}", flush=True)
for mod in ("streamlit", "pandas", "mlflow", "PIL", "plotly"):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"[viewer] {mod} = {ver}", flush=True)
    except ImportError as e:
        print(f"[viewer] FATAL: cannot import {mod} ({e})", flush=True)
        sys.exit(2)

_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
if _tracking_uri.startswith("sqlite:///"):
    _db_path = _tracking_uri.replace("sqlite:///", "", 1)
    if not Path(_db_path).exists():
        print(f"[viewer] WARNING: mlflow.db not found at {_db_path} (relative to cwd) — "
              f"viewer will show 'No experiments'.", flush=True)
print(f"[viewer] All imports OK, launching Streamlit UI on MLFLOW_TRACKING_URI={_tracking_uri}", flush=True)

try:
    from streamlit_autorefresh import st_autorefresh  # non-blocking JS-timer rerun
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False
    print(f"[viewer] streamlit-autorefresh not installed → auto-refresh falls back to manual", flush=True)

import re
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Face-occ analytics", layout="wide")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Params we show in hover tooltips on the trials comparison chart. Order matters
# (top to bottom in the tooltip). Anything else still queryable via the param table.
HOVER_PARAMS = [
    "pretrained_source",
    "sampler_strategy",
    "loss_rw_strategy",
    "feature_fairness",
    "pooling_type",
    "learning_rate",
    "weight_decay",
    "augmentation_level",
    "num_train_epochs",
    "loss_fairness_lambda",
    "layer_decay",
]


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
        m = r.data.metrics
        # New explicit names (v3+), fallback to legacy short names (pre-v3 runs)
        def pick(*keys: str) -> Any:
            for k in keys:
                v = m.get(k)
                if v is not None:
                    return v
            return None
        out.append({
            "run_id": r.info.run_id,
            "name": r.info.run_name or r.info.run_id[:8],
            "status": r.info.status,
            "challenge_score_test_estimated": pick("eval_challenge_score_test_estimated", "eval_score", "val_score"),
            "challenge_score_val":            pick("eval_challenge_score_val", "eval_score_raw"),
            "err_F_test_estimated":           pick("eval_err_F_test_estimated", "eval_err_F"),
            "err_M_test_estimated":           pick("eval_err_M_test_estimated", "eval_err_M"),
            "err_diff_test_estimated":        pick("eval_err_diff_test_estimated", "eval_err_diff"),
            "mae_pct_test_estimated":         pick("eval_mae_pct_test_estimated"),
            "r2_test_estimated":              pick("eval_r2_test_estimated"),
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


# === Helpers for "Trials comparison" mode ===

@st.cache_data(ttl=30, show_spinner=False)
def _list_runs_full(tracking_uri: str, experiment_id: str) -> List[Dict[str, Any]]:
    """Like _list_runs but also returns params + final metrics for tooltip use."""
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
            "start_time": r.info.start_time,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
        })
    return out


@st.cache_data(ttl=30, show_spinner=False)
def _list_available_metrics(tracking_uri: str, experiment_ids: Tuple[str, ...]) -> List[str]:
    """Union of metric keys logged across the latest few runs of the given experiments."""
    client = _get_client(tracking_uri)
    keys: set = set()
    for eid in experiment_ids:
        runs = client.search_runs([eid], max_results=20)
        for r in runs:
            keys.update(r.data.metrics.keys())
    keys = {k for k in keys if k.startswith("eval_") or k in {"train_loss", "loss"}}
    return sorted(keys)


@st.cache_data(ttl=20, show_spinner=False)
def _fetch_metric_history(tracking_uri: str, run_id: str, metric: str) -> List[Tuple[int, float]]:
    client = _get_client(tracking_uri)
    try:
        hist = client.get_metric_history(run_id, metric)
    except Exception:
        return []
    return [(int(m.step), float(m.value)) for m in hist]


# Match `_trial<N>` followed by underscore (timestamp suffix in optimize.py)
# or end of string (legacy / direct). Captures the real Optuna trial number.
_TRIAL_NUM_RE = re.compile(r"_trial(\d+)(?:_|$)")


def _parse_trial_number(run_name: str, fallback: int) -> int:
    """Extract the real Optuna trial number from MLflow run names like
    `dinov3-vitb16-3090-v4_trial28_20260527_093412` → 28. Falls back to the provided
    index when the name doesn't match."""
    m = _TRIAL_NUM_RE.search(run_name or "")
    return int(m.group(1)) if m else fallback


def _clear_trials_caches() -> None:
    """Clear only the caches relevant to the trials-comparison view, not the
    expensive qualitative-artifact cache."""
    _list_experiments.clear()
    _list_runs_full.clear()
    _list_available_metrics.clear()
    _fetch_metric_history.clear()


def _build_hover_text(params: Dict[str, str], name: str, exp_name: str) -> str:
    lines = [f"<b>{name}</b>", f"<i>{exp_name}</i>"]
    for key in HOVER_PARAMS:
        v = params.get(key)
        if v is None:
            continue
        # truncate long values (e.g. full ibot:runs:/.../encoder paths)
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        lines.append(f"{key}={v_str}")
    return "<br>".join(lines)


def _render_trials_comparison() -> None:
    import plotly.graph_objects as go

    experiments = _list_experiments(TRACKING_URI)
    if not experiments:
        st.error("No MLflow experiments found.")
        return

    # Default selection: keep optuna-* experiments (one per arch); fallback to all
    default_exps = [(eid, name) for eid, name in experiments if name.startswith("optuna-")]
    if not default_exps:
        default_exps = experiments

    with st.sidebar:
        st.header("Trials comparison")
        view_mode = st.radio(
            "View",
            ["Inter-trial (convergence)", "Intra-trial (training curves)"],
            index=0,
            help=(
                "Inter-trial: 1 point per trial = score final, vue convergence d'Optuna.\n"
                "Intra-trial: 1 courbe par trial = évolution training step-par-step."
            ),
        )
        exp_label_to_id = {name: eid for eid, name in experiments}
        default_labels = [name for _, name in default_exps]
        selected_labels = st.multiselect(
            "Experiments",
            list(exp_label_to_id.keys()),
            default=default_labels,
        )
        if not selected_labels:
            st.info("Select at least one experiment.")
            return
        selected_exp_ids = tuple(exp_label_to_id[label] for label in selected_labels)

        available_metrics = _list_available_metrics(TRACKING_URI, selected_exp_ids)
        if not available_metrics:
            st.warning("No eval_* metrics found yet.")
            return
        default_metric = (
            "eval_challenge_score_test_estimated"
            if "eval_challenge_score_test_estimated" in available_metrics
            else available_metrics[0]
        )
        selected_metric = st.selectbox(
            "Metric",
            available_metrics,
            index=available_metrics.index(default_metric),
        )

        max_per_exp = st.slider("Max trials per experiment", 5, 200, 50)
        log_y = st.checkbox("Log Y axis", value=True)
        show_running_only = st.checkbox("Hide finished trials", value=False)
        if view_mode == "Inter-trial (convergence)":
            show_best_so_far = st.checkbox("Show best-so-far envelope", value=True)
            hide_inf = st.checkbox("Hide failed trials (score=inf/NaN)", value=True)
            normalize_x = False
        else:
            show_best_so_far = False
            hide_inf = False
            normalize_x = st.checkbox(
                "Normalize x-axis to % progress",
                value=True,
                help="x = step / max(step) per trial. Aligns curves with different num_train_epochs.",
            )
        if _HAS_AUTOREFRESH:
            auto_refresh_sec = st.selectbox(
                "Auto-refresh",
                [0, 15, 30, 60, 120],
                index=2,  # default 30s — non-blocking via JS timer
                format_func=lambda s: "off" if s == 0 else f"every {s}s",
            )
        else:
            auto_refresh_sec = 0
            st.caption("Auto-refresh disabled (streamlit-autorefresh not installed)")
        if st.button("Refresh now"):
            _clear_trials_caches()
            st.rerun()

    # Schedule non-blocking JS-timer rerun BEFORE rendering so the timer survives
    # any subsequent st.stop() or exceptions in the render path.
    if _HAS_AUTOREFRESH and auto_refresh_sec > 0:
        st_autorefresh(interval=auto_refresh_sec * 1000, key="trials_auto_refresh")
        _clear_trials_caches()  # ensure each refresh tick fetches fresh data

    if view_mode == "Inter-trial (convergence)":
        _render_inter_trial(
            selected_labels, exp_label_to_id, selected_metric,
            max_per_exp, log_y, show_running_only, show_best_so_far, hide_inf,
        )
    else:
        _render_intra_trial(
            selected_labels, exp_label_to_id, selected_metric,
            max_per_exp, log_y, show_running_only, normalize_x,
        )


def _final_metric_value(
    tracking_uri: str, run: Dict[str, Any], metric: str
) -> Optional[float]:
    """Return the final (last logged) value of `metric` for the given run.
    Falls back to run.data.metrics if no history available."""
    hist = _fetch_metric_history(tracking_uri, run["run_id"], metric)
    if hist:
        return float(hist[-1][1])
    v = run["metrics"].get(metric)
    return float(v) if v is not None else None


def _render_inter_trial(
    selected_labels: List[str],
    exp_label_to_id: Dict[str, str],
    selected_metric: str,
    max_per_exp: int,
    log_y: bool,
    show_running_only: bool,
    show_best_so_far: bool,
    hide_inf: bool,
) -> None:
    import math
    import plotly.graph_objects as go

    st.title("Trials comparison — convergence (inter-trial)")
    st.caption(
        f"1 point = 1 trial (score final). Enveloppe noire = best-so-far. "
        f"Hover pour les params. Source: `{TRACKING_URI}`"
    )

    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    rows_for_table: List[Dict[str, Any]] = []

    for exp_idx, label in enumerate(selected_labels):
        eid = exp_label_to_id[label]
        runs = _list_runs_full(TRACKING_URI, eid)
        if show_running_only:
            runs = [r for r in runs if r["status"] == "RUNNING"]
        runs = runs[:max_per_exp]
        # Sort by start_time ASC so trial index = chronological order
        runs = sorted(runs, key=lambda r: r.get("start_time", 0) or 0)

        color = palette[exp_idx % len(palette)]
        xs: List[int] = []
        ys: List[float] = []
        hovers: List[str] = []
        for i, r in enumerate(runs, start=1):
            val = _final_metric_value(TRACKING_URI, r, selected_metric)
            if val is None:
                continue
            if hide_inf and (math.isinf(val) or math.isnan(val)):
                continue
            # Use the real Optuna trial number when parseable (e.g. "_trial28" → 28),
            # so x-axis matches MLflow run names & Optuna dashboard. Falls back to
            # chronological index for legacy runs without the suffix.
            trial_idx = _parse_trial_number(r["name"], i)
            xs.append(trial_idx)
            ys.append(val)
            hovers.append(_build_hover_text(r["params"], r["name"], label))
            rows_for_table.append({
                "experiment": label,
                "trial_idx": trial_idx,
                "trial": r["name"],
                "status": r["status"],
                "final_value": val,
                **{k: r["params"].get(k) for k in HOVER_PARAMS},
            })

        if not xs:
            continue

        # Scatter of per-trial finals
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=f"{label}  ({len(xs)} trials)",
            marker=dict(color=color, size=8, opacity=0.7, line=dict(width=0.5, color="white")),
            hovertext=hovers,
            hovertemplate="trial=%{x}<br>"
                          f"{selected_metric}=%{{y:.5g}}<br>"
                          "%{hovertext}<extra></extra>",
        ))

        # Best-so-far envelope (running min)
        if show_best_so_far and ys:
            best = []
            cur = float("inf")
            for v in ys:
                if v < cur:
                    cur = v
                best.append(cur)
            fig.add_trace(go.Scatter(
                x=xs, y=best, mode="lines",
                name=f"best-so-far {label[:20]}",
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.update_layout(
        height=600,
        xaxis_title="Trial index (chronological)",
        yaxis_title=selected_metric,
        yaxis_type="log" if log_y else "linear",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if rows_for_table:
        df = pd.DataFrame(rows_for_table).sort_values("final_value", na_position="last")

        # Top-N per experiment — focus on balancing params for cross-arch comparison
        st.markdown("### Top trials per experiment")
        top_n = st.slider("Top N per experiment", 1, 20, 5, key="topn_per_exp")
        top_per_exp = df.groupby("experiment", as_index=False).head(top_n)
        # Re-sort: experiments alphabetically, then by final_value ASC within each
        top_per_exp = top_per_exp.sort_values(["experiment", "final_value"], na_position="last")
        # Focus columns on balancing for the comparison
        focus_cols = [
            "experiment", "trial_idx", "trial", "final_value",
            "pretrained_source", "sampler_strategy", "loss_rw_strategy", "feature_fairness",
            "pooling_type", "learning_rate", "loss_fairness_lambda",
        ]
        focus_cols = [c for c in focus_cols if c in top_per_exp.columns]
        st.dataframe(top_per_exp[focus_cols], use_container_width=True, hide_index=True)

        st.markdown("### All trials")
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_intra_trial(
    selected_labels: List[str],
    exp_label_to_id: Dict[str, str],
    selected_metric: str,
    max_per_exp: int,
    log_y: bool,
    show_running_only: bool,
    normalize_x: bool = False,
) -> None:
    import plotly.graph_objects as go

    st.title("Trials comparison — training curves (intra-trial)")
    x_label = "% progress" if normalize_x else "step"
    st.caption(
        f"1 courbe par trial = évolution {x_label}-par-{x_label}. "
        f"Hover pour les params. Source: `{TRACKING_URI}`"
    )

    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    rows_for_table: List[Dict[str, Any]] = []

    for exp_idx, label in enumerate(selected_labels):
        eid = exp_label_to_id[label]
        runs = _list_runs_full(TRACKING_URI, eid)
        if show_running_only:
            runs = [r for r in runs if r["status"] == "RUNNING"]
        runs = runs[:max_per_exp]
        color = palette[exp_idx % len(palette)]
        trials_with_data = 0
        for r in runs:
            hist = _fetch_metric_history(TRACKING_URI, r["run_id"], selected_metric)
            if not hist:
                continue
            trials_with_data += 1
            steps = [pt[0] for pt in hist]
            values = [pt[1] for pt in hist]
            # Normalize x to % progress so trials with different num_train_epochs align
            if normalize_x and steps and max(steps) > 0:
                xmax = max(steps)
                x_plot = [s / xmax * 100 for s in steps]
            else:
                x_plot = steps
            hover = _build_hover_text(r["params"], r["name"], label)
            x_template = "% =%{x:.1f}" if normalize_x else "step=%{x}"
            fig.add_trace(go.Scatter(
                x=x_plot,
                y=values,
                mode="lines",
                name=f"{label[:20]}/{r['name'][:12]}",
                line=dict(width=1.5, color=color),
                opacity=0.7,
                hovertext=hover,
                hovertemplate=f"{x_template}<br>"
                              f"{selected_metric}=%{{y:.5g}}<br>"
                              "%{hovertext}<extra></extra>",
                showlegend=False,
            ))
            rows_for_table.append({
                "experiment": label,
                "trial": r["name"],
                "status": r["status"],
                "last_step": steps[-1] if steps else None,
                "last_value": values[-1] if values else None,
                **{k: r["params"].get(k) for k in HOVER_PARAMS},
            })

        # Legend dummy per experiment
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"{label}  ({trials_with_data} trials)",
            line=dict(color=color, width=3),
        ))

    fig.update_layout(
        height=600,
        xaxis_title="step",
        yaxis_title=selected_metric,
        yaxis_type="log" if log_y else "linear",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if rows_for_table:
        st.markdown("### Trials sweep table")
        df = pd.DataFrame(rows_for_table)
        st.dataframe(df, use_container_width=True, hide_index=True)


# === UI dispatcher ===

with st.sidebar:
    mode = st.radio(
        "Mode",
        ["Trials comparison (live)", "Qualitative viewer (per-run)"],
        index=0,
    )
    st.divider()

if mode == "Trials comparison (live)":
    _render_trials_comparison()
    st.stop()

# === Qualitative viewer (legacy) ===

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

    # Sort by score ASC (best first); runs without score go to the bottom.
    def _score_key(r: Dict[str, Any]) -> float:
        s = r.get("challenge_score_test_estimated")
        return float(s) if isinstance(s, (int, float)) and s == s else float("inf")
    runs = sorted(runs, key=_score_key)

    sort_order = st.radio("Order", ["Best → worst", "Worst → best", "Most recent first"], index=0)
    if sort_order == "Worst → best":
        runs = list(reversed(runs))
    elif sort_order == "Most recent first":
        # _list_runs already returns DESC by start_time
        runs = _list_runs(TRACKING_URI, selected_exp_id)

    def _fmt(r: Dict[str, Any]) -> str:
        s = r["challenge_score_test_estimated"]
        score_str = f"{s:.5f}" if isinstance(s, (int, float)) else "n/a"
        return f"{r['name']}  ·  score={score_str}  ·  [{r['status']}]"

    selected_run = st.selectbox("Run", runs, format_func=_fmt)
    show_metrics = st.checkbox("Show metric panel", value=True)

# === Main panel ===

def _fmt_val(v: Any, decimals: int = 5, suffix: str = "") -> str:
    return f"{v:.{decimals}f}{suffix}" if isinstance(v, (int, float)) else "n/a"


if show_metrics:
    st.markdown("### Score total")
    st.caption("`Score = (err_F + err_M)/2 + |err_F - err_M|` (formule officielle). `test-estimated` = val reweightée par `P_test/P_train` (estimation perf test) — c'est ce que pilote Optuna. `val direct` = même formule sans reweight (diagnostic).")
    cols = st.columns(2)
    cols[0].metric("Score (test-estimated)",   _fmt_val(selected_run["challenge_score_test_estimated"]),
                   help="LA cible Optuna. Plus c'est bas, mieux c'est.")
    cols[1].metric("Score (val direct)",       _fmt_val(selected_run["challenge_score_val"]),
                   help="Pas de correction du shift Y. Diff vs test-estimated = magnitude du shift.")

    # Decompose score = mean_err + |err_diff|  (because (err_F+err_M)/2 + |err_F-err_M| = mean + diff)
    err_f = selected_run["err_F_test_estimated"]
    err_m = selected_run["err_M_test_estimated"]
    mean_err = (err_f + err_m) / 2 if isinstance(err_f, (int, float)) and isinstance(err_m, (int, float)) else None
    st.markdown("### Composantes du score (test-estimated)")
    cols = st.columns(4)
    cols[0].metric("mean_err",       _fmt_val(mean_err),
                   help="(err_F + err_M) / 2 — performance moyenne sur les 2 genres")
    cols[1].metric("err_diff",       _fmt_val(selected_run["err_diff_test_estimated"]),
                   help="|err_F - err_M| — pénalité de fairness genre")
    cols[2].metric("err_F",          _fmt_val(err_f),
                   help="Σwᵢ(pᵢ-yᵢ)² / Σwᵢ sur les samples Female (wᵢ=1/30+yᵢ)")
    cols[3].metric("err_M",          _fmt_val(err_m),
                   help="Σwᵢ(pᵢ-yᵢ)² / Σwᵢ sur les samples Male")

    st.markdown("### Métriques humaines (test-estimated)")
    cols = st.columns(2)
    cols[0].metric("MAE",            _fmt_val(selected_run["mae_pct_test_estimated"], decimals=2, suffix=" %"),
                   help="Erreur absolue moyenne en points de % d'occlusion : le modèle se trompe en moyenne de X points")
    cols[1].metric("R²",             _fmt_val(selected_run["r2_test_estimated"], decimals=3),
                   help="0 = modèle trivial (moyenne constante), 1 = parfait")

qual_dir = _download_qualitative(TRACKING_URI, selected_run["run_id"])
if qual_dir is None:
    st.info("No qualitative/ artifacts logged for this run "
            "(needs `save_qualitative_k > 0` and a non-skipped trial).")
    st.stop()

qual_path = Path(qual_dir)
worst_df = _load_csv(qual_path / "worst", "worst")
best_df = _load_csv(qual_path / "best", "best")

tab_diag, tab_worst, tab_best, tab_csv = st.tabs([
    "📊 Diagnostic charts",
    "Worst-K (highest weighted_err)",
    "Best-K (lowest weighted_err)",
    "Raw CSVs",
])

with tab_diag:
    diag_dir = qual_path / "diagnostics"
    chart_path = diag_dir / "error_vs_occlusion_and_density.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
        st.caption(
            "**Gauche** : MAE moyen par bin de Y (occlusion), avec courbes séparées F / M / overall. "
            "Permet de repérer où le modèle pèche (low/mid/high occlusion) et si le gap F-M est constant "
            "ou explose avec Y.  \n"
            "**Droite** : densité de `|pred - gt|` par genre. Permet de distinguer "
            "biais systématique (distribution shiftée) vs queues lourdes (quelques très mauvaises preds)."
        )
    else:
        st.info(
            "Aucun diagnostic chart pour ce run.\n\n"
            "Les charts sont produits par `_save_diagnostic_charts()` dans `train.py` lors du dump qualitatif "
            "(activé via `save_qualitative_k > 0`). Run nécessaire avec le code v3+."
        )

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
