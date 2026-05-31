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
    "axis1_power",
    "axis2_power",
    "feature_fairness",
    "mmd_lambda",
    "adv_lambda",
    "loss_focal_gamma",
    "loss_fairness_lambda",
    "pretrained_source",
    "pooling_type",
    "augmentation_level",
    "learning_rate",
    "weight_decay",
    "num_train_epochs",
    "layer_decay",
]


@st.cache_resource(show_spinner=False)
def _get_client(tracking_uri: str):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=tracking_uri)


_EXP_FILTER = os.environ.get("FACE_OCC_EXP_FILTER", "").strip()


@st.cache_data(ttl=30, show_spinner=False)
def _list_experiments(tracking_uri: str) -> List[Tuple[str, str]]:
    """List MLflow experiments, optionally filtered by FACE_OCC_EXP_FILTER env var.

    Filter is a substring match against experiment name (e.g. "v6" → only `*-v6` experiments).
    Empty filter → all experiments shown.
    """
    client = _get_client(tracking_uri)
    exps = client.search_experiments()
    pairs = [(e.experiment_id, e.name) for e in exps]
    if _EXP_FILTER:
        pairs = [(eid, name) for eid, name in pairs if _EXP_FILTER in name]
    return sorted(pairs, key=lambda x: x[1])


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
            # v8 clean : noms canoniques uniquement (pas de legacy fallback).
            "challenge_score":     pick("eval_challenge_score"),
            "challenge_score_raw": pick("eval_challenge_score_raw"),
            "err_F":               pick("eval_err_F"),
            "err_M":               pick("eval_err_M"),
            "err_diff":            pick("eval_err_diff"),
            "mae_pct":             pick("eval_mae_pct"),
            "r2":                  pick("eval_r2"),
        })
    return out


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_run_params(tracking_uri: str, run_id: str) -> Dict[str, str]:
    """Return all MLflow params logged for the given run (config dump)."""
    client = _get_client(tracking_uri)
    try:
        return dict(client.get_run(run_id).data.params)
    except Exception as e:
        st.warning(f"Could not fetch params for {run_id}: {e}")
        return {}


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

    # Default selection: v9 only (current sweep version on this branch).
    # Fallback ladders: optuna-*-v9 → all optuna-* → all experiments. v6/v7/v8 et antérieurs
    # restent sélectionnables manuellement via le multiselect.
    import re
    _CURRENT_VERSION_RE = re.compile(r"-v10$")
    default_exps = [(eid, name) for eid, name in experiments
                    if name.startswith("optuna-") and _CURRENT_VERSION_RE.search(name)]
    if not default_exps:
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
        # v8 : noms canoniques uniquement, pas de legacy fallback.
        if "eval_challenge_score" in available_metrics:
            default_metric = "eval_challenge_score"
        else:
            default_metric = available_metrics[0]
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
            sort_order = st.radio(
                "Sort tables",
                ["Best score → worst", "Worst score → best", "Best fairness (low err_diff)", "Worst fairness (high err_diff)"],
                index=0,
                help="Affecte les tables 'Top trials per experiment' et 'All trials'. "
                     "Fairness sort = trier par eval_err_diff (utile pour voir les trials équitables "
                     "même si score global moins bon).",
            )
            normalize_x = False
        else:
            show_best_so_far = False
            hide_inf = False
            sort_order = "Best → worst"
            normalize_x = st.checkbox(
                "Normalize x-axis to % progress",
                value=True,
                help="x = step / max(step) per trial. Aligns curves with different num_train_epochs.",
            )
        if _HAS_AUTOREFRESH:
            auto_refresh_sec = st.selectbox(
                "Auto-refresh",
                [0, 3, 5, 10, 15, 30, 60, 120],
                index=2,  # default 5s — non-blocking via JS timer
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
        # v8 : 4 sort orders. "Best fairness" = trier par eval_err_diff ascending
        sort_by_err_diff = sort_order in ("Best fairness (low err_diff)", "Worst fairness (high err_diff)")
        sort_descending = sort_order in ("Worst score → best", "Worst fairness (high err_diff)")
        _render_inter_trial(
            selected_labels, exp_label_to_id, selected_metric,
            max_per_exp, log_y, show_running_only, show_best_so_far, hide_inf,
            sort_descending=sort_descending,
            sort_by_err_diff=sort_by_err_diff,
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
    sort_descending: bool = False,
    sort_by_err_diff: bool = False,
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
            # v8 : récupère err_diff depuis metrics MLflow pour le sort fairness
            err_diff_val = None
            for k in ("eval_err_diff",):
                v = r["metrics"].get(k)
                if v is not None:
                    err_diff_val = float(v)
                    break
            rows_for_table.append({
                "experiment": label,
                "trial_idx": trial_idx,
                "trial": r["name"],
                "status": r["status"],
                "final_value": val,
                "err_diff": err_diff_val,
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
        # v8 : sort par score OU par err_diff (fairness), asc/desc selon sort_descending.
        sort_col = "err_diff" if sort_by_err_diff else "final_value"
        df = pd.DataFrame(rows_for_table).sort_values(
            sort_col, ascending=not sort_descending, na_position="last",
        )

        focus_cols = [
            "experiment", "trial_idx", "trial", "final_value", "err_diff",
            # v10 — unified rebalancing
            "axis1_power", "axis2_power",
            "feature_fairness", "mmd_lambda", "adv_lambda",
            "loss_focal_gamma", "loss_fairness_lambda",
            # Architecture + hyperparams
            "pretrained_source", "pooling_type",
            "learning_rate", "weight_decay",
        ]

        def _family(name: str) -> str:
            n = name.lower()
            if "sapiens" in n:
                return "sapiens"
            if "dino" in n:
                return "dino"
            return "other"

        df["_family"] = df["experiment"].map(_family)

        families = [("Dino", "dino"), ("Sapiens", "sapiens"), ("Other", "other")]
        for fam_label, fam_key in families:
            fam_df = df[df["_family"] == fam_key]
            if fam_df.empty:
                continue
            n_per_exp = fam_df.groupby("experiment").size()
            max_n = int(n_per_exp.max())
            st.markdown(f"### Top trials per experiment — {fam_label}")
            st.caption(
                f"{len(fam_df)} trials across {len(n_per_exp)} expé · "
                f"max/expé={max_n} · min/expé={int(n_per_exp.min())}"
            )
            top_n = st.slider(
                f"Top N per experiment — {fam_label}",
                1, max(20, max_n), min(5, max_n),
                key=f"topn_per_exp_{fam_key}",
            )
            top_per_exp = fam_df.groupby("experiment", as_index=False).head(top_n)
            top_per_exp = top_per_exp.sort_values(
                ["experiment", sort_col],
                ascending=[True, not sort_descending],
                na_position="last",
            )
            cols = [c for c in focus_cols if c in top_per_exp.columns]
            st.dataframe(top_per_exp[cols], use_container_width=True, hide_index=True)

        # Per-axis breakdown : agrégation (best/avg/med/n_trials) par valeur (catégoriels)
        # ou par bin (continus). Permet d'identifier d'un coup d'œil quels choix gagnent.
        AXIS_CATEGORICAL = [
            "pretrained_source", "pooling_type", "feature_fairness",
        ]
        # v8 : breakdown des continus binnés en quartiles. Permet de voir si γ haut/bas
        # marche mieux, si focal_gamma converge vers une zone, etc.
        AXIS_CONTINUOUS = [
            "axis1_power", "axis2_power",
            "mmd_lambda",
            "loss_focal_gamma",
            "learning_rate", "weight_decay",
            "head_dropout", "backbone_drop_path_rate",
        ]
        N_BINS_CONTINUOUS = 4
        for fam_label, fam_key in families:
            fam_df = df[df["_family"] == fam_key]
            if fam_df.empty:
                continue
            st.markdown(f"### Per-axis breakdown — {fam_label}")
            st.caption(
                f"Catégoriels : 1 ligne par choix. Continus : 4 quantiles. "
                f"Best = min score dans le groupe. Trier par best ASC → identifier les "
                f"valeurs gagnantes pour pruner le search space dans une future version."
            )
            rows: List[Dict[str, Any]] = []
            # Catégoriels
            for axis in AXIS_CATEGORICAL:
                if axis not in fam_df.columns:
                    continue
                axis_df = fam_df[["final_value", axis]].dropna(subset=[axis])
                axis_df = axis_df[axis_df["final_value"].notna()]
                if axis_df.empty:
                    continue
                for choice, sub in axis_df.groupby(axis):
                    if len(sub) == 0:
                        continue
                    rows.append({
                        "axis": axis, "choice": str(choice), "n": int(len(sub)),
                        "best": float(sub["final_value"].min()),
                        "avg": float(sub["final_value"].mean()),
                        "med": float(sub["final_value"].median()),
                    })
            # Continus binnés en quartiles
            for axis in AXIS_CONTINUOUS:
                if axis not in fam_df.columns:
                    continue
                axis_df = fam_df[["final_value", axis]].copy()
                # Convert string params → float (MLflow stores all as strings)
                axis_df[axis] = pd.to_numeric(axis_df[axis], errors="coerce")
                axis_df = axis_df.dropna(subset=[axis])
                axis_df = axis_df[axis_df["final_value"].notna()]
                if len(axis_df) < N_BINS_CONTINUOUS:
                    continue
                try:
                    bins = pd.qcut(axis_df[axis], q=N_BINS_CONTINUOUS, duplicates="drop")
                except ValueError:
                    continue
                for interval, sub in axis_df.groupby(bins, observed=True):
                    if len(sub) == 0:
                        continue
                    # Format interval as readable range
                    label = f"[{interval.left:.3f}, {interval.right:.3f}]"
                    rows.append({
                        "axis": axis, "choice": label, "n": int(len(sub)),
                        "best": float(sub["final_value"].min()),
                        "avg": float(sub["final_value"].mean()),
                        "med": float(sub["final_value"].median()),
                    })
            if not rows:
                st.info("No params available for this family.")
                continue
            ax_df = pd.DataFrame(rows).sort_values(["axis", "best"]).reset_index(drop=True)
            for col in ("best", "avg", "med"):
                ax_df[col] = ax_df[col].round(5)
            st.dataframe(ax_df, use_container_width=True, hide_index=True)

        st.markdown("### All trials")
        st.dataframe(df.drop(columns=["_family"]), use_container_width=True, hide_index=True)


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
        ["Trials comparison (live)", "Qualitative viewer (per-run)", "Post-processing effect (test holdout)"],
        index=0,
        key="mode_selector",
    )
    st.divider()

if mode == "Trials comparison (live)":
    _render_trials_comparison()
    st.stop()


def _render_isotonic_effect() -> None:
    """Visualize the effect of post-hoc calibration (isotonic + linear + PCHIP) on
    the test holdout (v12+).

    Requires artifacts qualitative/test_holdout_predictions.csv and
    qualitative/calibrator_mappings.csv from the run.
    """
    st.title("Post-processing effect — test holdout (5%)")
    st.caption(
        "Test holdout = ~5% des samples sampled to match P_test via H_C. "
        "3 calibrators (isotonic, linear, PCHIP spline) sont fit sur val (iid P_train) "
        "puis appliqués au test holdout. Compare pred RAW vs chaque calibration."
    )

    experiments = _list_experiments(TRACKING_URI)
    if not experiments:
        st.error("No MLflow experiments found.")
        return
    exp_label_to_id = {f"{name} ({eid})": eid for eid, name in experiments}
    exp_labels = list(exp_label_to_id.keys())
    # Default to v12 experiment if present
    default_idx = 0
    for i, label in enumerate(exp_labels):
        if "v12" in label.lower():
            default_idx = i
            break
    selected_exp_label = st.sidebar.selectbox("Experiment", exp_labels, index=default_idx, key="pp_exp")
    selected_exp_id = exp_label_to_id[selected_exp_label]

    runs = _list_runs(TRACKING_URI, selected_exp_id)
    if not runs:
        st.warning("No runs in this experiment.")
        return
    runs = sorted(runs, key=lambda r: float(r.get("challenge_score") or float("inf")))

    def _fmt(r: Dict[str, Any]) -> str:
        s = r["challenge_score"]
        score_str = f"{s:.5f}" if isinstance(s, (int, float)) else "n/a"
        return f"{r['name']}  ·  score={score_str}"
    selected_run = st.sidebar.selectbox("Run", runs, format_func=_fmt, key="pp_run")

    folder_str = _download_qualitative(TRACKING_URI, selected_run["run_id"])
    if not folder_str:
        st.error("Pas d'artifact `qualitative/` dans ce run — test holdout pas activé "
                 "(ajouter `test_split_ratio: 0.05` dans le yaml data).")
        return
    folder = Path(folder_str)
    pred_csv = folder / "test_holdout_predictions.csv"
    cal_curves_csv = folder / "calibrator_mappings.csv"
    iso_csv = folder / "isotonic_mapping.csv"  # legacy fallback
    if not pred_csv.exists():
        st.error(f"Pas de `test_holdout_predictions.csv` dans {folder} — v11 ou avant probablement.")
        return

    df_pred = pd.read_csv(pred_csv)
    df_cal = pd.read_csv(cal_curves_csv) if cal_curves_csv.exists() else None
    df_iso = pd.read_csv(iso_csv) if iso_csv.exists() else None

    # Detect available calibrator columns
    cal_methods = [c.replace("pred_", "") for c in df_pred.columns
                    if c.startswith("pred_") and c not in ("pred_raw",)]
    # Fall back to legacy "iso" only
    if not cal_methods and "pred_iso" in df_pred.columns:
        cal_methods = ["iso"]

    # === Metrics summary ===
    st.markdown("### Métriques test holdout")
    raw_score = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_score_raw")
    best_cal_score = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_score_best_cal")
    raw_diff = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_err_diff_raw")

    cols = st.columns(2 + len(cal_methods))
    cols[0].metric("Score RAW", f"{raw_score:.5f}" if raw_score is not None else "n/a")
    if best_cal_score is not None:
        delta_best = f"{(best_cal_score - raw_score):+.5f}" if raw_score is not None else None
        cols[1].metric("Score BEST cal", f"{best_cal_score:.5f}",
                        delta=delta_best, delta_color="inverse")
    else:
        cols[1].metric("Score BEST cal", "n/a")
    for i, m in enumerate(cal_methods):
        s_m = _final_metric_value(TRACKING_URI, selected_run, f"test_holdout_score_{m}")
        delta_m = f"{(s_m - raw_score):+.5f}" if (s_m is not None and raw_score is not None) else None
        cols[2 + i].metric(f"Score {m}", f"{s_m:.5f}" if s_m is not None else "n/a",
                             delta=delta_m, delta_color="inverse")
    st.caption("Delta négatif (vert) = amélioration. Test holdout = ~4985 samples, distribution P_test.")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly required for these plots — pip install plotly")
        return

    # === 1. Calibration mapping curves (3 methods) ===
    if df_cal is not None and len(df_cal) > 0:
        st.markdown("### 1. Courbes de calibration apprises (par méthode, par gender)")
        st.caption("Mapping pred_raw → pred_corrigé. Diagonale = pas de correction.")
        c1, c2 = st.columns(2)
        method_colors = {"isotonic": "#d62728", "linear": "#2ca02c", "pchip": "#9467bd", "iso": "#d62728"}
        with c1:
            fig_F = go.Figure()
            fig_F.add_trace(go.Scatter(x=df_cal["x"], y=df_cal["x"], mode="lines",
                                         name="y=x", line=dict(color="gray", dash="dash")))
            for m in cal_methods:
                col = f"{m}_F"
                if col in df_cal.columns:
                    fig_F.add_trace(go.Scatter(x=df_cal["x"], y=df_cal[col], mode="lines",
                                                 name=m, line=dict(color=method_colors.get(m, "black"), width=3)))
            fig_F.update_layout(title="Female", xaxis_title="pred_raw", yaxis_title="pred_corrigé",
                                  height=400, hovermode="x")
            st.plotly_chart(fig_F, use_container_width=True)
        with c2:
            fig_M = go.Figure()
            fig_M.add_trace(go.Scatter(x=df_cal["x"], y=df_cal["x"], mode="lines",
                                         name="y=x", line=dict(color="gray", dash="dash")))
            for m in cal_methods:
                col = f"{m}_M"
                if col in df_cal.columns:
                    fig_M.add_trace(go.Scatter(x=df_cal["x"], y=df_cal[col], mode="lines",
                                                 name=m, line=dict(color=method_colors.get(m, "black"), width=3)))
            fig_M.update_layout(title="Male", xaxis_title="pred_raw", yaxis_title="pred_corrigé",
                                  height=400, hovermode="x")
            st.plotly_chart(fig_M, use_container_width=True)
    elif df_iso is not None and len(df_iso) > 0:
        # Legacy fallback: only isotonic mapping available
        st.markdown("### 1. Courbe isotonic apprise (par gender) — legacy v11")
        fig_iso = go.Figure()
        fig_iso.add_trace(go.Scatter(x=df_iso["x"], y=df_iso["x"], mode="lines",
                                       name="y=x", line=dict(color="gray", dash="dash")))
        fig_iso.add_trace(go.Scatter(x=df_iso["x"], y=df_iso["iso_F"], mode="lines",
                                       name="F iso", line=dict(color="#d62728", width=3)))
        fig_iso.add_trace(go.Scatter(x=df_iso["x"], y=df_iso["iso_M"], mode="lines",
                                       name="M iso", line=dict(color="#1f77b4", width=3)))
        fig_iso.update_layout(xaxis_title="pred_raw", yaxis_title="pred_corrigé",
                                height=400, hovermode="x")
        st.plotly_chart(fig_iso, use_container_width=True)

    # === 2. Distributions pred RAW + chaque cal + GT vs P_test ref, par gender ===
    st.markdown("### 2. Distribution prédictions vs GT vs P_test (réf PDF), par gender")
    st.caption("`P_test PDF` (noir pointillé) = distribution Y du test challenge extraite du PDF. "
                "Cible vers laquelle preds doivent converger.")

    # P_test reference (marginal Y) — same overlay on F and M panels
    try:
        from src.utils.distribution import _TEST_PMF, N_BINS as _NB, BIN_WIDTH as _BW
        ref_x = np.array([(b + 0.5) * _BW for b in range(_NB)])
        ref_density = np.asarray(_TEST_PMF) / _BW
    except ImportError:
        ref_x = ref_density = None

    method_color_F = {"raw": "#d62728", "isotonic": "#ff7f0e", "linear": "#2ca02c", "pchip": "#9467bd", "iso": "#ff7f0e"}
    method_color_M = {"raw": "#1f77b4", "isotonic": "#17becf", "linear": "#2ca02c", "pchip": "#9467bd", "iso": "#17becf"}

    g_F = df_pred[df_pred["gender"] < 0.5]
    g_M = df_pred[df_pred["gender"] >= 0.5]
    c1, c2 = st.columns(2)
    with c1:
        fig_F = go.Figure()
        fig_F.add_trace(go.Histogram(x=g_F["pred_raw"], name="pred RAW", opacity=0.45, nbinsx=40,
                                       marker_color=method_color_F["raw"], histnorm="probability density"))
        for m in cal_methods:
            col = f"pred_{m}"
            if col in g_F.columns:
                fig_F.add_trace(go.Histogram(x=g_F[col], name=f"pred {m}", opacity=0.45, nbinsx=40,
                                               marker_color=method_color_F.get(m, "#888"),
                                               histnorm="probability density"))
        fig_F.add_trace(go.Histogram(x=g_F["gt"], name="GT (holdout F)", opacity=0.30, nbinsx=40,
                                       marker_color="gray", histnorm="probability density"))
        if ref_x is not None:
            fig_F.add_trace(go.Scatter(x=ref_x, y=ref_density, mode="lines+markers",
                                         name="P_test PDF (réf)", line=dict(color="black", width=3, dash="dot"),
                                         marker=dict(size=8, symbol="diamond")))
        fig_F.update_layout(barmode="overlay", title=f"Female (n={len(g_F)})",
                              xaxis_title="Y", yaxis_title="density", height=400,
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_F, use_container_width=True)
    with c2:
        fig_M = go.Figure()
        fig_M.add_trace(go.Histogram(x=g_M["pred_raw"], name="pred RAW", opacity=0.45, nbinsx=40,
                                       marker_color=method_color_M["raw"], histnorm="probability density"))
        for m in cal_methods:
            col = f"pred_{m}"
            if col in g_M.columns:
                fig_M.add_trace(go.Histogram(x=g_M[col], name=f"pred {m}", opacity=0.45, nbinsx=40,
                                               marker_color=method_color_M.get(m, "#888"),
                                               histnorm="probability density"))
        fig_M.add_trace(go.Histogram(x=g_M["gt"], name="GT (holdout M)", opacity=0.30, nbinsx=40,
                                       marker_color="gray", histnorm="probability density"))
        if ref_x is not None:
            fig_M.add_trace(go.Scatter(x=ref_x, y=ref_density, mode="lines+markers",
                                         name="P_test PDF (réf)", line=dict(color="black", width=3, dash="dot"),
                                         marker=dict(size=8, symbol="diamond")))
        fig_M.update_layout(barmode="overlay", title=f"Male (n={len(g_M)})",
                              xaxis_title="Y", yaxis_title="density", height=400,
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_M, use_container_width=True)

    st.caption("Comment lire : si `pred ISO` (orange/cyan) suit mieux `P_test PDF` (noir pointillé) "
                "que `pred RAW` (rouge/bleu), l'isotonic a rapproché la distribution prédite de la "
                "référence test. `GT (holdout)` doit être cohérent avec `P_test PDF` (puisqu'on a "
                "stratifié le holdout via H_C) — sert de sanity check du holdout.")

    # === 3. Calibration plot: pred vs gt for RAW + each calibrator (small multiples) ===
    st.markdown("### 3. Calibration (pred vs gt), par méthode")
    st.caption("Points proches de y=x = bien calibré. Au-dessus = sur-estimation, en-dessous = sous-estimation. F=rouge/orange, M=bleu/cyan.")
    line_diag = go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", name="y=x",
                              line=dict(color="black", dash="dash"), showlegend=False)
    methods_for_calibration = ["raw"] + cal_methods
    cols_cal = st.columns(min(len(methods_for_calibration), 4))
    for i, m in enumerate(methods_for_calibration):
        col = "pred_raw" if m == "raw" else f"pred_{m}"
        if col not in df_pred.columns:
            continue
        with cols_cal[i % len(cols_cal)]:
            fig_c = go.Figure()
            fig_c.add_trace(line_diag)
            fig_c.add_trace(go.Scatter(x=g_F["gt"], y=g_F[col], mode="markers", name="F",
                                         marker=dict(color="#d62728", size=3, opacity=0.35)))
            fig_c.add_trace(go.Scatter(x=g_M["gt"], y=g_M[col], mode="markers", name="M",
                                         marker=dict(color="#1f77b4", size=3, opacity=0.35)))
            fig_c.update_layout(title=m.upper(), xaxis_title="gt", yaxis_title="pred", height=350)
            st.plotly_chart(fig_c, use_container_width=True)

    # === 4. MAE par bin Y, RAW vs chaque méthode ===
    st.markdown("### 4. Δ MAE par bin Y (méthode - raw). Négatif = amélioration.")
    n_bins = 15
    bin_w = 0.5 / n_bins
    df_pred["bin"] = (df_pred["gt"] / bin_w).clip(0, n_bins - 1).astype(int)
    df_pred["err_raw"] = (df_pred["pred_raw"] - df_pred["gt"]).abs()
    for m in cal_methods:
        col = f"pred_{m}"
        if col in df_pred.columns:
            df_pred[f"err_{m}"] = (df_pred[col] - df_pred["gt"]).abs()

    # Build long-format dataframe: for each method × bin × gender, compute Δ MAE
    rows = []
    for gi, gname in [(False, "M"), (True, "F")]:
        sub = df_pred[(df_pred["gender"] < 0.5) == gi]
        for b in range(n_bins):
            sb = sub[sub["bin"] == b]
            if len(sb) == 0:
                continue
            mae_raw = sb["err_raw"].mean()
            for m in cal_methods:
                if f"err_{m}" not in sb.columns:
                    continue
                mae_m = sb[f"err_{m}"].mean()
                rows.append({"bin_center": (b + 0.5) * bin_w, "gender": gname,
                              "method": m, "delta_mae": mae_m - mae_raw})
    if rows:
        df_long = pd.DataFrame(rows)
        fig_mae = px.bar(df_long, x="bin_center", y="delta_mae", color="method",
                           facet_row="gender", height=500,
                           labels={"delta_mae": "Δ MAE = MAE_cal - MAE_raw", "bin_center": "Y bin center"},
                           barmode="group", title="Δ MAE par méthode × bin × gender")
        fig_mae.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_mae, use_container_width=True)

    with st.expander("Raw data (preview)"):
        st.dataframe(df_pred.head(50), use_container_width=True)


if mode == "Post-processing effect (test holdout)":
    _render_isotonic_effect()
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
        s = r.get("challenge_score")
        return float(s) if isinstance(s, (int, float)) and s == s else float("inf")
    runs = sorted(runs, key=_score_key)

    sort_order = st.radio("Order", ["Best → worst", "Worst → best", "Most recent first"], index=0)
    if sort_order == "Worst → best":
        runs = list(reversed(runs))
    elif sort_order == "Most recent first":
        # _list_runs already returns DESC by start_time
        runs = _list_runs(TRACKING_URI, selected_exp_id)

    def _fmt(r: Dict[str, Any]) -> str:
        s = r["challenge_score"]
        score_str = f"{s:.5f}" if isinstance(s, (int, float)) else "n/a"
        return f"{r['name']}  ·  score={score_str}  ·  [{r['status']}]"

    selected_run = st.selectbox("Run", runs, format_func=_fmt)
    show_metrics = st.checkbox("Show metric panel", value=True)

# === Main panel ===

def _fmt_val(v: Any, decimals: int = 5, suffix: str = "") -> str:
    return f"{v:.{decimals}f}{suffix}" if isinstance(v, (int, float)) else "n/a"


if show_metrics:
    st.markdown("### Score total")
    st.caption("`Score = (err_F + err_M)/2 + |err_F - err_M|` (formule officielle). v6 (B'): val matche P_test → `Score` est directement l'estimateur de la perf test. `Score raw` = même formule sans aucune correction (diagnostic, identique au principal en v6).")
    cols = st.columns(2)
    cols[0].metric("Score",                    _fmt_val(selected_run["challenge_score"]),
                   help="LA cible Optuna. Plus c'est bas, mieux c'est.")
    cols[1].metric("Score raw (sans reweight)", _fmt_val(selected_run["challenge_score_raw"]),
                   help="v6 (B'): identique au Score principal car val matche déjà P_test. v4/v5: ancien `_val` (avant correction).")

    # Decompose score = mean_err + |err_diff|  (because (err_F+err_M)/2 + |err_F-err_M| = mean + diff)
    err_f = selected_run["err_F"]
    err_m = selected_run["err_M"]
    mean_err = (err_f + err_m) / 2 if isinstance(err_f, (int, float)) and isinstance(err_m, (int, float)) else None
    st.markdown("### Composantes du score")
    cols = st.columns(4)
    cols[0].metric("mean_err",       _fmt_val(mean_err),
                   help="(err_F + err_M) / 2 — performance moyenne sur les 2 genres")
    cols[1].metric("err_diff",       _fmt_val(selected_run["err_diff"]),
                   help="|err_F - err_M| — pénalité de fairness genre")
    cols[2].metric("err_F",          _fmt_val(err_f),
                   help="Σwᵢ(pᵢ-yᵢ)² / Σwᵢ sur les samples Female (wᵢ=1/30+yᵢ)")
    cols[3].metric("err_M",          _fmt_val(err_m),
                   help="Σwᵢ(pᵢ-yᵢ)² / Σwᵢ sur les samples Male")

    st.markdown("### Métriques humaines")
    cols = st.columns(2)
    cols[0].metric("MAE",            _fmt_val(selected_run["mae_pct"], decimals=2, suffix=" %"),
                   help="Erreur absolue moyenne en points de % d'occlusion : le modèle se trompe en moyenne de X points")
    cols[1].metric("R²",             _fmt_val(selected_run["r2"], decimals=3),
                   help="0 = modèle trivial (moyenne constante), 1 = parfait")

qual_dir = _download_qualitative(TRACKING_URI, selected_run["run_id"])
if qual_dir is None:
    st.info("No qualitative/ artifacts logged for this run "
            "(needs `save_qualitative_k > 0` and a non-skipped trial).")
    st.stop()

qual_path = Path(qual_dir)
worst_df = _load_csv(qual_path / "worst", "worst")
best_df = _load_csv(qual_path / "best", "best")

tab_diag, tab_params, tab_worst, tab_best, tab_csv = st.tabs([
    "📊 Diagnostic charts",
    "⚙️ Training params",
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

with tab_params:
    params = _fetch_run_params(TRACKING_URI, selected_run["run_id"])
    if not params:
        st.info("No params logged for this run.")
    else:
        # Highlight panel : v10 search-space params ordonnés par lisibilité.
        OPTUNA_KEYS = [
            # === v10 unified rebalancing target ===
            "axis1_power", "axis2_power",
            # === Feature fairness ===
            "feature_fairness", "mmd_lambda",
            # === Loss ===
            "loss_focal_gamma",
            # === Architecture ===
            "pretrained_source", "pooling_type",
            # === Hyperparams ===
            "learning_rate", "weight_decay",
            "head_dropout", "backbone_drop_path_rate",
            "pool_attn_dropout", "pool_proj_dropout",
            "tau_focal_init", "tau_diffuse_init",
            "n_focal", "n_diffuse", "n_free", "num_heads",
            "loss_query_diversity_lambda",
        ]
        # v9 : afficher TOUS les OPTUNA_KEYS (même les manquants → "—") pour qu'on voie
        # explicitement les params absents (legacy v6-v8, conditionnels non samplés, etc.).
        highlight = [(k, params.get(k, "—")) for k in OPTUNA_KEYS]
        st.markdown("### Optuna search-space (params samplés)")
        st.caption("'—' = param absent de ce run (legacy v6/v7, ou conditionnel non samplé)")
        st.dataframe(
            pd.DataFrame(highlight, columns=["param", "value"]),
            use_container_width=True, hide_index=True,
        )

        # Full dump — toutes les params (train_*, pretrain_*, model_*, data_*, etc.)
        # avec filtre texte pour naviguer.
        st.markdown("### All params (full MLflow dump)")
        query = st.text_input(
            "Filter params (substring match on key or value)",
            value="",
            placeholder="e.g. train_, pretrain_lr, sapiens, ...",
        )
        items = sorted(params.items())
        if query:
            q = query.lower()
            items = [(k, v) for k, v in items if q in k.lower() or q in str(v).lower()]
        st.caption(f"{len(items)} / {len(params)} params shown")
        st.dataframe(
            pd.DataFrame(items, columns=["param", "value"]),
            use_container_width=True, hide_index=True, height=600,
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
