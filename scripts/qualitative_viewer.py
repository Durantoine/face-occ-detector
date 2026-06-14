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
import re
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
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Face-occ analytics", layout="wide")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Params we show in hover tooltips on the trials comparison chart. Order matters
# (top to bottom in the tooltip). DON'T filter df construction — keep this short
# just for tooltip readability. df underlying uses ALL params via r["params"].
HOVER_PARAMS = [
    "correction_strength",
    "feature_fairness",
    "ot_lambda", "ot_method", "sinkhorn_eps",
    "adv_lambda",
    "loss_focal_gamma",
    "loss_lambda_premultiplier",   # v20
    "pretrained_source",
    "pooling_type",
    "grid_size", "mil_k_top",
    "learning_rate",
    "weight_decay",
    "layer_decay",
    "min_lr_rate",
]

# v20: conditional dependencies (mirror of yaml conditional_on). When the parent
# condition isn't met, the param is irrelevant (Optuna still logged a default).
# Used to mask inactive conditionals in df → Per-axis breakdown ignores those rows
# (e.g. mil_k_top affiché seulement quand pooling_type=mil).
CONDITIONAL_PARAMS: Dict[str, Dict[str, Any]] = {
    "ot_lambda":                   {"feature_fairness": "ot"},
    "ot_method":                   {"feature_fairness": "ot"},
    "sinkhorn_eps":                {"feature_fairness": "ot", "ot_method": "sinkhorn"},
    "adv_lambda":                  {"feature_fairness": "dann"},
    "pool_attn_dropout":           {"pooling_type": "attention_k_query"},
    "pool_proj_dropout":           {"pooling_type": "attention_k_query"},
    "loss_query_diversity_lambda": {"pooling_type": "attention_k_query"},
    "tau_focal_init":              {"pooling_type": "attention_k_query"},
    "tau_diffuse_init":            {"pooling_type": "attention_k_query"},
    "n_focal":                     {"pooling_type": "attention_k_query"},
    "n_diffuse":                   {"pooling_type": "attention_k_query"},
    "n_free":                      {"pooling_type": "attention_k_query"},
    "pool_proj_out_dim":           {"pooling_type": "attention_k_query"},
    "mil_hidden":                  {"pooling_type": "mil"},
    "mil_k_top":                   {"pooling_type": "mil"},
    "grid_size":                   {"pooling_type": "grid"},
    "grid_cell_proj_dim":          {"pooling_type": "grid"},
    "head_type":                   {"pooling_type": "grid"},
    "head_hidden_dim":             {"pooling_type": "grid", "head_type": "mlp"},
}


def _params_with_conditional_mask(run_params: Dict[str, str]) -> Dict[str, Any]:
    """Return ALL run params with conditional ones nulled-out if their parent
    condition isn't satisfied. Optuna logs default values even for inactive
    conditionals → without masking, Per-axis breakdown shows e.g. mil_k_top stats
    on trials that don't use mil pooling."""
    out: Dict[str, Any] = dict(run_params)
    for p, conds in CONDITIONAL_PARAMS.items():
        if p not in out:
            continue
        active = True
        for parent, required in conds.items():
            actual = run_params.get(parent)
            if actual is None or str(actual) != str(required):
                active = False
                break
        if not active:
            out[p] = None
    return out


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
            # `best_score` est loggé par optimize.py post-train = score du best checkpoint
            # (load_best_model_at_end). C'est la valeur que voit Optuna et qui détermine si
            # on sauve un modèle (min_score_to_save). Fallback eval_challenge_score = dernier
            # epoch eval (peut être pire que le best si overfit en fin d'entraînement).
            "challenge_score":     pick("best_score", "eval_challenge_score"),
            "challenge_score_raw": pick("eval_challenge_score_raw"),
            "err_F":               pick("best_err_F", "err_F", "eval_err_F"),
            "err_M":               pick("best_err_M", "err_M", "eval_err_M"),
            "err_diff":            pick("best_err_diff", "err_diff", "eval_err_diff"),
            "mae_pct":             pick("best_mae_pct", "eval_mae_pct"),
            "r2":                  pick("best_r2", "eval_r2"),
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


def _download_test_predictions(tracking_uri: str, run_id: str) -> Optional[pd.DataFrame]:
    """Load the `test_predictions` artifact (real best-epoch predictions on the test set),
    computed once at the end of the trial. Returns the DataFrame, or None if absent."""
    client = _get_client(tracking_uri)
    try:
        if not client.list_artifacts(run_id, "test_predictions"):
            return None
    except Exception:
        return None
    tmp_root = Path(tempfile.gettempdir()) / "face_occ_testpred_cache" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        local = client.download_artifacts(run_id, "test_predictions", dst_path=str(tmp_root))
        csvs = list(Path(local).glob("*.csv"))
        return pd.read_csv(csvs[0]) if csvs else None
    except Exception:
        return None


def _download_val_predictions(tracking_uri: str, run_id: str) -> Optional[pd.DataFrame]:
    """Load the `val_predictions` artifact (best-epoch val preds: gt + pred + gender),
    used to fit the supervised calibrators (isotonic). None if absent (pre-feature run)."""
    client = _get_client(tracking_uri)
    try:
        if not client.list_artifacts(run_id, "val_predictions"):
            return None
    except Exception:
        return None
    tmp_root = Path(tempfile.gettempdir()) / "face_occ_valpred_cache" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        local = client.download_artifacts(run_id, "val_predictions", dst_path=str(tmp_root))
        csvs = list(Path(local).glob("*.csv"))
        return pd.read_csv(csvs[0]) if csvs else None
    except Exception:
        return None


def _isotonic_fit(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
    """Weighted isotonic regression via PAVA (numpy-only). Returns a monotone lookup (ux, uy)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    order = np.argsort(x, kind="mergesort")
    xo, yo, wo = x[order], y[order], w[order]
    val, wt, cnt = [], [], []
    for yi, wi in zip(yo, wo):
        val.append(yi); wt.append(wi); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, w2, c2 = val.pop(), wt.pop(), cnt.pop()
            v1, w1, c1 = val.pop(), wt.pop(), cnt.pop()
            val.append((v1 * w1 + v2 * w2) / (w1 + w2)); wt.append(w1 + w2); cnt.append(c1 + c2)
    fitted = np.empty(len(yo)); i = 0
    for v, c in zip(val, cnt):
        fitted[i:i + c] = v; i += c
    ux, first = np.unique(xo, return_index=True)
    last = np.append(first[1:] - 1, len(xo) - 1)
    return ux, np.clip(fitted[last], 0.0, 1.0)


def _isotonic_apply(fit, q: np.ndarray) -> np.ndarray:
    ux, uy = fit
    if len(ux) < 2:
        return np.clip(np.asarray(q, float), 0.0, 1.0)
    return np.clip(np.interp(np.asarray(q, float), ux, uy), 0.0, 1.0)


def _quantile_map_to_ptest(p: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """Map predictions onto a target PMF by rank (1D optimal transport, numpy-only)."""
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return p
    pmf = np.clip(np.asarray(pmf, float), 0.0, None); pmf = pmf / pmf.sum()
    edges = np.linspace(0.0, 0.5, len(pmf) + 1)
    cdf = np.concatenate([[0.0], np.cumsum(pmf)])
    cdf, keep = np.unique(cdf, return_index=True)
    order = np.argsort(p, kind="mergesort")
    u = np.empty(n); u[order] = (np.arange(n) + 0.5) / n
    return np.clip(np.interp(u, cdf, edges[keep]), 0.0, 1.0)


# Official P_test PMF (100 bins over [0,0.5]) — inlined so the viewer stays import-free (no torch/scipy).
_PTEST_PMF_05 = np.array([
    0.023880, 0.021935, 0.018046, 0.018046, 0.014993, 0.013726, 0.011194, 0.011194, 0.008277, 0.011940,
    0.019267, 0.019267, 0.016282, 0.015671, 0.014450, 0.014450, 0.013704, 0.013161, 0.012618, 0.012618,
    0.017096, 0.016926, 0.016757, 0.016757, 0.015332, 0.014823, 0.014314, 0.014314, 0.012822, 0.015468,
    0.018113, 0.018113, 0.016757, 0.015909, 0.015061, 0.014608, 0.013704, 0.013229, 0.012754, 0.014269,
    0.017299, 0.016485, 0.015671, 0.015694, 0.015739, 0.015264, 0.014789, 0.014156, 0.012890, 0.013839,
    0.014789, 0.013941, 0.013093, 0.012890, 0.012686, 0.012449, 0.012211, 0.011601, 0.010990, 0.010651,
    0.010312, 0.009362, 0.008412, 0.007734, 0.007055, 0.006682, 0.006309, 0.006038, 0.005766, 0.005178,
    0.004885, 0.004613, 0.004342, 0.003347, 0.002849, 0.002578, 0.002307, 0.002035, 0.001900, 0.001492,
    0.001085, 0.000950, 0.000882, 0.000814, 0.000746, 0.000339, 0.000339, 0.000373, 0.000407, 0.000339,
    0.000339, 0.000237, 0.000136, 0.000136, 0.000136, 0.000170, 0.000204, 0.000068, 0.000068, 0.000068,
], dtype=np.float64)


def _continuous_density(vals: np.ndarray, bw: float = 0.02, nb: int = 200):
    """Light Gaussian-KDE-like continuous density (fine histogram + gaussian smoothing), numpy-only —
    same continuous philosophy as the training pipeline, but with no torch/sklearn/scipy dependency."""
    grid = (np.arange(nb) + 0.5) / nb
    hist, _ = np.histogram(np.clip(np.asarray(vals, float), 0.0, 1.0), bins=nb, range=(0.0, 1.0), density=True)
    dx = 1.0 / nb
    half = max(1, int(4 * bw / dx))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / bw) ** 2)
    k /= k.sum()
    return grid, np.convolve(hist, k, mode="same")


def _ptest_density(grid: np.ndarray) -> np.ndarray:
    """Continuous P_test target density on `grid` (the official PMF, 0 above its [0,0.5] support)."""
    centers = (np.arange(100) + 0.5) * 0.005
    pmf = _PTEST_PMF_05 / _PTEST_PMF_05.sum()
    d = np.interp(grid, centers, pmf * 200.0, left=0.0, right=0.0)  # PMF × N_BINS → density
    d[grid > 0.5] = 0.0
    return d


@st.cache_data(show_spinner=False)
def _ptrain_density_cached():
    """Continuous P_train target density from the train labels (read once). Used to draw the
    intermediate target P_target=(1-a)*P_train + a*P_test that v38 actually aims at."""
    try:
        y = pd.read_csv("data/raw/train.csv")["FaceOcclusion"].values
    except Exception:
        return None, None
    return _continuous_density(np.asarray(y, dtype=float), bw=0.02)


def _w1(grid: np.ndarray, da: np.ndarray, db: np.ndarray) -> float:
    """Wasserstein-1 (Earth Mover's Distance) between two densities on a uniform grid, in occlusion
    units = integral of |CDF_a - CDF_b|. 0 = identical; robust where KL would blow up (P_test=0 >0.5)."""
    dx = float(grid[1] - grid[0])
    ca, cb = np.cumsum(da) * dx, np.cumsum(db) * dx
    ca, cb = ca / max(ca[-1], 1e-12), cb / max(cb[-1], 1e-12)
    return float(np.sum(np.abs(ca - cb)) * dx)


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
    by_rank: Dict[int, List[Tuple[Optional[float], Path]]] = {}
    for p in sorted(imgs):
        try:
            rank = int(p.stem.split("_", 1)[0])  # p.stem strips .png -> "000" parses (and "5_gt0.39" too)
        except (ValueError, IndexError):
            continue
        m = re.search(r"gt([0-9.]+)", p.name)
        gt_in_name = float(m.group(1)) if m else None
        by_rank.setdefault(rank, []).append((gt_in_name, p))

    def _pick(rank: int, gt: float) -> Optional[Path]:
        cands = by_rank.get(rank)
        if not cands:
            return None
        typed = [(g, p) for g, p in cands if g is not None]
        if typed:
            return min(typed, key=lambda gp: abs(gp[0] - gt))[1]
        return cands[-1][1]

    for chunk_start in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            i = chunk_start + j
            if i >= len(df):
                break
            row = df.iloc[i]
            with cols[j]:
                rank = int(row["rank"])
                img_path = _pick(rank, float(row["gt"]))
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
                fn = row.get("filename") if "filename" in df.columns else None
                if isinstance(fn, str) and fn:
                    parts = fn.rsplit("/", 2)
                    ref = "/".join(parts[-2:]).replace("_align.webp", "")
                    st.caption(f"`{ref}`")


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
    # v34: include all eval_* and train_* monitoring metrics
    keys = {k for k in keys if k.startswith("eval_") or k.startswith("train_") or k == "loss"}
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

    import re
    _CURRENT_VERSION_RE = re.compile(r"-v34$|scout-v34$")
    default_exps = [(eid, name) for eid, name in experiments
                    if _CURRENT_VERSION_RE.search(name)]
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
        # `best_score` (post-train) = score du best checkpoint (load_best_model_at_end),
        # cohérent avec la valeur Optuna et avec ce qui déclenche min_score_to_save.
        # Fallback eval_challenge_score = dernier epoch (peut être pire si overfit).
        if "best_score" in available_metrics:
            default_metric = "best_score"
        elif "eval_challenge_score" in available_metrics:
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
    """Return the final (last logged) value of `metric` for the given run."""
    hist = _fetch_metric_history(tracking_uri, run["run_id"], metric)
    if hist:
        return float(hist[-1][1])
    v = run.get(metric)
    return float(v) if isinstance(v, (int, float)) else None


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
            # v8 : err_diff pour le sort fairness. v35: best_err_diff (best epoch) en priorité.
            err_diff_val = None
            for k in ("best_err_diff", "eval_err_diff"):
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
                **_params_with_conditional_mask(r["params"]),
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

        # v20: ALL HPO params from yaml search_space (covers scout, 3090, A100 variants)
        focus_cols = [
            "experiment", "trial_idx", "trial", "final_value", "err_diff",
            "correction_strength",
            "feature_fairness", "ot_lambda", "ot_method", "sinkhorn_eps", "adv_lambda",
            "loss_focal_gamma", "loss_lambda_premultiplier",
            "pretrained_source", "pooling_type",
            "learning_rate", "min_lr_rate", "weight_decay", "layer_decay",
            "head_dropout", "backbone_drop_path_rate",
            # pooling-conditional knobs
            "pool_attn_dropout", "pool_proj_dropout", "loss_query_diversity_lambda",
            "tau_focal_init", "tau_diffuse_init",
            "n_focal", "n_diffuse", "n_free",
            "mil_hidden", "mil_k_top",
            "grid_size", "grid_cell_proj_dim",
            "pool_proj_out_dim",
            "head_type", "head_hidden_dim",
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
        # ou par bin (continus). v20: AUTO-DISCOVER tous les params loggés.
        # Hint lists = forcing categorical (parsable comme numeric mais sémantiquement cat,
        # ex: grid_size=3,4,5,...). Le reste est auto-classifié en numeric/cat via dtype.
        FORCE_CATEGORICAL = {
            "pretrained_source", "pooling_type", "feature_fairness", "ot_method",
            "mil_hidden", "grid_size", "pool_proj_out_dim", "grid_cell_proj_dim",
            "head_type", "head_hidden_dim",
        }
        # Meta cols à exclure (pas des HPO knobs)
        META_COLS = {
            "experiment", "trial_idx", "trial", "final_value", "err_diff",
            "_family", "run_id", "run_name", "status", "start_time", "end_time",
            "exp_name", "name", "score", "best_score", "eval_challenge_score",
            "test_holdout_score_raw", "test_holdout_score_selected_cal",
            "val_score_best_combo_is_eval", "best_alpha_for_submission_value",
            "best_cal_for_submission", "best_alpha_for_submission",
        }
        # v20: order matters — important axes first (fairness then learning then conditional)
        PRIORITY_ORDER = [
            # Fairness & IS-reweight (cœur v20)
            "correction_strength", "feature_fairness",
            "loss_lambda_premultiplier", "loss_focal_gamma",
            "ot_lambda", "adv_lambda", "ot_method", "sinkhorn_eps",
            # Architecture
            "pretrained_source", "pooling_type",
            # Learning dynamics
            "learning_rate", "min_lr_rate", "weight_decay", "layer_decay",
            "head_dropout", "backbone_drop_path_rate",
            # Pooling-conditional
            "loss_query_diversity_lambda",
            "pool_attn_dropout", "pool_proj_dropout",
            "tau_focal_init", "tau_diffuse_init",
            "n_focal", "n_diffuse", "n_free",
            "mil_hidden", "mil_k_top",
            "grid_size", "grid_cell_proj_dim",
            "pool_proj_out_dim",
            "head_type", "head_hidden_dim",
        ]
        # Auto-discover axes from df columns (post family-filter)
        def _split_axes(fam_df_local: pd.DataFrame) -> Tuple[List[str], List[str]]:
            cat, cont = [], []
            for col in fam_df_local.columns:
                if col in META_COLS:
                    continue
                ser = fam_df_local[col].dropna()
                if ser.empty:
                    continue
                if col in FORCE_CATEGORICAL:
                    cat.append(col)
                    continue
                # Tentative numeric parse — si tout convertit, c'est continu
                num = pd.to_numeric(ser, errors="coerce")
                n_nan = num.isna().sum()
                if n_nan / len(ser) > 0.2:   # > 20% pas convertibles → catégoriel
                    cat.append(col)
                else:
                    cont.append(col)
            # Order: priority first (in PRIORITY_ORDER sequence), then leftover alphabetical
            def _ordered(axes: List[str]) -> List[str]:
                axes_set = set(axes)
                head = [a for a in PRIORITY_ORDER if a in axes_set]
                tail = sorted([a for a in axes if a not in head])
                return head + tail
            return _ordered(cat), _ordered(cont)
        AXIS_CATEGORICAL, AXIS_CONTINUOUS = _split_axes(df)
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
            # Sort: maintain PRIORITY_ORDER for axes (cat then cont), best ASC within each axis
            axis_order = {a: i for i, a in enumerate(AXIS_CATEGORICAL + AXIS_CONTINUOUS)}
            ax_df = pd.DataFrame(rows)
            ax_df["_order"] = ax_df["axis"].map(lambda a: axis_order.get(a, 999))
            ax_df = ax_df.sort_values(["_order", "best"]).drop(columns=["_order"]).reset_index(drop=True)
            for col in ("best", "avg", "med"):
                ax_df[col] = ax_df[col].round(5)
            st.dataframe(ax_df, use_container_width=True, hide_index=True)

            # v20: Optuna exploration coverage per axis. Sert à voir si Optuna a déjà
            # exploré largement (high n_distinct, range proche du search_space yaml) ou
            # s'il s'est concentré sur une zone (low n_distinct, range étroite).
            # Indicateurs :
            #   - n_distinct : # valeurs uniques explorées (cat) / # bins (continu)
            #   - explored_range : (min, max) observés
            #   - coverage_pct : (max-min)/search_range × 100 (continu seulement, NaN si inconnu)
            #   - n_trials : nombre de trials ayant cet axe (post-conditional)
            st.markdown(f"#### Exploration coverage — {fam_label}")
            cov_rows: List[Dict[str, Any]] = []
            all_axes = list(AXIS_CATEGORICAL) + list(AXIS_CONTINUOUS)
            for axis in all_axes:
                if axis not in fam_df.columns:
                    continue
                col = fam_df[axis].dropna()
                if col.empty:
                    continue
                is_cat = axis in AXIS_CATEGORICAL
                if is_cat:
                    distinct = sorted(set(str(v) for v in col.unique()))
                    cov_rows.append({
                        "axis": axis, "kind": "cat", "n_trials": int(len(col)),
                        "n_distinct": len(distinct),
                        "values": ", ".join(distinct[:6]) + ("…" if len(distinct) > 6 else ""),
                        "range_min": None, "range_max": None, "coverage_pct": None,
                    })
                else:
                    num = pd.to_numeric(col, errors="coerce").dropna()
                    if num.empty:
                        continue
                    vmin, vmax = float(num.min()), float(num.max())
                    cov_rows.append({
                        "axis": axis, "kind": "cont", "n_trials": int(len(num)),
                        "n_distinct": int(num.nunique()),
                        "values": f"std={num.std():.4f}",
                        "range_min": round(vmin, 5), "range_max": round(vmax, 5),
                        "coverage_pct": None,  # search_space yaml range parsing skipped (ambigu cross-yaml)
                    })
            if cov_rows:
                cov_df = pd.DataFrame(cov_rows).sort_values(["kind", "axis"]).reset_index(drop=True)
                st.dataframe(cov_df, use_container_width=True, hide_index=True)
                st.caption(
                    f"Total trials in family: {len(fam_df)}. "
                    f"n_distinct faible + n_trials élevé = Optuna a convergé sur une zone (= exploration saturée). "
                    f"n_distinct élevé = exploration encore active."
                )

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
                **_params_with_conditional_mask(r["params"]),
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
        ["Trials comparison (live)", "Qualitative viewer (per-run)", "Post-processing effect (test holdout)",
         "v35 data cleaning (proxy)"],
        index=0,
        key="mode_selector",
    )
    st.divider()

if mode == "Trials comparison (live)":
    _render_trials_comparison()
    st.stop()


def _apply_dark_theme(fig):
    """Force dark template + transparent bg + light text — fix le noir-sur-noir Streamlit."""
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e5e7eb"))
    fig.update_xaxes(color="#e5e7eb", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(color="#e5e7eb", gridcolor="rgba(255,255,255,0.05)")
    return fig


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
    default_idx = 0
    for i, label in enumerate(exp_labels):
        if "v22" in label.lower():
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

    # === Metrics summary : val IS-rew (estimator P_test) | oracle (direct P_test sample) ===
    st.markdown("### Métriques : val IS-reweighted vs oracle test holdout")
    st.caption(
        "**Val IS-rew** = `compute_score_stratified_is` sur val (15k iid P_train) avec poids IS pour simuler P_test "
        "— c'est l'estimateur sans biais que voit Optuna. "
        "**Oracle** = score direct sur 5k test holdout sampled to match P_test(Y). "
        "Les 2 estiment la même perf P_test, avec variance et biais différents (val plus haute var via IS, oracle plus basse var mais N=5k)."
    )

    # Sources MLflow
    val_raw  = _final_metric_value(TRACKING_URI, selected_run, "best_score")
    orc_raw  = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_score_raw")
    orc_sel  = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_score_selected_cal")
    orc_orcl = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_score_oracle_cal")

    def _delta_str(v: Optional[float], ref: Optional[float]) -> Optional[str]:
        if v is None or ref is None:
            return None
        d = v - ref
        return f"{d:+.5f} ({d / max(abs(ref), 1e-9) * 100:+.1f}%)"

    # Build rows: raw + 1 row per cal method
    rows = [("raw (no cal)", val_raw, orc_raw, None)]   # None = no delta vs itself
    for m in cal_methods:
        v_m = _final_metric_value(TRACKING_URI, selected_run, f"val_score_{m}_at_best_alpha_is_eval")
        o_m = _final_metric_value(TRACKING_URI, selected_run, f"test_holdout_score_{m}")
        rows.append((m, v_m, o_m, _final_metric_value(TRACKING_URI, selected_run, f"best_alpha_{m}")))

    # Sort by oracle score (best calibrator first after raw)
    rows_sorted = [rows[0]] + sorted(rows[1:], key=lambda r: (r[2] if r[2] is not None else float("inf")))

    # Render table 2 cols (val | oracle), highlight SELECTED cal
    selected_cal_param = (selected_run.get("params") or {}).get("best_cal_for_submission") if isinstance(selected_run.get("params"), dict) else None
    # Fallback: get from a params fetch helper
    if selected_cal_param is None:
        _p = _fetch_run_params(TRACKING_URI, selected_run["run_id"])
        selected_cal_param = _p.get("best_cal_for_submission")

    orc_diff_raw = _final_metric_value(TRACKING_URI, selected_run, "test_holdout_err_diff_raw")
    import pandas as _pd
    df_rows = []
    for name, vv, ov, alpha in rows_sorted:
        is_sel = name == selected_cal_param
        diff_key = "test_holdout_err_diff_raw" if name == "raw (no cal)" else f"test_holdout_err_diff_{name}"
        od = _final_metric_value(TRACKING_URI, selected_run, diff_key)
        df_rows.append({
            "calibrator": ("⭐ " if is_sel else "") + name,
            "best α": f"{alpha:.2f}" if isinstance(alpha, (int, float)) else ("—" if name == "raw (no cal)" else "?"),
            "val IS-rew score": f"{vv:.5f}" if isinstance(vv, (int, float)) else "n/a",
            "Δ val vs raw": _delta_str(vv, val_raw) or "—",
            "oracle score": f"{ov:.5f}" if isinstance(ov, (int, float)) else "n/a",
            "Δ oracle vs raw": _delta_str(ov, orc_raw) or "—",
            "oracle err_diff": f"{od:.5f}" if isinstance(od, (int, float)) else "n/a",
            "Δ err_diff vs raw": _delta_str(od, orc_diff_raw) or "—",
        })
    st.dataframe(_pd.DataFrame(df_rows), hide_index=True, use_container_width=True)
    st.caption(
        "`oracle err_diff` = |err_F − err_M| sur le holdout. Une calibration peut baisser le score "
        "mais **creuser** le gap genre (err_diff ↑) — à surveiller. v21 : la calibration n'est appliquée "
        "que si elle améliore le **score oracle** (sinon raw)."
    )

    # Quick summary metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Val IS-rew (raw)", f"{val_raw:.5f}" if val_raw is not None else "n/a",
              help="= best_score MLflow. Estimateur sans biais de P_test perf (Optuna target).")
    c2.metric("Oracle (raw)", f"{orc_raw:.5f}" if orc_raw is not None else "n/a",
              help="= test_holdout_score_raw. Score direct sur 5k échantillonnés selon P_test(Y).")
    if orc_sel is not None:
        c3.metric("Oracle (selected cal ⭐)", f"{orc_sel:.5f}",
                  delta=_delta_str(orc_sel, orc_raw), delta_color="inverse",
                  help="Oracle après application du cal sélectionné via val IS-strat — ce qu'on submit.")
    elif orc_orcl is not None:
        c3.metric("Oracle (oracle cal, biased)", f"{orc_orcl:.5f}",
                  delta=_delta_str(orc_orcl, orc_raw), delta_color="inverse",
                  help="Upper bound biaisé : best cal post-hoc directement sur test holdout.")
    st.caption("Delta négatif (vert) = amélioration. ⭐ = calibrator sélectionné pour la submission.")

    try:
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.error("numpy + plotly required for these plots — pip install numpy plotly")
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
                                                 name=m, line=dict(color=method_colors.get(m, "#9ca3af"), width=3)))
            fig_F.update_layout(title="Female", xaxis_title="pred_raw", yaxis_title="pred_corrigé",
                                  height=400, hovermode="x")
            _apply_dark_theme(fig_F)
            st.plotly_chart(fig_F, use_container_width=True)
        with c2:
            fig_M = go.Figure()
            fig_M.add_trace(go.Scatter(x=df_cal["x"], y=df_cal["x"], mode="lines",
                                         name="y=x", line=dict(color="gray", dash="dash")))
            for m in cal_methods:
                col = f"{m}_M"
                if col in df_cal.columns:
                    fig_M.add_trace(go.Scatter(x=df_cal["x"], y=df_cal[col], mode="lines",
                                                 name=m, line=dict(color=method_colors.get(m, "#9ca3af"), width=3)))
            fig_M.update_layout(title="Male", xaxis_title="pred_raw", yaxis_title="pred_corrigé",
                                  height=400, hovermode="x")
            _apply_dark_theme(fig_M)
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
        _apply_dark_theme(fig_iso)
        st.plotly_chart(fig_iso, use_container_width=True)

    # === 2. Déformation des distributions par calibrator (small multiples) ===
    st.markdown("### 2. Alignement pred → P_test par calibrator (au best α de chacun)")
    st.caption(
        "Barres = density du blend `α·cal + (1-α)·raw` à son α optimal (val IS-strat). "
        "**Ambre** = oracle GT (= test_holdout sampled pour matcher P_test via H_C, n=5k). "
        "**Vert clair pointillé** = P_test exact extrait du PDF (référence cible). "
        "Bonne calibration ⇒ barres qui s'alignent sur l'ambre (= proche du vert pointillé). "
        "L'écart ambre ↔ vert pointillé = bruit d'échantillonnage du test_holdout 5k vs PDF n=29 980."
    )

    # v20 fix: cible per-gender via H_C, pas la marginale globale.
    #   ANCIEN BUG: on affichait _TEST_PMF_FINE (P_test(Y) marginal aggregated sur G)
    #   pour les 2 genres → fausse cible (e.g. y=0 affichait density 4.78 pour F alors
    #   que la vraie P_test(Y=0|F) = 0.93 car femmes train rares à y=0).
    #   FIX: compute P_test(Y|G=g) = P_train(G=g|Y) · P_test(Y) / P_test(G=g) via
    #   estimate_test_pmf_joint sur le train.csv complet (ou val si pas dispo).
    ref_x = ref_density_F = ref_density_M = None
    _NB = 100
    try:
        from src.utils.distribution import (
            _TEST_PMF_FINE, N_BINS_FINE as _NB_FINE, BIN_WIDTH_FINE as _BW_FINE,
            estimate_test_pmf_joint, _resolve_edges,
        )
        _NB = _NB_FINE
        ref_x = np.array([(b + 0.5) * _BW_FINE for b in range(_NB_FINE)])
        # Per-gender cible : use train data to estimate P_train(G|Y) (cf H_C derivation)
        try:
            _train_df = pd.read_csv("data/raw/train.csv")
            _train_y = _train_df["FaceOcclusion"].astype(float).values
            _train_g = (_train_df["gender"].astype(float).values >= 0.5).astype(float)
            _edges = _resolve_edges(n_bins=_NB_FINE, bin_width=_BW_FINE)
            _joint = estimate_test_pmf_joint(_train_y, _train_g,
                                              bin_edges=_edges, test_pmf_y=_TEST_PMF_FINE)
            # P_test(Y|G=g) = joint[g,:] / sum_y joint[g,:]
            _p_f = _joint[0, :] / max(_joint[0, :].sum(), 1e-12)
            _p_m = _joint[1, :] / max(_joint[1, :].sum(), 1e-12)
            ref_density_F = _p_f / _BW_FINE
            ref_density_M = _p_m / _BW_FINE
        except Exception as _e_jt:
            # Fallback : marginale globale (l'ancien comportement, mais on warn)
            st.warning(f"P_test per-gender cible : fallback marginale globale ({_e_jt}). "
                        "data/raw/train.csv introuvable.")
            ref_density_F = ref_density_M = np.asarray(_TEST_PMF_FINE) / _BW_FINE
    except ImportError:
        ref_x = ref_density_F = ref_density_M = None

    # Fetch best alpha per cal from MLflow metrics
    best_alpha = {}
    for m in cal_methods:
        a = _final_metric_value(TRACKING_URI, selected_run, f"best_alpha_{m}")
        if a is not None:
            best_alpha[m] = float(a)
    # raw is always α=0 (no correction)
    best_alpha["raw"] = 0.0

    g_F = df_pred[df_pred["gender"] < 0.5]
    g_M = df_pred[df_pred["gender"] >= 0.5]
    methods_show = ["raw"] + cal_methods
    # Neon-style palette (vibrant on dark background)
    bar_color = {
        "raw": "#60a5fa",              # Bright Blue
        "isotonic": "#f87171",         # Bright Red
        "isotonic_continuous": "#fb923c", # Orange
        "isotonic_regime": "#fb7185",  # Rose
        "isotonic_tailboost": "#c084fc", # Purple
        "linear": "#4ade80",           # Bright Green
        "pchip": "#2dd4bf",            # Teal
    }
    gt_color = "#fbbf24"        # Amber — oracle GT (test_holdout = P_test sampled, n=5k)
    target_color = "#86efac"    # Bright green — P_test exact (PDF extraction, n=29980)

    def _density(values, bins):
        h, _ = np.histogram(values, bins=bins)
        w = bins[1] - bins[0]
        s = h.sum()
        return (h / (s * w)) if s > 0 else h.astype(float)

    def _smooth_density(values, x_grid, bandwidth=0.012):
        """KDE-lite gaussian smoothing → ligne lissée pour la GT (au lieu d'un histogramme blocky)."""
        v = np.asarray(values, dtype=np.float64)
        v = v[(v >= 0.0) & (v <= 0.5)]
        if len(v) == 0:
            return np.zeros_like(x_grid)
        # Vectorized gaussian KDE
        diff = (x_grid[:, None] - v[None, :]) / bandwidth
        k = np.exp(-0.5 * diff ** 2) / (bandwidth * np.sqrt(2 * np.pi))
        return k.mean(axis=1)

    bin_edges = np.linspace(0.0, 0.5, _NB + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_w = bin_edges[1] - bin_edges[0]
    # Finer grid for the smoothed GT line (visible smoothness, no blockiness)
    smooth_grid = np.linspace(0.0, 0.5, 200)

    for gname, gdf in [("Female", g_F), ("Male", g_M)]:
        st.markdown(f"**{gname}** (n={len(gdf)})")
        cols = st.columns(len(methods_show))
        # Smoothed GT density on fine grid (vs blocky 25-bin step plot before)
        gt_smooth = _smooth_density(gdf["gt"].values, smooth_grid, bandwidth=0.012)
        # v20 fix: pick the right per-gender P_test target
        ref_density_for_gender = ref_density_F if gname == "Female" else ref_density_M
        for i, m in enumerate(methods_show):
            col = "pred_raw" if m == "raw" else f"pred_{m}"
            if col not in gdf.columns:
                continue
            alpha = best_alpha.get(m, 1.0)
            # Blend: α·cal + (1-α)·raw
            if m == "raw":
                pred_vals = gdf["pred_raw"].values
            else:
                pred_vals = np.clip(alpha * gdf[col].values + (1.0 - alpha) * gdf["pred_raw"].values, 0.0, 1.0)
            pred_density = _density(pred_vals, bin_edges)
            title = f"{m.upper()} (α={alpha:.1f})" if m != "raw" else "RAW"
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bin_centers, y=pred_density, name="pred",
                                  marker_color=bar_color.get(m, "#9ca3af"), opacity=0.85,
                                  width=bin_w * 0.95))
            fig.add_trace(go.Scatter(x=smooth_grid, y=gt_smooth, name="oracle GT (P_test sampled, n=5k, KDE)",
                                      mode="lines", line=dict(color=gt_color, width=2.5)))
            if ref_x is not None and ref_density_for_gender is not None:
                fig.add_trace(go.Scatter(x=ref_x, y=ref_density_for_gender,
                                          name=f"P_test cible | {gname[0]}",
                                          mode="lines", line=dict(color=target_color, width=3, dash="dot"),
                                          fill="tozeroy", fillcolor="rgba(134,239,172,0.08)"))
            fig.update_layout(title=title, height=280,
                                xaxis=dict(title="Y", range=[0, 0.5]),
                                yaxis=dict(title="density"),
                                margin=dict(l=30, r=10, t=35, b=30),
                                showlegend=(i == 0),
                                legend=dict(orientation="h", y=-0.25, x=0))
            _apply_dark_theme(fig)
            cols[i].plotly_chart(fig, use_container_width=True)

    # === 2bis. Drift toward P_test — quantification du glissement ===
    if ref_x is not None:
        st.markdown("### 2bis. Distance pred ↔ P_test (Wasserstein-1)")
        st.caption(
            "Pour chaque méthode × gender : W1(pred_density, P_test). "
            "Plus la barre est basse, plus la prédiction est proche de P_test (cible). "
            "Référence ambre pointillée = W1(oracle_GT, P_test) (= bruit d'échantillonnage du test_holdout 5k vs PDF 29 980, "
            "borne inférieure incompressible)."
        )
        # W1(p, q) = sum |CDF_p - CDF_q| * bin_width (1D, bins identiques)
        def _w1(p_density: np.ndarray, q_density: np.ndarray, bw: float) -> float:
            cdf_p = np.cumsum(p_density) * bw
            cdf_q = np.cumsum(q_density) * bw
            return float(np.sum(np.abs(cdf_p - cdf_q)) * bw)

        # v20: use the FINE grid (100 bins, 0.005 width) for W1 computation — précision 4×
        # supérieure au 25 bins legacy. Pred densities are computed on the same fine grid
        # for fair comparison with the high-resolution P_test reference.
        # v20 fix: W1 vs per-gender P_test cible (ref_density_F / ref_density_M), pas la marginale.
        fine_n_bins_w1 = _NB
        fine_bin_edges_w1 = np.linspace(0.0, 0.5, fine_n_bins_w1 + 1)
        fine_bin_w = fine_bin_edges_w1[1] - fine_bin_edges_w1[0]

        drift_rows: List[Dict[str, Any]] = []
        oracle_w1_by_g: Dict[str, float] = {}
        for gname, gdf in [("Female", g_F), ("Male", g_M)]:
            ptest_for_gender = ref_density_F if gname == "Female" else ref_density_M
            if ptest_for_gender is None:
                continue
            oracle_density = _density(gdf["gt"].values, fine_bin_edges_w1)
            oracle_w1_by_g[gname] = _w1(oracle_density, ptest_for_gender, fine_bin_w)
            for m in methods_show:
                col = "pred_raw" if m == "raw" else f"pred_{m}"
                if col not in gdf.columns:
                    continue
                alpha = best_alpha.get(m, 1.0)
                if m == "raw":
                    pred_vals = gdf["pred_raw"].values
                else:
                    pred_vals = np.clip(alpha * gdf[col].values + (1.0 - alpha) * gdf["pred_raw"].values, 0.0, 1.0)
                pred_density = _density(pred_vals, fine_bin_edges_w1)
                drift_rows.append({
                    "gender": gname, "method": m.upper(),
                    "W1_to_Ptest": _w1(pred_density, ptest_for_gender, fine_bin_w),
                })

        if drift_rows:
            df_drift = pd.DataFrame(drift_rows)
            fig_drift = px.bar(
                df_drift, x="method", y="W1_to_Ptest", color="method",
                facet_col="gender", height=380,
                color_discrete_map={m.upper(): bar_color.get(m, "#9ca3af") for m in methods_show},
                labels={"W1_to_Ptest": "W1(pred, P_test) — lower is closer"},
            )
            for gname, w1_ref in oracle_w1_by_g.items():
                # Reference: W1(oracle_GT, P_test_PDF) = inherent sampling noise of test_holdout vs PDF
                fig_drift.add_hline(
                    y=w1_ref, line_dash="dot", line_color=gt_color, line_width=2,
                    annotation_text=f"oracle GT noise floor: {w1_ref:.4f}",
                    annotation_position="top right",
                    annotation_font_color=gt_color,
                    col=1 if gname == "Female" else 2,
                )
            fig_drift.update_layout(showlegend=False, margin=dict(l=30, r=10, t=50, b=30))
            _apply_dark_theme(fig_drift)
            # facet titles + axis labels couleur claire (override sur les annotations facet de px)
            fig_drift.for_each_annotation(lambda a: a.update(font=dict(color="#e5e7eb"))
                                           if a.text and "oracle GT" not in a.text else None)
            st.plotly_chart(fig_drift, use_container_width=True)

    # === 3. Calibration plot: pred vs gt for RAW + each calibrator (small multiples) ===
    st.markdown("### 3. Calibration (pred vs gt), par méthode")
    st.caption("Points proches de y=x = bien calibré. Au-dessus = sur-estimation, en-dessous = sous-estimation. F=rose, M=cyan.")
    line_diag = go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", name="y=x",
                              line=dict(color="#e5e7eb", dash="dash", width=1.5), showlegend=False)
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
                                         marker=dict(color="#fb7185", size=3, opacity=0.5)))
            fig_c.add_trace(go.Scatter(x=g_M["gt"], y=g_M[col], mode="markers", name="M",
                                         marker=dict(color="#38bdf8", size=3, opacity=0.5)))
            fig_c.update_layout(title=m.upper(), xaxis_title="gt", yaxis_title="pred", height=350)
            _apply_dark_theme(fig_c)
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
        fig_mae.add_hline(y=0, line_dash="dash", line_color="#e5e7eb")
        _apply_dark_theme(fig_mae)
        fig_mae.for_each_annotation(lambda a: a.update(font=dict(color="#e5e7eb")))
        st.plotly_chart(fig_mae, use_container_width=True)

    with st.expander("Raw data (preview)"):
        st.dataframe(df_pred.head(50), use_container_width=True)


def _render_data_cleaning() -> None:
    """v35 step2: show proxy-flagged hard test images and the low-occ train anchors they remove,
    plus the train density at each cleaning stage."""
    import re as _re
    import plotly.graph_objects as go

    st.title("v35 — Proxy data cleaning (step 2)")
    import dataclasses
    from src.config import Config as _Cfg
    RAW = Path("data/raw")
    MID = _re.compile(r"(m\.[0-9a-zA-Z_]+)")
    # Thresholds are read from Config defaults (single source of truth) so the UI never drifts from
    # training. occ-probe is OFF (proxy_occ_thr huge): the anti-shortcut flags only degraded quality.
    _pd = {f.name: f.default for f in dataclasses.fields(_Cfg)}
    BLUR_T, TINT_T, CLIP_T, CUT = _pd["proxy_blur_thr"], _pd["proxy_tint_thr"], _pd["proxy_clip_thr"], _pd["proxy_cut"]

    # --- shared: test-flag projection (mirrors inference-time anchor removal) ---
    import os
    te = pd.read_csv("results/test_defects.csv") if os.path.exists("results/test_defects.csv") else None
    flag = fids = rm = clean = keepmax = None
    if te is not None:
        te["mid"] = te.filename.str.extract(MID.pattern)[0]
        col = lambda name, default: te[name] if name in te.columns else pd.Series(default, index=te.index)
        nb_mask = (col("clipv", -1e9) > CLIP_T) if "clipv" in te.columns else (col("sat", 1e9) < 12)
        te["why"] = np.where(col("blur", 1e9) < BLUR_T, "FLOU",
                      np.where(col("tint", -1e9) > TINT_T, "TEINTE",
                      np.where(nb_mask, "N&B", "")))
        flag = te[te.why != ""].copy()
        clean = pd.read_csv(Path(st.session_state.get("clean_csv", "data/raw/train_clean.csv")))
        clean["mid"] = clean.filename.str.extract(MID.pattern)[0]
        keepmax = set(clean.loc[clean.groupby("mid").FaceOcclusion.idxmax().values].filename)
        fids = set(flag.mid.dropna())
        rm = clean[clean.mid.isin(fids) & (clean.FaceOcclusion < CUT) & ~clean.filename.isin(keepmax)]

    tab_flag, tab_dist = st.tabs(["Images flaggées & ancres retirées", "Distributions par étape"])

    with tab_flag:
        if te is None:
            st.warning("Scores proxy test manquants (results/test_defects.csv).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Images test flaggées", f"{len(flag):,}")
            c2.metric("Identités triggées", f"{len(fids):,}")
            c3.metric("Ancres train retirées (projection)", f"{len(rm):,}")
            st.caption(f"Seuils (Config): FLOU blur<{BLUR_T:g} · TEINTE tint>{TINT_T:g} · N&B "
                       f"{'clipv>'+format(CLIP_T,'g') if 'clipv' in te.columns else 'sat<12 (clipv absent de test_defects.csv → fallback)'} · cut={CUT:g}. "
                       "⚠️ Projection inférence (flags test) — distincte de la dédup qui produit train_clean.csv.")
            opts = ["FLOU", "TEINTE", "N&B"]
            why = st.selectbox("Type de proxy", opts, key="dc_why")
            anchored = set(clean[clean.FaceOcclusion < CUT].mid)
            sub = flag[(flag.why == why) & (flag.mid.isin(anchored))].drop_duplicates("mid").head(12)
            st.caption("Gauche = image test flaggée (pourrie) · droite = ancres train basse-occ retirées de cette identité")
            for _, r in sub.iterrows():
                cols = st.columns([1, 4])
                with cols[0]:
                    p = RAW / r.filename
                    if p.exists(): st.image(str(p), caption=f"TEST {why}", use_container_width=True)
                with cols[1]:
                    anch = clean[(clean.mid == r.mid) & (clean.FaceOcclusion < CUT) &
                                 (~clean.filename.isin(keepmax))].head(8)
                    if len(anch):
                        ic = st.columns(len(anch))
                        for k, (_, a) in enumerate(anch.iterrows()):
                            ap = RAW / a.filename
                            if ap.exists(): ic[k].image(str(ap), caption=f"occ {a.FaceOcclusion:.2f}", use_container_width=True)

    with tab_dist:
        try:
            from src.utils.distribution import get_p_test_spline
            base = pd.read_csv("data/raw/train.csv").FaceOcclusion.values
            clean_df = pd.read_csv("data/raw/train_clean.csv")
            stages = [("base train (100k)", base, dict(color="#888", dash="dash", width=2)),
                      ("train_clean / dédup", clean_df.FaceOcclusion.values, dict(color="#1f77b4", width=2))]
            # step2 stage computed on the fly from the SAME projection as the metric (consistent)
            if rm is not None and len(rm):
                after = clean_df[~clean_df.filename.isin(set(rm.filename))].FaceOcclusion.values
                stages.append((f"après step2 (−{len(rm)} ancres)", after, dict(color="#2ca02c", width=2)))
            bins = np.linspace(0, 0.5, 51); ctr = 0.5 * (bins[:-1] + bins[1:])
            fig = go.Figure()
            for name, vals, style in stages:
                h, _ = np.histogram(vals, bins=bins, density=True)
                fig.add_trace(go.Scatter(x=ctr, y=h, mode="lines", name=f"{name} (n={len(vals):,})", line=style))
            pte = np.clip(get_p_test_spline()(ctr), 1e-9, None); pte /= pte.sum() * (bins[1] - bins[0])
            fig.add_trace(go.Scatter(x=ctr, y=pte, mode="lines", name="P_test", line=dict(color="red", width=3)))
            fig.add_vline(x=CUT, line_dash="dot", line_color="gray")
            fig.update_layout(title="Densité d'occlusion par étape de nettoyage", xaxis_title="occlusion", height=520)
            st.plotly_chart(_apply_dark_theme(fig), use_container_width=True)
            st.caption(f"base train et train_clean ne diffèrent que de {len(base)-len(clean_df):,} imgs (dédup strict) → "
                       f"courbes quasi superposées (base = pointillés gris). cut={CUT:g} ; le step2 amincit juste sous le cut.")
        except Exception as e:
            st.warning(f"Distributions indisponibles: {e}")


if mode == "v35 data cleaning (proxy)":
    _render_data_cleaning()
    st.stop()

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
    _exp_labels = list(exp_label_to_id.keys())
    # Default to the most-recently-created experiment (rename/version-proof; the old hardcoded "v22"
    # sent you to a stale experiment -> you'd land on the wrong run / "the last trial").
    try:
        _ct = {e.experiment_id: (e.creation_time or 0) for e in _get_client(TRACKING_URI).search_experiments()}
        _default_idx = max(range(len(_exp_labels)), key=lambda i: _ct.get(exp_label_to_id[_exp_labels[i]], 0))
    except Exception:
        _default_idx = 0
    selected_exp_label = st.selectbox("Experiment", _exp_labels, index=_default_idx)
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

# === Distribution des prédictions sur le TEST SET (best epoch, inférence réelle) ===
# Calculée une seule fois en fin de trial (artefact `test_predictions`). Affichée avant le bloc
# qualitative (qui peut st.stop()), donc visible même si save_qualitative_k=0.
st.subheader("Distribution des prédictions — test set (best epoch, inférence réelle)")
_tp_df = _download_test_predictions(TRACKING_URI, selected_run["run_id"])
if _tp_df is None or len(_tp_df) == 0:
    st.info("Pas de prédictions test pour ce run (artefact `test_predictions` absent — "
            "run antérieur à la feature, trial non terminé, ou test set indisponible au run).")
else:
    _num_cols = list(_tp_df.select_dtypes("number").columns)
    _col = "FaceOcclusion" if "FaceOcclusion" in _num_cols else (_num_cols[0] if _num_cols else None)
    _vals = pd.to_numeric(_tp_df[_col], errors="coerce").dropna().to_numpy() if _col else np.array([])
    if len(_vals) == 0:
        st.warning("Artefact `test_predictions` présent mais aucune colonne de prédiction numérique.")
    else:
        import plotly.graph_objects as go
        # Continuous density of the test predictions (KDE-like) vs the continuous P_test target.
        _grid, _dens = _continuous_density(_vals, bw=0.02)
        _ptest = _ptest_density(_grid)
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Histogram(x=_vals, nbinsx=80, histnorm="probability density",
                                      marker_color="rgba(76,120,168,0.20)", name="prédictions (hist)"))
        fig_tp.add_trace(go.Scatter(x=_grid, y=_dens, mode="lines", name="prédictions (densité continue)",
                                    line=dict(color="#4C78A8", width=2.6)))
        fig_tp.add_trace(go.Scatter(x=_grid, y=_ptest, mode="lines", name="P_test (cible finale)",
                                    line=dict(color="crimson", width=2, dash="dash")))
        _alpha = float(selected_run.get("eval_alpha", 0.5) or 0.5)
        _pg, _pdn = _ptrain_density_cached()
        if _pdn is not None:
            _inter = (1.0 - _alpha) * np.interp(_grid, _pg, _pdn) + _alpha * _ptest
            fig_tp.add_trace(go.Scatter(x=_grid, y=_inter, mode="lines",
                                        name=f"cible intermédiaire α={_alpha:.2f} (entre P_train/P_test)",
                                        line=dict(color="#2E8B57", width=2.2, dash="dot")))
        fig_tp.add_vline(x=0.3, line_dash="dot", line_color="gray")
        fig_tp.update_layout(xaxis_title="FaceOcclusion prédite", yaxis_title="densité",
                             height=400, bargap=0.02, margin=dict(t=10, b=10),
                             legend=dict(orientation="h", yanchor="bottom", y=1.0))
        st.plotly_chart(fig_tp, use_container_width=True)
        st.caption("Densité **continue** des prédictions test (best epoch) vs **P_test** (cible finale, rouge) "
                   "et la **cible intermédiaire** (vert, entre P_train et P_test) que v38 vise réellement. "
                   "La courbe bleue devrait épouser la **verte** (pas la rouge) tant qu'on cible l'intermédiaire.")
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("n (test)", f"{len(_vals):,}")
        _c2.metric("moyenne", f"{_vals.mean():.3f}")
        _c3.metric("médiane", f"{np.median(_vals):.3f}")
        _c4.metric("% > 0.3", f"{(_vals > 0.3).mean() * 100:.1f}%")
        _e1, _e2 = st.columns(2)
        _e1.metric("W₁(préd, P_test)", f"{_w1(_grid, _dens, _ptest):.4f}",
                   help="Wasserstein-1 (EMD) en unités d'occlusion entre la densité des prédictions et P_test. "
                        "0 = identiques. Plus robuste que KL (qui exploserait là où P_test=0, >0.5).")
        if _pdn is not None:
            _e2.metric("W₁(préd, cible interm.)", f"{_w1(_grid, _dens, _inter):.4f}",
                       help="Distance à la cible intermédiaire (verte) que v38 vise réellement.")

        # === Effet des post-calibrations sur la forme résultante (test) ===
        st.markdown("#### Formes résultantes selon le mode de post-calibration")
        _valdf = _download_val_predictions(TRACKING_URI, selected_run["run_id"])
        _modes = [("original", _vals, "#4C78A8"),
                  ("quantile-map → P_test", _quantile_map_to_ptest(_vals, _PTEST_PMF_05), "#E45756")]
        if _valdf is not None and {"gt", "pred"}.issubset(_valdf.columns) and len(_valdf) >= 50:
            _vgt = _valdf["gt"].to_numpy(float); _vpr = _valdf["pred"].to_numpy(float)
            _vw = 1.0 / 30.0 + _vgt
            _modes.append(("isotonic (val)", _isotonic_apply(_isotonic_fit(_vpr, _vgt, _vw), _vals), "#9467bd"))
            if "gender" in _valdf.columns:
                _gg = _valdf["gender"].to_numpy(float) >= 0.5
                if int(_gg.sum()) >= 25 and int((~_gg).sum()) >= 25:
                    _calF = _isotonic_apply(_isotonic_fit(_vpr[~_gg], _vgt[~_gg], _vw[~_gg]), _vals)
                    _calM = _isotonic_apply(_isotonic_fit(_vpr[_gg], _vgt[_gg], _vw[_gg]), _vals)
                    _modes.append(("isotonic per-gender (val, mix)", (float((~_gg).mean()), _calF, _calM), "#54A24B"))
            _iso_note = (f" Isotonic pondéré (w=1/30+gt) fit sur **val** (n={len(_valdf)}, iid P_train → caveat "
                         "covariate-shift). Per-gender = mélange par le prior genre du val (le test n'a pas de label genre).")
        else:
            _iso_note = " (artefact `val_predictions` absent → isotonic indisponible ; relance un trial après cette feature.)"

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=_grid, y=_ptest, mode="lines", name="P_test (cible finale)",
                                     line=dict(color="crimson", width=2, dash="dash")))
        _w1_rows = []
        for _name, _payload, _color in _modes:
            if isinstance(_payload, tuple):
                _pF, _cF, _cM = _payload
                _dmode = _pF * _continuous_density(_cF, bw=0.02)[1] + (1.0 - _pF) * _continuous_density(_cM, bw=0.02)[1]
                _mean = _pF * float(_cF.mean()) + (1.0 - _pF) * float(_cM.mean())
            else:
                _dmode = _continuous_density(_payload, bw=0.02)[1]
                _mean = float(np.mean(_payload))
            fig_cal.add_trace(go.Scatter(x=_grid, y=_dmode, mode="lines", name=_name,
                                         line=dict(color=_color, width=2.4)))
            _w1_rows.append({"mode": _name, "W₁ → P_test": round(_w1(_grid, _dmode, _ptest), 4), "moyenne": round(_mean, 3)})
        fig_cal.add_vline(x=0.3, line_dash="dot", line_color="gray")
        fig_cal.update_layout(xaxis_title="FaceOcclusion", yaxis_title="densité", height=380,
                              margin=dict(t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.0))
        st.plotly_chart(fig_cal, use_container_width=True)
        st.dataframe(pd.DataFrame(_w1_rows), hide_index=True, use_container_width=True)
        st.caption("Forme des prédictions **test** après chaque post-calibration vs P_test (rouge). "
                   "**quantile-map** vise directement la marginale P_test (non supervisé, recale aussi la masse à y≈0)."
                   + _iso_note)

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
    holdout_chart = qual_path / "diagnostics_holdout" / "error_vs_occlusion_and_density.png"
    val_chart = qual_path / "diagnostics" / "error_vs_occlusion_and_density.png"

    # v21: the HOLDOUT diagnostics are the honest ones (P_test distribution). The val
    # stratified-IS metric equalises err_F/err_M and hides the gender gap; the holdout
    # shows where it really lives (high-y tail, men). Show holdout first.
    if holdout_chart.exists():
        st.subheader("Test holdout (P_test) — vue honnête du gap")
        st.image(str(holdout_chart), use_container_width=True)
        st.caption(
            "**A** MAE par bin Y × genre : le gap F-M est ~0 partout SAUF la queue haut-Y où "
            "`err_M` explose. **B** contribution à `err_g` par bin : confirme que tout le gap "
            "vient du haut-Y. **C** densité d'échantillons. **D** distribution de `|pred-gt|`. "
            "C'est la distribution test réelle — contrairement au val, elle ne masque pas le biais genre."
        )
    if val_chart.exists():
        st.subheader("Val (iid P_train) — pour comparaison")
        st.image(str(val_chart), use_container_width=True)
        st.caption(
            "Mêmes panneaux sur le val. Le val suit P_train (peu d'hommes haut-Y) → l'estimateur "
            "stratifié-IS y égalise `err_F`/`err_M` et **sous-estime** le gap. À comparer au holdout ci-dessus."
        )
    if not holdout_chart.exists() and not val_chart.exists():
        st.info(
            "Aucun diagnostic chart pour ce run.\n\n"
            "Produits par `_save_diagnostic_charts()` dans `train.py` (val + holdout via subdir "
            "`diagnostics_holdout`, v21). Le holdout nécessite `test_split_ratio > 0`."
        )

with tab_params:
    params = _fetch_run_params(TRACKING_URI, selected_run["run_id"])
    if not params:
        st.info("No params logged for this run.")
    else:
        # Highlight panel : v10 search-space params ordonnés par lisibilité.
        OPTUNA_KEYS = [
            # === v16+ single-axis correction under H_C ===
            "correction_strength",
            # === Feature fairness ===
            "feature_fairness", "ot_lambda", "ot_method", "sinkhorn_eps", "adv_lambda",
            # === Loss / Lagrangien ===
            "loss_focal_gamma", "loss_lambda_threshold",
            # === Architecture ===
            "pretrained_source", "pooling_type", "grid_size", "mil_k_top", "mil_hidden",
            # === Hyperparams ===
            "learning_rate", "weight_decay", "layer_decay", "min_lr_rate",
            "head_dropout", "backbone_drop_path_rate",
            "pool_attn_dropout", "pool_proj_dropout",
            "tau_focal_init", "tau_diffuse_init",
            "n_focal", "n_diffuse", "n_free",
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
