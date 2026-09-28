# %% [cell 0]
# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [cell 1]
# ==============================================================================
# 0. Imports, configuration, reproducibility
# ==============================================================================
import os
import sys
import gc
import glob
import json
import time
import math
import random
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

warnings.filterwarnings("ignore")
pd.set_option("mode.chained_assignment", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("traffic_flow_bench")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    log.info("lightgbm not available -- Task 1 will use the profile-only fallback.")

LGBM_DEVICE = "cpu"


@dataclass
class Config:
    dataset_root_hints: List[str] = field(default_factory=lambda: [
        "/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench",
        "/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public",
        "/kaggle/input/2026-ieee-big-data-traffic-flow-bench",
        "kaggle_public", ".",
    ])
    working_dir: Path = Path("/kaggle/working")
    eda_dir: Path = Path("/kaggle/working/eda")
    submission_path: Path = Path("/kaggle/working/submission.csv")
    chunk_size: int = 500_000
    slots_per_week: int = 7 * 288
    queue_speed_ratio: float = 0.60
    queue_horizons_min: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    odme_lambda_grid: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0, 100.0)
    odme_lambda_default: float = 15.0
    min_train_rows_for_lgbm: int = 30_000
    lgbm_sample_cap: int = 600_000          # reservoir cap for residual-model training rows
    fd_alpha_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 0.81, 0.1), 2))
    tod_smooth_grid: Tuple[int, ...] = (1, 3, 5, 7)   # candidate circular smoothing windows, chosen on holdout
    tod_smooth_window: int = 3              # fallback/default if grid search is skipped
    queue_trend_weight_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))
    queue_threshold_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.80, 1.21, 0.025), 3))
    n_holdout_files: int = 3                # use up to this many most-recent files as a pooled holdout
    lgbm_n_estimators: int = 600
    lgbm_learning_rate: float = 0.04
    lgbm_num_leaves: int = 63
    lgbm_min_child_samples: int = 40
    use_lgbm_queue_model: bool = True       # train a gradient-boosted queue classifier when possible
    queue_min_train_rows: int = 800         # minimum backtest rows required before trusting the LGBM queue model


CFG = Config()


def ensure_dirs(cfg: Config = CFG) -> None:
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    cfg.eda_dir.mkdir(parents=True, exist_ok=True)


def timeit(label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            log.info(f"START  {label}")
            return self

        def __exit__(self, *exc):
            log.info(f"DONE   {label}  ({time.time() - self.t0:.1f}s)")
            return False
    return _Timer()


# %% [cell 2]
# ==============================================================================
# 1. Dataset location and manifest
# ==============================================================================
def locate_release_root(cfg: Config = CFG) -> Path:
    candidates = list(cfg.dataset_root_hints)
    candidates += glob.glob("/kaggle/input/**/kaggle_public", recursive=True)
    candidates += glob.glob("/kaggle/input/**/corridors", recursive=True)

    for c in candidates:
        p = Path(c)
        if (p / "config" / "corridors.json").exists() or (p / "corridors").exists():
            return p
        if p.name == "corridors" and p.parent.exists():
            return p.parent

    for p in Path("/kaggle/input").rglob("submission_key.csv"):
        return p.parent

    raise FileNotFoundError(
        "Could not locate the competition dataset. Checked: "
        f"{candidates}. Verify the competition data is attached to this "
        "notebook (Add Data) and the path matches /kaggle/input/competitions/"
        "2026-ieee-big-data-traffic-flow-bench."
    )


def load_panels(release_root: Path) -> List[str]:
    manifest_path = release_root / "config" / "corridors.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            panels = [p["corridor_id"] for p in manifest["panels"]]
            if panels:
                return panels
        except Exception as e:
            log.warning(f"Failed to parse corridors.json ({e}); falling back to directory scan.")

    corridor_dir = release_root / "corridors"
    if corridor_dir.exists():
        return sorted(p.name for p in corridor_dir.iterdir() if p.is_dir())

    raise FileNotFoundError("No corridors.json manifest and no corridors/ directory found.")


def safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        log.warning(f"Failed reading {path}: {e}")
        return None


def safe_read_parquet_files(files: List[Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=columns))
        except Exception as e:
            log.warning(f"Skipping unreadable parquet {f}: {e}")
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def compute_time_slots(df: pd.DataFrame, ts_col: str = "timestamp") -> Tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    weekday = ts.dt.weekday.fillna(0).to_numpy(dtype=np.int64)
    tod = ((ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)) // 5).to_numpy(dtype=np.int64)
    tod = np.clip(tod, 0, 287)
    return weekday, tod




# %% [cell 3]
# ==============================================================================
# 2. Exploratory data analysis 
# ==============================================================================
def run_eda(release_root: Path, panels: List[str], cfg: Config = CFG, sample_panels: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = panels[:sample_panels] if len(panels) > sample_panels else panels
    all_rows = []
    for panel in sample:
        train_dir = release_root / "corridors" / panel / "train"
        files = sorted((train_dir / "mainline_states").glob("**/*.parquet"))
        if not files:
            files = sorted((train_dir / "mainline_states_masked").glob("**/*.parquet"))
        if not files:
            log.warning(f"[EDA] No training state files for panel {panel}; skipping.")
            continue
        df = safe_read_parquet_files(
            files[: min(len(files), 5)],
            columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"],
        )
        if df.empty:
            continue
        df["panel"] = panel
        all_rows.append(df)

    if not all_rows:
        log.warning("[EDA] No data available to sample; skipping EDA plots.")
        return

    eda_df = pd.concat(all_rows, ignore_index=True)
    for col in ("speed_kmh", "flow_vph", "pct_observed"):
        eda_df[col] = pd.to_numeric(eda_df[col], errors="coerce")

    eligible = eda_df["pct_observed"].ge(75)
    log.info(f"[EDA] Sampled rows: {len(eda_df):,} across {len(sample)} panels")
    log.info(f"[EDA] Eligible fraction (pct_observed >= 75): {eligible.mean():.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(eda_df["speed_kmh"].dropna(), bins=60, color="steelblue")
    axes[0, 0].set_title("Speed distribution (km/h)")
    axes[0, 1].hist(eda_df["flow_vph"].dropna(), bins=60, color="darkorange")
    axes[0, 1].set_title("Flow distribution (veh/h)")

    ts = pd.to_datetime(eda_df["timestamp"], utc=True, errors="coerce")
    tod_hour = ts.dt.hour + ts.dt.minute / 60.0
    prof = eda_df.assign(tod_hour=tod_hour).groupby(
        pd.cut(tod_hour, bins=48)
    )[["speed_kmh", "flow_vph"]].mean()
    axes[1, 0].plot(range(len(prof)), prof["speed_kmh"], color="steelblue")
    axes[1, 0].set_title("Mean speed by time of day (sampled panels)")
    axes[1, 0].set_xlabel("30-min bin")

    axes[1, 1].hist(eda_df["pct_observed"].dropna(), bins=40, color="seagreen")
    axes[1, 1].axvline(75, color="red", linestyle="--", label="eligibility threshold")
    axes[1, 1].set_title("pct_observed distribution")
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = cfg.eda_dir / "eda_overview.png"
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info(f"[EDA] Saved overview figure to {out_path}")

    del eda_df, all_rows
    gc.collect()




# %% [cell 4]
# ==============================================================================
# 3. Fundamental-diagram parameters and  corridor topology
# ==============================================================================
def load_fd_parameters(panel_dir: Path) -> pd.DataFrame:
    fd_path = panel_dir / "network" / "fd_parameters.csv"
    fd = safe_read_csv(fd_path, dtype={"station_id": str, "link_id": str})

    if fd is not None and {"link_id"}.issubset(fd.columns):
        for col in ("v_f", "k_crit", "w", "k_jam", "lanes"):
            if col in fd.columns:
                fd[col] = pd.to_numeric(fd[col], errors="coerce")
        keep = ["link_id"] + [c for c in ("v_f", "k_crit", "w", "k_jam", "lanes") if c in fd.columns]
        fd = fd[keep].drop_duplicates("link_id")
        return fd

    return pd.DataFrame(columns=["link_id", "v_f", "k_crit", "w", "k_jam", "lanes"])


def estimate_free_flow_speed(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
    df["pct_observed"] = pd.to_numeric(df["pct_observed"], errors="coerce")
    eligible = df[df["pct_observed"].ge(75) & df["speed_kmh"].notna()]
    if eligible.empty:
        return pd.DataFrame(columns=["link_id", "v_f_est"])
    est = eligible.groupby("link_id")["speed_kmh"].quantile(0.95).rename("v_f_est").reset_index()
    return est


def load_link_topology(panel_dir: Path) -> Optional[pd.DataFrame]:
    """
    Best-effort loader for a corridor's link ordering, used only to smooth
    Task 2 queue predictions spatially (queues propagate to the adjacent
    upstream link). The release schema for this file is not guaranteed by
    the competition brief, so several plausible filenames/column names are
    tried and the function returns None (propagation simply skipped) if
    nothing usable is found -- Task 2 degrades gracefully to the per-link
    forecast with no spatial smoothing.
    """
    candidates = ["link_topology.csv", "topology.csv", "corridor_topology.csv", "links.csv"]
    seq_cols = ["seq", "sequence", "order", "milepost", "position"]
    for name in candidates:
        df = safe_read_csv(panel_dir / "network" / name, dtype={"link_id": str})
        if df is None or "link_id" not in df.columns:
            continue
        seq_col = next((c for c in seq_cols if c in df.columns), None)
        if seq_col is None:
            continue
        out = df[["link_id", seq_col]].rename(columns={seq_col: "seq"}).drop_duplicates("link_id")
        out["seq"] = pd.to_numeric(out["seq"], errors="coerce")
        out = out.dropna(subset=["seq"]).sort_values("seq").reset_index(drop=True)
        if len(out) >= 2:
            return out
    return None




# %% [cell 5]
# ==============================================================================
# 3. Task 1 -- time-of-week profile, vectorized lookup, LGBM residual
# ==============================================================================
def _smooth_circular(mat: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis=1, wrapping at the 288-slot boundary.
    Reduces noise in sparsely-observed (link, tod) cells without discarding
    the daily shape of the profile."""
    if window <= 1:
        return mat
    n = mat.shape[1]
    k = window // 2
    stack = np.stack([np.roll(mat, s, axis=1) for s in range(-k, k + 1)], axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(stack, axis=0)
    return out


def _read_link_ids(path: Path) -> List[str]:
    return sorted(pd.read_parquet(path, columns=["link_id"])["link_id"].astype(str).unique())


def _accumulate_profile(files: List[Path], link_idx: Dict[str, int], n_slots: int,
                         sample_cap: int, rng: np.random.Generator):
    """Streams a list of parquet files once, accumulating per-link/per-slot
    sum & count for speed and flow, plus a bounded reservoir sample of raw
    eligible rows (used later as LGBM training data)."""
    n_links = len(link_idx)
    speed_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    speed_cnt = np.zeros((n_links, n_slots), dtype=np.int32)
    flow_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    flow_cnt = np.zeros((n_links, n_slots), dtype=np.int32)
    sample_rows: List[pd.DataFrame] = []
    sample_total = 0
    fd_raw: List[pd.DataFrame] = []

    for path in files:
        try:
            df = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        except Exception as e:
            log.warning(f"Skipping unreadable file {path}: {e}")
            continue

        df["link_id"] = df["link_id"].astype(str)
        known = df["link_id"].isin(link_idx)
        df_known = df[known]

        if not df_known.empty:
            idx = df_known["link_id"].map(link_idx).to_numpy(dtype=np.int64)
            weekday, tod = compute_time_slots(df_known)
            slot = weekday * 288 + tod

            pct = pd.to_numeric(df_known["pct_observed"], errors="coerce")
            sp = pd.to_numeric(df_known["speed_kmh"], errors="coerce").to_numpy(dtype=float)
            fl = pd.to_numeric(df_known["flow_vph"], errors="coerce").to_numpy(dtype=float)
            eligible = (pct.ge(75)).to_numpy()

            v_sp = eligible & np.isfinite(sp)
            v_fl = eligible & np.isfinite(fl)

            np.add.at(speed_sum, (idx[v_sp], slot[v_sp]), sp[v_sp])
            np.add.at(speed_cnt, (idx[v_sp], slot[v_sp]), 1)
            np.add.at(flow_sum, (idx[v_fl], slot[v_fl]), fl[v_fl])
            np.add.at(flow_cnt, (idx[v_fl], slot[v_fl]), 1)

            both = v_sp & v_fl
            if both.any() and sample_total < sample_cap:
                sub = df_known.loc[both, ["link_id", "timestamp", "speed_kmh", "flow_vph"]]
                take = min(len(sub), max(1, sample_cap // max(1, len(files))))
                if take < len(sub):
                    sub = sub.sample(n=take, random_state=rng.integers(0, 2**31 - 1))
                sample_rows.append(sub)
                sample_total += len(sub)

        if len(fd_raw) < 3:
            fd_raw.append(df[["link_id", "speed_kmh", "pct_observed"]])

        del df
    gc.collect()

    sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame(
        columns=["link_id", "timestamp", "speed_kmh", "flow_vph"])
    fd_fallback_raw = pd.concat(fd_raw, ignore_index=True) if fd_raw else pd.DataFrame(
        columns=["link_id", "speed_kmh", "pct_observed"])
    return speed_sum, speed_cnt, flow_sum, flow_cnt, sample_df, fd_fallback_raw


def _finalize_profile(link_idx, speed_sum, speed_cnt, flow_sum, flow_cnt,
                       fd_fallback_raw, cfg: Config, smooth_window: Optional[int] = None) -> dict:
    """Builds the (link, slot) and (link, tod) lookup tables. `smooth_window`
    overrides cfg.tod_smooth_window so callers can grid-search it against the
    real holdout objective instead of using a fixed value."""
    n_slots = cfg.slots_per_week
    window = cfg.tod_smooth_window if smooth_window is None else smooth_window
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_mean = np.where(speed_cnt > 0, speed_sum / np.maximum(speed_cnt, 1), np.nan)
        flow_mean = np.where(flow_cnt > 0, flow_sum / np.maximum(flow_cnt, 1), np.nan)

    tod_index = np.arange(n_slots) % 288

    def collapse_by_tod(mean_mat, cnt_mat):
        out = np.full((len(link_idx), 288), np.nan)
        for t in range(288):
            cols = np.where(tod_index == t)[0]
            sub_sum = np.nansum(np.where(cnt_mat[:, cols] > 0, mean_mat[:, cols] * cnt_mat[:, cols], 0.0), axis=1)
            sub_cnt = np.nansum(cnt_mat[:, cols], axis=1)
            out[:, t] = np.where(sub_cnt > 0, sub_sum / np.maximum(sub_cnt, 1), np.nan)
        return out

    speed_by_tod = collapse_by_tod(speed_mean, speed_cnt)
    flow_by_tod = collapse_by_tod(flow_mean, flow_cnt)

    # Circular smoothing over time-of-day to stabilize sparsely-observed slots.
    speed_by_tod = _smooth_circular(speed_by_tod, window)
    flow_by_tod = _smooth_circular(flow_by_tod, window)

    panel_speed_by_tod = np.nanmean(speed_by_tod, axis=0)
    panel_flow_by_tod = np.nanmean(flow_by_tod, axis=0)
    panel_speed_scalar = np.nanmean(panel_speed_by_tod) if np.isfinite(np.nanmean(panel_speed_by_tod)) else 60.0
    panel_flow_scalar = np.nanmean(panel_flow_by_tod) if np.isfinite(np.nanmean(panel_flow_by_tod)) else 800.0

    fd_fallback = estimate_free_flow_speed(fd_fallback_raw) if not fd_fallback_raw.empty else \
        pd.DataFrame(columns=["link_id", "v_f_est"])

    return {
        "link_idx": link_idx,
        "speed_mean": speed_mean, "flow_mean": flow_mean,
        "speed_by_tod": speed_by_tod, "flow_by_tod": flow_by_tod,
        "panel_speed_by_tod": panel_speed_by_tod, "panel_flow_by_tod": panel_flow_by_tod,
        "panel_speed_scalar": float(panel_speed_scalar), "panel_flow_scalar": float(panel_flow_scalar),
        "fd_fallback": fd_fallback,
    }


def vectorized_profile_lookup(profile: dict, link_ids: np.ndarray, weekday: np.ndarray,
                               tod: np.ndarray, kind: str) -> np.ndarray:
    """Fully vectorized replacement for the original per-row Python loop.
    Same hierarchical fallback semantics: exact (link, slot) mean -> (link, tod)
    mean across weeks -> panel-wide (tod) mean -> panel scalar."""
    mean_mat = profile["speed_mean"] if kind == "speed" else profile["flow_mean"]
    by_tod = profile["speed_by_tod"] if kind == "speed" else profile["flow_by_tod"]
    panel_by_tod = profile["panel_speed_by_tod"] if kind == "speed" else profile["panel_flow_by_tod"]
    scalar = profile["panel_speed_scalar"] if kind == "speed" else profile["panel_flow_scalar"]

    n = len(link_ids)
    out = np.full(n, scalar, dtype=float)

    panel_vals = panel_by_tod[tod]
    valid_panel = np.isfinite(panel_vals)
    out[valid_panel] = panel_vals[valid_panel]

    link_idx_map = profile["link_idx"]
    idx = np.array([link_idx_map.get(lid, -1) for lid in link_ids], dtype=np.int64)
    known = idx >= 0

    if known.any():
        by_tod_vals = np.full(n, np.nan)
        by_tod_vals[known] = by_tod[idx[known], tod[known]]
        valid_bytod = known & np.isfinite(by_tod_vals)
        out[valid_bytod] = by_tod_vals[valid_bytod]

        slot = weekday * 288 + tod
        exact_vals = np.full(n, np.nan)
        exact_vals[known] = mean_mat[idx[known], slot[known]]
        valid_exact = known & np.isfinite(exact_vals)
        out[valid_exact] = exact_vals[valid_exact]

    return out


def link_categorical_code(profile: dict, link_ids: np.ndarray) -> np.ndarray:
    """Integer-codes link_id for use as a LightGBM categorical feature, so the
    residual model can learn per-link biases (systematic over/under-shoot of
    the time-of-week profile) that the tod/weekday cyclic features alone
    cannot express. Unknown links (not seen while building the profile) get
    a single shared out-of-vocabulary code."""
    link_idx_map = profile["link_idx"]
    oov = len(link_idx_map)
    return np.array([link_idx_map.get(lid, oov) for lid in link_ids], dtype=np.int64)


def clip_to_fundamental_diagram(speed: np.ndarray, flow: np.ndarray, v_f: np.ndarray,
                                 k_crit: np.ndarray, w: np.ndarray, k_jam: np.ndarray,
                                 alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Nudges (speed, flow) toward the triangular fundamental diagram to reduce
    Task 3 (S_physics/S_FD) violations. `alpha` is the blend weight toward the
    FD-consistent flow and is chosen per panel by held-out validation against
    the composite 0.35*S_state + 0.15*S_physics objective, rather than fixed."""
    speed = np.asarray(speed, dtype=float)
    flow = np.asarray(flow, dtype=float)
    v_f = np.asarray(v_f, dtype=float)
    k_crit = np.asarray(k_crit, dtype=float)
    w = np.asarray(w, dtype=float)
    k_jam = np.asarray(k_jam, dtype=float)

    has_params = np.isfinite(v_f) & np.isfinite(k_crit) & np.isfinite(w) & np.isfinite(k_jam) & (speed > 0)
    out_flow = flow.copy()

    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(speed > 0, flow / np.maximum(speed, 1e-6), 0.0)
        free_branch = v_f * density
        cong_branch = w * (k_jam - density)
        fd_flow = np.where(density <= k_crit, free_branch, cong_branch)
        fd_flow = np.clip(fd_flow, 0, None)

    blend = np.where(has_params & np.isfinite(fd_flow), (1 - alpha) * flow + alpha * fd_flow, flow)
    out_flow = np.where(has_params, blend, out_flow)
    out_speed = np.clip(speed, 0, np.where(np.isfinite(v_f) & (v_f > 0), v_f * 1.05, speed))
    return out_speed, out_flow


# %% [cell 6]
def score_state_component(speed_true, speed_pred, flow_true, flow_pred, lanes=None) -> float:
    """Implements S_state(r) = 0.54*S_v + 0.46*S_q from the competition brief,
    with flow RMSE computed per lane (falls back to per-link flow if lanes
    are unavailable, which only affects the internal validation score, not
    the official one)."""
    speed_true = np.asarray(speed_true, dtype=float)
    speed_pred = np.asarray(speed_pred, dtype=float)
    flow_true = np.asarray(flow_true, dtype=float)
    flow_pred = np.asarray(flow_pred, dtype=float)
    if lanes is not None:
        lanes = np.asarray(lanes, dtype=float)
        lanes = np.where(np.isfinite(lanes) & (lanes > 0), lanes, 1.0)
    else:
        lanes = np.ones_like(flow_true)

    mask = np.isfinite(speed_true) & np.isfinite(speed_pred) & np.isfinite(flow_true) & np.isfinite(flow_pred)
    if mask.sum() == 0:
        return 0.0
    rmse_v = float(np.sqrt(np.mean((speed_true[mask] - speed_pred[mask]) ** 2)))
    rmse_q = float(np.sqrt(np.mean(((flow_true[mask] - flow_pred[mask]) / lanes[mask]) ** 2)))
    s_v = max(0.0, 1 - rmse_v / 25.0)
    s_q = max(0.0, 1 - rmse_q / 600.0)
    return 0.54 * s_v + 0.46 * s_q


def score_fd_consistency(speed, flow, v_f, k_crit, w, k_jam) -> float:
    """Rough proxy for S_FD: 1 minus the mean normalized absolute gap between
    submitted flow and the FD-consistent flow at the submitted density.
    Used only to pick the blend weight alpha on internal validation data --
    not a substitute for the organizers' exact metric."""
    speed = np.asarray(speed, dtype=float)
    flow = np.asarray(flow, dtype=float)
    v_f = np.asarray(v_f, dtype=float)
    k_crit = np.asarray(k_crit, dtype=float)
    w = np.asarray(w, dtype=float)
    k_jam = np.asarray(k_jam, dtype=float)
    has_params = np.isfinite(v_f) & np.isfinite(k_crit) & np.isfinite(w) & np.isfinite(k_jam) & (speed > 0)
    if not has_params.any():
        return 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(speed > 0, flow / np.maximum(speed, 1e-6), 0.0)
        fd_flow = np.where(density <= k_crit, v_f * density, w * (k_jam - density))
        fd_flow = np.clip(fd_flow, 0, None)
    denom = np.maximum(np.maximum(flow, fd_flow), 1e-6)
    gap = np.abs(flow - fd_flow) / np.maximum(denom, 1e-6)
    return float(1 - np.clip(np.nanmean(gap[has_params]), 0, 1))


def _lgbm_features(link_ids, weekday, tod, profile_pred_speed, profile_pred_flow, lanes,
                    profile: Optional[dict] = None) -> pd.DataFrame:
    tod_frac = tod / 288.0
    wd_frac = weekday / 7.0
    feats = pd.DataFrame({
        "tod_slot": tod.astype(np.int32),
        "weekday": weekday.astype(np.int32),
        "is_weekend": (weekday >= 5).astype(np.int32),
        "tod_sin": np.sin(2 * np.pi * tod_frac), "tod_cos": np.cos(2 * np.pi * tod_frac),
        "wd_sin": np.sin(2 * np.pi * wd_frac), "wd_cos": np.cos(2 * np.pi * wd_frac),
        "profile_speed": profile_pred_speed, "profile_flow": profile_pred_flow,
        "lanes": lanes,
    })
    if profile is not None:
        # link_id as a categorical feature lets the tree learn per-link biases
        # (e.g. a consistently-congested on-ramp link) that tod/weekday cycles
        # alone cannot express -- this was previously missing entirely.
        link_code = link_categorical_code(profile, np.asarray(link_ids))
        feats["link_code"] = pd.Series(link_code, index=feats.index).astype("category")
    return feats


def _pooled_holdout(files: List[Path]) -> Optional[pd.DataFrame]:
    """Reads and concatenates a list of holdout files, applying the same
    eligibility filter used everywhere else. Pooling several files (instead
    of a single most-recent file) gives a much less noisy estimate of
    S_state when deciding whether to keep the LGBM residual model and which
    alpha/smoothing-window to use -- a single-file holdout can easily be won
    or lost by chance."""
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        except Exception as e:
            log.warning(f"Could not read holdout file {f}: {e}")
            continue
        df["link_id"] = df["link_id"].astype(str)
        df["pct_observed"] = pd.to_numeric(df["pct_observed"], errors="coerce")
        df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
        df["flow_vph"] = pd.to_numeric(df["flow_vph"], errors="coerce")
        df = df[df["pct_observed"].ge(75) & df["speed_kmh"].notna() & df["flow_vph"].notna()]
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def fit_task1_panel_model(panel_dir: Path, cfg: Config = CFG, rng: Optional[np.random.Generator] = None) -> Optional[dict]:
    """
    Builds the time-of-week profile for one panel, then uses a pooled holdout
    of the most recent files (up to cfg.n_holdout_files) to:
      (a) choose the circular tod-smoothing window,
      (b) decide whether a LightGBM residual correction (now including a
          link-identity categorical feature) improves S_state,
      (c) choose the fundamental-diagram blend weight alpha that maximizes
          0.35*S_state + 0.15*S_physics on the holdout (both computed with
          the exact formulas from the competition brief).
    Returns a model dict consumed by run_task1_predict(). Falls back to a
    profile-only model (alpha=0.3, no LGBM) when there isn't enough data to
    validate safely.
    """
    rng = rng or np.random.default_rng(SEED)
    train_dir = panel_dir / "train"
    files = sorted((train_dir / "mainline_states").glob("**/*.parquet"))
    if not files:
        files = sorted((train_dir / "mainline_states_masked").glob("**/*.parquet"))
    if not files:
        log.warning(f"No training state files under {train_dir}; Task 1 will use global fallback for this panel.")
        return None

    try:
        link_ids = _read_link_ids(files[0])
    except Exception as e:
        log.warning(f"Could not read link ids from {files[0]}: {e}")
        return None
    link_idx = {lid: i for i, lid in enumerate(link_ids)}

    n_holdout = min(cfg.n_holdout_files, max(0, len(files) - 1))
    holdout_files = files[-n_holdout:] if n_holdout > 0 else []
    train_files = files[:-n_holdout] if n_holdout > 0 else files

    speed_sum, speed_cnt, flow_sum, flow_cnt, sample_df, fd_raw = _accumulate_profile(
        train_files, link_idx, cfg.slots_per_week, cfg.lgbm_sample_cap, rng)

    fd = load_fd_parameters(panel_dir)

    def build_profile(window):
        prof = _finalize_profile(link_idx, speed_sum, speed_cnt, flow_sum, flow_cnt, fd_raw, cfg,
                                  smooth_window=window)
        return prof

    profile = build_profile(cfg.tod_smooth_window)
    if fd.empty and not profile["fd_fallback"].empty:
        fd = profile["fd_fallback"].rename(columns={"v_f_est": "v_f"})
        fd["k_crit"], fd["w"], fd["k_jam"] = np.nan, np.nan, np.nan
    fd_lookup = fd.set_index("link_id") if not fd.empty else pd.DataFrame(
        columns=["v_f", "k_crit", "w", "k_jam", "lanes"])

    model = {"profile": profile, "fd_lookup": fd_lookup, "lgbm_speed": None, "lgbm_flow": None,
             "alpha": 0.3}

    if not holdout_files:
        return model

    hold = _pooled_holdout(holdout_files)
    if hold is None or hold.empty:
        return model

    weekday_h, tod_h = compute_time_slots(hold, "timestamp")
    link_ids_h = hold["link_id"].to_numpy()
    speed_true = hold["speed_kmh"].to_numpy()
    flow_true = hold["flow_vph"].to_numpy()

    lanes_h = fd_lookup.reindex(link_ids_h)["lanes"].to_numpy() if "lanes" in fd_lookup.columns else np.ones(len(link_ids_h))
    v_f_h = fd_lookup.reindex(link_ids_h)["v_f"].to_numpy() if "v_f" in fd_lookup.columns else np.full(len(link_ids_h), np.nan)
    k_crit_h = fd_lookup.reindex(link_ids_h)["k_crit"].to_numpy() if "k_crit" in fd_lookup.columns else np.full(len(link_ids_h), np.nan)
    w_h = fd_lookup.reindex(link_ids_h)["w"].to_numpy() if "w" in fd_lookup.columns else np.full(len(link_ids_h), np.nan)
    k_jam_h = fd_lookup.reindex(link_ids_h)["k_jam"].to_numpy() if "k_jam" in fd_lookup.columns else np.full(len(link_ids_h), np.nan)
    lanes_h = np.where(np.isfinite(lanes_h), lanes_h, 1.0)

    # (a) circular tod-smoothing window, chosen on the pooled holdout
    best_window, best_window_score = cfg.tod_smooth_window, -np.inf
    for window in cfg.tod_smooth_grid:
        prof_w = build_profile(window)
        sp = vectorized_profile_lookup(prof_w, link_ids_h, weekday_h, tod_h, "speed")
        fl = vectorized_profile_lookup(prof_w, link_ids_h, weekday_h, tod_h, "flow")
        s = score_state_component(speed_true, sp, flow_true, fl, lanes_h)
        if s > best_window_score:
            best_window_score, best_window = s, window
    if best_window != cfg.tod_smooth_window:
        profile = build_profile(best_window)
        model["profile"] = profile
        if fd.empty and not profile["fd_fallback"].empty:
            fd = profile["fd_fallback"].rename(columns={"v_f_est": "v_f"})
            fd["k_crit"], fd["w"], fd["k_jam"] = np.nan, np.nan, np.nan
            fd_lookup = fd.set_index("link_id")
            model["fd_lookup"] = fd_lookup
    log.info(f"  Chosen tod-smoothing window={best_window} (holdout S_state={best_window_score:.4f})")

    profile_speed_h = vectorized_profile_lookup(profile, link_ids_h, weekday_h, tod_h, "speed")
    profile_flow_h = vectorized_profile_lookup(profile, link_ids_h, weekday_h, tod_h, "flow")

    best_speed_pred, best_flow_pred = profile_speed_h, profile_flow_h
    best_score = score_state_component(speed_true, best_speed_pred, flow_true, best_flow_pred, lanes_h)

  
    if HAS_LGBM and len(sample_df) >= cfg.min_train_rows_for_lgbm:
        try:
            s_link_ids = sample_df["link_id"].to_numpy()
            s_weekday, s_tod = compute_time_slots(sample_df, "timestamp")
            s_profile_speed = vectorized_profile_lookup(profile, s_link_ids, s_weekday, s_tod, "speed")
            s_profile_flow = vectorized_profile_lookup(profile, s_link_ids, s_weekday, s_tod, "flow")
            s_lanes = fd_lookup.reindex(s_link_ids)["lanes"].to_numpy() if "lanes" in fd_lookup.columns else np.ones(len(s_link_ids))
            s_lanes = np.where(np.isfinite(s_lanes), s_lanes, 1.0)

            feats = _lgbm_features(s_link_ids, s_weekday, s_tod, s_profile_speed, s_profile_flow, s_lanes,
                                    profile=profile)
            y_speed = sample_df["speed_kmh"].to_numpy() - s_profile_speed
            y_flow = sample_df["flow_vph"].to_numpy() - s_profile_flow

            # small internal train/val split for early stopping so the larger
            # n_estimators budget doesn't just overfit the residuals
            n = len(feats)
            perm = rng.permutation(n)
            n_val = max(1, int(0.15 * n))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            cat_cols = ["link_code"] if "link_code" in feats.columns else "auto"

            lgbm_kwargs = dict(
                n_estimators=cfg.lgbm_n_estimators, num_leaves=cfg.lgbm_num_leaves,
                learning_rate=cfg.lgbm_learning_rate, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=cfg.lgbm_min_child_samples, reg_alpha=0.1, reg_lambda=0.1,
                random_state=SEED, device=LGBM_DEVICE, verbosity=-1,
            )
            lgbm_speed = lgb.LGBMRegressor(**lgbm_kwargs)
            lgbm_flow = lgb.LGBMRegressor(**lgbm_kwargs)
            try:
                lgbm_speed.fit(feats.iloc[tr_idx], y_speed[tr_idx],
                                eval_set=[(feats.iloc[val_idx], y_speed[val_idx])],
                                eval_metric="l2", callbacks=[lgb.early_stopping(50, verbose=False)],
                                categorical_feature=cat_cols)
                lgbm_flow.fit(feats.iloc[tr_idx], y_flow[tr_idx],
                               eval_set=[(feats.iloc[val_idx], y_flow[val_idx])],
                               eval_metric="l2", callbacks=[lgb.early_stopping(50, verbose=False)],
                               categorical_feature=cat_cols)
            except TypeError:
                # older lightgbm without callbacks API
                lgbm_speed.fit(feats, y_speed, categorical_feature=cat_cols)
                lgbm_flow.fit(feats, y_flow, categorical_feature=cat_cols)

            feats_h = _lgbm_features(link_ids_h, weekday_h, tod_h, profile_speed_h, profile_flow_h,
                                      lanes_h, profile=profile)
            speed_pred_lgbm = profile_speed_h + lgbm_speed.predict(feats_h)
            flow_pred_lgbm = profile_flow_h + lgbm_flow.predict(feats_h)
            score_lgbm = score_state_component(speed_true, speed_pred_lgbm, flow_true, flow_pred_lgbm, lanes_h)

            if score_lgbm > best_score:
                model["lgbm_speed"] = lgbm_speed
                model["lgbm_flow"] = lgbm_flow
                best_speed_pred, best_flow_pred, best_score = speed_pred_lgbm, flow_pred_lgbm, score_lgbm
                log.info(f"  LGBM residual model (+link_code) improved holdout S_state to {best_score:.4f} "
                         f"({os.path.basename(str(panel_dir))})")
            else:
                log.info(f"  LGBM residual model did not beat profile baseline "
                         f"({score_lgbm:.4f} <= {best_score:.4f}); keeping profile-only.")
        except Exception as e:
            log.warning(f"LightGBM residual training failed ({e}); using profile-only predictions.")

    # (c) fundamental-diagram blend weight, chosen against the true composite objective
    best_alpha, best_obj = 0.0, -np.inf
    for alpha in cfg.fd_alpha_grid:
        sp, fl = clip_to_fundamental_diagram(best_speed_pred, best_flow_pred, v_f_h, k_crit_h, w_h, k_jam_h, alpha=alpha)
        s_state = score_state_component(speed_true, sp, flow_true, fl, lanes_h)
        s_fd = score_fd_consistency(sp, fl, v_f_h, k_crit_h, w_h, k_jam_h)
        objective = 0.35 * s_state + 0.15 * s_fd
        if objective > best_obj:
            best_obj, best_alpha = objective, alpha
    model["alpha"] = float(best_alpha)
    log.info(f"  Chosen FD blend alpha={best_alpha} (holdout objective={best_obj:.4f})")

    return model


# %% [cell 7]
def run_task1_predict(panel_dir: Path, model: dict, rows: pd.DataFrame) -> pd.DataFrame:
    profile, fd_lookup, alpha = model["profile"], model["fd_lookup"], model["alpha"]

    weekday, tod = compute_time_slots(rows, "timestamp")
    link_ids = rows["link_id"].astype(str).to_numpy()

    speed_pred = vectorized_profile_lookup(profile, link_ids, weekday, tod, "speed")
    flow_pred = vectorized_profile_lookup(profile, link_ids, weekday, tod, "flow")

    if model.get("lgbm_speed") is not None and model.get("lgbm_flow") is not None:
        lanes = fd_lookup.reindex(link_ids)["lanes"].to_numpy() if "lanes" in fd_lookup.columns else np.ones(len(link_ids))
        lanes = np.where(np.isfinite(lanes), lanes, 1.0)
        feats = _lgbm_features(link_ids, weekday, tod, speed_pred, flow_pred, lanes, profile=profile)
        speed_pred = speed_pred + model["lgbm_speed"].predict(feats)
        flow_pred = flow_pred + model["lgbm_flow"].predict(feats)

    if not fd_lookup.empty:
        v_f = fd_lookup.reindex(link_ids)["v_f"].to_numpy() if "v_f" in fd_lookup.columns else np.full(len(link_ids), np.nan)
        k_crit = fd_lookup.reindex(link_ids)["k_crit"].to_numpy() if "k_crit" in fd_lookup.columns else np.full(len(link_ids), np.nan)
        w = fd_lookup.reindex(link_ids)["w"].to_numpy() if "w" in fd_lookup.columns else np.full(len(link_ids), np.nan)
        k_jam = fd_lookup.reindex(link_ids)["k_jam"].to_numpy() if "k_jam" in fd_lookup.columns else np.full(len(link_ids), np.nan)
        speed_pred, flow_pred = clip_to_fundamental_diagram(speed_pred, flow_pred, v_f, k_crit, w, k_jam, alpha=alpha)

    out = rows[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
    out["speed_kmh"] = np.clip(speed_pred, 0, None)
    out["flow_vph"] = np.clip(flow_pred, 0, None)
    return out


def run_task1(release_root: Path, panels: List[str], key: pd.DataFrame, cfg: Config = CFG) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    state_rows = key[key["task"] == "state"].copy()
    task1_models: Dict[str, dict] = {}
    if state_rows.empty:
        return pd.DataFrame(columns=["panel", "timestamp", "station_id", "link_id", "mask_regime",
                                      "speed_kmh", "flow_vph"]), task1_models

    state_rows["timestamp"] = pd.to_datetime(state_rows["timestamp"], utc=True, errors="coerce")
    results = []
    rng = np.random.default_rng(SEED)

    for panel in panels:
        panel_dir = release_root / "corridors" / panel
        rows = state_rows[state_rows["panel"] == panel]
        if rows.empty:
            continue

        model = fit_task1_panel_model(panel_dir, cfg, rng)
        if model is None:
            out = rows[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
            out["speed_kmh"] = 60.0
            out["flow_vph"] = 800.0
            results.append(out)
            continue

        task1_models[panel] = model
        results.append(run_task1_predict(panel_dir, model, rows))
        gc.collect()

    if not results:
        return pd.DataFrame(columns=["panel", "timestamp", "station_id", "link_id", "mask_regime",
                                      "speed_kmh", "flow_vph"]), task1_models
    return pd.concat(results, ignore_index=True), task1_models


# %% [cell 8]
# ==============================================================================
# 4. Task 2 -- queue state
# ==============================================================================
def build_free_flow_lookup(panel_dir: Path) -> Dict[str, float]:
    fd = load_fd_parameters(panel_dir)
    if not fd.empty and "v_f" in fd.columns:
        vf = fd.dropna(subset=["v_f"]).set_index("link_id")["v_f"]
        if not vf.empty:
            return vf.to_dict()

    train_dir = panel_dir / "train"
    files = sorted((train_dir / "mainline_states").glob("**/*.parquet"))[:2]
    if not files:
        files = sorted((train_dir / "mainline_states_masked").glob("**/*.parquet"))[:2]
    if not files:
        return {}
    df = safe_read_parquet_files(files, columns=["link_id", "speed_kmh", "pct_observed"])
    if df.empty:
        return {}
    est = estimate_free_flow_speed(df)
    return est.set_index("link_id")["v_f_est"].to_dict()


def _trend_features(g: pd.DataFrame) -> pd.Series:
    g = g.dropna(subset=["speed_kmh"])
    if g.empty:
        return pd.Series({"last_speed": np.nan, "slope_per_min": 0.0, "accel": 0.0,
                           "speed_std": 0.0, "n_obs": 0})
    tail = g.tail(6)
    last_speed = tail["speed_kmh"].iloc[-1]
    speed_std = float(tail["speed_kmh"].std()) if len(tail) > 1 else 0.0
    if len(tail) >= 2:
        t_minutes = (tail["timestamp"] - tail["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
        y = tail["speed_kmh"].to_numpy()
        slope = np.polyfit(t_minutes, y, 1)[0] if np.ptp(t_minutes) > 0 else 0.0
        if len(tail) >= 3 and np.ptp(t_minutes) > 0:
            quad = np.polyfit(t_minutes, y, 2)
            accel = float(2 * quad[0])
        else:
            accel = 0.0
    else:
        slope = 0.0
        accel = 0.0
    return pd.Series({"last_speed": last_speed, "slope_per_min": slope, "accel": accel,
                       "speed_std": speed_std, "n_obs": len(tail)})


def _forecast_speed(last_speed, slope_per_min, horizon_min, free_flow, w_trend):
    """Ensemble of damped-trend extrapolation and reversion to free flow.
    w_trend in [0,1] controls how much weight the trend keeps at long
    horizons (1 = pure trend/persistence, 0 = reverts fully to free flow);
    it is chosen by backtesting against real held-out history (see
    calibrate_queue_model)."""
    damp = 1.0 / (1.0 + 0.03 * np.clip(horizon_min, 0, None))
    trend_component = last_speed + slope_per_min * horizon_min * damp
    horizon_frac = np.clip(horizon_min / 30.0, 0, 1)
    reversion_weight = (1 - w_trend) * horizon_frac
    blended = (1 - reversion_weight) * trend_component + reversion_weight * free_flow
    return np.clip(blended, 0, free_flow * 1.1)


def _build_backtest_table(history: pd.DataFrame, ff_lookup: Dict[str, float], default_ff: float,
                           cfg: Config = CFG) -> pd.DataFrame:
    """Splits each (window_id, link_id) history series in half: the early
    half is used to compute trend features (as at real prediction time), the
    late half supplies horizons + ground-truth queue labels. Returns one row
    per (group, late-timestamp) -- i.e. a proper supervised table, used both
    by the heuristic grid search and by the optional LGBM classifier."""
    h = history.dropna(subset=["speed_kmh"]).sort_values(["window_id", "link_id", "timestamp"])
    if h.empty:
        return pd.DataFrame()

    groups = list(h.groupby(["window_id", "link_id"]))
    if len(groups) > 1200:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(groups), size=1200, replace=False)
        groups = [groups[i] for i in idx]

    rows = []
    for (wid, lid), g in groups:
        g = g.reset_index(drop=True)
        if len(g) < 8:
            continue
        split = max(2, len(g) // 2)
        early, late = g.iloc[:split], g.iloc[split:]
        if len(early) < 2 or late.empty:
            continue
        feat = _trend_features(early)
        ff = ff_lookup.get(lid, default_ff)
        origin_t = early["timestamp"].iloc[-1]
        horizon_min = (late["timestamp"] - origin_t).dt.total_seconds().to_numpy() / 60.0
        true_speed = late["speed_kmh"].to_numpy()
        true_queue = (true_speed <= cfg.queue_speed_ratio * ff).astype(int)
        for hm, tq in zip(horizon_min, true_queue):
            rows.append({
                "window_id": wid, "link_id": lid,
                "last_speed": feat["last_speed"], "slope_per_min": feat["slope_per_min"],
                "accel": feat["accel"], "speed_std": feat["speed_std"], "n_obs": feat["n_obs"],
                "free_flow": ff, "horizon_min": hm, "queue_true": tq,
            })
    return pd.DataFrame(rows)


def calibrate_queue_model(table: pd.DataFrame, cfg: Config = CFG) -> Tuple[float, float]:
    """Grid-searches w_trend and the threshold multiplier to maximize the
    true space-time IoU (as defined in the brief) on the backtest table."""
    if table.empty:
        return 0.5, 1.0

    last_speed = table["last_speed"].to_numpy()
    slope = table["slope_per_min"].to_numpy()
    horizon_min = table["horizon_min"].to_numpy()
    ff = table["free_flow"].to_numpy()
    true_queue = table["queue_true"].to_numpy()

    best_w, best_thr, best_iou = 0.5, 1.0, -1.0
    for w_trend in cfg.queue_trend_weight_grid:
        fs = _forecast_speed(last_speed, slope, horizon_min, ff, w_trend)
        for thr_mult in cfg.queue_threshold_grid:
            pred_queue = (fs <= thr_mult * cfg.queue_speed_ratio * ff).astype(int)
            inter = int(np.sum((pred_queue == 1) & (true_queue == 1)))
            union = int(np.sum((pred_queue == 1) | (true_queue == 1)))
            iou = 1.0 if union == 0 else inter / union
            if iou > best_iou:
                best_iou, best_w, best_thr = iou, w_trend, thr_mult

    log.info(f"  Task 2 heuristic calibration: w_trend={best_w}, threshold_mult={best_thr}, "
             f"backtest IoU={best_iou:.4f}")
    return float(best_w), float(best_thr)


def _queue_lgbm_features(table: pd.DataFrame) -> pd.DataFrame:
    ratio = table["last_speed"] / np.maximum(table["free_flow"], 1e-6)
    return pd.DataFrame({
        "last_speed": table["last_speed"].to_numpy(),
        "slope_per_min": table["slope_per_min"].to_numpy(),
        "accel": table["accel"].to_numpy(),
        "speed_std": table["speed_std"].to_numpy(),
        "n_obs": table["n_obs"].to_numpy(),
        "free_flow": table["free_flow"].to_numpy(),
        "horizon_min": table["horizon_min"].to_numpy(),
        "speed_ratio": ratio.to_numpy(),
    })


def fit_queue_model(table: pd.DataFrame, cfg: Config = CFG) -> dict:
    """Always fits the tuned heuristic (fast, robust, works with little data).
    When there's enough backtest data and lightgbm is available, additionally
    fits a gradient-boosted classifier with richer lag/curvature features and
    keeps it only if it beats the heuristic's IoU on a held-out split of the
    backtest table -- this can adapt per-corridor far better than a single
    global (w_trend, threshold) pair."""
    w_trend, thr_mult = calibrate_queue_model(table, cfg)
    model = {"type": "heuristic", "w_trend": w_trend, "thr_mult": thr_mult, "clf": None,
             "clf_threshold": 0.5}

    if not (cfg.use_lgbm_queue_model and HAS_LGBM) or len(table) < cfg.queue_min_train_rows:
        return model

    try:
        rng = np.random.default_rng(SEED)
        n = len(table)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        X = _queue_lgbm_features(table)
        y = table["queue_true"].to_numpy()

        clf = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, device=LGBM_DEVICE, verbosity=-1,
        )
        try:
            clf.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[val_idx], y[val_idx])],
                    eval_metric="binary_logloss", callbacks=[lgb.early_stopping(40, verbose=False)])
        except TypeError:
            clf.fit(X.iloc[tr_idx], y[tr_idx])

        # threshold search on the held-out split, by IoU (matching the brief's metric)
        proba_val = clf.predict_proba(X.iloc[val_idx])[:, 1]
        y_val = y[val_idx]
        best_iou_clf, best_thr_clf = -1.0, 0.5
        for thr in np.round(np.arange(0.1, 0.91, 0.025), 3):
            pred = (proba_val >= thr).astype(int)
            inter = int(np.sum((pred == 1) & (y_val == 1)))
            union = int(np.sum((pred == 1) | (y_val == 1)))
            iou = 1.0 if union == 0 else inter / union
            if iou > best_iou_clf:
                best_iou_clf, best_thr_clf = iou, thr

        # compare against the heuristic's IoU on the SAME held-out split for a fair decision
        fs_val = _forecast_speed(table["last_speed"].to_numpy()[val_idx], table["slope_per_min"].to_numpy()[val_idx],
                                  table["horizon_min"].to_numpy()[val_idx], table["free_flow"].to_numpy()[val_idx],
                                  w_trend)
        pred_heur = (fs_val <= thr_mult * cfg.queue_speed_ratio * table["free_flow"].to_numpy()[val_idx]).astype(int)
        inter_h = int(np.sum((pred_heur == 1) & (y_val == 1)))
        union_h = int(np.sum((pred_heur == 1) | (y_val == 1)))
        iou_heur = 1.0 if union_h == 0 else inter_h / union_h

        log.info(f"  Task 2 LGBM classifier held-out IoU={best_iou_clf:.4f} vs heuristic IoU={iou_heur:.4f}")
        if best_iou_clf > iou_heur:
            model["type"] = "lgbm"
            model["clf"] = clf
            model["clf_threshold"] = float(best_thr_clf)
    except Exception as e:
        log.warning(f"Task 2 LGBM classifier training failed ({e}); using heuristic ensemble only.")

    return model


def apply_spatial_smoothing(out: pd.DataFrame, panel: str, panel_dir: Path,
                             margin: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """If a corridor topology file is available, nudge a link's prediction
    toward 'queued' when it sits close to the decision boundary (margin near
    1.0) and an adjacent link (by corridor sequence) at the same timestamp is
    already predicted queued -- queues propagate upstream along a link
    sequence. No-op when no topology file is found. `margin` should be >1
    when comfortably NOT queued and <=1 when queued/near the boundary,
    regardless of whether the heuristic or LGBM model produced it."""
    topo = load_link_topology(panel_dir)
    if topo is None:
        return out["queue_pred"].to_numpy()

    seq_map = dict(zip(topo["link_id"], topo["seq"]))
    df = out.reset_index(drop=True).copy()
    df["seq"] = df["link_id"].map(seq_map)
    df["margin"] = margin
    if df["seq"].isna().all():
        return out["queue_pred"].to_numpy()

    pred = df["queue_pred"].to_numpy().copy()
    for ts, grp in df.groupby("timestamp"):
        grp = grp.dropna(subset=["seq"]).sort_values("seq")
        if len(grp) < 2:
            continue
        idxs = grp.index.to_numpy()
        seqs = grp["seq"].to_numpy()
        q = pred[idxs].copy()
        margin_grp = grp["margin"].to_numpy()
        close_call = (margin_grp > 0.9) & (margin_grp <= 1.15) & (q == 0)
        neighbor_queued = np.zeros(len(idxs), dtype=bool)
        neighbor_queued[1:] |= q[:-1] == 1
        neighbor_queued[:-1] |= q[1:] == 1
        flip = close_call & neighbor_queued
        q[flip] = 1
        pred[idxs] = q
    return pred


def run_task2(release_root: Path, panels: List[str], key: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    queue_rows = key[key["task"] == "queue"].copy()
    if queue_rows.empty:
        return pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"])

    wi_files = sorted(release_root.glob("**/window_index.csv"))
    hist_files = sorted(release_root.glob("**/window_history.parquet"))
    if not (wi_files and hist_files):
        log.warning("Task 2 support files (window_index / window_history) not found; predicting no-queue everywhere.")
        out = queue_rows[["window_id", "timestamp", "link_id"]].copy()
        out["queue_pred"] = 0
        return out

    windows = pd.concat([safe_read_csv(f, dtype=str) for f in wi_files if f.exists()], ignore_index=True)
    windows = windows.dropna(how="all")

    history = safe_read_parquet_files(hist_files, columns=["window_id", "timestamp", "link_id", "speed_kmh"])
    history["window_id"] = history["window_id"].astype(str)
    history["link_id"] = history["link_id"].astype(str)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history["speed_kmh"] = pd.to_numeric(history["speed_kmh"], errors="coerce")
    history = history.sort_values(["window_id", "link_id", "timestamp"])

    panel_col = "panel" if "panel" in windows.columns else None
    ff_lookup: Dict[str, float] = {}
    if panel_col:
        for panel in windows[panel_col].dropna().unique():
            if panel in panels:
                ff_lookup.update(build_free_flow_lookup(release_root / "corridors" / panel))
    else:
        for panel in panels:
            ff_lookup.update(build_free_flow_lookup(release_root / "corridors" / panel))

    default_ff = float(np.nanmedian(list(ff_lookup.values()))) if ff_lookup else 90.0

    backtest_table = _build_backtest_table(history, ff_lookup, default_ff, cfg)
    queue_model = fit_queue_model(backtest_table, cfg)
    log.info(f"  Task 2 model selected: {queue_model['type']}")

    trend = history.groupby(["window_id", "link_id"]).apply(_trend_features).reset_index()

    queue_rows["window_id"] = queue_rows["window_id"].astype(str)
    queue_rows["link_id"] = queue_rows["link_id"].astype(str)
    queue_rows["timestamp"] = pd.to_datetime(queue_rows["timestamp"], utc=True, errors="coerce")

    merged = queue_rows.merge(trend, on=["window_id", "link_id"], how="left").reset_index(drop=True)

    origin = merged.groupby("window_id")["timestamp"].transform("min")
    horizon_min = (merged["timestamp"] - origin).dt.total_seconds() / 60.0

    merged["free_flow"] = merged["link_id"].map(ff_lookup).fillna(default_ff)
    merged["last_speed"] = merged["last_speed"].fillna(merged["free_flow"] * 0.9)
    merged["slope_per_min"] = merged["slope_per_min"].fillna(0.0)
    merged["accel"] = merged.get("accel", 0.0)
    merged["accel"] = merged["accel"].fillna(0.0)
    merged["speed_std"] = merged.get("speed_std", 0.0)
    merged["speed_std"] = merged["speed_std"].fillna(0.0)
    merged["n_obs"] = merged.get("n_obs", 0)
    merged["n_obs"] = merged["n_obs"].fillna(0)

    if queue_model["type"] == "lgbm":
        feat_table = pd.DataFrame({
            "last_speed": merged["last_speed"].to_numpy(),
            "slope_per_min": merged["slope_per_min"].to_numpy(),
            "accel": merged["accel"].to_numpy(),
            "speed_std": merged["speed_std"].to_numpy(),
            "n_obs": merged["n_obs"].to_numpy(),
            "free_flow": merged["free_flow"].to_numpy(),
            "horizon_min": horizon_min.to_numpy(),
        })
        feat_table["speed_ratio"] = feat_table["last_speed"] / np.maximum(feat_table["free_flow"], 1e-6)
        proba = queue_model["clf"].predict_proba(feat_table)[:, 1]
        thr = queue_model["clf_threshold"]
        queue_pred = (proba >= thr).astype(int)
       
        margin = np.where(proba > 1e-9, thr / np.maximum(proba, 1e-9), 10.0)
    else:
        forecast_speed = _forecast_speed(
            merged["last_speed"].to_numpy(), merged["slope_per_min"].to_numpy(),
            horizon_min.to_numpy(), merged["free_flow"].to_numpy(), queue_model["w_trend"])
        thr_speed = queue_model["thr_mult"] * cfg.queue_speed_ratio * merged["free_flow"].to_numpy()
        queue_pred = (forecast_speed <= thr_speed).astype(int)
        margin = forecast_speed / np.maximum(thr_speed, 1e-6)

    out = merged[["window_id", "timestamp", "link_id"]].copy()
    if "panel" in merged.columns:
        out["panel"] = merged["panel"]
    else:
        out["panel"] = out["window_id"].astype(str).str.rsplit("_W", n=1).str[0]
    out["queue_pred"] = queue_pred

    smoothed_chunks = []
    for panel, grp in out.groupby("panel"):
        panel_dir = release_root / "corridors" / panel
        if not panel_dir.exists():
            smoothed_chunks.append(grp["queue_pred"].to_numpy())
            continue
        m = margin[grp.index.to_numpy()]
        smoothed_chunks.append(apply_spatial_smoothing(grp, panel, panel_dir, m, cfg))
    if smoothed_chunks:
        out["queue_pred"] = np.concatenate(smoothed_chunks)

    out = out.drop(columns=["panel"])
    del history, trend, merged
    gc.collect()
    return out


# %% [cell 9]
# ==============================================================================
# 5. Task 4 --  NNLS OD estimation with per-corridor lambda search
# ==============================================================================
def _solve_nnls(A: np.ndarray, counts: np.ndarray, anchor: np.ndarray, lam: float) -> np.ndarray:
    aa = np.vstack([A, math.sqrt(lam) * np.eye(A.shape[1])])
    bb = np.concatenate([counts, math.sqrt(lam) * anchor])
    flows, _ = nnls(aa, bb)
    return flows


def _select_lambda(A: np.ndarray, counts: np.ndarray, anchor: np.ndarray, cfg: Config,
                    rng: np.random.Generator) -> float:
    """Leave-out validation on the observed link counts: hold out a random
    subset of links' counts, fit NNLS with the remaining counts at each
    candidate lambda, and score by reconstruction error on the held-out
    counts. This directly targets S_link/S_od's implicit trade-off between
    fitting counts and staying close to the prior, instead of a fixed
    lambda for every corridor."""
    n_links = A.shape[0]
    if n_links < 4:
        return cfg.odme_lambda_default
    holdout_frac = 0.25
    n_hold = max(1, int(n_links * holdout_frac))
    idx = rng.permutation(n_links)
    hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]

    best_lam, best_err = cfg.odme_lambda_default, np.inf
    for lam in cfg.odme_lambda_grid:
        try:
            flows = _solve_nnls(A[fit_idx], counts[fit_idx], anchor, lam)
            pred_hold = A[hold_idx] @ flows
            err = float(np.sqrt(np.mean((pred_hold - counts[hold_idx]) ** 2)))
        except Exception:
            continue
        if err < best_err:
            best_err, best_lam = err, lam
    return best_lam


def run_task4(release_root: Path, panels: List[str], key: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    odme_rows = key[key["task"] == "odme"].copy()
    if odme_rows.empty:
        return pd.DataFrame(columns=["panel", "departure_time", "path_id", "path_flow"])

    rng = np.random.default_rng(SEED)
    results = []
    for panel in panels:
        network = release_root / "corridors" / panel / "network"
        paths_csv = network / "path_set.csv"
        incidence_csv = network / "path_link_incidence.csv"
        base_od_csv = network / "base_od.csv"
        counts_csv = network / "link_counts.csv"

        panel_rows = odme_rows[odme_rows["panel"] == panel] if "panel" in odme_rows.columns else pd.DataFrame()
        if panel_rows.empty:
            continue

        if not (paths_csv.exists() and incidence_csv.exists() and base_od_csv.exists()):
            log.warning(f"Task 4 network files missing for {panel}; falling back to prior flow.")
            base = safe_read_csv(base_od_csv, dtype={"path_id": str}) if base_od_csv.exists() else None
            if base is not None and "base_flow" in base.columns:
                out = panel_rows[["panel", "departure_time", "path_id"]].merge(
                    base[["path_id", "base_flow"]].rename(columns={"base_flow": "path_flow"}),
                    on="path_id", how="left")
            else:
                out = panel_rows[["panel", "departure_time", "path_id"]].copy()
                out["path_flow"] = 0.0
            results.append(out)
            continue

        try:
            paths = safe_read_csv(paths_csv, dtype={"path_id": str})
            paths["path_id"] = paths["path_id"].astype(str)
            path_ids = paths["path_id"].tolist()
            path_index = {p: i for i, p in enumerate(path_ids)}

            incidence = safe_read_csv(incidence_csv, dtype={"path_id": str, "link_id": str})
            incidence["path_id"] = incidence["path_id"].astype(str)
            incidence["link_id"] = incidence["link_id"].astype(str)
            link_ids = incidence["link_id"].drop_duplicates().tolist()
            link_index = {l: i for i, l in enumerate(link_ids)}

            rr = incidence["link_id"].map(link_index).to_numpy(dtype=np.int64)
            cc = incidence["path_id"].map(path_index).to_numpy(dtype=np.int64)
            A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
            valid = np.isfinite(rr) & np.isfinite(cc)
            A[rr[valid], cc[valid]] = 1.0

            base = safe_read_csv(base_od_csv, dtype={"path_id": str})
            base["path_id"] = base["path_id"].astype(str)
            base = base.set_index("path_id").reindex(path_ids)
            anchor = pd.to_numeric(base.get("base_flow", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy()

            if counts_csv.exists():
                counts_df = safe_read_csv(counts_csv, dtype={"link_id": str})
                counts_df["link_id"] = counts_df["link_id"].astype(str)
                counts_df["count"] = pd.to_numeric(counts_df.get("count", counts_df.get("flow_vph")), errors="coerce")
                counts_map = counts_df.set_index("link_id")["count"].to_dict()
                counts = np.array([counts_map.get(l, np.nan) for l in link_ids], dtype=float)
                prior_loaded = A @ anchor
                counts = np.where(np.isfinite(counts), counts, prior_loaded)
            else:
                counts = A @ anchor

            lam = _select_lambda(A, counts, anchor, cfg, rng)
            flows = _solve_nnls(A, counts, anchor, lam)
            log.info(f"  Task 4 [{panel}]: selected lambda={lam}")

            flow_map = dict(zip(path_ids, flows))
            out = panel_rows[["panel", "departure_time", "path_id"]].copy()
            out["path_flow"] = out["path_id"].map(flow_map).fillna(0.0).clip(lower=0)
            results.append(out)
        except Exception as e:
            log.warning(f"Task 4 solve failed for {panel} ({e}); writing zero flows.")
            out = panel_rows[["panel", "departure_time", "path_id"]].copy()
            out["path_flow"] = 0.0
            results.append(out)

    if not results:
        return pd.DataFrame(columns=["panel", "departure_time", "path_id", "path_flow"])
    return pd.concat(results, ignore_index=True)




# %% [cell 10]
# ==============================================================================
# 6. Submission assembly with coverage diagnostics
# ==============================================================================
def load_submission_key(release_root: Path) -> pd.DataFrame:
    key_file = release_root / "submission_key.csv"
    if not key_file.exists():
        found = list(release_root.glob("**/submission_key.csv"))
        if not found:
            raise FileNotFoundError("submission_key.csv not found anywhere under the dataset root.")
        key_file = found[0]
    key = pd.read_csv(key_file, dtype=str)
    return key


def assemble_submission(key: pd.DataFrame, task1_df: pd.DataFrame, task2_df: pd.DataFrame,
                         task4_df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    out = pd.DataFrame({
        "submission_id": pd.to_numeric(key["submission_id"], errors="coerce").astype("int64"),
        "task": key["task"].astype(str),
    })
    for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[col] = 0.0

    key_ts = pd.to_datetime(key["timestamp"], utc=True, errors="coerce") if "timestamp" in key.columns else None

    # --- Task 1 (state) ---
    n_state = int((key["task"] == "state").sum())
    if not task1_df.empty:
        left = key.loc[key["task"] == "state", ["submission_id", "panel", "station_id", "link_id", "mask_regime"]].copy()
        left["timestamp"] = key_ts.loc[left.index] if key_ts is not None else pd.NaT
        left["submission_id"] = pd.to_numeric(left["submission_id"], errors="coerce").astype("int64")
        merged = left.merge(task1_df, on=["panel", "timestamp", "station_id", "link_id", "mask_regime"], how="left")
        merged = merged.drop_duplicates("submission_id").set_index("submission_id")
        matched = int(merged["speed_kmh"].notna().sum())
        if n_state and matched < 0.98 * n_state:
            log.warning(f"Task 1 join matched only {matched}/{n_state} rows ({matched/max(n_state,1):.1%}); "
                        f"check timestamp/id dtypes if this is unexpectedly low.")
        out.loc[out["task"] == "state", "speed_kmh"] = out.loc[out["task"] == "state", "submission_id"].map(
            merged["speed_kmh"]).fillna(0.0).to_numpy()
        out.loc[out["task"] == "state", "flow_vph"] = out.loc[out["task"] == "state", "submission_id"].map(
            merged["flow_vph"]).fillna(0.0).to_numpy()

    # --- Task 2 (queue) ---
    n_queue = int((key["task"] == "queue").sum())
    if not task2_df.empty:
        left = key.loc[key["task"] == "queue", ["submission_id", "window_id", "link_id"]].copy()
        left["timestamp"] = key_ts.loc[left.index] if key_ts is not None else pd.NaT
        left["submission_id"] = pd.to_numeric(left["submission_id"], errors="coerce").astype("int64")
        merged = left.merge(task2_df, on=["window_id", "timestamp", "link_id"], how="left")
        merged = merged.drop_duplicates("submission_id").set_index("submission_id")
        matched = int(merged["queue_pred"].notna().sum())
        if n_queue and matched < 0.98 * n_queue:
            log.warning(f"Task 2 join matched only {matched}/{n_queue} rows ({matched/max(n_queue,1):.1%}); "
                        f"check timestamp/id dtypes if this is unexpectedly low.")
        out.loc[out["task"] == "queue", "queue_pred"] = out.loc[out["task"] == "queue", "submission_id"].map(
            merged["queue_pred"]).fillna(0).to_numpy()

    # --- Task 4 (odme) ---
    n_odme = int((key["task"] == "odme").sum())
    if not task4_df.empty:
        left = key.loc[key["task"] == "odme", ["submission_id", "panel", "departure_time", "path_id"]].copy()
        left["submission_id"] = pd.to_numeric(left["submission_id"], errors="coerce").astype("int64")
        merged = left.merge(task4_df, on=["panel", "departure_time", "path_id"], how="left")
        merged = merged.drop_duplicates("submission_id").set_index("submission_id")
        matched = int(merged["path_flow"].notna().sum())
        if n_odme and matched < 0.98 * n_odme:
            log.warning(f"Task 4 join matched only {matched}/{n_odme} rows ({matched/max(n_odme,1):.1%}); "
                        f"check timestamp/id dtypes if this is unexpectedly low.")
        out.loc[out["task"] == "odme", "path_flow"] = out.loc[out["task"] == "odme", "submission_id"].map(
            merged["path_flow"]).fillna(0.0).to_numpy()

    out["queue_pred"] = out["queue_pred"].round().clip(0, 1).astype(int)
    out["path_flow"] = out["path_flow"].clip(lower=0)
    out["speed_kmh"] = out["speed_kmh"].clip(lower=0)
    out["flow_vph"] = out["flow_vph"].clip(lower=0)
    out = out.fillna(0.0)

    return out[["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]].sort_values(
        "submission_id").reset_index(drop=True)


def validate_submission(sub: pd.DataFrame, key: pd.DataFrame) -> None:
    expected_rows = len(key)
    assert len(sub) == expected_rows, f"Row count mismatch: expected {expected_rows}, got {len(sub)}"
    assert not sub.isna().any().any(), "Null values present in submission."
    assert sub["submission_id"].is_unique, "Duplicate submission_id values found."
    assert set(sub["queue_pred"].unique()).issubset({0, 1}), "queue_pred must be binary."
    assert (sub["path_flow"] >= 0).all(), "Negative path_flow values found."
    assert np.isfinite(sub[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()).all(), "Non-finite values found."

    zero_state = int(((sub["task"] == "state") & (sub["speed_kmh"] == 0) & (sub["flow_vph"] == 0)).sum())
    n_state = int((sub["task"] == "state").sum())
    if n_state and zero_state / n_state > 0.1:
        log.warning(f"{zero_state}/{n_state} state rows are exactly (0,0) -- likely unmatched joins, "
                    f"not real predictions of zero speed/flow. Investigate before submitting.")

    log.info(f"Validation passed: {len(sub):,} rows, task breakdown -> "
              f"{sub['task'].value_counts().to_dict()}")




# %% [cell 11]
# ==============================================================================
# 7. Orchestration
# ==============================================================================
def main(cfg: Config = CFG) -> pd.DataFrame:
    ensure_dirs(cfg)
    t_start = time.time()
    try:
        with timeit("locate dataset & load manifest"):
            release_root = locate_release_root(cfg)
            panels = load_panels(release_root)
            log.info(f"Dataset root: {release_root}")
            log.info(f"Panels ({len(panels)}): {panels}")

        with timeit("exploratory data analysis"):
            run_eda(release_root, panels, cfg)

        with timeit("load submission_key.csv"):
            key = load_submission_key(release_root)
            log.info(f"submission_key rows: {len(key):,}; tasks: {key['task'].value_counts().to_dict()}")

        with timeit("Task 1 -- state (speed/flow, profile + optional LGBM + tuned FD blend)"):
            task1_df, task1_models = run_task1(release_root, panels, key, cfg)

        with timeit("Task 2 -- queue (ensemble forecast, IoU-tuned)"):
            task2_df = run_task2(release_root, panels, key, cfg)

        with timeit("Task 4 -- ODME path flows (lambda-tuned NNLS)"):
            task4_df = run_task4(release_root, panels, key, cfg)

        with timeit("assemble submission"):
            submission = assemble_submission(key, task1_df, task2_df, task4_df, cfg)

        with timeit("validate submission"):
            validate_submission(submission, key)

        with timeit("write submission.csv"):
            submission.to_csv(cfg.submission_path, index=False)
            size_mb = cfg.submission_path.stat().st_size / (1024 * 1024)
            log.info(f"Submission written to {cfg.submission_path} ({size_mb:.1f} MB)")

        return submission

    except FileNotFoundError as e:
        log.error(f"Dataset access error: {e}")
        raise
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise
    finally:
        log.info(f"Total runtime: {time.time() - t_start:.1f}s")


# %% [cell 12]
submission = main()
submission.head()

