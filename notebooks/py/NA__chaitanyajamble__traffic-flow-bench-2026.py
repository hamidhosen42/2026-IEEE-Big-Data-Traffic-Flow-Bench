# %% [cell 0]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

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
 
LGBM_DEVICE = "cpu"  # set to "gpu" only if a GPU accelerator is attached
 
 
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
    chunk_size: int = 500_000          # rows per chunk for large csv streaming
    slots_per_week: int = 7 * 288      # 5-minute resolution, 7 days
    queue_speed_ratio: float = 0.60    # queued if speed <= ratio * free-flow speed
    queue_horizons_min: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    odme_lambda: float = 15.0          # prior-anchoring strength for NNLS
    min_train_rows_for_lgbm: int = 50_000
 
 
CFG = Config()
CFG.working_dir.mkdir(parents=True, exist_ok=True)
CFG.eda_dir.mkdir(parents=True, exist_ok=True)
 
 
def timeit(label: str):
    """Small context manager for stage timing without extra dependencies."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            log.info(f"START  {label}")
            return self
 
        def __exit__(self, *exc):
            log.info(f"DONE   {label}  ({time.time() - self.t0:.1f}s)")
            return False
    return _Timer()
 
 
# ==============================================================================
# 1. Dataset location and manifest
# ==============================================================================
def locate_release_root(cfg: Config = CFG) -> Path:
    """Find the competition data root robustly across Kaggle path variants."""
    candidates = list(cfg.dataset_root_hints)
    candidates += glob.glob("/kaggle/input/**/kaggle_public", recursive=True)
    candidates += glob.glob("/kaggle/input/**/corridors", recursive=True)
 
    for c in candidates:
        p = Path(c)
        if (p / "config" / "corridors.json").exists() or (p / "corridors").exists():
            return p
        if p.name == "corridors" and p.parent.exists():
            return p.parent
 
    # last resort: search for a known filename anywhere under /kaggle/input
    for p in Path("/kaggle/input").rglob("submission_key.csv"):
        return p.parent
 
    raise FileNotFoundError(
        "Could not locate the competition dataset. Checked: "
        f"{candidates}. Verify the competition data is attached to this "
        "notebook (Add Data) and the path matches /kaggle/input/competitions/"
        "2026-ieee-big-data-traffic-flow-bench."
    )
 
 
def load_panels(release_root: Path) -> List[str]:
    """Return the list of directional panel (corridor) ids."""
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
    """Return (weekday[0-6], time-of-day slot[0-287]) arrays for a 5-min grid."""
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    weekday = ts.dt.weekday.fillna(0).to_numpy(dtype=np.int64)
    tod = ((ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)) // 5).to_numpy(dtype=np.int64)
    tod = np.clip(tod, 0, 287)
    return weekday, tod
 
 
# ==============================================================================
# 2. Exploratory data analysis
# ==============================================================================
def run_eda(release_root: Path, panels: List[str], sample_panels: int = 3) -> None:
  
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
        # cap number of files per panel to keep EDA fast on huge releases
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
    log.info(
        "[EDA] speed_kmh  -- mean {:.1f}, median {:.1f}, p05 {:.1f}, p95 {:.1f}".format(
            eda_df["speed_kmh"].mean(), eda_df["speed_kmh"].median(),
            eda_df["speed_kmh"].quantile(0.05), eda_df["speed_kmh"].quantile(0.95),
        )
    )
    log.info(
        "[EDA] flow_vph   -- mean {:.1f}, median {:.1f}, p05 {:.1f}, p95 {:.1f}".format(
            eda_df["flow_vph"].mean(), eda_df["flow_vph"].median(),
            eda_df["flow_vph"].quantile(0.05), eda_df["flow_vph"].quantile(0.95),
        )
    )
 
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
    out_path = CFG.eda_dir / "eda_overview.png"
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info(f"[EDA] Saved overview figure to {out_path}")
 
    del eda_df, all_rows
    gc.collect()
 
 
# ==============================================================================
# 3. Fundamental-diagram parameters 
# ==============================================================================
def load_fd_parameters(panel_dir: Path) -> pd.DataFrame:
    """
    Returns per-link triangular fundamental diagram parameters:
    link_id, v_f (free-flow speed), k_crit, w (wave speed), k_jam, lanes.
    Falls back to data-driven estimates when fd_parameters.csv is absent
    or missing columns, so downstream steps never depend on this file
    being complete.
    """
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
    """Data-driven fallback: free-flow speed = 95th percentile of eligible speed per link."""
    df = train_df.copy()
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
    df["pct_observed"] = pd.to_numeric(df["pct_observed"], errors="coerce")
    eligible = df[df["pct_observed"].ge(75) & df["speed_kmh"].notna()]
    if eligible.empty:
        return pd.DataFrame(columns=["link_id", "v_f_est"])
    est = eligible.groupby("link_id")["speed_kmh"].quantile(0.95).rename("v_f_est").reset_index()
    return est
 
 
# ==============================================================================
# 4. Task 1 -- historical time-of-week profiles (+ optional LightGBM refinement)
# ==============================================================================
def build_time_of_week_profile(panel_dir: Path, cfg: Config = CFG) -> Optional[dict]:
    """
    Streams every training parquet file for one panel once, accumulating
    per-link, per-(weekday*288+tod-slot) sum/count for speed and flow.
    Memory footprint is O(n_links * 2016) regardless of how many rows /
    files exist, so this scales to arbitrarily large training sets.
    """
    train_dir = panel_dir / "train"
    files = sorted((train_dir / "mainline_states").glob("**/*.parquet"))
    if not files:
        files = sorted((train_dir / "mainline_states_masked").glob("**/*.parquet"))
    if not files:
        log.warning(f"No training state files under {train_dir}; Task 1 will use global fallback for this panel.")
        return None
 
 
    try:
        first_ids = pd.read_parquet(files[0], columns=["link_id"])["link_id"].astype(str).unique()
    except Exception as e:
        log.warning(f"Could not read link ids from {files[0]}: {e}")
        return None
 
    link_ids = sorted(first_ids)
    link_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)
    n_slots = cfg.slots_per_week
 
    speed_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    speed_cnt = np.zeros((n_links, n_slots), dtype=np.int32)
    flow_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    flow_cnt = np.zeros((n_links, n_slots), dtype=np.int32)
 
    all_raw_for_fd = []  
 
    for path in files:
        try:
            df = pd.read_parquet(
                path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"]
            )
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
 
        if len(all_raw_for_fd) < 3:
            all_raw_for_fd.append(df[["link_id", "speed_kmh", "pct_observed"]])
 
        del df
    gc.collect()
 
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_mean = np.where(speed_cnt > 0, speed_sum / np.maximum(speed_cnt, 1), np.nan)
        flow_mean = np.where(flow_cnt > 0, flow_sum / np.maximum(flow_cnt, 1), np.nan)
 
    
    tod_index = np.arange(n_slots) % 288
 
    def collapse_by_tod(mean_mat, cnt_mat):
        out = np.full((n_links, 288), np.nan)
        for t in range(288):
            cols = np.where(tod_index == t)[0]
            sub_sum = np.nansum(np.where(cnt_mat[:, cols] > 0, mean_mat[:, cols] * cnt_mat[:, cols], 0.0), axis=1)
            sub_cnt = np.nansum(cnt_mat[:, cols], axis=1)
            out[:, t] = np.where(sub_cnt > 0, sub_sum / np.maximum(sub_cnt, 1), np.nan)
        return out
 
    speed_by_tod = collapse_by_tod(speed_mean, speed_cnt)
    flow_by_tod = collapse_by_tod(flow_mean, flow_cnt)
 
    panel_speed_by_tod = np.nanmean(speed_by_tod, axis=0)
    panel_flow_by_tod = np.nanmean(flow_by_tod, axis=0)
    panel_speed_scalar = np.nanmean(panel_speed_by_tod) if np.isfinite(np.nanmean(panel_speed_by_tod)) else 60.0
    panel_flow_scalar = np.nanmean(panel_flow_by_tod) if np.isfinite(np.nanmean(panel_flow_by_tod)) else 800.0
 
    fd_fallback = estimate_free_flow_speed(pd.concat(all_raw_for_fd, ignore_index=True)) if all_raw_for_fd else \
        pd.DataFrame(columns=["link_id", "v_f_est"])
 
    return {
        "link_idx": link_idx,
        "speed_mean": speed_mean, "flow_mean": flow_mean,
        "speed_by_tod": speed_by_tod, "flow_by_tod": flow_by_tod,
        "panel_speed_by_tod": panel_speed_by_tod, "panel_flow_by_tod": panel_flow_by_tod,
        "panel_speed_scalar": float(panel_speed_scalar), "panel_flow_scalar": float(panel_flow_scalar),
        "fd_fallback": fd_fallback,
    }
 
 
def profile_lookup(profile: dict, link_id: str, weekday: int, tod: int, kind: str) -> float:
    """Hierarchical lookup with graceful fallback through progressively coarser layers."""
    mean_mat = profile["speed_mean"] if kind == "speed" else profile["flow_mean"]
    by_tod = profile["speed_by_tod"] if kind == "speed" else profile["flow_by_tod"]
    panel_by_tod = profile["panel_speed_by_tod"] if kind == "speed" else profile["panel_flow_by_tod"]
    scalar = profile["panel_speed_scalar"] if kind == "speed" else profile["panel_flow_scalar"]
 
    idx = profile["link_idx"].get(link_id)
    if idx is not None:
        slot = weekday * 288 + tod
        v = mean_mat[idx, slot]
        if np.isfinite(v):
            return float(v)
        v = by_tod[idx, tod]
        if np.isfinite(v):
            return float(v)
    v = panel_by_tod[tod]
    if np.isfinite(v):
        return float(v)
    return float(scalar)
 
 
def clip_to_fundamental_diagram(speed: np.ndarray, flow: np.ndarray, v_f: np.ndarray,
                                 k_crit: np.ndarray, w: np.ndarray, k_jam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nudges (speed, flow) pairs toward the triangular fundamental diagram
    q_FD(k) = v_f * k for k <= k_crit, and w * (k_jam - k) for k > k_crit,
    to reduce Task 3 (S_physics / S_FD) violations without discarding the
    Task 1 estimate: density is derived (k = q / v), the FD-consistent flow
    at that density is computed, and the reported flow is blended 70/30
    toward the FD-consistent value. Rows with missing FD parameters are
    left unchanged.
    """
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
 
    blend = np.where(has_params & np.isfinite(fd_flow), 0.7 * flow + 0.3 * fd_flow, flow)
    out_flow = np.where(has_params, blend, out_flow)
    out_speed = np.clip(speed, 0, np.where(np.isfinite(v_f) & (v_f > 0), v_f * 1.05, speed))
    return out_speed, out_flow
 
 
def fit_lgbm_residual_model(train_df: pd.DataFrame, target_col: str) -> Optional["lgb.LGBMRegressor"]:
    """Optional refinement: gradient-boosted correction on top of the profile baseline."""
    if not HAS_LGBM or len(train_df) < CFG.min_train_rows_for_lgbm:
        return None
    feature_cols = ["tod", "weekday", "profile_pred", "lanes"]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    X = train_df[feature_cols]
    y = train_df[target_col]
    try:
        model = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            device=LGBM_DEVICE, verbosity=-1,
        )
        model.fit(X, y)
        return model
    except Exception as e:
        log.warning(f"LightGBM training failed ({e}); using profile-only predictions.")
        return None
 
 
def run_task1(release_root: Path, panels: List[str], key: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """
    Produces speed_kmh / flow_vph for every submission_key row with task=='state'.
    Returns a DataFrame indexed by the join keys used for final assembly.
    """
    state_rows = key[key["task"] == "state"].copy()
    if state_rows.empty:
        return pd.DataFrame(columns=["panel", "timestamp", "station_id", "link_id", "mask_regime",
                                      "speed_kmh", "flow_vph"])
 
    state_rows["timestamp"] = pd.to_datetime(state_rows["timestamp"], utc=True, errors="coerce")
    results = []
 
    for panel in panels:
        panel_dir = release_root / "corridors" / panel
        rows = state_rows[state_rows["panel"] == panel]
        if rows.empty:
            continue
 
        profile = build_time_of_week_profile(panel_dir, cfg)
        if profile is None:
          
            out = rows[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
            out["speed_kmh"] = 60.0
            out["flow_vph"] = 800.0
            results.append(out)
            continue
 
        weekday, tod = compute_time_slots(rows, "timestamp")
        link_ids = rows["link_id"].astype(str).to_numpy()
 
        speed_pred = np.array([
            profile_lookup(profile, lid, wd, t, "speed")
            for lid, wd, t in zip(link_ids, weekday, tod)
        ])
        flow_pred = np.array([
            profile_lookup(profile, lid, wd, t, "flow")
            for lid, wd, t in zip(link_ids, weekday, tod)
        ])
 
        # Physics-consistency clipping using FD parameters (Task 3 improvement).
        fd = load_fd_parameters(panel_dir)
        if fd.empty and not profile["fd_fallback"].empty:
            fd = profile["fd_fallback"].rename(columns={"v_f_est": "v_f"})
            fd["k_crit"], fd["w"], fd["k_jam"] = np.nan, np.nan, np.nan
 
        if not fd.empty:
            fd_lookup = fd.set_index("link_id")
            v_f = fd_lookup.reindex(link_ids)["v_f"].to_numpy() if "v_f" in fd_lookup.columns else np.full(len(link_ids), np.nan)
            k_crit = fd_lookup.reindex(link_ids)["k_crit"].to_numpy() if "k_crit" in fd_lookup.columns else np.full(len(link_ids), np.nan)
            w = fd_lookup.reindex(link_ids)["w"].to_numpy() if "w" in fd_lookup.columns else np.full(len(link_ids), np.nan)
            k_jam = fd_lookup.reindex(link_ids)["k_jam"].to_numpy() if "k_jam" in fd_lookup.columns else np.full(len(link_ids), np.nan)
            speed_pred, flow_pred = clip_to_fundamental_diagram(speed_pred, flow_pred, v_f, k_crit, w, k_jam)
 
        out = rows[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        out["speed_kmh"] = np.clip(speed_pred, 0, None)
        out["flow_vph"] = np.clip(flow_pred, 0, None)
        results.append(out)
 
        del profile
        gc.collect()
 
    if not results:
        return pd.DataFrame(columns=["panel", "timestamp", "station_id", "link_id", "mask_regime",
                                      "speed_kmh", "flow_vph"])
    return pd.concat(results, ignore_index=True)
 
 
# ==============================================================================
# 5. Task 2 -- queue state
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
 
 
def run_task2(release_root: Path, panels: List[str], key: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """
    For every (window_id, link_id) in the queue template, extrapolates the
    speed trend observed in the last 30 minutes of window_history forward
    to each forecast horizon and applies the queued threshold
    (speed <= 0.60 * free_flow_speed). Falls back to the last-observed
    speed (pure persistence) when there is not enough history to fit a
    trend, and to a link-level historical queue frequency prior when no
    speed observations exist at all for that link in the window.
    """
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
 
    # Free-flow speed per (panel, link) -- build once per panel present in windows.
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
 
    # --- Trend extrapolation, vectorized per window/link group ---
    def last_and_slope(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["speed_kmh"])
        if g.empty:
            return pd.Series({"last_speed": np.nan, "slope_per_min": 0.0})
        tail = g.tail(6)  # ~last 30 minutes at 5-min resolution
        last_speed = tail["speed_kmh"].iloc[-1]
        if len(tail) >= 2:
            t_minutes = (tail["timestamp"] - tail["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
            y = tail["speed_kmh"].to_numpy()
            if np.ptp(t_minutes) > 0:
                slope = np.polyfit(t_minutes, y, 1)[0]
            else:
                slope = 0.0
        else:
            slope = 0.0
        return pd.Series({"last_speed": last_speed, "slope_per_min": slope})
 
    trend = history.groupby(["window_id", "link_id"]).apply(last_and_slope).reset_index()
 
    queue_rows["window_id"] = queue_rows["window_id"].astype(str)
    queue_rows["link_id"] = queue_rows["link_id"].astype(str)
    queue_rows["timestamp"] = pd.to_datetime(queue_rows["timestamp"], utc=True, errors="coerce")
 
    merged = queue_rows.merge(trend, on=["window_id", "link_id"], how="left").reset_index(drop=True)
 
    # Determine horizon offset in minutes relative to each window's origin (min timestamp per window in the queue template).
    origin = merged.groupby("window_id")["timestamp"].transform("min")
    horizon_min = (merged["timestamp"] - origin).dt.total_seconds() / 60.0
 
    merged["free_flow"] = merged["link_id"].map(ff_lookup).fillna(default_ff)
    merged["last_speed"] = merged["last_speed"].fillna(merged["free_flow"] * 0.9)
    merged["slope_per_min"] = merged["slope_per_min"].fillna(0.0)
 

    damp = 1.0 / (1.0 + 0.03 * horizon_min.clip(lower=0))
    forecast_speed = merged["last_speed"] + merged["slope_per_min"] * horizon_min * damp
    forecast_speed = forecast_speed.clip(lower=0, upper=merged["free_flow"] * 1.1)
 
    queue_pred = (forecast_speed <= cfg.queue_speed_ratio * merged["free_flow"]).astype(int)
 
    out = merged[["window_id", "timestamp", "link_id"]].copy()
    out["queue_pred"] = queue_pred.to_numpy()
 
    del history, trend, merged
    gc.collect()
    return out
 
 
# ==============================================================================
# 6. Task 4 -- prior-regularized NNLS origin-destination matrix estimation
# ==============================================================================
def run_task4(release_root: Path, panels: List[str], key: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """
    For each corridor: builds the path-link incidence matrix A, solves
        min || [A; sqrt(lambda) I] f - [counts; sqrt(lambda) * prior] ||^2   s.t. f >= 0
    which anchors the solution toward the released prior (base_od.csv) while
    still fitting the observed link counts, avoiding both degenerate cases
    described in the competition brief (unmodified prior, and counts fit
    with an implausible demand pattern).
    """
    odme_rows = key[key["task"] == "odme"].copy()
    if odme_rows.empty:
        return pd.DataFrame(columns=["panel", "departure_time", "path_id", "path_flow"])
 
    results = []
    for panel in panels:
        network = release_root / "corridors" / panel / "network"
        paths_csv = network / "path_set.csv"
        incidence_csv = network / "path_link_incidence.csv"
        base_od_csv = network / "base_od.csv"
        counts_csv = network / "link_counts.csv"  # observed mainline / ramp counts, if released
 
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
                counts = A @ anchor  # no independent counts released -> anchor is the only signal
 
            aa = np.vstack([A, math.sqrt(cfg.odme_lambda) * np.eye(A.shape[1])])
            bb = np.concatenate([counts, math.sqrt(cfg.odme_lambda) * anchor])
            flows, _ = nnls(aa, bb)
 
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
 
 
# ==============================================================================
# 7. Submission assembly (memory-bounded, streams submission_key.csv)
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
 
 
def assemble_submission(key: pd.DataFrame,
                         task1_df: pd.DataFrame,
                         task2_df: pd.DataFrame,
                         task4_df: pd.DataFrame,
                         cfg: Config = CFG) -> pd.DataFrame:
    """
    Left-joins each task's predictions back onto the full submission_key
    row set so that every required row is present exactly once, with the
    non-applicable value columns for that task left at 0 as specified in
    the competition's submission format.
    """
    out = pd.DataFrame({
        "submission_id": pd.to_numeric(key["submission_id"], errors="coerce").astype("int64"),
        "task": key["task"].astype(str),
    })
    for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[col] = 0.0
 
    key_ts = pd.to_datetime(key["timestamp"], utc=True, errors="coerce") if "timestamp" in key.columns else None
 
    # --- Task 1 (state) ---
    if not task1_df.empty:
        left = key.loc[key["task"] == "state", ["submission_id", "panel", "station_id", "link_id", "mask_regime"]].copy()
        left["timestamp"] = key_ts.loc[left.index] if key_ts is not None else pd.NaT
        left["submission_id"] = pd.to_numeric(left["submission_id"], errors="coerce").astype("int64")
        merged = left.merge(
            task1_df, on=["panel", "timestamp", "station_id", "link_id", "mask_regime"], how="left"
        )
        merged = merged.drop_duplicates("submission_id").set_index("submission_id")
        out.loc[out["task"] == "state", "speed_kmh"] = out.loc[out["task"] == "state", "submission_id"].map(
            merged["speed_kmh"]).fillna(0.0).to_numpy()
        out.loc[out["task"] == "state", "flow_vph"] = out.loc[out["task"] == "state", "submission_id"].map(
            merged["flow_vph"]).fillna(0.0).to_numpy()
 
    # --- Task 2 (queue) ---
    if not task2_df.empty:
        left = key.loc[key["task"] == "queue", ["submission_id", "window_id", "link_id"]].copy()
        left["timestamp"] = key_ts.loc[left.index] if key_ts is not None else pd.NaT
        left["submission_id"] = pd.to_numeric(left["submission_id"], errors="coerce").astype("int64")
        merged = left.merge(task2_df, on=["window_id", "timestamp", "link_id"], how="left")
        merged = merged.drop_duplicates("submission_id").set_index("submission_id")
        out.loc[out["task"] == "queue", "queue_pred"] = out.loc[out["task"] == "queue", "submission_id"].map(
            merged["queue_pred"]).fillna(0).to_numpy()
 
    # --- Task 4 (odme) ---
    if not task4_df.empty:
        left = key.loc[key["task"] == "odme", ["submission_id", "panel", "departure_time", "path_id"]].copy()
        left["submission_id"] = pd.to_numeric(left["submission_id"], errors="coerce").astype("int64")
        merged = left.merge(task4_df, on=["panel", "departure_time", "path_id"], how="left")
        merged = merged.drop_duplicates("submission_id").set_index("submission_id")
        out.loc[out["task"] == "odme", "path_flow"] = out.loc[out["task"] == "odme", "submission_id"].map(
            merged["path_flow"]).fillna(0.0).to_numpy()
 
    # Enforce required per-task ranges / types from the submission spec.
    out["queue_pred"] = out["queue_pred"].round().clip(0, 1).astype(int)
    out["path_flow"] = out["path_flow"].clip(lower=0)
    out["speed_kmh"] = out["speed_kmh"].clip(lower=0)
    out["flow_vph"] = out["flow_vph"].clip(lower=0)
    out = out.fillna(0.0)
 
    return out[["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]].sort_values(
        "submission_id").reset_index(drop=True)
 
 
def validate_submission(sub: pd.DataFrame, key: pd.DataFrame) -> None:
    """Fails loudly (before upload) rather than letting the scorer reject a malformed file."""
    expected_rows = len(key)
    assert len(sub) == expected_rows, f"Row count mismatch: expected {expected_rows}, got {len(sub)}"
    assert not sub.isna().any().any(), "Null values present in submission."
    assert sub["submission_id"].is_unique, "Duplicate submission_id values found."
    assert set(sub["queue_pred"].unique()).issubset({0, 1}), "queue_pred must be binary."
    assert (sub["path_flow"] >= 0).all(), "Negative path_flow values found."
    assert np.isfinite(sub[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()).all(), "Non-finite values found."
    log.info(f"Validation passed: {len(sub):,} rows, task breakdown -> "
              f"{sub['task'].value_counts().to_dict()}")
 
 
# ==============================================================================
# 8. Orchestration
# ==============================================================================
def main() -> None:
    t_start = time.time()
    try:
        with timeit("locate dataset & load manifest"):
            release_root = locate_release_root()
            panels = load_panels(release_root)
            log.info(f"Dataset root: {release_root}")
            log.info(f"Panels ({len(panels)}): {panels}")
 
        with timeit("exploratory data analysis"):
            run_eda(release_root, panels)
 
        with timeit("load submission_key.csv"):
            key = load_submission_key(release_root)
            log.info(f"submission_key rows: {len(key):,}; tasks: {key['task'].value_counts().to_dict()}")
 
        with timeit("Task 1 -- state (speed/flow)"):
            task1_df = run_task1(release_root, panels, key)
 
        with timeit("Task 2 -- queue"):
            task2_df = run_task2(release_root, panels, key)
 
        with timeit("Task 4 -- ODME path flows"):
            task4_df = run_task4(release_root, panels, key)
 
        with timeit("assemble submission"):
            submission = assemble_submission(key, task1_df, task2_df, task4_df)
 
        with timeit("validate submission"):
            validate_submission(submission, key)
 
        with timeit("write submission.csv"):
            submission.to_csv(CFG.submission_path, index=False)
            size_mb = CFG.submission_path.stat().st_size / (1024 * 1024)
            log.info(f"Submission written to {CFG.submission_path} ({size_mb:.1f} MB)")
 
    except FileNotFoundError as e:
        log.error(f"Dataset access error: {e}")
        raise
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise
    finally:
        log.info(f"Total runtime: {time.time() - t_start:.1f}s")
 
 
if __name__ == "__main__":
    main()
 
