# %% [md 0]
# # TrafficFlowBench — 0.6970 LB writeup
# 
# **Scores:** baseline reproduction 0.5534 → structural-KF state + near-cut queue trigger 0.6638 → abrupt-onset queue rules 0.6970 → hybrid rule+LightGBM queues **0.7269** (public LB).
# Everything runs on CPU (numpy/pandas/scipy/lightgbm only, no DL). Code cells are condensed but faithful to the submitted pipeline.
# 
# | Task | Weight | Method | Local check |
# |---|---|---|---|
# | 1 state | 0.35 | weekday×time-of-day profile + vectorized local-level **Kalman filter / RTS smoother** on same-day deviations | train macro S_state 0.90 vs 0.73 historical mean |
# | 2 queue | 0.30 | **hybrid**: onset = recurrent-bottleneck Top-K at T+30 with flow/capacity gate; ongoing = LightGBM per-(link, step) classifier (window-level CV IoU 0.54) | official train windows |
# | 3 physics | 0.15 | scored on Task 1 values; soft-FD projection + conservation post-processing **evaluated and dropped** (see below) | S_FD ≈ 0.99, S_LWR not locally reproducible |
# | 4 odme | 0.20 | NNLS against the **scored split's own** released link counts, anchored on the weak prior | equals official `public_reference` to <1e-8 |

# %% [md 1]
# ## Task 1 — profile + Kalman-smoothed same-day deviations
# 
# Following the official structural-KF teaching baseline (Zhou & Mahmassani 2007): a historical **weekday × 288-slot (5-min)** profile
# carries the regular pattern; the *same-day deviations* (observation − profile) are smoothed by a **local-level Kalman filter with an
# RTS backward pass**, per link. The official scalar implementation vectorizes over all links, so a panel-day takes milliseconds.
# 
# Two details matter:
# - the filter runs on the deviation field indexed by **time-of-day** (288 rows), not raw time — the day is one "series of slots";
# - process/measurement variances scale with each link's own deviation std: `q = (0.35·σ)²`, `r = (0.8·σ)²`.
# Train-side scoring against the released unmasked layer gives macro **S_state = 0.899** (historical mean alone: 0.732); every one of the
# 10 panels improves, worst case +0.04 (D7_I210_E, where same-day signal is weak).

# %% [cell 2]
import numpy as np, pandas as pd
from pathlib import Path

REL = Path("/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public")

def kalman_smooth_rows(y, q, r):
    """Local-level Kalman filter + RTS smoother, vectorized over rows.
    y: (L, T) observations, NaN = missing. q, r: (L,) variances."""
    L, T = y.shape
    big = 1e12
    has0 = np.isfinite(y[:, 0])
    x = np.where(has0, y[:, 0], 0.0); p = np.where(has0, r, big)
    x_f = np.zeros((L, T)); p_f = np.zeros((L, T)); x_p = np.zeros((L, T)); p_p = np.zeros((L, T))
    finite = np.isfinite(y)
    for t in range(T):
        if t > 0: p = p + q
        x_p[:, t], p_p[:, t] = x, p
        obs = finite[:, t]
        k = p / (p + r)
        x = np.where(obs, x + k * (y[:, t] - x), x)
        p = np.where(obs, (1.0 - k) * p, p)
        x_f[:, t], p_f[:, t] = x, p
    x_s = x_f.copy()
    for t in range(T - 2, -1, -1):
        g = p_f[:, t] / p_p[:, t + 1]
        x_s[:, t] = x_f[:, t] + g * (x_s[:, t + 1] - x_p[:, t + 1])
    return x_s

def smooth_deviations(dev):
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(dev, axis=1)
    sd = np.where(np.isfinite(sd), np.maximum(sd, 1e-3), 1e-3)
    return kalman_smooth_rows(dev, (0.35 * sd) ** 2, (0.8 * sd) ** 2)

def build_profile(panel, panel_dir):
    """weekday x 288 means over clean train cells (pct_observed>=75), plus per-link fallback."""
    import glob
    S = {}; C = {}
    link_ids, link_index = None, None
    for path in sorted(glob.glob(str(panel_dir / "train/mainline_states/**/*.parquet"), recursive=True)):
        f = pd.read_parquet(path, columns=["timestamp","link_id","speed_kmh","flow_vph","pct_observed","is_score_eligible"])
        f = f[f.is_score_eligible.astype(bool) & (pd.to_numeric(f.pct_observed, errors="coerce") >= 75)]
        f["link_id"] = f.link_id.astype(str)
        if link_ids is None:
            link_ids = sorted(f.link_id.unique()); link_index = {l: i for i, l in enumerate(link_ids)}
        li = f.link_id.map(link_index).to_numpy()
        ts = pd.to_datetime(f.timestamp, utc=True)
        slot = (ts.dt.weekday * 288 + ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        for ch in ("speed_kmh", "flow_vph"):
            v = pd.to_numeric(f[ch], errors="coerce").to_numpy()
            tot = S.setdefault(ch, np.zeros((len(link_ids), 2016)))
            cnt = C.setdefault(ch, np.zeros((len(link_ids), 2016), dtype=np.int64))
            np.add.at(tot, (li, slot), np.nan_to_num(v))
            np.add.at(cnt, (li, slot), np.isfinite(v))
    prof = {}
    for ch, tot in S.items():
        cnt = C[ch]
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        fb = np.nanmean(np.where(cnt > 0, mean, np.nan), axis=1)
        fb = np.where(np.isfinite(fb), fb, np.nanmean(tot) / max(np.nanmean(cnt), 1))
        prof[ch] = np.where(cnt > 0, mean, fb[:, None])
    return {"link_ids": link_ids, "link_index": link_index,
            "speed": prof["speed_kmh"], "flow": prof["flow_vph"]}, C

# %% [cell 3]
def predict_day(profile, counts, frame):
    """Profile + Kalman-smoothed same-day deviation for one masked day."""
    link_index = profile["link_index"]
    frame = frame.copy(); frame["link_id"] = frame.link_id.astype(str)
    li = frame.link_id.map(link_index).fillna(-1).to_numpy(np.int64)
    known = li >= 0; safe_li = np.where(known, li, 0)
    ts = pd.to_datetime(frame.timestamp, utc=True)
    wslot = (ts.dt.weekday * 288 + ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int64)
    tod = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int64)
    base_s = profile["speed"][safe_li, wslot]; base_f = profile["flow"][safe_li, wslot]
    out_s, out_f = base_s.copy(), base_f.copy()
    for ch, base, out in (("speed_kmh", base_s, out_s), ("flow_vph", base_f, out_f)):
        val = pd.to_numeric(frame[ch], errors="coerce").to_numpy()
        obs = known & frame.is_score_eligible.astype(bool).to_numpy() & np.isfinite(val) & np.isfinite(base)
        n_links = len(profile["link_ids"]); total = np.zeros((n_links, 288)); n = np.zeros((n_links, 288), np.int64)
        np.add.at(total, (li[obs], tod[obs]), (val - base)[obs]); np.add.at(n, (li[obs], tod[obs]), 1)
        dev = np.full((n_links, 288), np.nan); np.divide(total, n, out=dev, where=n > 0)
        sm = smooth_deviations(dev)
        pred = np.clip(base + sm[safe_li, tod], 0.0, None)
        out[known] = pred[known]
    return out_s, out_f

# %% [md 4]
# ## Task 2 — what the train windows actually look like
# 
# The release ships **official train windows** (`task2/<panel>/train/`) built by the same window builder as the scored splits, and the
# train split has unmasked state — a perfect tuning ground truth. Three measured facts drive the rule:
# 
# 1. **Queue tails do not propagate.** Median upstream propagation of the queued front between 5-min cells is ≈ 0 km/h:
#    congestion forms abruptly and holds its ground. Upstream BFS expansion of the queued map is pure false positives.
# 2. **Onset has no speed precursor — the truth only exists at T+30.** In all 40 train `queue_onset` windows the origin has zero queued
#    links, the links that queue up within the horizon sit **40–58 km/h above the cut** at the origin (no margin rule can see them),
#    and queued cells appear **only in the final horizon step**. Predicting 1 before T+30 is guaranteed FP.
# 3. **Over-capacity without a queue predicts an empty horizon.** Windows whose max sensored flow/capacity at the origin is ≥ ~1.0
#    never develop a queue in the horizon (separation 0.994 vs 1.003 on train). This gates off the recurrent-bottleneck guess.
# 
# **Ongoing** = repeat the origin queued map (sensored links only — that is all the history shows), then fill non-sensored links lying
# between two queued sensored links (their truth is usually queued too; this is worth a lot on corridor-wide queues).
# **Onset** = the Top-3 recurrent-bottleneck links for that hour-of-day, fired at T+30 only, unless the flow/capacity gate says "nothing forms".

# %% [cell 5]
HORIZON = 6
HIST_CELLS = 12

def mine_bottlenecks(release, panels):
    """freq[link, hour] = fraction of train days with >=1 queued eligible cell in that hour."""
    import glob
    tables = {}
    for panel in panels:
        net_links = None
        topo = pd.read_csv(release/"corridors"/panel/"network"/"lwr_mainline_topology.csv", dtype={"link_id": str})
        topo = topo[topo.link_id.str.match(r"^(?!CONN)")].sort_values("order_index")
        link_ids = topo.link_id.tolist(); v_cut = 0.60 * pd.to_numeric(topo.free_speed_kmh, errors="coerce").fillna(105.0).to_numpy()
        sensored = topo.has_sensor.eq(1).to_numpy(); idx = {l: i for i, l in enumerate(link_ids)}
        x = (pd.to_numeric(topo.length_km, errors="coerce").fillna(0.5).cumsum() - 0.5 * pd.to_numeric(topo.length_km, errors="coerce").fillna(0.5)).to_numpy()
        freq = np.zeros((len(link_ids), 24)); n_days = 0
        for path in sorted(glob.glob(str(release/"corridors"/panel/"train/mainline_states/**/*.parquet"), recursive=True)):
            f = pd.read_parquet(path, columns=["timestamp","link_id","speed_kmh","is_score_eligible"])
            f["link_id"] = f.link_id.astype(str)
            li = f.link_id.map(idx)
            ok = li.notna().to_numpy() & f.is_score_eligible.astype(bool).to_numpy() & f.speed_kmh.notna().to_numpy()
            li = li.fillna(-1).to_numpy(np.int64)
            q = ok & (f.speed_kmh.to_numpy() <= v_cut[li])
            tod = (pd.to_datetime(f.timestamp, utc=True).dt.hour).to_numpy()[q]
            hit = np.zeros((24, len(link_ids)), bool)
            for h in range(24): hit[h] = np.bincount(li[q][tod == h], minlength=len(link_ids)) > 0
            freq += hit.T; n_days += 1
        tables[panel] = {"links": link_ids, "freq": freq / max(n_days, 1), "sensored": sensored,
                         "v_cut": v_cut, "index": idx, "x": x}
    return tables

# %% [cell 6]
def predict_window(S, E, F, table, cap, hour, topk=3, gate_ratio=1.01):
    """S,E,F: (12, L) history matrices (speed, eligible, flow). Returns (L, 6) bool prediction."""
    L = len(table["links"]); v = table["v_cut"]
    valid_last = E[-1] & np.isfinite(S[-1])
    queued_now = valid_last & (S[-1] <= v)
    if queued_now.any():                              # ongoing: persistence
        pred = np.repeat(queued_now[:, None], HORIZON, axis=1)
    else:                                             # onset
        if F is not None and np.isfinite(F[-1]).any():
            ratio = np.where(valid_last, F[-1], np.nan) / np.where(valid_last, cap, np.nan)
            if np.nanmax(ratio) >= gate_ratio:        # over-capacity, no queue -> nothing forms
                return np.zeros((L, HORIZON), bool)
        freq = table["freq"][:, hour]
        cands = np.where(table["sensored"] & (freq > 0))[0]
        cands = cands[np.argsort(-freq[cands])][:topk]
        pred = np.zeros((L, HORIZON), bool)
        pred[cands, HORIZON - 1] = True               # fire at T+30 only
    # fill non-sensored links between two queued sensored links (corridor order)
    order = np.argsort(table["x"])
    for step in range(HORIZON):
        col = pred[:, step][order]; sens = table["sensored"][order]
        hot = col & sens
        if hot.sum() < 2: continue
        ranks = np.where(hot)[0]
        for a, b in zip(ranks[:-1], ranks[1:]):
            if b - a > 1:
                col[a + 1:b] |= ~sens[a + 1:b]
        pred[:, step] = col[np.argsort(order)]
    return pred

# %% [md 7]
# ## Task 2 (part 2) — LightGBM for ongoing windows: rules and trees are complementary
# 
# A per-(link, step) **LightGBM classifier** trained on the 80 official train windows (87k rows, 8% positive) with only
# released-history features (speed statistics, margins, deceleration, history queue count, flow/capacity, bottleneck
# frequencies, position, hour, step) tells a clean story under 5-fold **window-level** CV:
# 
# | predictor | onset IoU | ongoing IoU | overall |
# |---|---|---|---|
# | rule (this notebook) | **0.440** | 0.442 | 0.441 |
# | LightGBM @ p≥0.3 | 0.322 | **0.638** | 0.480 |
# | **hybrid: rule for onset, ML for ongoing** | **0.440** | **0.638** | **0.539** |
# 
# The abrupt onset formation is best caught by the recurrent-bottleneck rule (the model cannot learn a signal that is not
# in the features), while the *evolution* of an existing queue (growth/clearance per step) is exactly what gradient
# boosting picks up from the history statistics. The window `condition` is published, so the hybrid switch is legal at
# inference time. Feature gains are dominated by `queued_hist`, `v_last`, `margin_last`, then `pos`/`freq_hour`/
# `window_ratio_max`.

# %% [md 8]
# ## Task 3 — why we dropped the physics post-processing
# 
# `S_physics = (S_FD + 2·S_LWR)/3` is scored on the Task 1 values. Locally, with the public release, **S_FD ≈ 0.99** out of the box
# (our KF flows already sit on the triangular FD). **S_LWR is not locally reproducible**: without the organizer-held boundary-flux file,
# the scorer derives boundary flows from topology applied to the participant's own flows, and its own comments note this puts even the
# *exact truth* at S_LWR = 0 — we verified E_LWR ≈ 1.001 with corr(dN, rhs) ≈ 0.000 on released cells. Soft-FD projection + inter-link
# conservation smoothing (the 0.809-LB recipe) moved our local S_physics by +0.0001 only, so it was dropped rather than shipped blind.
# If the server scores with the organizer flux, accuracy alone is the lever; if it runs the same fallback, S_LWR ≈ 0 for everyone and
# S_FD is already saturated. Either way the post-processing had no measurable upside.

# %% [md 9]
# ## Task 4 — NNLS on the scored split's own counts
# 
# Each scored split publishes `task4/<panel>/<split>/synthetic_link_counts.csv` — the count vector the task is actually posed on
# (v1 fitted path flows on *train-month* counts; that is the single biggest ODME mistake to avoid). Solving
# `min ‖A·f − c‖²` over observed links only, with the released weak prior as anchor, reproduces the official `public_reference`
# path flows to <1e-8. We use the official builder's solver:
# 
# ```
# flows = builder.solve(A[measured], counts[measured], base_prior)
# ```
# 
# and **never** mix in `network/base_od.csv` (that is the private-month table; the docs warn against exactly this).

# %% [md 10]
# ## Merge + validation
# 
# The three task CSVs are joined against `submission_key.csv` with the official `merge_submissions.py`, then checked:
# row count 6,980,503, template exact match, all values finite & non-negative, queue cells ∈ {0,1}, every natural key covered.
# Only template (`sample_submission_state`) cells are emitted for Task 1 — target sets come from the release, not from hunting NaNs.
# 
# ## Lessons
# 
# 1. **Mine the release before modelling.** The train task-2 windows + unmasked train state answered every design question
#    empirically (no speed precursor, T+30-only truth, the flow gate) — no blind rule tuning against the 5/day submission budget.
# 2. **Reproduce the scorer locally before optimizing for it.** The physics post-processing that helped others was invisible under
#    the local S_LWR fallback; measuring first saved a wasted submission.
# 3. **Persistence is a strong ongoing baseline** — every addition must beat it on the official train windows, not on ad-hoc proxies.
# 4. Task 1 accuracy (0.35 weight) is the biggest remaining lever: spatial blending into the KF and regime-aware q/r tuning are next.
