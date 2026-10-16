# Capacity scenario (H1) — 2026-10-11

Status: IN PROGRESS. Sections are filled as results come in.

## 0. PRE-REGISTRATION (written 2026-10-11 01:35 +06, after the H1 data test (section 1), before any model result)

H1 itself (a capacity factor of about 1.2 or 1/1.2 between train and validation/private) is **rejected by the data**
(section 1). So no capacity rescaling (a), x1.2 augmentation (b) or recalibrated FD constant (c) is tested: there is no factor to apply.

What the data does show is a **related scaling**: per-ramp demand is re-drawn in the new scenario.
- Per-link ramp ON/OFF means move by factors of 0.2–2.7. 82–90 % of ramps change by more than 15 %, against 0–6 % for Feb vs the other train months.
- Mainline flow stays within ±3 %.

The production ongoing model uses 5 ramp columns (`ramp_on3`, `ramp_off3`, `ramp_on_surge`, `ramp_on_pm3`, `ramp_off_pm3`), all scaled by link capacity. Their per-link level is therefore scenario-specific. The production onset model is already NORAMP.

**Correction arms (ongoing, lab np recipe = og_np_base: F03 rows, 255 leaves, 500 trees, lr 0.05, folds 0 and 1):**
- `og_np_noramp`: the 5 ramp columns dropped.
- `og_np_ramprel`: the 5 ramp columns replaced by one scenario-invariant ratio, `rel_on = on3 / on12`, i.e. the recent on-ramp flow relative to the same ramp's flow 4–60 min earlier in the same window. It is computed as `ramp_on3 / (ramp_on3 − ramp_on_surge)`, NaN when the denominator ≤ 1e-6, and clipped to [0, 5].

**Gates (binding, vs og_np_base, fold 0+1 mean probabilities):**
- Ongoing: FM-PM validation ≥ +0.0015 LB **and** FM-PM private ≥ +0.0005 LB.
  - FM-PM settings as in TRANSFER_ANALYSIS §5.0: panel-exact, ±2 h, all-cell predicted counts, pseudo-March offlike_m + runstart_m lab pairs.
  - Footprint = decision changes between the two lab models on the 40 + 40 official ongoing windows (lab inputs, all-train prior).
- Reported, not binding: train offlike / runstart OOF (folds 0, 1); pseudo-March offlike_m / runstart_m with CI and panels positive.
- Onset: no arm. Production onset already excludes ramp columns, and no capacity scaling exists to correct. The gate (pseudo-March ≥ +0.01 IoU, CI > 0, official-40 sign not negative) is recorded but unused.
- T1: no FD-constant arm, because the bottleneck discharge equals fd capacity in every split (section 1.3). Gate J ≥ +0.0003, 7/8 recorded but unused.

## 1. H1 data test: REJECTED

Data: released masked observations only (`data/t2local/t2data_<P>.npz`).
- Train is subsampled to the March/April visibility wherever a statistic depends on sample size.
- Scripts: `analysis/t2lab/capscen_stats.py`, `capscen_links.py`, `capscen_heads.py`, `capscen_ramps.py`.
- Logs: `analysis/t2lab/logs/capscen_*.log`. Tables: `analysis/t2lab/results/capscen/` (`stats_panel_month.csv`, `tod_share.csv`, `links.csv`, `heads.csv`, `heads_links.csv`, `ramps.csv`).

### 1.1 High flow quantiles vs fd capacity (per panel; train-month median, then the MAR / APR ratio to it)

| panel | q99 lane flow / cap, train median | MAR / train | APR / train | fitted capacity (median of the top 1 %) MAR / APR vs train |
|---|---:|---:|---:|---|
| D12_I5_N | 0.770 | 0.95 | 0.96 | 0.97 / 0.97 |
| D12_I5_S | 0.789 | 0.95 | 0.94 | 0.95 / 0.94 |
| D7_I10_E | 0.765 | 0.98 | 1.00 | 0.98 / 1.00 |
| D7_I10_W | 0.792 | 0.98 | 1.00 | 0.98 / 1.00 |
| D7_I210_E | 0.656 | 0.98 | 0.95 | 0.99 / 0.96 |
| D7_I210_W | 0.649 | 0.98 | 0.97 | 0.99 / 0.97 |
| D7_I405_N | 0.962 | 1.00 | 0.98 | 1.00 / 0.99 |
| D7_I405_S | 0.773 | 1.01 | 1.05 | 1.03 / 1.06 |

Per link (1,450 links), the q99.5 lane-flow ratio:

| comparison | median | q05–q95 | links in [1.15, 1.25] | links in [0.80, 0.87] |
|---|---:|---|---:|---:|
| MAR / train | 0.998 | 0.957–1.031 | 0.5 % | 1.4 % |
| APR / train | 0.993 | 0.907–1.049 | – | – |
| Feb / other train months (null) | 0.998 | 0.978–1.008 | – | – |

There is **no cluster at 1.2 or at 1/1.2**. The train months are flat (spread ±1 %), so train does not "already include" a different factor from the rest.

### 1.2 Speed–flow / flow–density relation

- Free-flow speed / vf: 0.998–1.002 in every split.
- Congested branch (lane flow / cap against density / per-lane kcrit, queued cells):
  - slope −0.15..−0.21, identical in train, MAR and APR (e.g. D7_I405_N −0.183 / −0.182 / −0.181);
  - intercept-implied jam density 5.5–7.5 kcrit, equal within the train-month range except D12_I5_S (+6 %).

### 1.3 Breakdown and bottleneck discharge (the decisive test)

**Discharge** = flow on the free link just downstream of a queue head, divided by *its* fd capacity:
- q90: 1.00–1.03 in **every** split and panel. In train, MAR and APR the simulator discharges at the published fd capacity, not at 1.2× or 1/1.2× of it.
- Per bottleneck, the discharge is identical across splits: e.g. D12_I5_S link 39: 0.500 / 0.509 / 0.500; D7_I10_E link 81: 0.438 / 0.437 / 0.441; D12_I5_N link 239: 0.785 / 0.792 / 0.793.

**Flow just before breakdown** (t−1, free → queued at t, / cap):
- MAR / train median 0.85–1.05; APR 0.91–1.04;
- all inside or near the train-month range (train-month ranges are wide, e.g. 0.52–0.72 on D7_I210_E).

### 1.4 Queued share by time of day (pooled over 8 panels, % of observed cells)

| hour | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6.5 | 11.7 | 16.2 | 18.4 | 17.5 | 14.6 | 9.4 |
| MAR | 6.4 | 10.0 | 14.1 | 15.5 | 14.8 | 12.2 | 8.6 |
| APR | 7.3 | 11.2 | 14.7 | 15.9 | 14.8 | 12.0 | 7.8 |

- Timing is unchanged: the flow peak slot is within ±1–2 slots, and the queue peak slot is unchanged pooled.
- The level is panel-specific, outside the train-month range on 3 panels (15–18 h share):
  - D12_I5_N: 0.099 MAR / 0.132 APR vs train 0.152–0.250;
  - D7_I210_W: 0.188 / 0.147 vs 0.218–0.241;
  - D7_I405_S: 0.221 / 0.223 vs 0.148–0.189.
- Demand is unchanged. Mean v/c (all slots, AM, PM) changes by −4..+4 %, and the per-link PM mainline flow ratio has median 0.985–0.997.

**Verdict on H1: rejected.**
- Relative to train: every capacity-type statistic is ≈1.00, and the 3–6 % panel-level moves are not 1.2.
- Relative to fd_parameters: bottleneck discharge equals fd capacity in all splits.
- `capacity_multiplier 1.2` is most likely already baked into the synthetic fd capacities. The manifest lists "FD capacity parameters" under `replaced_with_synthetic`, and observed flows on ordinary links top out at about 0.65–0.95 × cap. In any case, it applies to train and to validation/private alike.
- A train-trained model, our FD features and our recon constants are **not** miscalibrated by a capacity factor.
