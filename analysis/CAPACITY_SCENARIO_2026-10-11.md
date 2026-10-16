# Capacity scenario (H1) — 2026-10-11

Status: **DONE** (resumed and finished on 2026-10-16 after the 10-11 reboot). Analysis and lab only; nothing was pushed or submitted.

**Bottom line**
1. **H1 is rejected.** No ×1.2 (or 1/1.2) capacity factor separates train from validation/private. Bottleneck discharge equals the published fd capacity in all three splits, and per-bottleneck discharge is identical across splits.
2. **The real scenario differences:**
   - a per-ramp demand re-draw (0.2–2.7×), with mainline flow within ±3 %;
   - a different set of active bottlenecks (location), with unchanged capacities;
   - no change in demand level or peak timing.
3. **The pre-registered ongoing ramp corrections (noramp, ramprel) FAIL every gate.** FM-PM is −0.0004 on validation and −0.0008 on private, and pseudo-March is negative. The ramp columns still help in the new scenario. No kernel was written.
4. **Public-LB selection bias** (section 4): a team with 48–100 submissions can plausibly sit +0.005..+0.02 above its true level. Our own 0.88198 is not bias-free either.

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
- q90: 1.00–1.03 in 20 of 24 panel × split cells (train-month median, MAR, APR). The exceptions are D7_I10_W at 0.94–0.98 in all three splits (few queues) and D7_I10_E MAR at 0.90. In train, MAR and APR the simulator discharges at the published fd capacity, not at 1.2× or 1/1.2× of it.
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

## 2. Other systematic train → March/April differences (≤ T data could see them; data only)

| difference | evidence | correctable with ≤ T data? |
|---|---|---|
| **Ramp demand re-drawn per ramp** | Per-link ramp ON/OFF mean, AM and PM: MAR/train q05–q95 0.16–2.7, median 0.78–0.91; 82–90 % of ramps move by more than 15 %. Null: Feb vs the other train months, 0–6 %. Mainline flow at the same links: q05–q95 0.93–1.04 (`capscen_ramps.py`) | It is a pure input shift for every ramp-level feature. Ramps carry about 1 % of mainline flow and do not explain where queues move (corr(Δ downstream on-ramp, Δ queued share) −0.40..+0.39, sign-inconsistent). Tested in section 3 |
| **Bottleneck activation set** | Discharge is identical at shared bottlenecks (1.3), but some train bottlenecks switch off and new ones switch on. Off: D12_I5_N link 114, 3,319 head cells in train, 0 in MAR/APR (17 % → 0 % queued share on links 111–114 at an unchanged flow of 5,944 / 5,753 / 5,930 vph). On: D12_I5_S links 91/95, 0 → 8–25 %; D7_I10_W link 20 (MAR only); D7_I10_E links 19/23 active in train + APR but not MAR | This is the scenario-specific location effect. Only same-scenario history (earlier March/April days) identifies it, which is the sp4 / pv lever already in use. Same-day: of the 10 % of new-scenario onsets at non-train-recurrent locations, only 15 % had an earlier queue on the same day at that link ±1 (`capscen_onsetloc.py`) |
| **Onset location is mostly at train-recurrent places** | 90 % of MAR/APR onset events lie within ±1 link / ±30 min of a cell with train prior ≥ 0.05 (train itself: 97 %). Previous ≤ 4 same-split days had a queue there 84–89 % of the time (train: 87 %) | The location loss is a choice among known candidates, not unseen places. Same-month recency (prior blend) is the right tool; there is no capacity fix |
| **"Incident-like" queues** | Days with a ≥ 30-min queued run on a link whose train prior is ≤ 0.02: train median 4–10 %. MAR/APR: D12_I5_S 94 % / 97 %, D7_I405_S 87 % / 87 %, D7_I10_E 48 % / 53 %; 0.3–13× on the other panels | Mostly the new recurrent bottlenecks above, not 5–7 incidents. Correctable only via same-split recurrence |
| **Queue level** | Pooled PM queued share −15 % (16 h: 18.4 → 15.5 / 15.9). Outside the train-month range: D12_I5_N (lower), D7_I210_W (lower), D7_I405_S (higher) | Partly via same-split priors. A global level shift is not supported |
| **Demand level, peak timing** | v/c −4..+4 %; flow-peak slot ±1–2; queue-peak slot unchanged pooled | Nothing to correct |

## 3. Pre-registered ramp-correction arms: FAIL (both)

- Code: `analysis/t2lab/capscen_ramp_arm.py` (arms) and `capscen_ramp_report.py` (gates).
- Logs: `logs/capscen_ramp_{noramp,ramprel,report}.log`.
- Outputs:
  - `data/t2local/cache/ogfeat/arms/og_np_{noramp,ramprel}/` (fold pickles, pm_f0/1, off_f0/1 with the base-model official probabilities);
  - `results/capscen/ramp_fm.csv` and `official_footprint_*.csv`.
- Reference: og_np_base (fold 0 + 1, existing models).

| gate / set | noramp | ramprel |
|---|---:|---:|
| train official-40 (fold 0) | −0.0046 [−0.015, +0.005] 1/8 | −0.0079 [−0.019, +0.002] 2/8 |
| train offlike (1,473, folds 0+1) | −0.0053 [−0.0074, −0.0032] 1/8 | −0.0061 [−0.0090, −0.0037] 1/8 |
| train pseudo (1,055) | −0.0067 [−0.0097, −0.0036] 0/8 | −0.0070 [−0.0118, −0.0020] 3/8 |
| **pseudo-March offlike_m (298)** | **−0.0059 [−0.0113, −0.0004] 3/8** | **−0.0105 [−0.0165, −0.0044] 1/8** |
| pseudo-March runstart_m (1,124) | −0.0020 [−0.0053, +0.0014] 3/8 | −0.0007 [−0.0039, +0.0026] 4/8 |
| official footprint, validation: windows / cells / Δn | 28 / 155 / +7 | 30 / 166 / +18 |
| official footprint, private: windows / cells / Δn | 31 / 172 / +28 | 31 / 168 / +28 |
| **FM-PM validation (gate ≥ +0.0015)** | **−0.00039** | **−0.00039** |
| **FM-PM private (gate ≥ +0.0005)** | **−0.00080** | **−0.00079** |
| decision | **FAIL** | **FAIL** |

Interpretation:
- The ramp columns' *temporal* content (recent surge relative to the ramp's own level, and the neighbour sums) still carries signal in the new scenario. Dropping them, or reducing them to a within-window ratio, loses 0.006–0.011 IoU on pseudo-March offlike_m.
- The per-ramp level shift does not cost the production ongoing model anything measurable. The earlier "+ramp gate −0.007" (TRANSFER_ANALYSIS §1a row 27) is not reproduced on the current recipe.
- No kernel change (t2_occ_v26 / t2_ongoing_pvogf) is warranted. Nothing was written.
- Onset and T1: no arm, as pre-registered. There is no capacity factor to apply, and production onset is already NORAMP.

## 4. Public-LB noise vs selection bias

Script: `analysis/t2lab/capscen_selbias.py`; log: `logs/capscen_selbias.log`.

- **Absolute sampling noise.** 40 + 40 official windows per split give an S_total SD of about 0.004–0.006 for one submission. The organizers' own figure for the persistence baseline is 0.0126, from a per-window IoU SD of 0.375. Public and private are independent draws, so a single team's public-to-private shift has SD ≈ 0.006–0.008 even with zero selection.
- **Best-of-N selection.** The bias of picking the best of N submissions is σ·√(1−ρ)·E[max_N], where ρ is the correlation of public noise between a team's submissions. E[max] is 2.23 for N = 48 and 2.51 for N = 100.
  - Our measured probe noise (TRANSFER_ANALYSIS §1) puts the difference SD between two variants at 0.0012–0.0027 LB (ongoing) and 0.003–0.005 (onset), which implies ρ ≈ 0.6–0.95.
  - So best-of-N bias is about **+0.003..+0.007** (σ 0.004–0.006, ρ 0.8–0.9), and up to +0.010..+0.015 if the variants are diverse (ρ ≤ 0.5).
- **Greedy hill-climbing (keep if public Δ > 0)** is worse, because accepted noise accumulates.
  - A window-level simulation with bounded IoU on the fixed 80 public windows: half onset-type probes re-drawing 3–7 windows, half ongoing-type probes perturbing 25 windows, every probe zero-mean in truth.
  - It gives a public-only gain of +0.025 (N = 48) to +0.035 (N = 100).
  - That is an upper bound, because real probes are mostly genuine changes with correlated window effects. A naive additive model (independent noise per probe) gives an unphysical +0.05..+0.10.
- **Estimate.** A team at 0.90 with 48–100 submissions plausibly carries **+0.005..+0.02** of public selection bias, central ≈ +0.01. Its true level is then roughly 0.88–0.895.
- **Our 0.88198 (~70 submissions, KEEP rule at +0.0001 that "decided on noise") also carries positive bias**, probably +0.003..+0.01. So our true gap to a 0.90 team is likely 0.005–0.015, not 0.018. It is not zero, and it cannot be measured on the public LB.
- **Consequence for the final picks.** Prefer changes whose evidence is lab / pseudo-March / FM-PM, not public deltas below ≈ +0.003. The two picks should differ on bets that are *independent* of public noise (the A/B scenario hedge already does this).

## 5. What could plausibly reach 0.89 within the rules (honest view)

- **No physics constant is miscalibrated** (H1 rejected). There is no cheap global correction.
- **The remaining loss is scenario-specific location / extent** (TRANSFER_ANALYSIS §3: onset location about 0.02 LB, ongoing extent about 0.009 LB). Only same-scenario history identifies it. On the public windows (March days 1–5, 0–4 earlier March days) that history barely exists.
  - So **public 0.89 is probably not reachable honestly** without either luck or public-LB fitting (section 4).
  - The private month has 30+ March days plus earlier April days before most windows. There the same-scenario lever (pv / pvo / sp4, i.e. pick A) is real: the lab estimates +0.03..+0.09 onset IoU and +0.009..+0.014 ongoing IoU on many-day pseudo-March.
- **The most plausible remaining private gains**, all ≤ T and legitimate:
  1. Scenario-recurrence location features built from earlier same-split days at the candidate links. This is about 90 % of onsets in the new scenario: they sit at train-recurrent places, chosen among candidates. The work is to strengthen and validate on many-day pseudo-March (pv2 a1–a3 showed +0.007..+0.014, concentrated in 1 panel).
  2. Per-step extent calibration of growth calls on new-scenario data, i.e. fitting a shrink on released-March windows for April use. It falls into the "decode calibration" family, so it needs FM-PM private ≥ +0.0005.
  3. Nothing from capacity, demand-level or timing corrections: the data shows none.
- **Expected private upside is +0.002..+0.006 LB**, not +0.01. A final *public* number of 0.89 should not be the target.
