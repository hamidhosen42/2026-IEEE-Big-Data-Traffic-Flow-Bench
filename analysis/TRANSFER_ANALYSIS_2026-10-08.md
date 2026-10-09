# Why lab gates fail to predict the public LB — transfer analysis, 2026-10-08

Analysis only. No kernel was pushed and nothing was submitted. New local code: `analysis/t2lab/xfer_ogf_diag.py`
(lab run, logs `analysis/t2lab/logs/xfer_ogf_diag_f{0,2}.log`, output `analysis/t2lab/results/xfer/`). The analysis scripts
(footprint, noise, FM-PM forecast, combos, decompositions) are copied to `analysis/t2lab/results/xfer/scripts/`. They read kernel outputs
downloaded to the session scratchpad: tfb-t2occ, t2ogfa/b/c, t2v20/21/22/22af/24 queue CSVs and probs. Outputs: `fm_validation.csv`, `h4_windows_forecast.csv`,
`h4_cells.csv`, `transfer_table.csv`.

Scoring reminder: one ongoing (or onset) IoU point = 0.15 LB; every official window has the same weight (8 panels × 5 windows,
4 families of 2 panels), so **one window = 0.15 / 40 = 0.00375 LB per unit of IoU**.

**Main results**

1. **Ongoing: a gate that predicts the LB exists.** Call it **FM-PM**, the footprint-matched pseudo-March forecast (section 2.5).
   - How it works: take the candidate's *label-free* decision changes on the 40 official windows. Price each changed window with the
     pseudo-March paired delta of lab windows in the same panel that received a similar change (Δ predicted cells) at a similar time of day.
   - On the 5 ongoing probes with lab pairs, FM-PM gives F03 +0.0043 (LB +0.0047), allfin +0.0008 (+0.0009), occ +0.0011 (≈ +0.0001),
     g72 +0.0004 (−0.0002, 4 windows) and **ogfeat −0.0002..−0.0007 (−0.0012)**.
   - Mean absolute error 0.0005 LB. The unconditional pseudo-March gain errs by 0.0015 and train OOF by 0.0013. Both of those predicted ogfeat at +0.003..+0.004.
2. **Why ogfeat failed.**
   - The March change was a *growth* change: +197 / −107 cells, positives +1.7 %, 159 of the 197 added cells outside the queue at T.
   - It was concentrated in 2 D7_I10_E windows at about 14:00 (+36 and +54 cells, 30 % of all changed cells).
   - In the train lab, all2 *shrinks* predictions (−0.9 %) and its big growth calls are right (Δn ≥ 10: dIoU +0.06).
   - In pseudo-March, the same growth calls are wrong (Δn ≥ 10: −0.074, Δn ≥ 20: −0.13, 80 % negative). D7_I10_E 13–15 h is −0.10..−0.11, and growth ("new") cells are over-confident (precision 0.68–0.74 at mean p 0.81).
   - FM-PM prices those 2 windows at −0.18 and −0.20 IoU, ≈ −0.0014 LB, which is the whole observed loss.
   - Cause: the effect of the features on outcomes changes with the scenario (where and how far queues grow is scenario-specific). The feature distributions themselves do not shift: official-window SMD median 0.04.
   - The train lab cannot see this. The pseudo-March *average* hid it, because the regime is a small part of the pseudo-March windows but 2 of the 40 official windows.
3. **Onset LB probes cannot be read.** An onset change flips 3–7 official windows, and a flipped window moves by ±0.45–0.6 IoU.
   - So a probe has an LB noise SD of 0.003–0.005, above every expected onset gain (+0.001..+0.004). A single onset probe carries no information.
   - The 1-in-8 record is still weak evidence that the lab is biased upward (P = 3.5 % under a zero-mean null).
4. **Where the remaining loss is.**
   - Ongoing (LB ≈ 0.863):
     - About 55 % of the loss is at T+25/T+30.
     - About 60 % is missed new cells plus false new cells.
     - Growing windows (12–25 % of windows) carry 27–53 % of the loss.
     - A per-step size oracle with our own ranking recovers +0.05..+0.06 IoU, about half of the loss: the problem is **how far the queue extends**, not where it is.
   - Onset (LB ≈ 0.765): in the new scenario, 57–78 % of the loss comes from windows with zero overlap, i.e. the wrong location.
5. **Experiment run (lab, ≈ 1 h): ogfeat with a growth veto (veto10).**
   - Rule: take all2's decision unless it adds more than 10 cells outside the queue at T compared with fs4occ; otherwise keep fs4occ.
   - Positive on every gate:

     | Gate | veto10 | all2 |
     |---|---:|---:|
     | Train offlike | +0.017 | +0.020 |
     | Pseudo-March offlike_m / runstart_m | +0.027 / +0.010 | +0.025 / +0.007 |
     | FM-PM, validation (LB-equivalent) | +0.0004 | −0.0007 |
     | **FM-PM, private** | **+0.0036** | +0.0027 |

   - Recommended for the private rows. It needs no new training: the kernel probabilities already exist.

---

## 1. Lab → LB table

**Sources and conventions.**
- Full table: 53 rows with sources and PRE / POST tags, in `analysis/t2lab/results/xfer/transfer_table.csv`. Conventions are in `transfer_notes.txt`.
- **LB Δ** is measured against the probe's *actual* base, taken from the chain logs and state files:
  - G1, G2 and G3 are on E5 (the oct-6 chain started from E5), so **G3 = allfin + occ vs fs4**.
  - H1–H5 are on G4 / H1.
- **IoU Δ** = LB Δ / 0.15 (LB Δ / 0.30 for the two rows that changed both conditions).
- **Ratio** = LB IoU Δ / lab IoU Δ. A negative ratio means the sign was wrong.
- Lab sets:

  | Set | Meaning |
  |---|---|
  | off40 | the 40 official train windows |
  | offlike | official-like train OOF, 2,992 windows (= "official-like" / "train OOF") |
  | pseudo | current-style train pseudo windows |
  | offlike_m / runstart_m | pseudo-March sets |
  | gate | other pseudo-March numbers (EXP_024 gate, pmarch2, the 214-window onset set) |

- A "†" marks a number computed after the probe (POST).

### 1a. Ongoing (and both-condition) probes

| # | change | LB Δ | IoU Δ | off40 | offlike | pseudo | offlike_m | runstart_m | gate | ratio (best set) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | og_noprior (prior-free only) | +0.00454 | +0.030 | −0.012 | | −0.009 | | | **+0.080** | gate 0.38; train sets wrong sign |
| 15 | v9 weekday priors (both conditions) | +0.00330 | +0.011 | +0.008 | | +0.001 | | | | off40 1.45 |
| 16 | og pers30 | −0.00183 | −0.012 | −0.019 | | −0.015 | | | **+0.005** ✗ | off40 0.65 |
| 18 | 63 → 127 leaves | +0.00109 | +0.007 | +0.010 | | +0.007 | | | | 0.71 / 1.0 |
| 21 | 127 → 255 leaves | +0.00033 | +0.002 | +0.006 | | +0.003 | | | | 0.39 / 0.65 |
| 24 | 255 → 511 leaves (onset retrained too) | +0.00182 | +0.012 | +0.008 | | +0.006 | | | +0.0015 | 1.6 / 2.1 |
| 25 | id511 | +0.00028 | +0.002 | +0.006 | +0.005† | +0.010 | | | +0.0036 | 0.3 / 0.5 |
| 27 | + ramp T−3..T−1 | +0.00030 | +0.002 | +0.004 | | +0.002 | | | −0.007 ✗ | 0.48 / 1.0 |
| 30 | np511r instead of id511r | +0.00047 | +0.003 | −0.006 ✗ | −0.005† ✗ | −0.010 ✗ | | | **+0.0095** | gate 0.33 |
| 35 / 39 | og4 / og8 bags | +0.00069 / +0.00009 | | | | | | | | no lab number |
| **40** | **F03 fs2 (selection-faithful)** | **+0.00474** | **+0.032** | +0.038 | +0.036 | **−0.028** ✗ | +0.021 | +0.022 | | offlike 0.88, off40 0.84, offlike_m 1.5 |
| 41 | fs4 bag vs fs2 | +0.00040 | +0.003 | | −0.0005† | +0.0002 | +0.002 | +0.0015 | | ~1.3 (noise) |
| 46 | g72 switch (F10 on 8 windows) | −0.00022 | −0.0015 | +0.0007 | +0.004 | | +0.003 | | | wrong sign (4 windows changed) |
| **47** | **allfin** | **+0.00090** | +0.006 | +0.002 | +0.0027 | +0.0017 | +0.0065 | +0.0033 | | offlike_m 0.92 |
| 50 | allfin + occ vs fs4 | +0.00102 | +0.0068 | +0.0016 | +0.0055 | +0.010 | +0.0065 | +0.004 | | offlike_m 1.05 |
| **56** | **ogfeat all2** | **−0.00121** | **−0.008** | +0.0155 | +0.0196 | +0.024 | +0.0255 | +0.0094 | | **wrong sign everywhere** |

### 1b. Onset probes

| # | change | LB Δ | IoU Δ | off40 | offlike | pseudo / CV | gate (pseudo-March) | forecast | sign by set |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 12 | on30 (more prior-free) | −0.00392 | −0.026 | −0.088† | | −0.075 | −0.042 | | all ✓ (ratio 0.3–0.6) |
| 13 | CatBoost | +0.00057 | +0.004 | +0.022 | | +0.009 | **−0.025** ✗ | | off40 ✓ 0.17 |
| 31 | COMBO14 50/50 cat / LGB | −0.00207 | −0.014 | **−0.026**† | | −0.001 | **+0.018** ✗ | | off40 ✓ |
| 36 | COMBO19 stacker | **−0.01303** | −0.087 | | | +0.006 (in-kernel CV) ✗ | | +0.0006 LB | CV ✗ |
| 42 | sp4 scenario prior | +0.00104 | +0.007 | | | | +0.090 (days 1–5 +0.026) | +0.0033 LB† | gate ✓ (0.08; 0.26 on days 1–5) |
| 44 | LODO bag | −0.00496 | −0.033 | | −0.002..+0.001 | | **+0.028** ✗ | | train ≈ 0 |
| 45 | mix9 early-month prior | −0.00144 | −0.010 | | | | **+0.030** ✗ | **+0.0042 LB** ✗ | gate ✗ |
| 49 | all-finite onset (G2) | −0.00107 | −0.007 | **−0.0125**† | +0.0056 ✗ | +0.0055 ✗ | +0.0039 ✗ | | off40 ✓ |

### 1c. Task 1 probes

The lab proxy is on the LB scale: 0.35 × holdout ΔS_state, or the composite J.

| # | change | LB Δ | lab proxy | ratio |
|---|---|---:|---:|---:|
| 14 | low-rank B | +0.00146 | +0.00126 | 1.16 |
| 22 / 20 / 23 | D / E / F | +0.00064 / +0.00215 / +0.00110 | +0.00024 / +0.00084 / +0.00042 | 2.6 / 2.6 / 2.6 |
| 26 / 28 | FT / FTA | +0.00170 / +0.00031 | +0.00070 / +0.00017 | 2.4 / 1.8 |
| 34 / 37 | FDR / TV | +0.00167 / +0.00026 | +0.00130 / +0.00050 | 1.28 / 0.52 |
| 38 | E1 mix | +0.00063 | +0.00073 | 0.86 |
| 48 | t1dark (G1) | +0.00029 | +0.00030 | 0.95 |
| 53 / 54 | ramp W 0.75 / W 0.5 | +0.00037 / +0.00028 | +0.00069 / +0.00052 | 0.54 / 0.54 |
| 57 | H5 (FD bag of 5) | +0.00016 | +0.00010 | 1.6 |

All 15 T1 probes have the sign right.

### 1d. Which lab set predicts the LB

| | best set for **sign** | best set for **size** | notes |
|---|---|---|---|
| **Ongoing** | **FM-PM** (5 / 5 within noise; section 2.5). Among the plain sets: runstart_m 4/5, offlike_m 4/6, off40 9/13, pseudo 9/13 | **FM-PM** (MAE 0.0005 LB). offlike_m when it agrees on sign (ratio 0.9–1.5) | Every plain set failed on the two new *feature* sets (g72, ogfeat). Train sets get month-shift changes (og_noprior, np511r) wrong; pseudo-March gets them right. **Training-population changes** (F03) transfer at about 0.9× offlike |
| **Onset** | **off40**: 4/4, including a PRE −0.0125 for G2 and a −0.026 analog for COMBO14. pseudo-March gate 2/7 (median ratio −0.33); CV 3/5 | none. sp4 reached 0.08 of its gate value; mix9's +0.0042 LB forecast came out −0.0014 | **An LB probe cannot read an onset change at all.** 3–7 windows change and a changed window moves ±0.45–0.6 IoU, so one probe has SD 0.003–0.005 LB, above every expected gain. off40's 4/4 is *n* = 4 on 40 windows: suggestive, not proven |
| **T1** | holdout or J proxy: 15/15 | J: 0.55–1.0× for blackout members. Holdout ΔS_state: about 2.5× (0.35 × ΔS × 2.5, i.e. LB ≈ 0.9 × ΔS), because the LWR / FD terms move with S_state | stable; no action needed |

**Noise floor for every ongoing probe.** Resampling 5 lab windows per panel gives a 40-window SD of the mean dIoU of 0.008–0.018, i.e. **±0.0012–0.0027 LB**.
- Ongoing probes under about +0.002 LB (og8, fs4, occ, g72, id511, ramp) are therefore unreadable on the public LB.
- The KEEP rule at +0.0003 has been deciding on noise.

---

## 2. Why ogfeat all2 (tfb-h4) failed

### 2.1 What changed on the 40 official March ongoing windows (fs4occ → fs4ogf)

Source: kernel outputs `tfb-t2occ` (`queue_submission_fs4occ.csv`, `t2v26occ_probs.npz`), `tfb-t2ogfa/b` (`t2v27{a,b}_probs.npz`),
`tfb-t2ogfc` (`queue_submission_fs4ogf.csv`, the file tfb-h4 swapped in on base tfb-h1).
- Decoding the averaged probabilities reproduces both CSVs exactly.
- These are the true rows of h1 and h4; h3 has the same validation rows as h1.

| | value |
|---|---|
| changed cells | 304 / 43,500 (0.70 %), in 28/40 windows |
| added / removed | **197 / 107**; positives 5,268 → 5,358 (**+1.7 %**) |
| added cells outside / inside the visible queue at T | **159 / 38** (growth into new links) |
| removed cells inside / outside the queue at T | **85 / 22** (earlier clearance) |
| changed cells by step h1..h6 | 14, 26, 61, 66, 65, 72 (late steps) |
| concentration | the top 5 windows hold 163 of the 304; **D7_I10_E validation_007 (13:55, +36 cells) and _010 (13:50, +54)** hold 90 |
| per panel (changed / added / removed) | D7_I10_E 103/90/13, D7_I405_N 54/18/36, D12_I5_N 35/31/4, D7_I405_S 34/13/21, D7_I210_W 31/17/14, D7_I10_W 24/24/0, D12_I5_S 19/4/15, D7_I210_E 4/0/4 |
| g72 (mid-run) windows | 8 windows, 98 changed cells; FM-PM sum +0.03, so **not** the driver |
| near the queue boundary | 172 of the 304 changed cells are ≤ 2 links from a queued-at-T link; 132 are further away |

Each model's expected IoU of the two decisions, computed under its own probabilities:
- **fs4occ's view:** switching costs −0.033 IoU (−0.0049 LB).
- **all2's view:** switching gains +0.037 (+0.0055 LB).
- **LB:** −0.0081 IoU. That puts March at λ = 0.36 on the line between the two views, i.e. closer to the base model.
- **Lab:** on changed windows, λ = 0.99–1.39 (train official-40 / offlike / runstart). In the train scenario all2 is right whenever the two models disagree.

### 2.2 The same comparison in the lab

`xfer_ogf_diag.py` uses the same kernel functions, all-finite and occupancy inputs, the F03 rows and fold 0 at 255 leaves. The gain reproduces: official +0.0155, offlike +0.0196, runstart +0.0133.

| | train offlike (726) | train runstart (2,731) | pseudo-March offlike_m / runstart_m | **March official 40** |
|---|---|---|---|---|
| cells changed | 0.50 % | 0.40 % | — | 0.70 % |
| added vs removed | 1,504 vs **2,524** | 4,671 vs **6,790** | — | **197** vs 107 |
| positives all2 vs base | **−0.9 %** | −0.5 % | **+0.4 % / +0.9 %** (D7_I10_E **+6 % / +8 %**) | **+1.7 %** |
| dIoU of windows with large growth (Δn ≥ 10) | **+0.059** (n 22) | +0.052 (n 49) | **−0.074** (n 55, 67 % negative); Δn ≥ 20: **−0.13** (80 % negative) | the 2 D7_I10_E windows: Δn +36, +54 |
| dIoU of windows with large shrink (Δn ≤ −10) | +0.052 | +0.065 | +0.090 | 9 windows remove 3–29 cells |
| D7_I10_E, origins 13–15 h | +0.008 (n 37) | — | **−0.114 (n 8) / −0.096 (n 45)** | 007, 010 |

The direction of the March footprint is visible without labels: growth, and more positives. It matches pseudo-March and is the opposite of train.

**Calibration of the cells each model predicts outside the visible queue at T ("new" cells).** Each entry is precision / mean p.

| | base | all2 |
|---|---|---|
| train offlike | 0.84 / 0.81 | 0.90 / 0.85 |
| pseudo-March offlike_m | 0.73 / 0.77 | **0.74 / 0.81** |
| pseudo-March runstart_m | 0.72 / 0.77 | **0.68 / 0.81** |

- "Old" cells (inside the queue at T) are calibrated everywhere: 0.96–0.98.
- In the new scenario, growth calls are over-confident. all2 makes 11–16 % more of them and is more over-confident.

### 2.3 Hypotheses tested

| Hypothesis | Test | Verdict |
|---|---|---|
| **H1. Lab windows are distributed differently from the official ones**: masking, early month, g72 | Official validation history coverage is 0.72–0.78 (mean 0.763), the same as train offlike; pseudo-March is masked (~0.5). Masked control: the gain survives 30 % masking (+0.020, 8/8). Early month: pseudo-March days 1–12 gives **+0.030** (larger, not smaller). g72 windows: +0.03 FM-PM, not the driver | **Rejected** as the main cause |
| **H2. Lab labels come from a different selector** | offlike uses the faithful t36 rule, which reproduces 36/40 official origins. FM-PM built on pseudo-March windows from the masked month predicts the LB correctly, so the selector of the lab windows is not what breaks transfer | **Rejected** |
| **H3. Small n**: 40 windows | Resample 5 windows per panel from the lab per-window deltas. The 40-window SD of the mean dIoU is 0.008 (official-40), 0.012 (offlike), 0.011 (runstart_m) and 0.018 (offlike_m), i.e. **0.0012–0.0027 LB per ongoing probe**. P(dIoU ≤ −0.0081) is 0.0003–0.09 (0.009–0.14 after the usual 0.6× shrink) | **Contributes, but is not sufficient alone.** It does mean that **every ongoing probe below about +0.002 LB is a coin flip on the public LB** |
| **H4. Feature distribution shift between train and March** | `results/ogfeat/shift_new_features.csv` (98 static ogfeat columns, keep-region cells): standardised mean difference vs train. Official validation windows: median abs 0.036, 8 % above 0.2. Pseudo-March: 0.13. Private: 0.10 | **Marginal shift rejected** for the official windows. What shifted is the **conditional** P(queue \| features): the growth calls (2.2) |
| **H5 (new). Scenario-conditional growth regime** | pseudo-March (new scenario, same features): Δn ≥ 10 windows lose −0.074; D7_I10_E 13–15 h −0.10; new-cell precision 0.68–0.74 at p 0.81. Train: the opposite sign. FM-PM prices the 2 official D7_I10_E windows at −0.18 and −0.20 IoU (−0.0014 LB together) and the whole change at −0.0002..−0.0007 LB | **Supported**: this is the failure |

### 2.4 What the lab predicted on comparable windows

- **Train lab** (official-40 / offlike, fold 0, production inputs):
  - It forecasts +0.022 IoU for the official footprint and +0.0033 LB for the changed windows.
  - Its windows with growth calls of the size seen in March are all positive.
- **Pseudo-March aggregate:** +0.025 (offlike_m) / +0.0065 (runstart_m). It is dominated by shrink and clearance calls, which do transfer (+0.07..+0.09).
- **Pseudo-March conditioned on the March footprint** (same panel, ±2 h, similar Δn): −0.0016 IoU (panel-exact) to +0.0017 (pooled), i.e. −0.0002..+0.0003 LB.
  - The variant with all-cell counts, consistent with the official template, gives −0.0047 IoU (−0.0007 LB).

### 2.5 The gate that would have caught it: FM-PM (footprint-matched pseudo-March forecast)

Steps:
1. Compute the candidate's decisions on the official validation (and private) windows. This needs no labels: decode both probability files.
2. For each changed window, take the mean pseudo-March paired delta over lab windows that meet all of these:
   - same panel (fall back to pooled when n < 5);
   - origin within ±2 h;
   - same sign of Δ predicted cells, and |Δn − Δn_w| ≤ max(3, 0.5 |Δn_w|).
3. Average the 40 values and multiply by 0.15.

Validation on every ongoing LB probe that has a pseudo-March per-window lab pair:

| Probe (official files) | LB Δ | FM-PM | pseudo-March offlike_m × 0.15 | train offlike × 0.15 | windows / cells changed |
|---|---:|---:|---:|---:|---|
| F03 fs2 (og8 → fs2) | **+0.00474** | **+0.00434** | +0.00315 | +0.0054 | 34 / 206 |
| allfin (fs4 → fs4af) | **+0.00090** | **+0.00075** | +0.00097 | +0.0004 | 22 / 108 |
| occ (fs4af → fs4occ; LB ≈ G3 − F5) | ≈ +0.0001 | +0.0011 | +0.00025 | +0.0005 | 21 / 96 |
| g72 switch (fs4 → sw03 on g72 windows) | −0.00022 | +0.00035 | +0.00045 | +0.0006 | 4 / 11 |
| **ogfeat all2 (fs4occ → fs4ogf)** | **−0.00121** | **−0.0002..−0.0007** | +0.0037 | +0.0029 | 28 / 304 |
| mean abs. error | | **0.0005** | 0.0015 | 0.0013 | |

**Limits.**
- 5 probes only.
- The lab arms are 255-leaf single models, not the 511-leaf production bags.
- FM-PM needs the dense runstart_m windows: with offlike_m alone the matching cells are too sparse (ogfeat then reads +0.012).
- The residual is within the outcome noise of a single probe (±0.001–0.002).
- Use FM-PM as a *veto* (do not probe when FM-PM ≤ 0), not as a precise forecast.
- It uses pseudo-March labels from all March days, for model *choice* only, like the public LB itself; never as a model input.

---

## 3. Error decomposition of the production-like model

**Ongoing.**
- Model: `base_afocc` = F03 + all-finite + occupancy, np, 255 leaves, fold 0 (`results/xfer/ogf_diag_f0_L255.pkl`). The fold-2 replicate is in `ogf_diag_f2_L255.pkl`: base offlike 0.881, runstart 0.914, the same picture.
- Most LB-faithful lab set: **offlike (official-like train OOF), read through the FM-PM footprint**. Plain train OOF is faithful for training-population changes (F03 ×0.9). For feature changes, read it only together with the pseudo-March conditional check.
- Shares are shares of 1 − IoU with the competition weights.

| set | IoU | missed new (fn_new) | dropped old | kept cleared (fp_old) | false new | h1 | h2 | h3 | h4 | **h5** | **h6** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official-40 | 0.891 | **0.57** | 0.03 | 0.27 | 0.13 | 0.07 | 0.12 | 0.14 | 0.11 | 0.21 | **0.35** |
| offlike (726) | 0.883 | **0.36** | 0.08 | 0.33 | 0.23 | 0.08 | 0.11 | 0.15 | 0.16 | 0.20 | **0.29** |
| runstart (2,731) | 0.907 | 0.33 | 0.11 | 0.26 | **0.30** | 0.05 | 0.07 | 0.11 | 0.15 | 0.23 | **0.38** |

Growing / shrinking / stable windows. Each window is classed by its true size at h6 against the queue at T (±25 %).

| set | grow: windows / loss / IoU | shrink: windows / loss / IoU | stable: windows / loss / IoU |
|---|---|---|---|
| official-40 | 25 % / **53 %** / 0.77 | 47 % / 34 % / 0.92 | 28 % / 13 % / 0.95 |
| offlike | 21 % / **32 %** / 0.82 | 48 % / 43 % / 0.90 | 31 % / 25 % / 0.91 |
| runstart | 12 % / **27 %** / 0.81 | 35 % / 31 % / 0.92 | 52 % / 42 % / 0.93 |

**Size vs location.** Oracles keep the model's own ranking and change only the number of cells.

| set | IoU | window size oracle | per-step size oracle |
|---|---:|---:|---:|
| official-40 | 0.891 | +0.024 | **+0.053** |
| offlike | 0.883 | +0.038 | **+0.060** |
| runstart | 0.907 | +0.025 | **+0.048** |

- The per-step oracle recovers about half of the loss. The ranking (*where*) is good; the per-step extent (*how far*) is what is wrong.
- In the new scenario, the extent errors come from **over-confident growth calls**: new-cell precision 0.72–0.73 at mean p 0.77 for the base model (2.2).

**By panel** (offlike loss share):

| Panel | Loss share |
|---|---:|
| D7_I10_E | 0.19 |
| D12_I5_N | 0.18 |
| D12_I5_S | 0.17 |
| D7_I10_W | 0.13 |
| D7_I405_S | 0.10 |
| D7_I210_W | 0.09 |
| D7_I405_N | 0.07 |
| D7_I210_E | 0.06 |

The official-40 set has the same top 3, plus D7_I405_S.

**Onset.**
- Train OOF: CatBoost `elig_bag`, `results/allfin/onset_cv.pkl`.
- New scenario: pseudo-March 214 windows, `results/splitprior/march_onset_variants.pkl`, arms base = train prior and s4 = production prior. Pseudo-March s4 (0.754) is level-faithful to the LB onset estimate (≈ 0.765); train OOF (0.89) is not.

| set | IoU | loss in zero-overlap windows (wrong location) | under-size | over-size | exact size |
|---|---:|---:|---:|---:|---:|
| train official-40 | 0.945 | 0 | 0.033 | 0.008 | 90 % |
| train offlike (522) | 0.888 | 0.057 (51 % of loss) | 0.064 | 0.021 | 85 % |
| pseudo-March base | 0.665 | **0.261 (78 %)** | 0.131 | 0.073 | 68 % |
| pseudo-March s4 (production) | 0.754 | **0.141 (57 %)** | 0.133 | 0.052 | 68 % |

- Worst panels (pseudo-March s4): D12_I5_S (0.61), D7_I10_W (0.60), D7_I405_S (0.63).

**Largest recoverable loss, in LB points.**

| Loss | Size | Assessment |
|---|---:|---|
| Onset location | ≈ 0.235 × 0.15 × 0.6 ≈ **0.02** | Reachable only with same-scenario location information. Public windows are days 1–5 with 0–4 earlier March days: sp4 is in, and LODO / mix9 failed. It is the private lever, already in pick A's pv bag. On public it cannot be verified (section 1) |
| Ongoing per-step extent | ≈ 0.137 × 0.15 × 0.45 ≈ **0.009** | Growing windows at h5/h6, i.e. missed new cells and false new cells. The *only* part of it that fails to transfer is the scenario-specific growth call. This is where the next experiments aim |

---

## 4. New experiments

**Gate for every ongoing candidate.** A candidate needs all three to keep going:
- **FM-PM > 0 on the validation footprint and on the private footprint**;
- train offlike > 0;
- the label-free footprint check: the direction of the change on the official windows must match the lab regime that produced the gain.

**For onset**, no public probe can pass or fail a change (SD 0.003–0.005 per probe). Onset changes go to the private file only, behind pseudo-March evidence.

### X1 (run, ≈ 1 h of lab): ogfeat with a growth veto, `veto10`

**Rule.** Per window, take all2's decision unless all2 adds more than 10 cells outside the visible queue at T compared with fs4occ; in that case keep fs4occ.
- It keeps all2's clearance and shrink calls, which transfer (+0.07..+0.09 on pseudo-March).
- It drops the scenario-specific growth calls, which do not.

| gate | all2 | **veto10** | veto3 | blend 0.5 | A∧B |
|---|---:|---:|---:|---:|---:|
| train official-40 (fold 0) | +0.0155 | +0.0107 | +0.0074 | +0.0107 | +0.0063 |
| train offlike (fold 0, 726) | +0.0196 | **+0.0171** | +0.0133 | +0.0152 | +0.0093 |
| train runstart (fold 0) | +0.0133 | +0.0124 | +0.0107 | +0.0117 | +0.0074 |
| pseudo-March offlike_m | +0.0245 | **+0.0269** (7/8) | +0.0257 | +0.0109 | +0.0252 |
| pseudo-March runstart_m | +0.0065 | **+0.0096** (7/8) | +0.0091 | +0.0084 | +0.0054 |
| FM-PM validation, LB-equivalent | −0.0007 | **+0.0004** | +0.0005 | +0.0003 | +0.0004 |
| FM-PM private, LB-equivalent | +0.0027 | **+0.0036** | +0.0036 | +0.0004 | +0.0026 |

**Fold-2 replicate** (train, 748 offlike / 2,633 runstart windows): all2 +0.0236 / +0.0126, **veto10 +0.0221 / +0.0121**, veto3 +0.0180 / +0.0109, blend 0.5 +0.0162 / +0.0110, A∧B +0.0145 / +0.0092. The veto costs 0.001–0.002 of in-scenario gain.

**Use.**
- Private rows.
  - Pick B (train-only April rows): replace the fs4occ April ongoing rows with veto10. It is positive under both scenario hypotheses: train scenario +0.017, March scenario FM-PM +0.0036.
  - Pick A: port all2 to the pvo recipe (LEVERS E2), then apply the same veto.
- Public probe: optional. Expected +0.0004, noise ±0.0015, so it cannot confirm anything; spend a slot only if one is free.
- Cost: no training. One combine kernel reads `tfb-t2occ` and `tfb-t2ogfa/b` probabilities, decodes, and applies the veto, as `t2_ogfeat_comb.py` already does.
- Caveat: the threshold 10 was chosen after seeing the H4 failure. veto3 gives the same picture, so the result is not sensitive to the exact value.

### X2 (proposed, lab ≈ 3 h): physics shock gate for growth calls

**Target.** The largest remaining ongoing loss: growing windows at h5/h6, i.e. missed new cells and false new cells. Also the scenario-specificity of learned growth.

**Method.** This is not a feature. It is a rule on decisions.
1. For each window, estimate the upstream tail's shock speed from data ≤ T: w = (q_up − q_q) / (k_up − k_q), using the measured flow / density upstream of the tail and inside the queue (all-finite history, slot T).
2. From it, compute the kinematic tail position at T+h.
3. Accept model cells outside the queue at T only where they lie within the kinematic tail ± 1 link. Otherwise apply the fs4occ decision.

This applies to fs4occ itself and to all2.

**Why it should transfer.** It uses only the current measurements and conservation, not learned scenario patterns. The ratio is physics, not a train-month association.

**Gate.** Train offlike > 0, FM-PM validation > 0 and FM-PM private > 0 (both footprints).
- Kill it if the pseudo-March new-cell precision of accepted growth calls is not at least 0.05 above that of rejected ones.

**Rules.** Uses data ≤ T only.

### X3 (proposed, lab ≈ 2 h + 1 kernel): FM-PM as a pre-registered gate, and two probes it decides

- Re-run FM-PM, with the private footprint included, on every pending ongoing candidate.
- **Decision rule:**
  - Push only candidates with FM-PM validation ≥ +0.0005 *and* private ≥ +0.0005.
  - Use public slots only when FM-PM validation ≥ +0.0015, i.e. above about 1 SD of single-probe noise.
- Under this rule the ogfeat all2 public probe would have been rejected (−0.0002..−0.0007) and F03 accepted (+0.0043).
- A cheap extension: build FM-PM for the T1 blackout members on the official blackout rows. The T1 transfer ratio is already stable at 0.55–1.0×, so this is low priority.

**Not proposed** (rejected or saturated):
- decoder morphology / thresholds;
- equal-window weights;
- LODO onset;
- more ongoing features (X1 combines models that already exist);
- Task 4, recon / TV.

A *scenario-fitted shrink of new-cell probabilities* would directly fix the over-confidence in 2.2. It is a decode-calibration change, i.e. the rejected category, so it is listed only as an observation.

---

## 5. Follow-ups (2026-10-08, afternoon)

### 5.0 PRE-REGISTERED decision rule for X2 (written 11:39 +06 before any FM-PM number for X2 was computed)

The shock-gate variants are shock(fs4occ) and shock(all2).

**PUSH only if** FM-PM is ≥ +0.0005 (LB-equivalent) on **both** footprints against fs4occ:
- the validation footprint (40 official March ongoing windows);
- the private footprint (40 April windows).

FM-PM settings, fixed in advance:
- panel-exact matching, ±2 h, all-cell predicted counts;
- pseudo-March offlike_m + runstart_m lab pairs (lab models og_np_base / og_np_all2, fold 0+1 mean).

**Reported but not binding:**
- train offlike and runstart, folds 0 and 2 (production inputs, 255 leaves);
- pseudo-March offlike_m and runstart_m.

The margin m = 1 link and the kinematic definition below are fixed before evaluation. No variant is tuned on FM-PM.

### 5.1 X2 shock gate: result → **FAIL, DROP**

Code: `analysis/t2lab/xfer_shock_lib.py` and `xfer_shock_eval.py`. Log: `logs/xfer_shock_eval.log`. Outputs: `results/xfer/shock_*.{pkl,csv}`.

**Definition (as pre-registered):**
- Clusters are the runs of qlast. Downstream is index +1 for E/N panels and −1 for W/S panels.
- Recent state = the last finite value in T−3..T−1.
- Shock speed w = (q_u − q_c) / (k_u − k_c), per lane, using the 3 upstream non-queued links and the tail plus 1 link inside the queue. It is used only when k_c − k_u > 5, and clipped to ±30 km/h.
- Over h steps the tail reaches max(0, −w) · 5h/60 km upstream, plus a margin of 1 link. 1 link is also allowed downstream of the head.
- On the official windows the median w is −0.6 km/h (q10 −12, q90 +21), and 51 % of clusters are growing.

| gate (vs base / fs4occ) | shockA = shock(fs4occ) | shockB = shock(all2) | all2 | veto10 |
|---|---:|---:|---:|---:|
| train offlike f0 / f2 | −0.062 / −0.062 | −0.050 / −0.046 | +0.020 / +0.024 | +0.017 / +0.022 |
| train runstart f0 / f2 | −0.041 / −0.041 | −0.033 / −0.034 | +0.013 / +0.013 | +0.012 / +0.012 |
| pseudo-March offlike_m / runstart_m | −0.017 / −0.010 | +0.005 / −0.004 | +0.025 / +0.007 | +0.027 / +0.010 |
| **FM-PM validation** (LB) | **−0.0013** | **−0.0005** | −0.0007 | +0.0004 |
| **FM-PM private** (LB) | **−0.0034** | **−0.0002** | +0.0027 | **+0.0036** |

- Under the pre-registered rule, both shock variants fail (FM-PM < +0.0005 on both footprints) and are dropped.
- The gate removes 3–4 % of positives. These are mostly *correct* growth cells beyond the kinematic reach: queues grow faster than the instantaneous shock estimated from T−3..T−1.
- No `queue_submission_shock.csv` variant was written.
- Applying the same rule to veto10: it passes on private (+0.0036) but not on validation (+0.0004 < +0.0005). That confirms: **no public slot for veto10; private only.**

### 5.2 X1 kernel: `kaggle_jobs/t2_ogveto.py` (written, NOT pushed)

**Kernel sources it needs:**
- **tfb-t2occ:** `queue_submission_fs4occ.csv` and `t2v26occ_probs.npz` (used only for qlast and the window ids);
- **tfb-t2ogfc:** `queue_submission_fs4ogf.csv`;
- the competition data: `corridors/<panel>/network/lwr_mainline_topology.csv` for the link order.

It does no training and no decode; it runs in seconds.

**Output:** `queue_submission_veto10.csv`, with all 174,000 queue rows in fs4occ row order. A per-window file `t2ogveto_windows.csv` is written alongside.

**CONTROL lines:**
- onset rows equal fs4occ (assert);
- windows vetoed per split;
- rows changed vs fs4occ and vs fs4ogf per split;
- positives.

**Local test** (`analysis/t2lab/xfer_ogveto_localtest.py`, log `logs/xfer_ogveto_localtest.log`):
- `ast.parse` passes.
- Decoding the probability files reproduces both input CSVs on all 87,000 ongoing rows.
- **The kernel's veto10 rows equal the lab veto10 decisions on 43,500 / 43,500 validation and 43,500 / 43,500 private rows.**

| split | windows vetoed | rows changed vs fs4occ | rows changed vs fs4ogf | positives (fs4occ / fs4ogf / veto10) |
|---|---|---:|---:|---|
| validation | 4: D7_I10_E 007 and 010, D7_I10_W 008, D7_I210_W 009 | 190 | 114 | 5,268 / 5,358 / 5,244 |
| private | 1: D7_I210_W private_006 | 281 | 12 | 4,750 / 4,721 / 4,709 |

**Use:** April ongoing rows of pick B (hedge) via `swap_queue2.py`, with part `(tfb-t2ogveto, queue_submission_veto10.csv, queue_ongoing, private)`. Check that the public score is unchanged (max_val_changed 0).

**For pick A:** port ogfeat to the pvo recipe first (LEVERS E2), then apply the same veto.
