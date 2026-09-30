# Source family A: competition resources, generator analysis, Traffic4cast

Written 2026-09-30. Resource IDs refer to `A_resources.csv`.

**What was read**
- The official repository, including its full commit history (29 commits, with bodies).
- The Kaggle forum dump.
- Seven participant repositories, with code inspected.
- Eight public Kaggle notebooks, pulled through the API.
- The Traffic4cast 2019–2022 papers and repositories. arXiv lists no Traffic4cast 2023 edition, and nothing about one was found.

The most informative source by far is **Hisernberg/kraggle-3-in-one `trafficflow/`** (A0057–A0084): public LB 0.87249 on 2026-09-28, with every submission logged, single-factor LB deltas, and exact probe decompositions. It has no licence, so we take ideas only and copy no code.

---

## 0. Where we stand vs the best documented public pipeline

| Component | Ours (EXISTING_MAP) | Hisernberg H2/H11P (A0058) | Note |
|---|---|---|---|
| S_ODME | ≈0.994 | 0.9938 (probe) | Same. Saturated. |
| Task 1 + Task 3 (weighted) | S_state ≈0.93–0.94 val-holdout, S_phys ≈0.67 | 0.4327 weighted at H2, then +0.0038 more from transductive training | Their T1+T3 lead comes from transductive training, seed ensembles, TV smoothing and gated reconciliation. |
| Onset IoU (March) | 0.743 (measured at LB 0.84855) | 0.732 (H2) | Ours is **not** behind on onset. |
| Ongoing IoU (March) | 0.747 at LB 0.84855; later + prior-free + weekday priors, current value unmeasured | 0.843 (H2) | **Likely our largest Task 2 gap.** Re-probe it (zero-ongoing probe on COMBO5). |

---

## 1. Candidate experiments (ranked by expected value)

### I-1. Transductive Task 1 training on observed test-month cells (HIGH-VALUE, **NEW**)

**Basis:** A0058 (EXPERIMENTS.md, "Transductive Task 1 member"), A0066 (`t1_pseudo.py`), rule A0008.

**Weakness it addresses:** our T1 model is fitted on train only (EXP_T1_l uses the val-month cells only as a holdout). The val and private months are independent demand draws (A0006, A0035, A0045), so train-only fits leave month-specific structure unlearned.

**Label:** NEW. We already built the val-month holdout, which is the evaluation half of this idea.

**Exact change:**
- For each panel and each of March and April, sample about 90k observed, eligible, non-target, non-blackout cells as extra training rows, labelled with their released values.
- Build their features with those cells hidden. Hide them in rounds of about 30k per round (about 5% extra masking) so the neighbour context looks like a real target's.
- Keep them disjoint from the evaluation cells (our 15% val-month holdout, plus an April holdout built the same way).
- Add no absolute date or day-index feature.
- Hisernberg's increments were 1× → 2× → 3× rows with different seeds, averaged.

**Data:** Task 1 is offline, and any released observation may be used (A0008). Not a Task 2 input.

**Leakage risk:** low, as long as evaluation cells are excluded from training rows and hidden when their own features are built.

**Compute:** one more feature build over about 1.8M extra rows, plus one T1 fit. Fits a Kaggle CPU kernel.

**Expected gain:**
- Hisernberg LB: +0.00229 (1×), +0.00109 (2×), +0.00047 (3×), about **+0.004 total**.
- Their pseudo-holdout measured only the S_state part. The LB gain was about 1.2–2× the pseudo figure, the rest being LWR from better test-month density.

**Validation and reject rule:**
- Use a pseudo-holdout on **both** March and April, scored with the official S_state formula.
- Reject if 0.35·ΔS_state < +0.0005 on either month, or if fewer than 8/10 panels improve.

### I-2. Ongoing model: blend a full-feature model with a "no location prior, keep link identity" model (**HIGH-VALUE RETEST**)

**Basis:** A0059 §11–12 (the v4/v5 recipes). A0058: v5 was the only ongoing change that transferred to March, with S_queue +0.0081. Their March ongoing was 0.843.

**Weakness it addresses:** ongoing IoU. Our last measured value was 0.747. Our current ongoing is "prior-free (no location/time/prior features)".

**Hisernberg's ablation on 3,001 selector windows:**

| Model | Score |
|---|---|
| Dropping only the location/TOD priors (pq_*, rq7_*) | 0.877 (unchanged vs full) |
| Also dropping link identity, TOD and weekday | 0.829 vs 0.853 at the fast config. Hurts everywhere, **including the non-recurrent slice**. |
| 50/50 blend of the full model and the "noloc" model | best on every slice |

**Label:** RELATED / HIGH-VALUE RETEST.
- Our "prior-free only" beat the 70/30 blend on the LB (+0.0045).
- But if our prior-free model also removed link identity, position and static attributes, the Hisernberg variant (priors dropped, identity kept) has never been tested.

**Exact change:**
- Train ongoing model B with every feature except the location/TOD/weekday prior families. Keep link id or position, lanes, capacity and panel code.
- Final probability = 0.5·B + 0.5·(current prior-free model). Also try 0.5·B + 0.5·(full model).
- Their v5 weights were 0.35 LWR-all + 0.35 LWR-noloc + 0.15 v2-all + 0.15 v2-noloc.

**Data:** ≤T only. **Leakage:** none. **Compute:** one more ongoing model (255 leaves). Low.

**Expected gain:** +0.005 to +0.02 on ongoing IoU, i.e. +0.001 to +0.003 on S_total. Uncertain: Hisernberg's other ongoing CV gains failed on March (see §4).

**Validation and reject rule:**
- Gate on the **test-month pseudo-holdout** (I-4), not on train CV.
- Reject if its March-only ongoing delta is ≤ 0.

### I-3. Task 2 label repair: hybrid truth with a LightGBM imputer for missing train cells (**RELATED**, medium–high)

**Basis:** A0059 §13 ("The label fix"), A0068 (`truthfix.py`). LB: onset **+0.030** on March (D1, +0.00453 S_total), against CV +0.005 to +0.012.

**Weakness it addresses:** our Task 2 truth is observed speed ≤ 0.6·v_free on eligible cells only (TRACKER). About 24% of cells are missing or ineligible.
- Official truth is the complete noise-free state (A0021, A0011).
- At a queue's first slot the state is sharp in time and smooth in space. So missing cells distort the replicated selector (which origin is chosen and the T+30 set) and the size of the onset positive set. Hisernberg's first-slot blocks came out 7–17% too small.

**Label:** RELATED.
- We exclude missing cells rather than time-filling them, so our bias is smaller than theirs was (their old fill had queued recall 0.266 at onset slots). The gain may be smaller for us.
- Our onset (0.743) already exceeds theirs after the fix (0.728–0.732).

**Exact change:**
- Train a LightGBM imputer of speed on observed train cells. Features: the cell's neighbours at t±1,2, links ±1,2 and the diagonals. Train on even days, check on odd days.
- Hybrid truth: fill a missing cell with the imputer **only** within ±5 min and ±2 links of an observed queued cell. Elsewhere keep the observed-only truth.
- This avoids scattering the imputer's 1–2% false positives through free flow.
- Re-run our selector replica on the hybrid truth and retrain onset (CatBoost) on the redrawn windows.

**Data:** train labels only, so no ≤T problem. **Leakage:** none. **Compute:** medium (imputer plus one selector and feature rebuild).

**Expected gain:** +0.00 to +0.004 S_total.

**Validation and reject rule:**
- Compare on the **test-month pseudo-holdout** (I-4, observed labels), because the train CV truth changes with the fix.
- Reject if the March-only pseudo onset IoU (see the leakage note in I-4) does not improve by ≥ 0.005.

### I-4. Task 2 test-month pseudo-holdout as the adoption gate (**NEW**, validation infrastructure, high)

**Basis:** A0074 (§21/21b on branch `claude/focused-allen-uzq8gr`), A0075 (`t2/pseudo.py`).

**Weakness it addresses:** our CV→LB ratio is about 0.84 and systematic. Hisernberg found 4 of 5 ongoing changes that passed plain and shift-weighted CV lost on March (§4).

**What the pseudo-holdout did for Hisernberg:**
- It caught the G3 stacking failure: −0.0048 at 4 SE on pseudo-March, while CV said +0.0065.
- It gives about 3,000 ongoing and 230 onset windows per month (about 330 ongoing in `sim` mode, which replays the official greedy selector from every start day).
- The official windows sit in days 0–5 of each month, because the selector is chronological.

**Exact change:**
- On the masked March/April view, run our selector replica with its rules: coverage ≥ 0.7, onset/ongoing definitions, queued horizon cell, persistence ≤ 0.9, greedy with 360-min spacing, started from every day.
- Exclude windows touching released blackout rows ([T−12, T+6]).
- Forward-fill hidden Task 1 targets in the history causally, ≤ 3 slots.
- Labels are the observed eligible horizon cells. Leave hidden cells out of the IoU.
- Score with the official aggregation.

**Leakage risk: medium, and it decides protocol.**
- The pseudo labels are observations after the official windows' T.
- Using April labels to pick the model for April (private) windows means post-T data steers the forecast. Treat that as off-limits.
- **Use March pseudo-windows to gate decisions for the private file.** All March timestamps precede every private T.
- Treat April only as a sanity readout, never as a selection criterion. If in doubt, ask the organizers.

**Compute:** low (feature build on about 3k windows per month).

**Expected value:** it prevents LB-noise-driven mistakes. Hisernberg estimates the SD of a 40-window March ongoing LB delta at 0.005–0.009, so small Task 2 LB deltas are mostly noise.

### I-5. Task 1 seed/bagging ensemble (**NEW** for Task 1, medium)

**Basis:** A0062 (T1_ENSEMBLE.md), A0058.
- H2 (2-seed mean): +0.00066 LB against a +0.00077 local prediction.
- H3 (3rd seed): +0.00022 against +0.00023 predicted.
- Variance-reduction gains transferred at 85–95%. Capacity/row gains transferred at only 15–50%.

**Weakness it addresses:** T1 variance; our capacity-driven gains (EXP_T1_E/F) may transfer only partly.

**Label:** NEW. Our "3-seed averaging: no gain" was measured on Task 2 ongoing, not Task 1.

**Exact change:** train 2–3 T1 members, each with a different row subsample (seed changes both the row sample and the LightGBM seeds) and early stopping, and average raw speed, flow and density.

**Leakage:** none. **Compute:** 2–3× T1 training, which may need split kernels.

**Expected gain:** +0.0003 to +0.0007. **Reject rule:** val-month holdout J gain < +0.0003, or fewer than 7/10 panels improve.

### I-6. Task 3: TV smoothing of density inside runs of target cells (**NEW**, medium)

**Basis:** A0061 (T3_SMOOTHING.md), A0067. LB G1 +0.00060 against a local +0.00062.

**Weakness it addresses:** S_LWR. Our physics score is about 0.67, and every Task 3 post-processing we tried failed. Ours was spatial flow smoothing, which is a different operation.

**Exact change:**
- Per link, find runs of consecutive target cells.
- Apply total-variation smoothing to per-lane k = q/v within the run. Increments smaller than τ·(mean run density) are set to zero, so queue fronts are kept.
- τ values: 0.0075 in free flow, 0.02 inside the gate (v < 0.6·v_f), 0.05 in blackouts.
- Add weak L1 anchors to the observed neighbours at the run ends.
- Outside the gate move only the flow. Inside it, split the change between speed and flow.

**Where the error sits:** only within-run transitions (about 29% of the LWR loss) are affected. Isolated cells carry 46–48% of the loss and need per-cell accuracy instead.

**Leakage:** none. **Compute:** negligible.

**Expected gain:** +0.0004 to +0.0006. Tripling the smoothing lost on LB (−0.0004), so it is already near its optimum.

**Reject rule:** our LWR proxy + S_state J on at least 4 panels must be ≥ +0.0003, with none negative.

### I-7. Gated density reconciliation with speed-heavy split (**HIGH-VALUE RETEST** of our rejected EXP_T1_f/g/m)

**Basis:** A0058 (gated reconciliation table), A0065.
- Gate: J +0.0016 on 4/4 panels.
- Split a = 0.75: S_state +0.0018 on 4/4 panels, LB +0.00064, exactly as predicted.

**What they did:**
- A separate density model (near-L2, Huber α = 1; the L1-leaning version was worse).
- Reconcile to q/v = k̂ **only where predicted v < 0.6·v_f**. Below 0.4·v_f the density model's L1 error is 1.50 against 1.77 for FD(v) and 2.99 for q/v. Above the gate, q/v is better.
- Split the adjustment roughly 75/25 between the two variables, putting more of the change on speed. In queues, flow sits near discharge capacity.

**Label:** RELATED / partly CONTRADICTED. Our tests were ungated or on a single panel (D12_I405_S) and found 0.

**Caveat:** Hisernberg later found that LWR gains from *more accurate density models* transferred at only about 15% (H7). The gate and split themselves transferred.

**Reject rule:** J on ≥ 4 panels including D7_I10_W and D12_I5_S. Discount the LWR part to about 30% when predicting the LB.

### I-8. Blackout rows: test-month simulated blackouts + ramp features (**NEW**, low–medium)

**Basis:** A0058 (fullPD, and the ramp member hold7), A0066.
- Ramp flows are released even inside mainline blackouts (about 80% valid).
- Ramp features cut blackout speed RMSE 6.89 → 6.11.
- Training on simulated March/April blackouts cut blackout speed RMSE by 12–14% (7.33 → 6.42 March).

**Weakness it addresses:** about 1.9–2.1% of targets are in blackouts, and they dominate the MSE.

**Exact change:**
- Add ramp features: own link and ±3 links, prev/next slot, net inflow, and on dark rows the change since blackout start.
- Add training blackouts drawn at queue-like origins in the masked test months, away from the released blackouts, blanking T+1..T+18.

**Leakage:** low (Task 1 is offline).

**Expected gain:** +0.0003 to +0.0006. Discount blackout-only local gains to about 1/3; Hisernberg saw 5–50% transfer.

**Reject rule:** blackout RMSE on the simulated test-month blackouts must improve by ≥ 5% on both months.

### I-9. For the **private** file only: add March (validation-month) pseudo-windows as Task 2 training data (**NEW**, medium, rule-sensitive)

**Basis:** rule A0008 ("released data with a timestamp at or before T … any released file"), I-4 windows, the Traffic4cast temporal-domain-adaptation lesson (A0125, A0142, overview A0131), and the month-shift evidence in A0059 §8 (bottleneck activity shifts in March/April on D12_I5_N/S, D7_I405_S and D7_I210_W).

**Weakness it addresses:** onset and ongoing transfer to a new demand draw. Private has 7 incidents against 5 in validation.

**Exact change:** train Task 2 models for private windows on train plus March pseudo-windows, labelled from observed eligible March cells. The validation-file models stay train-only.

**Leakage risk:** medium. Every March timestamp precedes every April T, so this is literally allowed, but **confirm with the organizers** (forum or e-mail) before using it, and disclose it in the report.

**Expected gain:** unknown, probably small (+0.002 to +0.01 private S_queue). It cannot be checked on the public LB.

**Validation:** train on train + March days 1–15, test on March days 16–31 pseudo-windows, against a train-only model. Reject if the delta is ≤ 0.

### I-10. kNN analog-state features for onset (**RELATED**, low–medium)

**Basis:** Traffic4cast 2022 team TSE, 2nd in the extended track: kNN similarity features into a GBDT (A0134, A0206). Bolt, 2nd in core: target encodings plus a PCA "global traffic fingerprint" (A0128, A0225). Overview A0136: target encodings and global-state features were the most important everywhere.

**Weakness it addresses:** onset site and extent when the site is non-recurrent. Our TOD/weekday priors cannot adapt to the current state.

**Exact change:**
- For each onset window, build the corridor vector of speed/v_cut and flow/capacity at T−15..T.
- Find the k = 20–50 nearest train onset-candidate origins (same panel) and average their T+30 queue indicators per link as a feature.
- Also add the top-3 PCA scores of the corridor state as global features.

**Label:** RELATED. Hisernberg's library shape-prior decoder, a posterior over training truth shapes, gained ≤ 0.004 and lost on non-recurrent windows (A0059 §13). Our onset-start prior was rejected.

**Leakage:** none, if neighbours come from train, or from dates before T. **Compute:** low.

**Reject rule:** pseudo-holdout onset (March) ≤ +0.003.

### I-11. Small items (low value; do only if cheap)

- **Month-specific per-(link, month, speed-band) speed-level correction** (A0076, `t1_bias.py`).
  - Correction: log-speed shift with flow scaled by the same factor, so q/v is unchanged. Cross-fitted by day parity and shrunk by n/(n+30).
  - About +0.0001. Only useful after I-1, because transductive members pool both months.
- **Eligibility-aware expected-IoU decoder** (A0070). The official IoU counts only eligible cells (A0029). Eligibility is uniform by design (A0043), so the expected gain is about 0.
- **Onset stage-2 stacking plus more seeds** (A0072): +0.0006 LB for Hisernberg. Ours already uses CatBoost onset; test only if I-3 lands.
- **Final-pick hedge.**
  - Kaggle scores the better of the two selected files on private.
  - Keep one file with the conservative ongoing model, and one with the best pseudo-holdout Task 2 variant (A0074 decision).

---

## 2. FACTS about top teams / generator

### FACT (with source)

**Leaderboard (A0003, Kaggle API `leaderboard/view`, 2026-09-30)**
- Top 5: Pardheev Krishna 0.92406 (date shown 09-12), Musa Peker 0.92226 (09-28), Jason Karpeles 0.91665 (09-19), Kushal Khemani 0.91004 (09-10), KTK 0.90187 (09-30).
- Next: gichang 0.89702 (09-30), Kanak Khemani 0.89449 (09-10), Lukas 0.89260 (09-30), George Daniel Gherasim 0.89198 (09-29), Inocchi 0.88697 (09-30), Giorgio Ottoboni 0.88658 (09-30).
- The board still mixes pre-rebuild scores (rebuild 2026-09-11, A0006). The organizer is asking Kaggle to remove them (A0016).

**Hisernberg's post-rebuild tracking (A0058)**
- On 2026-09-27 they list the post-rebuild top as KTK 0.90085, gichang 0.89628, George Daniel Gherasim 0.89145, Inocchi 0.88697, Giorgio Ottoboni 0.88548. They place themselves 15th.
- They treat Pardheev, Musa Peker, Jason Karpeles and the Khemanis as pre-rebuild, but the dates the API reports for some of these are after the rebuild (see INFERENCE).

**Only documented pipelines above 0.85**
- Hisernberg 0.87249 (A0057). S_queue 0.7878 at H2: onset 0.732, ongoing 0.843. S_ODME 0.9938. S_phys about 0.69 (S_LWR about 0.55).
- happyc0der 0.84481 (A0085).
- No team above 0.88 has published code or a write-up. No top-team technique is publicly known.

**Rules**
- Task 2 may use any released data timestamped ≤ T, including earlier days and the full train split (A0008).
- Onset windows have truth only at T+30 (120/120, A0010).
- Labels come from the noise-free underlying state (A0011, A0021).
- Official IoU is computed only over `is_score_eligible` target cells. A window with no eligible cell scores 0 (A0029).

**Generator: provenance**
- "Calibrated traffic-flow model". Parameters measured from real detector data on the same corridors: free-flow speed, per-lane capacity, jam density, demand by weekday × time of day, measurement-noise size, detector availability. No real rows are replayed (A0018, A0019).
- The pre-synthetic design used Caltrans PeMS, train 2025-06-01..2026-02-28 and validation 2026-03. D12 I-405 was dropped from Queue for poor real observation quality (A0052).

**Generator: scenario contract (A0026)**
- `capacity_multiplier 1.2`, `incident_severity 1.0`.
- A removed field (A0037) said the master seed is a "stable integer from synthetic_release_v1, panel, split", with child seed offsets mainline_observation = 10, ramp_observation = 11, task4_prior = 12.
- So the mainline and ramp observation-noise streams and the Task 4 prior noise are independent random streams per (panel, split).
- After the rebuild, the val/private scenario seed is drawn at random and held by the organizers.

**Generator: rebuild**
- Val and private were regenerated from new seeds. Train is identical cell for cell (A0006).
- Val and private are independent demand draws and incident schedules: 5 incidents vs 7 (A0019, A0045).

**Generator: noise, as measured by participants**
- Speed noise sd about 1.0–1.9 km/h; flow noise about 28 vph/lane, or 3.5–6% (A0063, A0086).
- 5-minute fluctuations are white in time but correlated across links at the same instant: adjacent-link correlation 0.36–0.60 for speed, about 0.3 for flow (A0086).
- Congested observations sit on the congested branch of the released triangular FD: median deviation 0%, IQR ±2.4% (A0058).

**Generator: detector health and masking**
- Detector health is uniform by design. Ramp coverage is 0.7560 on every corridor, so all corridors are Mode A (A0043). About 76% of cells are eligible (A0019).
- The mask is blake2b(panel|regime|date|timestamp|link_id)/2^64 < rate (0.2/0.3/0.5), with one regime per day (A0020).
- Around every scored Task 2 window, the horizon and the hour after it are blanked in all four channels, which blacks out T+1..T+18 (A0036, A0063).
- The Task 1 template = the hash mask minus the window-history cells (2,837 on D7_I210_W val). The visible blanks add 6,980 horizon/buffer cells (A0036).
- Ramps are still released inside mainline blackouts (A0058).
- Lowest flow anywhere in the withheld data: 347 vph (commit 801bba8).

**Task 2 window selector** (A0026, A0059 §1, matched on train)
- History [T−60, T) with eligible coverage ≥ 0.7.
- Ongoing iff ≥ 2 queued eligible observations with ≥ 2 on one link.
- At least one queued horizon cell.
- Persistence IoU of the true state at T ≤ 0.9.
- Greedy chronological, 5 per condition, ≥ 360 min between **any** two picks.
- As a result the official windows sit in the first 0–5 days of each month (A0074).
- The organizer notes the synthetic corridor yields "normal" windows almost only at morning-peak onset (A0026).
- Onset is solvable: an hour before a queue forms, the road is at 1.28× its queue threshold against 1.54× where none follows, carrying 6912 vs 4620 vph (A0026).

**Task 3**
- The conservation signal is about 0.75% of N; the topology flux error is about 0.22% (A0043).
- The practical ceiling is about 0.9636 (A0022).
- Perfect / enhanced-baseline / congestion-erasing submissions score S_LWR 0.9446 / 0.4947 / 0.0 (commit 6eb935c).

**Task 4 (A0060)**
- One path per OD pair, n(n−1)/2 paths. The counted links are pure screenlines, and counts within a segment are identical.
- The prior is gravity-like with power-law deterrence (−1.05·ln(d−o)) and per-split lognormal noise, σ ≈ 0.61.
- The organizer baseline 0.8359 is reproduced (0.8357) under the hypothesis "f* = ridge fit of the split prior to hidden mainline flows, c = A f*".

**The organizer's old enhanced T1 baseline** (profile + same-day temporal/spatial interpolation) scored S_state 0.9083 on validation (A0039, A0053).

### INFERENCE (ours, not stated by any source)

- **The 0.91–0.92 scores.** Four entries in this range carry dates after the rebuild (09-12, 09-19, 09-28). Either the API returns the date of a team's latest submission rather than its best, or some teams really are above 0.91.
  - Hisernberg's tracker treats them as stale, but gives no evidence.
  - Treat the credible post-rebuild frontier as roughly 0.90–0.92.
- **Where the gap to the frontier sits.** Task 4 (0.994) and S_state (≈0.94, ceiling about 0.955) are near saturation for serious teams. The remaining room is S_queue (+0.1 is worth +0.03) and S_LWR (+0.1 is worth +0.01).
  - A team at 0.90 or above probably has S_queue ≥ 0.85, which needs ongoing ≥ 0.90 and onset ≥ 0.80 on March.
  - For us, ongoing (0.747 at the last measurement) is the likeliest large deficit. Re-probe it.
- **Private month.** Private has more incidents (7 vs 5), so possibly more non-recurrent queues than March. Hisernberg's recurrence metric (≤T data) found only 2.5% of private ongoing windows below recurrence 0.05, against 12.5% in March.
  - Model choices tuned on March's non-recurrent mix may be less important on private. Robustness still matters.
- **Noise streams.** Because observation-noise streams have their own child seeds, measurement noise is independent of the incident/demand scenario. Denoising cannot learn anything about the scenario beyond smoothing.

---

## 3. Traffic4cast: transferable lessons (2019–2022)

- **2019–2021 (A0149, A0152, A0131, A0124, A0125, A0137, A0142, A0145).**
  - U-Nets on gridded movies won. In 2021 the task added temporal (COVID) and spatial (unseen city) domain shift, handled with domain adaptation, fine-tuning and multi-task learning.
  - Test-time overlapping patches improved stability.
  - Our corridor is 1-D and tabular GBDT already works, so the transferable part is the **domain-adaptation / transductive** lesson: fit on the target period's own unlabelled or observed data. That is I-1 (Task 1) and I-9 (Task 2 private).
- **2022 (A0136 overview; A0140 ustc-gobbler 1st; A0128/A0225 Bolt 2nd; A0132 oahciy 3rd; A0134/A0206 TSE).**
  - GBDT with heavy feature engineering matched or beat GNNs.
  - The most important features were target encodings (historical label distributions per edge, which match our priors), global-state and time features (PCA fingerprint, inferred time slot), then edge ids and static attributes.
  - The winner imputed missing sensors (VAE) before forecasting. That is analogous to I-3, repairing labels and inputs by imputation.
  - Rare congestion events were the hardest class everywhere. That matches our onset and non-recurrent ongoing weakness.
- **Nothing** in Traffic4cast addresses the IoU-set decoding or the selection artifact specific to this competition.

---

## 4. NEGATIVE evidence (do not repeat without new reason)

**Task 2 decoding**
- Onset logit bias of +0.5 (larger sets): onset −0.016 on March (A0058, F1).
- Site-commit decoders (fewer hedges): −0.012 (F2).
- Contiguous-block, head+extent and library shape-prior decoders all lose (A0059 §12–13). This matches our DROP of morphology decoders.

**Task 2 ongoing** (A0058, A0059 §16–20)
- Stage-2 stacking: −0.023 on March while CV said +0.0065.
- Shrink-only stacking: −0.010.
- Hybrid labels for ongoing: −0.0013.
- 255-leaf capacity: −0.0027 on March, although CV said +0.005 and every gate passed.
- Our own 255/511-leaf ongoing gains may therefore not be real on the LB either. Check them on the March pseudo-holdout (I-4).

**Task 2 onset**
- Covariate-shift importance-weighted onset training loses (A0059 §18).
- Ramp-demand features for onset: −0.0004 (A0086).
- 1-D dilated CNN second opinion for ongoing: OOF 0.8997 vs LightGBM 0.9230, and blends were negative (A0090).
- Onset decoder thresholds are saturated (A0086; our EXISTING_MAP agrees).

**Task 2 validation methods**
- Onset first-slot extent is nearly unpredictable from anything observable at T (|r| ≤ 0.24). It is set within the 5-minute slot (A0059 §13).
- A "post-blackout site check" that reads T+19..T+21 observations was used by Hisernberg only as a diagnostic. **It reads data after T. Do not use it, even for model selection** (A0059 §19).

**Task 1 / Task 3**
- The L1-leaning density model is worse than near-L2 (A0058). Our EXP_T1_m agrees.
- Blackout smoothing is at the LWR floor (A0077).
- Per-day offsets hurt, and partial reconciliation above the 0.6·v_f gate is used up (A0074/A0076).
- Treiber ASM smoothing: speed error 3.88 → 10.81 (A0013).
- Ramp flux injection into conservation made things worse (A0013).
- Wider same-link local-linear smoothing did not help (A0086).
- Snapping v to v_free for S_FD: −0.008 (A0086).
- Kalman/RTS on profile deviations is worse than interpolation (A0122; our EXP_T1_c).
- **Transfer lesson (A0058):** capacity and data-volume gains validated on a train holdout transfer only 15–50% to the LB. Variance reduction and post-processing transfer 85–100%. Validate capacity changes on the test-month pseudo-holdout.

**Task 4**
- Every deviation from the L2 projection / NNLS λ = 0.05 on the split's own counts and prior lost on the LB:
  - λ = 50: −0.084 S_ODME.
  - Relative-deviation form: −0.180.
  - Gravity-smoothed prior: −0.097 / −0.215 (A0086).
  - Rescaling the prior first is harmful (A0060).
- Stop working on Task 4.

**Selection-artifact caution.** Training onset on T+5..T+25 is contradicted by the truth definition (A0010). This matches our EXISTING_MAP DROP.

---

## 5. Honest counts for this partial

A_resources.csv has 228 unique resources:

| Status | Count |
|---|---:|
| DISCOVERED (seen only in search results or listings) | 73 |
| SCREENED | 62 |
| DEEP_READ | 57 |
| CODE_INSPECTED | 36 |

**Gaps**
- The web-search budget ran out mid-task, and the arXiv API rate-limited follow-up queries. Traffic4cast coverage is therefore limited to the 26 arXiv papers, 39 GitHub repositories and about 30 mirror/overview pages found.
- The Kaggle notebook list returned only 8 competition kernels.
