# EXP_021 – EXP_040 roadmap

Written 2026-10-01, after the four prerequisites:
- organizer facts: `organizer_verified_facts.md`;
- the 0.86217 vs 0.85860 diff: `86217_vs_85860_diff.md`;
- the Task 2 ceiling: `SCORE_GAP_ANALYSIS.md`;
- the topology audit: `topology_audit.md`.

**What the evidence says about where the gap is:**
- The gap to about 0.90 is almost entirely **Task 2**. A frontier score of 0.90 implies S_queue ≈ 0.89, against our ≈ 0.78.
- Task 3 LWR is pinned by noise: every reachable predictor scores 0.52 in the local prototype.
- Task 1 has at most about +0.0035 left.
- Task 4 is saturated.

**Common rules:**
- Runtimes are for Kaggle 4-CPU kernels.
- A "smoke" run is the `SMOKE = True` copy: 2 panels, every 9th day. It must finish with no NaN, all rows mapped, and plausible ranges.
- The T2-OOF gate uses the same day-block folds, windows and seeds as oofX; pseudo is 1,000+ windows and official is 80.
- The T1 gate is the val-month holdout.
- LB probes are one-component swaps on the current best, with CONTROL agreement = 1.0.

## P0

### EXP_021 — Task 2: travel-direction consistent link order — **DONE: DROP**

Result: pseudo −0.0032, negative in 3 of 4 folds. Ongoing was unchanged (+0.0006); onset dropped, most in the W/S panels (−0.007). Official was +0.0037, but that is only 80 windows. The model already compensates for the direction mix.
- **Hypothesis:** the pooled Task 2 model sees "downstream" features that are physically upstream in the W/S panels (topology audit). Consistent direction lets it learn queue spill-back and upstream growth.
- **Where:** `t2_model_v13.py` / `t2_oofF.py`, `load_panel()`.
- **Change:** after `sort_values('order_index')`, reverse the link order when `panel[-1] in 'WS'`.
- **Runtime:** OOF about 2 h; production about 5 h.
- **Metric:** onset and ongoing IoU (T2-OOF).
- **Smoke:** PASS (`tfb-ooff-smoke`).
- **Validation:** window-matched comparison with the oofX P255 / O_cat8 controls, per fold and per panel group (E/N vs W/S).
- **KEEP:** Δ ≥ +0.003 pseudo in ≥ 3/4 folds and official not below −0.003. Then build a production v15 (v13 + flip) and run an LB probe.
- **DROP:** otherwise.

### EXP_022 — Task 1: transductive target-month rows (RUNNING, `tfb-t1v9-full`)
- **Hypothesis:** released val/private cells, hidden in rounds, are in-distribution training rows. Task 1 is offline, which is organizer-confirmed.
- **Where:** `t1_pipeline_v9.py`.
- **Change:** `TR_ROUNDS=2`, `TR_HIDE=0.20`, FT model (F params).
- **Runtime:** about 5 h.
- **Metric:** S_state, and S_LWR through the same rows.
- **Smoke:** PASS, 0.9364 vs 0.9350.
- **KEEP:** holdout Δ ≥ +0.0007 on ≥ 6/8 families, then an LB swap Δ > +0.0005.
- **DROP:** otherwise.

### EXP_023 — Task 2: ongoing with link identity (RUNNING, `tfb-t2v14`)
- **Hypothesis:** location *priors* overfit the month, but link identity (lanes, capacity, length, position) does not.
- **Change:** `og_id` (drops only prior / tod / dow), `og_full`, and 50/50 blends.
- **Metric:** ongoing IoU on the LB. CV cannot see month shift.
- **KEEP:** LB Δ ≥ +0.001 against COMBO7.
- **Note:** re-run on the flipped order if EXP_021 is kept.

### EXP_024 — Task 2 test-month pseudo-holdout gate
- **Hypothesis:** the CV→LB ratio of about 0.84 is systematic, and persistence shows the same 0.82. The val-month windows are intrinsically different, and we need a gate built on that month.
- **Where:** new `t2_pseudo_march.py`:
  - reuse `build_windows()` / `window_features()` on `validation/mainline_states_masked`;
  - use the observed eligible labels;
  - exclude windows touching released blackouts;
  - score cached models.
- **Runtime:** about 1 h.
- **Metric:** onset and ongoing IoU on March pseudo-windows.
- **Smoke:** 2 panels.
- **Validation of the gate itself:** it must reproduce the signs of three known LB deltas: prior-free ongoing +, persistence blend −, onset 30/70 −.
- **KEEP as the gate:** if it matches 3/3. **Protocol:** March is used to decide the **private** file only.

### EXP_025 — measure S_LWR on the val holdout inside the Task 1 kernel (diagnostic)
- **Change:** in the T1 kernel, compute the LWR proxy on holdout transitions whose two endpoints are known (holdout or released), for the temporal, F and FT variants.
- **Runtime:** a few minutes.
- **Purpose:** confirm on real val-month data the local finding that the model does not raise LWR, before any further physics work.
- **DROP physics work:** if F − temporal < +0.01 S_LWR.

## P1

### EXP_026 — Task 1 seed bagging
- **Change:** 3 × FT with seeds 0/1/2 and different row subsamples, averaged.
- **Runtime:** about 3 × FT.
- **KEEP:** holdout Δ ≥ +0.0005.

### EXP_027 — Task 1 mask-severity features (user C1)
- **New features:**
  - longest-gap length through the cell;
  - local observed fraction in 3×3, 5×5 and 7×7;
  - observed fraction of the link-day and the time-slot;
  - distance in km (not index) to the nearest observation along space.
- **Existing features:** `rate`, `gp` and `gn` already cover regime and gap distance.
- **KEEP:** Δ ≥ +0.0007, with R3 not worse.

### EXP_028 — Task 1 per-regime residual scaling (user C2)
- **Change:** `pred = interp + α_r · resid`, with α_r for R1, R2 and R3 chosen on train CV folds (not on the holdout), then checked on the holdout.
- **Runtime:** minutes.
- **KEEP:** Δ ≥ +0.0003 with α ≠ 1 stable across folds.
- **Note:** per-regime *models* (C3) were already rejected (−0.002).

### EXP_029 — graph-aware distances for Task 1 and Task 2 (user EXP_B)
- **Change:** km spacing to the neighbours at ±1..3; lane-drop and lane-add indicators up- and downstream; capacity-drop ratio in km; distance to the nearest on/off ramp.
- **Why:** spacing ranges from 0.1 to 7.8 km (topology audit), but the neighbour features are index-based.
- **Task 2:** EXP_021 was dropped, so use symmetric ±k distances.
- **KEEP:** T1 holdout Δ ≥ +0.0007 / T2-OOF Δ ≥ +0.003.

### EXP_030 — released ramp flows as features
- **Task 1:** any time.
- **Task 2:** at or before T only; the organizer confirmed ramp_states are allowed up to T.
- **Change:** on/off-ramp flow at t (T1) or at T and T−1 (T2) for the ramps attached to the link and its neighbours; on-ramp surge as an onset precursor.
- **KEEP:** as in EXP_029.

### EXP_031 — Task 2 onset: bottleneck propensity × live instability (user T2_D)
- **Change:** explicit interaction features:
  - `prior_now × (1 − vr_last)`;
  - `prior(T+30) × fr_last / cap`;
  - `prior × downstream-min-speed drop`;
  - `prior × on-ramp surge`.
- **Runtime:** OOF about 2 h.
- **KEEP:** onset T2-OOF Δ ≥ +0.004 in ≥ 3/4 folds.
- **Note:** plain hazard/gradient features were noise; this is a different representation.

### EXP_032 — Task 2 onset as ranking (user T2_B)
- **Change:** LightGBM `lambdarank` with group = window and label = queued at T+30. Scores are converted to probabilities by per-window softmax calibration fitted on OOF, then passed to the decoder.
- **KEEP:** onset T2-OOF Δ ≥ +0.004.

### EXP_033 — Task 2 decoders bundled
- **Change:** block-aware onset decoder plus exact Poisson-binomial ongoing decoder. Measured on OOF: ≈ +0.0015 onset and +0.0006 ongoing.
- **Runtime:** minutes.
- **KEEP:** 4/4 folds ≥ 0. Ship it only inside another submission.

### EXP_034 — Task 2 same-split, earlier-days priors (organizer-confirmed legal)
- **Change:** for a private window at T, add prior tables built from released masked observations of the whole March month plus April days before T. For validation windows, use March days before T.
- **KEEP:** onset Δ ≥ +0.003 on the EXP_024 gate.
- **Risk:** val and private are independent draws, so the transfer may be weak.

## P2 and rejected

- **EXP_035 — Task 1 low-rank auxiliary with masked SoftImpute (user EXP_D).** KEEP if Δ ≥ +0.0007, with R3 reported separately.
- **EXP_036 — Task 1 cross-fitted two-stage imputation (user EXP_F).**
  - Stage-1 OOF fill, then refeaturise.
  - It replaces the "error-conditional second stage", which is weaker: it corrects only flagged cells.
  - KEEP if Δ ≥ +0.0015.
- **EXP_037 — Task 2 probability calibration (user T2_A).** **Already tested and DROPPED:** pseudo and official signs disagree, and B's temperature scan hurts. Not re-run.
- **EXP_038 — Task 2 uncertainty-aware m (user T2_E).** Covered by EXP_033 and B's window-size analysis. The expected-IoU decoder already picks m per window, not a fixed m.
- **EXP_039 — Task 1 mixture of experts (user EXP_E).** **Deprioritised.**
  - The residual LightGBM already takes interpolation, spatial, low-rank and profile inputs as features, which is a learned gate.
  - Per-regime experts lost 0.002.
  - Run only if EXP_027 shows regime-dependent residual structure.
- **EXP_040 — single-entry reproducible notebook** (awards; `deterministic=True`). Needed before Nov 6.

**Not repeated (Phase 5):**
- onset T+5..T+25;
- shockwave / queue-boundary features;
- window weighting;
- simple LWR / density pulls;
- OD λ sweep;
- CatBoost ongoing blend;
- contiguity / morphology;
- Task 2 calibration;
- same-instant high-pass noise features: new, but tested locally today, and they reduced S_LWR;
- Task 2 label denoising: onset cells are 26 km/h below the cut, and only 0.9% of queued cells are isolated.

## TOP 5 to execute now

| # | EXP | Status |
|---:|---|---|
| 1 | EXP_021 direction fix | **DROP** (−0.0032 pseudo, 3/4 folds negative) |
| 2 | EXP_022 transductive Task 1 | RUNNING → LB swap if KEEP |
| 3 | EXP_023 ongoing identity | RUNNING (v14) → LB probes of `bif` / `bnf` |
| 4 | EXP_024 March pseudo-holdout gate | build next; needs a free Kaggle slot |
| 5 | EXP_031 onset propensity × instability + EXP_030 ramp features | one OOF run (`t2_oofR.py`) on the original order, queued |
