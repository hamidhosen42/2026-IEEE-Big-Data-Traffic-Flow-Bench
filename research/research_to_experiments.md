# Research → experiments (EXP_NEW_001 … EXP_NEW_055)

Updated 2026-10-01. Baseline for every entry: **COMBO5, public LB 0.86217** (see `EXISTING_MAP.md`).

**Sources.** Resource ids `A…`, `B…`, `C…`, `D…` point to rows in `partials/*_resources.csv`. Merged ids (`R…`) are in `resource_index.csv`, whose `source_ids` column maps back to the partial ids.

**Labels:**
- **NEW** = not in our history.
- **RELATED** = a variant of something we did.
- **DUPLICATE** = already done.
- **CONTRADICTED** = our data or evidence says no.
- **HIGH-VALUE RETEST** = we rejected it, but the new evidence says our test was too narrow.

**Status:**
- EXECUTED-KEEP / EXECUTED-DROP = run and decided.
- RUNNING = on Kaggle now.
- PROPOSED = not executed.

**Priority** ≈ gain × evidence × novelty × confidence / (compute × leakage risk), scored 1–5 by judgement, not computed.

**Standard validation gates** (referred to below):
- **T1-holdout:** the val-month internal holdout, 15% of released eligible cells with seed 7, family-mean S_state. KEEP if Δ ≥ +0.0007 and ≥ 6/8 panels improve.
- **T2-CV:**
  - 4 contiguous day-block folds on pseudo windows plus the 80 official train windows, with the competition aggregation.
  - KEEP if 4/4 folds are ≥ 0 and official does not drop by more than 0.003.
  - For shift-driven ideas (prior-free, identity), CV is not informative; an **LB probe** decides.
- **LB probe:** a swap kernel on the exact COMBO base with the CONTROL agreement check at 100%. KEEP if LB Δ > +0.0005 (T1) or > +0.001 (T2 ongoing; SD of a 40-window delta ≈ 0.005–0.009 per A0074, so a single probe is weak evidence).
- **Smoke:** the `SMOKE = True` copy of the kernel (2 panels, every 9th day) must finish without error, with no NaN, all template rows mapped, and plausible ranges.

---

## Task 1 / Task 3 (reconstruction; S_physics follows Task 1 density accuracy)

### EXP_NEW_001: Transductive Task 1 rows from the target months
- **Basis:** A0059/A0058 (Hisernberg log: LB +0.0023 / +0.0011 / +0.0005 for 1× / 2× / 3× rows); D0153 (SAITS masked-imputation training); D0154 (test-time training).
- **Label:** NEW.
- **Weakness:** Task 1 learns only from the 9 train months, but the val and private months have their own demand and incidents.
- **Hypothesis:** released cells of the target month, re-hidden, are the most in-distribution training rows available. Task 1 is offline, so this is allowed.
- **Exact change:** `t1_pipeline_v9.py`, in the per-panel block before the final fit.
  - For split in {validation, private}, run `TR_ROUNDS=2` rounds, hiding `TR_HIDE=0.20` of the released eligible cells (excluding the 15% val holdout, which stays hidden in every view).
  - Compute features on the hidden view. The target is the released value.
  - Append these rows, capped at 4×MAX_ROWS, to the F training rows. Model FT: 511 leaves, lr 0.04, 1500 trees.
- **Task:** T1 + T3.
- **Metric:** S_state holdout, LWR proxy.
- **Smoke:** PASS. FT 0.9364 vs F 0.9350 on 2 panels.
- **Validation:** T1-holdout, then LB probe (swap T1 rows into COMBO6/7).
- **Scores:** compute Medium | leakage Low (holdout disjoint; offline task) | gain High | confidence High.
- **Threshold:** holdout Δ ≥ +0.0007 → LB probe; LB Δ > +0.0005 → KEEP.
- **Status:** RUNNING (`tfb-t1v9-full`). **Priority 5.**

### EXP_NEW_002: ASM/GASM characteristic-kernel features
- **Basis:** C0001/C0002 (ASMx code and parameters, code inspected); C0013, C0014, C0006.
- **Label:** NEW.
- **Weakness:** interpolation features ignore the direction in which traffic waves travel. Congestion moves upstream at about 18 km/h.
- **Exact change:** `t1_pipeline_v10.py`, `asm_feats()` called from `features()`.
  - Mask-normalised kernel averages of visible v/v_f and q/lane.
  - Waves ±80 and ±18 km/h, ±5 slots × ±5 stations, σ=0.6 km, τ=5 min, with real milepost distances.
  - Adds 12 features: 4 waves × (v, q, support).
- **Task:** T1 + T3.
- **Smoke:** PASS. FA 0.9351 vs F 0.9350; the ASM features have low gain importance in the smoke run.
- **Validation:** T1-holdout.
- **Scores:** compute Low–Medium | leakage Low | gain Medium (agent estimate +0.002..0.006, untested on top of GBDT) | confidence Low–Medium.
- **Threshold:** holdout Δ ≥ +0.0007.
- **Status:** RUNNING (`tfb-t1v10-full`). **Priority 4.**

### EXP_NEW_003: Cross-fitted two-stage imputation ("tree message passing")
- **Basis:** C0031 (MICE), C0033 (missForest), C0735, C0260/C0339 (Traffic4cast two-stage GBDT), C0009/C0010 (GRIN/SPIN message passing), C0185 (cross-fitting).
- **Label:** RELATED. Our EXP_T1_e "stacked HGB" was a local single-stage stack, not a fill-and-refeature stage 2.
- **Weakness:** at 20–50% masking, many neighbour inputs are interpolations (holdout 0.907), not model predictions (0.939).
- **Exact change:**
  - Stage 1 = the E-size model, with out-of-fold predictions on train from the 3 frozen day blocks and a full fit for val/private.
  - Fill the masked cells with the stage-1 predictions.
  - Re-run `features()` on the filled cube, then add the stage-1 own prediction as a feature.
  - Stage 2 = the F model.
- **Smoke:** the SMOKE copy; also check that the stage-2 CV gain is ≤ 2× the holdout gain (leakage signal).
- **Scores:** compute High (+2–4 h) | leakage Medium (OOF discipline needed) | gain Medium–High (+0.002..0.008 est.) | confidence Medium.
- **Threshold:** holdout Δ ≥ +0.0015.
- **Status:** PROPOSED. **Priority 4.**

### EXP_NEW_004: Task 1 seed / bagging ensemble
- **Basis:** A0059 (Hisernberg +0.0007 LB; variance-reduction gains transferred 85–95%).
- **Label:** NEW for T1. Our "3 seeds no gain" was measured on T2 ongoing.
- **Exact change:** in the final fit, train the F model with 3 seeds (`seed`, `bagging_seed`, `feature_fraction_seed` = 0/1/2) on different row subsamples, and average the residual predictions.
- **Scores:** compute Medium (3× F) | leakage Low | gain Low–Medium | confidence Medium–High.
- **Threshold:** holdout Δ ≥ +0.0005.
- **Status:** PROPOSED. **Priority 4.**

### EXP_NEW_005: TV smoothing of density inside runs of consecutive target cells
- **Basis:** A0059 §TV (+0.0006 LB). No TV-denoising paper is in the index; this is the classic Condat 1-D TV algorithm (not indexed).
- **Label:** NEW.
- **Weakness:** LWR scores ΔN between adjacent links, and per-cell errors in q/v create spurious spatial jumps.
- **Exact change:** post-process predicted k = q/v along space, only inside runs of ≥ 2 adjacent target cells. Use 1-D TV (Condat) with λ tuned on the holdout LWR proxy, then re-split k into (v, q) with 75/25 speed/flow weighting.
- **Scores:** compute Low | leakage Low | gain Low–Medium | confidence Medium.
- **Threshold:** holdout S_state not worse by more than 0.0002 **and** LWR proxy Δ ≥ +0.002.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_006: Gated density reconciliation (v < 0.6·v_f only)
- **Basis:** A0059 (+0.0006..0.002 local).
- **Label:** HIGH-VALUE RETEST. Our ungated density pull (single panel) was 0.
- **Exact change:** reconcile q/v toward the interpolated density only where the predicted v < 0.6·v_f, with a 75/25 speed/flow split.
- **Scores:** compute Low | leakage Low | gain Low | confidence Low–Medium.
- **Threshold:** T1+T3 holdout Δ > 0 on ≥ 6/8 panels.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_007: Blackout-specific training rows plus ramp features
- **Basis:** A0058, A0063 (ramps are released inside mainline blackouts; blackout speed RMSE −12..−14%).
- **Label:** NEW. We already simulate the 90-min blackouts but use no ramp channel.
- **Exact change:**
  - Load `ramp_states` for the same panel and day.
  - Add on-ramp and off-ramp flows, at t and t±1, at the nearest ramp up- and downstream of each station.
  - Add the ramp-implied mainline flow balance.
  - Used by all rows, which matters mostly in the gaps (about 1.9% of targets).
- **Scores:** compute Medium | leakage Low (Task 1 is offline) | gain Low–Medium | confidence Medium.
- **Threshold:** holdout Δ on the gap cells > 0 and overall ≥ 0.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_008: Mask-support features and mask-rate matching
- **Basis:** C0008 (ImputeFormer Table 7), C0007.
- **Label:** RELATED (we have `rate` and `gp`/`gn`).
- **Exact change:**
  - Add the local observed fraction in 3×3, 5×5 and 7×7 windows.
  - Check whether the neighbour-availability distribution of the holdout is sparser than in train. The holdout hides an extra 15%; the test does not.
- **Scores:** compute Low | leakage Low | gain Low | confidence Medium.
- **Threshold:** Δ ≥ +0.0005.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_009: Masked low-rank completion (SoftImpute / LATC) as features
- **Basis:** C0737–C0739 (SoftImpute implementations), C0005 (LATC, code inspected), C0216/C0217, D-B5.
- **Label:** HIGH-VALUE RETEST. Today's SVD runs on the *interpolated* field and inherits the gap bias.
- **Exact change:** `lowrank()` becomes an iterative SoftImpute on the observed cells only (rank 5, λ from the singular-value gap, 20 iterations), per day and cross-day.
- **Scores:** compute Medium | leakage Low | gain Low–Medium | confidence Medium.
- **Threshold:** Δ ≥ +0.0007.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_010: Conservation / CTM one-step / Newell N-curve features
- **Basis:** C0015 (CTM), C0062/C0063 (Newell simplified kinematic waves), C0275; D0131 (Newell three-detector features).
- **Label:** NEW as features. Only post-processing versions were tried.
- **Exact change:**
  - Upstream and downstream flow-balance estimates, using k = q/max(v, 1).
  - A Godunov one-step forecast from t−1 and a backward step from t+1, with the per-link triangular FD from `fd_parameters.csv`.
  - Newell minimum of the free- and congested-shifted cumulative counts.
- **Scores:** compute Low | leakage Low | gain Low–Medium | confidence Low–Medium.
- **Threshold:** T1+T3 proxy Δ > 0.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_011: Per-link FD-implied features
- **Basis:** C0099, C0109.
- **Label:** NEW. The occupancy→speed identity sub-idea is CONTRADICTED; see EXP_NEW_054.
- **Exact change:** features v_FD(k̂) and q_FD(k̂) from the interpolated density, using each link's own v_f, capacity and k_jam.
- **Scores:** compute Low | leakage Low | gain Low | confidence Low.
- **Threshold:** Δ ≥ +0.0005.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_012: LightGBM `linear_tree`
- **Basis:** D0171, D0176.
- **Label:** NEW.
- **Exact change:** E/F params plus `linear_tree=True, linear_lambda=1.0`. Test at 2× rows first, because of memory.
- **Scores:** compute Medium | leakage Low | gain −0.002..+0.003 | confidence Low.
- **Threshold:** Δ ≥ +0.0007.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_013: LightGBM `extra_trees` / `path_smooth`
- **Basis:** D0171, D0174.
- **Label:** NEW.
- **Scores:** gain ±0.001 | confidence Low.
- **Threshold:** Δ ≥ +0.0007.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_014: `deterministic=True, force_row_wise=True` in every LightGBM model
- **Basis:** D0171.
- **Label:** NEW (process).
- **Weakness:** the awards require a rerun within 1% of the private score.
- **Gain:** none; it is a reproducibility guard.
- **Threshold:** the rerun matches bit-for-bit, or up to 1e-6.
- **Status:** PROPOSED; must be in the final notebook. **Priority 5 (process).**

### EXP_NEW_015: Small CPU space-time U-Net inpainting net as a stacked feature
- **Basis:** C0045 (U-Net), C0049 (partial-convolution inpainting), C0051; A (Traffic4cast U-Net solutions).
- **Label:** NEW.
- **Scores:** compute High on CPU | gain uncertain | confidence Low.
- **Threshold:** Δ ≥ +0.0015.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_016: Calibrated CTM data assimilation (nudging) as a feature
- **Basis:** C0275 (CTM density estimation), C0483, C0488, C0489 (traffic data assimilation / EnKF), C0068 (CTM calibration).
- **Label:** NEW, risky.
- **Weakness:** the data come from a CTM-like generator, so a calibrated simulator could fill gaps better than statistics can.
- **Scores:** compute High | gain uncertain | confidence Low.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_017: Characteristic-sheared or Hankel low-rank completion
- **Basis:** C-B4.
- **Label:** NEW, speculative.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_018: Month-specific speed-level correction per (link, month, speed band)
- **Basis:** A0076 (+0.0001).
- **Label:** NEW.
- **Scores:** gain very low.
- **Status:** PROPOSED; only after EXP_NEW_001. **Priority 1.**

### EXP_NEW_019: Denoised neighbour inputs (median-3 on visible values)
- **Basis:** C-A6.
- **Label:** NEW.
- **Scores:** gain Low. Noise is white in time, sd ≈ 1–1.9 km/h (A0063).
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_020: Bigger Task 1 model on top of F (1023 leaves, or 8× rows)
- **Basis:** own history (E +0.0024, F +0.0012, diminishing).
- **Label:** RELATED.
- **Scores:** expected gain < 0.0006; compute High.
- **Status:** PROPOSED. **Priority 1.**

---

## Task 2 (queue forecast)

### EXP_NEW_021: Ongoing — identity-kept prior-free model and full model, 50/50 blends
- **Basis:** A0059 (Hisernberg v5: the only ongoing change that transferred to March, +0.008 S_queue; dropping link identity hurt every slice).
- **Label:** HIGH-VALUE RETEST. Our prior-free model also drops `pos`, `lanes`, `cap_*` and `length`.
- **Hypothesis:** location *priors* overfit the month; link *identity* (geometry) does not.
- **Exact change:** `t2_model_v14.py`.
  - `og_id` = LightGBM with 511 leaves, lr 0.03, 1800 trees, trained without {prior, prior_now, prior_alldays, tod, dow}.
  - `og_full` = all columns.
  - Variants: `id511`, `full511`, `bnf` = ½(np511 + full511), `bif` = ½(id511 + full511).
  - The onset rows are identical to v13.
- **Validation:** an LB probe per variant, since CV cannot see the month shift. Use the Task 2 pseudo-holdout (EXP_NEW_025) once it exists.
- **Scores:** compute Medium | leakage Low | gain Medium–High (ongoing is the largest gap: Hisernberg 0.843 vs our last measured 0.747) | confidence Medium.
- **Threshold:** LB Δ ≥ +0.001 vs COMBO7.
- **Status:** RUNNING (`tfb-t2v14`). **Priority 5.**

### EXP_NEW_022: Ongoing 511 leaves / lr 0.03 / 1800 trees
- **Basis:** own oofY run.
- **Label:** RELATED (capacity).
- **Result:** pseudo 0.9017 vs 0.8958 and official 0.8626 vs 0.8551, better in 4/4 folds. Doubling the data at 255 leaves gave ≈ 0.
- **Status:** EXECUTED-KEEP → `tfb-t2v13` → COMBO7 (auto-submit 2026-10-01).

### EXP_NEW_023: Re-measure ongoing and onset IoU on the current best (zero-probe)
- **Basis:** A0058 (their probes); our 09-28 probes.
- **Label:** RELATED (diagnostic).
- **Weakness:** our 0.747 ongoing figure is three model generations old. Estimated now ≈ 0.81 (INFERENCE from the LB deltas).
- **Exact change:** a swap kernel from COMBO7 with all ongoing `queue_pred` set to 0. Then ΔLB ≈ 0.30 × 0.5 × IoU_ongoing.
- **Scores:** compute Low | costs 1 submission.
- **Status:** PROPOSED; do it when a submission slot is free. **Priority 3.**

### EXP_NEW_024: Hybrid label repair
- **Basis:** A0059 §13, A0068 (onset +0.030 on their LB).
- **Label:** RELATED. Our labels exclude missing cells instead of time-filling them, so our bias is smaller than theirs was.
- **Exact change:**
  - Train a LightGBM speed imputer on observed train cells.
  - Fill missing cells within ±5 min and ±2 links of an observed queued cell.
  - Rerun the selector replica, then retrain onset CatBoost.
- **Scores:** compute Medium | leakage Low | gain Low–Medium | confidence Low–Medium.
- **Threshold:** March pseudo-holdout onset Δ ≥ +0.005.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_025: Task 2 test-month pseudo-holdout as the adoption gate
- **Basis:** A0074, A0075 (Hisernberg: it caught a stacking failure that CV missed).
- **Label:** NEW (validation).
- **Exact change:**
  - Run our selector replica on the **masked validation month**: coverage ≥ 0.7, the same condition rules, greedy with 360 min from every start day. Exclude windows touching released blackouts.
  - Labels = observed eligible horizon cells.
  - Score with the official aggregation.
  - Use it to gate **private-file** decisions only. All March timestamps precede every April T. Never select using April labels.
- **Scores:** compute Low | leakage Medium (protocol-sensitive) | gain: decision quality | confidence Medium.
- **Threshold:** it must reproduce the sign of known LB deltas (prior-free +, persistence blend −, onset 30/70 −) before we trust it.
- **Status:** PROPOSED. **Priority 4.**

### EXP_NEW_026: For the private file only, add validation-month pseudo-windows as Task 2 training data
- **Basis:** A-I-9.
- **Label:** NEW.
- **Rule note:** March data is timestamped before every April T, so it satisfies "≤ T". Confirm with the organizers before using it.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_027: kNN analog-state features for onset
- **Basis:** Traffic4cast 2022 GBDT solutions (C0260, C0339, and the A partial's Traffic4cast rows).
- **Label:** RELATED. Hisernberg's shape-prior decoder gained ≤ 0.004 and lost on non-recurrent windows.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_028: Onset stage-2 stacking plus more seeds
- **Basis:** A0072 (+0.0006 LB).
- **Label:** RELATED.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_029: Dependence-aware (component-mixture) onset decoder
- **Basis:** B (Nowozin 2014 deep-read; RankSEG).
- **Local:** onset +0.0015 pseudo, 4/4 folds ≥ 0, official unchanged. About +0.0002 LB.
- **Label:** NEW.
- **Status:** PROPOSED (bundle with another change). **Priority 2.**

### EXP_NEW_030: Exact Poisson-binomial expected-IoU decoder for ongoing
- **Basis:** B (RankSEG code inspected).
- **Local:** ongoing +0.0006 in 4/4 folds; official unchanged.
- **Label:** NEW.
- **Status:** PROPOSED (bundle). **Priority 2.**

### EXP_NEW_031: Shift diagnostic
- **Basis:** B-C3.
- **Label:** NEW (diagnostic).
- **Exact change:** compare max p and Σp of our predictions on the 40 + 40 val/private windows against train OOF. Predict LB onset IoU from the confidence bins: 91% of OOF onset windows have max p > 0.9 (IoU 0.937), and the 3% with max p < 0.5 score IoU 0.10.
- **Leakage note:** this is a read-out only; nothing is fitted on val.
- **Status:** PROPOSED. **Priority 3.**

### EXP_NEW_032: Tail-mass safeguard (zero p < ε before decoding)
- **Basis:** B-C4.
- **Local:** ±0.0002.
- **Status:** PROPOSED (robustness only). **Priority 1.**

### EXP_NEW_033: Denoised / soft training labels
- **Basis:** B0004, B0041, B0042 (B-R1, D-A2).
- **Label:** NEW.
- **Pre-check (EXECUTED 2026-10-01 on the local D12_I405_S train, 120 days):**
  - queued share 1.28%;
  - **isolated queued cells 0.9%**;
  - a median-3 relabel changes about 1.9% of queued cells.
- **Decision:** label noise is too small to explain the 0.84 CV→LB ratio. **Deprioritized**, per the agent's own rule (skip if isolated < 1%).
- **Status:** EXECUTED pre-check → DEFER. **Priority 1.**

### EXP_NEW_034: Breakdown-probability (stochastic-capacity) features for onset
- **Basis:** B (Brilon et al.; Elefteriadou).
- **Label:** RELATED. Our hazard / flow-capacity features were noise.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_035: Window-level activation / queue-size model feeding the decoder
- **Basis:** B-R3, D-B2 (D0145, D0146).
- **Label:** NEW.
- **Exact change:** predict the queue size per window and use it when choosing m in top-m decoding.
- **Status:** PROPOSED. **Priority 2.**

### EXP_NEW_036: Causal (≤ T) prior-shift / EM prevalence correction
- **Basis:** D0161, D0163, D0165.
- **Label:** NEW, but rule-constrained: only windows with origin < T may be pooled.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_037: Change-point (CUSUM / BOCPD) history features
- **Basis:** B-R5.
- **Label:** DUPLICATE of our rejected hazard / gradient features.
- **Status:** DROP.

### EXP_NEW_038: LB-faithful local truth (denoised state, all links)
- **Basis:** B-V1.
- **Label:** NEW (validation).
- **Note:** the pre-check in EXP_NEW_033 shows label noise is small, so the expected change in rankings is small.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_039: Importance-weighted CV toward weekend share
- **Basis:** B-V2, D-A6.
- **Label:** CONTRADICTED. Weekends are *easier* locally (onset 0.926 vs 0.881), so re-weighting widens the gap.
- **Status:** DROP.

### EXP_NEW_040: Probability calibration (isotonic, power) before expected-IoU decoding
- **Basis:** D-A1, B-C (Platt / isotonic / beta).
- **Label:** NEW.
- **Result (EXECUTED 2026-09-30, oofX cache):**

  | Variant | Pseudo | Official |
  |---|---|---|
  | Base | 0.8831 | 0.8904 |
  | Power 1.1 | 0.8845 | 0.8898 |
  | Nested isotonic | 0.8808 | 0.8924 |

  The signs disagree between pseudo and official. B measured that temperature ≥ 2 collapses onset.
- **Status:** EXECUTED-DROP.

### EXP_NEW_041: CTM forward rollout from the state at T as an onset feature
- **Basis:** D-B1; C0015.
- **Label:** RELATED. Likely noise, like our other physics features.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_042: Neural onset member (RealMLP / TabM) blended with CatBoost
- **Basis:** D-B3.
- **Label:** RELATED.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_043: Onset — keep CatBoost depth 8; bigger variants are noise
- **Basis:** own oofX.
- **Label:** DUPLICATE (done).
- **Status:** EXECUTED-DROP.

---

## Task 4 (ODME)

### EXP_NEW_044: KL / IPF or prior-weighted GLS projection
- **Basis:** D-B4.
- **Label:** NEW, but capped: with exact counts, methods differ only in the null space the prior fixes. At most about 0.0012 of S_total, and it cannot be checked locally.
- **Status:** PROPOSED. **Priority 1.**

### EXP_NEW_045: Stop Task 4 work
- **Basis:**
  - D: `repo/src/task4/build_task4_odme_artifacts.py` is exactly our NNLS with √0.05·I.
  - A0060: the truth is a ridge fit to the prior.
- **Status:** DECISION.

---

## Process / validation / selection

### EXP_NEW_046: Final-pick hedge
- **Basis:** A-I-11.
- **Exact change:** of the 2 final selections, one uses the conservative ongoing model and the other the best pseudo-holdout Task 2 variant.
- **Status:** PROPOSED (for the Nov 6 selection). **Priority 4.**

### EXP_NEW_047: LB-delta noise discipline
- **Basis:** D-A7, A0074.
- **Rule:** a single Task 2 ongoing LB Δ < 0.002 is within noise. Require CV or pseudo-holdout agreement before keeping it.
- **Status:** ADOPTED.

### EXP_NEW_048: Single-entry reproducible notebook for the final pipeline
- **Basis:** the rules (rerun within 1%).
- **Label:** NEW (required).
- **Status:** PROPOSED, before Nov 6. **Priority 5 (process).**

---

## Rejected by evidence (listed so no compute goes to them)

### EXP_NEW_049: Time-series foundation models (Chronos / TimesFM / Moirai) for Task 2
- **Label:** CONTRADICTED. A 12-step history is shorter than one input patch, and these models are weakest on transport data in GIFT-Eval (D rows).

### EXP_NEW_050: GNN / spatio-temporal forecasting frameworks (LibCity, BasicTS, STGCN, DCRNN, …)
- **Label:** CONTRADICTED for us.
- **Reasons:**
  - 1-D corridors, CPU only;
  - plain MLPs match ST-GNNs (D);
  - windowed GBRT beat 8 deep models on 9 datasets (D).

### EXP_NEW_051: PINN / physics-loss training
- **Label:** CONTRADICTED.
- **Reasons:**
  - it helps only at very sparse sensing and fails at shocks (C0011, C0012);
  - our density reconciliations failed.

### EXP_NEW_052: Mixture-of-experts / regime gating for Task 1
- **Label:** CONTRADICTED by our per-regime result (−0.002 on every panel).

### EXP_NEW_053: Deep imputers (SAITS, BRITS, GRIN, SPIN, ImputeFormer, CSDI / PriSTI diffusion)
- **Label:** CONTRADICTED on cost and evidence.
- **Reasons:**
  - their papers never compare against GBDT;
  - they need 4–26 GPU-hours;
  - diffusion is too slow on CPU (C).

### EXP_NEW_054: Occupancy → speed identity (speed ≈ flow·g / occupancy at masked cells)
- **Basis:** C-A5 step 0.
- **Label:** CONTRADICTED by our diagnostic (2026-10-01, local private D12_I405_S). Where speed is masked, flow and occupancy are also masked in 100% of cells; they are co-masked.
- **Status:** EXECUTED-DROP.

### EXP_NEW_055: Task 1 profile features / Kalman / per-lane spatial interpolation
- **Label:** DUPLICATE. All were already rejected (see `EXISTING_MAP.md`).

---

## Cheap gains (no retraining, or a single config change)

| ID | What | Status |
|---|---|---|
| 022 | ongoing 511 leaves | done → COMBO7 |
| 029 / 030 | block-aware onset decoder / exact ongoing decoder | ≈ +0.0002 each, bundle |
| 014 | deterministic flags | required, no gain |
| 023 | zero-probe re-measure | diagnostic, 1 submission |
| 031 | shift diagnostic | diagnostic |
| 040 | calibration | **tested, no gain** |
| 032 | tail-mass safeguard | ±0.0002 |

## High-upside structural gains

| ID | What | Expected |
|---|---|---|
| 001 | transductive Task 1 | RUNNING; LB +0.002..0.004 expected (Hisernberg evidence) |
| 021 | ongoing identity / full blends | RUNNING; up to +0.003..0.005 if the ongoing gap is as large as estimated |
| 003 | cross-fitted two-stage Task 1 | +0.002..0.008 on holdout (est.) |
| 025 | March pseudo-holdout gate | decision quality for the private file |
| 024 / 026 | label repair; March pseudo-windows for the private file | rule check needed for 026 |
