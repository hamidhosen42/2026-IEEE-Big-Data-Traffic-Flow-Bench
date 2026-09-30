# D: model families, ensembling, HPO, shift-aware validation, ODME — experiment ideas

Scope: source family D. Resource IDs refer to `D_resources.csv`. Each idea was checked against `EXISTING_MAP.md` (2026-09-30) and `experiments/TRACKER.md`.

**Two facts from the code shaped this list.**

1. `kaggle_jobs/t1_pipeline_v8.py` builds Task 1 training rows **only from the `train` split**. The validation and private months are used only for prediction and for the internal 15% holdout. Nothing in the pipeline learns from the target month's own released cells.
2. The organizer's Task 4 reference (`repo/src/task4/build_task4_odme_artifacts.py`, D0363) is the same NNLS with `REG_LAMBDA = 0.05` on `[A; √λ·I]` that we run. Our Task 4 therefore already equals the public reference. The remaining headroom is at most 0.006 of S_ODME, which is at most **0.0012** of the total score.

Labels:
- **NEW**: not tried in the map.
- **RELATED**: close to something tried, but the mechanism differs.
- **DUPLICATE**: already done.
- **CONTRADICTED**: the evidence or our own results argue against it.
- **HIGH-VALUE RETEST**: tried before, but worth re-running under a changed condition.

---

## A. Cheap gains (post-hoc, or one retrain of an existing model)

### A1. Task 2: calibrate probabilities, then adjust for causal prior shift, before expected-IoU decoding. Label: NEW.

- **Basis:**
  - D0256: boosted trees are miscalibrated; Platt or isotonic scaling fixes this.
  - D0163: Saerens–Latinne–Decaestecker EM re-estimates target priors from unlabeled posteriors.
  - D0165: the EM approach (MLLS) plus calibration beats BBSE.
  - D0161: AdapTable, a model-agnostic calibrator plus label-distribution handler for tabular shift.
  - D0160: TableShift, where the shift gap tracks label-distribution shift.
  - D0144: for F-type metrics, the optimal threshold depends on calibrated probabilities.
- **Weakness addressed:**
  - The CV→LB ratio is about 0.84 and systematic.
  - The validation month has more non-recurrent queues and more weekend windows.
  - Expected-IoU top-m decoding uses probability magnitudes to choose m. A prevalence shift therefore moves the IoU-optimal m even when the cell ranking is right.
- **Exact change:**
  1. On the OOF cache (tfb-t2oof / oofY), fit an isotonic map per condition (onset, ongoing) on 3 folds. Apply it to the 4th, and rotate.
  2. At inference, for each split × condition, estimate the target prevalence π̂ with the SLD EM on the calibrated probabilities of candidate cells. Only windows with origin ≤ the current T may be used (a running estimate), so the Task 2 "data ≤ T" rule holds. Then set `p' = r·p / (r·p + s·(1−p))`, where `r = π̂/π_tr` and `s = (1−π̂)/(1−π_tr)`.
  3. Shrink halfway (`r ← √r`), and clip r to [0.67, 1.5]. Incidents violate the p(x|y)-invariance assumption, so full correction is unsafe.
  4. Decode exactly as now.
- **Compute:** minutes on the OOF cache. Needs no retraining.
- **Expected gain:** 0 to +0.004 LB. Isotonic scaling alone changes only m, not the ordering.
- **Validation:**
  1. Leave one pseudo fold out and estimate π̂ blind on it. Compare IoU for raw, calibrated, and calibrated + SLD.
  2. Also check the 40 official onset windows.
  3. Log π̂/π_tr per month to see whether the validation month really is shifted.
- **Reject rule:** reject if it is not ≥ +0.002 IoU in ≥ 3/4 folds, or if the official-window IoU drops by more than 0.005. If the causal-only estimate cannot be implemented cleanly, drop the SLD step and keep calibration only.

### A2. Task 2: denoise the training labels to match the noise-free truth. Label: NEW, high value.

- **Basis:**
  - D0155: Noise2Self. A J-invariant denoiser can be calibrated from noisy data alone.
  - D0253: confident learning. Estimate and prune label noise from OOF probabilities.
  - D0151: Robust-GBDT, with non-convex losses under label noise.
  - D0147: Jaccard losses work with soft labels.
  - D0169: label noise breaks "accuracy on the line".
- **Weakness addressed:**
  - Truth is `speed ≤ 0.6·v_free` on the noise-free state, but our labels threshold noisy observed speed.
  - Cells near the threshold, and isolated one-cell blips, are systematically mislabeled.
  - This is a plausible reason why the CV→LB ratio is 0.84 for both our model and persistence.
- **Exact change:** try three variants on the current v9/v13 recipe.
  - (a) **Denoised hard labels.** Apply a small space×time smoother to `v/v_free` on the complete train layer before thresholding. Candidates are a 3×3 median, or the ASM kernel from A5. Pick the bandwidth with a Noise2Self criterion: the filter that best predicts held-out observed cells when the centre cell is excluded.
  - (b) **Soft labels.** Use the fraction of the 3×3 neighbourhood that is queued as the target, with LightGBM `objective=cross_entropy`.
  - (c) **Confident-learning down-weighting.** Down-weight to 0.3 any training cells whose OOF probability disagrees strongly with their noisy label and that are isolated (no queued neighbour within ±1 link and ±5 min).
- **Compute:** one Task 2 retrain per variant, the same cost as the current model.
- **Expected gain:** +0.002 to +0.01 LB. The effect is uncertain, because the local proxy truth is itself noisy.
- **Validation:**
  - Local scoring uses noisy labels, so a gain may be invisible or slightly negative locally. Score each variant on (i) the raw proxy and (ii) a denoised-truth proxy.
  - Then submit **one** LB probe for the best variant, swapping Task 2 only.
- **Reject rule:** reject if the LB probe is < +0.001, or if the raw-proxy IoU drops by more than 0.01. The latter would mean the smoother erased real short queues.

### A3. Task 1: add training rows from the target month itself via a masked-imputation task (MIT / test-time training). Label: NEW, high value.

- **Basis:**
  - D0153: the SAITS masked-imputation task. Re-mask observed values and learn to impute them.
  - D0154: test-time training under distribution shift.
  - D0155: self-supervision on noisy data.
  - D0166/D0167: DG tricks do not help, but target-domain data does.
- **Weakness addressed:**
  - The validation and private months are generated independently, with different incidents.
  - The Task 1 model has never seen rows from the month it predicts. The profile-free switch was the first sign that month-specific structure matters.
- **Exact change:**
  1. For each target split, draw R = 2 replicas.
  2. In each replica, hide an extra 8–10% of the released eligible cells. Use the same iid-hash-style mask, stratified by the cell's regime, and also add simulated 90-minute blackouts.
  3. Recompute features on the re-masked view. Targets are the released values in the same residual form (`v/v_f − tv`, `q/lanes − tq`).
  4. Either append these rows to the train rows with weight 0.5, or fit a small second LightGBM on target-only rows and blend 70/30.
  5. Keep the 15% evaluation holdout disjoint from every re-masked cell.
  6. Keep the extra mask rate low, so the feature missingness stays close to the real targets.
- **Compute:** +30–50% Task 1 rows. It fits inside the current kernel if MAX_ROWS is respected.
- **Expected gain:** +0.001 to +0.004 on the validation-month holdout S_state. It also lifts Task 3, which depends on the density implied by the Task 1 predictions.
- **Validation:** use the validation-month internal holdout (disjoint), per panel.
- **Reject rule:** reject if the gain over model E/F is < +0.001, or if fewer than 7/10 panels improve.
- **Leakage and rules:** Task 1 is offline, and only released observations are used as targets. Never use Task 2 blackout cells as targets.

### A4. Task 1: LightGBM knobs not yet tried. Label: NEW.

- **Basis:** D0171 (LightGBM parameter docs, inspected), D0176 (GBDT with piece-wise linear leaves), D0174 (ExtraTrees).
- **Exact change:** on model E settings, test one change at a time:
  - (a) `linear_tree=True, linear_lambda=1.0`. The residual target is locally near-linear in the neighbour values, and linear leaves extrapolate more smoothly to an unseen month. Test at 2× rows first, because linear_tree raises memory use.
  - (b) `extra_trees=True, path_smooth=10`.
  - (c) `deterministic=True, force_row_wise=True`. This is not a score change; it is a free guard for the 1% reproducibility rule.
- **Compute:** one Task 1 run per variant.
- **Expected gain:**
  - (a): −0.002 to +0.003.
  - (b): ±0.001.
  - (c): 0.
- **Validation:** validation-month holdout. **Reject rule:** reject if < +0.0007, the size of the smallest change we accepted before.

### A5. Task 1 features: physics-aware interpolation along characteristics (ASM and Newell three-detector). Label: NEW.

These are features for the GBDT, not post-processing, which we already tried and dropped.

- **Basis:**
  - D0137/D0138/D0139: the Treiber–Helbing adaptive smoothing method. Kernels run along the free-flow (+v_f) and congested (−w) characteristics. Calibrated parameters transfer across sites.
  - D0131: Newell-shifted cumulative-count features. Measured: RMSE −0.5 to −2%, with a larger help under transfer.
  - D0128: the three-detector method.
  - D0251: PDFormer, where delay-aware features help.
- **Weakness addressed:**
  - Our temporal and spatial interpolations are isotropic.
  - In the generator (CTM/LWR), information moves downstream at v_f and upstream at w = cap/(k_jam − k_crit). Both are known per link from `fd_parameters.csv`.
- **Exact change:** add the following for every target cell:
  - `asm_v`: an ASM speed estimate with `c_free = v_f`, `c_cong = −w`, σ = 1 station, τ = 1 slot, and switching from `v_crit` = critical speed.
  - `newell_q`: the flow from `min(N_up(t − d/v_f), N_down(t − d/w) + k_jam·d)`, using observed cumulative counts from the nearest observed upstream and downstream stations. Where no counts are available, fall back to the interpolated flow.
  - `asm_cong_weight`: the weight between the two regimes.
- **Compute:** vectorised numpy, roughly the cost of the current interpolation features.
- **Expected gain:** +0.001 to +0.004 holdout. Density at congested masked cells, which Task 3 scores, is the most likely beneficiary.
- **Validation:** validation-month holdout, overall and for R3 (50% masking). **Reject rule:** reject if < +0.0007, or if Task 3's local LWR proxy gets worse.

### A6. Task 2: weight weekend windows toward the target month's calendar share. Label: NEW, small.

- **Basis:** D0156 (importance weighting under covariate shift), D0158 (robust/clipped weights), D0082/D0083 (adversarial validation and reweighting).
- **Weakness addressed:** weekend share is 35% in the validation month and 24% in train, and the private month has its own calendar. Weekday-class priors already helped (+0.0033). Row weights are the cheaper complement.
- **Exact change:** weight each training window by `share_target(dow class) / share_train(dow class)`, clipped to [0.5, 2]. The target share comes from the target month's calendar, which is known without looking at data. Retrain the ongoing and onset models.
  - EXP_T2_l tested "equal window weights", which is a different weighting and gave ≈0.
- **Compute:** one Task 2 retrain. **Expected gain:** 0 to +0.002.
- **Validation:** weight the pseudo folds the same way, then run the LB probe. **Reject rule:** reject if < +0.001 in 3/4 folds.

### A7. Process: leaderboard discipline and final selection. Label: RELATED; this is methodology, not a model.

- **Basis:**
  - D0087: the Ladder. Accept only when the gain exceeds a noise step.
  - D0086: overfitting from test reuse is small, but distribution shift is not.
  - D0081: overtuning.
  - D0068: the forecast-combination puzzle. Estimated weights add variance.
  - D0093: rolling-origin validation can flip rankings.
  - D0347: shake-up survival.
- **Rules:**
  1. Accept a change only if the LB gain is ≥ 0.001 **and** the local folds agree in ≥ 3/4. Prefer fixed simple blend ratios, such as 70/30 or 50/50, over ratios tuned on 4 folds.
  2. Final pick of two submissions:
     - (i) the best public LB;
     - (ii) the most shift-robust recipe: prior-free wherever that tied, and no public-LB-tuned knobs.
  3. The private month has 7 incidents against 5 in validation. It therefore likely has more non-recurrent queues, which argues for (ii).

---

## B. Structural (new component, more compute or risk)

### B1. Task 2 onset: CTM forward rollout from the state at T as a feature. Label: RELATED.

Queue-block, shockwave and hazard features gave ≈0. A simulator rollout is a different signal.

- **Basis:**
  - D0124: the CTM (Daganzo).
  - D0210: VecCTM, a vectorised numpy CTM for batch rollouts.
  - D0303: the link transmission model, a cheaper Newell-based loading model.
  - D0131: Newell features help most under transfer.
  - D0255: congestion fronts are anchored at bottlenecks.
- **Weakness addressed:**
  - Onset IoU (0.743) has the most headroom.
  - Onset windows by definition show no queue in the history. A queue appears at T+30 when upstream demand exceeds a bottleneck's capacity.
  - A CTM that uses the per-link FD parameters and the demand observed at T can predict where supply falls short. This is the generator's own mechanism.
- **Exact change:**
  1. For each window, set the initial density from the last observed (interpolated) speed and flow via the per-link FD.
  2. Set upstream boundary inflow to the mean of the last 15 minutes, held constant.
  3. Run 6 × 5-min steps, with sub-steps to satisfy CFL.
  4. Features: `ctm_queued_T30` (speed ≤ 0.6·v_f), `ctm_density_ratio_T30`, `ctm_first_queued_step`, and `ctm_margin = min over t of (receiving capacity − sending demand)`.
  5. Add these to the CatBoost onset model and the LightGBM ongoing model.
- **Compute:** less than 1 CPU-minute for all windows (vectorised). The cost is mainly implementation, about one day.
- **Expected gain:** onset IoU +0.00 to +0.02. This is high variance; the LB effect is ≤ +0.006.
- **Validation:** 4 pseudo folds plus the 40 official onset windows.
- **Reject rule:** reject if onset is not ≥ +0.004 in ≥ 3/4 folds. Given our feature history, expect rejection unless the rollout carries capacity information the GBDT cannot reconstruct.

### B2. Task 2 decoder: use window-level queue-size information instead of the independence assumption. Label: NEW.

Earlier decoder experiments changed thresholds, steps or morphology. This one changes the model of |Y|.

- **Basis:**
  - D0145: GFM, an exact expected-F maximiser that uses P(y_i = 1, |y| = s) and needs no independence assumption.
  - D0146: independence-based plug-in decoding can be far from optimal when labels are correlated.
  - D0143: plug-in decoding is only as good as the probability model.
- **Weakness addressed:** queue cells are strongly correlated. Our expected-IoU top-m decoder treats them as independent, so it mis-estimates the IoU of each candidate m.
- **Exact change:**
  1. Train a window-level LightGBM quantile or Poisson regressor for |Y| (truth-set size) from window features. These include aggregates of the cell probabilities: their sum, the sum of the top-k, and spatial spread.
  2. At decode time, sample s from the predicted size distribution. For each s, rescale the cell probabilities so they sum to s, and compute the expected IoU for each m.
  3. Choose the m that maximises the average over samples.
- **Compute:** a small model trained on the OOF cache. The decode itself costs minutes.
- **Expected gain:** +0.002 to +0.006 IoU on both conditions.
- **Validation and reject rule:** OOF cache over 4 folds. Reject if the gain is < +0.002 in ≥ 3/4 folds.

### B3. Task 2 onset: a diverse neural member (RealMLP or TabM) blended with CatBoost. Label: RELATED.

CatBoost ongoing and seed averaging brought no diversity; a neural net gives more.

- **Basis:**
  - D0244: RealMLP plus tuned-default GBDT beats either alone.
  - D0245: TabM.
  - D0246: TabArena. Cross-model ensembles lead; watch for validation overfitting.
  - D0002: an XGBoost + deep ensemble was the best entry.
  - D0012 and D0344: GBDT+NN winners in KDD Cup 2022 and Ubiquant.
  - D0239: 2–3 models are enough.
- **Exact change:**
  - Model: a 3-layer MLP on the onset feature table. Use quantile-transformed inputs and a BCE loss. Train on CPU with torch, 8–15 epochs, with fixed seeds and `torch.use_deterministic_algorithms`.
  - Blend: rank-average with CatBoost at a fixed 70/30, then decode.
- **Compute:** 20–40 CPU-minutes for onset. Onset has fewer rows than ongoing.
- **Expected gain:** onset IoU 0 to +0.01, which is ≤ +0.003 LB.
- **Reject rule:** reject if the correlation with CatBoost is > 0.98, or if the gain is < +0.003 in 3/4 folds.

### B4. Task 4: KL (entropy/IPF) projection, or prior-weighted GLS, instead of the identity-weighted L2. Label: NEW, but low value.

- **Basis:**
  - D0098: Van Zuylen–Willumsen minimum information, a KL projection.
  - D0097/D0100: GLS and Bayesian estimators. With exact counts, the posterior mean equals a covariance-weighted projection.
  - D0112/D0114: IPF, or generalised IPF, for linear constraints.
  - D0111: constrained Poisson GLM.
  - D0109: L1 recovery for sparse path flows.
  - D0363: the scorer. S_dev rewards matching the *true* deviation magnitude from the prior.
- **Mechanics:**
  - With noise-free counts, every estimator satisfies A·f = c. They differ only in the null-space component of A, and that component is set entirely by the prior plus the regulariser (D0102).
  - KL or `diag(prior)`-weighted GLS keeps prior *ratios* and is better if the prior's error is multiplicative. L1 is better only if the true flows are sparse.
- **Exact change:** add `min Σ (f−b)²/b` with a hard or soft A·f = c as a variant beside the current NNLS.
- **Expected gain:** ≤ +0.0012 total at most; probably less than 0.0005.
- **Validation:** none is possible locally. S_od, S_dev and S_attr need withheld truth; the organizer's docstring says so explicitly. Only an LB probe can decide.
- **Reject rule:** do not spend a submission on this unless one is left over on the final days.

### B5. Task 1: tensor-completion (LATC or BGCP) output as an extra feature. Label: RELATED to the low-rank SVD features we kept.

- **Basis:** D0217 (LATC), D0216 (BGCP), D0249 (ImputeFormer: low-rank structure is the key inductive bias).
- **Exact change:** run LATC or truncated CP completion on the day × time × station tensor of each split. Add the reconstruction as `lr_tensor_v` and `lr_tensor_q`.
- **Expected gain:** +0.000 to +0.002. The rank-5 and cross-day SVD features already capture most of this.
- **Reject rule:** reject if < +0.0007.

---

## C. Negative evidence (why not to spend time on these)

- **Time-series foundation models (Chronos, TimesFM, Moirai, TabPFN-TS, LLM forecasters) for Task 2:**
  - GIFT-Eval (D0031): TSFMs are weakest on Transport-type data and with short contexts.
  - Tuned AR or supervised baselines match or beat specialised foundation models (D0035), and synthetic-pretrained zero-shot models are more than 2× worse than supervised SOTA (D0036).
  - LLM components add nothing (D0254).
  - The TSFM input patch (32 steps in TimesFM, D0029) is longer than our whole 12-step history.
  - The target is a per-cell binary IoU set, not a numeric forecast.
  - Verdict: CONTRADICTED.
- **Long-horizon transformers and decomposition/MLP forecasters (Informer, Autoformer, FEDformer, PatchTST, TimesNet, iTransformer, TimeMixer, N-HiTS, N-BEATS):**
  - They are designed for 96–720-step lookbacks (D0004, D0021, D0022).
  - DLinear already beats the older transformers on those benchmarks.
  - A windowed GBRT beats 8 deep models on 9 datasets, including traffic (D0238).
  - Verdict: CONTRADICTED as replacements. At most they are diverse members (B3).
- **Deep tabular models as replacements for the GBDTs:**
  - Trees win on irregular, medium-to-large tabular data (D0001, D0002, D0003, D0243).
  - Neural nets help only as ensemble members (D0244, D0246).
- **Spatio-temporal GNNs (DCRNN, Graph WaveNet, GRIN, SPIN, STAEformer):**
  - Plain MLPs with identity embeddings match STGNNs (D0047, D0048).
  - Our corridor is a 1D line graph, and ±k neighbour features already encode it.
  - Identity embeddings are location priors, which hurt under month shift, as our prior-free ongoing result shows.
  - GPU cost and nondeterminism conflict with the 1% reproducibility rule.
- **Domain generalisation and invariance training:**
  - Tuned ERM ties or beats every DG method (D0167).
  - On temporal shift benchmarks, DG, SSL and continual learning show no gain over ERM (D0166).
  - Verdict: CONTRADICTED. Target-domain *data* (A3) and label-prior correction (A1) are the evidence-backed levers, not invariance.
- **Heavy stacking and large ensembles:**
  - Estimated combination weights add variance (D0068).
  - Hillclimb and ensemble selection overfit the selection set (D0062, D0246).
  - 2–3 models capture nearly all the gain (D0239).
  - Our own data agree: per-panel, per-TOD and weekday blends ≈0, 3 seeds ≈0, and CatBoost ongoing had correlation 0.997.
  - If we stack at all, use a linear or logistic meta-model on OOF probabilities (D0259), not a GBDT meta-model.
- **Per-regime mixture of experts with hard splits:**
  - The soft gating of D0067 is the principled version.
  - Our hard R1/R2/R3 Task 1 split lost 0.002 on every panel, and TSE's missing-rate model (D0181) was a special case in a different setting.
  - Do not retry without soft gating.
- **Physics-informed losses and post-hoc physics reconciliation for Task 1/Task 3:**
  - D0134 documents PIML failures when the physics is misspecified or the loss terms are imbalanced.
  - This matches our dropped density, FD-pull and L1-reconciliation experiments.
  - Physics should enter as **features** (A5, B1), not constraints.
- **HPO:**
  - Tuned-over-default gains for GBDT are modest (D0077), and only a few hyperparameters matter per dataset (D0079).
  - Overtuning is common with small or noisy validation sets (D0081).
  - Our capacity sweeps paid off (+0.003..0.008), but further blind Optuna sweeps on the same 4 folds are likely to overfit. Use nested folds if you sweep at all (D0258).
- **ODME literature beyond projection:**
  - Bilevel, SUE path-flow estimators, dynamic and Kalman methods, and deep ODME (D0101, D0105, D0106, D0120) all either:
    - need extra information (routes, costs, dynamics), or
    - protect against count noise, which is absent here.
  - With noise-free counts none of them adds information. Task 4 is saturated (B4).

## D. Priority order (value ÷ cost)

1. **A3** — Task 1 target-month MIT rows. NEW; affects Task 1 and Task 3 (50% of the score).
2. **A2** — Task 2 label denoising. NEW; targets the systematic CV→LB gap.
3. **A1** — calibration plus causal prior-shift. Post-hoc; one probe.
4. **A5** — ASM and Newell features for Task 1.
5. **B2** — size-aware decoder.
6. **B1** — CTM rollout onset feature.
7. **A4**, **A6**, then **B3**.
8. **B4** and **B5**: only with spare submissions or compute.
