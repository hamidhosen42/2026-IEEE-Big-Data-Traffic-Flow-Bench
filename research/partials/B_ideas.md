# Source family B: experiment ideas for Task 2 (queue forecast)

Scope covered: breakdown and onset, stochastic capacity, bottlenecks, precursors, queue propagation, survival and hazard models, change-points and regimes, expected-IoU decoding, calibration, imbalance, PU learning and label noise, shift and domain adaptation, and validation design.

- Resource ids (Bxxxx) refer to `research/partials/B_resources.csv`.
- Labels are relative to `research/EXISTING_MAP.md`: NEW / RELATED / DUPLICATE / CONTRADICTED / HIGH-VALUE RETEST.

## 0. Evidence I measured on the local OOF cache

These are scratch runs only. They are not in `TRACKER.md`.

**Setup.**
- Cache: `kaggle_jobs/out/t2oof`, 4,269 windows, the same folds as EXP_T2_q.
- Recipe: onset = CatBoost `Pc`, ongoing = prior-free LightGBM `Pn`, with the current ratio decoder `decode()`.
- Aggregation is the competition one. Pseudo scores use 4 day-block folds; official scores use 40 + 40 windows.

| Variant (post-hoc, no retraining) | pseudo total | pseudo onset | pseudo ongoing | official total | official onset | official ongoing |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: ratio decoder E[I]/E[U] (current) | 0.8770 | 0.8930 | 0.9082 | 0.8876 | 0.9393 | 0.8359 |
| Exact Poisson-binomial expected-IoU decoder (RankIoU, independence) | 0.8769 | 0.8918 | **0.9088** (+ in 4/4 folds) | 0.8798 | 0.9237 | 0.8359 |
| Onset component-mixture Monte Carlo decoder (activation = max p, thr 0.02) | **0.8780** | **0.8945** (fold Δ +.0001/.0000/+.0053/+.0010) | 0.9082 | 0.8876 | 0.9393 | 0.8359 |
| Onset temperature T=1.5 | 0.8765 | 0.8931 | – | 0.8876 | 0.9394 | – |
| Onset temperature T=2 / T=3 | 0.8677 / 0.7062 | 0.8776 / **0.5298** | – | 0.8741 / 0.6889 | 0.9123 / 0.5419 | – |
| Onset logit bias ±1 | 0.8738 / 0.8722 | 0.8863 / 0.8856 | – | 0.8808 / 0.8860 | – | – |
| Tail truncation (p < ε → 0), ε = 0.005…0.1 | 0.8770–0.8772 | ±0.0003 | +0.0000…+0.0007 | 0.8876–0.8877 | = | = |

**Other facts from the same cache.**
- **Calibration.** In-distribution calibration is already good.
  - Ongoing reliability is almost on the diagonal.
  - Onset is slightly under-confident in the middle (mean p 0.24 → 0.33 true, 0.65 → 0.73).
  - Onset Σp per window is 3.04 against 3.57 true cells. The decoder picks 3.12 cells.
- **Onset confidence is bimodal.**
  - 91% of pseudo onset windows have max p > 0.9, and their IoU is 0.937.
  - The 3% of windows with max p < 0.5 have IoU 0.10.
  - All 39/40 official onset windows have max p > 0.9.
- **Weekends are easier, not harder.**
  - Onset IoU: weekend 0.926 vs weekday 0.881 (pseudo); 0.969 vs 0.920 (official).
  - Ongoing IoU: 0.910 vs 0.908 (pseudo); 0.898 vs 0.802 (official).
  - The validation month has more weekend windows, so on its own that should raise its LB. The weekend share therefore does not explain the CV→LB gap.
  - It follows that re-weighting CV toward weekends would widen the gap, not predict it.
- **Conditioning on "the truth is non-empty" does not change the decision.**
  - Every onset window contains a queue by construction.
  - Under independence, E[IoU | Y≠∅] = E[IoU]/P(Y≠∅) for any non-empty prediction. The factor is constant in the prediction, so the arg-max is unchanged.
  - Renormalising p by 1−Π(1−p) is therefore pointless for decoding.
- **Code check.** The `rankseg` RMA IoU rule, `cumsum_p / (Σp − cumsum_p + τ)`, is our `decode()` line for line (B0007). Our decoder is the published RankSEG-RMA / EIoEU approximation (B0005, B0037).

---

## CHEAP GAINS (no retraining; post-hoc on the cached OOF probabilities)

### C1. Onset: dependence-aware (component-mixture) expected-IoU decoding — NEW
- **Basis:**
  - B0037 (Ahmed et al.: EIoEU over mixtures centred on candidate solutions beats both plain EIoEU and exact EIoU).
  - B0033 and B0297 (Waegeman et al. / GFM: marginal decoding is inconsistent when labels are dependent).
  - B0310, B0005.
- **Weakness addressed.**
  - The current decoder assumes independent cells.
  - In reality one recurrent bottleneck activates as a contiguous block at T+30, so the cells are strongly positively dependent.
  - Onset windows are tiny (median 3 cells, B0003), which is where the E[I]/E[U] approximation error is O(1/n) (B0005).
- **Exact change.**
  - At h = T+30, form components: runs of adjacent links with p ≥ 0.02.
  - Give each component an activation a_c = max p and cell probabilities q_i = p_i / a_c.
  - Draw 400 Monte Carlo joint samples. Drop empty samples, since onset truth is never empty.
  - Choose τ for the top-τ-by-p set by maximising the mean sampled IoU.
  - Ongoing windows keep the current decoder.
  - Next variant: replace a_c with a learned window-level activation probability (see R3).
- **Availability ≤ T:** yes. It is a pure function of this window's probabilities.
- **Leakage:** none.
- **Compute:** about 12 s for all 4,269 windows. Use a fixed RNG seed so results are reproducible.
- **Measured:** onset +0.0015 pseudo (≥0 in 4/4 folds); official unchanged. LB-equivalent is only about +0.0002.
- **Validation:** nested-fold choice of the threshold (0.02 / 0.05) and the activation rule. Report per-fold Δ.
- **Reject rule:**
  - Mean pseudo onset Δ < +0.002 with the learned activation, **or** any fold < −0.002, **or** official onset < baseline.
  - Otherwise bundle it into the next submission. Do not spend a probe on it alone.

### C2. Ongoing: exact Poisson-binomial expected-IoU decoder (RankIoU) — NEW, tiny
- **Basis:** B0006, B0007, B0035, B0034, B0317.
- **Exact change.** For ongoing windows only, replace the ratio with the exact value.
  - Formula: E[IoU](τ) = (Σ_{top τ} p) · ∫₀¹ x^{τ−1} Π_{j∉top τ}(1−p_j+p_j x) dx.
  - Evaluate it with 160-node Gauss–Legendre quadrature, which is exact under independence.
  - **Do not cap τ.** A cap of 400 silently broke large queues in my first run.
- **Measured:**
  - Ongoing: +0.0006 pseudo (4/4 folds), official ±0. LB-equivalent is about +0.0001.
  - Onset: −0.001 pseudo and −0.016 official, so keep the ratio decoder for onset.
- **Leakage / compute:** none; about 20 s.
- **Reject rule:** ship only bundled, and only if a re-run on the v9/255-leaf OOF (`oofX`/`oofY`) still shows ≥ 0 in 4/4 folds.

### C3. Shift diagnostic on val/private windows using our own predictions — NEW, HIGH-VALUE for decisions, not for score
- **Basis:**
  - B0026 (Ben-David: shift is measurable from unlabeled inputs).
  - B0017 (Ovadia: calibration degrades under shift).
  - B0043 (BBSE uses predictions on the target domain), B0023.
- **Question.** Is the LB onset gap (0.743 vs CV 0.89) a *mixture* shift?
  - In CV, IoU is 0.937 when max p > 0.9 and 0.10 when max p < 0.5.
  - So if the validation month simply contains more "no-confident-candidate" windows (non-recurrent or incident onsets), the gap is largely irreducible from data ≤ T. Effort should then move to ongoing.
- **Exact change (analysis only).**
  - For the 40 validation + 40 private onset windows (and the ongoing ones), histogram max p, Σp, the number of components, and the tail mass Σ_{p<0.05} p. Use the submitted models.
  - Predicted LB onset IoU = Σ_bins share_val(bin) × IoU_CV(bin), with bins from the table in §0.
  - Compare this with the measured 0.743.
- **Leakage:**
  - The diagnostic reads validation *inputs* (≤ T per window) and our outputs only.
  - Its result must NOT be turned into per-window rules fitted on validation data. It only steers where effort goes.
- **Compute:** minutes.
- **Decision rule.**
  - If the mixture-predicted onset IoU is within ±0.03 of 0.743: stop onset modelling work and spend the probes on ongoing.
  - If it predicts ≫ 0.743: the model is confidently wrong on the validation month (concept shift). Prioritise R1 and V1.

### C4. Tail-mass safeguard — NEW, robustness only
- **Basis:** B0005 (the decision depends on the full Σp), plus my temperature run.
  - The ratio decoder is hypersensitive to a few thousand near-zero cells.
  - At onset T=3, IoU collapses from 0.893 to 0.530.
- **Change:** set p < 0.02 to 0 before decoding.
- **Measured:** CV-neutral (±0.0002).
- **When to use:** apply only if C3 shows inflated tail mass on validation windows compared with train OOF. It then protects against a covariate-shift-inflated background.
- **Reject rule:** do not submit without a C3 signal. It cannot be validated locally.

---

## Ideas that need retraining

### R1. Denoised / soft training labels to match the noise-free truth — NEW (HIGH-VALUE)
- **Basis:**
  - B0004: organizer says labels come from the underlying state, with no noise floor.
  - B0041 (Natarajan: class-conditional noise), B0042 (confident learning).
  - Frenay & Kaban survey; soft-label papers in the `label_noise` family.
- **Weakness addressed.**
  - Training labels are `observed speed ≤ 0.6·v_free` on *noisy* observations.
  - Near the threshold and at queue edges this creates flicker. The model learns to reproduce noise that the LB truth does not contain.
  - The CV→LB ratio is method-agnostic: 0.82 for persistence, 0.84 for our model. That fits a truth-definition mismatch better than a model-specific shift.
  - It also fits the weekend finding in §0: weekends are easier, yet the LB is lower.
- **Exact change.** Pick one of three variants:
  - (a) Build labels from a 3-step centred median of the speed (t−5, t, t+5) before thresholding. This is allowed because labels on train months may use their own future.
  - (b) Soft labels q = Φ((cut − v_obs)/σ), with σ estimated from lag-1 speed differences in free flow. Train LightGBM `xentropy` on q; CatBoost onset uses `CrossEntropy` with probabilistic targets.
  - (c) Confident-learning style: drop or flip isolated single-cell queue labels whose OOF p < 0.1.
  - Features stay unchanged. Windows and folds stay unchanged.
- **Leakage:** none. Label construction uses train-month data only; features stay ≤ T.
- **Compute:** one full T2 retrain on Kaggle (about the same as v12).
- **Expected gain:** uncertain, +0.000…+0.005 LB. It could lift ongoing, which has the largest absolute gap (0.78 vs 0.92).
- **Validation.**
  - Score CV against both the noisy proxy and the median-3 proxy (V1). A drop on the noisy proxy alone is expected and does not mean the change failed.
  - A single LB probe decides.
- **Pre-check (cheap):** the share of train queue cells that are isolated in both space and time, and the estimated σ. If isolated cells are under 1%, skip R1.
- **Reject rule:** LB Δ < +0.001 (5 subs/day allows it), **or** onset/ongoing CV on the median-3 proxy drops by more than 0.003.

### R2. Breakdown-probability (stochastic-capacity) features for onset — RELATED (likely small)
- **Basis:**
  - B0009, B0010: Brilon/Geistefeldt censored product-limit estimate, where capacity follows a Weibull distribution.
  - B0107, B0106: Elefteriadou; Persaud et al.
  - B0112, B0104 (Markov-chain breakdown), B0425.
- **Weakness addressed.** Onset = breakdown at a recurrent bottleneck. The current features give flow/capacity, but not a per-link empirical breakdown curve.
- **Exact change.**
  - Per link, from the 9 train months, fit a censored capacity distribution F_link(q):
    - an event = the 5-min flow just before speed first drops below v_cut;
    - censored = non-breakdown flows.
  - New features: F_link(q_T), F_link(max q over [T−15, T)), and 1 − Π over the next-upstream links.
  - Add these to CatBoost onset only.
- **Caveat.** The generator is CTM/LWR-like. If link capacity is deterministic in the simulator, breakdown is a demand-exceeds-capacity event and F_link collapses to a step. That is already represented by `cap_ratio` and demand features, and the "upstream demand" features in EXP_T2_H were noise.
- **Leakage:** fit per training fold only (fold-safe, like the priors).
- **Compute:** low.
- **Reject rule:** onset pseudo Δ < +0.003 mean or < 3/4 folds positive (the EXP_T2_H standard).

### R3. Window-level activation / volume model to feed C1 — NEW
- **Basis:**
  - B0033 (GFM needs the joint count distribution: P(y=0) and the size structure).
  - B0006 (volume τ is the second half of the Bayes rule).
  - B0037.
- **Exact change.**
  - For each pseudo onset window, enumerate candidate components (C1).
  - Train a small classifier P(component contains a T+30 queue cell) on component-level features: max p, Σp, component length, queue prior at the component, weekday class, recency of the last activation the same day.
  - A second, optional head: P(|Y| = k), k = 1..8.
  - Plug both into C1's Monte Carlo decoder instead of max p.
- **Leakage:** ≤ T features; nested folds for the component model.
- **Compute:** low (thousands of rows).
- **Expected gain:** +0.002–0.006 onset CV (speculative), roughly +0.0003–0.001 LB.
- **Reject rule:** onset pseudo Δ < +0.003 or official onset down.

### R4. Causal (≤ T) prior-shift correction — NEW but low / risky
- **Basis:** B0044 (Saerens EM), B0045 (BCTS+EM), B0043.
- **Change.**
  - Per panel, keep a running estimate of the queue-cell prevalence using only released data with timestamp ≤ T of the current window. That means earlier windows the same day or month, and history observations only.
  - Adjust logits by log(π_new/π_train).
- **Leakage: HIGH for the transductive version.** Pooling all validation windows uses data after T for earlier windows, which the Task 2 rule forbids (discussion 742068). Only the causal version is allowed.
- **Why low:** a logit bias of ±1 already costs −0.007 onset in CV, and there is no trustworthy prevalence signal ≤ T.
- **Reject rule:** skip unless C3 shows a large prevalence shift in history-visible quantities.

### R5. Change-point / BOCPD / CUSUM history features — DUPLICATE-ish (low)
- **Basis:** B0019, B0202, B0213.
- **Status:** RELATED to EXP_T2_H, where speed deltas, std, and acceleration were noise.
- Only retry as a single feature, "run length since the last speed regime change at the link and its downstream neighbour", if R1 changes the label regime. Otherwise skip.

---

## Validation-design ideas

### V1. LB-faithful local truth — NEW (HIGH-VALUE for decision quality)
- **Basis:** B0049 (Roberts: evaluation must mirror the target structure), B0050 (Wild-Time), B0048, B0004.
- **Weakness addressed.**
  - Local IoU uses noisy observed speed on sensored, eligible cells.
  - The LB uses the noise-free state on **every link**.
  - The ratio of about 0.82–0.84 for both persistence and our model indicates that the metric definition, not only the month, differs.
- **Exact change.**
  - Build a second local truth on train windows from a denoised state: centred median of 3 in time, plus a spatial 3-link median. Score it on all links, with unsensored or ineligible cells filled from the Task 1 v4 interpolation.
  - Re-score persistence, the base heuristic, the v2 model, and the current recipe.
  - Check whether this reproduces LB ratios: persistence 0.2518 LB vs local, and our 0.745 LB S_queue.
  - If it does, use it as the primary Task 2 CV metric. The existing LB probe history gives about 8 LB-validated deltas to rank-check against.
- **Leakage:** none (train months only).
- **Compute:** low.
- **Reject rule:** if its rank agreement with the known LB deltas (og prior-free +, onset 30/70 −, weekday priors +, persistence blend −, CatBoost onset +) is not better than the current proxy, drop it.

### V2. Shift-weighted CV (IWCV) — CONTRADICTED for the weekend factor
- **Basis:** B0021, B0020.
- Weekend windows are *easier* locally (§0). Weighting toward the validation weekend share raises CV and moves it away from the LB.
- Only re-open this with a non-recurrent / incident weight, and only if C3 or V1 identifies a measurable window type.
- EXP_T2_l already showed that a local non-recurrent subset does not reproduce the LB.

---

## NEGATIVE evidence (what not to do, with sources)

**From my measurements on the OOF cache:**

1. **Post-hoc calibration / temperature on onset.**
   - T = 2 → −0.015 and T = 3 → −0.36 onset IoU; logit bias ±1 → −0.007.
   - The expected-IoU decoder is hypersensitive to background mass.
   - Isotonic/Platt/beta (B0011–B0013) cannot help much because in-distribution reliability is already good.
   - Under shift, CV-fitted calibration does not transfer (B0017, B0016).
   - **CONTRADICTED** as a cheap gain.
2. **Exact independent-cell decoding for onset.** −0.001 pseudo and −0.016 official. The independence assumption is worse than the ratio approximation on tiny dependent onset blobs (B0033 explains why).
3. **Renormalising onset probabilities by P(≥1 queued cell).** It has no effect on the Bayes decision (derivation in §0).

**From the literature and the competition forum:**

4. **Gating onset on "quiet history".** The public 0.809 notebook reported that speed-based onset suppression collapsed onset IoU from 0.74 to 0.17 (B0001). Breakdown is stochastic at a given flow (B0107, B0106, B0009): the history can look identical on breakdown and non-breakdown days.
5. **Fixed thresholds or threshold tuning.**
   - They are provably not IoU-consistent (B0006, Remark 2; B0036).
   - This matches EXP_T2_D.
   - Structured/SSVM-style loss minimisation is inconsistent, whereas the plug-in approach is consistent (B0030). A training-time Lovász or soft-Jaccard loss (B0040, B0329) is therefore low value for our per-cell GBDT plus decoder.
6. **Class rebalancing / SMOTE / focal loss.**
   - Balancing does not help strong GBDTs (B0029).
   - Focal and cross-entropy are not IoU-calibrated (B0006).
   - Focal distorts probabilities (B0027), which our decoder depends on.
7. **Importance weighting / domain adaptation.**
   - The effect of importance weights vanishes for high-capacity models (B0022).
   - Domain generalisation and domain adaptation rarely beat tuned ERM (B0024, B0025, B0050).
   - Invariant representations can hurt when label marginals differ (B0254).
   - This is consistent with EXP_T2_l (window weights ≈ 0).
8. **Hazard / survival reformulation of onset timing** (B0046). It is **CONTRADICTED**: the selector fixes the first appearance at T+30 (B0002, B0003). A hazard model trained on all origins would also lose the "a queue appears somewhere at T+30" conditioning that the replicated selector provides.
9. **Transductive label-shift or covariate-shift correction on the validation month** (B0044, B0045, B0016). It pools inputs after T, which the Task 2 rule forbids (discussion 742068).
10. **Weekend-driven CV re-weighting.** Weekends are easier locally, so this is the wrong direction (§0).
11. **Adaptive smoothing (ASM) of history fields** (B0001). It blurred bottleneck transitions in Task 1. Avoid it for Task 2 history features too.

## Priority order (my recommendation)
1. C3 (diagnostic, minutes).
2. V1 (LB-faithful truth, low cost).
3. R1 (denoised labels; decide with a single LB probe).
4. C1 + R3 (onset dependence-aware decoding).
5. C2, bundled.
6. Everything else is low value or rejected.
