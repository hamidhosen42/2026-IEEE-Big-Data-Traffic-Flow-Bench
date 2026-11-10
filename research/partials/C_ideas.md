# C — Traffic state reconstruction / imputation / physics: experiment ideas for our pipeline

Scope: Task 1 (masked speed and per-lane flow), and Task 3 through the density q/v at masked cells.
Evidence ids point to `C_resources.csv`. Each idea is labelled against `EXISTING_MAP.md` (2026-09-30).

Baseline to beat: T1 LightGBM residual-to-interpolation, F model (511 leaves), val-month holdout **0.9386**.
It is the E model (0.9374) plus bigger capacity.

## Validation protocol (applies to every idea)

1. **Val-month internal holdout.** Hide 15% of released eligible cells.
   - Report per panel, per mask regime (20/30/50%), and separately for cells inside the 90-min post-origin blackouts.
   - Compute every new feature **after** the holdout cells are hidden.
2. **Train CV.** Use the 3 contiguous day-block folds on the official train templates, with the same aggregation as the competition.
3. **LWR proxy.** Compute `1 − Σ|ΔN_pred − ΔN_obs| / Σ|ΔN_obs|` on target cells, and report T1+T3 jointly.
   - Physics is about 0.67 on LB, so a density gain is worth more than a speed-RMSE gain of the same size.
4. **Generic reject rule.** Reject if any of these holds:
   - mean holdout Δ < +0.0010;
   - the change is worse on ≥ 3 of 8 panels;
   - the LWR proxy drops by more than 0.002.
   - Individual ideas below tighten this where noted.
5. **Leakage rule.** A feature may only use cells that are *visible in that row's view*.
   - Training rows: official train mask + simulated blackout.
   - Holdout rows: test mask + the extra 15%.
   - Never use the target cell itself.

---

## A. Cheap gains (≤ 1 day of work each, CPU cost ≤ 30 min per full run)

### A1. ASM/GASM characteristic-kernel features — **NEW, HIGH-VALUE** (priority 1)

**Basis.**
- C0013: Treiber–Helbing ASM, robust when up to 65% of detectors are dropped.
- C0002 / C0001: calibrated ASM with FFT code, CODE_INSPECTED. Calibrated values were c_cong ≈ −20 km/h, c_free ≈ 80–97 km/h, τ ≈ 8–12 s, δ ≈ 0.14 km at 30 s / 0.48 km resolution.
- C0014: GASM, which uses multiple wave speeds with weights from matrix completion.
- C0024: ASNN, which learns the ASM parameters.
- C0006: KW-CNN (DEEP_READ). Anisotropic kernels bounded by the free-flow speed (downstream) and the backward wave speed (upstream) reduce overfitting.
- C0697: fast ASM implementations.
- C0020, C0027: phase-based and oblique-grid smoothing variants.

**Weakness addressed.** Our neighbour features are isotropic and short-range: ±1..2 stations at t−1..t+1.
- In congestion, information travels upstream at about 15–25 km/h, which is about 1.3–2 km per 5-min slot.
- The informative cells for a masked congested cell therefore lie on a diagonal spanning several links and slots.
- At 50% masking, and inside the 90-min blackouts, the ±1 neighbours are often missing. Temporal interpolation then carries pre-blackout values into a queue that has moved.

**Code change (feature stage, `t1_features`).** Work per panel-day on the (slot × link) grid, using milepost x.
1. Build a mask M: 1 where the cell is visible in the row's view. Build the values V: speed, total flow, and occupancy.
2. For each target (i, t), sum over links j with |j−i| ≤ 6 and slots s with |s−t| ≤ 6. Let Δx = x_j − x_i in km and Δt = 5(s − t) in minutes.
   - `φ_c = exp(−|Δt − Δx/c_cong|/τ − |Δx|/δ)`
   - `φ_f = exp(−|Δt − Δx/c_free|/τ − |Δx|/δ)`
   - `Z_c = Σφ_c·M·V / Σφ_c·M` and `Z_f = Σφ_f·M·V / Σφ_f·M`
   - Also keep the support masses `S_c = Σφ_c·M` and `S_f = Σφ_f·M`.
3. Blend: `W = ½(1 + tanh((V_thr − min(Z_c^speed, Z_f^speed))/ΔV))` and `ASM = W·Z_c + (1−W)·Z_f`.
4. Features per variable (speed, flow, occupancy): Z_c, Z_f, W, ASM, S_c, S_f, and ASM − interp.
   - Add ASM density `ASM_flow / ASM_speed` for Task 3.
5. GASM variant: add Z at c ∈ {−25, −18, −12, +60, +90, +120} km/h as separate columns and let LightGBM do the weighting (C0014's idea).
   - Implement with 169 array shifts using the real per-pair Δx (links are non-uniform), in vectorised numpy.

**Calibration.** For each corridor, grid-search on two train weeks per panel, scoring RMSE of ASM alone on hidden released cells:
- c_cong ∈ {−12, −16, −20, −25};
- c_free ∈ {60, 80, 100};
- τ ∈ {2.5, 5, 10} min;
- δ ∈ {0.3, 0.6, 1.2} km.

Because τ and δ must match 5-min / link resolution, **do not copy the 30-s values from C0002**. Use calibrated values per corridor, or pooled values.

**Compute.** A 169-shift sum on a 288 × 350 grid × 3 variables × 3–9 kernels takes about 1–3 s per panel-day, so a few minutes per full run on 4 cores. Calibration is under 10 min.

**Expected gain.** +0.002 to +0.006 on the holdout, concentrated in the 50% regime, the blackout cells, and congested cells. There should be a visible LWR-proxy gain because congested speeds drive q/v errors. This is our estimate, not a literature number: no paper reports ASM on top of a GBDT.

**Validation.** Use the protocol above plus a per-regime and per-blackout breakdown. Extra check: feature importance must not be concentrated in S_c/S_f alone, which would mean the model is learning mask density rather than the state.

**Reject.** Holdout Δ < +0.0015, or no gain on the 50%-regime subset.

### A2. Mask-support and mask-rate matching of training views — **RELATED** (regime rate and gap distances exist)

**Basis.**
- C0008, ImputeFormer Table 7 (DEEP_READ): models do best when the training mask rate is close to the test missing rate. Mixed-rate masking is best when rates are unknown or blocky.
- C0007: the survey found that models fail at high missing rates unless missingness is indicated explicitly.

**Code change.**
1. Add the local observed fraction in windows 3×3, 5×5 and 7×7 (links × slots).
2. Add the number of visible cells on each characteristic cone. This comes for free from A1's S_c/S_f.
3. Check that the neighbour-availability distribution of training rows matches the test regimes.
   - In training, the official template is the only mask; in the holdout, test mask + 15% also applies.
   - If they differ, add extra random hiding to training views at the regime-specific rate, so the holdout and the test are not systematically sparser than training.

**Compute.** Negligible.

**Expected gain.** +0.000 to +0.0015.

**Reject.** Δ < +0.0005.

### A3. Masked low-rank completion (SoftImpute / LATC / LCR) instead of SVD of the interpolated field — **RELATED / HIGH-VALUE RETEST**

**Basis.**
- C0004: fancyimpute SoftImpute (CODE_INSPECTED).
- C0003 / C0005: transdim LATC (CODE_INSPECTED). Seattle speed at 30% random missing gave RMSE 3.16 vs 3.72 for BTMF.
- C0262 / C0261: LCR, an FFT solution in log-linear time.
- C0007: the benchmark found that low-rank or global priors are among the strongest, and LATC is the cheapest in time and memory.
- C0195: RPCA. C0205: TRMF.

**Weakness addressed.** Our rank-3/5 SVD features factorise the *interpolated* field. Inside gaps and blackouts they reproduce the interpolation bias. A proper masked completion fits only visible entries.

**Code change.**
1. Per panel-month and variable, build the matrix (days·288) × links, or the tensor links × 288 × days, from visible cells only.
2. Run SoftImpute with max_rank ∈ {5, 10}, shrinkage chosen on hidden released cells, 50–100 iterations, using randomized SVD.
3. Optional: LATC with time_lags {1, 2, 288} and θ ∈ {5, 10}.
4. Features: the completed value, completed − interp, and the completed density.
5. Keep the existing SVD features (the tree picks).

**Compute.** SoftImpute on an 8640 × 350 matrix takes seconds per iteration, so a few minutes per panel.
- LATC's per-sensor sparse solves take roughly 10–40 min per large panel (our estimate, not measured). Try SoftImpute first.

**Expected gain.** +0.0005 to +0.002.
- The previous low-rank steps gave +0.0036 (LB +0.0015) and then +0.0007.
- Limiting evidence: C0501 found completion adds little when local data are rich. C0008's Table 2 has LRTC-AR (LATC) failing in block missing (MAE 4.05 vs 0.94 for point missing).
- So use completions **as features only**, never as replacements.

**Reject.** Δ < +0.0007, which is the size of the last low-rank step.

### A4. Conservation / CTM one-step features — **NEW as features** (post-processing reconciliations failed, features were never tried)

**Basis.**
- C0015: Daganzo CTM.
- C0062 / C0063: Newell's simplified kinematic-wave theory (N-curves).
- C0071 / C0072: variational formulation.
- C0275: CTM density estimation, about 13% error.
- Map: "conservation (= spatial flow smoothing) is the only base-recipe step that helps".

**Weakness addressed.** Flow at masked cells, and the q/v density that Task 3 scores, are only implicitly tied to the neighbours' flows.

**Code change.** Build these features from the visible field, or from the A1 ASM-filled field when a neighbour is missing:
1. Upstream balance: `q_i(t) ≈ q_{i−1}(t) − L_i·(k_i(t+1) − k_i(t))/Δt`.
2. Downstream balance: `q_i(t) ≈ q_{i+1}(t) + L_{i+1}·(k_{i+1}(t+1) − k_{i+1}(t))/Δt`, with k = q/max(v, 1).
3. Godunov/CTM one-step forecast of k_i(t) from t−1, and a backward analogue from t+1. These need a per-link triangular FD (from A5): demand/supply `min(v_f·k, q_c)` and `min(q_c, w·(k_j − k))`. Report the implied speed `q/k`.
4. Newell N-curve feature: cumulative count shifted along the free characteristic (from upstream) and the congested characteristic (from downstream). Take the minimum, per Newell.

**Compute.** Seconds.

**Expected gain.** +0.001 to +0.003 on flow and LWR. Risk: differencing amplifies noise; the tree can ignore it.

**Reject.** Δ(T1+T3 proxy) ≤ 0.

### A5. Per-link FD calibration and FD-implied features, plus the occupancy→speed identity — **NEW (features)**; start with a diagnostic

**Basis.**
- C0122: automatic FD calibration on PeMS.
- C0109: g-factor single-loop speed estimation, 23% RMSE reduction.
- C0640, C0099: g-factor and length-based speed estimation from single loops, including the PeMS g-factor algorithm.
- C0103 / C0119 / C0114: capacity drop of about 6%, increasing with density.

**Step 0, diagnostic (30 min).** Measure the co-masking rate of speed vs occupancy vs flow at Task 1 target cells.
- If occupancy or flow is released where speed is masked, the g-factor identity `v ≈ q·g/occ` is a very strong feature. Fit g per link and time-of-day on train.
- If they are co-masked, which is likely given the hash mask, go to step 1.

**Step 1.** Fit a triangular FD per link on train months:
- v_f from the p85 of free-flow speed;
- q_c from the p98 of 5-min flow;
- w and k_j by fitting the congested branch to (k, q) points with v < 0.6·v_f.

Features:
- q/q_c and k/k_c from the interpolated or ASM values;
- the FD-implied speed on each branch given the interpolated flow;
- queue-discharge flow `(1 − 0.06)·q_c` of the nearest downstream active bottleneck, from the capacity-drop evidence;
- distance to that bottleneck head.

The calibrated w also provides c_cong for A1: w = q_c / (k_j − k_c).

**Compute.** Minutes.

**Expected gain.** +0.0005 to +0.002. Step 0 could be much larger if not co-masked.

**Reject.** Δ < +0.0005.

### A6. Denoised neighbour inputs — **NEW but low value**

**Basis.** C0007 (priors help); signal-processing rows (family `signal_processing`).

**Code change.** Savitzky–Golay or median filters over visible cells only, for the self station and the ±1 stations. Mostly subsumed by A1's kernel averaging, so only add it if A1 is rejected.

**Expected gain.** ≤ +0.0005.

---

## B. High-upside structural ideas (1–4 days each; run after A1–A4)

### B1. Cross-fitted two-stage ("tree message passing") imputation — **NEW**

**Precondition.** Confirm what "Stacked HGB" in EXP_T1_e meant. If it already fed stage-1 predictions of neighbouring masked cells into a second model, this idea is a DUPLICATE.

**Basis.**
- C0031: MICE chained equations. C0033: missForest. C0028: HyperImpute, iterative column-wise imputation.
- C0735: dual-stage error-corrected boosting for traffic imputation.
- C0260 / C0339: two-stage GBDT for Traffic4cast.
- C0180: stacked generalisation. C0185: cross-fitting (DML).
- C0009 / C0010: GRIN and SPIN get their gains from iterated spatio-temporal message passing.

**Weakness addressed.** At 20–50% i.i.d. masking, 20–50% of each target's neighbour inputs are interpolations rather than values. Stage-1 predictions at those neighbours are much better than interpolation (0.94 vs 0.91 state score).

**Code change.**
1. Stage 1 = the current model (or E-size, to save time). Produce **OOF** predictions for every target cell on train, using the 3 contiguous day-block folds. On val/test, use the full-fit stage-1 model.
2. Build a "filled" field: visible cells keep their observed values; masked cells get their stage-1 prediction.
3. Stage 2 = the same feature code run on the filled field: neighbours, spatial and temporal interpolation, and the A1 ASM on the filled field. Add the target's own stage-1 OOF prediction and its fold-disagreement.
4. Train a 255-leaf, 900-tree model. Optionally run a third round.

**Leakage.** Stage-1 values that feed any training row must come from a model that did not see that row's day-block. Val/test are clean by construction.
- Check: the stage-2 gain on train CV must be similar to the gain on the val holdout. A much larger CV gain means leakage.

**Compute.** Stage-1 OOF costs about 3× a small model; stage 2 costs about 1× F. That is roughly 2–4 h extra on 4 cores. It fits in 12 h if stage 1 uses E-size.

**Expected gain.** +0.002 to +0.008, largest in the 50% regime. Density errors should shrink where neighbours were interpolated through congestion.

**Reject.** Holdout Δ < +0.0015, or a train-CV gain more than 2× the holdout gain (leakage signal).

### B2. Small CPU space-time inpainting network as a stacked feature — **NEW**

**Basis.**
- C0098: deep CNN imputation on time–space diagrams, beating ASM.
- C0006: anisotropic kinematic-wave kernels.
- C0049: partial convolutions for irregular holes. C0570: masked autoencoders.
- C0007: priors plus mask indication.
- C0008: mixed-rate masking during training.

**Code change.**
- Input: 5-channel patches of 32 links × 48 slots — speed/v_f, flow/q_c, occupancy, visibility mask, Δx. The model is a U-Net of 3 levels with about 0.2–0.5M parameters, using partial convolutions or mask-concatenation.
- Train on the 9 train months with random i.i.d. masks at 20/30/50% plus random 90-min blocks. Mix the rates (C0008). Loss: MSE on hidden cells, plus an L1 on q/v for the Task 3 term.
- Output: the predicted field used as an LGBM feature, with OOF predictions by day-block for the train rows, as in B1.

**Compute.** CPU torch with 4 threads, deterministic seeds. Roughly 1–3 ms per patch step, so about 1–2 h for around 1M patch steps. Inference takes minutes. This is our estimate; benchmark first on one panel.

**Expected gain.** Uncertain: +0 to +0.005. It helps only if the CNN captures wave patterns beyond A1 + B1.

**Reject.** Run it only after A1 and B1. Reject if Δ < +0.0015 on top of them, or if reproducibility across two runs differs by more than 0.0003.

### B3. Calibrated CTM data assimilation ("hard physics") as a feature — **NEW, risky**

**Basis.**
- C0015: CTM. C0071: variational kinematic waves. C0275: CTM density estimation (SMM).
- C0076: stochastic CTM.
- C0534: an FNO trained on LWR simulations.
- Map fact: the data come from a calibrated CTM/LWR-like generator with measurement noise and incidents.

**Idea.** Using the per-link FD from A5, run a Godunov CTM forward over each day.
- Boundary inflow comes from the most upstream observed flow; the downstream boundary supply comes from the most downstream observed state.
- At every slot, nudge cells where observations are visible (Newtonian relaxation, weight tuned on hidden released cells).
- Also run a backward pass, and use the average.
- Output the CTM density, speed and flow as features. This differs from the dropped Kalman/RTS experiment, which ran on *profile deviations* without a flow model.

**Compute.** A CTM over 350 cells × 288 slots × 30 days is milliseconds per day.
- Needs a sub-step if L/v_f < 5 min; the CFL condition likely requires dt ≤ L_min/v_f, meaning sub-steps of about 20–30 s.
- Still only seconds per panel.

**Expected gain.** Unknown. It could be large on congested and blackout cells if the generator is close to a triangular-FD CTM. It could be about 0 if incidents and capacity drops dominate.

**Reject.** Δ < +0.001, or the LWR proxy is not improved.

### B4. Characteristic-sheared or Hankel low-rank completion — **NEW, speculative**

**Basis.**
- C0220: low-rank Hankel tensor completion for travelling waves.
- C0005: LATC.
- A1 gives the wave speeds.

**Idea.** Plain low-rank structure blurs moving jam fronts. Shift each link's series by its cumulative congested-wave travel time from a reference link, which makes jams roughly "vertical". Then run SoftImpute on the sheared matrix and un-shear. Alternatively, use a Hankel (delay-embedded) matrix per corridor.

**Compute.** Minutes.

**Expected gain.** +0 to +0.002.

**Reject.** Run only if A3 passes. Reject if Δ over A3 < +0.0007.

### B5. Deep spatio-temporal imputers (SAITS/BRITS via PyPOTS, SPIN, ImputeFormer, PriSTI/CSDI) — **NEW, not recommended**

**Basis.**
- C0009, C0010, C0008 (all DEEP_READ): gains are reported against MICE/MF/BRITS, never against temporal interpolation or GBDT.
- SPIN only matches GRIN on point missing (0.70 vs 0.67 MAE) and needs 4–26 GPU-hours.
- C0007: diffusion inference is very slow.
- C0264: PyPOTS gives one-line SAITS/BRITS if someone wants a sanity check.

**If tried at all.** Use one small panel, SAITS with d_model 64 on CPU, output used as a stacked feature.

**Reject.** Reject by default unless it beats B2 on that panel.

---

## C. Negative evidence (do not re-propose)

1. **PINN / PIDL / physics-loss training.**
   - C0011 (DEEP_READ): PIDL helps in *very sparse* sensing, where relative errors are 20–40%. EKF RE_ρ was 39.5%, NN 42.8%, PIDL 31.5%, PIDL+FD-learner 21.2%. Our fields are 50–80% observed.
   - C0012 (DEEP_READ): PIDL cannot place or size LWR shocks, and a diffusion term does not fix this.
   - C0549: low-resolution loop data make physics and data gradients misaligned. The paper is **withdrawn**, so treat it as weak evidence.
   - C0557: gains depend on sensing density.
   - C0376: an LWR prior is better only with dense observation.
   - Our own density reconciliations (EXP_T1_f/g/m) gave no gain. CONTRADICTED.
2. **Kalman / EKF / particle filters.**
   - Our Kalman/RTS was −0.019. C0275 CTM-SMM reports about 13% error, and the EKF in the C0011 table is the worst estimator. CONTRADICTED.
   - B3 differs because it uses the CTM as a feature, not a filter on profiles.
3. **Regime MoE / gating.**
   - Our per-regime models were −0.002 on every panel. C0505 and C0311 are forecasting- and GPU-oriented MoEs. CONTRADICTED.
   - Soft regime information should enter as *features* (A1's W, A5's k/k_c), not as separate experts.
4. **Forecasting STGNN architectures and frameworks.**
   - The survey (C0007) shows forecasting models degrade with missing rate unless masks are modelled.
   - Our graph is a 1-D chain, so graph learning adds nothing over neighbour indices.
   - LibCity, BasicTS (C0265), PyG-Temporal and UCTB are forecasting-oriented and GPU-bound.
   - STID-style identity embeddings (C0624) conflict with our profile-free result.
5. **Low-rank as a replacement.**
   - LATC fails on block missing (C0008 Table 2: 4.05 vs 0.94 MAE).
   - Tensor completion did not improve downstream prediction when local data were rich (C0501).
   - Keep low-rank as features only (A3).
6. **Diffusion / GAN imputers.** PriSTI and CSDI (C0324, C0320) have very slow inference and need GPUs; there is no CPU path at our cell counts.
7. **Literature baselines are weak for our setting.** GRIN, SPIN and ImputeFormer tables contain no linear-interpolation or GBDT baseline. Reported 20%+ gains over MICE/MF say nothing about gains over our LightGBM residual model.

## Suggested order

1. A5 step 0 (diagnostic)
2. A1
3. A2
4. A4
5. A3
6. B1
7. B3
8. B2
9. B4

Combine the survivors, then submit one LB probe per accepted block. Earlier probes showed that effects are additive.
