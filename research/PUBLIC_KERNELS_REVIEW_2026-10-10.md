# Public kernels review, 2026-10-10

Review only. Nothing was pushed or submitted.

**Sources.** Every kernel was pulled with `kjob.call('GET', '/kernels/pull?...')` and its run log with `/kernels/output`. Everything is saved under
`research/public_kernels_2026-10-10/`:

| File | Contents |
|---|---|
| `<user>__<slug>.ipynb` | exact source |
| `<user>__<slug>.txt` | the source flattened to text, with `# ===== CELL n` markers. The line numbers below refer to this file |
| `<user>__<slug>.log.txt` | the run log |
| `<user>__<slug>.meta.json` | the kernel metadata |
| `anubisyy_output/tfb_source/*.py`, `yeashusemwal_output/tfb/src/**.py` | the real model code. These two notebooks mostly load modules, and the code lives in their output files |
| `public_leaderboard_2026-10-10T05-54.csv` | full public LB, 387 teams, from `/competitions/.../leaderboard/download` |

**How the real scores were found.** The kernel metadata has no score field. The real public score comes from the team's row in the LB CSV, cross-checked against
the date of the kernel's run log. For anubisyy, the run log also records `recorded_solution_score` together with an asset sha256 check.

## 1. Real scores

| Notebook | Title claim | Real evidence | Verdict |
|---|---|---|---|
| sujanmajhisuzan/…top-rank-pipeline-0-90-target | "0.90+ Target" | Team ItsmeSujan: **0.85460**, 1 submission (2026-10-08 12:08; the kernel ran at 11:54) | The 0.90 is only a goal. The notebook is anubisyy's checkpoints plus a few tweaks, and it scores **−0.011** below anubisyy's 0.86517 |
| yanamiiannaa/…directional-features-safe-profiles | none | It writes no submission. Team Yanamii's best is 0.83779 | A 2-panel Task 1 pilot (Feb S_state 0.9300 → 0.9367) |
| anubisyy/…catboost-lgbm-nnls…-0-86517 | 0.86517 | The log has `recorded_solution_score 0.86517`, and every asset checksum passes. Team Anubisyy's best today is **0.87667** (rank 44) | The claim is genuine. The team's later 0.87667 code is not public |
| yeashusemwal/…trees-u-net-cnn-lb-0-85 | LB 0.85 | Team Yeashu Semwal: **0.85556** (rank 118, 2026-10-03) | Consistent with the claim |
| sachith7/…when-does-temporal-repair-fail | none | No submission; not on the LB | A Task 1 interpolation diagnostic on train dates |
| shuangranse/…06970-solution-writeup | 0.6970 (0.7269 hybrid) | Team sunshibo1101 (shuangranse + sunshibo1101): 0.82592 (rank 193) | The write-up itself is about the 0.7269 pipeline |

**Context from the same LB snapshot:**
- Our team Hack2Publish is at **0.88179**, rank 33. The submission is dated 2026-10-09 09:48. Note that this is 0.88179, not the 0.88176 in memory.
- The 0.89–0.903 group are KTK 0.90303, Lukas 0.90066, gichang and Gangmyung Doo 0.89859, Amin Mohamed 0.89667 and takaito 0.89664. All of them have 48–100 submissions.
- New since the 10-02 audit: **Musa Peker 0.92226** (94 submissions, last submission 10-05).
- **No author of these six notebooks scores above us.**

## 2. Techniques per notebook

The tasks are T1 state, T2 queue (onset / ongoing), T3 physics and T4 ODME.

**Rule compliance.** The rule is that a T2 forecast with origin T may only use data timestamped ≤ T. Every notebook passes:
- anubisyy and sujan use `window_history` only, which is < T (the assertion is at anubisyy txt l.1213).
- yeashusemwal uses `window_history` plus the masked mainline at slot T (`task2_v10.py` l.54). Its v15 adds 3 h of masked mainline from T−180 to T (`task2_v15.py` l.92: `mg["v"][:, lo:t0+1]`). Both are ≤ T.
- T1 and T4 are offline everywhere.

**No violation found.**

### anubisyy (0.86517; team now 0.87667)

**T1:**
- Residual to temporal interpolation, with flow averaged 50/50 with the mean flow of adjacent links (`state_model.py` l.112–186).
- 54 base features plus 40 lag/lead features at ±5/10/15/30/60 min. Each lag carries v, q/lane, q/v and an "eligible and finite" flag (`prepare_state.py` l.12–16).
- Training masks are simulated i.i.d. at 20/30/50 %, with 2 mask seeds per day.
- **CatBoost RMSE regressors at depth 8 (9000 iterations) and depth 10 (4500 iterations), averaged 50/50** (`train_state.py` l.9; txt l.90–117).
- Feb proxy 0.9349 → 0.9418, and **LB +0.0042** for "quality-aware features + CatBoost" on top of their own LightGBM Task 1.

**T2 onset:**
- 0.5 LightGBM (83 features, 1900 trees) + 0.5 CatBoost (depth 7). The CatBoost training histories get **5 % random cell dropout as augmentation**, with labels unchanged (`train_onset_catboost.py` l.18–22).
- T+30 only, fixed threshold 0.55, and a top-1 fallback.
- Training uses the first origin of each feasible run.
- The isolated LB gain was +0.0007, which is inside the onset-probe noise.

**T2 ongoing:**
- One LightGBM model, 36 features, 700 trees.
- Window weight 1000 / eligible cells (`queue_ongoing.py` l.210).
- Fixed threshold 0.40.

**T3:** clipping only.

**T4:**
- The same objective as ours, min ½‖Af−c‖² + λ/2‖f−b‖² with λ 0.05.
- Solved by L-BFGS-B from a 600-iteration warm start, tightened to maxiter 8000 and ftol 1e-14. This alone gave **0.84833 → 0.86025 (+0.012)** (txt l.1285, l.1337).

### sujanmajhisuzan (0.85460)

This is anubisyy's public assets and modules, with these changes:
- T1: clip v ∈ [5, 135].
- T2 onset: threshold 0.52 and a top-2 fallback.
- T2 ongoing: threshold 0.40 plus **1-D morphological closing**, which fills a single-link gap between two predicted links (txt l.102–117, 157–161).
- T4: unchanged.

It scores **0.0106 below** anubisyy's 0.86517. This independently confirms our EXP_T2_D result (morphology −0.010 to −0.015).

### yeashusemwal (0.85556)

**T1:**
- 3 LightGBM models with leave-one-day-out weekday × slot profiles.
- Blended **0.70 / 0.30 with a 2-seed day-grid U-Net**, which works on links × 288 slots with 13 input channels.
- The U-Net is trained with fresh masks every epoch plus simulated 90-min gaps.
- The U-Net loss is metric-shaped, **plus a per-lane density term weighted 0.05**: `dk = (qp/vp − qt/vt)/lanes` (`task1_unet2.py` l.1–6, l.110–115; blend at txt l.49 and l.126–135).
- Held-out S_state: LightGBM 0.9377, U-Net 0.9312, blend 0.9397.
- LB: +0.0010 at a 30 % U-Net share, and −0.0012 at 50 %.

**T2 onset:**
- An average of 3 dilated residual Conv1d CNNs run along the corridor. The input is history T−60..T, LOO profile slices and static link features. Predicts T+30 only, with threshold 0.5 and a top-2 fallback (`t2_cnn_infer.py` l.17–28; `task2_nn.py` l.1–5).
- LB +0.0019 vs trees.
- An unused v18 variant up-weights "non-recurrent" onset events ×3 (`task2_v18.py` l.2–4, l.102).

**T2 ongoing:**
- 7 LightGBM members:
  - v11–14: LOO profiles;
  - v15/16: 3 h of same-day context, queue age, queue-front speed and ramps, worth +0.0004;
  - v20: windows the selector could have picked, worth +0.0003, credited to our hosen42 t2v2 observation.
- Training uses **random history blanking of 15 % of links** (v18; `task2_v18.py` l.1–2, l.18–24).
- Fixed threshold 0.40.
- A CNN for ongoing cost −0.0049 LB.

**T4:** NNLS on the split's own counts and prior, λ 0.05, the same as ours. Its log is "S_link 1.0000".

### yanamiiannaa (no LB)

- T1 only, 2 panels. A small tree model on 27 features: before / after / upstream / downstream nearest visible observations within 5 hops, and whole-day-excluded profiles.
- Blended 70/30 with an archived temporal CNN.
- Feb macro S_state 0.9300 → 0.9367. Our members are about 0.942 on the val-month holdout, a different holdout.

### sachith7 (no LB)

- A T1 diagnostic on train dates.
- Temporal interpolation beats spatial interpolation and the 7/28-day profile on 5 corridors. A temporal-then-spatial fallback for long gaps (≥ 7 slots) does not help.
- **Day-edge cells** (no same-day observation on one side) are the weak spot.

### shuangranse (writes 0.7269; team now 0.82592)

- **T1:** weekday profile + Kalman/RTS on deviations. This is our EXP_T1_c (−0.019).
- **T2:**
  - Onset uses the top-3 recurrent-bottleneck links at T+30, gated by flow/capacity.
  - Ongoing is persistence + LightGBM.
  - Non-sensored links between two queued sensored links are filled (txt l.129–130, l.182).
- **T3:** soft-FD / conservation post-processing, dropped.
- **T4:** NNLS on the split's own counts, which matches `public_reference` to 1e-8.

## 3. What is new relative to TRACKER, ranked

**Already covered or rejected by us. Do not redo:**
- LOO / day-excluded profiles: T1 is profile-free, ongoing is prior-free, onset LODO gave LB −0.005.
- Equal window weights: EXP_T2_l.
- Thresholds, morphology and gap fill: EXP_T2_D.
- 3 h same-day context: EXP_T2_e and OGCLOCK (new-scenario gain ≈ 0).
- Selector-faithful ongoing windows: F03, already in production at LB +0.0047.
- Kalman/RTS: EXP_T1_c.
- Persistence blend: EXP_T2_n.
- CNN for ongoing: their own LB −0.005.
- **ODME solver tightening.** Our T4 is `scipy.optimize.nnls` on [A; √λ·I] (`consolidation/kaggle_parity/tfb-par-t4.py` l.1080–1081). That is an exact active-set solution of the same strictly convex objective. anubisyy's +0.012 only recovers what their under-converged L-BFGS lost, so it is worth **0** to us.

**Never tested by us:**

| # | Technique | Where | Our nearest test | Estimated public Δ for us | Cost | Rule |
|---:|---|---|---|---|---|---|
| 1 | **T1 day-grid U-Net with a density-consistency loss, as a low-weight blend member (≈ 0.15–0.3) before recon + TV** | yeashusemwal `task1_unet2.py` l.1–6, 110–115; blend txt l.126–135 | Proposed as EXP_NEW_015 (`research/research_to_experiments.md` l.177), never run. All 16 T1 members are LightGBM | Their result: +0.0020 holdout, +0.0010 LB on a weaker LightGBM. Our bag is stronger and the U-Net is ≈ 0.01 S_state weaker than our members. Expected **+0.0001 to +0.0005** | GPU training per panel (≈ 1–2 h), cheap inference; check on the val-month holdout first | ✓ offline |
| 2 | **T1 CatBoost residual member, depth 8 + depth 10, on our FD feature set** (model-family diversity) | anubisyy `train_state.py` l.9; txt l.90–117 | None. CatBoost was tried only for T2 (B4: correlation 0.997 on ongoing) | Their own +0.0042 LB is on a much weaker T1. For a 17th member of a different family, expected **+0.0001 to +0.0003** | GPU CatBoost on about 2.4 M rows × 98 columns ≈ 1–2 h per half | ✓ offline |
| 3 | **T1 long lag/lead self context: ±3/±6/±12 slots (15/30/60 min) of v, q/lane, q/v plus an eligibility flag** | anubisyy `prepare_state.py` l.12–16 | We have self ±1/±2 slots, ASM ±5 slots × ±5 stations, interpolation and gap distances | **+0 to +0.0002** (mostly redundant with interpolation + ASM) | One lab run on the val-month holdout; 10 extra columns | ✓ offline |
| 4 | T2 training augmentation: random link-history blanking of 15 % (yeashusemwal) or cell dropout of 5 % (anubisyy) | `task2_v18.py` l.18–24; `train_onset_catboost.py` l.18–22 | The masking-adaptation control (EXP_PV_VALIDITY) is related but is not augmentation | Ongoing **≈ 0 to +0.0005**, and it cannot be verified on 40 public windows. Onset: do not probe (8 onset failures; probe SD 0.003–0.005) | Retrain the F03 bag (≈ 5 h) | ✓ |
| 5 | Onset Conv1d corridor CNN (seed bag of 3) | `t2_cnn_infer.py` l.17–28; `task2_nn.py` | Listed as "CNN onset, no evidence" in `analysis/LEVERS_2026-10-07.md` l.110 | Their +0.0019 is inside the onset-probe noise. **Not recommended** (onset LB record 1 in 8) | GPU, 3 seeds | ✓ |
| 6 | T1 interpolation across midnight for day-edge targets (only inspired by sachith7's day-edge finding; they do not implement it) | sachith7 txt "Gap length and day-edge cells" | Our `interp_axis(c.mv, 1)` works within each day (`kaggle_jobs/t1_pipeline_v13a.py` l.218) | **< +0.0001** (night traffic is free-flow and edge cells are few) | Tiny | ✓ offline |

**Combined.** The realistic total of items 1–3 is about **+0.0003 to +0.001** public. All of it is in Task 1, so it transfers to private 1:1.

## 4. Verdict

**No public notebook explains how the 0.89–0.90 teams outscore us.**

- Every one of the six authors is below us on the same LB. The strongest, Anubisyy, is at 0.87667 against our 0.88179.
- Their public code is weaker than ours on T2:
  - fixed thresholds instead of expected-IoU decoding;
  - a single ongoing model;
  - no selector-faithful windows, except yeashusemwal's v20 at 20 % weight;
  - no same-scenario prior.
- On T4 they match ours. On T1 they are at best comparable.

**Where the gap must be.** The gap is 0.019 to KTK and 0.017 to the 0.898 group. Our own decomposition (`analysis/TRANSFER_ANALYSIS_2026-10-08.md` §3) puts the recoverable loss almost entirely in Task 2:
- onset location, ≈ 0.02 LB;
- ongoing per-step extent, ≈ 0.009 LB.

Nothing in these notebooks addresses either one:
- yeashusemwal names onset localisation as "the main open problem".
- The only T2 ideas here are augmentations and a CNN onset, both small or noise-level.

**What would move us.** The only items worth time are T1 diversity members (U-Net with a density loss, then CatBoost). They are cheap to check on our val-month holdout, rule-safe, and transfer 1:1 to private. Even so they are worth about 0.001, roughly 5 % of the gap.
