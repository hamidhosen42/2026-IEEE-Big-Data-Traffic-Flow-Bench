# Final research report — Traffic Flow Bench

**Date:** 2026-10-01. **Current best public LB:** 0.86217 (COMBO5). The prompt said 0.85309, which is out of date.

**Companion files:**
- [`EXISTING_MAP.md`](EXISTING_MAP.md): what we already did.
- [`resource_index.csv`](resource_index.csv): the merged resources.
- [`research_to_experiments.md`](research_to_experiments.md): EXP_NEW_001–055 in full.
- `partials/`: each research agent's raw list and notes.
- `merge_resources.py`: the dedup script.

**Method.**
- Four research agents covered four source families:
  - A — competition, generator and Traffic4cast;
  - B — Task 2 onset, IoU, calibration and shift;
  - C — reconstruction, imputation and physics;
  - D — model families, ensembles, validation and ODME.
- Every row needs a real URL the agent saw; nothing was invented.
- I checked every resource id cited in `research_to_experiments.md` against the partial CSVs (58/58 exist) and spot-checked 23 titles.

---

## A. Resource statistics

| Measure | Count |
|---|---:|
| Raw rows in the four partials | 2,243 |
| Harvested before relevance filtering (agent C: ~2,700 paper records + 480 repos) | ≈ 3,200+ |
| **Unique after dedup** (DOI, arXiv id, URL, title, repo root) | **2,114** |
| DISCOVERED (search / API metadata only) | 1,116 |
| SCREENED (abstract or summary read — see caveat) | 872 |
| DEEP_READ (full text or key sections) | 77 |
| CODE_INSPECTED | 49 |

**Caveats on the counts:**
- For agent C, SCREENED means the title plus a sentence taken from the abstract, not a full-abstract read. Its 684 SCREENED rows are the loosest part of the count.
- The web-search budget (200 calls) ran out for every agent. Discovery then continued through OpenAlex, Crossref, arXiv/DataCite and the GitHub API, and the arXiv API returned 429 errors.
- **Deep engagement is about 126 resources** (77 deep-read + 49 code-inspected). The larger numbers are breadth, not depth.

**Year distribution** (42 rows have no year):

| Years | Rows |
|---|---:|
| < 2000 | 47 |
| 2000–09 | 164 |
| 2010–15 | 215 |
| 2016–19 | 282 |
| 2020–21 | 274 |
| 2022–23 | 429 |
| 2024 | 202 |
| 2025 | 206 |
| 2026 | 252 |

**Resource types:**

| Type | Rows |
|---|---:|
| Paper / article / conference paper | ≈ 1,240 |
| Preprint | 425 |
| Code repository / code | ≈ 300 |
| Commits / PRs / official docs (organizer repo) | ≈ 45 |
| Forum threads | 13 |
| Blogs / book chapters / other | rest |

**By task tag:**

| Task tag | Rows |
|---|---:|
| T1 | 631 |
| T1;T3 | 380 |
| T2 (all sub-tags) | ≈ 710 |
| T4 | 61 |
| All tasks | 105 |

---

## B. Research map

### Task 1 reconstruction
- **Strongest new levers:**
  - transductive target-month rows (A0058/A0059; D0153/D0154);
  - cross-fitted two-stage imputation (C0031, C0033, C0735, C0260, C0339).
- **Plausible:** ASM wave-direction kernels (C0001/C0002, code inspected; C0013, C0014).
- Deep imputers (C0009, C0010, C0008) never compare against GBDT and are GPU-heavy.

### Task 2 onset
- Our decoder is exactly RankSEG's E[I]/E[U] rule (B, code inspected), so the decoder is near-optimal.
- The onset gap to CV is mostly unforecastable windows: 3% of OOF onset windows have max p < 0.5, and they score IoU 0.10.
- Hisernberg's onset (0.732) is not better than ours (0.743).

### Task 2 ongoing
- This is the largest measured gap: Hisernberg 0.843 vs our last measured 0.747. Our current value, by inference, is ≈ 0.81.
- Evidence (A0059) says: drop location *priors*, keep link *identity*, and blend with a full model.

### Task 3 physics
- It depends only on Task 1 density accuracy at masked cells.
- TV smoothing of density (+0.0006 LB) and gated reconciliation are the only documented post-hoc gains (A0059).
- The practical ceiling for S_LWR is about 0.96 (A0022).

### Task 4 ODME
- Saturated: our NNLS is the organizer's own build script (D).
- The truth is a ridge fit to the prior (A0060), so the most any change could add is ≈ 0.0012 of S_total.

### Validation
- CV→LB is ≈ 0.84 and systematic.
- A test-month (March) pseudo-holdout caught failures that CV missed (A0074/A0075).
- Using April labels to choose the private model would be post-T steering, so it is not used.

### Calibration
- The model is already calibrated in-distribution.
- Temperature ≥ 2 collapses onset.
- Isotonic and power calibration flip sign between pseudo and official windows (our test).

### Ensemble
- Our own ≈ 0 results for blends, seeds and CatBoost ongoing match the literature (D: heavy stacking overfits).
- The exception: Task 1 seed bagging transferred 85–95% for Hisernberg.

### Domain adaptation
- Importance weighting and invariance training rarely beat plain training (B, D).
- Weekend re-weighting is contradicted: weekends are *easier*.
- Shift corrections must use only data ≤ T.

### Generator-specific insights (FACT, with source)
- **Master seed:** a stable integer from (release, panel, split), with child seeds for the observation noise and the Task 4 prior (A0037).
- **Scenario contract:** capacity multiplier 1.2, incident severity 1.0 (A0026).
- **Noise:** speed noise sd ≈ 1–1.9 km/h; flow noise ≈ 28 vph/lane. The noise is white in time but correlated across links (A0063, A0086).
- **Masking:** the mask hash is blake2b with one regime per day (A0020). Mask cells blank **all channels** (our check: flow and occupancy are masked at 100% of speed-masked cells).
- **Ramps:** ramps stay released inside mainline blackouts (A0058).
- **Official windows:** they fall in days 0–5 of each month because of the greedy selector (A0074).

---

## C. What our current pipeline already does well

1. **Task 2 decoding is Bayes-consistent for IoU.**
   - Expected-IoU top-m equals RankSEG-RMA (B), so there is no decoder lever left: every alternative measured ≤ +0.0006.
   - Its probabilities are well calibrated (our test and B's).
2. **Shift robustness where it matters.**
   - Task 1 has no profile features.
   - The ongoing model is prior-free: LB +0.0045, which CV did not predict.
   - This matches the "location priors overfit the month" evidence (A0059).
3. **Onset** is on par with or ahead of the best documented public pipeline (0.743 vs 0.732), using the exact replicated selector and CatBoost.
4. **Task 4** is at the organizer's reference solution (≈ 0.994).
5. **Disciplined gates:**
   - a val-month holdout for Task 1;
   - frozen day-block folds and official windows for Task 2;
   - LB swaps with 100% CONTROL agreement.
   - Effects have been additive on the LB (COMBO1 matched the sum exactly).

---

## D. Current bottlenecks (evidence-supported)

1. **Task 2 ongoing on the new month.**
   - Our last measurement was 0.747 at LB 0.84855, against Hisernberg's 0.843.
   - The CV→LB ratio is ≈ 0.84 for us and ≈ 0.82 for persistence, so the gap does not come from our model alone.
   - Label noise is **not** the explanation: isolated queued cells are 0.9%, and a median-3 relabel changes 1.9% of queued cells (our pre-check).
2. **Task 1 never trains on the month it predicts.** This is the main documented Task 1 / Task 3 gain we have not taken: about +0.004 LB for Hisernberg.
3. **Physics (≈ 0.67)** is bounded by density accuracy at masked cells. No post-hoc fix has worked for us.
4. **Decision noise.** A 40-window ongoing LB delta has SD ≈ 0.005–0.009 (A0074), so single probes below 0.002 are unreliable.

---

## E. Rejected research directions (do not spend compute)

| Direction | Why |
|---|---|
| Time-series foundation models (Chronos / TimesFM / Moirai) | A 12-step history is shorter than one patch; they are weakest on transport data (D). |
| GNN / spatio-temporal frameworks (STGCN, DCRNN, GWNet, LibCity, BasicTS, …) | 1-D corridors on CPU; MLPs match ST-GNNs and windowed GBRT beats deep models (C, D). |
| Deep imputers (SAITS / BRITS / GRIN / SPIN / ImputeFormer / diffusion) | No GBDT comparison in the papers; GPU-hours; diffusion is too slow (C). |
| PINN / physics losses | They fail at shocks and help only at very sparse sensing (C0011, C0012); our density pulls failed. |
| MoE / regime gating (Task 1) | Our per-regime models lost 0.002 on every panel. |
| Kalman / profile features / per-lane spatial interpolation | Already rejected (map). |
| Probability calibration before decoding | Tested: signs disagree; temperature ≥ 2 collapses onset. |
| Onset hazard reformulation / T+5..T+25 onset | Contradicted by the truth definition (queue at T+30). |
| Weekend importance weighting | Weekends are easier, so re-weighting widens the gap. |
| Change-point / hazard / gradient features | Our tests found them to be noise. |
| Occupancy → speed identity | Channels are co-masked (our check). |
| Denoised / soft Task 2 labels | Noise share < 1% (our pre-check); deferred. |
| Task 4 method changes | Saturated; the ceiling is ≈ 0.0012. |

---

## F. Top 50 new experiments (ranked)

Full detail is in `research_to_experiments.md`. Status key: R = running, K = executed-keep, D = executed-drop / decided, P = proposed.

| # | EXP_NEW | Task | Idea | Status |
|---:|---|---|---|---|
| 1 | 001 | T1 / T3 | transductive target-month rows | R |
| 2 | 021 | T2 ongoing | identity-kept + full blends | R |
| 3 | 022 | T2 ongoing | 511-leaf model | K → COMBO7 |
| 4 | 003 | T1 | cross-fitted two-stage imputation | P |
| 5 | 004 | T1 | seed / bagging ensemble | P |
| 6 | 025 | T2 | March pseudo-holdout gate | P |
| 7 | 002 | T1 | ASM wave-kernel features | R |
| 8 | 048 | all | single-entry reproducible notebook | P |
| 9 | 014 | all | deterministic LightGBM | P |
| 10 | 046 | selection | final-pick hedge | P |
| 11 | 023 | T2 | zero-probe re-measure | P |
| 12 | 005 | T3 | TV density smoothing | P |
| 13 | 007 | T1 | ramp features / blackout rows | P |
| 14 | 009 | T1 | masked SoftImpute / LATC features | P |
| 15 | 010 | T1 / T3 | conservation / CTM / Newell features | P |
| 16 | 024 | T2 onset | hybrid label repair | P |
| 17 | 031 | T2 | shift diagnostic | P |
| 18 | 008 | T1 | mask-support features | P |
| 19 | 006 | T3 | gated density reconciliation | P |
| 20 | 029 | T2 onset | block-aware decoder | P |
| 21 | 030 | T2 ongoing | exact decoder | P |
| 22 | 026 | T2 | March pseudo-windows for the private file | P, rule check |
| 23 | 035 | T2 | window-size model → decoder | P |
| 24 | 028 | T2 onset | stacking + seeds | P |
| 25 | 027 | T2 onset | kNN analog features | P |
| 26 | 011 | T1 | FD-implied features | P |
| 27 | 012 | T1 | `linear_tree` | P |
| 28 | 047 | process | LB-noise discipline | adopted |
| 29 | 032 | T2 | tail-mass safeguard | P |
| 30 | 033 | T2 | denoised labels | D (deferred) |
| 31 | 013 | T1 | `extra_trees` | P |
| 32 | 015 | T1 | CPU U-Net inpainting feature | P |
| 33 | 016 | T1 | CTM data assimilation feature | P |
| 34 | 041 | T2 onset | CTM rollout feature | P |
| 35 | 042 | T2 onset | neural member | P |
| 36 | 034 | T2 onset | breakdown-probability features | P |
| 37 | 038 | T2 | LB-faithful local truth | P |
| 38 | 036 | T2 | causal prior-shift | P |
| 39 | 044 | T4 | KL / GLS projection | P |
| 40 | 018 | T1 | month speed-level correction | P |
| 41 | 019 | T1 | denoised neighbour inputs | P |
| 42 | 020 | T1 | 1023 leaves / 8× rows | P |
| 43 | 017 | T1 | sheared low-rank | P |
| 44 | 040 | T2 | calibration | D |
| 45 | 054 | T1 | occupancy identity | D |
| 46 | 043 | T2 onset | bigger CatBoost | D |
| 47 | 045 | T4 | stop Task 4 | decision |
| 48 | 039 | T2 | weekend IWCV | D (contradicted) |
| 49 | 037 | T2 | change-point features | D (duplicate) |
| 50 | 049–053, 055 | — | foundation / GNN / PINN / MoE / deep imputers / profile | D (contradicted) |

- **Top 30:** rows 1–30.
- **Top 15:** rows 1–15.
- **Top 5:** section H.

---

## G. Top 15 implementation candidates (plans)

| # | EXP | Plan (file → change → gate) |
|---:|---|---|
| 1 | 001 | `t1_pipeline_v9.py` → 2 × 20% re-hidden rounds of released val/private cells as FT rows → T1-holdout Δ ≥ +0.0007, then LB swap. Running as `tfb-t1v9-full`. |
| 2 | 021 | `t2_model_v14.py` → `og_id` (drops only prior/tod/dow), `og_full`, blends `bnf`/`bif` → LB probe each against COMBO7; keep if Δ ≥ +0.001. Running as `tfb-t2v14`. |
| 3 | 022 | Done: `t2_model_v13.py` → COMBO7 (`swap_combo7.py`), auto-submitted after the v13 kernel finishes. |
| 4 | 003 | New `t1_pipeline_v11.py` → stage 1 = E-size OOF over 3 day blocks; fill masked cells; rerun `features()`; stage 2 = F + own stage-1 prediction → holdout Δ ≥ +0.0015, train-CV gain ≤ 2× holdout gain. |
| 5 | 004 | In the v9/v11 final fit, train 3 seeds × F on different row subsamples and average → holdout Δ ≥ +0.0005. |
| 6 | 025 | New `t2_pseudo_march.py` → selector replica on masked validation-month data with observed labels → it must reproduce the signs of 3 known LB deltas. Then use it as the gate for private-file choices. |
| 7 | 002 | `t1_pipeline_v10.py` → `asm_feats()` → holdout Δ ≥ +0.0007. Running as `tfb-t1v10-full`. |
| 8 | 048 | `final/` → one notebook that trains Task 1 / 2 / 4 from scratch with fixed seeds → rerun twice; outputs identical, or within 1% of the scores. |
| 9 | 014 | All LightGBM params get `deterministic=True, force_row_wise=True` → part of #8. |
| 10 | 046 | Final selection: one conservative file plus one best-gated file. |
| 11 | 023 | `swap_zero_og.py` from COMBO7 → measure the ongoing IoU (costs 1 submission). |
| 12 | 005 | Post-process in the T1 kernel → 1-D TV on k inside runs of target cells, λ tuned on the holdout LWR proxy → LWR Δ ≥ +0.002 with no loss in S_state. |
| 13 | 007 | Load `ramp_states` → ramp flows at the nearest up/down ramps at t, t±1 → Δ on gap cells > 0. |
| 14 | 009 | Replace `lowrank()` with masked SoftImpute (rank 5, 20 iterations) → Δ ≥ +0.0007. |
| 15 | 010 | Balance / Godunov / Newell features from `fd_parameters.csv` → T1 + T3 proxy Δ > 0. |

---

## H. Top 5 next experiments (actionable against COMBO5 → COMBO7)

### 1. EXP_NEW_001 — transductive Task 1 (running)
- **File / function:** `kaggle_jobs/t1_pipeline_v9.py`, the per-panel block `# target-month cubes + transductive rows`.
- **Change:** extra training rows from the target months, as described in G#1.
- **Parameters:**
  - `TR_ROUNDS=2`, `TR_HIDE=0.20`;
  - holdout seed 7 (15%);
  - round seeds `1000 + 10r + (split == private)`;
  - F params: 511 leaves, lr 0.04, min_data 100, 1500 trees;
  - transductive rows capped at 4×MAX_ROWS.
- **Validation:** `val_F` vs `val_FT` family mean over 8 families, plus the per-panel count.
- **Smoke:** PASS (`tfb-t1v9-smoke`: FT 0.9364 vs F 0.9350).
- **KEEP / DROP:** KEEP if Δ ≥ +0.0007 and ≥ 6/8 panels improve, and then the LB swap (`state_submission_FT.csv` into COMBO7) gives > +0.0005. Otherwise DROP.

### 2. EXP_NEW_021 — ongoing identity / full blends (running)
- **File / function:** `kaggle_jobs/t2_model_v14.py`, after `og_big`: `og_id`, `og_full`; variants `id511`, `full511`, `bnf`, `bif`.
- **Parameters:** 511 leaves, lr 0.03, min_data 40, 1800 trees; `MAX_OG=3000`; `ID_DROP = {prior, prior_now, prior_alldays, tod, dow}`.
- **Validation:** LB swaps of ongoing rows only, against COMBO7. Order: `bif` first (the Hisernberg recipe), then `bnf`.
- **Smoke:** a config-level change of v12/v13, which has already run in full. The combo kernel gates (row count, 0/1 values, control agreement) act as the smoke.
- **KEEP / DROP:** KEEP if LB Δ ≥ +0.001, and the same sign on the March pseudo-holdout once #5 exists. DROP otherwise.

### 3. EXP_NEW_003 — cross-fitted two-stage Task 1
- **File / function:** a new `t1_pipeline_v11.py` based on v9.
  - A new `stage1_oof()` trains E-size models (255 leaves, lr 0.05, 900 trees) on 2 of the 3 frozen day blocks and predicts the third.
  - `features()` runs on `Cube` copies whose masked cells are filled with the stage-1 values.
  - New feature: `s1_self` (the stage-1 prediction at the cell).
- **Parameters:** stage 2 = F params.
- **Validation:** T1-holdout. Val and private use the full-fit stage 1.
- **Smoke:** the SMOKE copy on 2 panels; also check that the train-CV gain is ≤ 2× the holdout gain.
- **KEEP / DROP:** KEEP if holdout Δ ≥ +0.0015 on top of the #1 winner. Otherwise DROP.

### 4. EXP_NEW_004 — Task 1 seed bagging
- **File / function:** the final-fit block of the winning T1 pipeline. `MODELS['FTs']` = the mean of 3 models with `seed` / `bagging_seed` / `feature_fraction_seed` ∈ {0, 1, 2}, each on its own row subsample drawn with `rng(k)`.
- **Validation:** T1-holdout.
- **Smoke:** the SMOKE copy.
- **KEEP / DROP:** KEEP if Δ ≥ +0.0005. It costs 3× the training time, so check the 12-hour budget.

### 5. EXP_NEW_025 — March pseudo-holdout for Task 2 decisions
- **File / function:** a new `kaggle_jobs/t2_pseudo_march.py`.
  - Reuse `build_windows()` and `window_features()` from `t2_model_v13.py` on `validation/mainline_states_masked`.
  - Keep eligible coverage ≥ 0.7 and exclude windows within [T−12, T+18] of a released blackout.
  - Labels = observed eligible horizon cells ≤ 0.6·v_f.
  - Score cached model probabilities from the v13 / v14 models.
- **Validation of the gate itself:** it must reproduce the signs of three known LB results:
  - ongoing prior-free > full (LB +0.0045);
  - persistence blend < base (LB −0.0018);
  - onset 30/70 < 70/30 (LB −0.0039).
- **Smoke:** 2 panels, every 9th day.
- **KEEP / DROP:** if it matches 3/3 signs, it becomes the gate for private-file choices. If not, DROP it as a gate.

---

## Why are top teams > 0.90? (FACT vs INFERENCE)

**FACT:**
- **Post-rebuild scores, from the leaderboard API on 2026-09-30:** KTK 0.90187, gichang 0.89702, Lukas 0.89260, George Daniel Gherasim 0.89198 (A0003).
- **Scores with doubtful dates:** four entries at 0.91–0.92 (dated 09-12 / 09-19 / 09-28). They may be pre-rebuild; the organizer asked Kaggle to remove old scores (A0016).
- **No team above 0.88 has published code or a write-up.**
- **The best documented pipeline is 0.87249 (Hisernberg).** Its decomposition:

  | Component | Score |
  |---|---:|
  | Onset | 0.732 |
  | Ongoing | 0.843 |
  | ODME | 0.994 |
  | S_phys | ≈ 0.69 |

**INFERENCE:**
- A score of 0.90 needs S_queue ≈ 0.85+, which means ongoing ≥ 0.90 and onset ≥ 0.80 on the new month. Task 1, Task 3 and Task 4 are near saturation for every serious team.
- So the top teams most likely have much better *ongoing* transfer to the new month, and possibly better onset.
  - One possibility is heavy use of released val/private data where the rules allow it: Task 1 transductive training, and train/val pseudo-windows for the private file.
  - Another is models with link identity but without calendar priors.
- This is not verifiable, because no public evidence exists.

---

## Executed today (2026-09-30 / 10-01)

| Experiment | Result | Decision |
|---|---|---|
| T1 F (511 leaves) | holdout 0.9386 vs 0.9374 | KEEP → COMBO6, auto-submits 00:02 UTC |
| T2 ongoing 511 (oofY) | +0.006 pseudo, +0.0075 official, 4/4 folds | KEEP → t2v13 → COMBO7 |
| Calibration | signs disagree | DROP |
| Channel co-masking check | flow and occupancy are masked wherever speed is | DROP of the occupancy-identity idea |
| Label-noise pre-check | isolated queued cells 0.9% | denoising deferred |
| Transductive T1 (v9), ASM features (v10), ongoing identity blends (v14) | — | running on Kaggle |
