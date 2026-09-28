# TrafficFlowBench experiment tracker

Base notebook: `notebooks/0.8094_LB__soukeaizenz__physics-pipeline.ipynb`. A read-only copy is kept at
`kaggle_jobs/base/EXP000_base_original.ipynb`, and it runs unchanged as the Kaggle kernel `hosen42/tfb-exp000-base`.
Its public LB score (0.80939) is the notebook's own claim.

## Validation design

**Frozen folds.** Every Kaggle run uses the same fixed folds.

**Task 1 and Task 3 proxy.** Scored on the train split, using the targets from the official `task1/<panel>/train` template.
- Folds are 3 contiguous day blocks.
- Aggregation matches the competition: regime → panel → family → overall.
- LWR proxy: `1 - Σ|ΔN_pred - ΔN_obs| / Σ|ΔN_obs|`, with `N = q/max(v,1)·L`. Released truth is used in non-target cells.
  - This matches the organizer docstring: boundary flows are "projected onto the released observations".
  - Calibration on D12_I405_S: the naive baseline scores 0.10 on the proxy, which gives S_physics ≈ 0.39 (official naive figure: 0.355).

**Task 2.** Windows come from a replica of the official selector.
- Selector rules:
  - History is [T-60, T), with at least 70% of cells eligible.
  - Ongoing means the history has ≥2 queued observations, with ≥2 on one link; otherwise the window is onset.
  - The horizon must contain a queue.
  - Persistence IoU must be ≤ 0.9.
  - Windows are picked greedily in time order, with global spacing of ≥ 360 min.
- The onset selection reproduces all 5 official onset windows in each of the 8 panels.
- In 97% of pseudo windows, and in all 40 official onset windows, the onset queue first appears at T+30.
- Folds are 4 contiguous day blocks.
- Metric is IoU on sensored, eligible cells. Truth proxy is observed speed ≤ 0.6·v_free.

## Experiments

| ID | Experiment | Change | Local score | Δ vs base | Fold std | Smoke | Leakage check | Status / decision |
|---|---|---|---:|---:|---:|---|---|---|
| EXP_000 | Base notebook, unchanged | none | LB 0.80939 (author claim) | 0 | – | – | – | Running on Kaggle, used for reproduction |
| EXP_T1_a | Temporal interp only (D12_I405_S, local) | – | S_state 0.9132, LWR 0.523 | – | – | PASS | clean | Reference |
| EXP_T1_b | Base Task 1 recipe, re-implemented (D12_I405_S, local) | – | 0.9219 / 0.568 | 0 | – | PASS | clean | Reference |
| EXP_T1_c | shuangranse Kalman/RTS | – | 0.9029 / 0.475 | −0.019 | – | PASS | clean | **REJECT** |
| EXP_T1_d | Spatial interp on per-lane flow | per-lane spatial | 0.8672 / 0.000 | −0.055 | – | PASS | clean | **REJECT**: total flow must be interpolated |
| EXP_T1_e | Stacked HGB, neighbour features (local, 3-fold by day) | model | 0.9391 / 0.589 | +0.017 / +0.021 | – | PASS | day-grouped | KEEP → LightGBM on Kaggle |
| EXP_T1_f | Density model: q = k̂·v̂ | target | 0.9335 / 0.557 | −0.006 vs e | – | PASS | clean | **REJECT** |
| EXP_T1_g | Density pull toward interpolated k (w 0.25–1, a 0–0.5) | post-proc | best T1+T3 0.4370 vs 0.4373 | −0.0003 | – | PASS | clean | **REJECT** |
| EXP_T1_h | LightGBM residual model, smoke (2 panels) | model | 0.9389 vs base 0.9337 | +0.0052 | 0.0010 | PASS | contiguous blocks | Full run executing |
| EXP_T1_i | + simulated 90-min blackout gaps, smoke | train view | valmix 0.9368 vs base 0.9320 | +0.0048 | 0.0016 | PASS | gap origins from train only | Full run executing |
| EXP_T2_a | Base heuristic (clusters + persistence), 8 panels | – | pseudo 0.7015 / official 0.7348 | 0 | – | PASS | clusters were hand-mined on train (in-sample) | Reference |
| EXP_T2_b | Onset template per hour + ongoing persistence∪template | rule | pseudo 0.7393 / official 0.7657 | +0.038 / +0.031 | – | PASS | 4-fold by day | Superseded by EXP_T2_c |
| EXP_T2_c | LightGBM cell classifier per condition + expected-IoU decoding (8 panels, 4 day blocks) | model | pseudo 0.8791 / official 0.8755 | +0.178 / +0.141 | 0.012 | PASS | features use data < T plus the masked slot T; the history/T alignment bug was fixed before the full run | KEEP |
| EXP_T2_d | + 30% prior-free model blend (blend_70) | ensemble | pseudo 0.8823 / official 0.8827 | +0.003 / +0.007 vs c | 0.009 | PASS | clean | **KEEP**. LB probe (T2 only): **0.84357** vs base 0.80939 |
| EXP_T2_e | + context features (same day before T, previous 1/7 days) from the masked layer; 800 trees, 3000 ongoing windows | features | pseudo 0.8822 / official 0.8839 | ≈0 | 0.010 | PASS | data ≤ T only | REJECT (no gain; keep the simpler model) |
| EXP_T2_f | Hypothesis: flagged-unsensored links are never observed | diagnostic | they are observed at 83%, same as sensored links | – | – | – | – | REJECTED hypothesis |
| EXP_T1_j | Task 1 v3: LightGBM + simulated blackouts, 10 panels, 300 trees | model | train valmix 0.9309 vs base 0.9102; LWR 0.483 vs 0.468 | +0.021 | 0.0014 | PASS | contiguous blocks | KEEP. SUB_A (T1 v3 + T2 d + T4) LB **0.84499** |
| EXP_T1_k | Base notebook's own Task 1 code run on train (3 panels) | reproduction | speed/flow RMSE 1.72/31, 3.23/38, 2.14/43 | – | – | – | – | Confirms the re-implemented base is faithful |
| EXP_T1_l | Task 1 v4: + occupancy features, speed cap 1.05·v_free, validation-month internal holdout (15% of released cells hidden) | features | val-month holdout: base 0.9141, lgbm 0.9280, **lgbm without profile features 0.9307** (better on 10/10 panels) | +0.017 vs base | – | PASS | holdout cells come from released data only | **KEEP, profile-free for the final predictions** |
| EXP_T2_g | LB decomposition probes: FINAL v2 with onset (resp. ongoing) queue zeroed | LB probe | LB 0.73711 / 0.73657 → **LB onset IoU 0.743, ongoing 0.747, S_queue 0.745** | – | – | PASS | exact: drop = 0.3·0.5·S_cond | Diagnostic |
| EXP_T2_h | Hypothesis: truth uses topology v_free (105) instead of links.csv (≈114) | diagnostic | label IoU between the two definitions 0.85–0.999 (mostly ≈0.98); observed p85 speed ≈ links.csv | – | – | PASS | – | REJECTED |
| EXP_T2_i | Official persistence baseline scored locally | calibration | local 0.307 (official windows) / 0.331 (pseudo) vs LB 0.2518 → ratio 0.82; our model ratio 0.745/0.883 = 0.84 | – | – | PASS | – | CV→LB gap is systematic (proxy / month), not model-specific |
| EXP_T2_j | Ongoing windows: prior-free model only (og_noprior) | ensemble | local ongoing 0.908 vs 0.916 (slightly worse); only 0.4% of queue rows change | LB **0.85309** vs 0.84855 (+0.0045 → LB ongoing IoU ≈ +0.030) | – | PASS | clean; control check: rebuilt main = FINAL rows 100% | **KEEP — new best** |
| EXP_T2_k | Onset: 30/70 main/prior-free (with og_noprior) | ensemble | local onset worse (blend_50 0.862 < blend_70 0.884); only ~70 cells change | LB 0.84917 (−0.0039 → onset IoU ≈ −0.026) | – | PASS | clean | **REJECT** — onset needs the location/time priors; next: test MORE prior on onset (on85 / main only) |
| EXP_T2_l | v6: queue-block features (length, growth, relative position, 30-min distance) + equal-window weights + non-recurrent eval subset | features | pseudo blend_70 0.8825 (= v2 0.8823); weighted no gain; non-recurrent subset (145 windows) does NOT reproduce the LB effect of priors | 0 | 0.008 | PASS | clean | REJECT (features/weights); the local non-recurrent subset is not a valid LB proxy |
| EXP_T1_m | L1-loss density model, reconciled into q/v (w 0.25–1, speed share 0–1), D12_I405_S local | post-proc | best T1+T3 0.4383 = baseline 0.4383; stronger weights worse | 0 | – | PASS | clean | **REJECT** |
| EXP_T2_q | OOF probability cache (tfb-t2oof): same v2 features/folds, LGB main + LGB prior-free + CatBoost | infra | baseline (0.85309 recipe) pseudo 0.8732 / official 0.8765 | – | – | PASS | – | all post-hoc tests below use the same windows and folds |
| EXP_001/002 (proposed) | Onset: also train/predict T+5..T+25 | – | evidence: 39/40 official onset windows first queued at T+30 (1 has no observed queue), 96.9% of 2,128 pseudo onset windows; organizer: 120/120 | – | – | – | – | **REJECTED without running** (any earlier-step positive is a false positive) |
| EXP_T2_A1 | Onset blend sweep 1.0→0.3 (ongoing prior-free) | post-hoc | 0.80/0.20: pseudo 0.8739 / off 0.8786; 1.0: 0.8743 / 0.8709; ≤0.6 clearly worse (0.3: 0.8358) | ≈ +0.001 | – | PASS | – | within noise; confirms LB (more prior-free onset hurts) |
| EXP_T2_A2/A3 | Per-panel / per-TOD / weekday-weekend onset blend (nested CV) | post-hoc | pseudo 0.8749 / 0.8739 / 0.8739; official 0.8709 / 0.8709 / 0.8772 | ≈ 0 | – | PASS | nested | REJECT |
| **EXP_T2_p** | **Onset = CatBoost (all features, depth 8, 600 it)** | model | **pseudo 0.8770 / official 0.8876; onset IoU +0.004/+0.009/+0.014/+0.008 in 4/4 pseudo folds, +0.022 official** | **+0.004 / +0.011** | – | PASS | clean | **KEEP → tfb-t2v10** |
| EXP_T2_B4 | Ongoing = CatBoost or 50/50 with LGB prior-free | model | 0.8715–0.8727 / 0.8712–0.8742; prob. correlation 0.997 | − | – | PASS | – | REJECT (no diversity) |
| EXP_T2_D | Decoder: fixed thresholds 0.3–0.5, close 1-link gaps, drop isolated, per-step decode | post-hoc | all ≤ baseline; morphology −0.010 to −0.015 | − | – | PASS | – | REJECT (expected-IoU top-m stays) |

## Research
About 426 resources were opened and read across 9 research threads, and about 750 more were seen only as search results.
Numbers are the agents' own honest tallies. See the final report for the synthesis.

## Leaderboard submissions (public LB = validation month)
| Submission | Content | Public LB |
|---|---|---:|
| (earlier, user) | copy of the lamhuy8904 pipeline | 0.53880 |
| EXP_000 base | soukeaizenz notebook unchanged (submitted earlier by the user, ref 56616281) | 0.80939 (confirmed on LB) |
| PROBE_T2 | base + Task 2 only (EXP_T2_d) | 0.84357 |
| SUB_A | T1 v3 + T2 d + T4 | 0.84499 |
| FINAL v2 | final notebook `hosen42/tfb-final` v2: T2 d + T1 v4 profile-free + T4 | 0.84855 (rank 49/208 at submission time) |
| PROBE zero-onset / zero-ongoing | FINAL v2 with one condition's queue zeroed | 0.73711 / 0.73657 |
| **OG_NOPRIOR** | FINAL v2 + ongoing prior-free (tfb-swap-v5) | **0.85309** |
| ON30 | OG_NOPRIOR + onset 30/70 | 0.84917 |
