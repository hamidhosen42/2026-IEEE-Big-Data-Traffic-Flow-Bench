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
| **EXP_T2_p** | **Onset = CatBoost (all features, depth 8, 600 it)** | model | **pseudo 0.8770 / official 0.8876; onset IoU +0.004/+0.009/+0.014/+0.008 in 4/4 pseudo folds, +0.022 official** | **LB 0.85366 (+0.0006)** | – | PASS | clean | **KEEP** |
| EXP_T2_B4 | Ongoing = CatBoost or 50/50 with LGB prior-free | model | 0.8715–0.8727 / 0.8712–0.8742; prob. correlation 0.997 | − | – | PASS | – | REJECT (no diversity) |
| EXP_T2_D | Decoder: fixed thresholds 0.3–0.5, close 1-link gaps, drop isolated, per-step decode | post-hoc | all ≤ baseline; morphology −0.010 to −0.015 | − | – | PASS | – | REJECT (expected-IoU top-m stays) |
| EXP_T2_H | Onset/ongoing hazard + gradient features (speed deltas/acceleration/std, density over critical, neighbour gradients, upstream demand) — OOF cache tfb-t2oofH | features | CatBoost-onset recipe: pseudo 0.8780 vs 0.8770, official 0.8965 vs 0.8876; onset folds +0.004/−0.004/+0.000/−0.004; ongoing +0.001 in 3/4 folds | ≈ 0 | – | PASS | history only | REJECT (inside noise) |
| EXP_016 (T1 C) | Task 1: separate LightGBM per mask regime R1/R2/R3 | model | val-month holdout 0.9285 vs A 0.9307 (worse on every panel) | −0.002 | – | PASS | – | REJECT |
| **EXP_017 (T1 B)** | **Task 1: + per-day rank-3 SVD reconstruction of the interpolated speed/flow fields as features** | features | **val-month holdout 0.9343 vs A 0.9307 (B ≥ A on every panel shown)** | **LB 0.85455 (+0.0015, predicted +0.0013)** | – | PASS | holdout = hidden released cells | **KEEP** |
| EXP_T2_m | v9: weekday/weekend-class priors + broad ongoing keep (coverage 97.6–99.8%) | features | pseudo 0.8802 / official 0.8831 (v2 0.8791 / 0.8755) | **LB 0.85639 (+0.0033)** | – | PASS | fold-safe priors | **KEEP (best)** |
| EXP_T2_n | Ongoing: 0.7 prior-free + 0.3 last-valid persistence | ensemble | – | LB 0.85126 (−0.0018) | – | PASS | – | REJECT |
| EXP_COMBO_2 | v9 T2 + CatBoost onset on v9 features (tfb-t2v11) + T1 low-rank | combo | additive estimate ≈ 0.8585 | scheduled 2026-09-30 00:02 UTC | – | PASS | control 100% | – |
| EXP_T2_oprior | Onset-start prior (share of train onset windows whose T+30 truth contains the link, ±60 min, weekday class; parity split so no row sees its own truth) | feature | onset −0.003..−0.016 in 4/4 pseudo folds (official +0.013 on 40 windows) | − | – | PASS | parity split | REJECT |
| EXP_T1_o (T1 D) | + rank-5 low-rank, occupancy low-rank, cross-day low-rank per station | features | val-month holdout 0.9350 vs B 0.9343 (9/10 panels better) | +0.0007 | – | PASS | – | KEEP (small) |
| EXP_T2_big | Ongoing prior-free LGB: 127 leaves / min_data 50 / 800 trees (vs 63 / 100 / 500); 3-seed averaging | model | pseudo ongoing 0.9201 vs 0.9128 (+0.008 in 4/4 folds), official 0.8456 vs 0.8354; seed averaging 0.9125 (no gain) | +0.007 ongoing | – | PASS | – | KEEP big; REJECT seeds |
| EXP_topology | Link order: order_index vs milepost (discussion 743819) | diagnostic | identical in all 8 Task 2 panels (Spearman 1.000) | – | – | – | – | no change needed |
| EXP_COMBO_4 / 5 | COMBO2D = v9 + CatBoost onset + T1 D; COMBO3 = + big ongoing | combo | additive estimates ≈ 0.8588 / ≈ 0.8597 | scheduled 00:40 / 01:00 UTC 2026-09-30 | – | PASS | controls 100% | – |
| EXP_T1_E | Task 1 bigger LightGBM: 255 leaves, lr 0.05, 900 trees, 2× training rows, D features | model | val-month holdout **0.9374** vs D 0.9350 | +0.0024 | – | PASS | – | **KEEP** |
| EXP_T2_cap | Ongoing capacity: 255 leaves/1000 trees; 255/lr 0.025/2000 trees | model | pseudo ongoing 0.9235 / 0.9243 vs 127-leaf 0.9201 (4/4 folds +); official 0.8513 / 0.8486 | +0.003..0.004 | – | PASS | – | **KEEP 255/1000** |
| EXP_T2_oncat | Onset CatBoost variants: depth 10, 1500 iters, LGB big, 3-model average | model | pseudo 0.8865 / 0.8879 / 0.8840 / 0.8877 vs cat8 0.8863 (mixed folds) | ≈ 0 | – | PASS | – | keep cat8 |
| EXP_COMBO_6 / 7 | COMBO4 = COMBO3 + T1 E; COMBO5 = + ongoing 255 leaves | combo | additive estimates ≈ 0.8605 / ≈ 0.861 | scheduled 00:30 / 00:35 UTC 2026-09-30 | – | PASS | controls 100% | – |
| EXP_T1_F | Task 1 LightGBM 511 leaves, 4× rows, 1500 trees (tfb-t1v8-full) | model | val-month holdout **0.9386** vs E 0.9374 (7/8 panels better, D12_I405_S −0.0001) | +0.0012 | – | PASS | – | **KEEP**; COMBO6 **LB 0.86327 (+0.0011)**, new best |
| EXP_T2_511 | Ongoing prior-free LGB 511 leaves / lr 0.03 / 1800 trees, 6000 windows; control 255 at half data (tfb-oofY) | model | pseudo ongoing 0.9017 vs 255 0.8958 (4/4 folds +), official 0.8626 vs 0.8551; 255 half-data 0.8961 (data volume ≈ 0) | +0.006 | – | PASS (config change of v12) | – | **KEEP**; COMBO7 (COMBO6 + t2v13 queue rows) **LB 0.86509 (+0.0018)**, new best. Caveat: v13 retrained the CatBoost onset too, and onset-row identity with v12 was not checked |
| EXP_T2_calib | Calibration before expected-IoU decode on oofX cache (onset cat8, ongoing 255): power 0.8/0.9/1.1/1.25; nested isotonic per condition (research D A1) | post-proc | pseudo 0.8831 → 0.8809/0.8826/0.8845/0.8842, isotonic 0.8808; official 0.8904 → 0.8915/0.8900/0.8898/0.8885, isotonic 0.8924 (signs disagree) | ≈0 | – | – | nested | REJECT (already calibrated) |
| EXP_T1_trans (v9) | Task 1 transductive rows: 2 rounds × 20% of released val/private cells hidden, released value as target; val holdout disjoint (research A I-1, D A3) | data | val-month holdout **FT 0.9406 vs F 0.9386 (+0.0020, 10/10 panels)** | +0.0020 | – | PASS | holdout disjoint; offline task | **KEEP**; COMBO9 **LB 0.86707 (+0.0017)**, new best |
| EXP_T1_asm (v10) | ASM/GASM wave-kernel features (±80, ±18 km/h; ±5 slots × ±5 stations), F model | features | val-month holdout FA **0.9391** vs F 0.9386 (+0.0005; 9/10 panels +, D7_I210_W −0.0002) | +0.0005 | – | PASS | – | MARGINAL (below the +0.0007 rule); consider combining with FT |
| EXP_T2_id | Ongoing variants: identity-kept prior-free 511, full 511, 50/50 blends (research A I-2; LB-only decision since the gain is month-shift) | model | – | – | – | config of v13 | – | tfb-t2v14 running |
| EXP_topo_audit | Topology audit, all 10 panels (tfb-topo-audit): order_index vs milepost, travel direction, adjacency, ramps, spacing | diagnostic | order_index = milepost rank (Spearman 1.000); 1 station/link; **travel direction: E/N panels increasing index, W/S panels decreasing** | – | – | – | – | Task 1 OK; **Task 2 pooled features direction-inconsistent** → EXP_021 |
| EXP_021 (T2 flip) | Reverse the link order in W/S panels so index+1 = downstream everywhere (t2_oofF vs oofX control, 4,269 window-matched windows) | structural | pseudo 0.8799 vs 0.8831 (−0.0032; folds −0.0035/−0.0031/+0.0024/−0.0048); ongoing +0.0006, onset −0.0028 (W/S onset −0.007); official +0.0037 (80 windows) | −0.003 | – | PASS | – | **DROP**: the pooled model already copes; symmetric up/dn features |
| EXP_LWR_probe | Local D12_I405_S (60 days) Task 3 LWR proxy sensitivity | diagnostic | interp 0.523; truth+1 km/h 0.89; truth+1% flow 0.89; smoothed-truth oracle (uses own truth) 0.65; S_FD truth 0.991 | – | – | – | – | Task 3 pinned by observation noise |
| EXP_LWR_model | Local residual-HGB model + LWR-aware post-processing (density blend / run smoothing / v-q smoothing) and same-instant high-pass neighbour features, 40/20-day split | post-proc / features | model S_state 0.936 (interp 0.924) but S_LWR 0.520 (interp 0.523); best post-proc +0.0005 score; HP features S_LWR 0.516 | ≈0 | – | – | – | DROP physics post-proc; DROP HP noise features |
| EXP_onset_noise | Label noise at onset slots (local, 193 onset events) | diagnostic | onset cells median 26 km/h below cut; 0.2% within ±2 km/h of cut | – | – | – | – | label-noise explanation of the CV→LB gap rejected |
| EXP_024 (pseudo-March gate) | Production T2 models scored on pseudo-windows selected from the released validation month (history masked cells filled with T1 F, labels = observed eligible cells); smoke = 2 panels, small models | validation | smoke: 91 ongoing + 51 onset windows; all 3 LB sign checks reproduced (og np−full +0.078; pers blend −0.0045; onset 30/70 −0.109); cat onset +0.032; og 255 vs 63 +0.021; 511 vs 255 −0.003 | – | – | PASS | gate for private-file decisions only | full run (8 panels, 419 ongoing + 215 onset windows): og full63 0.733 / np63 0.813 / np255 0.825 / np511 0.826; onset 70/30 0.744, 30/70 0.702, cat 0.719, full 0.700. Checks: np−full +0.080 ✓, 30/70 −0.042 ✓, pers blend +0.005 ✗ (LB −0.012 IoU on 40 windows), cat −0.025 ✗ (LB +0.004 IoU). Absolute scores close to the LB estimates (ongoing ≈0.80, onset ≈0.76) | **gate reliable for large effects; small ones disagree (either side may be noise)** |
| EXP_030+031 (oofR) | Released ramp flows (t−3..t−1 only) on attached links and ±3 links + prior × instability interactions; P255 ongoing (prior-free: ramp feats kept) + CatBoost onset; window-matched vs oofX | features | pseudo +0.0026 (folds −0.0041/+0.0068/+0.0038/−0.0023); **ongoing +0.0008/+0.0021/+0.0023/+0.0028 (4/4)**, official ongoing +0.004; onset −0.0024/−0.001/+0.0061/−0.002 | ongoing +0.002 | – | PASS | ≤T only | **KEEP ramp feats for ongoing** (small); DROP onset interactions |
| EXP_COMBO_12 (COMBO10) | COMBO9 + ongoing rows from t2v15 (id511 + released ramp flows T−3..T−1) | LB probe | 0.34% of ongoing rows changed; onset untouched | **LB 0.86737 (+0.0003)**, new best | – | PASS | ≤T only | **KEEP** |
| EXP_T1_FTA (v11a/b) | Transductive + ASM features (FTA) vs transductive (FT), panels split over 2 kernels | features | holdout FT 0.9361/0.9413 → FTA 0.9365/0.9418 (halves); 9/10 panels +, 1 tie | +0.0005 | – | PASS | – | KEEP; COMBO11 **LB 0.86768 (+0.0003)** |
| EXP_T1_bag (v12a/b) | Second FTA member: seed 1, different transductive hidden rounds and row subsample; averaged with v11 FTA | ensemble | v12b member 0.9419 vs v11b 0.9418 (equal members); bag effect not measurable on the holdout | – | – | PASS | – | v12a 0.9364 vs v11a 0.9365; COMBO12 (mean of 2 members) **LB 0.86852 (+0.0008)**, new best → seeds 2 and 3 running (4-member bag) |
| EXP_v16_gate | Ramp features on the pseudo-March gate (same models as production v15/v16) | validation | **onset with ramp: LGB 70/30 0.671 (no ramp 0.744), cat 0.676 (0.719)** → ramp features hurt onset on March; ongoing np511r 0.832 vs id511r 0.823 (no ramp: id 0.830 vs np 0.826: order flips → ±0.01 is gate noise); onset mix +0.0135 vs cat (with ramp) | – | – | – | – | never use ramp feats for onset; v17 = onset variants without ramp |
| EXP_og_dilation | Post-hoc ongoing spatial dilation of probabilities (γ 0.5–0.9 from step h0) on gate probs | post-proc | 0.6456–0.7794 vs 0.8226 | −0.04…−0.18 | – | – | – | DROP |
| EXP_v17_gate | Gate without ramp (v13 feature set): onset 70/30 0.7439, mix 0.7374, cat 0.7192; ongoing id511 0.8296, np511 0.8263 (identical to the earlier gate: deterministic) | validation | – | – | – | – | – | – |
| EXP_COMBO_15 (COMBO13) | COMBO12 + ongoing rows from v15 np511r (prior-free 511 + ramp), 0.32% of ongoing rows changed | LB probe | – | **LB 0.86899 (+0.00047)**, new best | – | PASS | – | **KEEP** |
| EXP_COMBO_16 (COMBO14) | COMBO13 + onset rows = 50/50 CatBoost / LGB 70-30 (v17, no ramp); gate +0.018 vs cat | LB probe | onset positives 308 → 315 | **LB 0.86692 (−0.0021)** | – | PASS | – | **DROP**. The LB favours CatBoost onset twice → **the gate is NOT reliable for onset decisions** (observed-label truth ≠ noise-free truth) |
| EXP_dens_blend | Task 3: at T1 target cells k ← 0.75·k_pred + 0.25·k_interp (temporal interpolation of the released density, short gaps ≤3 slots only), speed kept, flow = k·v (local prototype: blend 0.25 best of 10 variants, +0.0005 score; gated versions ≤ +0.0004) | post-proc | Kaggle test run: coverage 1.0, median Δk/k 0.43% | – | – | PASS | offline Task 1 | DROPPED before the LB: replica with the real bag predictions is negative (S_state −0.0020..−0.0028); replaced by FD-gated recon |
| EXP_T1_bag_more | T1 bag members seed 2/3 (2 rounds) and seed 4 (3 transductive rounds, cap 6× rows) | ensemble | – | – | – | – | – | COMBO15 = COMBO13 + bag of 4 (seeds 0–3): **LB 0.86944 (+0.00045)**, new best; seed 4 (3 transductive rounds, cap 6×) second half: holdout **0.9424** vs 2-round members 0.9418/0.9419 (+0.0005 per member) → more rounds help; seeds 5, 6 also use 3 rounds |
| EXP_og_bag (v18) | Ongoing bag: np511r + id511r × seeds 0/1 (og4), npid, np2 variants | ensemble | – | – | – | – | – | v18 complete (174,000 rows, 0 dupes); og4 input ready for slot 3 |
| WF_2026-10-02 | Multi-agent research and plan workflow (69 agents): propose → 3-lens adversarial verify → plan → implement | process | Frontier: 0.92406 is stale (leaderboard CSV header = LastSubmissionDate; Pardheev last submission 09-12, 12 h after the reopen; 0.924 would need March queue IoU 0.93–0.97). Density blend local replica with the real bag predictions is NEGATIVE (S_state −0.002) → replaced by FD-gated recon (proxy +0.0013) and TV (proxy +0.0005). New kernels: swap_fdrecon.py, t2_onset_v19.py (onset bag + stage-2), t1_pipeline_v13a/b (FD-implied features, seed changed to 9 to avoid a collision with s7) | – | – | – | – | plan in research/PLAN_2026-10-03.md |
| EXP_FDR_diag (tfb-fdrecon-smoke v2) | FD-gated reconciliation diagnostic on COMBO15 (no submission). 10 links with implausible FD parameters (w > 40 km/h, e.g. per-lane capacity 7,668) excluded link-wise | diagnostic | released congested cells: k_obs vs k_FD(v_obs) median −0.0014..+0.0019, IQR about ±2% in all 10 panels → no panel excluded; FDR changes 3.98% of state rows (gate share 4%), median |dv| 0.18 km/h, |dq| 11 vph | – | – | PASS | offline T1 | slot 2 on 2026-10-03 (FDR on the slot-1 winner) |
| AUTO_CHAIN_2026-10-03 | kaggle_jobs/auto_chain_oct3.py (nohup) runs 5 chained one-factor submissions from 00:02 UTC: S1 T1 bag-N, S2 FDR, S3 og4, S4 onset v19 (gated) or T1-FD probe (gated), S5 TV; KEEP rules encoded; state in auto_chain_oct3_state.json, log in auto_chain_oct3.log | process | – | – | – | – | – | scheduled |
| EXP_023 on gate (pmarch2) | v14 ongoing variants scored on pseudo-March (419 ongoing windows) | validation | vs np511: **id511 +0.0036**, np+id +0.0029, full511 −0.079, bnf −0.006, bif −0.007 (Hisernberg's blend recipe hurts here) | – | – | – | – | id511 → LB probe |
| EXP_COMBO_10 (COMBO8) | COMBO7 + ongoing rows from v14 id511 only (control: v14 np511 vs v13 0.99951; onset rows identical) | LB probe | 0.35% of ongoing rows changed | **LB 0.86537 (+0.00028)**, new best | – | PASS | – | **KEEP** (same sign as the gate) |

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
| T2 CatBoost onset (v10) | 0.85309 + onset CatBoost | 0.85366 |
| **V9 weekday prior** | 0.85309 recipe on v9 models (weekday/weekend priors + broad ongoing keep) | **0.85639** (best) |
| OG_PERS30 | + 30% persistence on ongoing | 0.85126 (reject) |
| T1B | 0.85309 + Task 1 low-rank features | 0.85455 |
| T1B + T2cat | both | 0.85513 (= sum of single-factor gains: effects are additive) |
