# All submitted notebooks

One notebook per Kaggle submission, in submission order (71 submissions; best public LB #71, 0.88225).
Each file is the exact source of the kernel whose output was submitted. Swap kernels read upstream kernel outputs; their upstream sources are in `submissions/<folder>/upstream__*`. #1, #3 and #5 were submitted straight from Kaggle notebooks (current version).
Rebuild: `.venv/bin/python kaggle_jobs/build_all_notebooks.py`.

| # | Date (UTC) | Notebook | Kaggle kernel | Public LB | Description |
|---:|---|---|---|---:|---|
| 1 | 2026-09-16 | 01_traffic_flow_bench_pipeline__LB0.53880.ipynb | `traffic-flow-bench-pipeline` | 0.53880 | Notebook Traffic Flow Bench Pipeline / Version 1 |
| 2 | 2026-09-16 | — (no notebook: file upload that errored) | `` | error | Submission has 6985307 rows but 6980503 were expected. |
| 3 | 2026-09-24 | 03_traffic_flow_bench_pipeline__LB0.53880.ipynb | `traffic-flow-bench-pipeline` | 0.53880 | Notebook Traffic Flow Bench Pipeline / Version 2 |
| 4 | 2026-09-27 | — (no notebook: file upload that errored) | `` | error | Submission contains null values |
| 5 | 2026-09-27 | 05_0_809_lb_top_50_open_sourcing_physics_pi__LB0.80939.ipynb | `0-809-lb-top-50-open-sourcing-physics-pipeline` | 0.80939 | Notebook [0.809 LB TOP 50] Open-Sourcing Physics Pipeline / Version 1 |
| 6 | 2026-09-27 | 06_PROBE_T2__LB0.84357.ipynb | `tfb-probe-t2` | 0.84357 | PROBE T2: EXP000 base + Task2 LGBM blend70 (only queue rows changed) |
| 7 | 2026-09-27 | 07_SUB_A__LB0.84499.ipynb | `tfb-sub-a` | 0.84499 | SUB_A: T1 LGBM v3 (gap-aware) + T2 LGBM blend70 + T4 NNLS own counts |
| 8 | 2026-09-28 | 08_FINAL_v2__LB0.84855.ipynb | `tfb-final` | 0.84855 | FINAL v2: T2 LGBM blend70 + T1 LGBM profile-free (gap-aware, occ) + T4 NNLS own counts |
| 9 | 2026-09-28 | 09_PROBE_COND__LB0.73711.ipynb | `tfb-probe-cond` | 0.73711 | PROBE: FINAL v2 with ONSET queue zeroed |
| 10 | 2026-09-28 | 10_PROBE_COND__LB0.73657.ipynb | `tfb-probe-cond` | 0.73657 | PROBE: FINAL v2 with ONGOING queue zeroed |
| 11 | 2026-09-28 | 11_OG_NOPRIOR__LB0.85309.ipynb | `tfb-swap-v5` | 0.85309 | EXP_T2_j: FINAL v2 + ongoing windows prior-free only (og_noprior) |
| 12 | 2026-09-28 | 12_ON30__LB0.84917.ipynb | `tfb-swap-v7` | 0.84917 | EXP_T2_k: og_noprior + onset 30/70 (more prior-free onset) |
| 13 | 2026-09-29 | 13_T2_CATBOOST_ONSET__LB0.85366.ipynb | `tfb-swap-v10` | 0.85366 | EXP_T2_p: onset CatBoost + ongoing prior-free LGB (else = 0.85309 pipeline) |
| 14 | 2026-09-29 | 14_T1B_LOWRANK__LB0.85455.ipynb | `tfb-swap-t1b` | 0.85455 | EXP_T1_n: 0.85309 + Task 1 low-rank SVD features (T1 v5 B) |
| 15 | 2026-09-29 | 15_V9_WEEKDAY_PRIOR__LB0.85639.ipynb | `tfb-swap-v9` | 0.85639 | EXP_T2_m: v9 (weekday-class priors + broad ongoing keep), og_noprior recipe |
| 16 | 2026-09-29 | 16_OG_PERS30__LB0.85126.ipynb | `tfb-swap-v8` | 0.85126 | EXP_T2_n: og_noprior + 30% last-valid persistence on ongoing |
| 17 | 2026-09-29 | 17_T1B_T2CAT__LB0.85513.ipynb | `tfb-swap-t1b` | 0.85513 | EXP_COMBO_1: 0.85309 + T1 low-rank + T2 onset CatBoost |
| 18 | 2026-09-30 | 18_COMBO3__LB0.85969.ipynb | `tfb-combo3` | 0.85969 | EXP_COMBO_5: v9 + CatBoost onset + big ongoing LGB (127 leaves) + T1 v6 D |
| 19 | 2026-09-30 | 19_COMBO2__LB0.85796.ipynb | `tfb-combo2` | 0.85796 | EXP_COMBO_2: v9 (0.85639) + CatBoost onset (v9 feats) + T1 low-rank |
| 20 | 2026-09-30 | 20_COMBO4__LB0.86184.ipynb | `tfb-combo4` | 0.86184 | EXP_COMBO_6: COMBO3 + T1 v7 E (bigger T1 model, val-month holdout 0.9374) |
| 21 | 2026-09-30 | 21_COMBO5__LB0.86217.ipynb | `tfb-combo5` | 0.86217 | EXP_COMBO_7: v9 + CatBoost onset + ongoing LGB 255 leaves + T1 v7 E |
| 22 | 2026-09-30 | 22_COMBO2D__LB0.85860.ipynb | `tfb-combo2d` | 0.85860 | EXP_COMBO_4: v9 (0.85639) + CatBoost onset + T1 extended low-rank (v6 D) |
| 23 | 2026-10-01 | 23_COMBO6__LB0.86327.ipynb | `tfb-combo6` | 0.86327 | EXP_COMBO_8: COMBO5 + T1 F (511 leaves, val-month holdout 0.9386 vs 0.9374) |
| 24 | 2026-10-01 | 24_COMBO7__LB0.86509.ipynb | `tfb-combo7` | 0.86509 | EXP_COMBO_9: COMBO6 + ongoing LGB 511 leaves/lr0.03/1800 trees (t2v13) |
| 25 | 2026-10-01 | 25_COMBO8__LB0.86537.ipynb | `tfb-combo8` | 0.86537 | EXP_COMBO_10: COMBO7 + ongoing rows from v14 id511 (identity-kept prior-free; pseudo-March +0.0036) |
| 26 | 2026-10-01 | 26_COMBO9__LB0.86707.ipynb | `tfb-combo9` | 0.86707 | EXP_COMBO_11: COMBO8 + Task 1 rows from T1 v9 FT (transductive target-month rows; holdout 0.9406 vs 0.9386) |
| 27 | 2026-10-01 | 27_COMBO10__LB0.86737.ipynb | `tfb-combo10` | 0.86737 | EXP_COMBO_12: COMBO9 + ongoing rows from v15 (id511 + ramp flows T-3..T-1) |
| 28 | 2026-10-02 | 28_COMBO11__LB0.86768.ipynb | `tfb-combo11` | 0.86768 | EXP_COMBO_13: COMBO10 + Task 1 rows from T1 v11 FTA (transductive + ASM; holdout +0.0005 vs FT) |
| 29 | 2026-10-02 | 29_COMBO12__LB0.86852.ipynb | `tfb-combo12` | 0.86852 | EXP_COMBO_14: COMBO11 + Task 1 bag (mean of v11 FTA seed0 and v12 FTA seed1) |
| 30 | 2026-10-02 | 30_COMBO13__LB0.86899.ipynb | `tfb-combo13` | 0.86899 | EXP_COMBO_15: COMBO12 + ongoing rows from v15 np511r (prior-free 511 + ramp; probe vs id511r) |
| 31 | 2026-10-02 | 31_COMBO14__LB0.86692.ipynb | `tfb-combo14` | 0.86692 | EXP_COMBO_16: COMBO13 + onset rows = 50/50 CatBoost / LGB 70-30 (v17, no ramp; pseudo-March +0.018 vs cat) |
| 32 | 2026-10-02 | 32_COMBO15__LB0.86944.ipynb | `tfb-combo15` | 0.86944 | EXP_COMBO_17: COMBO13 + Task 1 bag of 4 FTA members (seeds 0-3) |
| 33 | 2026-10-03 | 33_COMBO16__LB0.87009.ipynb | `tfb-c16` | 0.87009 | EXP_COMBO_18: Task 1 bag of 8 FTA members (t1v11,t1v12,t1v12s2,t1v12s3,t1v12s4,t1v12s5,t1v12s6,t1v12s7) |
| 34 | 2026-10-03 | 34_COMBO17_FDR__LB0.87176.ipynb | `tfb-c17` | 0.87176 | EXP_COMBO_19: chain base + FD-gated reconciliation of Task 1 congested cells (v<0.6vf) |
| 35 | 2026-10-03 | 35_COMBO18_OG4__LB0.87245.ipynb | `tfb-c18` | 0.87245 | EXP_COMBO_20: chain base + ongoing rows from v18 og4 bag (np511r+id511r x 2 seeds) |
| 36 | 2026-10-03 | 36_COMBO19_ON19__LB0.85942.ipynb | `tfb-c19` | 0.85942 | EXP_COMBO_21: chain base + onset rows from v19 ABC_st03 (CatBoost bag + stage-2; CV-gated) |
| 37 | 2026-10-03 | 37_COMBO20_TV__LB0.87271.ipynb | `tfb-c20` | 0.87271 | EXP_COMBO_22: chain base + 1-D TV smoothing of Task 1 density inside target runs |
| 38 | 2026-10-04 | 38_E1_T1MIX__LB0.87334.ipynb | `tfb-e1` | 0.87334 | EXP_COMBO_23: T1 = 0.5 FD-feature bag (3) + 0.5 old bag(8), then FD recon + TV |
| 39 | 2026-10-04 | 39_E2_OG8__LB0.87343.ipynb | `tfb-e2` | 0.87343 | EXP_COMBO_24: chain base + ongoing rows from v20 og8 bag (np511r+id511r x 4 seeds) |
| 40 | 2026-10-04 | 40_E3__LB0.87817.ipynb | `tfb-e3` | 0.87817 | S3: EXP_COMBO_v21: chain base + ongoing rows from v21 fs2 (selection-faithful ongoing training: faithful windows 0.3 + run-starts 1.0) |
| 41 | 2026-10-04 | 41_E4__LB0.87857.ipynb | `tfb-e4` | 0.87857 | S4: EXP_COMBO_v22: chain base + ongoing rows from v22 fs4 (selection-faithful F03 bag: np/id x 2 seeds) |
| 42 | 2026-10-04 | 42_E5__LB0.87961.ipynb | `tfb-e5` | 0.87961 | S5: EXP_COMBO_OSP: chain base + onset rows with the same-scenario prior (sp4: train prior blended per cell with the March/April masked-layer |
| 43 | 2026-10-05 | 43_F1__LB0.87963.ipynb | `tfb-f1` | 0.87963 | P1: T1 = 0.5 FD-feature bag (4 members) + 0.5 old bag(8), recon + TV |
| 44 | 2026-10-05 | 44_F2__LB0.87465.ipynb | `tfb-f2` | 0.87465 | P2: March onset rows from the LODO CatBoost bag (training rows carry day-excluded priors: leave-one-day-out x3 + leave-week-group-out x3), s |
| 45 | 2026-10-05 | 45_F3__LB0.87817.ipynb | `tfb-f3` | 0.87817 | P3: March onset rows: production CatBoost + early-month mixture prior mix9 (P2 LODO not kept) (per-cell two-hypothesis prior from the 1-4 Ma |
| 46 | 2026-10-05 | 46_F4__LB0.87939.ipynb | `tfb-f4` | 0.87939 | P4: March ongoing rows of the spacing-blocked (g72) windows from an F10 pair (np+id 511, all faithful windows 1.0); other windows and April  |
| 47 | 2026-10-05 | 47_F5__LB0.88051.ipynb | `tfb-f5` | 0.88051 | P5: March ongoing rows from the ALL-FINITE F03 bag (ineligible-but-valued cells as features); April unchanged |
| 48 | 2026-10-06 | 48_G1__LB0.87990.ipynb | `tfb-g1` | 0.87990 | Q1: T1 blackout rows (90 min after each official Task 2 origin) = 0.5 T1 mix + 0.5 member trained with simulated-blackout transductive rows; |
| 49 | 2026-10-06 | 49_G2__LB0.87883.ipynb | `tfb-g2` | 0.87883 | Q2: March onset rows from CatBoost onset with ALL-FINITE inputs (ineligible-but-valued cells as features), same sp4 scenario prior; April un |
| 50 | 2026-10-06 | 50_G3__LB0.88092.ipynb | `tfb-g3` | 0.88092 | Q3: March ongoing rows from the F03 bag with all-finite inputs + 13 OCCUPANCY features (occupancy / per-link critical occupancy, neighbours, |
| 51 | 2026-10-06 | 51_G4__LB0.88092.ipynb | `tfb-g4` | 0.88092 | Q4: FINAL CANDIDATE A (same-scenario bet): April onset = CatBoost bag trained with released-March windows + scenario prior; April ongoing =  |
| 52 | 2026-10-06 | 52_G5__LB0.88092.ipynb | `tfb-g5` | 0.88092 | Q5: FINAL CANDIDATE B (hedge): April onset = production CatBoost with the train prior; April ongoing = all-finite F03 bag (train-only models |
| 53 | 2026-10-07 | 53_H1__LB0.88129.ipynb | `tfb-h1` | 0.88129 | Q1: T1 blackout rows (90 min after each official Task 2 origin) = 0.25 T1 mix + 0.75 member with simulated-blackout rows + 9 RAMP features ( |
| 54 | 2026-10-07 | 54_H2__LB0.88120.ipynb | `tfb-h2` | 0.88120 | Q2: T1 blackout rows (90 min after each official Task 2 origin) = 0.5 T1 mix + 0.5 member with simulated-blackout rows + 9 RAMP features (ra |
| 55 | 2026-10-07 | 55_H3__LB0.88129.ipynb | `tfb-h3` | 0.88129 | Q3: FINAL CANDIDATE B (hedge) on the ramp-member base: April onset = production CatBoost with the train prior; April ongoing = all-finite F0 |
| 56 | 2026-10-07 | 56_H4__LB0.88008.ipynb | `tfb-h4` | 0.88008 | Q4: March ongoing rows from the F03 bag (all-finite + occupancy) with 103 extra ONGOING features from the 60-min history and slot T (corrido |
| 57 | 2026-10-07 | 57_H5__LB0.88145.ipynb | `tfb-h5` | 0.88145 | Q5: T1 FD bag on ALL rows + the blackout member and the ramp member (5 FD members: v13, s10, s11, t1dark, t1ramp; W_FD 0.5); blackout rows 0 |
| 58 | 2026-10-08 | 58_I1__LB0.88145.ipynb | `tfb-i1` | 0.88145 | Q1: FINAL CANDIDATE B (hedge) on tfb-h5 (5-member FD bag + ramp blackout): April onset = production CatBoost with the train prior; April ong |
| 59 | 2026-10-08 | 59_I2__LB0.88156.ipynb | `tfb-i2` | 0.88156 | Q2: T1: second ramp member (seed 17): blackout rows 0.25 mix + 0.75 mean of the two ramp members; FD bag of 6 on all rows (v13, s10, s11, t1 |
| 60 | 2026-10-08 | 60_I3__LB0.88156.ipynb | `tfb-i3` | 0.88156 | Q3: FINAL CANDIDATE B (hedge) on tfb-i2 (two ramp members): April onset = production CatBoost with the train prior; April ongoing = all-fini |
| 61 | 2026-10-08 | 61_I4__LB0.88176.ipynb | `tfb-i4` | 0.88176 | Q4: T1 blackout rows = 0.25 mix + 0.75 BLACKOUT SPECIALIST (trained on official-like blackout placement: onset queues start at T+30, ongoing |
| 62 | 2026-10-08 | 62_I5__LB0.88176.ipynb | `tfb-i5` | 0.88176 | Q5: FINAL CANDIDATE B (hedge) on tfb-i4 (blackout specialist): April onset = production CatBoost with the train prior; April ongoing = all-f |
| 63 | 2026-10-09 | 63_J1__LB0.88179.ipynb | `tfb-j1` | 0.88179 | Q1: T1 FD bag on ALL rows + the blackout specialist (7 FD members: v13, s10, s11, t1dark, t1ramp, t1ramp2, t1bspec; W_FD 0.5); blackout rows |
| 64 | 2026-10-09 | 64_J3__LB0.88176.ipynb | `tfb-j3` | 0.88176 | Q3: FINAL CANDIDATE B2 (hedge) on tfb-i4: April onset = production CatBoost with the train prior; April ongoing = ogfeat F03 bag with a grow |
| 65 | 2026-10-09 | 65_J4__LB0.88176.ipynb | `tfb-j4` | 0.88176 | Q4: FINAL CANDIDATE A2 (same-scenario bet) on tfb-i4: April ongoing = March-augmented F03 np/id/full bag + scenario prior + 103 ogfeat colum |
| 66 | 2026-10-10 | 66_K1__LB0.88198.ipynb | `tfb-k1` | 0.88198 | K1: T1 mix + CatBoost FD member (depth 8+10, W 0.35; lab J +0.00071 8/8) on tfb-j4 (A2); blackout specialist, recon + TV unchanged |
| 67 | 2026-10-10 | 67_K2__LB0.88198.ipynb | `tfb-k2` | 0.88198 | K2: FINAL CANDIDATE B (hedge) on tfb-k1 (CatBoost T1): April onset = production CatBoost with the train prior; April ongoing = veto10 (ogfea |
| 68 | 2026-10-16 | 68_K3__LB0.88201.ipynb | `tfb-k3` | 0.88201 | K3: as K1 but the T1 CatBoost member is the mean of two seeds (tfb-t1catb + tfb-t1catb2), W 0.35; on tfb-j4 (A2); blackout specialist, recon |
| 69 | 2026-10-16 | 69_K4__LB0.88201.ipynb | `tfb-k4` | 0.88201 | K4: FINAL CANDIDATE B (hedge) on tfb-k3 (2 CatBoost seeds): April onset = production CatBoost with the train prior; April ongoing = veto10;  |
| 70 | 2026-10-16 | 70_K5__LB0.88225.ipynb | `tfb-k5` | 0.88225 | K5: tfb-k3 + T1 XGBoost FD member (W 0.35; lab J +0.00056 8/8, control-adjusted +0.00061); blackout specialist, recon + TV unchanged |
| 71 | 2026-10-16 | 71_K6__LB0.88225.ipynb | `tfb-k6` | 0.88225 | K6: FINAL CANDIDATE B (hedge) on tfb-k5 (XGBoost T1): April onset = production CatBoost with the train prior; April ongoing = veto10; March  |
| 72 | 2026-10-29 | 72_K7__LB0.88259.ipynb | `tfb-k7` | 0.88259 | K7: tfb-k5 + flow-only low-rank tensor completion member (W 0.35, ordinary rows); recon + TV |
| 73 | 2026-10-29 | 73_K8__LB0.88259.ipynb | `tfb-k8` | 0.88259 | K8: FINAL CANDIDATE B (hedge) on tfb-k7: April onset = train-prior CatBoost; April ongoing = veto10; March rows unchanged |
