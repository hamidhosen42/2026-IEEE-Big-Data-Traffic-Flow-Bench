# Submitted notebooks (kept locally)

| Output file | Submission ref | Public LB | Description |
|---|---|---:|---|
| submission_E5.csv | 56817463 | 0.87961 | S5: EXP_COMBO_OSP: chain base + onset rows with the same-scenario prior (sp4: train prior blended per cell with the March/April masked-layer prior of the days before T, smoothed +-2 slots, a=n/(n+4)) |
| submission_E4.csv | 56817377 | 0.87857 | S4: EXP_COMBO_v22: chain base + ongoing rows from v22 fs4 (selection-faithful F03 bag: np/id x 2 seeds) |
| submission_E3.csv | 56814685 | 0.87817 | S3: EXP_COMBO_v21: chain base + ongoing rows from v21 fs2 (selection-faithful ongoing training: faithful windows 0.3 + run-starts 1.0) |
| submission_E2.csv | 56810594 | 0.87343 | EXP_COMBO_24: chain base + ongoing rows from v20 og8 bag (np511r+id511r x 4 seeds) |
| submission_FDRTV.csv | 56810522 | 0.87334 | EXP_COMBO_23: T1 = 0.5 FD-feature bag (3) + 0.5 old bag(8), then FD recon + TV |
| submission_TV.csv | 56787882 | 0.87271 | EXP_COMBO_22: chain base + 1-D TV smoothing of Task 1 density inside target runs |
| submission_COMBO19.csv | 56787752 | 0.85942 | EXP_COMBO_21: chain base + onset rows from v19 ABC_st03 (CatBoost bag + stage-2; CV-gated) |
| submission_COMBO18.csv | 56787709 | 0.87245 | EXP_COMBO_20: chain base + ongoing rows from v18 og4 bag (np511r+id511r x 2 seeds) |
| submission_FDR.csv | 56787648 | 0.87176 | EXP_COMBO_19: chain base + FD-gated reconciliation of Task 1 congested cells (v<0.6vf) |
| submission_COMBO16.csv | 56787531 | 0.87009 | EXP_COMBO_18: Task 1 bag of 8 FTA members (t1v11,t1v12,t1v12s2,t1v12s3,t1v12s4,t1v12s5,t1v12s6,t1v12s7) |
| submission_COMBO15.csv | 56764679 | 0.86944 | EXP_COMBO_17: COMBO13 + Task 1 bag of 4 FTA members (seeds 0-3) |
| submission_COMBO14.csv | 56762903 | 0.86692 | EXP_COMBO_16: COMBO13 + onset rows = 50/50 CatBoost / LGB 70-30 (v17, no ramp; pseudo-March +0.018 vs cat) |
| submission_COMBO13.csv | 56762852 | 0.86899 | EXP_COMBO_15: COMBO12 + ongoing rows from v15 np511r (prior-free 511 + ramp; probe vs id511r) |
| submission_COMBO12.csv | 56761353 | 0.86852 | EXP_COMBO_14: COMBO11 + Task 1 bag (mean of v11 FTA seed0 and v12 FTA seed1) |
| submission_COMBO11.csv | 56761216 | 0.86768 | EXP_COMBO_13: COMBO10 + Task 1 rows from T1 v11 FTA (transductive + ASM; holdout +0.0005 vs FT) |
| submission_COMBO10.csv | 56741839 | 0.86737 | EXP_COMBO_12: COMBO9 + ongoing rows from v15 (id511 + ramp flows T-3..T-1) |
| submission_COMBO9.csv | 56741513 | 0.86707 | EXP_COMBO_11: COMBO8 + Task 1 rows from T1 v9 FT (transductive target-month rows; holdout 0.9406 vs 0.9386) |
| submission_COMBO8.csv | 56738051 | 0.86537 | EXP_COMBO_10: COMBO7 + ongoing rows from v14 id511 (identity-kept prior-free; pseudo-March +0.0036) |
| submission_COMBO7.csv | 56732225 | 0.86509 | EXP_COMBO_9: COMBO6 + ongoing LGB 511 leaves/lr0.03/1800 trees (t2v13) |
| submission_COMBO6.csv | 56722713 | 0.86327 | EXP_COMBO_8: COMBO5 + T1 F (511 leaves, val-month holdout 0.9386 vs 0.9374) |
| submission_COMBO2D.csv | 56692851 | 0.85860 | EXP_COMBO_4: v9 (0.85639) + CatBoost onset + T1 extended low-rank (v6 D) |
| submission_COMBO5.csv | 56692821 | 0.86217 | EXP_COMBO_7: v9 + CatBoost onset + ongoing LGB 255 leaves + T1 v7 E |
| submission_COMBO4.csv | 56692820 | 0.86184 | EXP_COMBO_6: COMBO3 + T1 v7 E (bigger T1 model, val-month holdout 0.9374) |
| submission_COMBO2.csv | 56692819 | 0.85796 | EXP_COMBO_2: v9 (0.85639) + CatBoost onset (v9 feats) + T1 low-rank |
| submission_COMBO3.csv | 56692012 | 0.85969 | EXP_COMBO_5: v9 + CatBoost onset + big ongoing LGB (127 leaves) + T1 v6 D |
| submission_T1B_T2cat.csv | 56658171 | 0.85513 | EXP_COMBO_1: 0.85309 + T1 low-rank + T2 onset CatBoost |
| submission_og_pers30.csv | 56658135 | 0.85126 | EXP_T2_n: og_noprior + 30% last-valid persistence on ongoing |
| submission_main.csv | 56657293 | 0.85639 | EXP_T2_m: v9 (weekday-class priors + broad ongoing keep), og_noprior recipe |
| submission_T1B.csv | 56657287 | 0.85455 | EXP_T1_n: 0.85309 + Task 1 low-rank SVD features (T1 v5 B) |
| submission_main.csv | 56656952 | 0.85366 | EXP_T2_p: onset CatBoost + ongoing prior-free LGB (else = 0.85309 pipeline) |
| submission_on30.csv | 56645587 | 0.84917 | EXP_T2_k: og_noprior + onset 30/70 (more prior-free onset) |
| submission_og_noprior.csv | 56643037 | 0.85309 | EXP_T2_j: FINAL v2 + ongoing windows prior-free only (og_noprior) |
| submission_zero_ongoing.csv | 56627217 | 0.73657 | PROBE: FINAL v2 with ONGOING queue zeroed |
| submission_zero_onset.csv | 56627188 | 0.73711 | PROBE: FINAL v2 with ONSET queue zeroed |
| submission.csv | 56626886 | 0.84855 | FINAL v2: T2 LGBM blend70 + T1 LGBM profile-free (gap-aware, occ) + T4 NNLS own counts |
| submission.csv | 56620877 | 0.84499 | SUB_A: T1 LGBM v3 (gap-aware) + T2 LGBM blend70 + T4 NNLS own counts |
| submission.csv | 56619654 | 0.84357 | PROBE T2: EXP000 base + Task2 LGBM blend70 (only queue rows changed) |
| submission.csv | 56616281 | 0.80939 | Notebook [0.809 LB TOP 50] Open-Sourcing Physics Pipeline | Version 1 |
| submission_352472917.csv | 56615110 | error |  |
| submission.csv | 56518779 | 0.53880 | Notebook Traffic Flow Bench Pipeline | Version 2 |
| submission.csv | 56282724 | error |  |
| submission.csv | 56282297 | 0.53880 | Notebook Traffic Flow Bench Pipeline | Version 1 |
