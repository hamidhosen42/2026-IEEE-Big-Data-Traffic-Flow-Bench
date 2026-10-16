# Submitted notebooks (kept locally)

| Output file | Submission ref | Public LB | Description |
|---|---|---:|---|
| submission_K4.csv | 57263111 | 0.88201 | K4: FINAL CANDIDATE B (hedge) on tfb-k3 (2 CatBoost seeds): April onset = production CatBoost with the train prior; April ongoing = veto10; March rows unchanged |
| submission_FDRTV.csv | 57263000 | 0.88201 | K3: as K1 but the T1 CatBoost member is the mean of two seeds (tfb-t1catb + tfb-t1catb2), W 0.35; on tfb-j4 (A2); blackout specialist, recon + TV unchanged |
| submission_K2.csv | 57049844 | 0.88198 | K2: FINAL CANDIDATE B (hedge) on tfb-k1 (CatBoost T1): April onset = production CatBoost with the train prior; April ongoing = veto10 (ogfeat bag with growth veto); March rows unchanged |
| submission_FDRTV.csv | 57049756 | 0.88198 | K1: T1 mix + CatBoost FD member (depth 8+10, W 0.35; lab J +0.00071 8/8) on tfb-j4 (A2); blackout specialist, recon + TV unchanged |
| submission_J4.csv | 57004474 | 0.88176 | Q4: FINAL CANDIDATE A2 (same-scenario bet) on tfb-i4: April ongoing = March-augmented F03 np/id/full bag + scenario prior + 103 ogfeat columns (pvogfv; many-day pseudo-March +0.020 IoU 8/8, FM-PM private +0.0015); March rows unchanged |
| submission_J3.csv | 56989460 | 0.88176 | Q3: FINAL CANDIDATE B2 (hedge) on tfb-i4: April onset = production CatBoost with the train prior; April ongoing = ogfeat F03 bag with a growth veto (veto10: window keeps fs4occ if ogfeat adds >10 cells outside the queue at T; FM-PM private +0.0036 vs fs4occ); March rows unchanged |
| submission_FDRTV.csv | 56981375 | 0.88179 | Q1: T1 FD bag on ALL rows + the blackout specialist (7 FD members: v13, s10, s11, t1dark, t1ramp, t1ramp2, t1bspec; W_FD 0.5); blackout rows 0.25 mix + 0.75 specialist; recon + TV |
| submission_I5.csv | 56956515 | 0.88176 | Q5: FINAL CANDIDATE B (hedge) on tfb-i4 (blackout specialist): April onset = production CatBoost with the train prior; April ongoing = all-finite F03 bag (train-only models); March rows unchanged |
| submission_FDRTV.csv | 56956277 | 0.88176 | Q4: T1 blackout rows = 0.25 mix + 0.75 BLACKOUT SPECIALIST (trained on official-like blackout placement: onset queues start at T+30, ongoing inside queues; 30 timing columns: onset/ongoing flag, offset, boundary values before/after the gap, ramp flow over the gap); FD bag unchanged; recon + TV |
| submission_I3.csv | 56940941 | 0.88156 | Q3: FINAL CANDIDATE B (hedge) on tfb-i2 (two ramp members): April onset = production CatBoost with the train prior; April ongoing = all-finite F03 bag (train-only models); March rows unchanged |
| submission_FDRTV.csv | 56940650 | 0.88156 | Q2: T1: second ramp member (seed 17): blackout rows 0.25 mix + 0.75 mean of the two ramp members; FD bag of 6 on all rows (v13, s10, s11, t1dark, t1ramp, t1ramp2; W_FD 0.5); recon + TV |
| submission_I1.csv | 56929045 | 0.88145 | Q1: FINAL CANDIDATE B (hedge) on tfb-h5 (5-member FD bag + ramp blackout): April onset = production CatBoost with the train prior; April ongoing = all-finite F03 bag (train-only models); March rows unchanged |
| submission_FDRTV.csv | 56908727 | 0.88145 | Q5: T1 FD bag on ALL rows + the blackout member and the ramp member (5 FD members: v13, s10, s11, t1dark, t1ramp; W_FD 0.5); blackout rows 0.25 mix + 0.75 ramp member; recon + TV |
| submission_H4.csv | 56908135 | 0.88008 | Q4: March ongoing rows from the F03 bag (all-finite + occupancy) with 103 extra ONGOING features from the 60-min history and slot T (corridor context, per-link kinematics, queue-end demand/supply, volatility-normalised speed; nothing after T); April unchanged [bag of 4] |
| submission_H3.csv | 56898576 | 0.88129 | Q3: FINAL CANDIDATE B (hedge) on the ramp-member base: April onset = production CatBoost with the train prior; April ongoing = all-finite F03 bag (train-only models); March rows unchanged |
| submission_FDRTV.csv | 56898478 | 0.88120 | Q2: T1 blackout rows (90 min after each official Task 2 origin) = 0.5 T1 mix + 0.5 member with simulated-blackout rows + 9 RAMP features (ramp-anchored flow transfer; ramps are released in the blackouts); recon + TV |
| submission_FDRTV.csv | 56897994 | 0.88129 | Q1: T1 blackout rows (90 min after each official Task 2 origin) = 0.25 T1 mix + 0.75 member with simulated-blackout rows + 9 RAMP features (ramp-anchored flow transfer; ramps are released in the blackouts); recon + TV |
| submission_G5.csv | 56865768 | 0.88092 | Q5: FINAL CANDIDATE B (hedge): April onset = production CatBoost with the train prior; April ongoing = all-finite F03 bag (train-only models, same recipe as the public best); March rows unchanged |
| submission_G4.csv | 56865673 | 0.88092 | Q4: FINAL CANDIDATE A (same-scenario bet): April onset = CatBoost bag trained with released-March windows + scenario prior; April ongoing = March-augmented F03 np/id/full + scenario prior, all-finite inputs; March rows unchanged |
| submission_G3.csv | 56865571 | 0.88092 | Q3: March ongoing rows from the F03 bag with all-finite inputs + 13 OCCUPANCY features (occupancy / per-link critical occupancy, neighbours, slope, density agreement); April unchanged |
| submission_G2.csv | 56865482 | 0.87883 | Q2: March onset rows from CatBoost onset with ALL-FINITE inputs (ineligible-but-valued cells as features), same sp4 scenario prior; April unchanged |
| submission_FDRTV.csv | 56865369 | 0.87990 | Q1: T1 blackout rows (90 min after each official Task 2 origin) = 0.5 T1 mix + 0.5 member trained with simulated-blackout transductive rows; recon + TV |
| submission_F5.csv | 56846866 | 0.88051 | P5: March ongoing rows from the ALL-FINITE F03 bag (ineligible-but-valued cells as features); April unchanged |
| submission_F4.csv | 56844607 | 0.87939 | P4: March ongoing rows of the spacing-blocked (g72) windows from an F10 pair (np+id 511, all faithful windows 1.0); other windows and April unchanged |
| submission_F3.csv | 56840738 | 0.87817 | P3: March onset rows: production CatBoost + early-month mixture prior mix9 (P2 LODO not kept) (per-cell two-hypothesis prior from the 1-4 March days before each window, W=6, pi=0.9); April rows unchanged |
| submission_F2.csv | 56840649 | 0.87465 | P2: March onset rows from the LODO CatBoost bag (training rows carry day-excluded priors: leave-one-day-out x3 + leave-week-group-out x3), same sp4 scenario prior; April rows unchanged |
| submission_FDRTV.csv | 56840469 | 0.87963 | P1: T1 = 0.5 FD-feature bag (4 members) + 0.5 old bag(8), recon + TV |
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
