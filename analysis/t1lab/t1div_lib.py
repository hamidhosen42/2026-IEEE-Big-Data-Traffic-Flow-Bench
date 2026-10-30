"""t1div: Task 1 DIVERSITY members (EXP_T1_UNET / EXP_T1_CATB) on the t1bspec pseudo-holdout (4 panels x 2 months).

Shared helpers. The evaluation cells, folds, rngs and month views are those of t1bspec_exp.run_fold (rebuilt here, then
checked against the saved t1bspec results), so the saved LightGBM arms base / base2 (stand-in T1 bag) and bs (blackout
specialist) are PAIRED with the new candidate members cell for cell.

PRE-REGISTERED GATE (10-10, written before any candidate result was computed):
  deployment arm A = mix (ordinary cells) / 0.25 mix + 0.75 bs (blackout cells), mix = mean(base, base2), then recon + TV;
  arm B(w)          = same with mix -> (1 - w) mix + w cand, w in {0.15, 0.25, 0.35}.
  PASS iff for some w: dJ(B - A) >= +0.0003 (J = 0.35 S_state + 0.10 S_LWR + 0.05 S_FD, mean over the 8 panel-months),
  95% bootstrap CI lower bound > 0, and dJ > 0 in >= 7/8 panel-months. Ordinary-only / blackout-only J are reported too.

ROUND 2 (EXP_T1_XGB / EXP_T1_CATBFTA) -- PRE-REGISTERED 2026-10-16 15:50 +06, written before any round-2 member was fitted:
  stand-in S (production-like, incl. the CatBoost member): mixC = 0.65 mean(base, base2) + 0.35 catb (catb = the 10-10
    EXP_T1_CATB lab member, one seed); ordinary cells mixC, blackout cells 0.25 mixC + 0.75 bs; then recon + TV.
  arm <cand>@w: mixC -> (1 - w) mixC + w cand, w in {0.15, 0.25, 0.35}; same blackout override, recon + TV.
  control base3@w: the same with cand = base3 (3rd LightGBM seed, FD columns, the existing 10-10 file).
  candidates (same train + transductive rows and evaluation cells as catb / base / base2):
    xgb      XGBoost hist residual regressors (speed/vf, flow/lane), 94 FD columns, max_depth 9, eta 0.06, 1500 rounds,
             subsample 0.8, colsample_bytree 0.8, min_child_weight 50, lambda 10, max_bin 256, seed 20261016
             (fallback decided on TIMING only, before any evaluation: 1000 rounds / eta 0.09 if > 25 min per panel-fold)
    catbfta  CatBoost exactly as catb (depth 8 / 2000 it + depth 10 / 1000 it averaged, lr 0.12, l2 10, Bernoulli 0.85,
             border 64) on the 74 OLD-FTA columns (= the 94 minus the 20 FD columns = v12 feature set without the profile
             columns, as the old FTA LightGBM bag), random_seed 20261016
  PASS (per candidate) iff for some w: (i) dJ(cand@w - S) >= +0.0003 (deploy scope, post recon + TV, mean of 8
    panel-months), (ii) 95% bootstrap CI lower bound > 0, (iii) dJ > 0 in >= 7/8 panel-months, AND
    (iv) dJ(cand@w - S) - dJ(base3@w - S) >= +0.00015 (diversity beyond 'one more member', same w).
    Deployment weight = the passing w with the largest dJ. Raw (no post) and ordinary / blackout scopes = info only.
  INFO (not gate), catbfta only: FTA-aware stand-in S_fta: mixC = 0.65 (0.5 mean(base, base2) + 0.5 lgbfta) + 0.35 catb,
    lgbfta = LightGBM seed 13 on the 74 old-FTA columns (production's mix already holds an old-FTA LightGBM bag at 0.5, so
    the feature diversity of catbfta is partly present there); reported as a transfer caveat.
  RESULT (16:50 +06, experiments/TRACKER.md): xgb PASS at 0.25 / 0.35 (0.35: dJ +0.00056 [+0.00044, +0.00066] 8/8, minus control
    +0.00061); catbfta FAIL. Post-hoc info (not part of the gate): xgb minus a production-capacity LightGBM control (lgbbig)
    = +0.00058 at 0.35, 8/8.

ROUND 3 (EXP_T1_DIV3) -- PRE-REGISTERED 2026-10-17 03:00 +06, written before any round-3 member was fitted:
  stand-in S3 (mirrors production steps 1-3 of tfb-k5): m0 = mean(base, base2); m1 = 0.65 m0 + 0.35 catb;
    mixX = 0.65 m1 + 0.35 xgb (catb / xgb = the round-1 / round-2 lab members, one seed each); ordinary cells mixX,
    blackout cells 0.25 mixX + 0.75 bs; then recon + TV (report: stand=xgb).
  MEMBER candidates (same rows / cells as catb / xgb; deploy scope, post recon + TV, mean of 8 panel-months):
    xgb2m    (a) second XGBoost seed: xgb params of round 2 with seed 20261017; arm = S3 with xgb -> mean(xgb, xgb2)
             (weights unchanged). Its control = base3 blended at w = 0.175 into mixX (0.175 = xgb2's effective weight).
             info: xgb2 blended on top at w (not gate; ~ an xgb up-weight).
    et       (b) sklearn ExtraTreesRegressor per target (yv, yq), 94 FD columns with NaN -> -999, 200 trees,
             max_features 0.5, min_samples_leaf 20, fitted on a 400k-row random subsample (rng 31; all rows if fewer),
             random_state 20261017 (fallback decided on TIMING / RAM only, before any evaluation: 120 trees / 250k rows
             if > 15 min per panel-fold or > 8 GB)
    lgblk    (c) different-target LightGBM (lab params, seed 14, 600 trees, lr 0.08, 255 leaves): targets
             zv = log(v/vf) - log(tv) and zk = log(k) - log(k_tpl) with k = (q/ln)/v, k_tpl = tq/(tv vf) (ratios
             clipped at 0.03 / 1e-3 before the logs); prediction v = vf tv exp(zv), q = ln k_tpl exp(zk) v
    ridge    (d) Ridge(alpha 10) on the 94 FD columns: NaN -> train median + missing indicator per column with NaN,
             standardised, 600k-row subsample (rng 32); targets yv, yq
    each blended at w in {0.15, 0.25, 0.35}: mixX -> (1 - w) mixX + w cand (blackout override, recon + TV).
    control base3@w (3rd LightGBM seed, existing file).
  MEMBER PASS iff for some w: (i) dJ(cand - S3) >= +0.0003, (ii) 95% bootstrap CI lo > 0, (iii) >= 7/8 panel-months > 0,
    (iv) dJ(cand) - dJ(control) >= +0.00015 (control: base3@w; for xgb2m base3@0.175). Deployment w = the passing w with
    the largest dJ.
  WEIGHT candidates (no new member): wx@v = S3 with the step-3 XGBoost weight 0.35 -> v, v in {0.45, 0.5, 0.6};
    wc@v = S3 with the step-2 CatBoost weight 0.35 -> v, v in {0.45, 0.5}.
  WEIGHT PASS iff dJ(arm - S3) >= +0.0002, 95% CI lo > 0, >= 7/8 panel-months. Deployment = the passing v with the
    largest dJ per family; if both families pass, the joint arm (both weights) is reported as info and deployed only if it
    passes the weight gate itself and beats each single family. Combinations member + weight = info only.
  Transfer caveat (pre-stated): the lab mix has 2 LightGBM + 1 CatBoost + 1 XGBoost seed, production 14 LightGBM + 2
    CatBoost seeds + 1 XGBoost, so lab dJ overstates LB gains (round 1-2 transferred at ~0.3-0.45x).
  RESULT (03:35 +06, experiments/TRACKER.md EXP_T1_DIV3): all FAIL. Members @best w: xgb2m +0.000017 (4/8), et -0.00018,
    lgblk -0.00008, ridge -0.00029 (control base3 negative at every w). Weights: wx@0.6 +0.00017 (7/8, below +0.0002),
    wc@0.45/0.5 +0.00006. POST-HOC info only (not gate-eligible): mean(xgb, xgb2) at XGBoost weight 0.6 / 0.7 / 0.8
    +0.00028 / +0.00033 / +0.00034 (8/8, 7/8, 7/8). No kernels written.

ROUND 4 (EXP_T1_PP4 post-processing + EXP_T1_DIV4 members; ideas of the public notebooks pulled 2026-10-29) --
PRE-REGISTERED 2026-10-30 00:20 +06, written before any round-4 arm was evaluated on an evaluation cell:
  base arm A = stand-in S3 of round 3 (production steps 1-4: m1 = 0.65 mean(base, base2) + 0.35 catb; mixX = 0.65 m1 + 0.35 xgb;
    blackout cells 0.25 mixX + 0.75 bs) followed by production step 5 (recon + TV). Same cells, units, J, bootstrap as round 3.
  Notation: per-lane density k = q / lanes / max(v, 1); anchors = RELEASED eligible cells of the fold's evaluation view
    (obs & ~(HO | EB)), per link, on the continuous time axis day-number x 288 (as the TV runs); clips at the end of every
    post-processing arm: v in [3, 1.05 vf], q >= 20.
  POST-PROCESSING (A) honghanhhh_trafficflowbench-lb-0-87141 cell [3] 'adjust': k_i = linear interpolation in time of the released
    k between the previous and the next released cell of the same link; run = number of non-released slots between them; at
    evaluation cells with both anchors and run <= r: delta = w (log k_i - log k); q <- q exp(alpha delta), v <- v exp(-(1 - alpha)
    delta). Grid w in {0.25, 0.5, 1.0} x alpha in {0, 0.5, 1.0} x r in {1, 2, 4} (27 configs). Two families:
      Aon   = recon + TV + interp (incremental on top of production; deployable)
      Ainst = recon + interp, no TV ('against TV')
    GATE per family, CROSS-FITTED because of the 27-config selection: for each month m the config with the largest mean dJ over
      the OTHER month's 4 panel-months is scored on month m's 4 panel-months; PASS iff the cross-fitted 8 panel-month dJ >= +0.0002,
      its 95% bootstrap CI lo > 0 (selection held fixed), >= 7/8 panel-months > 0, AND the full-data best config itself has
      dJ >= +0.0002, CI lo > 0, >= 7/8. Deployment = full-data best config of Aon (Ainst only if it passes and beats Aon's dJ).
  POST-PROCESSING (B) robinamirbahar_trafficflowbench-physics-guided-state-odme cell [9], three named arms on top of recon + TV:
      B1 nb94     at evaluation cells whose two time neighbours (t-1, t+1) are both released: log k <- 0.94 log k
                  + 0.03 (log k_prev + log k_next); change d (clipped +-0.35) split as the notebook: q <- q e^{0.78 d}, v <- v e^{-0.22 d}
      B2 nb94cap  (PRIMARY = the requested spec) B1, then an FD cap at density-congested cells (k > kc per lane, FD-valid links):
                  q <- min(q, lanes v w kj / (v + w)) = the congested-branch flow at the cell's own speed (speed unchanged)
      B3 robina   the notebook's literal 3 stages: (1) k > kc: q <- 0.05 q + 0.95 (0.3 lanes w (kj - k) + 0.7 lanes min(cap,
                  v w kj / (v + w))); (2) interp-N anchor (same day, run <= 4) with log weight 0.14 / 0.08 / 0.04 for run
                  1 / 2 / 3-4, then neighbour smoothing on the log-density grid of released + evaluation cells: weights 0.06 + 0.06
                  when both neighbours are released, otherwise 0.04 for a released / 0.05 for a predicted neighbour, d clipped
                  +-0.35, split 0.78 / 0.22; (3) gamma isodensity: ratio = clip(q_FD(k) / q, 0.8, 1.2), g = 0.035 (free) / 0.12
                  (congested), (v, q) <- gamma (v, q) if 5 <= gamma v <= 132.
    GATE per arm: dJ >= +0.0002, 95% CI lo > 0, >= 7/8 (deploy scope, post). Multiplicity (3 arms): a secondary arm (B1 / B3)
      must also have dJ_val > 0 and dJ_pri > 0. Deployment = B2 if it passes, else the passing arm with the largest dJ.
  MEMBERS (C) plarmuseau_trafficflowbench-eleven-anomalies-of-the-data section 12 ('month-itself' estimators; no training;
    computed per fold on the evaluation view = released cells minus the evaluation cells; per-link normalisation by the link's
    own mean over the view, u = v / mean_v(link), f = q / mean_q(link); script analysis/t1lab/t1div_lrtc.py):
      lrtc   EM low-rank tensor completion of (days x 288 x links) per channel: init = per-(tod, link) mean over days (fallback
             link mean, then 1); 60 iterations of Z = mean over the 3 unfoldings of the rank-r_i truncated-SVD projection,
             X = observed value where released else Z; prediction = Z. Ranks (r_day, r_tod, r_link) per channel chosen ONCE by
             self-validation RMSE on 2% of the released NON-evaluation cells (rng 41) of D12_I5_S validation fold 0 over
             {2, 4, 8} x {5, 10, 20, 40} x {5, 10, 20, 40} (the pilot touches no evaluation cell), then fixed for all 16 views.
      pw41   41-link position-window profile: P(c, tod, link) = mean of the normalised value over the days of day type c
             (Mon-Fri / Sat-Sun) and the links within +-20 positions (order_index), released cells only; prediction =
             link mean x P (fallback: all-day profile, then the link mean).
      lrtcQ / pw41Q  the same members with their speed replaced by the stand-in mixX speed (flow-only blend).
    Clips v in [3, 1.05 vf], q >= 20; blended at w in {0.15, 0.25, 0.35}: mixX -> (1 - w) mixX + w cand before the blackout
    override, then recon + TV; control base3@w (existing file).
    MEMBER PASS iff for some w: dJ >= +0.0003, 95% CI lo > 0, >= 7/8, and dJ - dJ(base3@w) >= +0.00015 (rounds 2-3 gate).
  Transfer caveats (pre-stated): post-processing deltas are not diluted by the production bag (14 LightGBM + 2 CatBoost + 1
    XGBoost), so they should transfer closer to 1:1 than members (~0.3-0.45x in rounds 1-2); lab ordinary cells are isolated
    3% holdouts, so runs of >= 2 hidden cells are under-represented relative to the release (mean masked run 1.4 slots).
  PILOT (00:40 +06, analysis/t2lab/results/t1div/lrtc_pilot.json): ranks speed (8, 20, 20) self-val RMSE 3.06 km/h, flow (8, 40, 10)
    33.0 vph/lane (references on the same cells: pw41 14.4 / 115, per-(tod, link) mean 12.3 / 158).
  RESULT (01:10 +06, experiments/TRACKER.md EXP_T1_PP4 / EXP_T1_DIV4; logs analysis/t1lab/logs/t1div4_*.log):
    (A) FAIL: all 27 configs negative in both families (best Aon 0.25/1.0/1 -0.00041 0/8, cross-fit -0.00041; Ainst -0.00039);
        on run-1 ordinary cells the model's mean |log k| error is 0.022-0.028 vs 0.030-0.035 for the interpolation.
    (B) FAIL: B1 -0.00003 (2/8), B2 +0.00006 [+0.00001, +0.00013] 6/8, B3 -0.00063 (0/8).
    (C) lrtcQ PASS at every w: @0.15 +0.00038 [+0.00035, +0.00042] 8/8, @0.25 +0.00051 [+0.00045, +0.00058] 8/8,
        @0.35 +0.00054 [+0.00046, +0.00064] 8/8 (minus control +0.00045 / +0.00066 / +0.00080, 8/8) -> deployment w = 0.35;
        ordinary-rows-only scope @0.35 +0.00054 [+0.00045, +0.00064] 8/8 (blackout scope +0.000008). lrtc (both channels:
        speed RMSE 3.03 vs 1.65) FAIL, pw41 / pw41Q FAIL (-0.003 .. -0.069). Parity: base3@w and wx@0.6 reproduce round 3 exactly.
    POST-HOC info (not gate-eligible): flow ranks (8, 20, 10) / (8, 40, 20) also pass (+0.00043 / +0.00050 at 0.25-0.35, 8/8),
        (4, 40, 10) weaker (+0.00027 @0.15, +0.00012 @0.35); pp arms re-run on S3 + lrtcQ@0.35 keep their signs.
    Kernels (NOT pushed): kaggle_jobs/swap_lrtc_k7.py (tfb-k7 = tfb-k5d + step M + recon + TV), swap_k8.py (hedge B on k7),
        t4_validate_k7.py, k7_followup.py (never submits); local tests t1lrtc_kernel_fntest.py / t1k8_swap_fntest.py PASSED.
    LB (10-29): tfb-k7 0.88259 (+0.00034 vs tfb-k5, ~0.6x the lab dJ).

ROUND 5 (EXP_T1_DIV5 members + EXP_T1_W5 weights; LRTC follow-ups) -- PRE-REGISTERED 2026-10-30 14:30 +06, written before any
round-5 member, pilot or arm was computed:
  stand-in S4 (mirrors tfb-k7 = production steps 1-5 + M): m1 = 0.65 mean(base, base2) + 0.35 catb; mixX = 0.65 m1 + 0.35 xgb;
    ORDINARY cells v = mixX_v, q = 0.65 mixX_q + 0.35 lrtc_q (lrtc = the round-4 flow LRTC file, ranks (8, 40, 10));
    BLACKOUT cells 0.25 mixX + 0.75 bs (untouched by step M); then recon + TV. Same cells, units, J, bootstrap as rounds 3-4.
  Members are computed per (panel, month, fold) on the evaluation view (released cells minus HO | EB, ineligible-valued cells kept),
    clips v in [3, 1.05 vf], q >= 20 (script analysis/t1lab/t1div5_lrtc.py, report t1div5_report.py). New ranks are fixed ONCE by
    self-validation on the round-4 pilot cells (2 % released NON-evaluation cells, rng 41, D12_I5_S validation fold 0; no evaluation
    cell is touched) and saved to lrtc5_pilot.json before any evaluation-cell number exists.
  MEMBER arms (ordinary cells only, flow only):
    (a) lrtc2m: S4 with lrtc_q -> mean(lrtc_q, lrtc2_q) at the same 0.35. Two variants:
          a_r  lrtc2r = flow LRTC with the best pilot triple of the EXTENDED grid r_day {12, 16, 24} x r_tod {20, 40, 60} x
               r_link {5, 10, 20} (the round-4 best day rank 8 was the edge of its grid), round-4 init
          a_i  lrtc2i = flow LRTC ranks (8, 40, 10) with a different initialisation: missing cells start from the 41-link day-type
               position-window profile (pw41) instead of the per-(tod, link) mean
        control ctl2m = S4 with lrtc_q -> mean(lrtc_q, base3_q) ('base3 in the second slot').
    (b) separate members ON TOP of S4: q <- (1 - w) q_S4 + w q_c, w in {0.15, 0.25, 0.35}:
          b_k  lrtcK = LRTC of the per-link-mean-normalised DENSITY k = q / max(v, 1) (cells with both channels), ranks = best pilot
               triple of the round-4 grid {2, 4, 8} x {5, 10, 20, 40} x {5, 10, 20, 40} by per-lane density RMSE; q = k_hat x v_S4
          b_l  lrtcL = LRTC of the link-centred LOG-flow log(max(q, 1)) - mean_link, same grid, ranks by per-lane flow RMSE of
               exp(.); q = exp(Z + mean_link)
        control base3Q@w = base3's flow on top of S4 at the same w (ordinary cells).
    (e) stkQ ('stack several such estimators', plarmuseau section 12): per (panel, month, fold) hide 2 % of the view's released cells
        (SV, rng 4300 + 10 split + fold; never evaluation cells), compute on the SV-hidden view the flow estimators {lrtc (8, 40, 10),
        lrtc2r, lrtcL, pw41 flow, per-(tod, link) mean}, fit NNLS weights (no intercept) of q / mean_link on the SV cells, predict the
        evaluation cells with the same estimators; ON TOP of S4 at w {0.15, 0.25, 0.35}; control base3Q@w.
    MEMBER PASS iff for some w (a: its single arm): dJ(arm - S4) >= +0.0003 (deploy scope, post recon + TV, mean of 8 panel-months),
      95 % bootstrap CI lo > 0, >= 7/8 panel-months > 0, AND dJ(arm) - dJ(control) >= +0.00015 (same w). Deployment = the passing
      variant / w with the largest dJ.
  WEIGHT / POST-PROCESSING arms:
    (c) wq@0.5: S4 with the step-M LRTC weight 0.35 -> 0.5 (ordinary cells).
    (d) blackout cells (after the blackout override; step M left them untouched), w in {0.15, 0.35}:
          d1 blkQ@w  (PRIMARY = the requested spec): q <- (1 - w) q_blk + w lrtc_q
          d2 blkVQ@w (secondary; round 4 showed the LRTC SPEED helps blackout cells, blk scope +0.00015 at an effective 0.0875):
               the same on both channels with the round-4 speed LRTC (ranks (8, 20, 20), same file)
    WEIGHT PASS iff dJ(arm - S4) >= +0.0002, 95 % CI lo > 0, >= 7/8 panel-months; d2 must also have dJ_val > 0 and dJ_pri > 0
      (multiplicity). Deployment = the passing config with the largest dJ per family.
  Combinations of passing arms: info only; deployed only if the combination passes its own gate vs S4 and beats each single arm.
  Transfer caveat (pre-stated): lrtcQ transferred at ~0.6x (lab +0.00054 -> LB +0.00034); members against S4 (which already holds a
    flow LRTC) are expected to be smaller than against S3.
  PILOT (14:50 +06, lrtc5_pilot.json): lrtc2r ranks (24, 40, 10) self-val flow RMSE 29.2 vs 33.0 for (8, 40, 10) (r_day 24 = the edge
    again); lrtcK (8, 20, 20) density RMSE 1.27 veh/km/lane (flow via the true speed 96 vph/lane: density is far less low-rank);
    lrtcL (8, 40, 10) 34.4; lrtc2i (pw41 init) 33.0 (init barely matters).
  RESULT (15:45 +06, logs/t1div5_report.log, jproxy_r5.csv; parity: dJ(S3 - S4) = -0.000536 = minus the round-4 lrtcQ@0.35 ord scope):
    ALL FAIL. lrtc2m_r +0.000178 [+0.000146, +0.000212] 8/8 (vs ctl2m +0.00035); lrtc2m_i +0.000017; lrtcK -0.0019 .. -0.0075;
    lrtcL +0.00005 / +0.00002 / -0.00005; stkQ +0.00014 / +0.00019 / +0.00021 (8/8, vs control +0.0002 .. +0.0004; NNLS weights
    ~0.6-0.85 on lrtc2r); wq@0.5 -0.00013; blkQ +0.000004 / -0.00006; blkVQ@0.15 +0.000185 [+0.000132, +0.000235] 7/8 (val +0.00017,
    pri +0.00020; just under +0.0002), blkVQ@0.35 +0.00006. Raw ordinary flow RMSE per lane: S4 28.3, lrtc 32.5, lrtc2r 30.2, stkQ 29.9.
    No kernel (pre-registered rule).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from t1ramp_exp import month_cube, GAP, SPLITS  # noqa: E402

PANELS = ('D12_I5_S', 'D7_I10_E', 'D7_I210_W', 'D7_I405_S')
BSPEC_CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', 't1bspec')
BSPEC_RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', 't1bspec')
RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', 't1div')
os.makedirs(RES, exist_ok=True)


def rows_of(sims, shape):
    r = np.zeros(shape[:2], bool)
    for d, t, _ in sims:
        r[d, t + 1:t + 1 + GAP] = True
    return r


def eval_masks(M, meta, split, fold):
    """= t1bspec_exp.run_fold: EB (evaluation-half blackout rows), HO (3 % ordinary eval cells), HP (validation internal
    holdout), hold, ev_id, simE."""
    obs = M['obs']; shape = obs.shape; si = SPLITS.index(split)
    simF = meta[f'{split}_simF']
    half = np.arange(len(simF)) % 2; eval_half = 1 if fold == 0 else 0
    simE = [tuple(s) for k, s in enumerate(simF) if half[k] == eval_half]
    rowsE = rows_of(simE, shape); ev_id = np.full(shape[:2], -1, np.int32)
    for k, (d, t, c) in enumerate(simF):
        if half[k] == eval_half:
            ev_id[d, t + 1:t + 1 + GAP] = k
    allsim = rows_of([tuple(s) for s in simF], shape) | rows_of([tuple(s) for s in meta[f'{split}_simR0']], shape) \
        | rows_of([tuple(s) for s in meta[f'{split}_simR1']], shape)
    EB = np.broadcast_to(rowsE[:, :, None], shape)
    HO = obs & ~np.broadcast_to(allsim[:, :, None], shape) & (np.random.default_rng(500 + si).random(shape) < 0.03)
    HP = (obs & (np.random.default_rng(7).random(shape) < 0.15)) if split == 'validation' else np.zeros(shape, bool)
    return dict(EB=EB, HO=HO, HP=HP, hold=HP | HO | (EB & obs), ev_id=ev_id, simE=simE)


def eval_cells(EM, obs):
    """(kind, mask) in the order the results store them."""
    return (('blk', EM['EB'] & obs), ('ord', EM['HO']))


def load_meta(panel):
    return np.load(f'{BSPEC_CACHE}/{panel}_meta.npz', allow_pickle=True)
