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
