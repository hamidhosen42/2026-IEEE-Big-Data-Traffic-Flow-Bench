"""t1div: Task 1 DIVERSITY members (EXP_T1_UNET / EXP_T1_CATB) on the t1bspec pseudo-holdout (4 panels x 2 months).

Shared helpers. The evaluation cells, folds, rngs and month views are those of t1bspec_exp.run_fold (rebuilt here, then
checked against the saved t1bspec results), so the saved LightGBM arms base / base2 (stand-in T1 bag) and bs (blackout
specialist) are PAIRED with the new candidate members cell for cell.

PRE-REGISTERED GATE (10-10, written before any candidate result was computed):
  deployment arm A = mix (ordinary cells) / 0.25 mix + 0.75 bs (blackout cells), mix = mean(base, base2), then recon + TV;
  arm B(w)          = same with mix -> (1 - w) mix + w cand, w in {0.15, 0.25, 0.35}.
  PASS iff for some w: dJ(B - A) >= +0.0003 (J = 0.35 S_state + 0.10 S_LWR + 0.05 S_FD, mean over the 8 panel-months),
  95% bootstrap CI lower bound > 0, and dJ > 0 in >= 7/8 panel-months. Ordinary-only / blackout-only J are reported too.
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
