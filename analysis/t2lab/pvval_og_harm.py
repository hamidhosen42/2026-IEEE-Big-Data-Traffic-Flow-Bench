"""pvval ONGOING raw-deployment harm test inside the TRAIN split (one scenario: the source rows carry no new scenario information,
so this measures the mechanics only).  Adapted from pvog_harm.py (killed before it finished; not edited).
Target  = official-like ongoing windows of fold 0 ('offlike', fold 0): 'raw0' = RAW histories (lab row cache, production slot-T
          simulation, fold-0-safe prior) = the April deployment regime; 'mask10' = the same windows with the emulated masked layer.
Base    = F03 rows of folds 1-3 WITHOUT the 31-day source month (days SRC0..SRC0+30, fold 1).
VARIANTS (argv[1], comma list; target blocks built once), each for the np / id / full column sets (LightGBM 255 leaves, lr 0.05, 500 trees, seed 0):
  base    base rows only
  raw     base + the faithful ongoing windows of the source month (selected on its emulated masked layer exactly as the kernel
          selects March windows: run-starts 1.0 / t%3 rest 0.3, x WM) with RAW history, production slot-T simulation, every label
  masked  base + the SAME windows built as the kernel builds its March rows (masked history, masked slot T, visible labels), x WM
Full member at evaluation: 'tr' = fold-0 train prior (in-distribution, right prior); 'sc' = train prior blended (a = n/(n+4)) with
the masked source month (31 days, the April deployment style).  Source rows of the full model carry the 'sc'-style blend.
Probabilities -> results/pvval/probs/H<SRC0>_<VARIANT>_<WM>.pkl ; tables: pvval_og_report.py harm."""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvval_lib as V
import pvog_lib as pv

VARS = sys.argv[1].split(',')
WM = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SRC0 = int(os.environ.get('SRC0', 69)); SRC = pv.src_days(SRC0)
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
PANELS = lab.PANELS[:2] if SMOKE else lab.PANELS
MASK_SEEDS = (10,)
sp = fc.ARMS['F03']
assert (lab.fold_blocks(273)[SRC] != pv.K0).all()
t00 = time.time()


def wfn_base(win, set_name, panel):
    w = np.asarray(sp['wfn'](win, set_name, panel), np.float32).copy()
    w[np.isin(win.d.to_numpy(), SRC)] = 0.0
    return w


EVT = pv.target_blocks(SRC, MASK_SEEDS, with_march=False, panels=PANELS)
lab.log('target blocks', {nm: sum(ev.W for ev in EVT if ev.rc.name == nm) for nm in dict.fromkeys(ev.rc.name for ev in EVT)},
        f'{time.time() - t00:.0f}s')
for VAR in VARS:
    ex = None
    if VAR in ('raw', 'masked'):
        parts = [pv.source_rows(p, SRC, VAR, WM) for p in PANELS]; parts = [x for x in parts if x is not None]
        ex = {k: np.concatenate([x[k] for x in parts]) for k in ('X', 'Xf', 'y', 'w')}
        lab.log(f'source rows {VAR}: windows {sum(x["n_windows"] for x in parts)} (run-starts {sum(x["n_rs"] for x in parts)}) rows {len(ex["y"])} '
                f'pos {ex["y"].mean():.3f} weight sum {ex["w"].sum():.0f}')
        for p in PANELS:
            lab.drop_panel(p)
        del parts; gc.collect()

    members = {}
    for cs in ('np', 'id', 'full'):
        t0 = time.time()
        X, y, w, info = fc.train_matrix(sp['train'], pv.K0, pv.COLS[cs], wfn_base, panels=PANELS)
        nb, wb = len(y), float(w.sum())
        if ex is None:
            mdl = pv.fit_plus(X, y, w, None, None, None, pv.COLS[cs], NT)
        else:
            Xe = ex['Xf'] if cs == 'full' else np.ascontiguousarray(ex['X'][:, pv.IX[cs]])
            mdl = pv.fit_plus(X, y, w, Xe, ex['y'], ex['w'], pv.COLS[cs], NT)
            del Xe
        lab.log(f'{VAR} {cs}: base rows {nb} (weight {wb:.0f})' + ('' if ex is None else f' + {len(ex["y"])} source rows (weight {ex["w"].sum():.0f})')
                + f', {time.time() - t0:.0f}s')
        del X, y, w; gc.collect()
        if cs == 'full':
            members[f'{VAR}_full_tr'] = (mdl, 'full', False); members[f'{VAR}_full_sc'] = (mdl, 'full', 'sc')
        else:
            members[f'{VAR}_{cs}'] = (mdl, cs, False)
    P6 = pv.predict_members(EVT, members)
    tag = f'H{SRC0}_{VAR}_{WM:g}' + ('_smoke' if SMOKE else '')
    pv.save_probs(tag, P6)
    sc = []
    for nm, wts in ((f'{VAR}:np+id', {f'{VAR}_np': 0.5, f'{VAR}_id': 0.5}),
                    (f'{VAR}:npid+full_tr', {f'{VAR}_np': 0.25, f'{VAR}_id': 0.25, f'{VAR}_full_tr': 0.5}),
                    (f'{VAR}:npid+full_sc', {f'{VAR}_np': 0.25, f'{VAR}_id': 0.25, f'{VAR}_full_sc': 0.5}),
                    (f'{VAR}:np', {f'{VAR}_np': 1.0}), (f'{VAR}:id', {f'{VAR}_id': 1.0}),
                    (f'{VAR}:full_tr', {f'{VAR}_full_tr': 1.0}), (f'{VAR}:full_sc', {f'{VAR}_full_sc': 1.0})):
        Rr = pv.score_members(EVT, P6, wts, nm, win_cols=())
        sc.append(Rr)
        print(f'{nm:24s} ' + '  '.join(f'{s} {lab.aggregate(Rr[Rr.set == s]):.4f} (n={int((Rr.set == s).sum())})' for s in dict.fromkeys(Rr.set)), flush=True)
    pd.concat(sc, ignore_index=True).to_pickle(f'{V.RES}/harm_scores_{tag}.pkl')
    del members, P6; gc.collect()
lab.log('DONE', f'{time.time() - t00:.0f}s')
