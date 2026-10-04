"""pvog EXP 3 / 4: raw-deployment harm test inside the TRAIN split (one scenario, so source rows carry NO scenario information).
Target = official-like ongoing windows of fold 0 with RAW histories ('raw0', lab row cache, fold-0-safe prior) and the same
windows with emulated masked histories ('mask<seed>').  Base = F03 rows of folds 1-3 without a 31-day source month (days
SRC0..SRC0+30, fold 1).  Variants (comma list in argv[1]), each for the np / id / full column sets:
  base    base rows only
  raw     base + the faithful ongoing windows of the source month, RAW history / every observed label, x WM
  masked  base + the SAME windows built as the kernel builds its March rows (masked history, masked slot T, visible labels), x WM
  march   base + the released-March faithful windows (all 31 days; WRONG scenario for a train target), x WM
Members saved to results/pvog/probs/H<SRC0>_<variant>_<WM>.pkl; pvog_report.py scores the blends."""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvog_lib as pv

VARS = sys.argv[1].split(',')
WM = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SRC0 = int(os.environ.get('SRC0', 69)); SRC = pv.src_days(SRC0)
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
PANELS = lab.PANELS[:2] if SMOKE else lab.PANELS
MASK_SEEDS = (10,) if SMOKE else (10, 11, 12, 13)
sp = fc.ARMS['F03']
assert (lab.fold_blocks(273)[SRC] != pv.K0).all()


def wfn_base(win, set_name, panel):
    w = np.asarray(sp['wfn'](win, set_name, panel), np.float32).copy()
    w[np.isin(win.d.to_numpy(), SRC)] = 0.0
    return w


EVT = pv.target_blocks(SRC, MASK_SEEDS, panels=PANELS)
lab.log('target blocks', {nm: sum(ev.W for ev in EVT if ev.rc.name == nm) for nm in dict.fromkeys(ev.rc.name for ev in EVT)})
extra = {}
for var in VARS:
    if var in ('raw', 'masked'):
        parts = [pv.source_rows(p, SRC, var, WM) for p in PANELS]; parts = [x for x in parts if x is not None]
        extra[var] = {k: np.concatenate([x[k] for x in parts]) for k in ('X', 'Xf', 'y', 'w')}
        lab.log(f'source rows {var}: windows {sum(x["n_windows"] for x in parts)} (run-starts {sum(x["n_rs"] for x in parts)}) rows {len(extra[var]["y"])} '
                f'pos {extra[var]["y"].mean():.3f} weight sum {extra[var]["w"].sum():.0f}')
        for p in PANELS:
            lab.drop_panel(p)
    elif var == 'march':
        TRm, _ = pv.march_data(31, WM, eval_from=99, panels=PANELS, verbose=False, prior_k=pv.K0)
        extra[var] = TRm
        lab.log(f'March rows (all days): {len(TRm["y"])} pos {TRm["y"].mean():.3f} weight sum {TRm["w"].sum():.0f}')
gc.collect()

for var in VARS:
    members = {}
    for cs in ('np', 'id', 'full'):
        t0 = time.time()
        X, y, w, info = fc.train_matrix(sp['train'], pv.K0, pv.COLS[cs], wfn_base, panels=PANELS)
        nb, wb = len(y), float(w.sum())
        if var == 'base':
            mdl = pv.fit_plus(X, y, w, None, None, None, pv.COLS[cs], NT)
        else:
            ex = extra[var]
            Xe = ex['Xf'] if cs == 'full' else np.ascontiguousarray(ex['X'][:, pv.IX[cs]])
            mdl = pv.fit_plus(X, y, w, Xe, ex['y'], ex['w'], pv.COLS[cs], NT)
            del Xe
        lab.log(f'{var} {cs}: base rows {nb} (weight {wb:.0f})' + ('' if var == 'base' else f' + {len(extra[var]["y"])} extra rows (weight {extra[var]["w"].sum():.0f})')
                + f', {time.time() - t0:.0f}s')
        del X, y, w; gc.collect()
        if cs == 'full':
            members[f'{var}_full_tr'] = (mdl, 'full', False)
            if var != 'march':
                members[f'{var}_full_sc'] = (mdl, 'full', 'sc')
            if var in ('base', 'march'):
                members[f'{var}_full_wrong'] = (mdl, 'full', 'wrong')
        else:
            members[f'{var}_{cs}'] = (mdl, cs, False)
    P6 = pv.predict_members(EVT, members)
    tag = f'H{SRC0}_{var}_{WM:g}' + ('_smoke' if SMOKE else '')
    pv.save_probs(tag, P6)
    fm = 'tr' if var == 'march' else 'sc'
    for nm, wts in ((f'{var}:np+id', {f'{var}_np': 0.5, f'{var}_id': 0.5}),
                    (f'{var}:npid+full_tr', {f'{var}_np': 0.25, f'{var}_id': 0.25, f'{var}_full_tr': 0.5})):
        Rr = pv.score_members(EVT, P6, wts, nm, win_cols=())
        print(f'{nm:24s} ' + '  '.join(f'{s} {lab.aggregate(Rr[Rr.set == s]):.4f} (n={int((Rr.set == s).sum())})' for s in dict.fromkeys(Rr.set)), flush=True)
    del members, P6; gc.collect()
lab.log('DONE')
