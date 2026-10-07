"""pvval ONGOING transfer test (leave-panels-out), adapted from pvog_lopo.py (killed before it finished; not edited).
The train split has ONE scenario, so "new scenario + a released masked month" is emulated by panels the base model never saw:
  base    F03 rows (all train days) of the 4 panels of the OTHER group
  raw     base + faithful ongoing windows of a 31-day source month of the TARGET panels (days SRC0..SRC0+30, selected on the
          emulated masked layer as the kernel selects March windows), RAW history / production slot-T simulation / every label, x WM
  masked  base + the SAME windows built as the kernel builds its March rows (masked history and slot T, visible labels), x WM
Targets = official-like ongoing windows of the target panels on every day outside the source month: 'raw0' = RAW histories (the
April deployment regime), 'mask10' = one emulated masked version.  np and id column sets (no prior columns).
Transfer efficiency of masked same-'scenario' rows on RAW targets = gain(masked) / gain(raw).  A panel shift is far bigger than
the scenario-seed shift: read the ratio, not the size.  Usage: pvval_og_lopo.py GROUP(A|B) [WM]"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvval_lib as V
import pvog_lib as pv

G = sys.argv[1]
WM = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SRC0 = int(os.environ.get('SRC0', 69)); SRC = pv.src_days(SRC0)
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
GROUPS = {'A': ['D7_I10_E', 'D7_I210_E', 'D7_I405_N', 'D12_I5_N'], 'B': ['D7_I10_W', 'D7_I210_W', 'D7_I405_S', 'D12_I5_S']}
if SMOKE:
    GROUPS = {'A': ['D7_I10_E'], 'B': ['D7_I10_W']}
sp = fc.ARMS['F03']
t00 = time.time()
tgt = GROUPS[G]
base_panels = [p for gg, pp in GROUPS.items() if gg != G for p in pp]
EVT = pv.target_blocks(SRC, (10,), k=4, panels=tgt, sel=lambda win: ~np.isin(win.d.to_numpy(), SRC), with_alt=False)
lab.log(f'group {G}: targets {tgt} windows', {nm: sum(ev.W for ev in EVT if ev.rc.name == nm) for nm in ('raw0', 'mask10')},
        '| base panels', base_panels, f'{time.time() - t00:.0f}s')
extra = {}
for var in ('raw', 'masked'):
    parts = [pv.source_rows(p, SRC, var, WM, k=4) for p in tgt]; parts = [x for x in parts if x is not None]
    extra[var] = {k: np.concatenate([x[k] for x in parts]) for k in ('X', 'y', 'w')}
    lab.log(f'source rows {var}: windows {sum(x["n_windows"] for x in parts)} rows {len(extra[var]["y"])} weight sum {extra[var]["w"].sum():.0f}')
for p in tgt:
    lab.drop_panel(p)
gc.collect()
P6 = {}
for cs in ('np', 'id'):
    X, y, w, info = fc.train_matrix(sp['train'], 4, pv.COLS[cs], sp['wfn'], panels=base_panels)
    for var in ('base', 'raw', 'masked'):
        t0 = time.time()
        if var == 'base':
            mdl = pv.fit_plus(X, y, w, None, None, None, pv.COLS[cs], NT)
        else:
            ex = extra[var]
            mdl = pv.fit_plus(X, y, w, np.ascontiguousarray(ex['X'][:, pv.IX[cs]]), ex['y'], ex['w'], pv.COLS[cs], NT)
        P6.update(pv.predict_members(EVT, {f'{var}_{cs}': (mdl, cs, False)}))
        lab.log(f'{G} {var} {cs}: base rows {len(y)} (weight {w.sum():.0f}) trained {time.time() - t0:.0f}s')
        del mdl
    del X, y, w; gc.collect()
tag = f'L{SRC0}_{G}_{WM:g}' + ('_smoke' if SMOKE else '')
pv.save_probs(tag, P6)
out = []
for var in ('base', 'raw', 'masked'):
    out.append(pv.score_members(EVT, P6, {f'{var}_np': 0.5, f'{var}_id': 0.5}, f'{var}:np+id', win_cols=()))
    for cs in ('np', 'id'):
        out.append(pv.score_members(EVT, P6, {f'{var}_{cs}': 1.0}, f'{var}:{cs}', win_cols=()))
Rr = pd.concat(out, ignore_index=True)
Rr.to_pickle(f'{V.RES}/lopo_scores_{tag}.pkl')
for sname in ('raw0', 'mask10'):
    for a, b in [('base:np+id', 'raw:np+id'), ('base:np+id', 'masked:np+id'), ('raw:np+id', 'masked:np+id'),
                 ('base:np', 'raw:np'), ('base:np', 'masked:np'), ('base:id', 'raw:id'), ('base:id', 'masked:id')]:
        print(V.fmt(V.pair(Rr, a, b, sname)), flush=True)
lab.log('DONE', f'{time.time() - t00:.0f}s')
