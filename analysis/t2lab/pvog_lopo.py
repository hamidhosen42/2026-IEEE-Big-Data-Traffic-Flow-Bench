"""pvog EXTRA: pseudo-scenario transfer test (leave-panels-out).  The train split has ONE scenario, so 'new scenario with a
released masked month' is emulated by panels the base model has never seen:
  base    F03 rows (all train days) of the 4 panels of the OTHER group
  raw     base + faithful ongoing windows of a 31-day source month of the TARGET panels, RAW histories, x WM
  masked  base + the SAME windows as the kernel builds March rows (masked history / slot T, visible labels), x WM
Target = official-like ongoing windows of the target panels on every day outside the source month, RAW histories (deployment
regime) and one emulated masked version.  Transfer efficiency of masked same-'scenario' rows = gain(masked) / gain(raw) on
the raw targets.  np and id column sets (no prior columns, so no location-prior leak).  A panel shift is a much bigger
shift than the scenario seed: read the RATIO, not the size."""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvog_lib as pv

WM = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
SRC0 = int(os.environ.get('SRC0', 69)); SRC = pv.src_days(SRC0)
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
GROUPS = {'A': ['D7_I10_E', 'D7_I210_E', 'D7_I405_N', 'D12_I5_N'], 'B': ['D7_I10_W', 'D7_I210_W', 'D7_I405_S', 'D12_I5_S']}
if SMOKE:
    GROUPS = {'A': ['D7_I10_E'], 'B': ['D7_I10_W']}
sp = fc.ARMS['F03']
out = []
for g, tgt in GROUPS.items():
    base_panels = [p for gg, pp in GROUPS.items() if gg != g for p in pp]
    EVT = pv.target_blocks(SRC, (10,), k=4, panels=tgt, sel=lambda win: ~np.isin(win.d.to_numpy(), SRC), with_alt=False)
    lab.log(f'group {g}: targets {tgt} windows', {nm: sum(ev.W for ev in EVT if ev.rc.name == nm) for nm in ('raw0', 'mask10')}, '| base panels', base_panels)
    extra = {}
    for var in ('raw', 'masked'):
        parts = [pv.source_rows(p, SRC, var, WM, k=4) for p in tgt]; parts = [x for x in parts if x is not None]
        extra[var] = {k: np.concatenate([x[k] for x in parts]) for k in ('X', 'y', 'w')}
        lab.log(f'  source rows {var}: windows {sum(x["n_windows"] for x in parts)} rows {len(extra[var]["y"])} weight sum {extra[var]["w"].sum():.0f}')
    for p in tgt:
        lab.drop_panel(p)
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
            lab.log(f'  {g} {var} {cs}: base rows {len(y)} (weight {w.sum():.0f}) {time.time() - t0:.0f}s')
            del mdl
        del X, y, w; gc.collect()
    pv.save_probs(f'L{SRC0}_{g}_{WM:g}' + ('_smoke' if SMOKE else ''), P6)
    for var in ('base', 'raw', 'masked'):
        out.append(pv.score_members(EVT, P6, {f'{var}_np': 0.5, f'{var}_id': 0.5}, f'{var}:np+id', win_cols=()))
        for cs in ('np', 'id'):
            out.append(pv.score_members(EVT, P6, {f'{var}_{cs}': 1.0}, f'{var}:{cs}', win_cols=()))
    del EVT, P6; gc.collect()
Rr = pd.concat(out, ignore_index=True)
Rr.to_pickle(f'{pv.RES}/lopo_{SRC0}_{WM:g}' + ('_smoke' if SMOKE else '') + '.pkl')
for sname in ('raw0', 'mask10'):
    T = pv.cmp_table(Rr, 'base:np+id', [sname], arms=['base:np+id', 'raw:np+id', 'masked:np+id'])
    print(f'\n== LOPO targets {sname} (base = base:np+id) =='); print(T.round(4).to_string(index=False))
    A = Rr[(Rr.set == sname) & (Rr.arm == 'raw:np+id')]; B = Rr[(Rr.set == sname) & (Rr.arm == 'masked:np+id')]
    bt = lab.paired_bootstrap(A, B); print(f'masked - raw on {sname}: {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] n={bt["n_windows"]}')
    for cs in ('np', 'id'):
        T = pv.cmp_table(Rr, f'base:{cs}', [sname], arms=[f'base:{cs}', f'raw:{cs}', f'masked:{cs}'])
        print(T.round(4).to_string(index=False))
lab.log('DONE')
