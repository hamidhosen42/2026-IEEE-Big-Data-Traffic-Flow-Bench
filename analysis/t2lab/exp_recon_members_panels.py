"""EXPERIMENT recon, robustness step: REDUCED replica members on other panels (heavy: run through run_heavy.py).
usage: exp_recon_members_panels.py <panels, comma separated> <seeds, e.g. 0> [n_trees]
Data = data/t2local/t2data_<PANEL>.npz (train truth of eligible observed cells: speed float16, flow; no occupancy; mask
rebuilt from the organizer hash). Outer fold 0 only (held-out = last 80 train days = pseudo-months A and B), fit days
thinned to keep memory low, fit rows capped at 650k and transductive rows at 350k. Same production feature / model code."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 't1lab'))
import t1lab as L

panels = sys.argv[1].split(','); seeds = [int(s) for s in sys.argv[2].split(',')]
n_trees = int(sys.argv[3]) if len(sys.argv) > 3 else 600
for panel in panels:
    Z = L.Data(panel)
    fit, months = L.folds(Z, 0)
    L.log(panel, 'days', Z.D, 'links', Z.S, '| fit days', len(fit), '| regimes held-out',
          {nm: {r: int((Z.reg[idx] == r).sum()) for r in L.REGS} for nm, idx in months.items()})
    M = L.fold_matrices(Z, 0)
    cols = list(M['cols'])
    MO = {nm: L.Month(Z, nm) for nm in months}
    for nm in months:
        m = MO[nm]; Xm = M[f'X_{nm}']
        a = L.aggregate(m.stats(Xm[:, cols.index('tv')] * m.vf_row, Xm[:, cols.index('tq')] * m.ln_row), m.reg)
        L.log(f'{panel} month {nm}: targets {int(m.tm.sum())} | temporal interp S_state {a["S_state"]:.4f} LWR {a["LWR"]:.4f} S_FD {a["S_FD"]:.4f} J {a["J"]:.5f}')
    del M
    for s in seeds:
        L.fit_members(Z, 0, s, n_trees=n_trees)
        for kind in ('nofd', 'fd'):
            for nm in months:
                m = MO[nm]
                a = L.aggregate(m.stats(*L.load_bag(nm, kind, [s], panel=panel)), m.reg)
                L.log(f'  {panel} {kind:4s} month {nm} seed {s}: S_state {a["S_state"]:.4f} LWR {a["LWR"]:.4f} S_FD {a["S_FD"]:.4f} J {a["J"]:.5f}')
    del Z, MO
L.log('DONE')
