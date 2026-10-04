"""EXPERIMENT recon, step 1: train the replica bag members (heavy: run through run_heavy.py).
usage: exp_recon_members.py <outer fold 0|1> <seeds, e.g. 0,1,2> [n_trees] [leaves] [lr]
Each seed gives one no-FD member (74 production columns) and one FD-feature member (94 columns) trained on the same rows
(fit-block targets + 3 transductive rounds x 20% of the released cells of the two held-out pseudo-months), production code
via analysis/t1lab/t1lab.py. Predictions at every held-out target cell are saved to data/t1local/members/."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 't1lab'))
import t1lab as L

outer = int(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(',')]
n_trees = int(sys.argv[3]) if len(sys.argv) > 3 else 600
leaves = int(sys.argv[4]) if len(sys.argv) > 4 else 255
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08
Z = L.Data()
L.log('outer', outer, 'seeds', seeds, 'trees', n_trees, 'leaves', leaves, 'lr', lr, 'threads', L.THREADS)
M = L.fold_matrices(Z, outer)
fit, months = L.folds(Z, outer)
L.log('fit days', Z.days[fit][0], '..', Z.days[fit][-1], len(fit), {r: int((Z.reg[fit] == r).sum()) for r in L.REGS})
MO = {nm: L.Month(Z, nm) for nm in months}
cols = list(M['cols'])
for nm, idx in months.items():
    m = MO[nm]
    Xm = M[f'X_{nm}']
    tvv = Xm[:, cols.index('tv')] * m.vf_row; tqq = Xm[:, cols.index('tq')] * m.ln_row       # temporal interpolation reference
    a = L.aggregate(m.stats(tvv, tqq), m.reg)
    L.log(f'month {nm}: days {Z.days[idx][0]} .. {Z.days[idx][-1]}', {r: int((m.reg == r).sum()) for r in L.REGS}, 'targets', int(m.tm.sum()),
          f'| temporal interp S_state {a["S_state"]:.4f} LWR {a["LWR"]:.4f} S_FD {a["S_FD"]:.4f} J {a["J"]:.5f}')
for s in seeds:
    L.fit_members(Z, outer, s, n_trees=n_trees, leaves=leaves, lr=lr)
    for kind in ('nofd', 'fd'):
        for nm in months:
            m = MO[nm]
            one = L.aggregate(m.stats(*L.load_bag(nm, kind, [s])), m.reg)
            done = [x for x in seeds if os.path.exists(L.member_path(outer, x, kind)) and x <= s]
            bag = L.aggregate(m.stats(*L.load_bag(nm, kind, done)), m.reg)
            L.log(f'  {kind:4s} month {nm} seed {s}: S_state {one["S_state"]:.4f} LWR {one["LWR"]:.4f} S_FD {one["S_FD"]:.4f} J {one["J"]:.5f}'
                  f' | bag{len(done)} S_state {bag["S_state"]:.4f} LWR {bag["LWR"]:.4f} S_FD {bag["S_FD"]:.4f} J {bag["J"]:.5f}')
L.log('DONE')
