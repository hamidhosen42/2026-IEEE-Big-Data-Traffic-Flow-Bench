"""Precision / calibration of predicted cells OUTSIDE the visible queue at T ('new' cells) vs inside, train (diag fold 0) vs pseudo-March."""
import sys
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd, lab, ogfeat_lib as og
F = 6
def stats(items):
    acc = {}
    for arm, P6, truth, ev, ql in items:
        for i in range(len(P6)):
            L = ql.shape[1]; p = P6[i].astype(np.float32); D = lab.decode(p.ravel()).reshape(F, L)
            q = np.repeat(ql[i][None], F, 0); E = ev[i]; T = truth[i] & E
            for nm, m in (('new', D & ~q & E), ('old', D & q & E)):
                a = acc.setdefault((arm, nm), [0, 0, 0.0]); a[0] += m.sum(); a[1] += (m & T).sum(); a[2] += p[m].sum()
            m = ~D & ~q & E; a = acc.setdefault((arm, 'missed_new'), [0, 0, 0.0]); a[0] += T.sum(); a[1] += (m & T).sum(); a[2] += 0
    return {k: (v[0], v[1] / max(v[0], 1), v[2] / max(v[0], 1)) for k, v in acc.items()}
Z = pd.read_pickle(sys.argv[1]); items = []
for (panel, s), (widx, truth, ev, ql, win) in Z['aux'].items():
    if s != 'offlike': continue
    for arm in ('base_afocc', 'all2_afocc'):
        items.append((arm, Z['P'][(arm, panel, s)][1], truth, ev, ql))
print('train offlike fold 0:', {k: tuple(np.round(v, 3)) for k, v in stats(items).items()})
def pm(name, pmk=(0, 1)):
    acc = None
    for k in pmk:
        z = pd.read_pickle(f'{og.ODIR}/arms/{name}/pm_f{k}.pkl')
        acc = {key: v.astype(np.float32) for key, v in z.items()} if acc is None else {key: acc[key] + z[key].astype(np.float32) for key in acc}
    return {k: v / len(pmk) for k, v in acc.items()}
for s in ('offlike_m', 'runstart_m'):
    items = []
    for arm in ('og_np_base', 'og_np_all2'):
        for (panel, ss), P6 in pm(arm).items():
            if ss != s: continue
            zz = og.build_pm(panel)[s]; items.append((arm, P6, zz['truth'], zz['ev'], zz['qlast']))
    print('pseudo-March', s, {k: tuple(np.round(v, 3)) for k, v in stats(items).items()})
