"""Combinations of base and all2 on pseudo-March (probabilities of og_np_base / og_np_all2, fold-0/1 model mean):
blend w (P = (1-w) pa + w pb), growth veto (all2 decision unless it adds > g cells outside the visible queue at T, else base),
shrink-only (A & B).  Per-window frames are saved for the footprint-matched forecast."""
import sys, types
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
OUT = sys.argv[1]
F = 6


def pm(name, pmk=(0, 1)):
    d = f'{og.ODIR}/arms/{name}'; acc = None
    for k in pmk:
        z = pd.read_pickle(f'{d}/pm_f{k}.pkl')
        acc = {key: v.astype(np.float32) for key, v in z.items()} if acc is None else {key: acc[key] + z[key].astype(np.float32) for key in acc}
    return {k: v / len(pmk) for k, v in acc.items()}


PA = pm('og_np_base'); PB = pm('og_np_all2')
rows = []
for (panel, s), pa in PA.items():
    pb = PB[(panel, s)]; zz = og.build_pm(panel)[s]
    truth, ev, ql, win = zz['truth'], zz['ev'], zz['qlast'], zz['win']
    for i in range(len(win)):
        T = truth[i] & ev[i]; E = ev[i]; q = np.repeat(ql[i][None], F, 0); L = ql.shape[1]
        A = lab.decode(pa[i].ravel()).reshape(F, L); B = lab.decode(pb[i].ravel()).reshape(F, L)
        cand = {'base': A, 'all2': B, 'and': A & B}
        for w in (0.25, 0.5, 0.75):
            cand[f'blend{w}'] = lab.decode(((1 - w) * pa[i] + w * pb[i]).ravel()).reshape(F, L)
        addout = (B & ~A & ~q).sum()
        for g in (3, 10):
            cand[f'veto{g}'] = A if addout > g else B
        def iou(P):
            u = ((T | P) & E).sum(); return 1.0 if u == 0 else (T & P & E).sum() / u
        r = dict(set=s, panel=panel, family=panel.rsplit('_', 1)[0], d=int(win.d.iloc[i]), t=int(win.t.iloc[i]), addout=int(addout))
        for k, P in cand.items():
            r['iou_' + k] = iou(P); r['n_' + k] = int((P).sum())
        rows.append(r)
W = pd.DataFrame(rows); W.to_pickle(f'{OUT}/ogf_combo_pm.pkl')
arms = [c[4:] for c in W if c.startswith('iou_')]
for s, g in W.groupby('set'):
    base = g.groupby(['family', 'panel']).iou_base.mean().groupby(level=0).mean().mean()
    out = []
    for a in arms:
        v = g.groupby(['family', 'panel'])['iou_' + a].mean().groupby(level=0).mean().mean()
        pp = (g.groupby('panel')['iou_' + a].mean() - g.groupby('panel').iou_base.mean())
        out.append(f'{a} {v - base:+.4f} ({int((pp > 0).sum())}/8)')
    print(s, len(g), '| vs base:', ', '.join(out))
