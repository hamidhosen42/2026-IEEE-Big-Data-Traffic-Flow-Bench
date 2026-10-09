import sys
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd, lab
Z = pd.read_pickle(sys.argv[1]); PS = Z['P']; AUX = Z['aux']; F = 6
rows = []
for (panel, s), (widx, truth, ev, ql, win) in AUX.items():
    _, Pa = PS[('base_afocc', panel, s)]; _, Pb = PS[('all2_afocc', panel, s)]; L = ql.shape[1]
    for j in range(len(widx)):
        pa = Pa[j].astype(np.float32); pb = Pb[j].astype(np.float32)
        A = lab.decode(pa.ravel()).reshape(F, L); B = lab.decode(pb.ravel()).reshape(F, L); q = np.repeat(ql[j][None], F, 0)
        T = truth[j] & ev[j]; E = ev[j]; addout = int((B & ~A & ~q).sum())
        cand = {'base': A, 'all2': B, 'and': A & B, 'blend0.5': lab.decode((0.5 * pa + 0.5 * pb).ravel()).reshape(F, L),
                'veto3': A if addout > 3 else B, 'veto10': A if addout > 10 else B}
        r = dict(set=s, panel=panel, family=panel.rsplit('_', 1)[0])
        for k, P in cand.items():
            u = ((T | P) & E).sum(); r[k] = 1.0 if u == 0 else (T & P & E).sum() / u
        rows.append(r)
W = pd.DataFrame(rows)
for s, g in W.groupby('set'):
    agg = lambda c: g.groupby(['family', 'panel'])[c].mean().groupby(level=0).mean().mean()
    b = agg('base'); print(s, len(g), ' | '.join(f'{c} {agg(c) - b:+.4f}' for c in ('all2', 'and', 'blend0.5', 'veto3', 'veto10')))
