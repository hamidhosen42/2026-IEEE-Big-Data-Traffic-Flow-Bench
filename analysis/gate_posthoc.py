"""Post-hoc variants on the pseudo-March gate probabilities saved by t2_model_v16 (gate_probs_<panel>.npz).
Decisions only for large, consistent effects (gate SD is large for small deltas). Family-mean aggregation."""
import glob, sys, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else 'kaggle_jobs/out/t2v16'
def decode(p):
    p = p.ravel().astype(np.float64); o = np.argsort(-p); cs = np.cumsum(p[o]); m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + cs[-1] - cs)); out = np.zeros(len(p), bool); out[o[:j + 1]] = True; return out
def iou(t, e, pr):
    t, e = t.ravel(), e.ravel(); u = (t & e | pr & e).sum(); return 1.0 if u == 0 else ((t & pr) & e).sum() / u
def dil(p, g, h0):
    q = p.astype(np.float32).copy()
    nb = np.maximum(np.pad(p, ((0, 0), (0, 0), (1, 0)))[:, :, :-1], np.pad(p, ((0, 0), (0, 0), (0, 1)))[:, :, 1:])
    q[:, h0:] = np.maximum(p[:, h0:], g * nb[:, h0:]); return q
rows = []
for f in sorted(glob.glob(f'{D}/gate_probs_*.npz')):
    panel = f.split('gate_probs_')[1][:-4]; z = np.load(f, allow_pickle=True)
    og = z['cond'] == 'queue_ongoing'; on = ~og
    I, N, C, L = [z[k].astype(np.float32) for k in ('id511r', 'np511r', 'cat', 'lgb7030')]
    V = {}
    if og.any():
        V['og_id'] = (I, og); V['og_np'] = (N, og); V['og_avg'] = (0.5 * I + 0.5 * N, og)
        for g in (0.5, 0.7, 0.9):
            for h0 in (0, 2, 4):
                V[f'og_id_dil{g}_h{h0}'] = (dil(I, g, h0), og)
    if on.any():
        for w in (1.0, 0.75, 0.5, 0.25, 0.0):
            V[f'on_cat{w}'] = (w * C + (1 - w) * L, on)
    for nm, (P, msk) in V.items():
        for i in np.where(msk)[0]:
            rows.append(dict(panel=panel, var=nm, cond='og' if og[i] else 'on', iou=iou(z['truth'][i], z['ev'][i], decode(P[i]))))
df = pd.DataFrame(rows)
g = df.groupby(['var', 'panel']).iou.mean().reset_index(); g['fam'] = g.panel.str.rsplit('_', n=1).str[0]
tab = g.groupby(['var', 'fam']).iou.mean().groupby('var').mean().sort_values(ascending=False)
print(tab.round(4).to_string())
