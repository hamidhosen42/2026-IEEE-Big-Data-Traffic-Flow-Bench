"""verify_occ_pmdrop.py : drop-best-panel on my seed-2 pseudo-March frame (no training)."""
import pandas as pd, lab, exp_firstcross_lib as fc, verify_occ_feat as vf
df = pd.read_pickle(f'{vf.VOUT}/pm_s2.pkl')
for s in ('offlike_m', 'runstart_m', 'faith3_m'):
    g = df[df.set == s]
    for c in ('np', 'id', 'npid'):
        a, b = fc.paired(g[g.arm == f'base_{c}'], g[g.arm == f'occ_{c}'])
        pp = b.groupby('panel').iou.mean() - a.groupby('panel').iou.mean(); best = pp.idxmax()
        r = lab.paired_bootstrap(a[a.panel != best], b[b.panel != best])
        line = f'PM seed2 [{s}] {c}: drop best {best} {r["diff"]:+.4f} [{r["lo"]:+.4f}, {r["hi"]:+.4f}] n={r["n_windows"]} | panels +{int((pp > 1e-9).sum())}/-{int((pp < -1e-9).sum())}'
        if s == 'offlike_m':
            rs = a.runstart.astype(bool).to_numpy()
            r2 = lab.paired_bootstrap(a[rs], b[rs]); r3 = lab.paired_bootstrap(a[~rs], b[~rs])
            line += f' | runstart {r2["diff"]:+.4f} [{r2["lo"]:+.4f}, {r2["hi"]:+.4f}] | midrun {r3["diff"]:+.4f} [{r3["lo"]:+.4f}, {r3["hi"]:+.4f}]'
        lab.log(line)
