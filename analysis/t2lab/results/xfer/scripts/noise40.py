"""40-window LB noise model for ogfeat: resample 5 windows per panel from lab per-window paired deltas.
Sources: train offlike fold-0 on production inputs (overlap_v27_f0.pkl), pseudo-March offlike_m / runstart_m (og_np_base vs og_np_all2,
mean of fold-0/1 models)."""
import sys, types
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
import lab, ogfeat_lib as og

OUT = sys.argv[1]
def pm(name, pmk=(0, 1)):
    d = f'{og.ODIR}/arms/{name}'; acc = None
    for k in pmk:
        z = pd.read_pickle(f'{d}/pm_f{k}.pkl')
        acc = {key: v.astype(np.float32) for key, v in z.items()} if acc is None else {key: acc[key] + z[key].astype(np.float32) for key in acc}
    res = []
    for (panel, s), P6 in acc.items():
        zz = og.build_pm(panel)[s]
        rc = types.SimpleNamespace(truth=zz['truth'], ev=zz['ev'], qlast=zz['qlast'], win=zz['win'], name=s, panel=panel, cond=lab.ONGOING, L=zz['qlast'].shape[1])
        df = lab.score_probs(rc, P6 / len(pmk), arm=name, set_name=s); df['vis'] = zz['win'].vis.to_numpy(); df['day'] = zz['win'].d.to_numpy()
        df['hist_vis'] = zz['win'].hist_vis.to_numpy(); df['size_T'] = zz['qlast'].sum(1)
        res.append(df)
    return pd.concat(res, ignore_index=True)

key = ['set', 'panel', 'd', 't']
A = pm('og_np_base'); B = pm('og_np_all2')
M = A.merge(B[key + ['iou', 'n_pred']], on=key, suffixes=('_a', '_b'))
O = pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/ogfeat/overlap_v27_f0.pkl')
oa = O[O.arm == 'base_afocc']; ob = O[O.arm == 'all2_afocc']
MO = oa.merge(ob[key + ['iou', 'n_pred']], on=key, suffixes=('_a', '_b'))
MO['size_T'] = np.nan
allm = pd.concat([MO.assign(src='train_prodinputs'), M.assign(src='pmarch')], ignore_index=True)
allm['dI'] = allm.iou_b - allm.iou_a
allm.to_pickle(f'{OUT}/ogf_perwindow_lab.pkl')
rng = np.random.default_rng(0)
obs = -0.00121 / 0.15
print(f'observed LB ongoing IoU delta {obs:+.4f}')
for (src, s), g in allm.groupby(['src', 'set']):
    groups = [gg.dI.to_numpy() for _, gg in g.groupby('panel')]
    fam = [p.rsplit('_', 1)[0] for p, _ in g.groupby('panel')]
    sims = np.zeros(20000)
    for b in range(len(sims)):
        pm_ = [x[rng.integers(0, len(x), 5)].mean() for x in groups]
        sims[b] = pd.Series(pm_).groupby(fam).mean().mean()
    print(f'{src:18s} {s:11s} n {len(g):5d} | window dI mean {g.dI.mean():+.4f} sd {g.dI.std():.4f} | share changed {np.mean(g.dI != 0):.2f} | '
          f'40-window replica: mean {sims.mean():+.4f} sd {sims.std():.4f} | P(<= obs) {np.mean(sims <= obs):.4f} | '
          f'P(<0) {np.mean(sims < 0):.3f} | 0.6x-shrunk P(<=obs) {np.mean(sims - 0.4 * sims.mean() <= obs):.4f}')
