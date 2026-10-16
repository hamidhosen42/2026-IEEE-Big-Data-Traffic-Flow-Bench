"""capscen_ramp_report.py -- pre-registered gate evaluation for the ramp correction arms (CAPACITY_SCENARIO_2026-10-11.md section 0).
Train OOF (folds 0, 1) and pseudo-March vs og_np_base; FM-PM (panel-exact, +-2 h, all-cell counts, offlike_m + runstart_m) on the
lab-model footprint of the 40 + 40 official ongoing windows.  Usage: capscen_ramp_report.py noramp ramprel"""
import sys, types
import numpy as np, pandas as pd
import lab, ogfeat_lib as og, exp_firstcross_lib as fc
sys.path.insert(0, f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts')
import fpmatch_lib as fpa
ARMS = sys.argv[1:]; A = f'{og.ODIR}/arms'; F = lab.F
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def pm_mean(name):
    z = [pd.read_pickle(f'{A}/{name}/pm_f{k}.pkl') for k in (0, 1)]
    return {key: (z[0][key].astype(np.float32) + z[1][key].astype(np.float32)) / 2 for key in z[0]}


def dec(p6):
    L = p6.shape[-1]; return lab.decode(p6.ravel()).reshape(F, L)


base_oof = pd.concat([pd.read_pickle(f'{A}/og_np_base/fold{k}.pkl')['df'] for k in (0, 1)], ignore_index=True)
PB = pm_mean('og_np_base')
PMZ = {panel: og.build_pm(panel) for panel in lab.PANELS}
out = []
for arm in ARMS:
    name = f'og_np_{arm}'
    oof = pd.concat([pd.read_pickle(f'{A}/{name}/fold{k}.pkl')['df'] for k in (0, 1)], ignore_index=True)
    rows = [og.delta_row(base_oof[base_oof.set == s], oof[oof.set == s], arm, 'train_' + s) for s in ('official', 'offlike', 'pseudo')]
    PN = pm_mean(name); lab_rows = []; sc = {}
    for panel in lab.PANELS:
        for s in og.PM_SETS:
            z = PMZ[panel][s]
            if not len(z['win']):
                continue
            rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=z['win'], name=s, panel=panel, cond=lab.ONGOING, L=z['qlast'].shape[1])
            a = lab.score_probs(rc, PB[(panel, s)], arm='base', set_name=s); b = lab.score_probs(rc, PN[(panel, s)], arm=arm, set_name=s)
            sc.setdefault(s, []).append((a, b))
            for j in range(len(z['win'])):
                na = int(dec(PB[(panel, s)][j]).sum()); nb = int(dec(PN[(panel, s)][j]).sum())
                lab_rows.append(dict(set=s, panel=panel, d=int(z['win'].d.iloc[j]), t=int(z['win'].t.iloc[j]), tod=z['win'].t.iloc[j] / 12,
                                     dI=b.iou.iloc[j] - a.iou.iloc[j], dn=nb - na))
    for s, lst in sc.items():
        a = pd.concat([x[0] for x in lst], ignore_index=True); b = pd.concat([x[1] for x in lst], ignore_index=True)
        bt = lab.paired_bootstrap(a, b); pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean()
        rows.append(dict(cmp=arm, set='pm_' + s, n=bt['n_windows'], base=lab.aggregate(a), new=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                         panels_pos=f'{int((pb - pa > 0).sum())}/{len(pa)}'))
    L_ = pd.DataFrame(lab_rows)
    fm = {}
    offr = []
    for split in ('validation', 'private'):
        o = []
        for k in (0, 1):
            Z = pd.read_pickle(f'{A}/{name}/off_f{k}.pkl'); o.append(Z)
        for panel in lab.PANELS:
            d0, d1 = o[0][(panel, split)], o[1][(panel, split)]
            assert (d0['window_id'] == d1['window_id']).all()
            pn = (d0['P_new'] + d1['P_new']) / 2; pbb = (d0['P_base'] + d1['P_base']) / 2
            for j, w in enumerate(d0['window_id']):
                Da, Db = dec(pbb[j]), dec(pn[j])
                offr.append(dict(split=split, window_id=w, panel=panel, tod=d0['tod'][j], nA=int(Da.sum()), nB=int(Db.sum()), changed=int((Da ^ Db).sum())))
    O = pd.DataFrame(offr); O['dn'] = O.nB - O.nA
    r = dict(arm=arm)
    for split in ('validation', 'private'):
        off = O[O.split == split]; f = fpa.forecast(L_, off)
        r[f'{split}_win'] = int((off.changed > 0).sum()); r[f'{split}_cells'] = int(off.changed.sum()); r[f'{split}_dn'] = int(off.dn.sum())
        r[f'{split}_fm_lb'] = 0.15 * f.mean()
    r['PASS'] = bool(r['validation_fm_lb'] >= 0.0015 and r['private_fm_lb'] >= 0.0005)
    out.append(r)
    print(f'\n===== {arm}'); print(pd.DataFrame(rows).round(4).to_string(index=False))
    O.to_csv(f'{lab.ROOT}/analysis/t2lab/results/capscen/official_footprint_{arm}.csv', index=False)
print('\n===== FM-PM (pre-registered gate: validation >= +0.0015 and private >= +0.0005)')
R = pd.DataFrame(out); print(R.round(5).to_string(index=False)); R.to_csv(f'{lab.ROOT}/analysis/t2lab/results/capscen/ramp_fm.csv', index=False)
