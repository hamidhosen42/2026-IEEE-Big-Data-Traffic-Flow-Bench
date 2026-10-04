"""ogfeat: distribution of the new features in four regimes (keep-region cells of ongoing windows):
  train offlike (raw history, train scenario) | pseudo-March offlike_m + runstart_m (MASKED history, new scenario)
  | official validation windows (RAW history, new scenario) | official private windows (RAW history, new scenario).
Standardised mean difference vs train offlike (train std).  Label-free for the official windows."""
import warnings
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
n1, n2 = og.static_names(), og.static_names2(); names = n1 + n2
acc = {k: [] for k in ('train', 'pmarch', 'val', 'priv')}
vis = {k: [] for k in acc}
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    rc = lab.rows(panel, 'offlike'); m = rc.keep5[:, 4] & (rc.h == 1)
    A = np.concatenate([np.asarray(og.build_fe(panel, 'offlike'))[rc.wi[m], rc.li[m]], np.asarray(og.build_fe2(panel, 'offlike'))[rc.wi[m], rc.li[m]]], 1)
    acc['train'].append(A)
    Vh, Fh, VT, d, t = og.train_inputs(P, rc.win); vis['train'].append((np.isfinite(Vh).mean(), np.isfinite(VT).mean()))
    pm = og.build_pm(panel); pm2 = og.build_pm2(panel)
    for s in og.PM_SETS:
        z = pm[s]; meta = z['meta']; k = meta[:, 2] == 1
        acc['pmarch'].append(np.concatenate([z['FE'][meta[k, 0], meta[k, 1]], pm2[s][meta[k, 0], meta[k, 1]]], 1))
    zz = np.load(P['npz'], allow_pickle=False)
    for split, key in (('validation', 'val'), ('private', 'priv')):
        R = lab.split_rows(P, split)
        og_w = np.where(R['cond'] == lab.ONGOING)[0]
        Vh = zz[f'{split}_Vh'].astype(np.float32); Fh = zz[f'{split}_Fh'].astype(np.float32); VT = zz[f'{split}_VT'].astype(np.float32)
        Fe = R['rows'][0][1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X1 = og.og_extra(P, Vh, Fh, VT, Fe); X2 = og.og_extra2(P, Vh, Fh, VT, Fe)
        keep = lab.ongoing_keep(R)
        FE = np.stack([X1[c] for c in n1] + [X2[c] for c in n2], -1)
        sel = np.zeros_like(keep); sel[og_w] = keep[og_w]
        acc[key].append(FE[sel]); vis[key].append((np.isfinite(Vh[og_w]).mean(), np.isfinite(VT[og_w]).mean()))
    lab.drop_panel(panel)
A = {k: np.concatenate(v) for k, v in acc.items()}
print({k: len(v) for k, v in A.items()})
print('history / slot-T visible share (mean over panels): ', {k: tuple(np.round(np.mean(v, 0), 3)) for k, v in vis.items() if v})
mu = np.nanmean(A['train'], 0); sd = np.nanstd(A['train'], 0) + 1e-9
T = pd.DataFrame(dict(col=names, train_mean=mu, train_std=sd, **{f'smd_{k}': (np.nanmean(A[k], 0) - mu) / sd for k in ('pmarch', 'val', 'priv')},
                      **{f'nan_{k}': np.isnan(A[k]).mean(0) for k in ('train', 'pmarch', 'val')}))
T['abs_pm_minus_val'] = (T.smd_pmarch - T.smd_val).abs()
T.to_csv(f'{og.RES}/shift_new_features.csv', index=False)
print(T.sort_values('abs_pm_minus_val', ascending=False).round(3).to_string(index=False))
