"""TRACK ogmodel: pseudo-March evaluation of final (all-train-days) structure models on the cached pseudo-March rows.
Regime: NEW scenario (March), train-only models, MASKED histories (about 50 % of cells visible, official windows have raw
histories at about 76 %), labels = visible masked-layer cells only.  Analysis / decision support only (uses March labels of
the released masked layer for scoring; nothing is trained or tuned on them here).
Usage: ogmodel_pmarch_eval.py <name>=<arm[+arm...]> [...]   first entry = base.
   arm = 'ref_np' / 'ref_id' / 'ref_full' (results/firstcross/final/model_F03_<cols>255_500.txt) or an ogmodel arm with saved
   final models (data/t2local/cache/ogmodel/models/<arm>.json).  Arms whose name contains 'noT' are fed the no-slot-T rows.
   Optional per-step blend: name=armA+armB@w1,w2,w3,w4,w5,w6  (weight of armB per step, armA gets 1 - w)."""
import json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import ogmodel_lib as og

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
F = lab.F
entries = []
for a in sys.argv[1:]:
    name, rhs = a.split('=')
    sw = None
    if '@' in rhs:
        rhs, sw = rhs.split('@'); sw = [float(x) for x in sw.split(',')]
    entries.append((name, rhs.split('+'), sw))
arms = sorted({x for _, parts, _ in entries for x in parts})
loaded = {}
for a in arms:
    if a.startswith('ref_'):
        c = a[4:]
        loaded[a] = ({}, [(0, lgb.Booster(model_file=f'{fc.RES}/final/model_F03_{c}255_500.txt'))], lab.colset(c))
    else:
        loaded[a] = og.load_models(a)

res = []
for panel in lab.PANELS:
    for sname in og.PM_SETS:
        pms = {vt: og.pm_load(panel, sname, vt) for vt in ('T', 'noT')}
        if pms['T'] is None:
            continue
        P6 = {}
        for a in arms:
            spec, models, cols = loaded[a]
            pm = pms['noT' if 'noT' in a else 'T']
            P6[a] = og.pm_predict(pm, lambda X, spec=spec, models=models, cols=cols: og.predict_struct(spec, models, X, cols), cols)
        for name, parts, sw in entries:
            pm = pms['noT' if all('noT' in x for x in parts) else 'T']
            if sw is not None:
                w = np.asarray(sw, np.float32)[None, :, None]; p = (1 - w) * P6[parts[0]] + w * P6[parts[1]]
            else:
                p = sum(P6[x] for x in parts) / len(parts)
            df = lab.score_probs(pm.rc, p, arm=name, set_name=sname)
            df['vis'] = pm.rc.win.vis.to_numpy(); df['hist_vis'] = pm.rc.win.hist_vis.to_numpy(); df['runstart'] = pm.rc.win.runstart.to_numpy()
            res.append(df)
R = pd.concat(res, ignore_index=True)
tag = '_'.join(n for n, _, _ in entries)[:120]
R.to_pickle(f'{og.RES}/pmarch_{tag}.pkl')
base = entries[0][0]
print('== PSEUDO-MARCH (train-only models; masked histories; visible labels): aggregated IoU ==')
rows = []
for sname in og.PM_SETS:
    for name, _, _ in entries:
        g = R[(R.set == sname) & (R.arm == name)]
        es = lab.error_split(g)
        rows.append(dict(set=sname, arm=name, n=len(g), iou=lab.aggregate(g), plain=g.iou.mean(), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(g))},
                         missed_new=es['missed_new'], kept_cleared=es['kept_cleared'], other=es['other']))
S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(f'{og.RES}/pmarch_summary_{tag}.csv', index=False)
print(f'\n== paired deltas vs {base} (lab.paired_bootstrap, windows resampled within panel) ==')
rows = []
for sname in og.PM_SETS:
    A = R[(R.set == sname) & (R.arm == base)]
    for name, _, _ in entries[1:]:
        rows.append(og.delta_row(A, R[(R.set == sname) & (R.arm == name)], name, sname))
Dl = pd.DataFrame(rows); print(Dl.round(4).to_string(index=False)); Dl.to_csv(f'{og.RES}/pmarch_deltas_{tag}.csv', index=False)
