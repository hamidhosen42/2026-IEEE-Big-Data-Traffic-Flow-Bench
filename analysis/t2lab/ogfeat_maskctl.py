"""ogfeat CONFOUND control: does a feature arm's gain survive when the 60-min history is MASKED like the released masked layer?
Train official-like windows (held-out fold k, fold-k models from ogfeat_run.py) are re-featurised with every observed history cell
(speed and flow) hidden i.i.d. with probability 1 - keep (the Task-1 mask regimes hide 20 / 30 / 50 %); slot-T draw, labels, prior
and scoring are unchanged.  keep = 1.0 reproduces the raw OOF numbers.  Prediction only (no training).
Usage: ogfeat_maskctl.py <base arm> <arm[,arm...]> [folds=01] [keeps=1.0,0.7,0.5]"""
import os, sys, types, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import lab, ogfeat_lib as og

base = sys.argv[1]; arms = sys.argv[2].split(',')
kw = dict(a.split('=', 1) for a in sys.argv[3:])
folds = [int(c) for c in kw.get('folds', '01')]; keeps = [float(x) for x in kw.get('keeps', '1.0,0.7,0.5').split(',')]
pd.set_option('display.width', 250)
import json
cfg = {a: json.load(open(f'{og.ODIR}/arms/{a}/config.json')) for a in [base] + arms}
models = {(a, k): lgb.Booster(model_file=f'{og.ODIR}/arms/{a}/model_f{k}.txt') for a in cfg for k in folds}
res = []
for panel in lab.PANELS:
    P = lab.load_panel(panel); rc = lab.rows(panel, 'offlike')
    Vh0, Fh0, VT, d, t = og.train_inputs(P, rc.win)
    dw = P['dow'][d]
    for keep in keeps:
        rng = np.random.default_rng(1000 + int(keep * 100))
        hide = rng.random(Vh0.shape) >= keep
        Vh = np.where(hide, np.nan, Vh0); Fh = np.where(hide, np.nan, Fh0)
        for k in folds:
            widx = np.where(rc.win.fold.to_numpy() == k)[0]
            if not len(widx):
                continue
            prior = lab.prior_table(P, P['block'] != k)
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                Fe, qlast = lab.window_features(P, Vh[widx], Fh[widx], t[widx], dw[widx], prior, VT[widx])
                lab.add_ramp(Fe, P['ON'], P['OFF'], d[widx], t[widx], P['cap'])
                X1 = og.og_extra(P, Vh[widx], Fh[widx], VT[widx], Fe); X2 = og.og_extra2(P, Vh[widx], Fh[widx], VT[widx], Fe)
            W, L = qlast.shape
            R = dict(d=d[widx], t=t[widx], cond=np.array([lab.ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
            for h in range(1, lab.F + 1):
                feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
                feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, dw[widx], t[widx], h)
                R['rows'].append((h, feats, None, None))
            X, _, _, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
            FE1 = np.stack([X1[c] for c in og.static_names()], -1); FE2 = np.stack([X2[c] for c in og.static_names2()], -1)
            ns = types.SimpleNamespace(truth=rc.truth[widx], ev=rc.ev[widx], qlast=qlast, win=rc.win.iloc[widx].reset_index(drop=True), name='offlike',
                                       panel=panel, cond=lab.ONGOING, L=L)
            for a, c in cfg.items():
                ix = [list(call).index(x) for x in c['prod_cols']]
                Xa = X[:, ix]
                if c['static'] or c['step']:
                    Xa = np.concatenate([Xa, og.extra_matrix(og.FE2(FE1, FE2), meta[:, 0], meta[:, 1], meta[:, 2], c['static'], c['step'])], 1)
                P6 = np.zeros((W, lab.F, L), np.float32)
                P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = models[(a, k)].predict(Xa, num_threads=lab.THREADS)
                df = lab.score_probs(ns, P6, arm=a, set_name='offlike'); df['keep'] = keep
                res.append(df)
    lab.drop_panel(panel)
R_ = pd.concat(res, ignore_index=True)
rows = []
for keep in keeps:
    A = R_[(R_.arm == base) & (R_.keep == keep)]
    for a in arms:
        B = R_[(R_.arm == a) & (R_.keep == keep)]
        r = og.delta_row(A, B, a, f'offlike keep={keep}'); rows.append(r)
D = pd.DataFrame(rows)
print(f'== masked-history control on train official-like windows, folds {folds} (B - A vs {base}) ==')
print(D.round(4).to_string(index=False))
D.to_csv(f'{og.RES}/maskctl_{base}_{"_".join(arms)}.csv'[:200], index=False)
