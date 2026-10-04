"""pvog (light, label-free): the released official ongoing windows (validation = March, private = April) in both regimes.
For every official ongoing window the RAW window_history is released, and the masked layer also holds the same 12 history
slots (only T+5..T+90 is blanked).  So each window's features can be computed twice: raw history (deployment) and masked
history (the regime of the pseudo-March evaluation and of the kernel's March training rows); slot T is masked in both.
No labels exist for these windows and none are used: the script measures footprints only
  * Jaccard of the decoded masks between recipes (a) R:np+id, (b) R:npid+full_s4, (c) RB:npid+full, per regime,
  * Jaccard of the decoded masks of one recipe between its raw-history and masked-history inputs (regime sensitivity).
Scenario prior for the blended members: released masked days strictly before the window's day (March, then April)."""
import sys, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import pvog_lib as pv

H, F = lab.H, lab.F
TAGB = sys.argv[1] if len(sys.argv) > 1 else 'R_15_5'
M = {'R_np': lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_np255_500.txt'), 'R_id': lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_id255_500.txt'),
     'R_full': lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_full255_500.txt')}
import os
HAVE_B = all(os.path.exists(f'{pv.RES}/models/model_{TAGB}_{cs}B.txt') for cs in ('np', 'id', 'full'))
RECIPES = {'a': {'R_np': 0.5, 'R_id': 0.5}, 'b': {'R_np': 0.25, 'R_id': 0.25, 'R_full': 0.5}}
if HAVE_B:                                             # March-trained lab models of pvog_marchtrain.py (exp 1)
    for cs in ('np', 'id', 'full'):
        M[f'RB_{cs}'] = lgb.Booster(model_file=f'{pv.RES}/models/model_{TAGB}_{cs}B.txt')
    RECIPES.update({'c': {'RB_np': 0.25, 'RB_id': 0.25, 'RB_full': 0.5}, 'c_npid': {'RB_np': 0.5, 'RB_id': 0.5}})
PAIRS = [('a', 'b')] + ([('a', 'c'), ('b', 'c'), ('a', 'c_npid')] if HAVE_B else [])
CS = {'R_np': 'np', 'R_id': 'id', 'R_full': 'full', 'RB_np': 'np', 'RB_id': 'id', 'RB_full': 'full'}


def jac(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


rows = []
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    _, Pm = pv.val_panel(panel, 'validation'); _, Pa = pv.val_panel(panel, 'private')
    days = Pm['days'] + Pa['days']; assert days == sorted(days)
    Pc = {k: np.concatenate([Pm[k], Pa[k]]) for k in ('V', 'Fq', 'E', 'Q', 'Em', 'ON', 'OFF', 'dow')}
    cQ, cE = pv.cum_tables(Pc)
    for split in ('validation', 'private'):
        wi = lab.window_index(panel, split)
        ts = pd.to_datetime(wi.forecast_origin, utc=True)
        dd = np.array([days.index(x) for x in ts.dt.strftime('%Y-%m-%d')]); tt = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        og = np.where((wi.condition == lab.ONGOING).to_numpy())[0]
        R1 = lab.split_rows(P, split)                                                 # raw history (production inference path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            R2 = pv.cube_rows(P, list(zip(dd.tolist(), tt.tolist())), prior, Pc, Pc, Pc['dow'], Pc['ON'], Pc['OFF'], vt='cube')   # masked history
        R2['cond'] = R1['cond']
        z = np.load(P['npz'], allow_pickle=False)
        Vh = z[f'{split}_Vh'].astype(np.float32); idx = tt[:, None] + np.arange(-H, 0)[None]; Vmh = Pc['V'][dd[:, None], idx]
        sub = np.isfinite(Vmh); agree = np.abs(Vh[sub] - Vmh[sub]).max() if sub.any() else 0.0
        masks = {}
        for reg, R in (('raw', R1), ('masked', R2)):
            Rd = dict(R); Rd['d'] = dd
            X, _, _, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
            Xf = pv.blended_full(X, Rd, meta, Pc['dow'], prior, cQ, cE)
            P6 = {}
            for nm, mdl in M.items():
                XX = Xf if CS[nm] == 'full' else np.ascontiguousarray(X[:, pv.IX[CS[nm]]])
                p = np.zeros((R['W'], F, R['L']), np.float32); p[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(XX, num_threads=lab.THREADS); P6[nm] = p
            for rn, wts in RECIPES.items():
                pr = sum(w * P6[m] for m, w in wts.items())
                masks[(reg, rn)] = {i: lab.decode(pr[i].ravel()).reshape(F, -1) for i in og}
        for i in og:
            r = dict(panel=panel, family=panel.rsplit('_', 1)[0], split=split, w=int(i), d=int(dd[i]), hist_raw=float(np.isfinite(Vh[i]).mean()),
                     hist_masked=float(np.isfinite(Vmh[i]).mean()), nq_raw=int(R1['qlast'][i].sum()), nq_masked=int(R2['qlast'][i].sum()), max_abs_diff_common=float(agree))
            for reg in ('raw', 'masked'):
                for x, y in PAIRS:
                    r[f'J_{x}{y}_{reg}'] = jac(masks[(reg, x)][i], masks[(reg, y)][i])
                for rn in RECIPES:
                    r[f'n_{rn}_{reg}'] = int(masks[(reg, rn)][i].sum())
            for rn in RECIPES:
                r[f'J_rawmask_{rn}'] = jac(masks[('raw', rn)][i], masks[('masked', rn)][i])
            rows.append(r)
    lab.drop_panel(panel)
D = pd.DataFrame(rows); D.to_pickle(f'{pv.RES}/official_footprint_{TAGB}' + ('' if HAVE_B else '_ab') + '.pkl')
print('history finite share: raw', round(D.hist_raw.mean(), 3), 'masked layer', round(D.hist_masked.mean(), 3), '| max abs speed diff on common cells', D.max_abs_diff_common.max())
cols = [c for c in D.columns if c.startswith('J_') or c.startswith('n_')]
T = D.groupby('split')[cols].mean().T
print(T.round(4).to_string())
print('\nwindows with any difference between recipes (raw regime):')
for x, y in [p for p in PAIRS if p[1] != 'c_npid']:
    print(f'  {x} vs {y}:', D.groupby('split').apply(lambda g: f'{int((g[f"J_{x}{y}_raw"] < 1).sum())}/{len(g)}').to_dict())
