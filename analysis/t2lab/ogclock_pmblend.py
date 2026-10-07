"""TRACK ogclock: pseudo-March (offlike_m, runstart_m) for production-like np+id BLENDS and the deployable g72 switch.
Models (k = 4, train split only): base F03 np255 / id255 (results/firstcross/final), ck models np_all / id_all (results/ogclock/final),
F10 np255 (reference for the in-flight tfb-t2v24 g72 switch).  g72 flag of offlike_m windows: results/ogmodel/g72_pmarch.pkl
(replay-derived: the previous official-like pick is exactly 360 min earlier); runstart_m windows are scored as not flagged
(as in exp_g72_pmarch / ogmodel_regime).  Prediction only."""
import types
import numpy as np, pandas as pd, lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import ogclock_lib as ck

F = lab.F
M = {'base_np': (lgb.Booster(model_file=f'{fc.RES}/final/model_F03_np255_500.txt'), lab.colset('np'), []),
     'base_id': (lgb.Booster(model_file=f'{fc.RES}/final/model_F03_id255_500.txt'), lab.colset('id'), []),
     'ck_np': (lgb.Booster(model_file=f'{ck.RES}/final/model_np_all_255_500.txt'), lab.colset('np'), ck.CK_ALL),
     'ck_id': (lgb.Booster(model_file=f'{ck.RES}/final/model_id_all_255_500.txt'), lab.colset('id'), ck.CK_ALL),
     'F10_np': (lgb.Booster(model_file=f'{fc.RES}/final/model_F10_np255_500.txt'), lab.colset('np'), [])}
G = pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/ogmodel/g72_pmarch.pkl')
allc = lab.all_cols()
res = []
for panel in lab.PANELS:
    pm = ck.build_pm(panel)
    for s in ck.PM_SETS:
        z = pm[s]; meta = z['meta']; win = z['win']
        if not len(win):
            continue
        P6 = {}
        for nm, (m, pc, xc) in M.items():
            X = z['X'][:, [allc.index(c) for c in pc]]
            if xc:
                X = np.concatenate([X, z['FE'][meta[:, 0][:, None], meta[:, 1][:, None], np.array([ck.CK_ALL.index(c) for c in xc])[None, :]]], 1)
            p = np.zeros((len(win), F, z['qlast'].shape[1]), np.float32)
            p[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = m.predict(X, num_threads=lab.THREADS)
            P6[nm] = p
        if s == 'offlike_m':
            g = win[['d', 't']].assign(panel=panel).merge(G[['panel', 'd', 't', 'g72_any']], on=['panel', 'd', 't'], how='left')
            flag = g.g72_any.fillna(False).to_numpy().astype(bool)
        else:
            flag = np.zeros(len(win), bool)
        f3 = flag[:, None, None]
        P6['blend_base'] = 0.5 * (P6['base_np'] + P6['base_id'])
        P6['blend_ck'] = 0.5 * (P6['ck_np'] + P6['ck_id'])
        P6['sw_blend_ck'] = np.where(f3, P6['blend_ck'], P6['blend_base'])
        P6['sw_np_ck'] = np.where(f3, P6['ck_np'], P6['base_np'])
        P6['sw_np_F10'] = np.where(f3, P6['F10_np'], P6['base_np'])
        rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=win, name=s, panel=panel, cond=lab.ONGOING, L=z['qlast'].shape[1])
        for nm, p in P6.items():
            df = lab.score_probs(rc, p, arm=nm, set_name=s); df['g72'] = flag; df['wtype'] = win.wtype.to_numpy(); res.append(df)
R = pd.concat(res, ignore_index=True)
R.to_pickle(f'{ck.RES}/pmblend.pkl')
rows = []
for s in ck.PM_SETS:
    for base, arm in (('base_np', 'ck_np'), ('base_id', 'ck_id'), ('blend_base', 'blend_ck'), ('blend_base', 'sw_blend_ck'), ('base_np', 'sw_np_ck'),
                      ('base_np', 'sw_np_F10'), ('sw_np_F10', 'sw_np_ck')):
        A = R[(R.set == s) & (R.arm == base)].drop(columns=['g72', 'wtype']); B = R[(R.set == s) & (R.arm == arm)].drop(columns=['g72', 'wtype'])
        rows.append(ck.delta_row(A, B, f'{arm} vs {base}', s))
        if s == 'offlike_m':
            for nm in (True, False):
                a = R[(R.set == s) & (R.arm == base) & (R.g72 == nm)].drop(columns=['g72', 'wtype'])
                b = R[(R.set == s) & (R.arm == arm) & (R.g72 == nm)].drop(columns=['g72', 'wtype'])
                rows.append(ck.delta_row(a, b, f'{arm} vs {base}', f'{s}:g72={nm}'))
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(D.round(4).to_string(index=False))
D.to_csv(f'{ck.RES}/pmblend_deltas.csv', index=False)
lab.log('DONE')
