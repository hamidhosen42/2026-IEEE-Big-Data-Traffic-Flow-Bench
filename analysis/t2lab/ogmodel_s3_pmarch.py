"""TRACK ogmodel S3 (pseudo-March part, light): calibration by step of the final F03 np255 / id255 models on the pseudo-March
windows (new scenario, masked histories, visible labels) and transfer test of the per-step Platt correction fitted on the
train official-like OOF cells (results/ogmodel/s3_calfull_<name>_platt_h.json; decode unchanged)."""
import json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import ogmodel_lib as og

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
F = lab.F


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def platt_fit(p, y):
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogisticRegression(C=1e6, max_iter=200).fit(lg(p)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


for name, kinds in (('np', ['np']), ('npid', ['np', 'id'])):
    models = [(lgb.Booster(model_file=f'{fc.RES}/final/model_F03_{c}255_500.txt'), lab.colset(c)) for c in kinds]
    cal = {int(eval(k)[0]): v for k, v in json.load(open(f'{og.RES}/s3_calfull_{name}_platt_h.json')).items()}
    calq = {eval(k): v for k, v in json.load(open(f'{og.RES}/s3_calfull_{name}_platt_hxq.json')).items()}
    for sname in og.PM_SETS:
        cells = []; base = []; rec = []; recq = []
        for panel in lab.PANELS:
            pm = og.pm_load(panel, sname, 'T')
            if pm is None:
                continue
            p = sum(m.predict(np.ascontiguousarray(pm.X[:, [pm.cols.index(c) for c in cols]]), num_threads=lab.THREADS) for m, cols in models) / len(models)
            w_, l_, h_ = pm.meta[:, 0], pm.meta[:, 1], pm.meta[:, 2]
            q = pm.rc.qlast[w_, l_]
            P6 = np.zeros((pm.W, F, pm.L), np.float32); P6[w_, h_ - 1, l_] = p
            base.append(lab.score_probs(pm.rc, P6, arm='base', set_name=sname))
            pc = p.copy(); pq = p.copy()
            for h in range(1, F + 1):
                k = h_ == h; a, b = cal[h]; pc[k] = 1 / (1 + np.exp(-(a * lg(p[k]) + b)))
                for qq in (False, True):
                    k2 = k & (q == qq)
                    if (h, qq) in calq and k2.any():
                        a, b = calq[(h, qq)]; pq[k2] = 1 / (1 + np.exp(-(a * lg(p[k2]) + b)))
            P6c = np.zeros_like(P6); P6c[w_, h_ - 1, l_] = pc; rec.append(lab.score_probs(pm.rc, P6c, arm='platt_h', set_name=sname))
            P6q = np.zeros_like(P6); P6q[w_, h_ - 1, l_] = pq; recq.append(lab.score_probs(pm.rc, P6q, arm='platt_hxq', set_name=sname))
            e = pm.e
            cells.append(pd.DataFrame(dict(panel=panel, h=h_[e], p=p[e], y=pm.y[e], q=q[e])))
        C = pd.concat(cells, ignore_index=True); B = pd.concat(base, ignore_index=True)
        t = []
        for h in range(1, F + 1):
            for qn, m in (('all', np.ones(len(C), bool)), ('queued at T', C.q.to_numpy()), ('free at T', ~C.q.to_numpy())):
                g = C[(C.h == h).to_numpy() & m]; a, b = platt_fit(g.p.to_numpy(), g.y.to_numpy())
                pmn = g.groupby('panel').apply(lambda x: x.y.mean() - x.p.mean(), include_groups=False)
                t.append(dict(h=h, regime=qn, cells=len(g), mean_pred=g.p.mean(), observed=g.y.mean(), obs_minus_pred=g.y.mean() - g.p.mean(), platt_slope=a, platt_icpt=b,
                              panels_under=int((pmn > 0).sum())))
        T = pd.DataFrame(t); print(f'\n== {name} / {sname}: calibration by step (final train-only model on pseudo-March cells) ==')
        print(T.round(4).to_string(index=False)); T.to_csv(f'{og.RES}/s3_calibration_pmarch_{name}_{sname}.csv', index=False)
        rows = [og.delta_row(B, pd.concat(rec, ignore_index=True), 'platt_h (fitted on train offlike OOF)', sname),
                og.delta_row(B, pd.concat(recq, ignore_index=True), 'platt_hxq (fitted on train offlike OOF)', sname)]
        print(pd.DataFrame(rows).round(4).to_string(index=False))
