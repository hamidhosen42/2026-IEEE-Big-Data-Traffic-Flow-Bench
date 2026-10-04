"""Ongoing for the private month: does adding SAME-SCENARIO training windows (released March, observed labels) to the F03 models
help, beyond feeding the same-scenario prior to a prior-using model?  Temporal split inside March: training windows from March
days < CUT only (faithful candidates on the t%3 grid at weight 0.3 w, run-starts at weight w - the F03 recipe), evaluation on the
pseudo-March offlike_m / runstart_m windows of days >= CUT.  Models: LightGBM 255 leaves / lr 0.05 / 500 trees (lab scale).
Arms: np+id (production-like, train only) | npid+full_s4 (0.5 np+id + 0.5 full fed the blended prior; train only)
      | npB+idB (np / id trained with March rows) | npBidB+fullB (0.5 (npB+idB)/2 + 0.5 fullB, full trained with March rows
      carrying the blended prior of the March days before their day).  Decision support only (March labels)."""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
CUT = int(sys.argv[1]) if len(sys.argv) > 1 else 15
WM = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
K = 4
OUT = 'results/splitprior_og'; os.makedirs(OUT, exist_ok=True)
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
MD = f'{fc.RES}/final'
cols_all = lab.all_cols(); cols_np, cols_id, cols_full = lab.colset('np'), lab.colset('id'), lab.colset('full')
ixn = [cols_all.index(c) for c in cols_np]; ixi = [cols_all.index(c) for c in cols_id]; ixf = [cols_all.index(c) for c in cols_full]
JPF = [cols_full.index(c) for c in PRIOR3]
m_np = lgb.Booster(model_file=f'{MD}/model_F03_np255_500.txt')
m_id = lgb.Booster(model_file=f'{MD}/model_F03_id255_500.txt')
m_full = lgb.Booster(model_file=f'{MD}/model_F03_full255_500.txt')
PRM = lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0)


def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def select(Pv):
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy()
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                og[d, t] = (1.0 if u == 0 else i / u) <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og


def val_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


def cum_tables(Pv):
    Q = Pv['Q'].astype(np.float32); E = Pv['E'].astype(np.float32)

    def box(a):
        c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
        return c[:, 5:] - c[:, :-5]
    Q, E = box(Q), box(E)
    D = Q.shape[0]; wk = Pv['dow'] >= 5
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
    return cQ, cE


def blended_full(X, R, meta, Pv, prior_tr, cQ, cE):
    """full-column matrix with the three prior columns blended per row (h-dependent slots)."""
    Xf = np.ascontiguousarray(X[:, ixf])
    wv, lv, hv = meta[:, 0], meta[:, 1], meta[:, 2]
    d_, t_ = R['d'], R['t']
    cr = (Pv['dow'][d_] >= 5).astype(int)[wv]; dr = d_[wv]; tr_ = t_[wv]; sh = np.minimum(tr_ + hv, 287)
    trv = [prior_tr[cr, sh, lv], prior_tr[cr, tr_, lv], prior_tr[2, sh, lv]]
    for j, (cc, ss) in enumerate(((cr, sh), (cr, tr_), (np.full_like(cr, 2), sh))):
        q = cQ[cc, dr, ss, lv]; ee = cE[cc, dr, ss, lv]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + K)
        Xf[:, JPF[j]] = (1 - a_) * trv[j] + a_ * sp
    return Xf


# ---- March rows: training (days < CUT: faith3 at 0.3, run-starts at 1.0) and evaluation (offlike_m / runstart_m, days >= CUT)
TR = dict(np=[], id=[], full=[], y=[], w=[]); EV = []
for panel in lab.PANELS:
    P, Pv = val_panel(panel)
    on, og = select(Pv)
    D = len(Pv['days'])
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    cQ, cE = cum_tables(Pv)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    trw = {}
    for a, b, r in zip(dd.tolist(), tt.tolist(), rs.tolist()):
        if a >= CUT:
            continue
        if r:
            trw[(a, b)] = 1.0
        elif b % 3 == 0:
            trw[(a, b)] = 0.3
    if trw:
        wl = sorted(trw)
        R = val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior_tr)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R)); assert list(call) == cols_all
        Xf = blended_full(X, R, meta, Pv, prior_tr, cQ, cE)
        ww = np.array([trw[w] for w in wl], np.float32)[meta[:, 0]] * WM
        TR['np'].append(np.ascontiguousarray(X[e][:, ixn])); TR['id'].append(np.ascontiguousarray(X[e][:, ixi])); TR['full'].append(Xf[e])
        TR['y'].append(y[e].astype(np.float32)); TR['w'].append(ww[e])
        del X, Xf
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    sets = {'offlike_m': sorted(w for w in s if w[0] >= CUT),
            'runstart_m': sorted((a, b) for a, b, r in zip(dd.tolist(), tt.tolist(), rs.tolist()) if r and a >= CUT)}
    for sname, wl in sets.items():
        if not wl:
            continue
        R = val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior_tr)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        Xf = blended_full(X, R, meta, Pv, prior_tr, cQ, cE)
        d_, t_ = R['d'], R['t']
        rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                   ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1), qlast=R['qlast'],
                                   win=pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONGOING, fold=0)), name=sname, panel=panel,
                                   cond=lab.ONGOING, L=R['L'])
        EV.append((rc, np.ascontiguousarray(X[:, ixn]), np.ascontiguousarray(X[:, ixi]), Xf, meta, R['W'], R['L']))
        del X
    lab.log(panel, 'March training windows', len(trw), '| eval', {k: len(v) for k, v in sets.items()})
    del cQ, cE; lab.drop_panel(panel); gc.collect()

Mn, Mi, Mf = np.concatenate(TR['np']), np.concatenate(TR['id']), np.concatenate(TR['full'])
My, Mw = np.concatenate(TR['y']), np.concatenate(TR['w'])
lab.log('March training rows', len(My), 'pos', round(float(My.mean()), 4), 'weight sum', round(float(Mw.sum()), 1))
sp = fc.ARMS['F03']
newm = {}
for nm, cset, Xm_ in (('npB', cols_np, Mn), ('idB', cols_id, Mi), ('fullB', cols_full, Mf)):
    t0 = time.time()
    X, y, w, info = fc.train_matrix(sp['train'], 4, cset, sp['wfn'])
    X = np.concatenate([X, Xm_]); y = np.concatenate([y, My]); w = np.concatenate([w, Mw])
    newm[nm] = fc.fit(X, y, w, PRM, 500, cset)
    lab.log(f'{nm} trained on {len(y)} rows ({len(My)} March) {time.time() - t0:.0f}s')
    del X, y, w; gc.collect()

res = []
for rc, Xn, Xi, Xf, meta, W, L in EV:
    P6 = {}
    for nm, mdl, XX in (('np', m_np, Xn), ('id', m_id, Xi), ('full_s4', m_full, Xf),
                        ('npB', newm['npB'], Xn), ('idB', newm['idB'], Xi), ('fullB', newm['fullB'], Xf)):
        p = np.zeros((W, F, L), np.float32); p[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(XX, num_threads=lab.THREADS); P6[nm] = p
    arms = {'np+id': 0.5 * (P6['np'] + P6['id']),
            'npid+full_s4': 0.25 * (P6['np'] + P6['id']) + 0.5 * P6['full_s4'],
            'npB+idB': 0.5 * (P6['npB'] + P6['idB']),
            'npBidB+fullB': 0.25 * (P6['npB'] + P6['idB']) + 0.5 * P6['fullB'],
            'npid+fullB': 0.25 * (P6['np'] + P6['id']) + 0.5 * P6['fullB'],
            'fullB': P6['fullB']}
    for nm, p in arms.items():
        res.append(lab.score_probs(rc, p, arm=nm, set_name=rc.name))
Rr = pd.concat(res, ignore_index=True); Rr.to_pickle(f'{OUT}/march_ongoing_marchtrain_{CUT}_{WM:g}.pkl')
for sname in ('offlike_m', 'runstart_m'):
    A0 = Rr[(Rr.set == sname) & (Rr.arm == 'np+id')]
    print(f'\n== {sname}, days >= {CUT}: n = {len(A0)} ==')
    for nm in dict.fromkeys(Rr.arm):
        B0 = Rr[(Rr.set == sname) & (Rr.arm == nm)]
        bt = lab.paired_bootstrap(A0, B0) if nm != 'np+id' else dict(diff=0, lo=0, hi=0)
        pa = A0.groupby('panel').iou.mean(); pb = B0.groupby('panel').iou.mean()
        print(f'{nm:14s} {lab.aggregate(B0):.4f}  vs np+id {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}]  panels+ {int((pb - pa > 1e-9).sum())}/{len(pa)}')
lab.log('DONE')
