"""Ongoing under the new scenario seed: does a prior-using F03 model fed the SAME-SCENARIO prior beat the prior-free one?
Models (train split only, all days, k = 4): F03 np255 (production-like prior-free; results/firstcross/final) and F03 full255
(all columns incl. prior / prior_now / prior_alldays / tod / dow / pos ...; trained here with the all-train prior as production).
Windows: pseudo-March offlike_m / runstart_m (faithful selector on the masked March month, blank-touching windows dropped; as
exp_firstcross_pmarch.py).  Variants: np | full_tr (train prior) | full_s<k> (the three prior columns blended per cell with the
prior of the March days < d, smoothed over slots +-2, a = n/(n+k)).  Keep region: production ongoing_keep under the train prior
for every variant.  Labels: visible masked-layer cells.  Decision support only (may steer the private file / motivate a probe)."""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
OUT = 'results/splitprior_og'; os.makedirs(OUT, exist_ok=True)
KS = (2, 4, 8, 16)
REF = sys.argv[1] if len(sys.argv) > 1 else 'np+id'
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
MD = f'{fc.RES}/final'
cols_np, cols_full = lab.colset('np'), lab.colset('full')
m_np = lgb.Booster(model_file=f'{MD}/model_F03_np255_500.txt')
f_full = f'{MD}/model_F03_full255_500.txt'
if os.path.exists(f_full):
    m_full = lgb.Booster(model_file=f_full)
else:
    t0 = time.time(); sp = fc.ARMS['F03']
    X, y, w, info = fc.train_matrix(sp['train'], 4, cols_full, sp['wfn'])
    m_full = fc.fit(X, y, w, lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0), 500, cols_full)
    m_full.save_model(f_full); lab.log(f'F03 full255 trained: {len(X)} rows {time.time() - t0:.0f}s {json.dumps(info)}')
    del X, y, w; gc.collect()
cols_id = lab.colset('id')
f_id = f'{MD}/model_F03_id255_500.txt'
if os.path.exists(f_id):
    m_id = lgb.Booster(model_file=f_id)
else:
    t0 = time.time(); sp = fc.ARMS['F03']
    X, y, w, info = fc.train_matrix(sp['train'], 4, cols_id, sp['wfn'])
    m_id = fc.fit(X, y, w, lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0), 500, cols_id)
    m_id.save_model(f_id); lab.log(f'F03 id255 trained: {len(X)} rows {time.time() - t0:.0f}s')
    del X, y, w; gc.collect()
imp = pd.Series(m_full.feature_importance('gain'), index=cols_full); imp = imp / imp.sum()
print('full model gain share of prior columns:', imp[list(PRIOR3)].round(4).to_dict(), '| top:', imp.sort_values(ascending=False).head(8).round(3).to_dict())


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


res = []
for panel in lab.PANELS:
    t1 = time.time()
    P, Pv = val_panel(panel)
    on, og = select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    sets = {'offlike_m': sorted(s), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist()))}
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    cQ, cE = cum_tables(Pv)
    for sname, wl in sets.items():
        if not wl:
            continue
        R = val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior_tr)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        call = list(call)
        ixn = [call.index(c) for c in cols_np]; ixf = [call.index(c) for c in cols_full]; ixi = [call.index(c) for c in cols_id]
        jf = [cols_full.index(c) for c in PRIOR3]
        d_, t_ = R['d'], R['t']
        wv, lv, hv = meta[:, 0], meta[:, 1], meta[:, 2]
        cr = (Pv['dow'][d_] >= 5).astype(int)[wv]; dr = d_[wv]; tr_ = t_[wv]; sh = np.minimum(tr_ + hv, 287)
        trv = [prior_tr[cr, sh, lv], prior_tr[cr, tr_, lv], prior_tr[2, sh, lv]]
        spv, nv = [], []
        for cc, ss in ((cr, sh), (cr, tr_), (np.full_like(cr, 2), sh)):
            q = cQ[cc, dr, ss, lv]; ee = cE[cc, dr, ss, lv]
            spv.append(np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0)); nv.append(ee / 5.0)
        win = pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONGOING, fold=0))
        rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                   ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                   qlast=R['qlast'], win=win, name=sname, panel=panel, cond=lab.ONGOING, L=R['L'])
        Xf = np.ascontiguousarray(X[:, ixf])
        variants = {'np': (m_np, np.ascontiguousarray(X[:, ixn])), 'id': (m_id, np.ascontiguousarray(X[:, ixi])), 'full_tr': (m_full, Xf)}
        for k in KS:
            Xk = Xf.copy()
            for j, jp in enumerate(jf):
                a_ = nv[j] / (nv[j] + k)
                Xk[:, jp] = (1 - a_) * trv[j] + a_ * spv[j]
            variants[f'full_s{k}'] = (m_full, Xk)
        P6s = {}
        for name, (mdl, Xv) in variants.items():
            P6 = np.zeros((R['W'], F, R['L']), np.float32)
            P6[wv, hv - 1, lv] = mdl.predict(Xv, num_threads=lab.THREADS)
            P6s[name] = P6
        P6s['np+full_s4'] = 0.5 * (P6s['np'] + P6s['full_s4'])
        P6s['np+full_tr'] = 0.5 * (P6s['np'] + P6s['full_tr'])
        P6s['np+id'] = 0.5 * (P6s['np'] + P6s['id'])
        P6s['np+id+full_s4'] = (P6s['np'] + P6s['id'] + P6s['full_s4']) / 3
        P6s['np+id+full_s2'] = (P6s['np'] + P6s['id'] + P6s['full_s2']) / 3
        P6s['np+id+full_tr'] = (P6s['np'] + P6s['id'] + P6s['full_tr']) / 3
        P6s['npid+full_s4_50'] = 0.25 * (P6s['np'] + P6s['id']) + 0.5 * P6s['full_s4']
        for name, P6 in P6s.items():
            res.append(lab.score_probs(rc, P6, arm=name, set_name=sname))
        del X, Xf, variants; gc.collect()
    lab.log(panel, {k: len(v) for k, v in sets.items()}, f'{time.time() - t1:.0f}s')
    del cQ, cE; lab.drop_panel(panel); gc.collect()

Rr = pd.concat(res, ignore_index=True); Rr.to_pickle(f'{OUT}/march_ongoing_variants.pkl')
Rr = Rr[Rr.arm.isin(['np', 'id', 'np+id', 'full_tr', 'full_s2', 'full_s4', 'np+full_tr', 'np+full_s4', 'np+id+full_tr', 'np+id+full_s2', 'np+id+full_s4', 'npid+full_s4_50'])]
Rr['dbin'] = pd.cut(Rr.d, [-1, 4, 14, 40], labels=['d1-5', 'd6-15', 'd16+'])
for sname in ('offlike_m', 'runstart_m'):
    print(f'\n== {sname}: pseudo-March ongoing IoU (window -> panel -> family -> mean) ==')
    rows = []
    for name in dict.fromkeys(Rr.arm):
        r = dict(arm=name)
        for b in ('all', 'd1-5', 'd6-15', 'd16+'):
            g = Rr[(Rr.set == sname) & (Rr.arm == name) & ((Rr.dbin == b) if b != 'all' else True)]
            r[b] = lab.aggregate(g); r[f'n_{b}'] = len(g)
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    A0 = Rr[(Rr.set == sname) & (Rr.arm == REF)]
    out = []
    for name in dict.fromkeys(Rr.arm):
        if name == REF:
            continue
        B0 = Rr[(Rr.set == sname) & (Rr.arm == name)]
        r = dict(arm=name)
        for b in ('all', 'd16+'):
            A = A0 if b == 'all' else A0[A0.dbin == b]; B = B0 if b == 'all' else B0[B0.dbin == b]
            bt = lab.paired_bootstrap(A.drop(columns='dbin'), B.drop(columns='dbin'))
            r[f'{b}_diff'] = bt['diff']; r[f'{b}_lo'] = bt['lo']; r[f'{b}_hi'] = bt['hi']
        pa = A0.groupby('panel').iou.mean(); pb = B0.groupby('panel').iou.mean(); r['panels_pos'] = f'{int((pb - pa > 0).sum())}/{len(pa)}'
        out.append(r)
    print(f'-- paired vs {REF} ({sname})'); print(pd.DataFrame(out).round(4).to_string(index=False))
lab.log('DONE')
