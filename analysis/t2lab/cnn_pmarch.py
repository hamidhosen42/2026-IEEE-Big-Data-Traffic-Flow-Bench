"""PSEUDO-MARCH check of the cnn ongoing member (new-scenario regime, MASKED histories).

Models: trained on the TRAIN split only (cnn: data/t2local/cache/cnn/<tag>/model_f4.pt = all train windows;
trees: results/firstcross/final/model_F03_{np,id}255_500.txt, reference = mean of the two).
Windows: offlike_m / runstart_m selected inside the released validation month exactly as exp_firstcross_pmarch.py
(val_panel / select / val_rows are copied from that script; blank-touching windows dropped, so no official window's hidden
horizon is involved).  Features of both model families: masked-layer history [T-60, T), the masked slot-T value as released,
ramp flows < T; keep region = production ongoing_keep on those features (all-train prior).  Labels: visible masked-layer
cells of T+5..T+30 -- March labels are used for SCORING only (analysis; nothing here is trained or tuned on them).
What does not match deployment: histories come from the masked layer (about 50 % of cells) instead of the raw window
history (about 76 %), and labels are the visible cells only.
Usage (through run_heavy): cnn_pmarch.py <tag[,tag...]> [name]"""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import lab
import cnn_lib as C
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

tags = sys.argv[1].split(',')
name = sys.argv[2] if len(sys.argv) > 2 else '+'.join(tags)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
H, F = lab.H, lab.F
_done = f'{C.RES}/pmarch_{name}_deltas.csv'
if os.path.exists(_done) and not os.environ.get('CNN_FORCE') and all(os.path.getmtime(_done) > os.path.getmtime(f'{C.CDIR}/{tg}/model_f4.pt') for tg in tags):
    print(f'cnn_pmarch {name}: results already on disk ({_done}); set CNN_FORCE=1 to recompute')
    sys.exit(0)
MD = f'{fc.RES}/final'
trees = {c: (lgb.Booster(model_file=f'{MD}/model_F03_{c}255_500.txt'), lab.colset(c)) for c in ('np', 'id')}
cnns = []
for tg in tags:
    cfg = json.load(open(f'{C.CDIR}/{tg}/config.json'))
    m = C.Net(c=cfg['c'], dils=tuple(int(x) for x in cfg['dils'].split(',')), drop=cfg['drop'])
    m.load_state_dict(torch.load(f'{C.CDIR}/{tg}/model_f4.pt', map_location='cpu')); m.eval(); cnns.append(m)
WS = (0.1, 0.2, 0.3, 0.4, 0.5)


def val_panel(panel):                                   # copied from exp_firstcross_pmarch.py
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def select(Pv):                                         # copied from exp_firstcross_pmarch.py
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy(); pers = np.full((D, 288), np.nan); stable = og.copy()
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
                pers[d, t] = 1.0 if u == 0 else i / u
                og[d, t] = pers[d, t] <= 0.9; stable[d, t] = pers[d, t] > 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, stable, pers, blank


def val_rows(P, Pv, wins, prior):                       # copied from exp_firstcross_pmarch.py
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


@torch.no_grad()
def cnn_predict(P, Pv, d, t, keep, qlast_ref):
    """mean probability of the final cnn models on masked-month windows; slot T as released (no extra simulation)."""
    vr, fr, vT = C.hist_arrays(P, Pv['V'], Pv['Fq'], d, t)
    ramp = C.ramp_arrays(Pv['ON'], Pv['OFF'], d, t, P['cap'])
    static = torch.from_numpy(C.static_channels(P))
    out = np.zeros((len(d), F, len(P['links'])), np.float32)
    for j in range(0, len(d), 128):
        sl = slice(j, j + 128)
        x, ql = C.assemble(torch.from_numpy(vr[sl]), torch.from_numpy(fr[sl]), torch.from_numpy(vT[sl]), torch.from_numpy(ramp[sl]), static, None)
        assert np.array_equal(ql.numpy().astype(bool), qlast_ref[sl]), 'qlast differs from the production features'
        out[sl] = sum(torch.sigmoid(m(x, ql)).numpy() for m in cnns) / len(cnns)
    return out * keep[:, None, :]


res = []; info = []; corr = []
for panel in C.PANELS:
    t1 = time.time()
    P, Pv = val_panel(panel)
    on, og, stable, pers, blank = select(Pv)
    D = len(Pv['days'])
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(D):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    sets = {'offlike_m': sorted(s), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist()))}
    vis_day = Pv['E'].sum((1, 2)) / np.maximum(Pv['Em'].sum((1, 2)), 1)
    info.append(dict(panel=panel, days=D, og_cand=int(og.sum()), **{k: len(v) for k, v in sets.items()}, visible_share=float(Pv['E'].sum() / Pv['Em'].sum())))
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    for sname, wl in sets.items():
        if not wl:
            continue
        R = val_rows(P, Pv, [(a, b, lab.ONGOING) for a, b in wl], prior)
        keep = lab.ongoing_keep(R)
        X, y, e, meta, call = lab.to_matrix(R, lab.ONGOING, None, keep)
        d_, t_ = R['d'], R['t']
        win = pd.DataFrame(dict(d=d_, t=t_, cond=lab.ONGOING, fold=0, pers_fill=pers[d_, t_], runstart=og[d_, t_] & ~og[d_, t_ - 1], faithful=og[d_, t_],
                                vis=vis_day[d_], hist_vis=[float(Pv['E'][a, b - H:b].sum() / max(Pv['Em'][a, b - H:b].sum(), 1)) for a, b in zip(d_, t_)]))
        rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                   qlast=R['qlast'], win=win, name=sname, panel=panel, cond=lab.ONGOING, L=R['L'])
        P6s = {}
        Pt = np.zeros((R['W'], F, R['L']), np.float32)
        for c, (mdl, cols_) in trees.items():
            ix = [list(call).index(x) for x in cols_]
            p6 = np.zeros_like(Pt); p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(X[:, ix], num_threads=lab.THREADS)
            P6s[f'tree_{c}'] = p6; Pt += 0.5 * p6
        P6s['trees'] = Pt
        Pc = cnn_predict(P, Pv, d_, t_, keep, R['qlast'])
        P6s['cnn'] = Pc
        for w in WS:
            P6s[f'blend{int(round(w * 100))}'] = w * Pc + (1 - w) * Pt
        P6s['persist'] = np.repeat(R['qlast'][:, None, :], F, 1).astype(np.float32) + 1e-6
        for a, P6 in P6s.items():
            df = lab.score_probs(rc, P6, arm=a, set_name=sname)
            df['vis'] = win.vis.to_numpy(); df['hist_vis'] = win.hist_vis.to_numpy()
            res.append(df)
        m = (Pt > 0) | (Pc > 0)
        corr.append(dict(set=sname, panel=panel, corr_cnn_trees=float(np.corrcoef(Pc[m], Pt[m])[0, 1]),
                         corr_np_id=float(np.corrcoef(P6s['tree_np'][m], P6s['tree_id'][m])[0, 1])))
        del X; gc.collect()
    lab.log(panel, json.dumps(info[-1]), f'{time.time() - t1:.0f}s')
    lab.drop_panel(panel)
R_ = pd.concat(res, ignore_index=True)
R_.to_pickle(f'{C.RES}/pmarch_{name}_windows.pkl'); pd.DataFrame(info).to_csv(f'{C.RES}/pmarch_{name}_info.csv', index=False)
print(pd.DataFrame(info).to_string(index=False))
allarms = list(dict.fromkeys(R_.arm)); base = 'trees'
print(f'\n== PSEUDO-MARCH ({name}; masked histories, visible-cell labels): aggregated IoU ==')
rows = []
for sname in ('offlike_m', 'runstart_m'):
    for a in allarms:
        g = R_[(R_.set == sname) & (R_.arm == a)]
        if not len(g):
            continue
        es = lab.error_split(g)
        rows.append(dict(set=sname, arm=a, n=len(g), iou=lab.aggregate(g), plain=g.iou.mean(), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(g))},
                         missed_new=es['missed_new'], kept_cleared=es['kept_cleared'], other=es['other']))
S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(f'{C.RES}/pmarch_{name}_summary.csv', index=False)
print(f'\n== PSEUDO-MARCH: paired deltas vs {base} (paired bootstrap n=2000) ==')
rows = []
for sname in ('offlike_m', 'runstart_m'):
    A = R_[(R_.set == sname) & (R_.arm == base)]
    for a in allarms:
        if a == base:
            continue
        B = R_[(R_.set == sname) & (R_.arm == a)]
        b = lab.paired_bootstrap(A, B); pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
        r = dict(set=sname, arm=a, **b, panels_pos=f'{int((pb - pa > 0).sum())}/{len(pa)}')
        for lo, hi, nm in ((0.7, 2, 'vis_hi'), (0.55, 0.7, 'vis_mid'), (0, 0.55, 'vis_lo')):
            m = (A.vis >= lo) & (A.vis < hi)
            if m.any():
                a2, b2 = fc.paired(A[m], B)
                r[nm] = lab.aggregate(b2) - lab.aggregate(a2); r[nm + '_n'] = len(a2)
        rows.append(r)
Dl = pd.DataFrame(rows); print(Dl.round(4).to_string(index=False)); Dl.to_csv(f'{C.RES}/pmarch_{name}_deltas.csv', index=False)
for sname in ('offlike_m', 'runstart_m'):
    g = R_[R_.set == sname]
    pp = g.pivot_table(index='panel', columns='arm', values='iou', aggfunc='mean')[['trees', 'cnn', 'blend20', 'blend30', 'blend50']]
    pp['n'] = g[g.arm == 'trees'].groupby('panel').size()
    for a in ('cnn', 'blend20', 'blend30', 'blend50'):
        pp[f'd_{a}'] = pp[a] - pp['trees']
    print(f'\nper panel ({sname})'); print(pp.round(4).to_string()); pp.to_csv(f'{C.RES}/pmarch_{name}_perpanel_{sname}.csv')
Cr = pd.DataFrame(corr); print('\nprobability correlation (cells with non-zero probability)'); print(Cr.groupby('set')[['corr_cnn_trees', 'corr_np_id']].mean().round(4).to_string())
lab.log('PMARCH DONE')
