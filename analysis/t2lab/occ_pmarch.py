"""occ on the NEW scenario seed (pseudo-March).  Usage: occ_pmarch.py og|on|og,on|dry
(dry = label-free footprint of the occ vs af np+id models on the released official ongoing windows)

Copied pattern of allfin_pmarch.py (val_panel / select / val_rows / rc_of / cum_tables), never edited there.
Models are trained on the TRAIN split only (all 273 days):
  ongoing  af  = cache/allfin/oof/af_all_{np,id}255/model_f4.txt       (all-finite rows, F03 recipe, lab scale)  = BASELINE
           occ = cache/occ/oof/occ_{np,id}255/model_f4.txt             (the same rows + 13 occupancy columns)
  onset    af  = cache/allfin/onset/cat_onset_noramp_all.cbm           (production-like CatBoost, NORAMP, all-finite rows)
           occ = the same fit with the occupancy columns (cache/occ/onset/cat_onset_noramp_occ.cbm)
Pseudo-March windows: the selections of allfin_pmarch.py on the masked validation month (labels / scoring cells = visible
eligible cells).  Features: all-finite masked layer (eligible + ineligible-valued cells) for the history and slot T; occupancy
from validation_Om (the masked layer = the deployment input there; co-masked with speed, checked), calibration k = 4 (all
train days).  Caveat (as allfin): the pseudo-March history is the masked layer (about 56 % of the cells visible) whereas the
real official windows carry the raw window history (about 83 %)."""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import lab
import occ_lib as oc
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

parts = sys.argv[1].split(',') if len(sys.argv) > 1 else ['og', 'on']
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
H, F, ONSET, ONGOING = lab.H, lab.F, lab.ONSET, lab.ONGOING
OUT = f'{oc.RES}/pmarch'; os.makedirs(OUT, exist_ok=True)
COLS = lab.all_cols(); CN = lab.colset('np'); CI = lab.colset('id')
KEEPC = [i for i, c in enumerate(COLS) if not c.startswith('ramp_')]
JP = [COLS.index(c) for c in lab.PRIOR3]
K_SP = 4


S1 = 'og4' in parts                                          # + seed-1 replicas -> bags of 4 (np, id x seeds 0, 1) = production fs4 shape
if S1:
    parts = [p for p in parts if p != 'og4'] + ['og']


def og_models():
    M = dict(af_np=lgb.Booster(model_file=f'{oc.AF}/oof/af_all_np255/model_f4.txt'), af_id=lgb.Booster(model_file=f'{oc.AF}/oof/af_all_id255/model_f4.txt'),
             occ_np=lgb.Booster(model_file=f'{oc.arm_dir("occ_np255")}/model_f4.txt'), occ_id=lgb.Booster(model_file=f'{oc.arm_dir("occ_id255")}/model_f4.txt'))
    if S1:
        for fam, arm in (('af', 'seed1'), ('occ', 'occs1')):
            for c in ('np', 'id'):
                M[f'{fam}_{c}1'] = lgb.Booster(model_file=f'{oc.arm_dir(f"{arm}_{c}255")}/model_f4.txt')
    return M


def on_models():
    base = CatBoostClassifier(); base.load_model(f'{oc.AF}/onset/cat_onset_noramp_all.cbm')
    mp = f'{oc.OC}/onset/cat_onset_noramp_occ.cbm'
    cat = CatBoostClassifier()
    if os.path.exists(mp):
        cat.load_model(mp)
    else:                                                    # allfin on_models recipe + the occupancy columns (k = 4 calibration)
        Xs, ys = [], []
        for p in lab.PANELS:
            rc = oc.rows_af(p, 'dense1500', ONSET); S = oc.Side(p, 'dense1500', ONSET)
            assert list(rc.cols) == COLS, p
            r = np.where(rc.e)[0]
            Xs.append(np.hstack([np.asarray(rc.X[r])[:, KEEPC], S.rows(rc.wi[r], rc.li[r], 4)])); ys.append(rc.y[r].astype(int))
        X = np.concatenate(Xs); y = np.concatenate(ys); del Xs
        sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
        cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0, allow_writing_files=False)
        t0 = time.time(); cat.fit(X[sub], y[sub])
        lab.log('occ catboost onset trained on', len(sub), 'rows', f'{time.time() - t0:.0f}s', 'pos rate', round(float(y.mean()), 4))
        tmp = f'{mp}.tmp{os.getpid()}'
        cat.save_model(tmp, format='cbm'); os.replace(tmp, mp); del X, y; gc.collect()
    return dict(af=base, occ=cat)


def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    x = np.load(f'{oc.XDIR}/t2x_{panel}.npz', allow_pickle=False)
    assert [str(a) for a in x['validation_days']] == list(days[o]) and [str(a) for a in x['links']] == P['links']
    Vx = x['validation_Vmx'].astype(np.float32); Fx = x['validation_Fmx'].astype(np.float32)
    assert not (np.isfinite(Vx) & Em).any()
    zo = np.load(f'{oc.ODIR}/t2occ_{panel}.npz', allow_pickle=False)
    assert [str(a) for a in zo['validation_days']] == list(days[o]) and [str(a) for a in zo['links']] == P['links'], 'occ day / link order'
    Om = zo['validation_Om'].astype(np.float32)
    V_all = np.where(E, V, Vx)
    assert np.array_equal(np.isfinite(Om), np.isfinite(V_all)), 'March occupancy not co-masked with the all-finite speed'
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32),
              V_all=V_all, F_all=np.where(E, Fq, Fx), Om=Om, x_share=float(np.isfinite(Vx).mean()))
    return P, Pv


def select(Pv):
    """allfin_pmarch.select (faithful ongoing / production onset candidates on the masked month; blank-touching dropped)."""
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
    return on, og, stable, pers


def val_rows(P, Pv, wins, prior, cond):
    """allfin_pmarch.val_rows in mode 'all' + the raw occupancy arrays of the same history / slot-T inputs."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    VV, FF = Pv['V_all'], Pv['F_all']
    Vh = VV[d[:, None], idx]; Fh = FF[d[:, None], idx]; VT = VV[d, t]
    Oh = Pv['Om'][d[:, None], idx]; OT = Pv['Om'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([cond] * W), qlast=qlast, W=W, L=L, rows=[],
             hist_vis=float(np.isfinite(Vh).mean()), T_vis=float(np.isfinite(VT).mean()), raw=oc.raw_occ(Oh, OT, Vh, Fh, P['lanes']))
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


def rc_of(Pv, R, name, panel, cond, extra=None):
    d, t = R['d'], R['t']
    win = pd.DataFrame(dict(d=d, t=t, cond=cond, fold=0, **(extra or {})))
    return types.SimpleNamespace(truth=np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1), ev=np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1),
                                 qlast=R['qlast'], win=win, name=name, panel=panel, cond=cond, L=R['L'])


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


def main():
    OG = og_models() if 'og' in parts else None
    ON = on_models() if 'on' in parts else None
    res = []; info = []
    ixn = [COLS.index(c) for c in CN]; ixi = [COLS.index(c) for c in CI]
    for panel in lab.PANELS:
        t1 = time.time()
        P, Pv = val_panel(panel)
        cal = np.load(f'{oc.OC}/calib/{panel}.npz'); ocr, cc = cal['ocrit5'][4], cal['c5'][4]
        on, og, stable, pers = select(Pv)
        D = len(Pv['days'])
        g_on, g_og = lab._glist(on), lab._glist(og); s_og = set(); s_on = set()
        for d0 in range(D):
            a, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s_on.update(a); s_og.update(b)
        prior = lab.prior_table(P, np.ones(len(P['days']), bool))
        inf = dict(panel=panel, x_share=Pv['x_share'], vis_elig=float(Pv['E'].mean()))
        if 'og' in parts:
            dd, tt = np.where(og); rs = ~og[dd, tt - 1]
            sets = {'offlike_m': sorted(s_og), 'runstart_m': sorted(zip(dd[rs].tolist(), tt[rs].tolist())),
                    'faith3_m': sorted((a, b) for a, b in zip(dd.tolist(), tt.tolist()) if b % 3 == 0)}
            rsset = set(zip(dd[rs].tolist(), tt[rs].tolist()))
            for sname, wl in sets.items():
                if not wl:
                    continue
                wins = [(a, b, ONGOING) for a, b in wl]
                R = val_rows(P, Pv, wins, prior, ONGOING)
                keep = lab.ongoing_keep(R)
                X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, keep)
                assert list(call) == COLS
                OF = oc.occ_feats(R['raw'], ocr, cc, meta[:, 0], meta[:, 1])
                inf[f'{sname}_n'] = len(wl); inf[f'{sname}_Tvis'] = R['T_vis']; inf[f'{sname}_histvis'] = R['hist_vis']
                rc = rc_of(Pv, R, sname, panel, ONGOING, extra=dict(runstart=np.array([(a, b) in rsset for a, b in wl], bool)))

                def pr(mdl, Xm):
                    p6 = np.zeros((R['W'], F, R['L']), np.float32)
                    p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(Xm, num_threads=lab.THREADS)
                    return p6
                Xo_n, Xo_i = np.hstack([X[:, ixn], OF]), np.hstack([X[:, ixi], OF])
                P6 = {'af_np': pr(OG['af_np'], X[:, ixn]), 'af_id': pr(OG['af_id'], X[:, ixi]),
                      'occ_np': pr(OG['occ_np'], Xo_n), 'occ_id': pr(OG['occ_id'], Xo_i)}
                if S1:
                    P6.update(af_np1=pr(OG['af_np1'], X[:, ixn]), af_id1=pr(OG['af_id1'], X[:, ixi]), occ_np1=pr(OG['occ_np1'], Xo_n), occ_id1=pr(OG['occ_id1'], Xo_i))
                for fam in ('af', 'occ'):
                    arms = [('np', P6[f'{fam}_np']), ('id', P6[f'{fam}_id']), ('npid', 0.5 * (P6[f'{fam}_np'] + P6[f'{fam}_id']))]
                    if S1:
                        arms += [('npid1', 0.5 * (P6[f'{fam}_np1'] + P6[f'{fam}_id1'])),
                                 ('bag4', 0.25 * (P6[f'{fam}_np'] + P6[f'{fam}_id'] + P6[f'{fam}_np1'] + P6[f'{fam}_id1']))]
                    for nm, p6 in arms:
                        res.append(lab.score_probs(rc, p6, arm=f'{fam}_{nm}', set_name=sname))
                del X, OF; gc.collect()
        if 'on' in parts and s_on:
            wl = sorted(s_on); wins = [(a, b, ONSET) for a, b in wl]
            cQ, cE = cum_tables(Pv)
            R = val_rows(P, Pv, wins, prior, ONSET)
            X, y, e, meta, call = lab.to_matrix(R, ONSET, None, None)
            assert list(call) == COLS
            wi, li = meta[:, 0], meta[:, 1]
            OF = oc.occ_feats(R['raw'], ocr, cc, wi, li)
            d, t = R['d'], R['t']; cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
            c_r, d_r, s6_r, t_r = cls[wi], d[wi], s6[wi], t[wi]
            Xs = X.copy()
            for j, (cc_, ss) in enumerate(((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r))):
                q = cQ[cc_, d_r, ss, li]; ee = cE[cc_, d_r, ss, li]
                sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + K_SP)
                Xs[:, JP[j]] = (1 - a_) * X[:, JP[j]] + a_ * sp
            rc = rc_of(Pv, R, 'march_on', panel, ONSET)
            inf['on_n'] = len(wl)
            for fam in ('af', 'occ'):
                for nm, Xv in (('trainprior', X), ('sp4', Xs)):
                    Xm = Xv[:, KEEPC] if fam == 'af' else np.hstack([Xv[:, KEEPC], OF])
                    p = ON[fam].predict_proba(Xm)[:, 1]
                    p6 = np.zeros((R['W'], F, R['L']), np.float32); p6[wi, F - 1, li] = p
                    res.append(lab.score_probs(rc, p6, arm=f'{fam}_{nm}', set_name='march_on'))
            del cQ, cE
        info.append(inf)
        lab.log(panel, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in inf.items()}), f'{time.time() - t1:.0f}s')
        lab.drop_panel(panel); gc.collect()
    tag = '_'.join(parts) + ('4' if S1 else '')
    R_ = pd.concat(res, ignore_index=True)
    R_.to_pickle(f'{OUT}/pmarch_{tag}.pkl')
    pd.DataFrame(info).to_csv(f'{OUT}/pmarch_info_{tag}.csv', index=False)
    rows = []
    for sname in dict.fromkeys(R_.set):
        for a in dict.fromkeys(R_[R_.set == sname].arm):
            g = R_[(R_.set == sname) & (R_.arm == a)]
            rows.append(dict(set=sname, arm=a, n=len(g), iou=lab.aggregate(g), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(g))}))
    S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(f'{OUT}/summary_{tag}.csv', index=False)
    print('\n== PSEUDO-MARCH paired deltas occ vs af (lab.paired_bootstrap) ==')
    out = []
    for sname in dict.fromkeys(R_.set):
        g = R_[R_.set == sname]
        fams = (('np', 'id', 'npid') + (('npid1', 'bag4') if S1 else ())) if sname != 'march_on' else ('trainprior', 'sp4')
        for nm in fams:
            A = g[g.arm == f'af_{nm}']; B = g[g.arm == f'occ_{nm}']
            r = oc.compare(A, B, f'{sname} occ_{nm} vs af_{nm}'); r.update(set=sname, arm=nm, wtype='all'); out.append(r); print(oc.fmt(r))
            if S1 and nm == 'npid1':                                   # seed noise: the same configuration, seed 1 vs seed 0
                for fam in ('af', 'occ'):
                    r = oc.compare(g[g.arm == f'{fam}_npid'], g[g.arm == f'{fam}_npid1'], f'{sname} NOISE {fam}_npid1 vs {fam}_npid')
                    r.update(set=sname, arm=f'noise_{fam}', wtype='all'); out.append(r); print(oc.fmt(r))
            if sname in ('offlike_m', 'faith3_m') and 'runstart' in A:
                for flag, wt in ((True, 'runstart'), (False, 'midrun')):
                    r = oc.compare(A[A.runstart == flag], B[B.runstart == flag], f'{sname} occ_{nm} vs af_{nm} [{wt}]')
                    r.update(set=sname, arm=nm, wtype=wt); out.append(r); print(oc.fmt(r))
    pd.DataFrame(out).to_csv(f'{OUT}/deltas_{tag}.csv', index=False)
    pp = (('offlike_m', 'af_bag4', 'occ_bag4'), ('runstart_m', 'af_bag4', 'occ_bag4')) if S1 else \
        (('offlike_m', 'af_npid', 'occ_npid'), ('runstart_m', 'af_npid', 'occ_npid'), ('march_on', 'af_sp4', 'occ_sp4'))
    for sname, a, b in pp:
        A = R_[(R_.set == sname) & (R_.arm == a)]; B = R_[(R_.set == sname) & (R_.arm == b)]
        if len(A) and len(B):
            a2, b2 = fc.paired(A, B)
            t_ = pd.DataFrame(dict(n=a2.groupby('panel').size(), base=a2.groupby('panel').iou.mean(), new=b2.groupby('panel').iou.mean())); t_['delta'] = t_.new - t_.base
            print(f'\nper panel ({sname}): {b} vs {a}'); print(t_.round(4).to_string())
            t_.to_csv(f'{OUT}/perpanel_{sname}_{b}.csv')


def split_rows_occ(P, split):
    """allfin_pmarch.split_rows_af (mode 'all') + the raw occupancy of the released official windows (window history / masked T)."""
    z = np.load(P['npz'], allow_pickle=False)
    wi = lab.window_index(P['panel'], split)
    org = pd.to_datetime(wi.forecast_origin, utc=True)
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
    Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32); VT = z[f'{split}_VT'].astype(np.float32)
    x = np.load(f'{oc.XDIR}/t2x_{P["panel"]}.npz', allow_pickle=False)
    assert [str(a) for a in x[f'{split}_window_id']] == wi.window_id.tolist()
    Vhx = x[f'{split}_Vhx'].astype(np.float32); Fhx = x[f'{split}_Fhx'].astype(np.float32); VTx = x[f'{split}_VTx'].astype(np.float32)
    el = np.isfinite(Vh); Fh = np.where(el, Fh, Fhx); Vh = np.where(el, Vh, Vhx); VT = np.where(np.isfinite(VT), VT, VTx)
    zo = np.load(f'{oc.ODIR}/t2occ_{P["panel"]}.npz', allow_pickle=False)
    assert [str(a) for a in zo[f'{split}_window_id']] == wi.window_id.tolist()
    Oh = zo[f'{split}_Oh'].astype(np.float32); OT = zo[f'{split}_OT'].astype(np.float32)
    assert np.array_equal(np.isfinite(Oh), np.isfinite(Vh)) and np.array_equal(np.isfinite(OT), np.isfinite(VT)), 'official occupancy not co-located'
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, tod, dow, prior, VT)
        wdays = sorted(wi.date.unique())
        mdays = [str(a) for a in z[f'{split}_days']]; ix = [mdays.index(d) for d in wdays]
        ONs = z[f'{split}_ON'][ix].astype(np.float32); OFFs = z[f'{split}_OFF'][ix].astype(np.float32)
        lab.add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, P['cap'])
    W, L = qlast.shape
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[], wi=wi, raw=oc.raw_occ(Oh, OT, Vh, Fh, P['lanes']))
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, dow, tod, h)
        R['rows'].append((h, feats, None, None))
    return R


def run_dry():
    """label-free footprint on the released official ONGOING windows: decoded np+id masks of the occ models vs the af models."""
    OG = og_models(); ixn = [COLS.index(c) for c in CN]; ixi = [COLS.index(c) for c in CI]; rows = []
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        cal = np.load(f'{oc.OC}/calib/{panel}.npz'); ocr, cc = cal['ocrit5'][4], cal['c5'][4]
        for split in ('validation', 'private'):
            R = split_rows_occ(P, split)
            X, _, _, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
            assert list(call) == COLS
            OF = oc.occ_feats(R['raw'], ocr, cc, meta[:, 0], meta[:, 1])
            M = {}
            for fam in ('af', 'occ'):
                p6 = np.zeros((R['W'], F, R['L']), np.float32)
                if len(X):
                    Xn = X[:, ixn] if fam == 'af' else np.hstack([X[:, ixn], OF]); Xi = X[:, ixi] if fam == 'af' else np.hstack([X[:, ixi], OF])
                    p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = 0.5 * (OG[f'{fam}_np'].predict(Xn, num_threads=lab.THREADS) + OG[f'{fam}_id'].predict(Xi, num_threads=lab.THREADS))
                M[fam] = p6
            for i in np.where(R['cond'] == ONGOING)[0]:
                a = lab.decode(M['af'][i].ravel()).reshape(F, -1); b = lab.decode(M['occ'][i].ravel()).reshape(F, -1)
                rows.append(dict(panel=panel, split=split, window_id=R['wi'].window_id.iloc[i], n_af=int(a.sum()), n_occ=int(b.sum()),
                                 changed=bool((a != b).any()), cells_changed=int((a != b).sum()), jacc=float((a & b).sum() / max((a | b).sum(), 1))))
        lab.drop_panel(panel)
    D = pd.DataFrame(rows); D.to_csv(f'{OUT}/dryrun_footprint.csv', index=False)
    g = D.groupby('split').agg(windows=('panel', 'size'), changed=('changed', 'sum'), cells_changed=('cells_changed', 'sum'),
                               cells_af=('n_af', 'sum'), cells_occ=('n_occ', 'sum'), mean_jacc=('jacc', 'mean'))
    print('\n== label-free footprint (official ongoing windows): occ np+id vs af np+id, decoded masks =='); print(g.round(4).to_string())


if 'og' in parts or 'on' in parts:
    main()
if 'dry' in parts:
    run_dry()
lab.log('DONE', parts)
