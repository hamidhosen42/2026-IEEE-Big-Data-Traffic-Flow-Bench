"""allfin on the NEW scenario seed (pseudo-March) + label-free footprint on the released official windows.

Models are trained on the TRAIN split only (all 273 days):
  ongoing  base = results/firstcross/final/model_F03_{np,id}255_500.txt (eligible-only rows; production recipe at lab scale)
           af   = cache/allfin/oof/af_all_{np,id}255/model_f4.txt        (all-finite rows, same recipe)
  onset    base = results/splitprior/cat_onset_noramp.cbm (production-like CatBoost, NORAMP)
           af   = the same fit on the all-finite dense1500 onset rows
Pseudo-March windows: exactly the selections of exp_firstcross_pmarch.py (ongoing) and exp_onset_splitprior.py (onset) on the
masked validation month (selection, labels and scoring cells: visible ELIGIBLE cells, unchanged).
Features: 'elig' = masked-layer history + masked slot-T value (as before);  'all' = the same plus the ineligible-valued cells
of the masked layer (Vmx / Fmx; always visible).  NOTE: the pseudo-March history comes from the masked layer (about half of
the cells visible) whereas the real official windows have the raw window_history (76 % eligible + 7 % ineligible-valued), so
the pseudo-March gain of the extra cells is an UPPER-biased estimate for the history part; the slot-T part is faithful.
Arms: elig (base model, elig features), all (af model, all features), cross (base model, all features = no retrain),
      afelig (af model, elig features).
Part C: released validation / private official windows, label-free: how much do the decoded masks change (footprint)."""
import gc, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import lab
import allfin_lib as af
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

parts = sys.argv[1].split(',') if len(sys.argv) > 1 else ['og', 'on', 'dry']
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
H, F, ONSET, ONGOING = lab.H, lab.F, lab.ONSET, lab.ONGOING
OUT = f'{af.RES}/pmarch'; os.makedirs(OUT, exist_ok=True)
COLS = lab.all_cols(); CN = lab.colset('np'); CI = lab.colset('id')
KEEPC = [i for i, c in enumerate(COLS) if not c.startswith('ramp_')]
JP = [COLS.index(c) for c in lab.PRIOR3]
K_SP = 4                                                    # production same-scenario prior: smoothed, a = n / (n + 4)


# ---------------------------------------------------------------- models
def og_models():
    MD = f'{fc.RES}/final'
    return dict(base_np=lgb.Booster(model_file=f'{MD}/model_F03_np255_500.txt'), base_id=lgb.Booster(model_file=f'{MD}/model_F03_id255_500.txt'),
                af_np=lgb.Booster(model_file=f'{af.arm_dir("af_all_np255")}/model_f4.txt'), af_id=lgb.Booster(model_file=f'{af.arm_dir("af_all_id255")}/model_f4.txt'))


def on_models():
    base = CatBoostClassifier(); base.load_model('results/splitprior/cat_onset_noramp.cbm')
    mp = f'{af.AF}/onset/cat_onset_noramp_all.cbm'
    cat = CatBoostClassifier()
    if os.path.exists(mp):
        cat.load_model(mp)
    else:                                                    # same recipe as exp_onset_splitprior.py on the all-finite rows
        Xs, ys = [], []
        for p in lab.PANELS:
            rc = af.rows_af(p, 'dense1500', ONSET)
            assert list(rc.cols) == COLS, p
            r = np.where(rc.e)[0]
            Xs.append(np.asarray(rc.X[r])); ys.append(rc.y[r].astype(int))
        X = np.concatenate(Xs); y = np.concatenate(ys); del Xs
        sub = np.random.default_rng(0).choice(len(X), min(len(X), 1_200_000), replace=False)
        cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0, allow_writing_files=False)
        t0 = time.time(); cat.fit(X[sub][:, KEEPC], y[sub])
        lab.log('all-finite catboost onset trained on', len(sub), 'rows', f'{time.time() - t0:.0f}s', 'pos rate', round(float(y.mean()), 4))
        tmp = f'{mp}.tmp{os.getpid()}'
        cat.save_model(tmp, format='cbm'); os.replace(tmp, mp); del X, y; gc.collect()      # atomic: parallel workers may train it too
    return dict(base=base, af=cat)


# ---------------------------------------------------------------- masked validation month (+ ineligible-valued layer)
def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    x = np.load(f'{af.XDIR}/t2x_{panel}.npz', allow_pickle=False)
    assert [str(a) for a in x['validation_days']] == list(days[o]) and [str(a) for a in x['links']] == P['links']
    Vx = x['validation_Vmx'].astype(np.float32); Fx = x['validation_Fmx'].astype(np.float32)
    assert not (np.isfinite(Vx) & Em).any()
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32),
              V_all=np.where(E, V, Vx), F_all=np.where(E, Fq, Fx), x_share=float(np.isfinite(Vx).mean()))
    return P, Pv


def select(Pv):
    """exp_firstcross_pmarch.select (faithful ongoing / production onset candidates on the masked month; blank-touching dropped)."""
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


def val_rows(P, Pv, wins, prior, cond, mode):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    VV, FF = (Pv['V'], Pv['Fq']) if mode == 'elig' else (Pv['V_all'], Pv['F_all'])
    Vh = VV[d[:, None], idx]; Fh = FF[d[:, None], idx]; VT = VV[d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([cond] * W), qlast=qlast, W=W, L=L, rows=[],
             hist_vis=float(np.isfinite(Vh).mean()), T_vis=float(np.isfinite(VT).mean()))
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
    """smoothed (slots +-2) cumulative queued / visible counts over the March days < d by weekday class (exp_onset_splitprior)."""
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


# ---------------------------------------------------------------- A + B: pseudo-March
def run_pmarch():
    OG = og_models() if 'og' in parts else None
    ON = on_models() if 'on' in parts else None
    res = []; info = []
    for panel in lab.PANELS:
        t1 = time.time()
        P, Pv = val_panel(panel)
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
            for sname, wl in sets.items():
                if not wl:
                    continue
                wins = [(a, b, ONGOING) for a, b in wl]; P6 = {}
                for mode in ('elig', 'all'):
                    R = val_rows(P, Pv, wins, prior, ONGOING, mode)
                    keep = lab.ongoing_keep(R)
                    X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, keep)
                    assert list(call) == COLS
                    ixn = [COLS.index(c) for c in CN]; ixi = [COLS.index(c) for c in CI]
                    pos = int(sum((lab_ & ev_).sum() for _, _, lab_, ev_ in R['rows'])); pin = int(sum((lab_ & ev_ & keep).sum() for _, _, lab_, ev_ in R['rows']))
                    inf[f'{sname}_{mode}_keepcov'] = pin / max(pos, 1); inf[f'{sname}_{mode}_kept'] = float(keep.mean())
                    inf[f'{sname}_{mode}_histvis'] = R['hist_vis']; inf[f'{sname}_{mode}_Tvis'] = R['T_vis']; inf[f'{sname}_n'] = len(wl)
                    rc = rc_of(Pv, R, sname, panel, ONGOING)

                    def pr(mdl, ix):
                        p6 = np.zeros((R['W'], F, R['L']), np.float32)
                        p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(X[:, ix], num_threads=lab.THREADS)
                        return p6
                    for fam in ('base', 'af'):
                        pn, pi = pr(OG[f'{fam}_np'], ixn), pr(OG[f'{fam}_id'], ixi)
                        tag = {('base', 'elig'): 'elig', ('af', 'all'): 'all', ('base', 'all'): 'cross', ('af', 'elig'): 'afelig'}[(fam, mode)]
                        for nm, p6 in (('np', pn), ('id', pi), ('npid', 0.5 * (pn + pi))):
                            res.append(lab.score_probs(rc, p6, arm=f'{tag}_{nm}', set_name=sname))
                    if mode == 'elig':
                        res.append(lab.score_probs(rc, np.repeat(R['qlast'][:, None, :], F, 1).astype(np.float32) + 1e-6, arm='persist_elig', set_name=sname))
                    del X; gc.collect()
        if 'on' in parts and s_on:
            wl = sorted(s_on); wins = [(a, b, ONSET) for a, b in wl]
            cQ, cE = cum_tables(Pv)
            for mode in ('elig', 'all'):
                R = val_rows(P, Pv, wins, prior, ONSET, mode)
                X, y, e, meta, call = lab.to_matrix(R, ONSET, None, None)
                assert list(call) == COLS
                wi, li = meta[:, 0], meta[:, 1]
                d, t = R['d'], R['t']; cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
                c_r, d_r, s6_r, t_r = cls[wi], d[wi], s6[wi], t[wi]
                Xs = X.copy()                                  # sp4: prior columns blended with the March prior of the days < d
                for j, (cc, ss) in enumerate(((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r))):
                    q = cQ[cc, d_r, ss, li]; ee = cE[cc, d_r, ss, li]
                    sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + K_SP)
                    Xs[:, JP[j]] = (1 - a_) * X[:, JP[j]] + a_ * sp
                rc = rc_of(Pv, R, 'march_on', panel, ONSET)
                inf[f'on_{mode}_histvis'] = R['hist_vis']; inf[f'on_{mode}_Tvis'] = R['T_vis']; inf['on_n'] = len(wl)
                for fam in ('base', 'af'):
                    tag = {('base', 'elig'): 'elig', ('af', 'all'): 'all', ('base', 'all'): 'cross', ('af', 'elig'): 'afelig'}[(fam, mode)]
                    for nm, Xv in (('trainprior', X), ('sp4', Xs)):
                        p = ON[fam].predict_proba(Xv[:, KEEPC])[:, 1]
                        p6 = np.zeros((R['W'], F, R['L']), np.float32); p6[wi, F - 1, li] = p
                        res.append(lab.score_probs(rc, p6, arm=f'{tag}_{nm}', set_name='march_on'))
            del cQ, cE
        info.append(inf)
        lab.log(panel, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in inf.items()}), f'{time.time() - t1:.0f}s')
        lab.drop_panel(panel); gc.collect()
    R_ = pd.concat(res, ignore_index=True)
    R_.to_pickle(f'{OUT}/pmarch_{"_".join(p for p in parts if p != "dry")}.pkl')
    pd.DataFrame(info).to_csv(f'{OUT}/pmarch_info_{"_".join(p for p in parts if p != "dry")}.csv', index=False)
    print('\n== PSEUDO-MARCH aggregated IoU (window -> panel -> family -> mean) ==')
    rows = []
    for sname in dict.fromkeys(R_.set):
        for a in dict.fromkeys(R_[R_.set == sname].arm):
            g = R_[(R_.set == sname) & (R_.arm == a)]
            rows.append(dict(set=sname, arm=a, n=len(g), iou=lab.aggregate(g), plain=g.iou.mean(), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(g))}))
    S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(f'{OUT}/summary_{"_".join(p for p in parts if p != "dry")}.csv', index=False)
    print('\n== PSEUDO-MARCH paired deltas (lab.paired_bootstrap) ==')
    out = []
    for sname in dict.fromkeys(R_.set):
        g = R_[R_.set == sname]
        fams = ('np', 'id', 'npid') if sname != 'march_on' else ('trainprior', 'sp4')
        for nm in fams:
            A = g[g.arm == f'elig_{nm}']
            for tag in ('all', 'cross', 'afelig'):
                B = g[g.arm == f'{tag}_{nm}']
                if len(B):
                    r = af.compare(A, B, f'{sname} {tag}_{nm} vs elig_{nm}'); r.update(set=sname, arm=f'{tag}_{nm}'); out.append(r); print(af.fmt(r))
            if sname == 'march_on':
                for lo, hi, lb in ((-1, 4, 'd1-5'), (14, 99, 'd16+')):
                    A2 = A[(A.d > lo) & (A.d <= hi)]; B2 = g[(g.arm == f'all_{nm}') & (g.d > lo) & (g.d <= hi)]
                    if len(A2):
                        r = af.compare(A2, B2, f'{sname} all_{nm} vs elig_{nm} [{lb}]'); r.update(set=f'{sname}_{lb}', arm=f'all_{nm}'); out.append(r); print(af.fmt(r))
    pd.DataFrame(out).to_csv(f'{OUT}/deltas_{"_".join(p for p in parts if p != "dry")}.csv', index=False)
    for sname, a, b in (('offlike_m', 'elig_npid', 'all_npid'), ('march_on', 'elig_sp4', 'all_sp4')):
        A = R_[(R_.set == sname) & (R_.arm == a)]; B = R_[(R_.set == sname) & (R_.arm == b)]
        if len(A) and len(B):
            a2, b2 = fc.paired(A, B)
            t = pd.DataFrame(dict(n=a2.groupby('panel').size(), base=a2.groupby('panel').iou.mean(), new=b2.groupby('panel').iou.mean())); t['delta'] = t.new - t.base
            print(f'\nper panel ({sname}): {b} vs {a}'); print(t.round(4).to_string())


# ---------------------------------------------------------------- C: label-free footprint on the released official windows
def split_rows_af(P, split, mode):
    """lab.split_rows with the all-finite history / slot-T value (mode 'all')."""
    z = np.load(P['npz'], allow_pickle=False)
    wi = lab.window_index(P['panel'], split)
    org = pd.to_datetime(wi.forecast_origin, utc=True)
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
    Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32); VT = z[f'{split}_VT'].astype(np.float32)
    if mode == 'all':
        x = np.load(f'{af.XDIR}/t2x_{P["panel"]}.npz', allow_pickle=False)
        assert [str(a) for a in x[f'{split}_window_id']] == wi.window_id.tolist()
        Vhx = x[f'{split}_Vhx'].astype(np.float32); Fhx = x[f'{split}_Fhx'].astype(np.float32); VTx = x[f'{split}_VTx'].astype(np.float32)
        assert not (np.isfinite(Vh) & np.isfinite(Vhx)).any() and not (np.isfinite(VT) & np.isfinite(VTx)).any()
        el = np.isfinite(Vh); Fh = np.where(el, Fh, Fhx); Vh = np.where(el, Vh, Vhx); VT = np.where(np.isfinite(VT), VT, VTx)
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, tod, dow, prior, VT)
        wdays = sorted(wi.date.unique())
        mdays = [str(a) for a in z[f'{split}_days']]; ix = [mdays.index(d) for d in wdays]
        ONs = z[f'{split}_ON'][ix].astype(np.float32); OFFs = z[f'{split}_OFF'][ix].astype(np.float32)
        lab.add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, P['cap'])
    W, L = qlast.shape
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[], wi=wi, dow=dow,
             hist_vis=float(np.isfinite(Vh).mean()), T_vis=float(np.isfinite(VT).mean()))
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, dow, tod, h)
        R['rows'].append((h, feats, None, None))
    return R


def run_dry():
    OG = og_models(); ON = on_models(); rows = []
    ixn = [COLS.index(c) for c in CN]; ixi = [COLS.index(c) for c in CI]
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        for split in ('validation', 'private'):
            M = {}; vis = {}
            for mode in ('elig', 'all'):
                R = split_rows_af(P, split, mode); vis[mode] = (R['hist_vis'], R['T_vis'])
                if mode == 'elig':                           # my eligible-only split rows must equal lab.split_rows
                    R0 = lab.split_rows(P, split)
                    assert all(np.array_equal(np.asarray(R['rows'][0][1][c]), np.asarray(R0['rows'][0][1][c]), equal_nan=True) for c in R0['rows'][0][1]), 'split_rows parity'
                X, _, _, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
                Xo, _, _, mo, _ = lab.to_matrix(R, ONSET, None, None)
                for fam in ('base', 'af'):
                    p6 = np.zeros((R['W'], F, R['L']), np.float32)
                    if len(X):
                        p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = 0.5 * (OG[f'{fam}_np'].predict(X[:, ixn], num_threads=lab.THREADS) + OG[f'{fam}_id'].predict(X[:, ixi], num_threads=lab.THREADS))
                    if len(Xo):
                        p6[mo[:, 0], F - 1, mo[:, 1]] = ON[fam].predict_proba(Xo[:, KEEPC])[:, 1]
                    M[(fam, mode)] = np.stack([lab.decode(p6[i].ravel()).reshape(F, -1) for i in range(R['W'])])
            for i in range(R['W']):
                a = M[('base', 'elig')][i]; r = dict(panel=panel, family=panel.rsplit('_', 1)[0], split=split, cond=R['cond'][i], window_id=R['wi'].window_id.iloc[i],
                                                       n_base=int(a.sum()), hist_vis_elig=vis['elig'][0], hist_vis_all=vis['all'][0], T_vis_elig=vis['elig'][1], T_vis_all=vis['all'][1])
                for nm, key in (('all', ('af', 'all')), ('cross', ('base', 'all')), ('afelig', ('af', 'elig'))):
                    b = M[key][i]
                    r[f'{nm}_n'] = int(b.sum()); r[f'{nm}_jacc'] = float((a & b).sum() / max((a | b).sum(), 1)); r[f'{nm}_changed'] = bool((a != b).any())
                    r[f'{nm}_cells_changed'] = int((a != b).sum())
                rows.append(r)
        lab.drop_panel(panel)
    D = pd.DataFrame(rows); D.to_csv(f'{OUT}/dryrun_footprint.csv', index=False)
    print('\n== label-free footprint on the released official windows: decoded masks vs base model on eligible-only features ==')
    g = D.groupby(['split', 'cond']).agg(windows=('panel', 'size'), cells_base=('n_base', 'sum'),
                                         all_changed=('all_changed', 'sum'), all_jacc=('all_jacc', 'mean'), all_cells=('all_cells_changed', 'sum'), all_n=('all_n', 'sum'),
                                         cross_changed=('cross_changed', 'sum'), cross_jacc=('cross_jacc', 'mean'), cross_cells=('cross_cells_changed', 'sum'),
                                         afelig_changed=('afelig_changed', 'sum'), afelig_jacc=('afelig_jacc', 'mean'),
                                         hist_vis_elig=('hist_vis_elig', 'mean'), hist_vis_all=('hist_vis_all', 'mean'), T_vis_elig=('T_vis_elig', 'mean'), T_vis_all=('T_vis_all', 'mean'))
    print(g.round(4).to_string())


if 'og' in parts or 'on' in parts:
    run_pmarch()
if 'dry' in parts:
    run_dry()
lab.log('DONE', parts)
