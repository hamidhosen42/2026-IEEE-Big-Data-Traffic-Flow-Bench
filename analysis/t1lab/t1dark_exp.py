"""t1dark: Task 1 transductive BLACKOUT rows, measured on a TARGET-MONTH pseudo-holdout (track t1dark).

Question: do simulated-blackout transductive rows built inside the released validation / private masked months reduce
the error on blackout cells (and not hurt ordinary masked cells)?

Pseudo-holdout (per panel and target month; released data only, labels = released values):
  * simulated blackouts: queue-like origins T on the masked month (selector-like rule: eligibility coverage >= 0.7 in
    T-12..T-1, an observed queued link (v <= 0.6 vf) in T+1..T+6; onset = no established queue in T-12..T-1 (no link
    queued at >= 2 history slots); ongoing = established queue and persistence (IoU of the queued state at T vs T+1..T+6
    over observed cells) <= 0.9), at least 18 slots away from the official (real) blackouts and from each other.
    Rows T+1..T+18 are blanked at every station (incl. the ineligible-valued cells), as in the release.
    A random subset (<= N_SIM per month) is kept, sorted in time and assigned alternately to halves A / B.
  * evaluation cells: (i) released eligible cells inside the evaluation-half blackouts; (ii) ordinary cells: 3% of the
    released eligible cells outside every simulated blackout (hidden in the evaluation view).
  * fold f: evaluation half = B (f=0) / A (f=1); the other half gives the treatment's training blackout rows.
Arms (identical except for the extra rows):
  base  production recipe of kaggle_jobs/t1_pipeline_v13a.py: darkened train targets (pseudo gap origins) + 3 transductive
        rounds per month hiding 20% of the released eligible cells; the validation-month 15% internal holdout (rng 7) is
        hidden in every transductive view and never a label (as production); evaluation cells are hidden in every view.
  dark  base + simulated-blackout rows: one view per month with the training-half blackouts blanked; rows = released
        eligible cells inside them, labels = released values.
Scale: one keep fraction p for train and transductive rows (production composition); blackout rows kept at min(1, 2p)
(the kernel uses both halves, so this matches the kernel's blackout-row share); LightGBM = t1lab reduced pF
(255 leaves, lr 0.08) with N trees.
Data: data/t2local/t2data_<P>.npz + data/t2local/x/t2x_<P>.npz (ineligible-valued cells merged into the view, as in the
masked parquet); no occupancy (all-NaN columns in both arms); station order = order_index; milepost = cumulative link
length (start positions, = the D12_I405_S parquet convention); train target mask = per-cell Bernoulli at the regime
rate (statistically identical to the organizer's per-cell blake2b hash).

Usage:  t1dark_exp.py prep <PANEL> [p=0.35]          train rows (cached) + simulated blackouts
        t1dark_exp.py fold <PANEL> <FOLD> [n_trees=600] [p=0.35] [n_sim=48]
        t1dark_exp.py all <PANEL> [n_trees=600] [lr=0.08] [p=0.35] [n_sim=48]   (prep + fold 0 + fold 1)
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t1lab  # noqa: E402  (production feature code compiled from kaggle_jobs/t1_pipeline_v13a.py via ast)
import lightgbm as lgb  # noqa: E402

prod = t1lab.prod
regime_of_dates = t1lab.regime_of_dates
TRACK = 't1dark'
RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', TRACK)
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', TRACK)
os.makedirs(RES, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
THREADS = int(os.environ.get('T2LAB_THREADS', 4))
GAP, H, F = 18, 12, 6
NOPROF = {'hv', 'hq', 'tod', 'dow'}
SPLITS = ('validation', 'private')
T0 = time.time()
SMOKE = os.environ.get('T1DARK_SMOKE', '') == '1'          # light correctness check: 1 train chunk, tiny model
SFX = '_smoke' if SMOKE else ''


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


class View:
    """duck-typed production Cube (only what features() reads)."""
    def __init__(self, mv, mq, dow, reg):
        self.mv, self.mq = mv, mq
        self.mo = np.full(mv.shape, np.nan, np.float32)
        self.dow, self.reg = dow, reg


def feats(view, st, hv, hq):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Fd, _, _ = prod.features(view, st, hv[view.dow], hq[view.dow])
    return Fd


# ------------------------------------------------------------------ data
def load_panel(panel):
    z = np.load(os.path.join(ROOT, 'data', 't2local', f't2data_{panel}.npz'), allow_pickle=True)
    x = np.load(os.path.join(ROOT, 'data', 't2local', 'x', f't2x_{panel}.npz'), allow_pickle=True)
    links = z['links'].astype(str)
    assert (x['links'].astype(str) == links).all() and (np.diff(z['order_index']) > 0).all()
    length = z['length'].astype(float)
    st = pd.DataFrame({'link_id': links, 'milepost': np.concatenate([[0.0], np.cumsum(length)[:-1]]),
                       'lanes': z['lanes'].astype(float), 'length_km': length, 'vf': z['vf'].astype(float),
                       'cap': z['cap'].astype(float), 'critical_density': z['fd_kcrit'].astype(float),
                       'k_jam': z['fd_kjam'].astype(float)}, index=pd.Index(links, name='station_id'))
    assert st[['lanes', 'vf', 'cap', 'critical_density', 'k_jam']].notna().all().all()
    return z, x, st


def train_cube(panel, z, x):
    td = z['train_days'].astype(str); assert (np.sort(td) == td).all() and len(td) == 273
    xd = x['train_days'].astype(str); ox = np.argsort(xd); assert (xd[ox] == td).all()
    V = z['train_V'].astype(np.float32); Q = z['train_F'].astype(np.float32)
    elig = np.isfinite(V) & np.isfinite(Q)
    V = np.where(np.isfinite(V), V, x['train_Vx'][ox].astype(np.float32))
    Q = np.where(np.isfinite(Q), Q, x['train_Fx'][ox].astype(np.float32))
    reg = regime_of_dates(panel, pd.Series(td)).astype(str)
    dow = pd.to_datetime(pd.Series(td)).dt.dayofweek.to_numpy()
    rate = np.array([prod.RATE[r] for r in reg], np.float32)
    target = elig & (np.random.default_rng(20251005).random(V.shape, dtype=np.float32) < rate[:, None, None])
    return dict(days=td, V=V, Q=Q, elig=elig, reg=reg, dow=dow, target=target)


def profile(mv, mq, dow):
    S = mv.shape[2]
    hv = np.full((7, 288, S), np.nan, np.float32); hq = hv.copy()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for w in range(7):
            sel = dow == w
            hv[w] = np.nanmean(mv[sel], 0); hq[w] = np.nanmean(mq[sel], 0)
    hv = np.where(np.isfinite(hv), hv, np.nanmean(hv)); hq = np.where(np.isfinite(hq), hq, np.nanmean(hq))
    return hv, hq


def month_cube(panel, z, x, split):
    d = z[f'{split}_days'].astype(str); o = np.argsort(d); days = d[o]
    xd = x[f'{split}_days'].astype(str); ox = np.argsort(xd); assert (xd[ox] == days).all()
    V = z[f'{split}_Vm'][o].astype(np.float32); Fq = z[f'{split}_Fm'][o].astype(np.float32); Em = z[f'{split}_Em'][o]
    obs = Em & np.isfinite(V) & np.isfinite(Fq)
    mv = np.where(np.isfinite(V), V, x[f'{split}_Vmx'][ox].astype(np.float32))
    mq = np.where(np.isfinite(Fq), Fq, x[f'{split}_Fmx'][ox].astype(np.float32))
    reg = regime_of_dates(panel, pd.Series(days)).astype(str)
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    real = []
    for s in z[f'{split}_wi_forecast_origin'].astype(str):
        real.append((list(days).index(s[:10]), int(s[11:13]) * 12 + int(s[14:16]) // 5))
    realrows = np.zeros((len(days), 288), bool)
    for dd, t in real:
        realrows[dd, t + 1:t + 1 + GAP] = True
    assert not np.isfinite(mv[realrows]).any(), 'real blackout rows must be blank'
    return dict(split=split, days=days, mv=mv, mq=mq, Em=Em, obs=obs, reg=reg, dow=dow, real=real, realrows=realrows)


def sim_candidates(M, vf):
    """queue-like origins on the masked month, >= 18 slots away from every real blackout."""
    obs, mv, Em, realrows = M['obs'], M['mv'], M['Em'], M['realrows']
    Qm = obs & (mv <= 0.6 * vf[None, None, :])
    D = mv.shape[0]
    out = []
    for d in range(D):
        for t in range(H, 288 - GAP):                      # rows t+1..t+18 inside the day
            if realrows[d, max(t + 1 - GAP, 0):min(t + 2 * GAP + 1, 288)].any():
                continue
            if Em[d, t - H:t].mean() < 0.7 or not Qm[d, t + 1:t + 1 + F].any():
                continue
            if (Qm[d, t - H:t].sum(0) >= 2).any():
                S = Qm[d, t]; Tf = Qm[d, t + 1:t + 1 + F]; Mf = obs[d, t + 1:t + 1 + F]
                u = ((Tf | S[None]) & Mf).sum(); i = (Tf & S[None] & Mf).sum()
                if (1.0 if u == 0 else i / u) > 0.9:
                    continue
                out.append((d, t, 1))
            else:
                out.append((d, t, 0))
    return out


def sim_select(cands, n_sim, seed):
    """random-order greedy with >= 2*GAP slots between origins, balanced like the official windows (half onset):
    pass 1 onset candidates up to n_sim // 2, pass 2 ongoing candidates up to n_sim, pass 3 onset again if still short."""
    rng = np.random.default_rng(seed)
    acc = []
    order = rng.permutation(len(cands))
    on = [i for i in order if cands[i][2] == 0]; og = [i for i in order if cands[i][2] == 1]
    for pool, cap in ((on, n_sim // 2), (og, n_sim), (on, n_sim)):
        for i in pool:
            if len(acc) >= cap:
                break
            d, t, c = cands[i]
            if (d, t, c) not in acc and all(dd != d or abs(tt - t) >= 2 * GAP for dd, tt, _ in acc):
                acc.append((d, t, c))
    return sorted(acc)


# ------------------------------------------------------------------ prep: train rows (cached) + simulated blackouts
def prep(panel, p=0.35, n_sim=48):
    z, x, st = load_panel(panel)
    f_rows = f'{CACHE}/{panel}_train_p{p}{SFX}.npz'
    f_meta = f'{CACHE}/{panel}_meta{SFX}.npz'
    T = train_cube(panel, z, x)
    mv = np.where(T['target'], np.nan, T['V']); mq = np.where(T['target'], np.nan, T['Q'])
    hv, hq = profile(mv, mq, T['dow'])
    pz = np.load(os.path.join(ROOT, 'kaggle_jobs', 'out', 't2lab_full', f'pseudo_{panel}.npz'), allow_pickle=True)
    origins = [(int(d), int(t)) for d, t in zip(pz['d'], pz['t'])]
    vf = st.vf.to_numpy(); ln = st.lanes.to_numpy()
    if not os.path.exists(f_rows):
        D = len(T['days']); chunks = np.array_split(np.arange(D), 7)
        p_tr = p * min(1.0, 2.4e6 / max(1, int(T['target'].sum())))   # production caps the train rows at 4 x 600k
        if SMOKE:
            chunks = chunks[:1]
        rng = np.random.default_rng(9)
        XS, YV, YQ, GP = [], [], [], []
        cols = None
        for ci, idx in enumerate(chunks):
            cmv, cmq = mv[idx].copy(), mq[idx].copy(); gapm = np.zeros(cmv.shape, bool)
            pos = {int(dd): j for j, dd in enumerate(idx)}
            for dd, t in origins:                                   # = production darken()
                if dd in pos:
                    j = pos[dd]; cmv[j, t + 1:t + 1 + GAP] = np.nan; cmq[j, t + 1:t + 1 + GAP] = np.nan; gapm[j, t + 1:t + 1 + GAP] = True
            Fd = feats(View(cmv, cmq, T['dow'][idx], T['reg'][idx]), st, hv, hq)
            if cols is None:
                cols = [k for k in Fd if k not in NOPROF]
                assert len(cols) == prod.V12_NFEAT + prod.N_FD - 4
            m = T['target'][idx] & (rng.random(cmv.shape, dtype=np.float32) < p_tr)
            yv = (T['V'][idx] / vf[None, None, :])[m] - Fd['tv'][m]; yq = (T['Q'][idx] / ln[None, None, :])[m] - Fd['tq'][m]
            XS.append(np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32))
            YV.append(yv.astype(np.float32)); YQ.append(yq.astype(np.float32)); GP.append(gapm[m])
            del Fd; gc.collect()
            log(panel, 'train chunk', ci, 'rows', int(m.sum()), 'in gaps', int(gapm[m].sum()))
        X = np.concatenate(XS); yv = np.concatenate(YV); yq = np.concatenate(YQ); gp = np.concatenate(GP)
        ok = np.isfinite(yv) & np.isfinite(yq)
        np.savez(f_rows, X=X[ok], yv=yv[ok], yq=yq[ok], gap=gp[ok], cols=np.array(cols))
        log(panel, 'train rows', int(ok.sum()), 'of', len(ok), '| in simulated gaps', int(gp[ok].sum()), '| target cells total',
            int(T['target'].sum()))
        del X, XS
    # simulated blackouts per target month
    meta = {}
    for si, split in enumerate(SPLITS):
        M = month_cube(panel, z, x, split)
        cands = sim_candidates(M, vf)
        sel = sim_select(cands, n_sim, seed=100 + si)
        nb_real = int(M['Em'][M['realrows']].sum())
        targets = M['Em'] & ~M['obs']
        meta[f'{split}_sim'] = np.array(sel, np.int64).reshape(-1, 3)
        meta[f'{split}_ncand'] = len(cands)
        # real target composition per regime (for the composite S_state). Scored blackout targets are only the
        # hash-masked cells inside the blackout rows (production log: D7_I10_W validation template rows 233060 =
        # hidden cells outside blackouts + rate x eligible cells inside; validation gap share of targets 0.017).
        for r in ('R1', 'R2', 'R3'):
            dm = M['reg'] == r
            inside = float((M['Em'] & M['realrows'][:, :, None])[dm].sum())
            outside = float((targets & ~M['realrows'][:, :, None])[dm].sum())
            meta[f'{split}_ntgt_{r}'] = outside + prod.RATE[r] * inside
            meta[f'{split}_nblk_{r}'] = prod.RATE[r] * inside
        log(panel, split, 'candidates', len(cands), '(onset', sum(c[2] == 0 for c in cands), ') selected', len(sel),
            '(onset', sum(c[2] == 0 for c in sel), ') | real blackout targets', nb_real, 'of', int(targets.sum()))
    np.savez(f_meta, **meta, hv=hv, hq=hq)
    return f_rows, f_meta


# ------------------------------------------------------------------ one fold: base vs dark
def lgb_params(seed, leaves=255, lr=0.08):
    return dict(objective='regression', learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100, max_bin=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                num_threads=THREADS, seed=seed, bagging_seed=seed + 10, feature_fraction_seed=seed + 20)


def run_fold(panel, fold, n_trees=600, p=0.35, n_sim=48, tag='', lr=0.08, seeds=(9,)):
    z, x, st = load_panel(panel)
    R = np.load(f'{CACHE}/{panel}_train_p{p}{SFX}.npz', allow_pickle=True)
    meta = np.load(f'{CACHE}/{panel}_meta{SFX}.npz', allow_pickle=True)
    cols = list(R['cols']); hv, hq = meta['hv'], meta['hq']
    vf = st.vf.to_numpy(); ln = st.lanes.to_numpy()
    p_dark = min(1.0, 2 * p)
    TXr, TYVr, TYQr, TXd, TYVd, TYQd = [], [], [], [], [], []
    EV = {}
    info = dict(panel=panel, fold=fold, n_trees=n_trees, lr=lr, p=p, p_dark=p_dark, seeds=list(seeds))
    for si, split in enumerate(SPLITS):
        M = month_cube(panel, z, x, split)
        obs = M['obs']; shape = obs.shape
        sim = meta[f'{split}_sim']
        half = np.arange(len(sim)) % 2                               # 0 = A, 1 = B
        eval_half = 1 if fold == 0 else 0
        rowsE = np.zeros(shape[:2], bool); rowsT = rowsE.copy(); ev_id = np.full(shape[:2], -1, np.int32)
        for k, (d, t, c) in enumerate(sim):
            if half[k] == eval_half:
                rowsE[d, t + 1:t + 1 + GAP] = True; ev_id[d, t + 1:t + 1 + GAP] = k
            else:
                rowsT[d, t + 1:t + 1 + GAP] = True
        allsim = rowsE | rowsT
        EB = np.broadcast_to(rowsE[:, :, None], shape); TB = np.broadcast_to(rowsT[:, :, None], shape)
        HO = obs & ~np.broadcast_to(allsim[:, :, None], shape) & (np.random.default_rng(500 + si).random(shape) < 0.03)
        HP = (obs & (np.random.default_rng(7).random(shape) < 0.15)) if split == 'validation' else np.zeros(shape, bool)
        hold = HP | HO | (EB & obs)
        vf3 = np.broadcast_to(vf[None, None, :], shape); ln3 = np.broadcast_to(ln[None, None, :], shape)
        rng_keep = np.random.default_rng(900 + 10 * fold + si)
        # --- shared transductive rounds (production)
        for r in range(prod.TR_ROUNDS):
            hid = obs & ~hold & ~EB & (np.random.default_rng(10000 + 10 * r + (split == 'private')).random(shape) < prod.TR_HIDE)
            gone = hid | hold | EB
            Fd = feats(View(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg']), st, hv, hq)
            m = hid & (rng_keep.random(shape) < p)
            TXr.append(np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32))
            TYVr.append(((M['mv'] / vf3)[m] - Fd['tv'][m]).astype(np.float32)); TYQr.append(((M['mq'] / ln3)[m] - Fd['tq'][m]).astype(np.float32))
            del Fd; gc.collect()
        # --- treatment: simulated-blackout rows (training half)
        gone = TB | hold | EB
        Fd = feats(View(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg']), st, hv, hq)
        m = obs & TB & ~hold & (rng_keep.random(shape) < p_dark)
        TXd.append(np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32))
        TYVd.append(((M['mv'] / vf3)[m] - Fd['tv'][m]).astype(np.float32)); TYQd.append(((M['mq'] / ln3)[m] - Fd['tq'][m]).astype(np.float32))
        info[f'{split}_dark_rows'] = int(m.sum())
        del Fd; gc.collect()
        # --- evaluation view (shared): evaluation-half blackouts blanked, ordinary cells hidden
        gone = HO | EB
        Fd = feats(View(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg']), st, hv, hq)
        for kind, m in (('blk', EB & obs), ('ord', HO)):
            d_, t_, s_ = np.nonzero(m)
            EV[f'{split}_{kind}'] = dict(X=np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32),
                                         tv=Fd['tv'][m].astype(np.float64), tq=Fd['tq'][m].astype(np.float64),
                                         vt=M['mv'][m].astype(np.float64), qt=M['mq'][m].astype(np.float64),
                                         vf=vf[s_], ln=ln[s_], d=d_, t=t_, s=s_, reg=M['reg'][d_],
                                         ev=ev_id[d_, t_] if kind == 'blk' else d_)
        del Fd; gc.collect()
        log(panel, f'fold {fold}', split, 'eval blackout cells', int((EB & obs).sum()), 'ordinary', int(HO.sum()),
            'dark rows', info[f'{split}_dark_rows'], 'rand rows so far', sum(len(a) for a in TYVr))
    Xr = np.concatenate([R['X']] + TXr); YVr = np.concatenate([R['yv']] + TYVr); YQr = np.concatenate([R['yq']] + TYQr)
    Xd = np.concatenate(TXd); YVd = np.concatenate(TYVd); YQd = np.concatenate(TYQd)
    ok_r = np.isfinite(YVr) & np.isfinite(YQr); ok_d = np.isfinite(YVd) & np.isfinite(YQd)
    info.update(train_rows=int(len(R['yv'])), rand_rows=int(sum(len(a) for a in TYVr)), dark_rows=int(ok_d.sum()),
                bad_r=int((~ok_r).sum()), bad_d=int((~ok_d).sum()))
    log(panel, f'fold {fold}', info)
    del TXr, TXd; gc.collect()
    preds = {}
    for si_, seed in enumerate(seeds):
      for arm0 in ('base', 'dark'):
        arm = arm0 + ('' if si_ == 0 else str(si_ + 1))           # seed 9 -> base / dark, second seed -> base2 / dark2
        if arm0 == 'base':
            XA, YV, YQ = Xr[ok_r], YVr[ok_r], YQr[ok_r]
        else:
            XA = np.concatenate([Xr[ok_r], Xd[ok_d]]); YV = np.concatenate([YVr[ok_r], YVd[ok_d]]); YQ = np.concatenate([YQr[ok_r], YQd[ok_d]])
        t0 = time.time()
        prm = lgb_params(seed, lr=lr)
        mv_ = lgb.train(prm, lgb.Dataset(XA, YV, free_raw_data=True), n_trees)
        mq_ = lgb.train(prm, lgb.Dataset(XA, YQ, free_raw_data=True), n_trees)
        for key, E in EV.items():
            pv = np.clip((E['tv'] + mv_.predict(E['X'])) * E['vf'], 3, 1.05 * E['vf'])
            pq = np.maximum((E['tq'] + mq_.predict(E['X'])) * E['ln'], 20)
            preds[f'{arm}|{key}|v'] = pv.astype(np.float32); preds[f'{arm}|{key}|q'] = pq.astype(np.float32)
        log(panel, f'fold {fold}', arm, 'rows', len(YV), f'trained+predicted in {time.time() - t0:.0f}s')
        del XA, YV, YQ, mv_, mq_; gc.collect()
    out = {}
    for key, E in EV.items():
        for c in ('vt', 'qt', 'ln', 'd', 't', 's', 'reg', 'ev', 'tv', 'tq', 'vf'):
            out[f'{key}|{c}'] = E[c]
    out.update(preds)
    np.savez_compressed(f'{RES}/{panel}_f{fold}{tag}{SFX}.npz', info=json.dumps(info), **out)
    # quick summary
    for key, E in EV.items():
        rv = {a: float(np.sqrt(np.mean((preds[f'{a}|{key}|v'] - E['vt']) ** 2))) for a in ('base', 'dark')}
        rq = {a: float(np.sqrt(np.mean(((preds[f'{a}|{key}|q'] - E['qt']) / E['ln']) ** 2))) for a in ('base', 'dark')}
        log(panel, f'fold {fold}', key, 'n', len(E['vt']), '| speed RMSE base %.3f dark %.3f | flow/lane RMSE base %.2f dark %.2f'
            % (rv['base'], rv['dark'], rq['base'], rq['dark']))
    log('done', f'{time.time() - T0:.0f}s')


if __name__ == '__main__':
    mode, panel = sys.argv[1], sys.argv[2]
    kw = dict(a.split('=') for a in sys.argv[3:] if '=' in a)
    p = float(kw.get('p', 0.35)); n_sim = int(kw.get('n_sim', 48))
    nt, lr = int(kw.get('n_trees', 600)), float(kw.get('lr', 0.08))
    if mode == 'prep':
        prep(panel, p=p, n_sim=n_sim)
    elif mode == 'fold':
        fold = int(sys.argv[3])
        run_fold(panel, fold, n_trees=nt, p=p, n_sim=n_sim, tag=kw.get('tag', ''), lr=lr)
    elif mode == 'all':                                   # prep + both folds in one process (one semaphore slot)
        seeds = tuple(int(v) for v in kw.get('seeds', '9').split(','))
        for pn in panel.split(','):
            prep(pn, p=p, n_sim=n_sim)
            for fold in (0, 1):
                run_fold(pn, fold, n_trees=nt, p=p, n_sim=n_sim, tag=kw.get('tag', ''), lr=lr, seeds=seeds)
    else:
        raise SystemExit(__doc__)
