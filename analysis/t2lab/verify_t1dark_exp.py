"""verify_t1dark_exp.py - adversarial, independent re-implementation of the t1dark target-month pseudo-holdout.

Independent of analysis/t1lab/t1dark_exp.py by design (only production feature code t1lab.prod.features and the organizer
regime_of_dates are shared):
  * own loaders / cubes, own queue-like candidate scan and greedy selection (seed SEL, not 100/101), RANDOM half assignment
    (not time-alternating), own train-target Bernoulli seed, own keep / transductive / ordinary-holdout seeds, own LGB seed;
  * evaluation views: 'std' (evaluation blackouts blanked, ordinary cells hidden) and 'sparse' (std + an extra 50% of the
    observed cells in rows T-12..T of every evaluation blackout hidden) -> dose response of the dark gain to history
    density (real blackouts have T-12..T-1 fully observed, i.e. DENSER history than the simulated ones);
  * per-cell output with step k (1..18), event id, condition, per-event history visibility.
Usage: verify_t1dark_exp.py <P1,P2> [folds=0,1] [leaves=255] [lr=0.1] [trees=450] [p=0.35] [seed=21] [sel=4242] [nsim=48] [tag=]
"""
import gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'analysis', 't1lab'))
import t1lab  # noqa: E402
import lightgbm as lgb  # noqa: E402

prod = t1lab.prod
OUTD = os.path.join(HERE, 'results', 'verify_t1dark')
os.makedirs(OUTD, exist_ok=True)
THREADS = int(os.environ.get('T2LAB_THREADS', 4))
GAP, HIST, HOR = 18, 12, 6
DROP = {'hv', 'hq', 'tod', 'dow'}
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


class V:
    def __init__(self, mv, mq, dow, reg):
        self.mv, self.mq, self.dow, self.reg = mv, mq, dow, reg
        self.mo = np.full(mv.shape, np.nan, np.float32)


def feats(mv, mq, dow, reg, st, hv, hq):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F, _, _ = prod.features(V(mv, mq, dow, reg), st, hv[dow], hq[dow])
    return F


def sorted_days(a):
    d = a.astype(str)
    o = np.argsort(d, kind='stable')
    return d[o], o


def load(panel):
    z = np.load(f'{ROOT}/data/t2local/t2data_{panel}.npz', allow_pickle=True)
    x = np.load(f'{ROOT}/data/t2local/x/t2x_{panel}.npz', allow_pickle=True)
    links = z['links'].astype(str)
    assert (x['links'].astype(str) == links).all()
    L = z['length'].astype(float)
    st = pd.DataFrame(dict(link_id=links, milepost=np.r_[0.0, np.cumsum(L)[:-1]], lanes=z['lanes'].astype(float),
                           length_km=L, vf=z['vf'].astype(float), cap=z['cap'].astype(float),
                           critical_density=z['fd_kcrit'].astype(float), k_jam=z['fd_kjam'].astype(float)),
                      index=pd.Index(links, name='station_id'))
    return z, x, st


def profile(mv, mq, dow):
    S = mv.shape[2]
    hv = np.full((7, 288, S), np.nan, np.float32)
    hq = hv.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for w in range(7):
            m = dow == w
            if m.any():
                hv[w] = np.nanmean(mv[m], 0)
                hq[w] = np.nanmean(mq[m], 0)
    hv = np.where(np.isfinite(hv), hv, np.nanmean(hv))
    hq = np.where(np.isfinite(hq), hq, np.nanmean(hq))
    return hv, hq


def build_train(panel, z, x, st, p):
    """train rows of the darkened train view (production recipe), own seeds; cached per (panel, p)."""
    path = f'{OUTD}/cache_{panel}_train_p{p}.npz'
    td, o = sorted_days(z['train_days'])
    xd, ox = sorted_days(x['train_days'])
    assert (xd == td).all()
    Vr = z['train_V'][o].astype(np.float32)
    Qr = z['train_F'][o].astype(np.float32)
    elig = np.isfinite(Vr) & np.isfinite(Qr)
    Vr = np.where(np.isfinite(Vr), Vr, x['train_Vx'][ox].astype(np.float32))
    Qr = np.where(np.isfinite(Qr), Qr, x['train_Fx'][ox].astype(np.float32))
    reg = np.asarray(t1lab.regime_of_dates(panel, pd.Series(td))).astype(str)
    dow = pd.to_datetime(pd.Series(td)).dt.dayofweek.to_numpy()
    rate = np.array([prod.RATE[r] for r in reg], np.float32)
    target = elig & (np.random.default_rng(31337).random(Vr.shape, dtype=np.float32) < rate[:, None, None])
    mv = np.where(target, np.nan, Vr)
    mq = np.where(target, np.nan, Qr)
    hv, hq = profile(mv, mq, dow)
    if os.path.exists(path):
        R = np.load(path, allow_pickle=True)
        return dict(X=R['X'], yv=R['yv'], yq=R['yq'], cols=list(R['cols'])), hv, hq
    pz = np.load(f'{ROOT}/kaggle_jobs/out/t2lab_full/pseudo_{panel}.npz', allow_pickle=True)
    origins = [(int(a), int(b)) for a, b in zip(pz['d'], pz['t'])]
    vf = st.vf.to_numpy()
    ln = st.lanes.to_numpy()
    p_tr = p * min(1.0, 2.4e6 / max(1, int(target.sum())))
    keep = np.random.default_rng(4711)
    XS, YV, YQ = [], [], []
    cols = None
    for idx in np.array_split(np.arange(len(td)), 7):
        cmv, cmq = mv[idx].copy(), mq[idx].copy()
        loc = {int(d): j for j, d in enumerate(idx)}
        for d, t in origins:
            if d in loc:
                cmv[loc[d], t + 1:t + 1 + GAP] = np.nan
                cmq[loc[d], t + 1:t + 1 + GAP] = np.nan
        F = feats(cmv, cmq, dow[idx], reg[idx], st, hv, hq)
        if cols is None:
            cols = [k for k in F if k not in DROP]
        m = target[idx] & (keep.random(cmv.shape, dtype=np.float32) < p_tr)
        XS.append(np.column_stack([np.asarray(F[k])[m] for k in cols]).astype(np.float32))
        YV.append(((Vr[idx] / vf[None, None, :])[m] - F['tv'][m]).astype(np.float32))
        YQ.append(((Qr[idx] / ln[None, None, :])[m] - F['tq'][m]).astype(np.float32))
        del F
        gc.collect()
    X = np.concatenate(XS)
    yv = np.concatenate(YV)
    yq = np.concatenate(YQ)
    ok = np.isfinite(yv) & np.isfinite(yq)
    np.savez(path, X=X[ok], yv=yv[ok], yq=yq[ok], cols=np.array(cols))
    log(panel, 'train rows', int(ok.sum()), 'p_tr', round(p_tr, 4))
    return dict(X=X[ok], yv=yv[ok], yq=yq[ok], cols=cols), hv, hq


def month(panel, z, x, split):
    days, o = sorted_days(z[f'{split}_days'])
    xd, ox = sorted_days(x[f'{split}_days'])
    assert (xd == days).all()
    Vm = z[f'{split}_Vm'][o].astype(np.float32)
    Fm = z[f'{split}_Fm'][o].astype(np.float32)
    Em = z[f'{split}_Em'][o]
    obs = Em & np.isfinite(Vm) & np.isfinite(Fm)
    mv = np.where(np.isfinite(Vm), Vm, x[f'{split}_Vmx'][ox].astype(np.float32))
    mq = np.where(np.isfinite(Fm), Fm, x[f'{split}_Fmx'][ox].astype(np.float32))
    reg = np.asarray(t1lab.regime_of_dates(panel, pd.Series(days))).astype(str)
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    di = {d: i for i, d in enumerate(days)}
    rr = np.zeros(Vm.shape[:2], bool)
    for s in z[f'{split}_wi_forecast_origin'].astype(str):
        ts = pd.Timestamp(str(s))
        rr[di[ts.strftime('%Y-%m-%d')], ts.hour * 12 + ts.minute // 5 + 1: ts.hour * 12 + ts.minute // 5 + 1 + GAP] = True
    assert not np.isfinite(mv[rr]).any()
    return dict(days=days, mv=mv, mq=mq, Em=Em, obs=obs, reg=reg, dow=dow, rr=rr)


def candidates(M, vf):
    """vectorised scan; same published rule (coverage, queue in T+1..T+6, onset/ongoing, persistence <= 0.9, real buffer)."""
    obs, mv, Em, rr = M['obs'], M['mv'], M['Em'], M['rr']
    Qm = obs & (mv <= 0.6 * vf[None, None, :])
    D = mv.shape[0]
    out = []
    for d in range(D):
        rcum = np.r_[0, np.cumsum(rr[d])]
        cov = np.array([Em[d, t - HIST:t].mean() for t in range(HIST, 288 - GAP)])
        for j, t in enumerate(range(HIST, 288 - GAP)):
            lo, hi = max(t + 1 - GAP, 0), min(t + 2 * GAP + 1, 288)
            if rcum[hi] - rcum[lo] > 0 or cov[j] < 0.7:
                continue
            fut = Qm[d, t + 1:t + 1 + HOR]
            if not fut.any():
                continue
            if (Qm[d, t - HIST:t].sum(0) >= 2).any():
                now = Qm[d, t][None, :]
                mf = obs[d, t + 1:t + 1 + HOR]
                uni = ((fut | now) & mf).sum()
                inter = (fut & now & mf).sum()
                if (inter / uni if uni else 1.0) > 0.9:
                    continue
                out.append((d, t, 1))
            else:
                out.append((d, t, 0))
    return out


def select(cands, n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cands))
    taken = []
    for want_c, cap in ((0, n // 2), (1, n), (0, n)):
        for i in perm:
            if len(taken) >= cap:
                break
            d, t, c = cands[i]
            if c != want_c or (d, t, c) in taken:
                continue
            if all(dd != d or abs(tt - t) >= 2 * GAP for dd, tt, _ in taken):
                taken.append((d, t, c))
    return sorted(taken)


def lgbp(seed, leaves, lr):
    return dict(objective='regression', learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100, max_bin=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                num_threads=THREADS, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)


def run_panel(panel, folds, leaves, lr, trees, p, seed, sel_seed, nsim, tag):
    z, x, st = load(panel)
    TR, hv, hq = build_train(panel, z, x, st, p)
    cols = TR['cols']
    vf = st.vf.to_numpy()
    ln = st.lanes.to_numpy()
    MS = {sp: month(panel, z, x, sp) for sp in ('validation', 'private')}
    SIM = {}
    for si, sp in enumerate(('validation', 'private')):
        c = candidates(MS[sp], vf)
        s = select(c, nsim, sel_seed + si)
        half = np.random.default_rng(sel_seed + 50 + si).permutation(len(s)) % 2
        SIM[sp] = (s, half)
        log(panel, sp, 'cands', len(c), 'onset', sum(q[2] == 0 for q in c), '| selected', len(s), 'onset', sum(q[2] == 0 for q in s))
    for fold in folds:
        TXr, TYv, TYq, DXr, DYv, DYq = [], [], [], [], [], []
        EV = []
        for si, sp in enumerate(('validation', 'private')):
            M = MS[sp]
            obs, shape = M['obs'], M['obs'].shape
            sims, half = SIM[sp]
            EBr = np.zeros(shape[:2], bool)
            TBr = EBr.copy()
            evid = np.full(shape[:2], -1, np.int32)
            kstep = np.zeros(shape[:2], np.int8)
            SHr = np.zeros(shape[:2], bool)
            hvis = np.zeros(len(sims))
            for k, (d, t, cnd) in enumerate(sims):
                if half[k] == fold:
                    EBr[d, t + 1:t + 1 + GAP] = True
                    evid[d, t + 1:t + 1 + GAP] = k
                    kstep[d, t + 1:t + 1 + GAP] = np.arange(1, GAP + 1)
                    SHr[d, t - HIST:t + 1] = True
                else:
                    TBr[d, t + 1:t + 1 + GAP] = True
                hvis[k] = obs[d, t - HIST:t].sum() / max(1, M['Em'][d, t - HIST:t].sum())
            assert not (EBr & TBr).any() and not (EBr & M['rr']).any() and not (TBr & M['rr']).any()
            EB = np.broadcast_to(EBr[:, :, None], shape)
            TB = np.broadcast_to(TBr[:, :, None], shape)
            HO = obs & ~(EB | TB) & (np.random.default_rng(777 + si).random(shape) < 0.03)
            HP = (obs & (np.random.default_rng(7).random(shape) < 0.15)) if sp == 'validation' else np.zeros(shape, bool)
            hold = HP | HO | (EB & obs)
            vf3 = vf[None, None, :]
            ln3 = ln[None, None, :]
            kp = np.random.default_rng(3000 + 10 * fold + si)
            for r in range(prod.TR_ROUNDS):
                hid = obs & ~hold & (np.random.default_rng(6000 + 10 * r + si).random(shape) < prod.TR_HIDE)
                gone = hid | hold | EB
                F = feats(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg'], st, hv, hq)
                m = hid & (kp.random(shape) < p)
                TXr.append(np.column_stack([np.asarray(F[k])[m] for k in cols]).astype(np.float32))
                TYv.append(((M['mv'] / vf3)[m] - F['tv'][m]).astype(np.float32))
                TYq.append(((M['mq'] / ln3)[m] - F['tq'][m]).astype(np.float32))
                del F
                gc.collect()
            gone = TB | hold | EB
            F = feats(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg'], st, hv, hq)
            m = obs & TB & ~hold & (kp.random(shape) < min(1.0, 2 * p))
            assert not (m & EB).any() and not (m & hold).any()
            DXr.append(np.column_stack([np.asarray(F[k])[m] for k in cols]).astype(np.float32))
            DYv.append(((M['mv'] / vf3)[m] - F['tv'][m]).astype(np.float32))
            DYq.append(((M['mq'] / ln3)[m] - F['tq'][m]).astype(np.float32))
            ndark = int(m.sum())
            del F
            gc.collect()
            SH = obs & np.broadcast_to(SHr[:, :, None], shape) & ~EB & (np.random.default_rng(888 + si).random(shape) < 0.5)
            for view, gone in (('std', HO | EB), ('sparse', HO | EB | SH)):
                F = feats(np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq']), M['dow'], M['reg'], st, hv, hq)
                kinds = (('blk', EB & obs), ('ord', HO)) if view == 'std' else (('blk', EB & obs),)
                for kind, m in kinds:
                    d_, t_, s_ = np.nonzero(m)
                    ev = evid[d_, t_]
                    EV.append(dict(split=sp, view=view, kind=kind,
                                   X=np.column_stack([np.asarray(F[k])[m] for k in cols]).astype(np.float32),
                                   tv=F['tv'][m].astype(np.float64), tq=F['tq'][m].astype(np.float64),
                                   vt=M['mv'][m].astype(np.float64), qt=M['mq'][m].astype(np.float64),
                                   vf=vf[s_], ln=ln[s_], d=d_, t=t_, s=s_, reg=M['reg'][d_], ev=ev, k=kstep[d_, t_],
                                   cond=np.array([sims[e][2] if e >= 0 else -1 for e in ev]),
                                   hvis=np.array([hvis[e] if e >= 0 else np.nan for e in ev])))
                del F
                gc.collect()
            log(panel, 'fold', fold, sp, 'eval blk cells', int((EB & obs).sum()), 'ord', int(HO.sum()), 'dark rows', ndark,
                'sparse extra hidden', int(SH.sum()))
        Xb = np.concatenate([TR['X']] + TXr)
        Yvb = np.concatenate([TR['yv']] + TYv)
        Yqb = np.concatenate([TR['yq']] + TYq)
        Xd = np.concatenate(DXr)
        Yvd = np.concatenate(DYv)
        Yqd = np.concatenate(DYq)
        okb = np.isfinite(Yvb) & np.isfinite(Yqb)
        okd = np.isfinite(Yvd) & np.isfinite(Yqd)
        Xb, Yvb, Yqb = Xb[okb], Yvb[okb], Yqb[okb]
        Xd, Yvd, Yqd = Xd[okd], Yvd[okd], Yqd[okd]
        del TXr, DXr
        gc.collect()
        log(panel, 'fold', fold, 'base rows', len(Yvb), 'dark extra', len(Yvd))
        P = {}
        for arm in ('base', 'dark'):
            if arm == 'base':
                XA, YV, YQ = Xb, Yvb, Yqb
            else:
                XA, YV, YQ = np.concatenate([Xb, Xd]), np.concatenate([Yvb, Yvd]), np.concatenate([Yqb, Yqd])
            t0 = time.time()
            prm = lgbp(seed, leaves, lr)
            mv_ = lgb.train(prm, lgb.Dataset(XA, YV), trees)
            mq_ = lgb.train(prm, lgb.Dataset(XA, YQ), trees)
            for j, E in enumerate(EV):
                P[(arm, j)] = (np.clip((E['tv'] + mv_.predict(E['X'])) * E['vf'], 3, 1.05 * E['vf']),
                               np.maximum((E['tq'] + mq_.predict(E['X'])) * E['ln'], 20))
            log(panel, 'fold', fold, arm, 'rows', len(YV), f'{time.time() - t0:.0f}s')
            del XA, YV, YQ, mv_, mq_
            gc.collect()
        out = {}
        for j, E in enumerate(EV):
            pre = f'{E["split"]}|{E["view"]}|{E["kind"]}'
            for c in ('vt', 'qt', 'ln', 'vf', 'd', 't', 's', 'reg', 'ev', 'k', 'cond', 'hvis'):
                out[f'{pre}|{c}'] = E[c]
            for arm in ('base', 'dark'):
                out[f'{pre}|{arm}|v'] = P[(arm, j)][0].astype(np.float32)
                out[f'{pre}|{arm}|q'] = P[(arm, j)][1].astype(np.float32)
            rv = {a: np.sqrt(np.mean((P[(a, j)][0] - E['vt']) ** 2)) for a in ('base', 'dark')}
            rq = {a: np.sqrt(np.mean(((P[(a, j)][1] - E['qt']) / E['ln']) ** 2)) for a in ('base', 'dark')}
            log(panel, 'fold', fold, pre, 'n', len(E['vt']), '| speed base %.3f dark %.3f (d %+.3f) | flow/lane base %.2f dark %.2f'
                % (rv['base'], rv['dark'], rv['dark'] - rv['base'], rq['base'], rq['dark']))
        info = dict(panel=panel, fold=fold, leaves=leaves, lr=lr, trees=trees, p=p, seed=seed, sel=sel_seed, nsim=nsim,
                    sims={sp: [list(map(int, q)) for q in SIM[sp][0]] for sp in SIM},
                    half={sp: SIM[sp][1].tolist() for sp in SIM})
        np.savez_compressed(f'{OUTD}/{panel}_f{fold}{tag}.npz', info=json.dumps(info), **out)
        del EV, P, out
        gc.collect()


if __name__ == '__main__':
    panels = sys.argv[1].split(',')
    kw = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
    folds = tuple(int(v) for v in kw.get('folds', '0,1').split(','))
    for pn in panels:
        run_panel(pn, folds, int(kw.get('leaves', 255)), float(kw.get('lr', 0.1)), int(kw.get('trees', 450)),
                  float(kw.get('p', 0.35)), int(kw.get('seed', 21)), int(kw.get('sel', 4242)), int(kw.get('nsim', 48)),
                  kw.get('tag', ''))
    log('done')
