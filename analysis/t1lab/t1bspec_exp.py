"""t1bspec: Task 1 BLACKOUT SPECIALIST with selection-faithful timing (LEVERS_2026-10-07 lever 3 / E3).

Machinery = t1ramp_exp.py (same panels, data, train rows, transductive rounds, LightGBM, J-proxy bookkeeping), changed so
that the simulated blackouts mimic the OFFICIAL placement and the specialist sees the timing:
  * EVALUATION blackouts (and the specialist's training blackouts) = selection-faithful origins on the masked month:
      onset   = first crossing: t1dark onset candidate (coverage >= 0.7, no established queue in T-12..T-1) with NO observed
                queued link in T+1..T+5 and an observed queued link at T+6 (the official onset truth queues exactly at T+30);
      ongoing = run-starts of the faithful ongoing map (coverage, established observed queue, a queued time-filled cell in
                T+1..T+6, persistence of the time-filled state <= 0.9 = lab.candidates RULE_FAITHFUL / pmarch select()).
    Official proportions: 5 onset + 5 ongoing per panel-month -> balanced selection (sim_select, half onset), >= 2*GAP
    apart, alternating halves A / B; fold f evaluates on one half and trains on the other (as t1dark / t1ramp).
  * the CURRENT member's training blackouts (random selector-like placement, t1dark rule) are drawn per fold from the
    t1dark candidates >= 2*GAP away from the fold's evaluation blackouts (n_sim/2 per month, half onset).
  * 30 timing columns BCOLS (bspec_feats; NaN outside blackout rows): condition flag (0 onset / 1 ongoing; official
    window_index condition at deployment, simulated type in the lab / pseudo cond on the train split), slot offset k,
    "breakdown expected" flag (onset & k >= 6), boundary values (normalised speed / per-lane flow) just before T and just
    after T+18 at the own station and +-2 stations ORIENTED (u = upstream, d = downstream), queued share of the corridor at
    both boundaries, onset-aware piecewise interpolation minus the linear one (speed, flow), cumulative net ramp flow over
    the gap and on/off-ramp flow relative to slot T.
  * train split: the darkened pseudo gap rows (tfb-t2lab-full pseudo origins = official-like replica windows, with cond)
    get the timing columns; every other row NaN.
Arms (seed 9 unless noted; ramp columns everywhere except base):
  base, base2(seed 10)  production bag members (no ramp, no blackout rows)
  dR, dR2(seed 10)      CURRENT ramp member: random-placement training blackouts, no timing columns
  dF                    faithful-placement training blackouts, no timing columns          (one factor: placement)
  bsR                   random-placement training blackouts + timing columns              (one factor: features)
  bs, bs2(seed 10)      SPECIALIST: faithful placement + timing columns
Usage: t1bspec_exp.py all <PANEL[,PANEL]> [n_trees=600] [p=0.35] [n_sim=48]
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t1ramp_exp as R0  # noqa: E402
from t1ramp_exp import (prod, View, feats, RCOLS, load_panel, train_cube, profile, month_cube, sim_candidates,  # noqa: E402
                        sim_select, ramp_cubes, rsub, lgb_params, direction, GAP, H, F, NOPROF, SPLITS)
import lightgbm as lgb  # noqa: E402

TRACK = 't1bspec'
RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', TRACK)
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', TRACK)
os.makedirs(RES, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
SMOKE = os.environ.get('T1BSPEC_SMOKE', '') == '1'
SFX = '_smoke' if SMOKE else ''
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ================================================================== BSPEC BLOCK (copied verbatim into the kernel)
BS_OFFS = (('0', 0), ('u1', -1), ('u2', -2), ('d1', 1), ('d2', 2))
BCOLS = (['bs_cond', 'bs_k', 'bs_brk'] + [f'bs_{w}_{ch}{o}' for w in ('pre', 'post') for ch in 'vq' for o, _ in BS_OFFS]
         + ['bs_qsh_pre', 'bs_qsh_post', 'bs_pwd_v', 'bs_pwd_q', 'bs_rnet_cum', 'bs_ron_rel', 'bs_roff_rel'])
N_BS = len(BCOLS)
BS_GAP, BS_ONSET_K = 18, 6                       # blackout rows T+1..T+18; official onset queue first at T+6 (T+30 min)


def bs_edge(A, d, t0, step, lim=3):
    """first finite value of A[d, t0 + step * j, :] for j = 0..lim-1 (per station; NaN if none)."""
    out = np.full(A.shape[2], np.nan, np.float32)
    for j in range(lim):
        t = t0 + step * j
        if 0 <= t < A.shape[1]:
            a = A[d, t]; m = np.isnan(out) & np.isfinite(a); out[m] = a[m]
    return out


def bs_orient(x, off):
    """value at station index s + off (NaN outside)."""
    y = np.full_like(x, np.nan)
    if off > 0:
        y[:-off] = x[off:]
    elif off < 0:
        y[-off:] = x[:off]
    else:
        y[:] = x
    return y


def bspec_feats(mv, mq, origins, vf, ln, dirn, tv, tq, rp_on, rp_off, rp_net):
    """timing columns of the blackout rows T+1..T+BS_GAP of every origin (d, T, c) (c = 0 onset / 1 ongoing); NaN elsewhere.
    mv / mq: the view (blackout rows blank), vph / km/h; vf, ln per station; dirn = +1 if traffic flows toward increasing
    station index; tv / tq / rp_* = production feature cubes (normalised per vf / per lane)."""
    D, T, S = mv.shape
    out = {k: np.full((D, T, S), np.nan, np.float32) for k in BCOLS}
    with np.errstate(all='ignore'):
        nv = mv / vf[None, None, :]; nq = mq / ln[None, None, :]
        for d, t0, c in origins:
            d, t0, c = int(d), int(t0), int(c)
            kk = np.arange(1, BS_GAP + 1); kk = kk[t0 + kk < T]
            if len(kk) == 0:
                continue
            rows = t0 + kk; K = kk[:, None].astype(np.float32)
            pre = dict(v=bs_edge(nv, d, t0, -1), q=bs_edge(nq, d, t0, -1))
            post = dict(v=bs_edge(nv, d, t0 + BS_GAP + 1, 1), q=bs_edge(nq, d, t0 + BS_GAP + 1, 1))
            out['bs_cond'][d, rows] = c
            out['bs_k'][d, rows] = K
            out['bs_brk'][d, rows] = ((c == 0) & (K >= BS_ONSET_K)).astype(np.float32)
            for w, Dd in (('pre', pre), ('post', post)):
                for ch in 'vq':
                    for o, off in BS_OFFS:
                        out[f'bs_{w}_{ch}{o}'][d, rows] = bs_orient(Dd[ch], off * dirn)[None, :]
                fin = np.isfinite(Dd['v'])
                out[f'bs_qsh_{w}'][d, rows] = float((Dd['v'][fin] <= 0.6).sum() / max(1, fin.sum()))
            frac = np.clip((K - (BS_ONSET_K - 1)) / (BS_GAP + 1 - (BS_ONSET_K - 1)), 0, 1) if c == 0 else K / (BS_GAP + 1)
            for ch, tx in (('v', tv), ('q', tq)):
                a, b = pre[ch][None, :], post[ch][None, :]
                base = tx[d, rows]
                pw = a + (b - a) * frac
                out[f'bs_pwd_{ch}'][d, rows] = np.where(np.isfinite(pw), pw - base, 0.0)
            out['bs_rnet_cum'][d, rows] = np.cumsum(np.nan_to_num(rp_net[d, rows]), 0) * (5.0 / 60.0)
            out['bs_ron_rel'][d, rows] = (rp_on[d, rows] + 1.0) / (rp_on[d, t0][None, :] + 1.0) - 1.0
            out['bs_roff_rel'][d, rows] = (rp_off[d, rows] + 1.0) / (rp_off[d, t0][None, :] + 1.0) - 1.0
    return out


def bs_fill_time36(V, maxgap=36):
    """linear interpolation along time within each day for interior gaps <= maxgap slots (SELECTION ONLY)."""
    out = V.copy(); D, T, L = V.shape; idx = np.arange(T)
    for d in range(D):
        okd = np.isfinite(V[d])
        for l in np.where(okd.any(0) & ~okd.all(0))[0]:
            ok = okd[:, l]
            if ok.sum() < 2:
                continue
            yi = np.interp(idx, idx[ok], V[d, ok, l])
            prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
            m = (~ok) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < T)
            out[d, m, l] = yi[m]
    return out


def bs_faithful_cands(mv, obs, Em, rr, vf, gap=18, h=12, f=6):
    """selection-faithful blackout origins (d, T, c) on a released masked month, >= gap slots from the blank rows rr:
    c = 0 onset FIRST CROSSING (coverage >= 0.7 in T-h..T-1, no established observed queue in the history, no observed
    queued cell in T+1..T+5, an observed queued cell at T+6); c = 1 ongoing RUN-STARTS of the faithful map (coverage,
    established observed queue, a queued time-filled cell in T+1..T+f, persistence of the time-filled state <= 0.9)."""
    cut = 0.6 * vf[None, None, :]
    Qm = obs & (mv <= cut)
    V = np.where(obs, mv, np.nan).astype(np.float32)
    VF = bs_fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= cut)
    D = mv.shape[0]
    og = np.zeros((D, 288), bool); out = []
    for d in range(D):
        for t in range(h, 288 - gap):
            if Em[d, t - h:t].mean() < 0.7:
                continue
            Qh = Qm[d, t - h:t]
            if Qh.sum() >= 2 and (Qh.sum(0) >= 2).any():
                Tf = QF[d, t + 1:t + 1 + f]; Mf = MF[d, t + 1:t + 1 + f]
                if not Tf.any():
                    continue
                S_ = np.repeat(QF[d, t][None], f, 0)
                u = ((Tf | S_) & Mf).sum(); i = (Tf & S_ & Mf).sum()
                og[d, t] = (1.0 if u == 0 else i / u) <= 0.9
            elif (not Qm[d, t + 1:t + f].any()) and Qm[d, t + f].any():
                if not rr[d, max(t + 1 - gap, 0):min(t + 2 * gap + 1, 288)].any():
                    out.append((d, t, 0))
    for d, t in zip(*np.where(og)):
        if not og[d, t - 1] and not rr[d, max(t + 1 - gap, 0):min(t + 2 * gap + 1, 288)].any():
            out.append((int(d), int(t), 1))
    return sorted(out)
# ================================================================== END BSPEC BLOCK


def add_bs(Fd, mv, mq, origins, st, dirn):
    Fd.update(bspec_feats(mv, mq, origins, st.vf.to_numpy(), st.lanes.to_numpy(), dirn, Fd['tv'], Fd['tq'],
                          Fd['rp_on'], Fd['rp_off'], Fd['rp_net']))
    return Fd


def away(cands, ref, gap=2 * GAP):
    rs = {}
    for d, t, _ in ref:
        rs.setdefault(int(d), []).append(int(t))
    return [c for c in cands if all(abs(c[1] - t) >= gap for t in rs.get(int(c[0]), []))]


# ------------------------------------------------------------------ prep
def prep(panel, p=0.35, n_sim=48):
    z, x, st = load_panel(panel)
    f_rows = f'{CACHE}/{panel}_train_p{p}{SFX}.npz'
    f_meta = f'{CACHE}/{panel}_meta{SFX}.npz'
    T = train_cube(panel, z, x)
    mv = np.where(T['target'], np.nan, T['V']); mq = np.where(T['target'], np.nan, T['Q'])
    hv, hq = profile(mv, mq, T['dow'])
    pz = np.load(os.path.join(ROOT, 'kaggle_jobs', 'out', 't2lab_full', f'pseudo_{panel}.npz'), allow_pickle=True)
    origins = [(int(d), int(t), 0 if str(c) == 'queue_onset' else 1) for d, t, c in zip(pz['d'], pz['t'], pz['cond'])]
    vf = st.vf.to_numpy(); ln = st.lanes.to_numpy(); dirn = direction(panel)
    # how faithful are the train pseudo onset origins (full train truth)?
    Qf = T['elig'] & (T['V'] <= 0.6 * vf[None, None, :])
    on = [(d, t) for d, t, c in origins if c == 0]
    fc = sum(bool(Qf[d, t + 6].any() and not Qf[d, t + 1:t + 6].any()) for d, t in on)
    log(panel, 'train pseudo origins', len(origins), 'onset', len(on), 'first-crossing at T+6 (truth)', fc)
    RT = ramp_cubes(panel, z, 'train')
    if not os.path.exists(f_rows):
        D = len(T['days']); chunks = np.array_split(np.arange(D), 7)
        p_tr = p * min(1.0, 2.4e6 / max(1, int(T['target'].sum())))
        if SMOKE:
            chunks = chunks[:1]
        rng = np.random.default_rng(9)
        XS, YV, YQ, GP = [], [], [], []
        cols = None
        for ci, idx in enumerate(chunks):
            cmv, cmq = mv[idx].copy(), mq[idx].copy(); gapm = np.zeros(cmv.shape, bool)
            pos = {int(dd): j for j, dd in enumerate(idx)}; org = []
            for dd, t, c in origins:
                if dd in pos:
                    j = pos[dd]; cmv[j, t + 1:t + 1 + GAP] = np.nan; cmq[j, t + 1:t + 1 + GAP] = np.nan; gapm[j, t + 1:t + 1 + GAP] = True
                    org.append((j, t, c))
            Fd = feats(View(cmv, cmq, T['dow'][idx], T['reg'][idx]), st, hv, hq, rsub(RT, idx))
            add_bs(Fd, cmv, cmq, org, st, dirn)
            if cols is None:
                cols = [k for k in Fd if k not in NOPROF]
                assert cols[-N_BS:] == BCOLS and cols[-N_BS - len(RCOLS):-N_BS] == RCOLS
            m = T['target'][idx] & (rng.random(cmv.shape, dtype=np.float32) < p_tr)
            yv = (T['V'][idx] / vf[None, None, :])[m] - Fd['tv'][m]; yq = (T['Q'][idx] / ln[None, None, :])[m] - Fd['tq'][m]
            XS.append(np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32))
            YV.append(yv.astype(np.float32)); YQ.append(yq.astype(np.float32)); GP.append(gapm[m])
            del Fd; gc.collect()
            log(panel, 'train chunk', ci, 'rows', int(m.sum()), 'in gaps', int(gapm[m].sum()),
                'bs finite in gaps', round(float(np.isfinite(XS[-1][GP[-1], -N_BS + 3]).mean()), 3) if GP[-1].any() else '-')
        X = np.concatenate(XS); yv = np.concatenate(YV); yq = np.concatenate(YQ); gp = np.concatenate(GP)
        ok = np.isfinite(yv) & np.isfinite(yq)
        assert not np.isfinite(X[ok][~gp[ok]][:, -N_BS:]).any(), 'timing columns must be NaN outside the gaps'
        fr = f'{R0.CACHE}/{panel}_train_p{p}{SFX}.npz'
        if os.path.exists(fr):                                          # paired with t1ramp: identical rows / base columns
            Rr = np.load(fr, allow_pickle=True)
            assert np.array_equal(Rr['X'], X[ok][:, :-N_BS], equal_nan=True) and np.array_equal(Rr['yv'], yv[ok]), 'rows differ from t1ramp'
            log(panel, 'train rows identical to the t1ramp cache (base + ramp columns)')
        np.savez(f_rows, X=X[ok], yv=yv[ok], yq=yq[ok], gap=gp[ok], cols=np.array(cols))
        log(panel, 'train rows', int(ok.sum()), '| in gaps', int(gp[ok].sum()))
        del X, XS
    meta = {}
    for si, split in enumerate(SPLITS):
        M = month_cube(panel, z, x, split)
        candR = sim_candidates(M, vf)
        candF = bs_faithful_cands(M['mv'], M['obs'], M['Em'], M['realrows'], vf, GAP, H, F)
        selF = sim_select(candF, n_sim, seed=200 + si)
        meta[f'{split}_simF'] = np.array(selF, np.int64).reshape(-1, 3)
        half = np.arange(len(selF)) % 2
        for fold in (0, 1):
            ev = [s for k, s in enumerate(selF) if half[k] == (1 if fold == 0 else 0)]
            selR = sim_select(away(candR, ev), n_sim // 2, seed=300 + 10 * fold + si)
            meta[f'{split}_simR{fold}'] = np.array(selR, np.int64).reshape(-1, 3)
        targets = M['Em'] & ~M['obs']
        for r in ('R1', 'R2', 'R3'):
            dm = M['reg'] == r
            inside = float((M['Em'] & M['realrows'][:, :, None])[dm].sum())
            outside = float((targets & ~M['realrows'][:, :, None])[dm].sum())
            meta[f'{split}_ntgt_{r}'] = outside + prod.RATE[r] * inside
            meta[f'{split}_nblk_{r}'] = prod.RATE[r] * inside
        log(panel, split, 'faithful candidates', len(candF), '(onset FC', sum(c[2] == 0 for c in candF), ', ongoing RS',
            sum(c[2] == 1 for c in candF), ') selected', len(selF), '(onset', sum(c[2] == 0 for c in selF), ') | random candidates',
            len(candR), 'selected per fold', [len(meta[f'{split}_simR{f}']) for f in (0, 1)])
    np.savez(f_meta, **meta, hv=hv, hq=hq)


# ------------------------------------------------------------------ one fold
ARMS = {  # name: (columns, blackout rows, seed index)
    'base': ('b', None, 0), 'base2': ('b', None, 1),
    'dR': ('r', 'R', 0), 'dR2': ('r', 'R', 1), 'dF': ('r', 'F', 0),
    'bsR': ('a', 'R', 0), 'bs': ('a', 'F', 0), 'bs2': ('a', 'F', 1),
}


def run_fold(panel, fold, n_trees=600, p=0.35, lr=0.08, seeds=(9, 10), arms=tuple(ARMS)):
    z, x, st = load_panel(panel)
    R = np.load(f'{CACHE}/{panel}_train_p{p}{SFX}.npz', allow_pickle=True)
    meta = np.load(f'{CACHE}/{panel}_meta{SFX}.npz', allow_pickle=True)
    cols = list(R['cols']); hv, hq = meta['hv'], meta['hq']
    nb = len(cols) - len(RCOLS) - N_BS
    CS = dict(b=slice(0, nb), r=slice(0, nb + len(RCOLS)), a=slice(None))
    vf = st.vf.to_numpy(); ln = st.lanes.to_numpy(); Ls = st.length_km.to_numpy(); dirn = direction(panel)
    kc_l = st.critical_density.to_numpy() / ln; kj_l = st.k_jam.to_numpy() / ln; cap_l = st.cap.to_numpy() / ln
    p_dark = min(1.0, 2 * p)
    SD = {}; EV = {}
    TXr, TYVr, TYQr = [], [], []
    DK = {k: ([], [], []) for k in ('R', 'F')}
    info = dict(panel=panel, fold=fold, n_trees=n_trees, lr=lr, p=p, p_dark=p_dark, seeds=list(seeds))

    def rows_of(sims, shape):
        r = np.zeros(shape[:2], bool)
        for d, t, _ in sims:
            r[d, t + 1:t + 1 + GAP] = True
        return r
    for si, split in enumerate(SPLITS):
        M = month_cube(panel, z, x, split)
        obs = M['obs']; shape = obs.shape
        RM = ramp_cubes(panel, z, split, order=np.argsort(z[f'{split}_days'].astype(str)))
        simF = meta[f'{split}_simF']
        half = np.arange(len(simF)) % 2; eval_half = 1 if fold == 0 else 0
        simE = [tuple(s) for k, s in enumerate(simF) if half[k] == eval_half]
        simTF = [tuple(s) for k, s in enumerate(simF) if half[k] != eval_half]
        simTR = [tuple(s) for s in meta[f'{split}_simR{fold}']]
        rowsE = rows_of(simE, shape); ev_id = np.full(shape[:2], -1, np.int32)
        for k, (d, t, c) in enumerate(simF):
            if half[k] == eval_half:
                ev_id[d, t + 1:t + 1 + GAP] = k
        allsim = rows_of([tuple(s) for s in simF], shape) | rows_of([tuple(s) for s in meta[f'{split}_simR0']], shape) \
            | rows_of([tuple(s) for s in meta[f'{split}_simR1']], shape)
        EB = np.broadcast_to(rowsE[:, :, None], shape)
        HO = obs & ~np.broadcast_to(allsim[:, :, None], shape) & (np.random.default_rng(500 + si).random(shape) < 0.03)
        HP = (obs & (np.random.default_rng(7).random(shape) < 0.15)) if split == 'validation' else np.zeros(shape, bool)
        hold = HP | HO | (EB & obs)
        vf3 = np.broadcast_to(vf[None, None, :], shape); ln3 = np.broadcast_to(ln[None, None, :], shape)
        rng_keep = np.random.default_rng(900 + 10 * fold + si)

        def view_feats(gone, origins):
            mv_, mq_ = np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq'])
            Fd = feats(View(mv_, mq_, M['dow'], M['reg']), st, hv, hq, RM)
            return add_bs(Fd, mv_, mq_, origins, st, dirn)
        for r in range(prod.TR_ROUNDS):
            hid = obs & ~hold & ~EB & (np.random.default_rng(10000 + 10 * r + (split == 'private')).random(shape) < prod.TR_HIDE)
            Fd = view_feats(hid | hold | EB, [])
            m = hid & (rng_keep.random(shape) < p)
            TXr.append(np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32))
            TYVr.append(((M['mv'] / vf3)[m] - Fd['tv'][m]).astype(np.float32)); TYQr.append(((M['mq'] / ln3)[m] - Fd['tq'][m]).astype(np.float32))
            del Fd; gc.collect()
        for kind, sims in (('R', simTR), ('F', simTF)):              # training blackout rows: random / faithful placement
            TB = np.broadcast_to(rows_of(sims, shape)[:, :, None], shape)
            Fd = view_feats(TB | hold | EB, sims)
            m = obs & TB & ~hold & (rng_keep.random(shape) < p_dark)
            DK[kind][0].append(np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32))
            DK[kind][1].append(((M['mv'] / vf3)[m] - Fd['tv'][m]).astype(np.float32)); DK[kind][2].append(((M['mq'] / ln3)[m] - Fd['tq'][m]).astype(np.float32))
            info[f'{split}_dark{kind}_rows'] = int(m.sum()); info[f'{split}_dark{kind}_n'] = len(sims)
            info[f'{split}_dark{kind}_onset'] = int(sum(s[2] == 0 for s in sims))
            del Fd; gc.collect()
        Fd = view_feats(HO | EB, simE)
        for kind, m in (('blk', EB & obs), ('ord', HO)):
            d_, t_, s_ = np.nonzero(m)
            EV[f'{split}_{kind}'] = dict(X=np.column_stack([np.asarray(Fd[k])[m] for k in cols]).astype(np.float32),
                                         tv=Fd['tv'][m].astype(np.float64), tq=Fd['tq'][m].astype(np.float64),
                                         vt=M['mv'][m].astype(np.float64), qt=M['mq'][m].astype(np.float64),
                                         vf=vf[s_], ln=ln[s_], d=d_, t=t_, s=s_, reg=M['reg'][d_],
                                         ev=ev_id[d_, t_] if kind == 'blk' else d_)
        assert np.isfinite(EV[f'{split}_blk']['X'][:, -N_BS]).all() and not np.isfinite(EV[f'{split}_ord']['X'][:, -N_BS:]).any()
        del Fd; gc.collect()
        Nt = np.where(obs, M['mq'] / np.clip(M['mv'], 1, None) * Ls[None, None, :], 0.0)
        okp = obs[:, :-1] & obs[:, 1:]; dNt = np.abs(Nt[:, 1:] - Nt[:, :-1])
        Ep = M['Em'][:, :-1] & M['Em'][:, 1:]
        ql_t = np.abs(np.where(obs, M['mq'] / ln[None, None, :], 0.0))
        for r in ('R1', 'R2', 'R3'):
            dm = M['reg'] == r
            if dm.any():
                info[f'{split}_lwrden_{r}'] = float(np.where(okp, dNt, 0)[dm].sum() * Ep[dm].sum() / max(1, okp[dm].sum()))
                info[f'{split}_fdden_{r}'] = float(ql_t[dm].sum() * M['Em'][dm].sum() / max(1, obs[dm].sum()))
        SD[split] = dict(Nt=Nt, okp=okp, E=(EB & obs) | HO)
        log(panel, f'fold {fold}', split, 'eval blackouts', len(simE), '(onset', sum(s[2] == 0 for s in simE), ') cells',
            int((EB & obs).sum()), 'ordinary', int(HO.sum()), 'dark rows R/F', info[f'{split}_darkR_rows'], info[f'{split}_darkF_rows'])
    Xr = np.concatenate([R['X']] + TXr); YVr = np.concatenate([R['yv']] + TYVr); YQr = np.concatenate([R['yq']] + TYQr)
    ok_r = np.isfinite(YVr) & np.isfinite(YQr)
    Xr, YVr, YQr = Xr[ok_r], YVr[ok_r], YQr[ok_r]
    DD = {}
    for kind in ('R', 'F'):
        Xd = np.concatenate(DK[kind][0]); YVd = np.concatenate(DK[kind][1]); YQd = np.concatenate(DK[kind][2])
        okd = np.isfinite(YVd) & np.isfinite(YQd)
        DD[kind] = (Xd[okd], YVd[okd], YQd[okd]); info[f'dark{kind}_rows'] = int(okd.sum())
    info.update(train_rows=int(len(R['yv'])), rand_rows=int(sum(len(a) for a in TYVr)))
    log(panel, f'fold {fold}', json.dumps(info))
    del TXr, DK; gc.collect()
    preds = {}
    for arm in arms:
        cset, dk, sidx = ARMS[arm]; cs = CS[cset]
        if dk is None:
            XA, YV, YQ = Xr[:, cs], YVr, YQr
        else:
            XA = np.concatenate([Xr[:, cs], DD[dk][0][:, cs]]); YV = np.concatenate([YVr, DD[dk][1]]); YQ = np.concatenate([YQr, DD[dk][2]])
        t0 = time.time()
        prm = lgb_params(seeds[sidx], lr=lr)
        mv_ = lgb.train(prm, lgb.Dataset(XA, YV, free_raw_data=True), n_trees)
        mq_ = lgb.train(prm, lgb.Dataset(XA, YQ, free_raw_data=True), n_trees)
        if cset == 'a':
            imp = pd.Series(mv_.feature_importance('gain'), index=cols).sort_values(ascending=False)
            imp = imp / imp.sum()
            log(panel, f'fold {fold}', arm, 'timing columns share of speed gain', round(float(imp[BCOLS].sum()), 4),
                'top', imp[BCOLS].head(6).round(4).to_dict())
        for key, E in EV.items():
            XE = E['X'][:, cs]
            pv = np.clip((E['tv'] + mv_.predict(XE)) * E['vf'], 3, 1.05 * E['vf'])
            pq = np.maximum((E['tq'] + mq_.predict(XE)) * E['ln'], 20)
            preds[f'{arm}|{key}|v'] = pv.astype(np.float32); preds[f'{arm}|{key}|q'] = pq.astype(np.float32)
            s_ = E['s']; kl = pq / np.maximum(pv, 1) / E['ln']; ql = pq / E['ln']
            qfd = np.where(kl <= kc_l[s_], vf[s_] * kl, cap_l[s_] / np.maximum(kj_l[s_] - kc_l[s_], 1e-9) * np.maximum(kj_l[s_] - kl, 0))
            preds[f'{arm}|{key}|fd'] = np.abs(ql - qfd).astype(np.float32)
        for split in SPLITS:
            G = SD[split]; NP = G['Nt'].copy(); E3 = G['E']
            for kind in ('blk', 'ord'):
                E = EV[f'{split}_{kind}']
                NP[E['d'], E['t'], E['s']] = preds[f'{arm}|{split}_{kind}|q'] / np.clip(preds[f'{arm}|{split}_{kind}|v'], 1, None) * Ls[E['s']]
            err = np.where(G['okp'], np.abs((NP[:, 1:] - NP[:, :-1]) - (G['Nt'][:, 1:] - G['Nt'][:, :-1])), 0.0)
            lw = np.zeros(NP.shape)
            lw[:, :-1] += np.where(E3[:, :-1], err, 0); lw[:, 1:] += np.where(~E3[:, :-1] & E3[:, 1:], err, 0)
            for kind in ('blk', 'ord'):
                E = EV[f'{split}_{kind}']
                preds[f'{arm}|{split}_{kind}|lw'] = lw[E['d'], E['t'], E['s']].astype(np.float32)
        log(panel, f'fold {fold}', arm, 'rows', len(YV), f'trained+predicted in {time.time() - t0:.0f}s')
        del XA, YV, YQ, mv_, mq_; gc.collect()
    out = {}
    for key, E in EV.items():
        for c in ('vt', 'qt', 'ln', 'd', 't', 's', 'reg', 'ev', 'tv', 'tq', 'vf'):
            out[f'{key}|{c}'] = E[c]
    out.update(preds)
    np.savez_compressed(f'{RES}/{panel}_f{fold}{SFX}.npz', info=json.dumps(info), **out)
    for key, E in EV.items():
        rv = {a: float(np.sqrt(np.mean((preds[f'{a}|{key}|v'] - E['vt']) ** 2))) for a in arms}
        rq = {a: float(np.sqrt(np.mean(((preds[f'{a}|{key}|q'] - E['qt']) / E['ln']) ** 2))) for a in arms}
        log(panel, f'fold {fold}', key, 'n', len(E['vt']), '| speed', ' '.join(f'{a} {rv[a]:.3f}' for a in arms),
            '| flow/lane', ' '.join(f'{a} {rq[a]:.2f}' for a in arms))


if __name__ == '__main__':
    mode, panels = sys.argv[1], sys.argv[2]
    kw = dict(a.split('=') for a in sys.argv[3:] if '=' in a)
    p = float(kw.get('p', 0.35)); n_sim = int(kw.get('n_sim', 48))
    nt, lr = int(kw.get('n_trees', 600)), float(kw.get('lr', 0.08))
    arms = tuple(kw.get('arms', ','.join(ARMS)).split(','))
    for pn in panels.split(','):
        if mode in ('prep', 'all'):
            prep(pn, p=p, n_sim=n_sim)
        if mode == 'all':
            for fold in (0, 1):
                run_fold(pn, fold, n_trees=nt, p=p, lr=lr, arms=arms)
    log('done')
