"""Track pvog: is the PRIVATE ongoing package real under RAW-history deployment?  Shared helpers (new file; nothing existing is edited).

Regimes
  raw     history [T-60, T) = every eligible cell (train cubes, official window_history); slot T through the masked layer.
  masked  history and slot T = the released masked layer (the Task-1 mask hides 50 / 30 / 20 % of the eligible cells i.i.d. per
          cell, one regime per day); labels = visible cells only.  Pseudo-March windows and the March training rows of
          kaggle_jobs/t2_ongoing_pv.py live in this regime.
The feature code is the production one (lab.window_features / add_ramp / prior_feats / to_matrix / ongoing_keep); only the
cubes that feed it are swapped.  val_panel / select / cum_tables / blended_full are copied from exp_ongoing_marchtrain.py.
"""
import gc, json, os, sys, time, types, warnings, zlib
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
ONGOING = lab.ONGOING
RES = f'{lab.ROOT}/analysis/t2lab/results/pvog'
CACHE = f'{lab.CACHE}/pvog'
for _d in (RES, CACHE, f'{RES}/models', f'{RES}/probs'):
    os.makedirs(_d, exist_ok=True)
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
K_BLEND = 4
COLS = {'all': lab.all_cols(), 'np': lab.colset('np'), 'id': lab.colset('id'), 'full': lab.colset('full')}
IX = {k: [COLS['all'].index(c) for c in v] for k, v in COLS.items()}
JPF = [COLS['full'].index(c) for c in PRIOR3]
assert COLS['full'] == COLS['all']
REGIMES = np.array([0.5, 0.7, 0.8])
MD_FINAL = f'{fc.RES}/final'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)


def prm(seed=0):
    return lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=seed)


# ------------------------------------------------------------------ released March month (masked layer)
def val_panel(panel, split='validation'):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z[f'{split}_days']]); o = np.argsort(days)
    V = z[f'{split}_Vm'][o].astype(np.float32); Fq = z[f'{split}_Fm'][o].astype(np.float32); Em = z[f'{split}_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z[f'{split}_ON'][o].astype(np.float32), OFF=z[f'{split}_OFF'][o].astype(np.float32))
    return P, Pv


def select(Pv):
    """faithful ongoing / production-rule onset candidates on a masked month (Pv: V, E, Q visible; Em eligibility)."""
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


def day_regime(Pv):
    """visible share of the eligible cells per day over the non-blanked slots, and the nearest regime (0.5 / 0.7 / 0.8)."""
    E, Em = Pv['E'], Pv['Em']
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    p = np.array([E[d][~blank[d]].sum() / max(Em[d][~blank[d]].sum(), 1) for d in range(E.shape[0])])
    return p, REGIMES[np.abs(p[:, None] - REGIMES[None]).argmin(1)]


def cube_rows(P, wins, prior, hist, labc, dow, ON, OFF, vt='cube', vt_cube=None):
    """production-shaped ongoing rows R for windows [(d, t)], history / slot-T from `hist` (dict V, Fq), labels from `labc` (dict Q, E).
    vt='cube' : slot-T value = hist['V'][d, t] (masked-layer style, as the pseudo-March construction)
    vt='sim62': production training simulation (rng(len(wins)), 62 % of the cells of hist['V'][d, t])."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = hist['V'][d[:, None], idx]; Fh = hist['Fq'][d[:, None], idx]
    if vt == 'cube':
        VT = (hist['V'] if vt_cube is None else vt_cube)[d, t]            # vt_cube: slot T from another cube than the history
    else:
        rng = np.random.default_rng(len(wins))
        VT = np.where(rng.random((len(d), hist['V'].shape[2])) < 0.62, hist['V'][d, t], np.nan)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, dow[d], prior, VT)
        lab.add_ramp(Fe, ON, OFF, d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([ONGOING] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, dow[d], t, h)
        R['rows'].append((h, feats, labc['Q'][d, np.minimum(t + h, 287)], labc['E'][d, np.minimum(t + h, 287)]))
    return R


def val_rows(P, Pv, wins, prior):
    """pseudo-March rows exactly as exp_ongoing_marchtrain.val_rows (masked history, masked slot T, visible labels)."""
    return cube_rows(P, wins, prior, Pv, Pv, Pv['dow'], Pv['ON'], Pv['OFF'], vt='cube')


def cum_tables(Pv):
    """cumulative same-scenario queue / visible counts over days < j by weekday class, slots +-2 (box of 5)."""
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


def blended_full(X, R, meta, dow, prior_tr, cQ, cE, jday=None, k=K_BLEND):
    """full-column matrix with the three prior columns blended per row: (1-a) train prior + a scenario prior, a = n/(n+k).
    jday: per-window index into the cumulative tables (default: the window's own day = scenario days BEFORE that day)."""
    Xf = np.ascontiguousarray(X[:, IX['full']])
    wv, lv, hv = meta[:, 0], meta[:, 1], meta[:, 2]
    d_, t_ = R['d'], R['t']
    jd = d_ if jday is None else np.asarray(jday)
    cr = (dow[d_] >= 5).astype(int)[wv]; dr = jd[wv]; tr_ = t_[wv]; sh = np.minimum(tr_ + hv, 287)
    trv = [prior_tr[cr, sh, lv], prior_tr[cr, tr_, lv], prior_tr[2, sh, lv]]
    for j, (cc, ss) in enumerate(((cr, sh), (cr, tr_), (np.full_like(cr, 2), sh))):
        q = cQ[cc, dr, ss, lv]; ee = cE[cc, dr, ss, lv]
        sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0; a_ = n / (n + k)
        Xf[:, JPF[j]] = (1 - a_) * trv[j] + a_ * sp
    return Xf


class Ev:
    """one evaluation block (panel, set): rc (truth / ev / qlast / win), X [n, all cols] (train prior), Xf (blended prior), meta."""

    def __init__(self, rc, X, Xf, meta, W, L):
        self.rc, self.X, self.Xf, self.meta, self.W, self.L = rc, X, Xf, meta, W, L

    def mat(self, cs, blended=True):
        if cs == 'full':
            return self.Xf if blended else np.ascontiguousarray(self.X[:, IX['full']])
        return np.ascontiguousarray(self.X[:, IX[cs]])

    def p6(self, p):
        out = np.zeros((self.W, F, self.L), np.float32); out[self.meta[:, 0], self.meta[:, 2] - 1, self.meta[:, 1]] = p
        return out


def march_data(cut=15, wm=5.0, eval_from=0, panels=None, verbose=True, prior_k=None):
    """March training rows (days < cut: faith3 at 0.3 wm, run-starts at wm; the F03 recipe) and the pseudo-March evaluation
    blocks offlike_m / runstart_m (windows of days >= eval_from).  Mirrors exp_ongoing_marchtrain.py.
    Returns TR = dict(X [n, all cols] train prior, Xf blended prior, y, w, d, panel) and EV = [Ev, ...]."""
    TR = dict(X=[], Xf=[], y=[], w=[], d=[], panel=[]); EV = []
    for pi, panel in enumerate(panels or lab.PANELS):
        P, Pv = val_panel(panel)
        on, og = select(Pv)
        D = len(Pv['days'])
        prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool) if prior_k is None else (P['block'] != prior_k))
        cQ, cE = cum_tables(Pv)
        pday, rday = day_regime(Pv)
        dd, tt = np.where(og); rs = ~og[dd, tt - 1]
        trw = {}
        for a, b, r in zip(dd.tolist(), tt.tolist(), rs.tolist()):
            if a >= cut:
                continue
            if r:
                trw[(a, b)] = 1.0
            elif b % 3 == 0:
                trw[(a, b)] = 0.3
        if trw:
            wl = sorted(trw)
            R = val_rows(P, Pv, [(a, b, ONGOING) for a, b in wl], prior_tr)
            X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R)); assert list(call) == COLS['all']
            Xf = blended_full(X, R, meta, Pv['dow'], prior_tr, cQ, cE)
            ww = np.array([trw[w] for w in wl], np.float32)[meta[:, 0]] * wm
            TR['X'].append(X[e]); TR['Xf'].append(Xf[e]); TR['y'].append(y[e].astype(np.float32)); TR['w'].append(ww[e])
            TR['d'].append(R['d'][meta[:, 0]][e].astype(np.int16)); TR['panel'].append(np.full(int(e.sum()), pi, np.int8))
            del X, Xf
        g_on, g_og = lab._glist(on), lab._glist(og); s = set()
        for d0 in range(D):
            _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
        sets = {'offlike_m': sorted(w for w in s if w[0] >= eval_from),
                'runstart_m': sorted((a, b) for a, b, r in zip(dd.tolist(), tt.tolist(), rs.tolist()) if r and a >= eval_from)}
        for sname, wl in sets.items():
            if not wl:
                continue
            R = val_rows(P, Pv, [(a, b, ONGOING) for a, b in wl], prior_tr)
            X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
            Xf = blended_full(X, R, meta, Pv['dow'], prior_tr, cQ, cE)
            d_, t_ = R['d'], R['t']
            win = pd.DataFrame(dict(d=d_, t=t_, cond=ONGOING, fold=0, vis=pday[d_], regime=rday[d_],
                                    hist_vis=[float(Pv['E'][a, b - H:b].sum() / max(Pv['Em'][a, b - H:b].sum(), 1)) for a, b in zip(d_, t_)]))
            rc = types.SimpleNamespace(truth=np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                       ev=np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1), qlast=R['qlast'],
                                       win=win, name=sname, panel=panel, cond=ONGOING, L=R['L'])
            EV.append(Ev(rc, X, Xf, meta, R['W'], R['L']))
        if verbose:
            lab.log(panel, 'March training windows', len(trw), '| eval', {k: len(v) for k, v in sets.items()})
        del cQ, cE; lab.drop_panel(panel); gc.collect()
    TR = {k: (np.concatenate(v) if v else None) for k, v in TR.items()}
    return TR, EV


def score_members(EV, P6, weights, arm, win_cols=('vis', 'regime', 'hist_vis')):
    """score a probability blend {member: weight} on every evaluation block; P6[(set, panel, member)] = [W, 6, L]."""
    out = []
    for ev in EV:
        rc = ev.rc
        p = sum(wt * P6[(rc.name, rc.panel, m)] for m, wt in weights.items())
        df = lab.score_probs(rc, p.astype(np.float32), arm=arm, set_name=rc.name)
        for c in win_cols:
            if c in rc.win:
                df[c] = rc.win[c].to_numpy()
        out.append(df)
    return pd.concat(out, ignore_index=True)


def cmp_table(R, base, sets, min_d=None, max_d=None, arms=None, extra=None):
    """paired bootstrap of every arm vs `base` per evaluation set (window -> panel -> family aggregation)."""
    rows = []
    for sname in sets:
        g = R[R.set == sname]
        if min_d is not None:
            g = g[g.d >= min_d]
        if max_d is not None:
            g = g[g.d <= max_d]
        if extra is not None:
            g = g[extra(g)]
        A0 = g[g.arm == base]
        for nm in (arms or list(dict.fromkeys(g.arm))):
            B0 = g[g.arm == nm]
            if not len(B0) or not len(A0):
                continue
            pa = A0.groupby('panel').iou.mean(); pb = B0.groupby('panel').iou.mean()
            if nm == base:
                bt = dict(diff=0.0, lo=0.0, hi=0.0)
            else:
                bt = lab.paired_bootstrap(A0, B0)
            rows.append(dict(set=sname, arm=nm, n=len(B0), iou=lab.aggregate(B0), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                             panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}'))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ masked-layer emulation on the TRAIN cubes
def march_regime_share():
    """empirical share of the day regimes (0.5 / 0.7 / 0.8 visible) over the 8 x 31 released March panel-days."""
    f = f'{CACHE}/regime_share.json'
    if os.path.exists(f):
        return np.array(json.load(open(f))['share'])
    cnt = np.zeros(3)
    for panel in lab.PANELS:
        _, Pv = val_panel(panel)
        _, r = day_regime(Pv)
        for i, x in enumerate(REGIMES):
            cnt[i] += (r == x).sum()
    json.dump(dict(regimes=REGIMES.tolist(), count=cnt.tolist(), share=(cnt / cnt.sum()).tolist()), open(f, 'w'))
    return cnt / cnt.sum()


def mask_cube(P, seed=0):
    """emulated masked layer of a train panel: every day gets a regime p_d drawn from the March regime shares, every observed
    cell stays visible with probability p_d (i.i.d. per cell; speed and flow share the mask).
    Returns dict(V, Fq, E, Q (visible), Em (= raw observed = eligible), p [D])."""
    key = f'mask{seed}'
    if key in P:
        return P[key]
    rng = np.random.default_rng(zlib.crc32(f'{P["panel"]}|{seed}'.encode()))
    D = P['V'].shape[0]
    p = rng.choice(REGIMES, size=D, p=march_regime_share())
    vis = rng.random(P['V'].shape, dtype=np.float32) < p[:, None, None].astype(np.float32)
    V = np.where(vis, P['V'], np.nan).astype(np.float32); Fq = np.where(vis, P['Fq'], np.nan).astype(np.float32)
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    P[key] = dict(V=V, Fq=Fq, E=E, Q=Q, Em=P['E'], p=p)
    return P[key]


def half_masked(win, set_name, panel):
    """deterministic 50 % split of a window frame (True = this window is taken in its MASKED version in the raw + masked mix)."""
    rng = np.random.default_rng(zlib.crc32(f'{panel}|{set_name}|half'.encode()))
    return rng.random(len(win)) < 0.5


def f03_windows(panel):
    """the F03 training windows of one panel for k = 4 (all train days): [(set, index in the lab set, d, t, weight)]."""
    sp = fc.ARMS['F03']; out = []
    for name, sw in sp['train']:
        win = lab.window_set(panel, name); win = win[win.cond == ONGOING].reset_index(drop=True)
        w = float(sw) * np.asarray(sp['wfn'](win, name, panel), np.float32)
        for i in np.where(w > 0)[0]:
            out.append((name, int(i), int(win.d.iloc[i]), int(win.t.iloc[i]), float(w[i])))
    return out


def masked_f03_rows(panel, seed=0, labels='all'):
    """masked-history version of the F03 training rows of one panel (all-train prior, production keep region), cached.
    labels='all': every observed (eligible) cell is a label (same label density as the raw F03 rows).
    Returns dict(X [n, all cols] (memmap), y, w, d, t, half (window taken masked in the mix), rs (run-start set))."""
    tag = f'{CACHE}/f03mask_s{seed}_{labels}_{panel}'
    if not os.path.exists(f'{tag}.npz'):
        P = lab.load_panel(panel); M = mask_cube(P, seed)
        wl = f03_windows(panel)
        prior = lab.prior_table(P, np.ones(len(P['days']), bool))
        labc = dict(Q=P['Q'], E=P['E']) if labels == 'all' else dict(Q=M['Q'], E=M['E'])
        R = cube_rows(P, [(a[2], a[3]) for a in wl], prior, M, labc, P['dow'], P['ON'], P['OFF'], vt='cube')
        X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R)); assert list(call) == COLS['all']
        halves = {}
        for name, _ in fc.ARMS['F03']['train']:
            win = lab.window_set(panel, name); win = win[win.cond == ONGOING].reset_index(drop=True)
            halves[name] = half_masked(win, name, panel)
        hw = np.array([halves[a[0]][a[1]] for a in wl]); ww = np.array([a[4] for a in wl], np.float32)
        rsw = np.array([a[0] == 'runstart' for a in wl])
        wi = meta[:, 0][e]
        tmp = f'{tag}.tmp{os.getpid()}'                                   # atomic write: parallel jobs may build the same cache
        np.save(f'{tmp}.npy', X[e]); os.replace(f'{tmp}.npy', f'{tag}.npy')
        np.savez_compressed(f'{tmp}.npz', y=y[e].astype(np.float32), w=ww[wi], d=R['d'][wi].astype(np.int16), t=R['t'][wi].astype(np.int16),
                            half=hw[wi], rs=rsw[wi], p_day=M['p'], n_windows=len(wl))
        os.replace(f'{tmp}.npz', f'{tag}.npz')
        P.pop(f'mask{seed}', None)
        del X, R; gc.collect()
    z = np.load(f'{tag}.npz')
    out = {k: z[k] for k in z.files}; out['X'] = np.load(f'{tag}.npy', mmap_mode='r')
    return out


def wfn_half(base_wfn, want_masked_half):
    """F03 weight function restricted to one half of the mix split (raw half: want_masked_half=False)."""
    def f(win, set_name, panel):
        w = np.asarray(base_wfn(win, set_name, panel), np.float32).copy()
        hm = half_masked(win, set_name, panel)
        w[hm != want_masked_half] = 0.0
        return w
    return f


# ------------------------------------------------------------------ member probabilities
def predict_members(EV, members):
    """members: {name: (booster, colset, blended)} -> {(set, panel, name): P6 float32 [W, 6, L]} on every evaluation block."""
    P6 = {}
    for ev in EV:
        for nm, (mdl, cs, bl) in members.items():
            P6[(ev.rc.name, ev.rc.panel, nm)] = ev.p6(mdl.predict(ev.mat(cs, bl), num_threads=lab.THREADS))
    return P6


def save_probs(tag, P6):
    pd.to_pickle({k: v.astype(np.float16) for k, v in P6.items()}, f'{RES}/probs/{tag}.pkl')


def load_probs(*tags):
    out = {}
    for tag in tags:
        z = pd.read_pickle(f'{RES}/probs/{tag}.pkl')
        out.update({k: v.astype(np.float32) for k, v in z.items()})
    return out


def fit_plus(Xtr, ytr, wtr, Xm, ym, wm_, cols, n_trees=500, seed=0):
    """LightGBM (lab scale) on train rows (+ optional extra rows)."""
    if Xm is not None and len(Xm):
        X = np.concatenate([Xtr, Xm]); y = np.concatenate([ytr, ym]); w = np.concatenate([wtr, wm_])
    else:
        X, y, w = Xtr, ytr, wtr
    mdl = fc.fit(X, y, w, prm(seed), n_trees, cols)
    return mdl


# ------------------------------------------------------------------ in-train raw-deployment test (fold 0 = target, source month in fold 1)
K0 = 0


def src_days(src0=69):
    return np.arange(src0, src0 + 31)


def source_month(P, src, seed=0):
    """Pv-like dict of the emulated masked layer of the train days `src` (a 31-day 'released month' of the SAME scenario)."""
    M = mask_cube(P, seed)
    return dict(panel=P['panel'], V=M['V'][src], Fq=M['Fq'][src], E=M['E'][src], Q=M['Q'][src], Em=P['E'][src], cut=P['cut'],
                dow=P['dow'][src], days=[P['days'][i] for i in src])


def source_rows(panel, src, kind, wm=5.0, seed=0, k=K0):
    """faithful ongoing windows of the masked source month (selected exactly as the kernel selects March windows: run-starts
    1.0 / t%3 rest 0.3, times wm) as training rows.  kind='masked': masked history + masked slot T + visible labels (the
    kernel's March rows); kind='raw': the SAME windows with raw history, production slot-T simulation and every observed label.
    Prior columns: fold-k train prior; Xf: blended with the masked source days before the window's day (as the kernel)."""
    P = lab.load_panel(panel); M = mask_cube(P, seed)
    prior = lab.prior_table(P, P['block'] != k)
    Pv = source_month(P, src, seed)
    on, og = select(Pv)
    cQ, cE = cum_tables(Pv)
    dd, tt = np.where(og); rs = ~og[dd, tt - 1]
    wl = sorted((int(a), int(b), 1.0 if r else 0.3) for a, b, r in zip(dd, tt, rs) if r or b % 3 == 0)
    if not wl:
        return None
    ww = np.array([c for _, _, c in wl], np.float32)
    wins = [(int(src[a]), b) for a, b, _ in wl]; jl = np.array([a for a, _, _ in wl])
    if kind == 'masked':
        R = cube_rows(P, wins, prior, M, dict(Q=M['Q'], E=M['E']), P['dow'], P['ON'], P['OFF'], vt='cube')
    else:
        R = cube_rows(P, wins, prior, dict(V=P['V'], Fq=P['Fq']), dict(Q=P['Q'], E=P['E']), P['dow'], P['ON'], P['OFF'], vt='sim62')
    X, y, e, meta, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R)); assert list(call) == COLS['all']
    Xf = blended_full(X, R, meta, P['dow'], prior, cQ, cE, jday=jl)
    return dict(X=X[e], Xf=Xf[e], y=y[e].astype(np.float32), w=(ww[meta[:, 0]] * wm)[e], n_windows=len(wl), n_rs=int(rs.sum()))


def target_blocks(src, mask_seeds=(), with_march=True, seed_src=0, k=K0, set_name='offlike', panels=None, sel=None, with_alt=True):
    """evaluation blocks of the fold-k windows of a train set (default: official-like ongoing windows of fold 0):
      'raw0'      raw histories (the lab row cache: production training-style slot-T simulation), truth = observed cells
      'mask<s>'   the same windows with the emulated masked layer of seed s as history / slot T (truth unchanged)
    Ev.alt: full-column matrices with other priors: 'sc' = fold-k train prior blended with the 31 masked source days (right
    scenario, a ~ 0.86), 'wrong' = blended with the released March month (wrong scenario)."""
    out = []
    for panel in panels or lab.PANELS:
        P = lab.load_panel(panel)
        prior = lab.prior_table(P, P['block'] != k)
        rc = lab.rows(panel, set_name, ONGOING)
        widx = np.where(rc.win.fold.to_numpy() == k)[0] if sel is None else np.where(sel(rc.win))[0]
        if not len(widx):
            continue
        win = rc.win.iloc[widx].reset_index(drop=True)
        tabs = {}
        if with_alt:
            cQs, cEs = cum_tables(source_month(P, src, seed_src))
            tabs['sc'] = (cQs, cEs, len(src))
        if with_alt and with_march:
            _, Pvm = val_panel(panel)
            cQm, cEm = cum_tables(Pvm); tabs['wrong'] = (cQm, cEm, len(Pvm['days']))
        loc = np.full(len(rc.win), -1); loc[widx] = np.arange(len(widx))

        def mk(name, X, meta, qlast):
            rcs = types.SimpleNamespace(truth=rc.truth[widx], ev=rc.ev[widx], qlast=qlast, win=win, name=name, panel=panel, cond=ONGOING, L=rc.L)
            ev = Ev(rcs, X, None, meta, len(widx), rc.L)
            Rr = dict(d=win.d.to_numpy(), t=win.t.to_numpy())
            ev.alt = {kk: blended_full(X, Rr, meta, P['dow'], prior, cq, ce, jday=np.full(len(win), j)) for kk, (cq, ce, j) in tabs.items()}
            return ev
        m = rc.keep5[:, k] & np.isin(rc.wi, widx)
        X = rc.matrix(m, COLS['all'], k)
        meta = np.stack([loc[rc.wi[m]], rc.li[m].astype(int), rc.h[m].astype(int)], 1)
        out.append(mk('raw0', X, meta, rc.qlast[widx]))
        for s in mask_seeds:
            M = mask_cube(P, s)
            R = cube_rows(P, list(zip(win.d.tolist(), win.t.tolist())), prior, M, dict(Q=P['Q'], E=P['E']), P['dow'], P['ON'], P['OFF'], vt='cube')
            Xm, _, _, metam, call = lab.to_matrix(R, ONGOING, None, lab.ongoing_keep(R))
            out.append(mk(f'mask{s}', Xm, metam, R['qlast']))
            P.pop(f'mask{s}', None)
        lab.drop_panel(panel); gc.collect()
    return out


def _ev_mat(self, cs, blended=True):
    if cs == 'full':
        if isinstance(blended, str):
            return self.alt[blended]
        return self.Xf if (blended and self.Xf is not None) else np.ascontiguousarray(self.X[:, IX['full']])
    return np.ascontiguousarray(self.X[:, IX[cs]])


Ev.mat = _ev_mat
