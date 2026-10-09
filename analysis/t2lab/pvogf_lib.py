"""TRACK pvogf (E2, 2026-10-09): bring the ogfeat 'all2' signal (103 ongoing columns = 98 static + 5 kx step columns, no tod / dow)
into pick A's April ongoing recipe pvo (kernel tfb-t2ogpvaf = kaggle_jobs/t2_ongoing_pvaf.py: F03 np / id / full bag + the faithful
windows of the released masked March month x WM 5, all-finite inputs, full member fed the same-scenario prior blended over the
March + April days < d).  New file; nothing existing is edited.

PRE-REGISTERED DECISION RULE (written 2026-10-09 10:00 +06, before any pvo-ogf number was computed; not changed afterwards)
  Candidates, each against pvo:
    (a) pvogf  = pvo trained with the 103 ogfeat columns appended to np / id / full, everything else equal;
    (b) pvogfv = per window the pvogf decision unless it adds > 10 cells outside the queue at T (qlast, all 6 steps) vs pvo, else pvo;
    (c) pvotv  = per window the train-only ogfeat np+id decision (T0 lab analog of fs4ogf) unless it adds > 10 cells outside the
                 queue at T vs pvo, else pvo.
  ADOPT a candidate for pick A only if BOTH
    (1) the many-day pseudo-March delta is >= +0.005 IoU with 95 % CI lower bound > 0.  Delta = aggregate(candidate) - aggregate(pvo)
        on the offlike_m windows of March days >= 15 (training March rows = days < 15 at WM 5; full-member prior blended over the
        March days < d), lab scale (255 leaves, lr 0.05, min_data 40, 500 trees), every member's probabilities averaged over seeds 0
        and 1 (same seeds in both arms), aggregation window -> panel -> family -> mean, CI = lab.paired_bootstrap (2000 resamples
        within panel);
    (2) FM-PM private >= +0.0005 (LB-equivalent) when it can be computed.  Footprint = the candidate's lab-emulated April decisions
        (seed-0 lab models trained on the F03 rows + all 31 March days at WM 5, full member fed the March + April days < d prior)
        vs lab-emulated pvo on the 40 official private ongoing windows.  Each changed window is priced with the mean seed-0 many-day
        per-window dIoU of the lab windows (offlike_m + runstart_m, days >= 15) of the same panel (pooled if n < 5), origin +-2 h,
        same sign of the change in predicted cells (all-cell counts), |dn - dn_w| <= max(3, 0.5 |dn_w|) (results/xfer/scripts/
        fpmatch_lib.forecast); FM-PM = 0.15 x the mean over the 40 windows (unchanged windows count 0).
  Reported, not binding: runstart_m many-day delta, per-panel deltas, each seed alone, train OOF (fold 0, pvo recipe) offlike /
  runstart / official-40.  If (a) and (b) both pass, the one with the higher FM-PM private is chosen; (c) only if (a) and (b) fail.

Data and code
  * Feature / row code is the KERNEL code, compiled with ast (main() never runs):
      K  = kaggle_jobs/t2_ogfeat_v27a.py : window_features, add_ramp, prior_*, to_matrix, ongoing_keep, decode, og_extra / og_extra2 /
           og_step (verbatim copies of ogfeat_lib.py), make_rows with the ogfeat columns (occupancy off: no 'O' in P) = pvaf make_rows
           + the 103 columns;
      K2 = kaggle_jobs/t2_ongoing_pvaf.py: scenario, march_faithful, blend_prior_og (file readers patched to the local exports).
  * Train split: all-finite feature cubes (t2x export), F03 lab windows (faith3 rest x 0.3 capped 1500 + run-starts x 1.0).
  * Masked months (March = validation, April = private): visible eligible V / Fq, eligibility Em, all-finite Vf / Ff (= V, else the
    ineligible-valued layer), exactly as the kernel's masked_month.
  * Official windows: history / slot T = eligible + ineligible-valued values (t2x export), as the kernel's load_split with ALLFIN.
"""
import ast, gc, glob, json, os, sys, time, types, warnings
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import pvog_lib as pv

ROOT = lab.ROOT
KV27 = f'{ROOT}/kaggle_jobs/t2_ogfeat_v27a.py'
KPVAF = f'{ROOT}/kaggle_jobs/t2_ongoing_pvaf.py'
XDIR = f'{lab.DATA}/x'
CACHE = f'{lab.CACHE}/pvogf'
RES = f'{ROOT}/analysis/t2lab/results/pvogf'
for _d in (CACHE, RES, f'{RES}/probs', f'{CACHE}/models'):
    os.makedirs(_d, exist_ok=True)
H, F, ONGOING = lab.H, lab.F, lab.ONGOING
CUT = 15                     # many-day regime: March training rows of days < CUT, evaluation windows of days >= CUT
WM = 5.0                     # March window weight multiplier (kernel WM)
PM_SETS = ('offlike_m', 'runstart_m')
CONSTS = {'H', 'F', 'SPACING', 'PARAMS', 'PRIOR_COLS', 'ID_DROP', 'OG_STATIC', 'OG_STEP', 'OG_ADD_PROD', 'CAPD', 'BIG', 'OG_STEP_ALL',
          'ONGOING', 'VARIANTS', 'NB', 'OCC_COLS', 'ALLFIN', 'OCC', 'PART', 'TAG', 'W_RS', 'W_DENSE', 'FILL_GAP', 'PERS_MAX', 'MAX_OG',
          'N_BIG', 'TIME_LIMIT_S', 'WM', 'K_BLEND', 'P_PV', 'N_PV', 'REF_V22', 'REF_BASE', 'REF_PVO', 'VETO_G'}


def compile_kernel(path):
    """top-level functions (except main) + constant assignments of a kernel file, executed in a fresh module (= ogfeat_overlap_v27)."""
    tree = ast.parse(open(path).read()); body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            body.append(node)
        elif isinstance(node, ast.Assign):
            tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if tg and tg <= CONSTS:
                body.append(node)
    Km = types.ModuleType(os.path.basename(path)[:-3])
    Km.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, warnings=warnings, T0=time.time(), SMOKE=False)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), Km.__dict__)
    return Km


K = compile_kernel(KV27)
K2 = compile_kernel(KPVAF)
NEW = list(K.OG_STATIC) + list(K.OG_STEP)
assert len(NEW) == 103 and K2.WM == WM and K2.K_BLEND == 4 and (K2.W_RS, K2.W_DENSE) == (1.0, 0.3)
for _f in ('window_features', 'add_ramp', 'prior_table', 'prior_feats', 'to_matrix', 'ongoing_keep', 'decode', 'established', 'fill_time'):
    assert ast.dump(ast.parse(__import__('inspect').getsource(getattr(K, _f)))) == ast.dump(ast.parse(__import__('inspect').getsource(getattr(K2, _f)))), _f


# ------------------------------------------------------------------ local replacements of the kernel's file readers
def local_masked_month(panel, split, li):
    """= K2.masked_month on the local exports: (days sorted, V visible eligible, Fq, Em eligibility, Vf all-finite, Ff all-finite)."""
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False); x = np.load(f'{XDIR}/t2x_{panel}.npz', allow_pickle=False)
    days = np.array([str(a) for a in z[f'{split}_days']]); o = np.argsort(days)
    xd = np.array([str(a) for a in x[f'{split}_days']]); ox = np.argsort(xd)
    assert xd[ox].tolist() == days[o].tolist() and [str(a) for a in x['links']] == [str(a) for a in z['links']]
    assert len(li) == len(z['links'])
    V = z[f'{split}_Vm'][o].astype(np.float32); Fq = z[f'{split}_Fm'][o].astype(np.float32); Em = z[f'{split}_Em'][o]
    Vx = x[f'{split}_Vmx'][ox].astype(np.float32); Fx = x[f'{split}_Fmx'][ox].astype(np.float32)
    assert not (np.isfinite(Vx) & Em).any()
    Vf = np.where(np.isfinite(V), V, Vx); Ff = np.where(np.isfinite(V), Fq, Fx)
    return [str(a) for a in days[o]], V, Fq, Em, Vf, Ff


def local_ramp_cube(panel, split, li, days):
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False)
    dd = [str(a) for a in z[f'{split}_days']]; ix = [dd.index(d) for d in days]
    return z[f'{split}_ON'][ix].astype(np.float32), z[f'{split}_OFF'][ix].astype(np.float32)


K2.masked_month = local_masked_month
K2.ramp_cube = local_ramp_cube
K2.log = lambda *a: None


# ------------------------------------------------------------------ panel inputs
def panel_pk(panel):
    """production-style P (lab.load_panel) + all-finite feature cubes Vf / Ff (kernel load_panel with ALLFIN) + prior_all."""
    P = lab.load_panel(panel)
    x = np.load(f'{XDIR}/t2x_{panel}.npz', allow_pickle=False)
    assert [str(a) for a in x['links']] == P['links'] and [str(a) for a in x['train_days']] == P['days']
    Vx = x['train_Vx'].astype(np.float32); Fx = x['train_Fx'].astype(np.float32)
    assert not (np.isfinite(Vx) & P['E']).any()
    PK = dict(P)
    PK['Vf'] = np.where(P['E'], P['V'], Vx); PK['Ff'] = np.where(P['E'], P['Fq'], Fx)
    PK['prior_all'] = K.prior_table(PK, np.ones(len(P['days']), bool))
    return PK


def f03_lab_windows(panel, k=4):
    """F03 lab training windows of one panel (fold != k): list of (set name, wins [(d, t, ONGOING)], weights)."""
    sp = fc.ARMS['F03']; out = []
    for name, sw in sp['train']:
        win = lab.window_set(panel, name); win = win[win.cond == ONGOING].reset_index(drop=True)      # = lab.rows(panel, name).win
        ww = np.full(len(win), float(sw), np.float32) * np.asarray(sp['wfn'](win, name, panel), np.float32)
        sel = np.where((win.fold.to_numpy() != k) & (ww > 0))[0]
        if len(sel):
            out.append((name, list(zip(win.d.to_numpy()[sel].tolist(), win.t.to_numpy()[sel].tolist(), [ONGOING] * len(sel))), ww[sel]))
    return out


def train_rows(PK, wins, prior, cols=None):
    """kernel make_rows (train-style slot-T simulation, all-finite inputs, ogfeat columns) -> X, y, e, meta, cols, R."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        R = K.make_rows(PK, wins, prior)
        X, y, e, meta, cols = K.to_matrix(R, ONGOING, cols, K.ongoing_keep(R))
    return X, y, e, meta, cols, R


def masked_rows(PK, M, Qm, Emv, d, t, cols):
    """rows of windows (d, t) of a masked month exactly as the pvogf kernel builds its March rows (pvaf main() + the ogfeat edits):
    all-finite masked history + all-finite masked slot T, visible labels Qm / Emv, all-train prior in the prior columns."""
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = M['Vf'][d[:, None], idx]; Fh = M['Ff'][d[:, None], idx]; VT = M['Vf'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = K.window_features(PK, Vh, Fh, t, M['dow'][d], PK['prior_all'], VT)
        K.add_ramp(Fe, M['ON'], M['OFF'], d, t, PK['cap'])
        OGX = K.og_extra(PK, Vh, Fh, VT, Fe); OGX.update(K.og_extra2(PK, Vh, Fh, VT, Fe))
        Fe.update({k: OGX[k] for k in K.OG_STATIC})
        W_, L_ = qlast.shape
        R = dict(d=d, t=t, cond=np.array([ONGOING] * W_), qlast=qlast, W=W_, L=L_, rows=[])
        for h in range(1, F + 1):
            feats = dict(Fe); feats['h'] = np.full((W_, L_), h, np.float32)
            feats['prior'], feats['prior_now'], feats['prior_alldays'] = K.prior_feats(PK['prior_all'], M['dow'][d], t, h)
            feats.update({k: v for k, v in K.og_step(OGX, h).items() if k in K.OG_STEP})
            R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))
        X, y, e, meta, _ = K.to_matrix(R, ONGOING, cols, K.ongoing_keep(R))
    return R, X, y, e, meta


def split_rows(PK, panel, split, cols):
    """official released windows (validation / private): all-finite history + slot T from the local exports (= kernel load_split with
    ALLFIN + the ogfeat edits).  Returns wi, R, X (ongoing rows in the keep region), meta, org."""
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False); x = np.load(f'{XDIR}/t2x_{panel}.npz', allow_pickle=False)
    wi = lab.window_index(panel, split)
    assert [str(a) for a in x[f'{split}_window_id']] == wi.window_id.tolist()
    org = pd.to_datetime(wi.forecast_origin, utc=True)
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
    Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32); VT = z[f'{split}_VT'].astype(np.float32)
    Vhx = x[f'{split}_Vhx'].astype(np.float32); Fhx = x[f'{split}_Fhx'].astype(np.float32); VTx = x[f'{split}_VTx'].astype(np.float32)
    assert not (np.isfinite(Vh) & np.isfinite(Vhx)).any() and not (np.isfinite(VT) & np.isfinite(VTx)).any()
    el = np.isfinite(Vh); Fh = np.where(el, Fh, Fhx); Vh = np.where(el, Vh, Vhx); VT = np.where(np.isfinite(VT), VT, VTx)
    W, L = len(wi), len(PK['links'])
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = K.window_features(PK, Vh, Fh, tod, dow, PK['prior_all'], VT)
        wdays = sorted(wi.date.unique())
        ONs, OFFs = local_ramp_cube(panel, split, PK['li'], wdays)
        K.add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, PK['cap'])
        OGX = K.og_extra(PK, Vh, Fh, VT, Fe); OGX.update(K.og_extra2(PK, Vh, Fh, VT, Fe))
        Fe.update({k: OGX[k] for k in K.OG_STATIC})
        R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
        for h in range(1, F + 1):
            feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
            feats['prior'], feats['prior_now'], feats['prior_alldays'] = K.prior_feats(PK['prior_all'], dow, tod, h)
            feats.update({k: v for k, v in K.og_step(OGX, h).items() if k in K.OG_STEP})
            R['rows'].append((h, feats, None, None))
        X, _, _, meta, _ = K.to_matrix(R, ONGOING, cols, K.ongoing_keep(R))
    return wi, R, X, meta, org


def onset_map(M, Qm):
    """production onset candidates on a masked month (pvog_lib.select, onset part): not established, visible queue in T+1..T+6."""
    V, Em = M['V'], M['Em']; E = np.isfinite(V); blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    D = V.shape[0]; on = np.zeros((D, 288), bool)
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if not K.established(Qm[d, t - H:t]):
                on[d, t] = Qm[d, t + 1:t + 1 + F].any()
    return on


def offlike_m(on, og):
    """official-like pseudo windows: interleaved greedy (5 + 5 quota, 72 slots) restarted at 00:00 of every day; union of ongoing picks."""
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(on.shape[0]):
        _, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(b)
    return sorted(s)


# ------------------------------------------------------------------ column sets
def colsets(cols):
    """{(arm, member): column list}: arm 'base' = pvaf columns (no ogfeat), 'ogf' = + the 103 columns; member np / id / full."""
    base = [c for c in cols if c not in NEW]
    out = {}
    for arm, cs in (('base', base), ('ogf', list(cols))):
        out[(arm, 'np')] = [c for c in cs if c not in K.PRIOR_COLS]
        out[(arm, 'id')] = [c for c in cs if c not in K.ID_DROP]
        out[(arm, 'full')] = list(cs)
    return out


def load_cols():
    return json.load(open(f'{CACHE}/cols.json'))


# ------------------------------------------------------------------ decisions / scoring
def decode6(p6):
    """[W, 6, L] probabilities -> [W, 6, L] bool decisions (production decode per window)."""
    W, _, L = p6.shape
    return np.stack([K.decode(p6[i].ravel()).reshape(F, L) for i in range(W)]) if W else np.zeros((0, F, L), bool)


def veto(A, B, qlast, g=10):
    """per window: B unless B adds > g cells outside the queue at T (qlast repeated over the 6 steps) vs A, then A."""
    q = np.repeat(qlast[:, None, :], F, 1)
    add = (B & ~A & ~q).sum((1, 2))
    return np.where((add > g)[:, None, None], A, B), add


def score_dec(dec, truth, ev, qlast):
    """per-window IoU on the visible cells + counts (n_all = all-cell predicted count, used by FM-PM)."""
    T = truth & ev; Pp = dec & ev
    u = (T | Pp).sum((1, 2)); it = (T & Pp).sum((1, 2))
    q = np.repeat(qlast[:, None, :], F, 1)
    return dict(iou=np.where(u == 0, 1.0, it / np.maximum(u, 1)), n_pred=Pp.sum((1, 2)), n_all=dec.sum((1, 2)), n_true=T.sum((1, 2)),
                new_all=(dec & ~q).sum((1, 2)))


def lgb_fit(get_X, y, w, cols, seed, n_trees=500, leaves=255):
    """LightGBM at lab scale; get_X() builds the matrix, which is released right after the Dataset is binned."""
    import lightgbm as lgb
    params = lab.lgb_params(num_leaves=leaves, learning_rate=0.05, min_data_in_leaf=40, seed=seed)
    X = get_X(); assert X.shape == (len(y), len(cols)), (X.shape, len(y), len(cols))
    ds = lgb.Dataset(X, y, weight=w, feature_name=list(cols), params=params, free_raw_data=True)
    ds.construct(); del X; gc.collect()
    return lgb.train(params, ds, n_trees)
