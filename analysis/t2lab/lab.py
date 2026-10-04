"""Local Task 2 lab (queue forecast) on the exported real Task 2 panels (data/t2local/t2data_<PANEL>.npz).

Production feature / window / decode code is NOT rewritten here: the function and constant definitions of
kaggle_jobs/t2_model_v20.py are extracted with `ast` (its module-level main code is never executed) and exposed as
`lab.prod.<name>` and re-exported (window_features, add_ramp, prior_table, prior_feats, make_rows, to_matrix,
ongoing_keep, decode, iou, established, build_windows, PARAMS, PRIOR_COLS, ID_DROP, H, F, SPACING).

Main entry points (see README.md):
    load_panel(panel)                      production-shaped panel dict from the local npz (+ block, family, npz path)
    official_windows(P, split='train')     official train windows as (day index, slot, condition)
    filled_state(P, kind='hb')             VF/QF/MF: filled state FOR SELECTION ONLY (uses the future; never a feature)
    selector_stats(P) / candidates(P, rule) / greedy(...) / replay_table(...)
    window_set(panel, name)                DataFrame d, t, cond, fold, flags (cached)
    rows(panel, name, cond)                cached feature matrix + fold-safe priors + keep masks (RowCache)
    run_arm(...)                           OOF LightGBM arm -> per-window result DataFrame (+ cached probabilities)
    aggregate / paired_bootstrap / step_profile / error_split
Online rule: everything that enters X is computed by the production feature code from data timestamped <= T.
The filled state and all window selection use train-split future values (offline label / window construction only).
"""
import ast, gc, glob, hashlib, json, os, pickle, time, types, warnings
import numpy as np
import pandas as pd

ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
PROD_FILE = f'{ROOT}/kaggle_jobs/t2_model_v20.py'
DATA = f'{ROOT}/data/t2local'
CACHE = f'{DATA}/cache'
LOGS = f'{ROOT}/analysis/t2lab/logs'
THREADS = int(os.environ.get('T2LAB_THREADS', 4))
for _d in (CACHE, LOGS):
    os.makedirs(_d, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ production code via ast (no main code executed)
_WANT_ASSIGN = {'H', 'F', 'SPACING', 'PARAMS', 'PRIOR_COLS', 'ID_DROP', 'PANELS'}


def _load_prod():
    src = open(PROD_FILE).read()
    tree = ast.parse(src)
    body, names = [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body.append(node); names.append(node.name)
        elif isinstance(node, ast.Assign):
            tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if tg and tg <= _WANT_ASSIGN:
                body.append(node); names.extend(sorted(tg))
    mod = types.ModuleType('t2_prod')
    mod.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, T0=time.time(), SMOKE=False,
                        REL='/nonexistent/kaggle_public')
    exec(compile(ast.Module(body=body, type_ignores=[]), PROD_FILE, 'exec'), mod.__dict__)
    mod.__extracted__ = names
    mod.__source_sha1__ = hashlib.sha1(src.encode()).hexdigest()
    return mod


prod = _load_prod()
H, F, SPACING = prod.H, prod.F, prod.SPACING
PANELS = list(prod.PANELS)
PARAMS, PRIOR_COLS, ID_DROP = dict(prod.PARAMS), set(prod.PRIOR_COLS), set(prod.ID_DROP)
window_features, add_ramp, prior_table, prior_feats = prod.window_features, prod.add_ramp, prod.prior_table, prod.prior_feats
to_matrix, ongoing_keep, decode, iou, established = prod.to_matrix, prod.ongoing_keep, prod.decode, prod.iou, prod.established
PRIOR3 = ['prior', 'prior_now', 'prior_alldays']
ONSET, ONGOING = 'queue_onset', 'queue_ongoing'


def make_rows(P, wins, prior, with_labels=True):
    """production make_rows (numpy all-NaN warnings silenced, values unchanged)."""
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        return prod.make_rows(P, wins, prior, with_labels)


def build_windows(P, global_spacing=True, dense_ongoing=False):
    return prod.build_windows(P, global_spacing=global_spacing, dense_ongoing=dense_ongoing)


# ------------------------------------------------------------------ data
_PANELS = {}


def fold_blocks(D):
    """4 contiguous day blocks exactly as production CV."""
    return np.minimum(np.arange(D) * 4 // D, 3)


def load_panel(panel, cache=True):
    """Same dict as production load_panel (panel, links, li, cut, vf, cap, lanes, length, days, dow, V, Fq, E, Q, ON, OFF)
    built from data/t2local/t2data_<panel>.npz, plus: block (fold id per day), family, milepost, npz (path)."""
    if cache and panel in _PANELS:
        return _PANELS[panel]
    path = f'{DATA}/t2data_{panel}.npz'
    z = np.load(path, allow_pickle=False)
    links = [str(x) for x in z['links']]; li = {l: i for i, l in enumerate(links)}
    vf = z['vf'].astype(np.float64); cut = 0.6 * vf
    V = z['train_V'].astype(np.float32); Fq = z['train_F'].astype(np.float32)
    E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
    days = [str(x) for x in z['train_days']]
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    P = dict(panel=panel, links=links, li=li, cut=cut, vf=vf, cap=z['cap'].astype(np.float64), lanes=z['lanes'].astype(np.float64),
             length=z['length'].astype(np.float64), days=days, dow=dow, V=V, Fq=Fq, E=E, Q=Q,
             ON=z['train_ON'].astype(np.float32), OFF=z['train_OFF'].astype(np.float32))
    P['block'] = fold_blocks(len(days)); P['family'] = str(z['train_wi_family_id'][0]); P['milepost'] = z['milepost']
    P['npz'] = path
    if cache:
        _PANELS[panel] = P
    return P


def drop_panel(panel=None):
    if panel is None:
        _PANELS.clear()
    else:
        _PANELS.pop(panel, None)
    gc.collect()


def window_index(panel, split='train'):
    z = np.load(f'{DATA}/t2data_{panel}.npz', allow_pickle=False)
    return pd.DataFrame({str(c): z[f'{split}_wi_{c}'].astype(str) for c in z[f'{split}_wi_columns']})


def official_windows(P, split='train'):
    """official train windows -> list of (day index, slot, condition) in window_index order."""
    wi = window_index(P['panel'], split)
    ts = pd.to_datetime(wi.forecast_origin, utc=True)
    day = ts.dt.strftime('%Y-%m-%d'); slot = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    if split != 'train':
        z = np.load(P['npz'], allow_pickle=False); dd = [str(x) for x in z[f'{split}_days']]
    else:
        dd = P['days']
    pos = {d: i for i, d in enumerate(dd)}
    return [(pos[d], int(s), c) for d, s, c in zip(day, slot, wi.condition) if d in pos]


def split_rows(P, split, prior=None):
    """Production-shaped R for the released validation / private windows (history, slot-T masked value, ramp slots < T).
    Mirrors the inference block of t2_model_v20.py; prior defaults to the all-train prior. No labels."""
    z = np.load(P['npz'], allow_pickle=False)
    wi = window_index(P['panel'], split)
    org = pd.to_datetime(wi.forecast_origin, utc=True)
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
    Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32); VT = z[f'{split}_VT'].astype(np.float32)
    prior = prior_table(P, np.ones(len(P['days']), bool)) if prior is None else prior
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = window_features(P, Vh, Fh, tod, dow, prior, VT)
        wdays = sorted(wi.date.unique())                       # production builds the ramp cube on the window days only
        mdays = [str(x) for x in z[f'{split}_days']]; ix = [mdays.index(d) for d in wdays]
        ONs = z[f'{split}_ON'][ix].astype(np.float32); OFFs = z[f'{split}_OFF'][ix].astype(np.float32)
        add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, P['cap'])
    W, L = qlast.shape
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[], wi=wi, dow=dow)
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, dow, tod, h)
        R['rows'].append((h, feats, None, None))
    return R


# ------------------------------------------------------------------ filled state (SELECTION ONLY: uses past and future)
def fill_time(a, maxgap):
    """linear interpolation along time within each day for interior gaps <= maxgap slots."""
    out = a.copy(); Dd, Tt, Ll = a.shape; idx = np.arange(Tt)
    for d in range(Dd):
        okd = np.isfinite(a[d])
        for l in np.where(okd.any(0) & ~okd.all(0))[0]:
            ok = okd[:, l]
            if ok.sum() < 2:
                continue
            y = a[d, :, l]
            yi = np.interp(idx, idx[ok], y[ok])
            prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
            m = (~ok) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < Tt)
            out[d, m, l] = yi[m]
    return out


def fill_space(a, maxgap):
    """linear interpolation along the link axis for interior gaps <= maxgap links."""
    out = a.copy(); Ll = a.shape[-1]
    A = a.reshape(-1, Ll); O = out.reshape(-1, Ll); ok = np.isfinite(A); idx = np.arange(Ll)
    for r in np.where((~ok).any(1) & (ok.sum(1) >= 2))[0]:
        k = ok[r]
        yi = np.interp(idx, idx[k], A[r][k])
        prev = np.maximum.accumulate(np.where(k, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(k, idx, 10 ** 6)[::-1])[::-1]
        m = (~k) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < Ll)
        O[r, m] = yi[m]
    return out


FILL_KINDS = {'hb': (3, 2, 12),          # time<=3, space<=2, time<=12  (the earlier sandbox / Hisernberg-style proxy)
              't1': (1, 0, 0), 't3': (3, 0, 0), 't6': (6, 0, 0), 't12': (12, 0, 0), 't24': (24, 0, 0), 't36': (36, 0, 0),
              't72': (72, 0, 0), 't288': (288, 0, 0), 'hb36': (3, 2, 36), 't3s2': (3, 2, 0), 's2': (0, 2, 0),
              't36s2': (36, 2, 0), 't36s300': (36, 300, 0)}


def filled_state(P, kind='hb'):
    """dict(VF, QF, MF): filled speed, filled queue state, finite mask. Selection / analysis only, never a feature."""
    key = f'fill_{kind}'
    if key in P:
        return P[key]
    f = f'{CACHE}/fill_{P["panel"]}_{kind}.npz'
    if os.path.exists(f):
        VF = np.load(f)['VF'].astype(np.float32)
    else:
        a, s, b = FILL_KINDS[kind]
        VF = fill_time(P['V'], a) if a else P['V'].copy()
        if s:
            VF = fill_space(VF, s)
        if b:
            VF = fill_time(VF, b)
        np.savez_compressed(f, VF=VF.astype(np.float32))
    MF = np.isfinite(VF)
    P[key] = dict(VF=VF, MF=MF, QF=MF & (VF <= P['cut'][None, None, :]))
    return P[key]


# ------------------------------------------------------------------ selector (vectorised) + greedy replays
TS = np.arange(H, 288 - F)            # candidate origin slots, as production build_windows


def _pers(state, truth, mask):
    """IoU(truth[t+1..t+6], state[t] repeated) over mask cells for every (d, t in TS); empty union -> 1.0."""
    D = state.shape[0]; inter = np.zeros((D, len(TS)), np.int32); union = inter.copy()
    S = state[:, H:288 - F]
    for h in range(1, F + 1):
        T = truth[:, H + h:288 - F + h]; M = mask[:, H + h:288 - F + h]
        inter += (T & S & M).sum(2); union += ((T | S) & M).sum(2)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def _full(a, fill):
    out = np.full((a.shape[0], 288), fill, a.dtype); out[:, H:288 - F] = a
    return out


def selector_stats(P, fill='hb'):
    """per (d, slot) arrays [D, 288] (False / NaN outside slots 12..281):
    cov_ok, est (history has an established queue), hq_obs / hq_fill (queue in horizon),
    pers[(state, cells)] for state in obs|fill, cells in E (observed eligible cells, truth Q) | all (filled cells, truth QF)."""
    key = f'sel_{fill}'
    if key in P:
        return P[key]
    Q, E = P['Q'], P['E']; D, _, L = Q.shape
    FS = filled_state(P, fill); QF, MF = FS['QF'], FS['MF']
    cE = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(E.sum(2), 1)], 1)
    cov = (cE[:, TS] - cE[:, TS - H]) / float(H * L)
    cQ = np.concatenate([np.zeros((D, 1, L), np.int16), np.cumsum(Q, 1, dtype=np.int16)], 1)
    est = ((cQ[:, TS] - cQ[:, TS - H]) >= 2).any(2)
    aq = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(Q.any(2), 1)], 1)
    hq_obs = (aq[:, TS + F + 1] - aq[:, TS + 1]) > 0
    aqf = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(QF.any(2), 1)], 1)
    hq_fill = (aqf[:, TS + F + 1] - aqf[:, TS + 1]) > 0
    S = dict(cov=_full(cov, np.nan), cov_ok=_full(cov >= 0.7, False), est=_full(est, False),
             hq_obs=_full(hq_obs, False), hq_fill=_full(hq_fill, False), pers={})
    for st, state in (('obs', Q), ('fill', QF)):
        S['pers'][(st, 'E')] = _full(_pers(state, Q, E), np.nan)
        S['pers'][(st, 'all')] = _full(_pers(state, QF, MF), np.nan)
    P[key] = S
    return S


RULE_PROD = dict(pers_state='obs', cells='E', fill='hb', pers_max=0.9)       # what production build_windows does
RULE_SANDBOX = dict(pers_state='fill', cells='E', fill='hb', pers_max=0.9)   # the earlier one-panel sandbox rule (34/40 exact)
# FAITHFUL (chosen by replay_selector.py, 36/40 exact and 39/40 within 5 min of the official train ongoing origins):
# persistence of the time-filled state at T against the time-filled horizon truth over ALL filled cells
RULE_FAITHFUL = dict(pers_state='fill', cells='all', fill='t36', pers_max=0.9)


def candidates(P, rule=None):
    """(on, og): boolean [D, 288] candidate maps. Onset candidates are always the production ones
    (coverage, no established queue in history, observed queue in horizon)."""
    rule = dict(RULE_FAITHFUL, **(rule or {}))
    S = selector_stats(P, rule['fill'])
    on = S['cov_ok'] & ~S['est'] & S['hq_obs']
    hq = S['hq_obs'] if rule['cells'] == 'E' else S['hq_fill']
    pers = S['pers'][(rule['pers_state'], rule['cells'])]
    og = S['cov_ok'] & S['est'] & hq & (np.nan_to_num(pers, nan=9.0) <= rule['pers_max'])
    return on, og


def _glist(m):
    d, t = np.where(m)
    return d.astype(np.int64) * 288 + t


def greedy(on, og, scan='interleaved', start_day=0, quota=5, spacing=SPACING, end_day=None):
    """Replay of the selector greedy from 00:00 of start_day. Returns (onset picks, ongoing picks) as lists of (d, t).
    scan='interleaved': one chronological scan over both conditions, every pair of picks >= spacing apart, quota per condition.
    scan='two_pass'  : production order: all onset picks first (spaced among themselves), then ongoing picks spaced from every pick.
    quota=None -> no quota (scan to end_day)."""
    g_on = on if on.ndim == 1 else _glist(on); g_og = og if og.ndim == 1 else _glist(og)
    g0 = start_day * 288; g1 = 10 ** 12 if end_day is None else end_day * 288
    q = 10 ** 9 if quota is None else quota
    sel_on, sel_og = [], []
    if scan == 'interleaved':
        allg = np.concatenate([g_on, g_og]); isog = np.concatenate([np.zeros(len(g_on), bool), np.ones(len(g_og), bool)])
        o = np.argsort(allg, kind='stable'); allg, isog = allg[o], isog[o]
        i = np.searchsorted(allg, g0)
        while i < len(allg) and allg[i] < g1 and (len(sel_on) < q or len(sel_og) < q):
            g = allg[i]
            tgt = sel_og if isog[i] else sel_on
            if len(tgt) < q:
                tgt.append(g); i = np.searchsorted(allg, g + spacing)
            else:
                i += 1
    else:
        i = np.searchsorted(g_on, g0)
        while i < len(g_on) and g_on[i] < g1 and len(sel_on) < q:
            sel_on.append(g_on[i]); i = np.searchsorted(g_on, g_on[i] + spacing)
        tk = np.array(sorted(sel_on), np.int64)
        i = np.searchsorted(g_og, g0)
        while i < len(g_og) and g_og[i] < g1 and len(sel_og) < q:
            g = g_og[i]
            j = np.searchsorted(tk, g)
            lo = tk[j - 1] if j > 0 else -10 ** 12; hi = tk[j] if j < len(tk) else 10 ** 12
            if g - lo >= spacing and hi - g >= spacing:
                sel_og.append(g); tk = np.insert(tk, j, g); i = np.searchsorted(g_og, g + spacing)
            else:
                i += 1
    return [(int(g // 288), int(g % 288)) for g in sel_on], [(int(g // 288), int(g % 288)) for g in sel_og]


def replay_match(P, rule, scan):
    """replay from the first train day with the 5 + 5 quota and count matches with the official train windows."""
    on, og = candidates(P, rule)
    s_on, s_og = greedy(on, og, scan)
    off = official_windows(P)
    out = {}
    for cond, sel in ((ONSET, s_on), (ONGOING, s_og)):
        o = [w[0] * 288 + w[1] for w in off if w[2] == cond]; s = np.array([d * 288 + t for d, t in sel], np.int64)
        tag = 'on' if cond == ONSET else 'og'
        out[f'{tag}_exact'] = int(sum(g in set(s.tolist()) for g in o))
        out[f'{tag}_pm1'] = int(sum((np.abs(s - g) <= 1).any() if len(s) else False for g in o))
        out[f'{tag}_n'] = len(o)
    return out


def replay_table(panels=None, fills=('hb',), verbose=True):
    rows_ = []
    for panel in panels or PANELS:
        P = load_panel(panel)
        for fill in fills:
            for st in ('obs', 'fill'):
                for cells in ('E', 'all'):
                    for scan in ('two_pass', 'interleaved'):
                        r = replay_match(P, dict(pers_state=st, cells=cells, fill=fill), scan)
                        rows_.append(dict(panel=panel, fill=fill, pers_state=st, cells=cells, scan=scan, **r))
        if verbose:
            log('replay', panel)
    return pd.DataFrame(rows_)


# ------------------------------------------------------------------ window sets
def _wins_df(P, wins, **extra):
    df = pd.DataFrame(dict(d=[int(w[0]) for w in wins], t=[int(w[1]) for w in wins], cond=[w[2] for w in wins]))
    df['fold'] = P['block'][df.d.to_numpy()] if len(df) else []
    for k, v in extra.items():
        df[k] = v
    return df


def _annotate(P, df):
    """selection-side diagnostics per window (use the future: never features)."""
    S = selector_stats(P, RULE_FAITHFUL['fill']); d, t = df.d.to_numpy(), df.t.to_numpy()
    df['pers_obs'] = S['pers'][('obs', 'E')][d, t]; df['pers_fill'] = S['pers'][('fill', RULE_FAITHFUL['cells'])][d, t]
    _, og = candidates(P)
    df['faithful'] = og[d, t] & (df.cond == ONGOING).to_numpy()
    prev = og[d, np.maximum(t - 1, 0)]
    df['runstart'] = df.faithful & ~prev
    return df


def window_set(panel, name, refresh=False):
    """Window sets (DataFrame d, t, cond, fold, pers_obs, pers_fill, faithful, runstart), cached under cache/windows/.
    name:
      'dense<cap>'     (a) production training windows: build_windows(global_spacing=False, dense_ongoing=True), ongoing
                           subsampled to <cap> per panel with rng(0) exactly as production (dense3000 = Kaggle setting)
      'faith'          (b) every faithful ongoing candidate origin (all slots)
      'faith3'             faithful candidates on the production dense grid (t % 3 == 0)
      'runstart'           faithful run-starts (candidate at t whose t-1 is not a candidate)
      'offlike'        (c) official-like windows: faithful greedy (FAITHFUL scan, 5 + 5 quota, 360 min) restarted at 00:00
                           of every train day, de-duplicated; both conditions
      'pseudo'         (d) current-style pseudo windows: build_windows(global_spacing=True)
      'official'       (e) the official train windows (5 onset + 5 ongoing per panel)
    """
    f = f'{CACHE}/windows/{panel}__{name}.pkl'
    os.makedirs(os.path.dirname(f), exist_ok=True)
    if os.path.exists(f) and not refresh:
        return pd.read_pickle(f)
    P = load_panel(panel)
    if name.startswith('dense'):
        cap = int(name[5:])
        base = f'{CACHE}/windows/{panel}__denseall.pkl'
        if os.path.exists(base):
            tw = pickle.load(open(base, 'rb'))
        else:
            tw = build_windows(P, global_spacing=False, dense_ongoing=True); pickle.dump(tw, open(base, 'wb'))
        rng = np.random.default_rng(0)
        og = [w for w in tw if w[2] == ONGOING]; on = [w for w in tw if w[2] == ONSET]
        if len(og) > cap:
            og = [og[i] for i in sorted(rng.choice(len(og), cap, replace=False))]
        wins = on + og
    elif name in ('faith', 'faith3', 'runstart'):
        _, og = candidates(P)
        d, t = np.where(og)
        if name == 'faith3':
            k = t % 3 == 0; d, t = d[k], t[k]
        if name == 'runstart':
            k = ~og[d, t - 1]; d, t = d[k], t[k]
        wins = [(int(a), int(b), ONGOING) for a, b in zip(d, t)]
    elif name == 'offlike':
        on, og = candidates(P); g_on, g_og = _glist(on), _glist(og); s = set()
        for d0 in range(len(P['days'])):
            a, b = greedy(g_on, g_og, FAITHFUL_SCAN, start_day=d0)
            s.update((d, t, ONSET) for d, t in a); s.update((d, t, ONGOING) for d, t in b)
        wins = sorted(s, key=lambda w: (w[2] != ONSET, w[0], w[1]))
    elif name == 'pseudo':
        wins = build_windows(P, global_spacing=True)
    elif name == 'official':
        wins = official_windows(P)
    else:
        raise KeyError(name)
    df = _annotate(P, _wins_df(P, wins))
    df.to_pickle(f)
    return df


FAITHFUL_SCAN = 'interleaved'


def register_window_set(panel, name, df, overwrite=False):
    """store a custom window list (DataFrame with d, t, cond) as a named set usable by rows() / run_arm()."""
    f = f'{CACHE}/windows/{panel}__{name}.pkl'
    if os.path.exists(f) and not overwrite:
        raise FileExistsError(f)
    P = load_panel(panel)
    out = _annotate(P, _wins_df(P, list(zip(df.d, df.t, df.cond))))
    for c in df.columns:
        if c not in out:
            out[c] = df[c].to_numpy()
    out.to_pickle(f)
    return out


# ------------------------------------------------------------------ row cache
class RowCache:
    """Feature rows of one (panel, window set, condition).
    X [n, ncol] float32 memmap (prior columns hold the ALL-train prior), cols, y, e (label / eligible-observed flag),
    wi, li, h (window index into .win, link index, step), prior5 [n, 5, 3] (prior, prior_now, prior_alldays computed
    without fold k for k = 0..3, k = 4 = all train days), keep5 [n, 5] (production ongoing_keep under prior k; onset: True),
    win (window DataFrame), truth / ev [W, 6, L], qlast [W, L]."""

    def __init__(self, path):
        self.path = path
        m = json.load(open(f'{path}/meta.json'))
        self.cols = m['cols']; self.panel = m['panel']; self.name = m['name']; self.cond = m['cond']; self.L = m['L']
        self.win = pd.read_pickle(f'{path}/win.pkl'); self.n = m['n']
        self.X = np.load(f'{path}/X.npy', mmap_mode='r') if self.n else np.zeros((0, len(self.cols)), np.float32)
        z = np.load(f'{path}/aux.npz')
        for k in ('y', 'e', 'wi', 'li', 'h', 'prior5', 'keep5', 'truth', 'ev', 'qlast'):
            setattr(self, k, z[k])
        self.fold = self.win.fold.to_numpy()[self.wi] if self.n else np.zeros(0, int)

    def matrix(self, mask, cols, k):
        """rows[mask] restricted to `cols`, prior columns replaced by the fold-k prior (k = 4: all-train prior)."""
        ix = [self.cols.index(c) for c in cols]
        r = np.where(mask)[0]
        X = np.ascontiguousarray(self.X[r][:, ix]) if len(r) else np.zeros((0, len(ix)), np.float32)
        for j, c in enumerate(cols):
            if c in PRIOR3:
                X[:, j] = self.prior5[r, k, PRIOR3.index(c)]
        return X


def _build_rows(P, win, cond):
    wins = list(zip(win.d.tolist(), win.t.tolist(), win.cond.tolist()))
    D = len(P['days'])
    priors = [prior_table(P, P['block'] != k) for k in range(4)] + [prior_table(P, np.ones(D, bool))]
    R = make_rows(P, wins, priors[4])
    d, t = R['d'], R['t']; dw = P['dow'][d]
    keeps = []
    for k in range(5):
        if cond == ONGOING:
            f0 = dict(R['rows'][0][1]); f0['prior_now'] = prior_feats(priors[k], dw, t, 1)[1]
            keeps.append(ongoing_keep(dict(R, rows=[(1, f0, None, None)])))
        else:
            keeps.append(np.ones((R['W'], R['L']), bool))
    keepU = np.logical_or.reduce(keeps)
    X, y, e, meta, cols = to_matrix(R, cond, None, keepU)
    wi_, li_, h_ = meta[:, 0], meta[:, 1], meta[:, 2]
    prior5 = np.zeros((len(X), 5, 3), np.float32); keep5 = np.zeros((len(X), 5), bool)
    for k in range(5):
        for h in np.unique(h_):
            m = h_ == h
            pf = prior_feats(priors[k], dw, t, int(h))
            for j in range(3):
                prior5[m, k, j] = pf[j][wi_[m], li_[m]]
        keep5[:, k] = keeps[k][wi_, li_]
    truth = np.stack([P['Q'][d, np.minimum(t + h, 287)] for h in range(1, F + 1)], 1)
    ev = np.stack([P['E'][d, np.minimum(t + h, 287)] for h in range(1, F + 1)], 1)
    return X, dict(y=y.astype(bool), e=e.astype(bool), wi=wi_.astype(np.int32), li=li_.astype(np.int16), h=h_.astype(np.int8),
                   prior5=prior5, keep5=keep5, truth=truth, ev=ev, qlast=R['qlast']), cols


def rows(panel, name, cond=ONGOING, refresh=False):
    """cached RowCache for (panel, window set, condition); built on first use with production make_rows / to_matrix."""
    path = f'{CACHE}/rows/{panel}__{name}__{cond}'
    if os.path.exists(f'{path}/meta.json') and not refresh:
        return RowCache(path)
    P = load_panel(panel)
    win = window_set(panel, name); win = win[win.cond == cond].reset_index(drop=True)
    os.makedirs(path, exist_ok=True)
    if len(win):
        X, aux, cols = _build_rows(P, win, cond)
    else:
        L = len(P['links']); cols = _empty_cols()
        X = np.zeros((0, len(cols)), np.float32)
        aux = dict(y=np.zeros(0, bool), e=np.zeros(0, bool), wi=np.zeros(0, np.int32), li=np.zeros(0, np.int16), h=np.zeros(0, np.int8),
                   prior5=np.zeros((0, 5, 3), np.float32), keep5=np.zeros((0, 5), bool), truth=np.zeros((0, F, L), bool),
                   ev=np.zeros((0, F, L), bool), qlast=np.zeros((0, L), bool))
    np.save(f'{path}/X.npy', X); np.savez_compressed(f'{path}/aux.npz', **aux); win.to_pickle(f'{path}/win.pkl')
    json.dump(dict(cols=list(cols), panel=panel, name=name, cond=cond, L=len(P['links']), n=int(len(X)),
                   prod_sha1=prod.__source_sha1__), open(f'{path}/meta.json', 'w'))
    del X, aux; gc.collect()
    return RowCache(path)


_COLS = None


def all_cols():
    """production column order (from any existing cache or a tiny build)."""
    global _COLS
    if _COLS is None:
        hits = sorted(glob.glob(f'{CACHE}/rows/*/meta.json'))
        hits = [h for h in hits if json.load(open(h))['n'] > 0]
        _COLS = json.load(open(hits[0]))['cols'] if hits else rows(PANELS[0], 'official', ONGOING).cols
    return list(_COLS)


_empty_cols = all_cols


def colset(kind='np'):
    """'np' prior-free production set (all columns except PRIOR_COLS, ramp features included); 'id' identity-kept
    (drop ID_DROP only); 'full' all columns."""
    c = all_cols()
    if kind == 'np':
        return [x for x in c if x not in PRIOR_COLS]
    if kind == 'id':
        return [x for x in c if x not in ID_DROP]
    if kind == 'full':
        return c
    raise KeyError(kind)


# ------------------------------------------------------------------ scoring
def score_probs(rc, P6, widx=None, arm='', set_name=None):
    """decode (production expected-IoU top-m over the 6 x L cells of each window) and score.
    P6 [W, 6, L] probabilities aligned with rc.win (rows of windows outside widx are ignored).
    Returns one row per window: iou, per-step inter/union (i1..i6, u1..u6), per-step IoU s1..s6 (empty union = 1),
    error split as shares of the union (fn_new: missed cells not queued at T; fn_old: dropped cells queued at T;
    fp_old: kept cells that cleared; fp_new: predicted new cells that stayed free), persistence IoU of qlast."""
    out = []
    widx = range(len(rc.win)) if widx is None else widx
    for i in widx:
        truth, ev, ql = rc.truth[i], rc.ev[i], rc.qlast[i]
        pred = decode(P6[i].ravel()).reshape(F, -1)
        T = truth & ev; Pp = pred & ev; q = np.repeat(ql[None], F, 0)
        u = (T | Pp).sum(); it = (T & Pp).sum()
        w = rc.win.iloc[i]
        r = dict(arm=arm, set=set_name or rc.name, panel=rc.panel, family=rc.panel.rsplit('_', 1)[0], cond=rc.cond,
                 d=int(w.d), t=int(w.t), fold=int(w.fold), iou=1.0 if u == 0 else it / u, union=int(u), n_true=int(T.sum()),
                 n_pred=int(Pp.sum()), pers_qlast=iou(truth, q, ev))
        un = max(u, 1)
        r.update(fn_new=(T & ~Pp & ~q).sum() / un, fn_old=(T & ~Pp & q).sum() / un, fp_old=(Pp & ~T & q).sum() / un,
                 fp_new=(Pp & ~T & ~q).sum() / un)
        for h in range(F):
            ih = int((T[h] & Pp[h]).sum()); uh = int((T[h] | Pp[h]).sum())
            r[f'i{h + 1}'] = ih; r[f'u{h + 1}'] = uh; r[f's{h + 1}'] = 1.0 if uh == 0 else ih / uh
        for c in ('pers_obs', 'pers_fill', 'faithful', 'runstart'):
            if c in w:
                r[c] = w[c]
        out.append(r)
    return pd.DataFrame(out)


def aggregate(df, col='iou'):
    """competition aggregation for one condition: window -> panel -> family -> mean (each condition is half of S_queue)."""
    if not len(df):
        return float('nan')
    pm = df.groupby(['family', 'panel'])[col].mean()
    return float(pm.groupby(level=0).mean().mean())


def s_queue(df, col='iou'):
    """full Task 2 score from a frame holding both conditions (window -> panel x condition -> panel -> family -> mean)."""
    pc = df.groupby(['family', 'panel', 'cond'])[col].mean().groupby(level=[0, 1]).mean()
    return float(pc.groupby(level=0).mean().mean())


def paired_bootstrap(dfa, dfb, col='iou', n=2000, seed=0):
    """paired bootstrap of aggregate(B) - aggregate(A): windows resampled with replacement within each panel.
    Returns dict(diff, lo, hi, p_pos, n_windows)."""
    key = ['panel', 'd', 't', 'cond']
    for nm, df in (('A', dfa), ('B', dfb)):
        if 'set' in df and df.set.nunique() > 1:
            raise ValueError(f'paired_bootstrap: frame {nm} holds several evaluation sets {sorted(df.set.unique())}; filter one set first')
        if df.duplicated(key).any():
            raise ValueError(f'paired_bootstrap: frame {nm} has duplicate windows (panel, d, t, cond); one row per window is required')
    m = dfa[key + ['family', col]].merge(dfb[key + [col]], on=key, suffixes=('_a', '_b'))
    rng = np.random.default_rng(seed)
    groups = [(fam, g[f'{col}_b'].to_numpy() - g[f'{col}_a'].to_numpy()) for (fam, _), g in m.groupby(['family', 'panel'])]
    fams = sorted({f for f, _ in groups}); diffs = np.zeros(n)
    for b in range(n):
        acc = {f: [] for f in fams}
        for f, dv in groups:
            acc[f].append(dv[rng.integers(0, len(dv), len(dv))].mean())
        diffs[b] = np.mean([np.mean(v) for v in acc.values()])
    obs = np.mean([np.mean([dv.mean() for f, dv in groups if f == ff]) for ff in fams])
    return dict(diff=float(obs), lo=float(np.quantile(diffs, 0.025)), hi=float(np.quantile(diffs, 0.975)),
                p_pos=float((diffs > 0).mean()), n_windows=int(len(m)))


def step_profile(df):
    """mean per-step IoU (window mean, empty union = 1) with the competition aggregation: 6 numbers for T+5..T+30."""
    return [aggregate(df, f's{h}') for h in range(1, F + 1)]


def error_split(df):
    """aggregated shares of the union: missed new cells, kept cleared cells, other (dropped persisting + false new)."""
    a = {c: aggregate(df, c) for c in ('fn_new', 'fp_old', 'fn_old', 'fp_new')}
    return dict(missed_new=a['fn_new'], kept_cleared=a['fp_old'], other=a['fn_old'] + a['fp_new'], dropped_old=a['fn_old'], false_new=a['fp_new'])


def persistence_probs(rc):
    """reference: last-valid-fill persistence (qlast repeated), as probabilities for score_probs."""
    return np.repeat(rc.qlast[:, None, :], F, 1).astype(np.float32) + 1e-6


# ------------------------------------------------------------------ OOF arm
def lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0, **kw):
    p = dict(PARAMS, num_leaves=num_leaves, learning_rate=learning_rate, min_data_in_leaf=min_data_in_leaf,
             num_threads=THREADS, seed=seed, bagging_seed=10 + seed, feature_fraction_seed=20 + seed, deterministic=True)
    p.update(kw)
    return p


def train_matrix(train, k, cols, cond=ONGOING, panels=None, weight_fn=None, max_rows=None, seed=0):
    """training matrix for held-out fold k (k = 4: all folds). train = [(set name, set weight), ...] or a set name.
    Rows: eligible-observed labels (e), inside the production keep region under the fold-k prior, windows outside fold k.
    weight_fn(win_df, set_name, panel) -> per-window weights (optional; multiplied by the set weight)."""
    train = [(train, 1.0)] if isinstance(train, str) else list(train)
    Xs, ys, ws = [], [], []
    for panel in panels or PANELS:
        for name, sw in train:
            rc = rows(panel, name, cond)
            if not rc.n:
                continue
            m = rc.e & rc.keep5[:, k] & (rc.fold != k)
            ww = np.full(len(rc.win), float(sw), np.float32)
            if weight_fn is not None:
                ww = ww * np.asarray(weight_fn(rc.win, name, panel), np.float32)
            m &= ww[rc.wi] > 0
            Xs.append(rc.matrix(m, cols, k)); ys.append(rc.y[m]); ws.append(ww[rc.wi[m]])
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.float32); w = np.concatenate(ws)
    del Xs
    if max_rows and len(X) > max_rows:
        s = np.sort(np.random.default_rng(seed).choice(len(X), max_rows, replace=False)); X, y, w = X[s], y[s], w[s]
    return X, y, w


def predict_set(models, rc, k, cols, widx=None, transform=None):
    """P6 [W, 6, L] for the windows widx of rc (default: windows of fold k) under the fold-k prior / keep region.
    models: one booster or a list (mean of probabilities). Cells outside the keep region get probability 0 (production)."""
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    if not rc.n:
        return P6
    wmask = (rc.win.fold.to_numpy() == k) if widx is None else np.isin(np.arange(len(rc.win)), widx)
    m = rc.keep5[:, k] & wmask[rc.wi]
    if m.any():
        X = rc.matrix(m, cols, k)
        if transform is not None:
            X = transform(X)
        ms = models if isinstance(models, (list, tuple)) else [models]
        p = sum(mdl.predict(X, num_threads=THREADS) for mdl in ms) / len(ms)
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = p
    return P6


def run_arm(arm, train='dense1500', evals=('official', 'offlike', 'pseudo'), cols='np', params=None, n_trees=500, folds=(0, 1, 2, 3),
            cond=ONGOING, panels=None, weight_fn=None, max_rows=None, seeds=(0,), save=True, verbose=True):
    """Train one LightGBM arm per held-out fold on the window set(s) `train` and predict the held-out windows of every
    evaluation set. Returns the per-window result frame (see score_probs); with save=True it is also written to
    cache/oof/<arm>/results.pkl and the probabilities to cache/oof/<arm>/probs__<set>__<panel>.npz (P6, d, t, done)."""
    import lightgbm as lgb
    panels = panels or PANELS
    cols_ = colset(cols) if isinstance(cols, str) else list(cols)
    odir = f'{CACHE}/oof/{arm}'
    if save:
        os.makedirs(odir, exist_ok=True)
    probs = {}; res = []
    for k in folds:
        t1 = time.time()
        X, y, w = train_matrix(train, k, cols_, cond, panels, weight_fn, max_rows)
        models = []
        for sd in seeds:
            p = dict(params or lgb_params()); p.update(seed=sd, bagging_seed=10 + sd, feature_fraction_seed=20 + sd, num_threads=THREADS)
            ds = lgb.Dataset(X, y, weight=None if np.all(w == 1) else w, feature_name=list(cols_), free_raw_data=True)
            models.append(lgb.train(p, ds, n_trees))
        if verbose:
            log(f'{arm} fold {k}: {len(X)} rows x {X.shape[1]} cols, pos {y.mean():.3f}, {len(models)} model(s) in {time.time() - t1:.0f}s')
        del X, y, w; gc.collect()
        for s in evals:
            for panel in panels:
                rc = rows(panel, s, cond)
                widx = np.where(rc.win.fold.to_numpy() == k)[0]
                if not len(widx):
                    continue
                P6 = predict_set(models, rc, k, cols_)
                key = (s, panel)
                if key not in probs:
                    probs[key] = dict(P6=np.zeros_like(P6), done=np.zeros(len(rc.win), bool), d=rc.win.d.to_numpy(), t=rc.win.t.to_numpy())
                probs[key]['P6'][widx] = P6[widx]; probs[key]['done'][widx] = True
                res.append(score_probs(rc, P6, widx, arm=arm, set_name=s))
        if verbose and res:
            df = pd.concat(res)
            log(f'{arm} fold {k} cumulative: ' + ', '.join(f'{s} {aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in evals))
    df = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    if save:
        df.to_pickle(f'{odir}/results.pkl')
        for (s, panel), v in probs.items():
            np.savez_compressed(f'{odir}/probs__{s}__{panel}.npz', P6=v['P6'].astype(np.float16), done=v['done'], d=v['d'], t=v['t'])
        json.dump(dict(arm=arm, train=train if isinstance(train, str) else [list(x) for x in train], evals=list(evals), cols=cols_,
                       params=params or lgb_params(), n_trees=n_trees, folds=list(folds), cond=cond, seeds=list(seeds),
                       max_rows=max_rows, weighted=weight_fn is not None), open(f'{odir}/config.json', 'w'), indent=1, default=str)
    return df


def load_arm(arm):
    return pd.read_pickle(f'{CACHE}/oof/{arm}/results.pkl')


def load_probs(arm, set_name, panel):
    z = np.load(f'{CACHE}/oof/{arm}/probs__{set_name}__{panel}.npz')
    return z['P6'].astype(np.float32), z['done']


def summary(df, sets=None):
    """one line per evaluation set: aggregated IoU, n windows, per-step profile, error split."""
    out = []
    for s in sets or list(dict.fromkeys(df.set)):
        g = df[df.set == s]
        if not len(g):
            continue
        es = error_split(g)
        out.append(dict(set=s, n=len(g), iou=aggregate(g), **{f'h{h + 1}': v for h, v in enumerate(step_profile(g))},
                        missed_new=es['missed_new'], kept_cleared=es['kept_cleared'], other=es['other']))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ self-test
def selftest(panel='D12_I5_S', deep=True):
    """parity checks; raises AssertionError on failure and returns a dict of measured facts."""
    out = {}
    src = open(PROD_FILE).read(); tree = ast.parse(src)
    fdefs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for nm in ('window_features', 'add_ramp', 'prior_table', 'prior_feats', 'make_rows', 'to_matrix', 'ongoing_keep', 'decode', 'iou',
               'established', 'build_windows', 'ffill_time', 'shift_links', 'dist_to_queued'):
        assert nm in prod.__dict__ and nm in fdefs, nm
        assert getattr(prod, nm).__code__.co_firstlineno == fdefs[nm].lineno, nm      # compiled from the production file itself
    assert (H, F, SPACING) == (12, 6, 72) and 'catboost' not in prod.__dict__ and 'panels' not in prod.__dict__
    out['extracted'] = len(prod.__extracted__)
    P = load_panel(panel)
    assert np.array_equal(P['cut'], 0.6 * P['vf']) and np.array_equal(P['E'], np.isfinite(P['V']))
    assert np.array_equal(P['Q'], P['E'] & (P['V'] <= P['cut'][None, None, :])) and P['V'].dtype == np.float32
    # official history_coverage: which definition reproduces it
    wi = window_index(panel); off = official_windows(P)
    cov = np.array([P['E'][d, t - H:t].mean() for d, t, _ in off]); offc = wi.history_coverage.astype(float).to_numpy()
    out['coverage_max_abs_diff_vs_official'] = float(np.abs(cov - offc).max())
    # vectorised selector == production build_windows
    on, og = candidates(P, RULE_PROD)
    a, b = greedy(on, og, 'two_pass', quota=None)
    mine = [(d, t, ONSET) for d, t in a] + [(d, t, ONGOING) for d, t in b]
    ref = build_windows(P, global_spacing=True)
    assert mine == ref, ('vectorised two-pass replay differs from production build_windows', len(mine), len(ref))
    out['vectorised_selector_equals_build_windows'] = len(ref)
    if deep:
        # row cache == direct production path under a fold prior
        win = window_set(panel, 'pseudo'); win = win[win.cond == ONGOING].reset_index(drop=True).iloc[:60]
        X, aux, cols = _build_rows(P, win, ONGOING)
        k = 1; prior = prior_table(P, P['block'] != k)
        R = make_rows(P, list(zip(win.d.tolist(), win.t.tolist(), win.cond.tolist())), prior)
        Xd, yd, ed, md, cd = to_matrix(R, ONGOING, None, ongoing_keep(R))
        m = aux['keep5'][:, k]; Xc = X[m].copy()
        for j, c in enumerate(cols):
            if c in PRIOR3:
                Xc[:, j] = aux['prior5'][m, k, PRIOR3.index(c)]
        assert list(cols) == list(cd) and Xc.shape == Xd.shape
        assert np.array_equal(Xc, Xd, equal_nan=True) and np.array_equal(aux['y'][m], yd) and np.array_equal(aux['e'][m], ed)
        assert np.array_equal(np.stack([aux['wi'][m], aux['li'][m], aux['h'][m]], 1), md)
        out['row_cache_equals_production_rows'] = int(Xd.shape[0])
    return out


if __name__ == '__main__':
    import sys
    print(json.dumps(selftest(sys.argv[1] if len(sys.argv) > 1 else 'D12_I5_S'), indent=1))
