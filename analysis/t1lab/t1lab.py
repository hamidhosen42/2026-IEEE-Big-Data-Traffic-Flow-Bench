"""t1lab: local Task 1 / Task 3 replica on corridor D12_I405_S (the only panel with local Task 1 truth).

What it is (see README.md in this folder):
  * data     267 train days with truth (mainline_states). The released masked view is REBUILT from the organizer's
             public hash (stable_mask / regime_of_dates in repo/src/task1/baseline_task1_historical_mean.py) and checked
             cell for cell against the 181 released masked files that exist locally -> real regimes for all 267 days
             (R1 70 / R2 85 / R3 112); the local masked download alone holds only 26 R3 days.
  * models   the production feature / model code of kaggle_jobs/t1_pipeline_v13a.py, extracted with `ast` (the Kaggle main
             code never runs): features() incl. ASM + low-rank + the 20 FD-implied columns, residual LightGBM for speed and
             flow, transductive rows from the evaluation block exactly like production does for the validation / private month.
  * folds    outer 0: fit = first 187 days, held-out = last 80 days = pseudo-months A (40 d) and B (40 d)
             outer 1: fit = last 187 days,  held-out = first 80 days = pseudo-months C and D
  * scores   S_state as repo/src/task1/score_task1.py, LWR proxy 1 - sum|dN_pred - dN_true| / sum|dN_true| (N = q/max(v,1)*L,
             consecutive eligible cells per link, within the day), S_FD as repo/src/task3/score_task3.py (full eligible field,
             per lane), J = 0.35*S_state + 0.10*LWR + 0.05*S_FD; everything is kept as per-day sufficient statistics so that
             regime breakdowns, nested selections and day bootstraps are cheap.
  * post     recon() / tv() are the functions of kaggle_jobs/swap_fdrecon.py themselves (ast), driven with other constants.

Nothing here writes outside data/t1local/ (cache, gitignored) and the caller's own result folders.
"""
from __future__ import annotations

import ast, glob, hashlib, json, os, re, sys, time, types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REL = os.path.join(ROOT, 'data', 'kaggle_public')
PANEL = 'D12_I405_S'
CACHE = os.path.join(ROOT, 'data', 't1local')
PROD_T1 = os.path.join(ROOT, 'kaggle_jobs', 't1_pipeline_v13a.py')
PROD_SWAP = os.path.join(ROOT, 'kaggle_jobs', 'swap_fdrecon.py')
THREADS = int(os.environ.get('T2LAB_THREADS', 4))
REGS = ('R1', 'R2', 'R3')
NOPROF = {'hv', 'hq', 'tod', 'dow'}                 # production drops the profile features (v4 decision)
N_HELD, N_MONTH = 80, 40
MAX_FIT_ROWS, MAX_TR_ROWS = 650_000, 350_000        # row caps of the REDUCED replica (other panels) only
W_STATE, W_LWR, W_FD = 0.35, 0.10, 0.05             # J = share of S_total carried by Task 1 + Task 3
os.makedirs(CACHE, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ production code via ast (no main code executed)
def _compile(path, funcs, consts, extra):
    src = open(path).read()
    body = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in funcs:
            body.append(node)
        elif isinstance(node, ast.Assign):
            names = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if names and names <= set(consts):
                body.append(node)
    mod = types.ModuleType('prod_' + os.path.basename(path)[:-3])
    mod.__dict__.update(extra)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), mod.__dict__)
    missing = [k for k in list(funcs) + list(consts) if k not in mod.__dict__]
    assert not missing, f'{path}: not found {missing}'
    return mod


T1_FUNCS = ('interp_axis', 'shift', 'gap_dist', 'base_recipe', 'lowrank', 'lowrank_days', 'features', 'fd_feats', 'asm_feats',
            'stack', 's_state', 'lwr_proxy', 'predict_variants')
T1_CONSTS = ('RATE', 'TR_ROUNDS', 'TR_HIDE', 'ASM_WAVES', 'ASM_K', 'ASM_SIGMA', 'ASM_TAU', 'NEW_LR', 'V12_NFEAT', 'FD_SPEEDS',
             'FD_COLS', 'N_FD')
SWAP_FUNCS = ('fd_params', 'kfd_of', 'recon', 'tv_solve', 'tv')
SWAP_CONSTS = ('GATE', 'A', 'LOGR_MAX', 'TV_FREE', 'TV_FREE_A', 'TV_GATE', 'TV_GATE_A', 'TV_ITERS', 'RUN_GAP_NS', 'W_RANGE',
               'KJL_RANGE', 'EXCL_MED', 'EXCL_IQR', 'EXCL_MIN_N')
prod = _compile(PROD_T1, T1_FUNCS, T1_CONSTS, dict(np=np, pd=pd))
PROD_DEFAULT = dict(gate=0.6, a=0.75, logr=0.5, tv=(0.0075, 0.001, 0.02, 0.005), iters=300)


def _swap_module():
    return _compile(PROD_SWAP, SWAP_FUNCS, SWAP_CONSTS, dict(np=np, pd=pd, REL=REL, print=lambda *a, **k: None))


_sw0 = _swap_module()
assert (_sw0.GATE, _sw0.A, _sw0.LOGR_MAX) == (PROD_DEFAULT['gate'], PROD_DEFAULT['a'], PROD_DEFAULT['logr'])
assert (_sw0.TV_FREE, _sw0.TV_FREE_A, _sw0.TV_GATE, _sw0.TV_GATE_A) == PROD_DEFAULT['tv'] and _sw0.TV_ITERS == PROD_DEFAULT['iters']


# ------------------------------------------------------------------ organizer code (mask hash, FD parameters)
sys.path.insert(0, os.path.join(ROOT, 'repo', 'src'))
from task1.baseline_task1_historical_mean import regime_of_dates, stable_mask  # noqa: E402


def organizer_params():
    """Per-link triangular FD parameters exactly as the Task 3 scorer builds them (network_parameters)."""
    from pathlib import Path
    from task3.score_task3 import network_parameters
    return network_parameters(Path(REL) / 'corridors' / PANEL).set_index('link_id')


# ------------------------------------------------------------------ data cache
def _slot(ts_text):
    return ts_text.str[11:13].astype(int).to_numpy() * 12 + ts_text.str[14:16].astype(int).to_numpy() // 5


def build_cache(force=False):
    """Truth cube + hash mask for every local train day (and the released private month). ~20 s."""
    path = f'{CACHE}/cube_{PANEL}.npz'
    if os.path.exists(path) and not force:
        return path
    C = f'{REL}/corridors/{PANEL}'
    key = lambda f: os.path.basename(f)[-18:-8].replace('_', '-')
    full = {key(f): f for f in sorted(glob.glob(f'{C}/train/mainline_states/*/*.parquet'))}
    msk = {key(f): f for f in sorted(glob.glob(f'{C}/train/mainline_states_masked/*/*.parquet'))}
    days = sorted(full)
    cols = ['date', 'timestamp', 'station_id', 'link_id', 'milepost', 'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible']
    first = pd.read_parquet(full[days[0]], columns=cols)
    st = first.drop_duplicates('station_id').set_index('station_id')[['link_id', 'milepost']].sort_values('milepost')
    stations = st.index.to_numpy()
    sidx = pd.Index(stations)
    D, S = len(days), len(stations)
    v = np.full((D, 288, S), np.nan, np.float32); q = v.copy(); o = v.copy()
    ef = np.zeros((D, 288, S), bool); mask = np.zeros((D, 288, S), bool)
    reg = []
    n_checked = n_bad = 0
    for di, day in enumerate(days):
        f = pd.read_parquet(full[day], columns=cols)
        assert (f.date == day).all() and len(f) == 288 * S, (day, len(f))
        ti = _slot(f.timestamp); si = sidx.get_indexer(f.station_id)
        assert (si >= 0).all()
        r = regime_of_dates(PANEL, f.date.astype(str).iloc[:1])[0]
        hm = stable_mask(PANEL, r, f.date, f.timestamp, f.link_id)
        v[di, ti, si] = f.speed_kmh.to_numpy(); q[di, ti, si] = f.flow_vph.to_numpy(); o[di, ti, si] = f.occupancy.to_numpy()
        ef[di, ti, si] = f.is_score_eligible.to_numpy() == 1; mask[di, ti, si] = hm
        reg.append(r)
        if day in msk:                                    # released masked file: must equal truth with the hash mask applied
            m = pd.read_parquet(msk[day], columns=['timestamp', 'station_id', 'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
            assert (m.timestamp.values == f.timestamp.values).all() and (m.station_id.values == f.station_id.values).all()
            rr = re.search(r'mask_regime=(R\d)', msk[day]).group(1)
            el = f.is_score_eligible.to_numpy() == 1
            gone = hm & el
            ok = (rr == r) and (m.is_score_eligible.to_numpy() == f.is_score_eligible.to_numpy()).all()
            for c in ('speed_kmh', 'flow_vph', 'occupancy'):
                a = np.where(gone, np.nan, f[c].to_numpy()); b = m[c].to_numpy()
                ok &= bool(np.array_equal(a, b, equal_nan=True))
            n_checked += 1; n_bad += not ok
    assert n_bad == 0, f'{n_bad} of {n_checked} released masked files differ from truth + hash mask'
    reg = np.array(reg)
    # private month (released masked layer; no truth)
    pf = {key(f): f for f in sorted(glob.glob(f'{C}/private/mainline_states_masked/*/*.parquet'))}
    pdays = sorted(pf)
    pv = np.full((len(pdays), 288, S), np.nan, np.float32); pq = pv.copy(); po = pv.copy()
    pe = np.zeros((len(pdays), 288, S), bool); preg = []
    for di, day in enumerate(pdays):
        m = pd.read_parquet(pf[day], columns=['timestamp', 'station_id', 'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
        ti = _slot(m.timestamp); si = sidx.get_indexer(m.station_id)
        pv[di, ti, si] = m.speed_kmh.to_numpy(); pq[di, ti, si] = m.flow_vph.to_numpy(); po[di, ti, si] = m.occupancy.to_numpy()
        pe[di, ti, si] = m.is_score_eligible.to_numpy() == 1
        preg.append(re.search(r'mask_regime=(R\d)', pf[day]).group(1))
    np.savez_compressed(path, days=np.array(days), stations=stations, link_id=st.link_id.to_numpy().astype(str),
                        milepost=st.milepost.to_numpy(), v=v, q=q, o=o, eligflag=ef, mask=mask, reg=reg,
                        n_checked=n_checked, priv_days=np.array(pdays), priv_mv=pv, priv_mq=pq, priv_mo=po, priv_elig=pe,
                        priv_reg=np.array(preg))
    log('cache built', path, '| days', D, '| released masked files checked', n_checked, '| regimes',
        {r: int((reg == r).sum()) for r in REGS})
    return path


T2_PANELS = ('D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S')


def build_mask_t2(panel, force=False):
    """Hash mask + regime per day for a Task 2 panel exported to data/t2local (truth = eligible observed train cells)."""
    path = f'{CACHE}/mask_{panel}.npz'
    if os.path.exists(path) and not force:
        return path
    z = np.load(os.path.join(ROOT, 'data', 't2local', f't2data_{panel}.npz'), allow_pickle=True)
    days = z['train_days'].astype(str); links = z['links'].astype(str); Lk = len(links)
    hhmm = [f'{t // 12:02d}:{(t % 12) * 5:02d}:00Z' for t in range(288)]
    mask = np.zeros((len(days), 288, Lk), bool); reg = []
    link_rep = pd.Series(np.tile(links, 288))
    for di, day in enumerate(days):
        r = regime_of_dates(panel, pd.Series([day]))[0]; reg.append(r)
        ts = pd.Series(np.repeat([f'{day}T{x}' for x in hhmm], Lk)); dd = pd.Series(np.full(288 * Lk, day))
        mask[di] = stable_mask(panel, r, dd, ts, link_rep).reshape(288, Lk)
    np.savez_compressed(path, mask=mask, reg=np.array(reg))
    log('mask built', panel, mask.shape, {r: int((np.array(reg) == r).sum()) for r in REGS})
    return path


class Data:
    """All local train days of a panel. target = eligible cell blanked by the released mask (= Task 1 target).
    panel = D12_I405_S (default): full local release (truth incl. occupancy and ineligible observed cells).
    panel in T2_PANELS: REDUCED replica from data/t2local/t2data_<panel>.npz - speed (float16) and flow of eligible observed
    cells only, no occupancy (those feature columns are NaN), station position = cumulative link length, and the released
    mask is the plain hash (the few cells inside official Task 2 train windows are not treated specially)."""

    def __init__(self, panel=PANEL):
        self.panel = panel
        self.sfx = '' if panel == PANEL else f'_{panel}'
        if panel != PANEL:
            return self._init_t2(panel)
        z = np.load(build_cache(), allow_pickle=True)
        self.days = z['days'].astype(str); self.stations = z['stations'].astype(str); self.link_id = z['link_id'].astype(str)
        self.v, self.q, self.o = z['v'], z['q'], z['o']
        self.eligflag = z['eligflag']; self.reg = z['reg'].astype(str)
        self.elig = self.eligflag & np.isfinite(self.v)                         # = production Cube.elig
        self.target = z['mask'] & self.elig
        self.dow = pd.to_datetime(pd.Series(self.days)).dt.dayofweek.to_numpy()
        self.n_checked = int(z['n_checked'])
        fdp = pd.read_csv(f'{REL}/corridors/{PANEL}/network/fd_parameters.csv', dtype={'link_id': str}).set_index('link_id')
        st = pd.DataFrame({'link_id': self.link_id, 'milepost': z['milepost']}, index=pd.Index(self.stations, name='station_id'))
        self.st = st.join(fdp.rename(columns={'free_speed_kmh': 'vf', 'capacity_vph': 'cap'})
                          [['lanes', 'length_km', 'vf', 'cap', 'critical_density', 'k_jam']], on='link_id')   # = production `st`
        self.priv = dict(days=z['priv_days'].astype(str), mv=z['priv_mv'], mq=z['priv_mq'], mo=z['priv_mo'], melig=z['priv_elig'],
                         reg=z['priv_reg'].astype(str))
        self.D, _, self.S = self.v.shape
        self.fit_cap = None

    def _init_t2(self, panel):
        z = np.load(os.path.join(ROOT, 'data', 't2local', f't2data_{panel}.npz'), allow_pickle=True)
        m = np.load(build_mask_t2(panel), allow_pickle=True)
        self.days = z['train_days'].astype(str); self.link_id = z['links'].astype(str); self.stations = self.link_id.copy()
        self.v = z['train_V'].astype(np.float32); self.q = z['train_F'].astype(np.float32)
        self.q = np.where(np.isfinite(self.v), self.q, np.nan).astype(np.float32)
        self.o = np.full(self.v.shape, np.nan, np.float32)
        self.eligflag = np.isfinite(self.v) & np.isfinite(self.q); self.reg = m['reg'].astype(str)
        self.elig = self.eligflag.copy(); self.target = m['mask'] & self.elig
        self.dow = pd.to_datetime(pd.Series(self.days)).dt.dayofweek.to_numpy()
        self.n_checked = 0
        lanes = np.clip(z['lanes'].astype(float), 1.0, None); vf = z['vf'].astype(float); cap = z['cap'].astype(float)
        length = z['length'].astype(float); kc = cap / np.clip(vf, 1, None)
        kj = np.where(np.isfinite(z['fd_kjam'].astype(float)), z['fd_kjam'].astype(float), 100.0 * lanes)
        self.st = pd.DataFrame({'link_id': self.link_id, 'milepost': np.cumsum(length) - length / 2, 'lanes': lanes, 'length_km': length,
                                'vf': vf, 'cap': cap, 'critical_density': kc, 'k_jam': kj}, index=pd.Index(self.stations, name='station_id'))
        self.priv = None
        self.D, _, self.S = self.v.shape
        self.fit_cap = max(30, min(193, int(9000 // self.S)))            # keep the fit cube under ~9000 station-days (memory)

    def net(self):
        """scorer-side FD parameters per link (index link_id), = repo/src/task3/score_task3.py network_parameters()."""
        if self.panel == PANEL:
            return organizer_params().reindex(self.link_id)
        st = self.st
        out = pd.DataFrame({'length_km': st.length_km.to_numpy(), 'free_speed_kmh': st.vf.to_numpy(), 'capacity_vph': st.cap.to_numpy(),
                            'lanes': st.lanes.to_numpy()}, index=pd.Index(self.link_id, name='link_id'))
        out['critical_density'] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
        out['k_jam'] = np.maximum(st.k_jam.to_numpy(), out.critical_density * 1.05)
        return out

    def fd_frame(self):
        """kernel-side FD table (= swap_fdrecon.fd_params) built from the station table; checked against it for D12_I405_S."""
        st = self.st
        m = pd.DataFrame({'link_id': self.link_id, 'free_speed_kmh': st.vf.to_numpy(float), 'capacity_vph': st.cap.to_numpy(float),
                          'lanes': np.clip(st.lanes.to_numpy(float), 1.0, None), 'k_jam': st.k_jam.to_numpy(float), 'n_rows': 1})
        m['critical_density'] = m.capacity_vph / m.free_speed_kmh.clip(lower=1)
        m['k_jam'] = m.k_jam.fillna(100.0 * m.lanes)
        m['k_jam'] = np.maximum(m.k_jam, 1.05 * m.critical_density)
        m['capl'] = m.capacity_vph / m.lanes; m['kcl'] = m.critical_density / m.lanes; m['kjl'] = m.k_jam / m.lanes
        m['w'] = m.capl / np.maximum(m.kjl - m.kcl, 1e-6)
        m['panel'] = self.panel
        return m

    def view(self, idx, hide=None):
        return View(self, np.asarray(idx), hide)


class View:
    """Duck-typed production Cube for a set of days: mv/mq/mo = released masked view (optionally with extra hidden cells)."""

    def __init__(self, Z, idx, hide=None):
        self.idx = idx
        self.days = list(Z.days[idx]); self.reg = Z.reg[idx]; self.dow = Z.dow[idx]; self.S = Z.S
        self.v, self.q = Z.v[idx], Z.q[idx]
        self.elig = Z.elig[idx]; self.melig = Z.eligflag[idx]; self.target = Z.target[idx]
        gone = self.target if hide is None else (self.target | hide)
        self.mv = np.where(gone, np.nan, self.v); self.mq = np.where(gone, np.nan, self.q)
        self.mo = np.where(gone, np.nan, Z.o[idx])


def folds(Z, outer):
    """fit day indices and the two held-out pseudo-months of an outer fold."""
    D = Z.D
    if outer == 0:
        fit = np.arange(0, D - N_HELD); held = np.arange(D - N_HELD, D); names = ('A', 'B')
    elif outer == 1:
        fit = np.arange(N_HELD, D); held = np.arange(0, N_HELD); names = ('C', 'D')
    else:
        raise ValueError(outer)
    if getattr(Z, 'fit_cap', None) and len(fit) > Z.fit_cap:               # reduced replica: evenly thinned fit days
        fit = fit[np.unique(np.linspace(0, len(fit) - 1, Z.fit_cap).round().astype(int))]
    return fit, {names[0]: held[:N_MONTH], names[1]: held[N_MONTH:]}


MONTH_OUTER = {'A': 0, 'B': 0, 'C': 1, 'D': 1}


def profile(c):
    """weekday x slot x station mean of the released (masked-layer) fit observations, as production."""
    S = c.mv.shape[2]
    hv = np.full((7, 288, S), np.nan, np.float32); hq = hv.copy()
    for w in range(7):
        sel = c.dow == w
        if sel.any():
            with np.errstate(all='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    hv[w] = np.nanmean(c.mv[sel], 0); hq[w] = np.nanmean(c.mq[sel], 0)
    hv = np.where(np.isfinite(hv), hv, np.nanmean(hv)); hq = np.where(np.isfinite(hq), hq, np.nanmean(hq))
    return hv, hq


# ------------------------------------------------------------------ feature matrices (cached per outer fold)
def _feat(c, st, hv, hq, where):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F, _, _ = prod.features(c, st, hv[c.dow], hq[c.dow])
    X, cols = prod.stack(F, where)
    return X, cols


def fold_matrices(Z, outer, force=False):
    """X/y of the fit targets and X of each pseudo-month's targets (all 98 production columns). Cached."""
    path = f'{CACHE}/fold{outer}{Z.sfx}.npz'
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    fit, months = folds(Z, outer)
    c = Z.view(fit); st = Z.st
    hv, hq = profile(c)
    X, cols = _feat(c, st, hv, hq, c.target)
    assert len(cols) == prod.V12_NFEAT + prod.N_FD and cols[-prod.N_FD:] == prod.FD_COLS
    vf = np.broadcast_to(st.vf.to_numpy()[None, None, :], c.v.shape)[c.target]
    ln = np.broadcast_to(st.lanes.to_numpy()[None, None, :], c.v.shape)[c.target]
    out = dict(cols=np.array(cols), hv=hv, hq=hq, X=X,
               yv=(c.v[c.target] / vf - X[:, cols.index('tv')]).astype(np.float32),
               yq=(c.q[c.target] / ln - X[:, cols.index('tq')]).astype(np.float32))
    if Z.panel != PANEL and len(X) > MAX_FIT_ROWS:                           # reduced replica: cap the fit rows
        keep = np.sort(np.random.default_rng(0).choice(len(X), MAX_FIT_ROWS, replace=False))
        out.update(X=X[keep], yv=out['yv'][keep], yq=out['yq'][keep])
    log(f'outer {outer}: fit rows {len(X)} cols {len(cols)}')
    for nm, idx in months.items():
        cm = Z.view(idx)
        Xm, cm_cols = _feat(cm, st, hv, hq, cm.target)
        assert cm_cols == cols
        out[f'X_{nm}'] = Xm
        log(f'outer {outer}: month {nm} target rows {len(Xm)}')
    np.savez(path, **out)
    return out


def colsets(cols):
    cols = list(cols)
    ixD = [i for i, k in enumerate(cols) if k not in NOPROF]                  # production FD member (94 columns)
    ixN = [i for i in ixD if cols[i] not in prod.FD_COLS]                     # production pre-FD member (74 columns)
    assert len(ixD) == prod.V12_NFEAT + prod.N_FD - 4 and len(ixN) == prod.V12_NFEAT - 4
    return dict(fd=ixD, nofd=ixN)


def lgb_params(seed, leaves=255, lr=0.08):
    """production pF (t1_pipeline_v13a.py) at reduced size (production: 511 leaves, lr 0.04, 1500 trees)."""
    return dict(objective='regression', learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100, max_bin=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=THREADS,
                seed=seed, bagging_seed=seed + 10, feature_fraction_seed=seed + 20)


def member_path(outer, seed, kind, tag='', panel=PANEL):
    return f'{CACHE}/members/outer{outer}_seed{seed}_{kind}{tag}{"" if panel == PANEL else "_" + panel}.npz'


def fit_members(Z, outer, seed, kinds=('nofd', 'fd'), rounds=3, n_trees=600, leaves=255, lr=0.08, tag='', force=False):
    """One bag member per kind (same transductive rounds and seeds for both kinds, so FD vs no-FD is a paired comparison).
    Saves the clipped predictions at every pseudo-month target cell: pv_<month>, pq_<month>."""
    import lightgbm as lgb
    todo = [k for k in kinds if force or not os.path.exists(member_path(outer, seed, k, tag, Z.panel))]
    if not todo:
        return
    os.makedirs(f'{CACHE}/members', exist_ok=True)
    M = fold_matrices(Z, outer)
    cols = list(M['cols']); ix = colsets(cols); st = Z.st
    fit, months = folds(Z, outer)
    TX, TYV, TYQ = [], [], []
    for j, (nm, idx) in enumerate(months.items()):                          # transductive rows from the evaluation block
        cs = Z.view(idx)
        obs = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq)
        vf3 = np.broadcast_to(st.vf.to_numpy()[None, None, :], cs.mv.shape)
        ln3 = np.broadcast_to(st.lanes.to_numpy()[None, None, :], cs.mv.shape)
        for r in range(rounds):
            hid = obs & (np.random.default_rng(1000 * seed + 10 * r + j).random(cs.mv.shape) < prod.TR_HIDE)
            cv = Z.view(idx, hide=hid)
            Xr, _ = _feat(cv, st, M['hv'], M['hq'], hid)
            TX.append(Xr)
            TYV.append((cs.mv[hid] / vf3[hid] - Xr[:, cols.index('tv')]).astype(np.float32))
            TYQ.append((cs.mq[hid] / ln3[hid] - Xr[:, cols.index('tq')]).astype(np.float32))
    TX = np.concatenate(TX); TYV = np.concatenate(TYV); TYQ = np.concatenate(TYQ)
    if Z.panel != PANEL and len(TX) > MAX_TR_ROWS:                           # reduced replica: cap the transductive rows
        keep = np.sort(np.random.default_rng(seed).choice(len(TX), MAX_TR_ROWS, replace=False))
        TX, TYV, TYQ = TX[keep], TYV[keep], TYQ[keep]
    XA = np.concatenate([M['X'], TX]); YV = np.concatenate([M['yv'], TYV]); YQ = np.concatenate([M['yq'], TYQ])
    del TX
    log(f'outer {outer} seed {seed}: train rows {len(M["X"])} + transductive {len(XA) - len(M["X"])}')
    p = lgb_params(seed, leaves, lr)
    for kind in todo:
        ii = ix[kind]
        Xk = np.ascontiguousarray(XA[:, ii])
        t = time.time()
        mv = lgb.train(p, lgb.Dataset(Xk, YV), n_trees); mq = lgb.train(p, lgb.Dataset(Xk, YQ), n_trees)
        out = {}
        for nm, idx in months.items():
            cm = Z.view(idx); Xm = M[f'X_{nm}']
            vf_ = np.broadcast_to(st.vf.to_numpy()[None, None, :], cm.v.shape)[cm.target]
            ln_ = np.broadcast_to(st.lanes.to_numpy()[None, None, :], cm.v.shape)[cm.target]
            tvn, tqn = Xm[:, cols.index('tv')], Xm[:, cols.index('tq')]
            Xp = np.ascontiguousarray(Xm[:, ii])
            out[f'pv_{nm}'] = np.clip((tvn + mv.predict(Xp)) * vf_, 3, 1.05 * vf_).astype(np.float32)      # = predict_variants
            out[f'pq_{nm}'] = np.maximum((tqn + mq.predict(Xp)) * ln_, 20).astype(np.float32)
        np.savez(member_path(outer, seed, kind, tag, Z.panel), **out)
        log(f'outer {outer} seed {seed} {kind}: {len(ii)} cols, {n_trees} trees x2 in {time.time() - t:.0f}s')


def load_bag(month, kind, seeds, tag='', panel=PANEL):
    """mean of the members' final predictions (as swap_t1bag*.py). kind 'mix' = 0.5 FD bag + 0.5 no-FD bag (swap_t1mix.py)."""
    if kind == 'mix':
        a = load_bag(month, 'nofd', seeds, tag, panel); b = load_bag(month, 'fd', seeds, tag, panel)
        return 0.5 * a[0] + 0.5 * b[0], 0.5 * a[1] + 0.5 * b[1]
    o = MONTH_OUTER[month]
    zs = [np.load(member_path(o, s, kind, tag, panel)) for s in seeds]
    return (np.mean([z[f'pv_{month}'] for z in zs], 0).astype(np.float64),
            np.mean([z[f'pq_{month}'] for z in zs], 0).astype(np.float64))


# ------------------------------------------------------------------ evaluation month (truth, scorers, post-processing)
STAT = ('ev2', 'eq2', 'n', 'lwr_num', 'lwr_den', 'fd_num', 'fd_den')


class Month:
    """A held-out pseudo-month with truth: scoring (per-day sufficient statistics) and the production recon / TV."""

    def __init__(self, Z, name):
        self.name = name
        _, months = folds(Z, MONTH_OUTER[name])
        idx = months[name]; c = Z.view(idx); self.c = c; st = Z.st
        self.reg = c.reg; self.days = np.array(c.days); self.nd = len(idx)
        self.tm = c.target; self.shape = c.v.shape
        self.d_of = np.broadcast_to(np.arange(self.nd)[:, None, None], self.shape)[self.tm]
        self.t_of = np.broadcast_to(np.arange(288)[None, :, None], self.shape)[self.tm]
        self.s_of = np.broadcast_to(np.arange(Z.S)[None, None, :], self.shape)[self.tm]
        self.vt, self.qt = c.v[self.tm].astype(np.float64), c.q[self.tm].astype(np.float64)
        self.regrow = self.reg[self.d_of]
        # scorer parameters: organizer network_parameters() (fd_parameters averaged per link; kc = cap/vf; kj >= 1.05 kc)
        P = Z.net()
        assert P[['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph', 'critical_density', 'k_jam']].notna().all().all()
        self.lanes_s = P.lanes.to_numpy(float); self.L_s = P.length_km.to_numpy(float)
        self.vf_s = P.free_speed_kmh.to_numpy(float); self.cap_s = P.capacity_vph.to_numpy(float)
        self.kc_s = P.critical_density.to_numpy(float); self.kj_s = P.k_jam.to_numpy(float)
        assert np.allclose(self.lanes_s, st.lanes.to_numpy()) and np.allclose(self.L_s, st.length_km.to_numpy())
        self.ln_row = self.lanes_s[self.s_of]; self.vf_row = self.vf_s[self.s_of]
        # truth-side constants
        self.V0 = c.v.astype(np.float64); self.Q0 = c.q.astype(np.float64)
        self.Nt = self.Q0 / np.clip(self.V0, 1, None) * self.L_s[None, None, :]
        self.ok = c.elig[:, :-1] & c.elig[:, 1:]
        self.dNt = self.Nt[:, 1:] - self.Nt[:, :-1]
        self.lwr_den = np.where(self.ok, np.abs(self.dNt), 0).sum((1, 2))
        self.touch = (self.tm[:, :-1] | self.tm[:, 1:]) & self.ok
        self._post = None
        self.Z = Z

    # ---- scoring
    def stats(self, pv, pq):
        """per-day sufficient statistics [nd, 7] (STAT order) of a prediction at the target cells."""
        pv = np.asarray(pv, np.float64); pq = np.asarray(pq, np.float64)
        ev2 = np.bincount(self.d_of, (pv - self.vt) ** 2, self.nd)
        eq2 = np.bincount(self.d_of, ((pq - self.qt) / self.ln_row) ** 2, self.nd)
        n = np.bincount(self.d_of, minlength=self.nd).astype(float)
        V = self.V0.copy(); Q = self.Q0.copy(); V[self.tm] = pv; Q[self.tm] = pq
        K = Q / np.clip(V, 1, None)
        N = K * self.L_s[None, None, :]
        dN = N[:, 1:] - N[:, :-1]
        lwr_num = np.where(self.ok, np.abs(dN - self.dNt), 0).sum((1, 2))
        ln = self.lanes_s[None, None, :]
        kl = K / ln; ql = Q / ln; kc = self.kc_s[None, None, :] / ln; kj = self.kj_s[None, None, :] / ln; cap = self.cap_s[None, None, :] / ln
        qfd = np.where(kl <= kc, self.vf_s[None, None, :] * kl, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
        e = self.c.elig
        fd_num = np.where(e, np.abs(ql - qfd), 0).sum((1, 2)); fd_den = np.where(e, np.abs(ql), 0).sum((1, 2))
        return np.column_stack([ev2, eq2, n, lwr_num, self.lwr_den, fd_num, fd_den])

    # ---- production post-processing (functions of swap_fdrecon.py, same formulas, other constants)
    def post(self):
        if self._post is None:
            self._post = Post(self)
        return self._post


def aggregate(stats, reg, w=None):
    """stats [nd, 7], reg [nd] -> dict with per-regime and regime-mean S_state / LWR / S_FD and J. w = day weights (bootstrap)."""
    out = {}
    for r in REGS:
        m = reg == r
        if not m.any():
            continue
        s = stats[m] if w is None else stats[m] * w[m][:, None]
        t = s.sum(0)
        if t[2] <= 0:
            continue
        rv, rq = np.sqrt(t[0] / t[2]), np.sqrt(t[1] / t[2])
        out[f'S_speed_{r}'] = max(0.0, 1 - rv / 25); out[f'S_flow_{r}'] = max(0.0, 1 - rq / 600)
        out[f'rmse_v_{r}'] = rv; out[f'rmse_q_{r}'] = rq
        out[f'S_state_{r}'] = 0.54 * out[f'S_speed_{r}'] + 0.46 * out[f'S_flow_{r}']
        out[f'LWR_{r}'] = max(0.0, 1 - t[3] / t[4]); out[f'S_FD_{r}'] = max(0.0, 1 - t[5] / (t[6] + 1e-9))
    for k in ('S_state', 'LWR', 'S_FD', 'S_speed', 'S_flow'):
        out[k] = float(np.mean([out[f'{k}_{r}'] for r in REGS if f'{k}_{r}' in out]))
    out['J'] = W_STATE * out['S_state'] + W_LWR * out['LWR'] + W_FD * out['S_FD']
    return out


def J_of(stats, reg, w=None):
    return aggregate(stats, reg, w)['J']


def boot_weights(reg, B=400, seed=0):
    """day-bootstrap weights [B, nd], resampling days within each regime (stratified)."""
    rng = np.random.default_rng(seed); nd = len(reg); W = np.zeros((B, nd))
    for r in REGS:
        ii = np.flatnonzero(reg == r)
        if len(ii):
            pick = rng.integers(0, len(ii), (B, len(ii)))
            for b in range(B):
                W[b] += np.bincount(ii[pick[b]], minlength=nd)
    return W


def paired_delta(sa, sb, reg, W, key='J'):
    """delta (b - a) of an aggregate key and its day-bootstrap SE."""
    d0 = aggregate(sb, reg)[key] - aggregate(sa, reg)[key]
    ds = np.array([aggregate(sb, reg, w)[key] - aggregate(sa, reg, w)[key] for w in W])
    return d0, float(ds.std())


class Post:
    """swap_fdrecon.py's recon() / tv() bound to one pseudo-month (its target rows play the role of the state rows)."""

    def __init__(self, M):
        Z = M.Z; sw = _swap_module(); self.sw = sw; self.M = M
        day_ns = pd.to_datetime(pd.Series(M.days)).to_numpy('datetime64[ns]').astype('int64')
        tsns = day_ns[M.d_of] + M.t_of.astype('int64') * 300 * 10 ** 9
        ks = pd.DataFrame({'panel': Z.panel, 'split': M.name, 'station_id': Z.stations[M.s_of], 'link_id': Z.link_id[M.s_of]})
        FD = Z.fd_frame()
        if Z.panel == PANEL:                                                 # the kernel's own table must agree
            K = sw.fd_params(PANEL).set_index('link_id').reindex(FD.link_id)
            for col in ('lanes', 'free_speed_kmh', 'capacity_vph', 'k_jam', 'critical_density', 'capl', 'kcl', 'kjl', 'w'):
                assert np.allclose(K[col].to_numpy(float), FD[col].to_numpy(float)), col
        bad = ((FD.w < sw.W_RANGE[0]) | (FD.w > sw.W_RANGE[1]) | ~np.isfinite(FD.w) | (FD.kjl < sw.KJL_RANGE[0])
               | (FD.kjl > sw.KJL_RANGE[1]) | ~np.isfinite(FD.kjl))
        self.n_bad_links = int(bad.sum())
        FD = FD[~bad].copy(); self.FD = FD
        p = ks[['panel', 'link_id']].merge(FD, on=['panel', 'link_id'], how='left', validate='many_to_one')
        hasfd = p.lanes.notna().to_numpy()
        # released eligible cells of the month (TV anchors), as swap_fdrecon.released()
        c = M.c
        rel_m = c.melig & np.isfinite(c.mv) & np.isfinite(c.mq)
        d_, t_, s_ = np.nonzero(rel_m)
        rel = pd.DataFrame({'station_id': Z.stations[s_], 'link_id': Z.link_id[s_], 'speed_kmh': c.mv[rel_m].astype(float),
                            'flow_vph': c.mq[rel_m].astype(float), 'ts': day_ns[d_] + t_.astype('int64') * 300 * 10 ** 9})
        rel = rel.merge(FD[['link_id', 'lanes', 'free_speed_kmh', 'kjl', 'w']], on='link_id', how='left')
        rel = rel[rel.lanes.notna()]
        rel['k'] = rel.flow_vph / rel.lanes / rel.speed_kmh.clip(lower=1)
        self.rel = rel
        sw.__dict__.update(ks=ks, tsns=tsns, hasfd=hasfd, ln=p.lanes.fillna(1.0).to_numpy(float),
                           vf=p.free_speed_kmh.fillna(1e9).to_numpy(float), kjl=p.kjl.fillna(0.0).to_numpy(float),
                           w=p.w.fillna(1.0).to_numpy(float), vf_ok=hasfd.copy(), released=lambda panel, split: rel)
        self.ln, self.vf, self.kjl, self.w = sw.ln, sw.vf, sw.kjl, sw.w

    def _set(self, gate, a, logr, tv=None, iters=None):
        d = self.sw.__dict__
        d.update(GATE=gate, A=a, LOGR_MAX=logr)
        if tv is not None:
            d.update(TV_FREE=tv[0], TV_FREE_A=tv[1], TV_GATE=tv[2], TV_GATE_A=tv[3])
        if iters is not None:
            d.update(TV_ITERS=iters)

    def recon(self, v, q, gate=0.6, a=0.75, logr=0.5):
        self._set(gate, a, logr)
        return self.sw.recon(np.asarray(v, float), np.asarray(q, float))

    def tv(self, v, q, gate=0.6, a=0.75, logr=0.5, tv=(0.0075, 0.001, 0.02, 0.005), iters=300):
        self._set(gate, a, logr, tv, iters)
        return self.sw.tv(np.asarray(v, float), np.asarray(q, float), 'lab')

    def kfd(self, v):
        return self.sw.kfd_of(v, self.w, self.kjl)


# ------------------------------------------------------------------ post-processing steps that are NOT in swap_fdrecon.py
def ffproj(P, v, q, gate_ff=0.95, b=1.0, keep='q'):
    """Free-flow branch projection. Cells with v >= gate_ff * vf are moved towards the free-flow branch q = vf * k, i.e.
    their speed towards vf: v' = v * (vf / v) ** b. keep='q': flow unchanged (density becomes q / v');
    keep='k': density unchanged, q' = q * v' / v (so the cell sits on q = v' * k)."""
    g = P.sw.vf_ok & (v >= gate_ff * P.vf)
    r = np.where(g, P.vf / np.clip(v, 1, None), 1.0) ** b
    return np.where(g, v * r, v), (np.where(g, q * r, q) if keep == 'k' else q)


def curveproj(P, v, q, gate=0.6, rho=20.0, logr=None):
    """Exact projection of gate cells (v < gate * vf) onto the congested FD branch q_lane = Q(v) = w kj v / (v + w):
    v* = argmin (v* - v)^2 + ((Q(v*) - q_lane) / rho)^2 on a 0.25 km/h grid within +-40 km/h of v (rho = vph per lane that
    counts like 1 km/h; rho -> inf keeps the speed (= recon with A = 0), rho -> 0 keeps the flow). Optional clip of the
    density change |log(k*/k)| <= logr is NOT applied (the result is always on the curve)."""
    g = P.sw.vf_ok & (v < gate * P.vf)
    vn, qn = v.copy(), q.copy()
    if not g.any():
        return vn, qn
    vg, qg, w, kj, ln, vf = v[g], q[g] / P.ln[g], P.w[g], P.kjl[g], P.ln[g], P.vf[g]
    off = np.arange(-40.0, 40.01, 0.25)[None, :]
    cand = np.clip(vg[:, None] + off, 3.0, (1.05 * vf)[:, None])
    Qc = w[:, None] * kj[:, None] * cand / (cand + w[:, None])
    cost = (cand - vg[:, None]) ** 2 + ((Qc - qg[:, None]) / rho) ** 2
    j = cost.argmin(1); ii = np.arange(len(vg))
    vn[g] = cand[ii, j]; qn[g] = np.maximum(Qc[ii, j] * ln, 20.0)
    return vn, qn


def run_pipeline(M, v, q, steps):
    """steps: sequence of ('R', gate, a, logr) | ('T', gate, a, logr, (tv_free, tv_free_a, tv_gate, tv_gate_a), iters)
    | ('F', gate_ff, b, keep) | ('C', gate, rho). R and T are the swap_fdrecon.py functions."""
    P = M.post()
    v = np.asarray(v, float); q = np.asarray(q, float)
    for s in steps:
        if s[0] == 'R':
            v, q = P.recon(v, q, *s[1:])
        elif s[0] == 'T':
            v, q = P.tv(v, q, *s[1:])
        elif s[0] == 'F':
            v, q = ffproj(P, v, q, *s[1:])
        elif s[0] == 'C':
            v, q = curveproj(P, v, q, *s[1:])
        else:
            raise ValueError(s)
    return v, q


def R(gate=0.6, a=0.75, logr=0.5):
    return ('R', gate, a, logr)


def T(gate=0.6, a=0.75, logr=0.5, tv=(0.0075, 0.001, 0.02, 0.005), iters=300):
    return ('T', gate, a, logr, tuple(tv), iters)


PROD_PIPE = (R(), T())                                # the current production setting (tfb-c20): recon then TV, kernel defaults


def selftest():
    """checks: production functions compiled from the kernel files; mask hash == released masked layer;
    cube S_state == organizer score_task1 on the released train files; cube S_FD == organizer score_task3 on a day subset."""
    import inspect
    from pathlib import Path
    Z = Data()
    assert prod.features.__code__.co_filename == PROD_T1 and _sw0.recon.__code__.co_filename == PROD_SWAP
    print('production functions compiled from', os.path.basename(PROD_T1), 'line', prod.features.__code__.co_firstlineno, '|',
          os.path.basename(PROD_SWAP), 'recon line', _sw0.recon.__code__.co_firstlineno, 'tv line', _sw0.tv.__code__.co_firstlineno)
    print('days', Z.D, 'stations', Z.S, '| released masked files equal to truth + hash mask:', Z.n_checked, '/', Z.n_checked,
          '| regimes', {r: int((Z.reg == r).sum()) for r in REGS}, '| target share of eligible',
          {r: round(float(Z.target[Z.reg == r].sum() / Z.elig[Z.reg == r].sum()), 4) for r in REGS})
    # --- S_state vs organizer (released train files only: the organizer evaluator reads the masked layer from disk)
    from task1 import score_task1
    C = f'{REL}/corridors/{PANEL}'
    key = lambda f: os.path.basename(f)[-18:-8].replace('_', '-')
    have = {key(f) for f in glob.glob(f'{C}/train/mainline_states_masked/*/*.parquet')} & set(Z.days)
    idx = np.array([i for i, d in enumerate(Z.days) if d in have])
    c = Z.view(idx)
    tvv = prod.interp_axis(c.mv, 1); tqq = prod.interp_axis(c.mq, 1)
    hv, hq = profile(c)
    tvv = np.where(np.isfinite(tvv), tvv, hv[c.dow]); tqq = np.where(np.isfinite(tqq), tqq, hq[c.dow])
    d_, t_, s_ = np.nonzero(c.target)
    ts = [f'{c.days[d]}T{t // 12:02d}:{(t % 12) * 5:02d}:00Z' for d, t in zip(d_, t_)]
    sub = pd.DataFrame({'panel': PANEL, 'timestamp': pd.to_datetime(ts, utc=True), 'station_id': Z.stations[s_], 'link_id': Z.link_id[s_],
                        'mask_regime': c.reg[d_], 'speed_kmh': tvv[c.target], 'flow_vph': tqq[c.target]})
    rows = score_task1.score_panel(PANEL, Path(REL), 'train', sub)
    org = {r['regime']: r for r in rows}
    lanes = Z.st.lanes.to_numpy()[s_]
    S_, per = prod.s_state(tvv[c.target].astype(float), tqq[c.target].astype(float), c.v[c.target].astype(float),
                           c.q[c.target].astype(float), lanes, c.reg[d_])
    # my per-day statistics path
    fake = types.SimpleNamespace()
    ev2 = np.bincount(d_, (tvv[c.target] - c.v[c.target]).astype(float) ** 2, len(idx))
    eq2 = np.bincount(d_, ((tqq[c.target] - c.q[c.target]).astype(float) / lanes) ** 2, len(idx))
    n = np.bincount(d_, minlength=len(idx)).astype(float)
    st7 = np.column_stack([ev2, eq2, n, np.zeros(len(idx)), np.ones(len(idx)), np.zeros(len(idx)), np.ones(len(idx))])
    ag = aggregate(st7, c.reg)
    for r in REGS:
        print(f'  S_state {r}: organizer {org[r]["S_state"]:.6f} (n {org[r]["n_target_cells"]}, missing {org[r]["n_missing_predictions"]})'
              f' | production s_state {per[r]:.6f} | lab aggregate {ag[f"S_state_{r}"]:.6f}')
        assert abs(org[r]['S_state'] - ag[f'S_state_{r}']) < 1e-6 and org[r]['n_missing_predictions'] == 0
        assert org[r]['n_target_cells'] == int(n[c.reg == r].sum())
    print('  S_state macro: organizer', round(org['macro_mean']['S_state'], 6), '| lab', round(ag['S_state'], 6))
    # --- S_FD vs organizer derive_physics + score_panel on month A (template replaced by the hash targets in memory)
    from task3 import score_task3
    M = Month(Z, 'A')
    pv, pq = M.vt * 1.02, M.qt * 0.97                                        # any prediction will do
    day_set = set(M.days)
    full_files = [Path(f) for f in sorted(glob.glob(f'{C}/train/mainline_states/*/*.parquet')) if key(f) in day_set]
    tsr = [f'{M.days[d]}T{t // 12:02d}:{(t % 12) * 5:02d}:00Z' for d, t in zip(M.d_of, M.t_of)]
    subm = pd.DataFrame({'panel': PANEL, 'timestamp': tsr, 'station_id': Z.stations[M.s_of], 'link_id': Z.link_id[M.s_of],
                         'mask_regime': M.regrow, 'speed_kmh': pv, 'flow_vph': pq})
    o_files, o_targets = score_task3.files, score_task3.published_targets
    try:
        score_task3.files = lambda panel_dir, split: full_files
        score_task3.published_targets = lambda panel_dir, panel, split, regime: set(zip(
            pd.to_datetime(subm.timestamp[subm.mask_regime == regime], utc=True), subm.station_id[subm.mask_regime == regime],
            subm.link_id[subm.mask_regime == regime]))
        cfg = json.load(open(os.path.join(ROOT, 'repo', 'config', 'task3_lwr_modes.json')))
        ag = aggregate(M.stats(pv, pq), M.reg)
        for r in REGS:
            phys, missing = score_task3.derive_physics(PANEL, Path(REL), 'train', subm, r)
            res = score_task3.score_panel(PANEL, phys, Path(REL), cfg, r)
            print(f'  S_FD {r}: organizer {res["S_FD"]:.6f} (missing {missing}) | lab {ag[f"S_FD_{r}"]:.6f}')
            assert abs(res['S_FD'] - ag[f'S_FD_{r}']) < 1e-6 and missing == 0
    finally:
        score_task3.files, score_task3.published_targets = o_files, o_targets
    # --- LWR proxy: lab == production lwr_proxy (which pools the regimes of the days it is given)
    V = M.c.v.astype(np.float64).copy(); Q = M.c.q.astype(np.float64).copy(); V[M.tm] = pv; Q[M.tm] = pq
    cc = types.SimpleNamespace(v=M.c.v.astype(np.float64), q=M.c.q.astype(np.float64), elig=M.c.elig, reg=M.reg)
    lp = prod.lwr_proxy(cc, Z.st, M.tm, V, Q)
    print('  LWR proxy: production lwr_proxy', round(lp, 6), '| lab', round(ag['LWR'], 6))
    assert abs(lp - ag['LWR']) < 1e-9
    print('selftest OK')


if __name__ == '__main__':
    selftest()
