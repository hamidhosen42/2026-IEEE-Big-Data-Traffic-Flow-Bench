"""TRACK ogphys (EXP_T2_PHYS): two ongoing-feature ideas of the 10-29 public notebooks on the production-like lab model.

PRE-REGISTERED 2026-10-30 14:45 +06, written before any ogphys feature, model or number was computed.

Base (mirror of production fs4occ at lab scale = the occ track, analysis/t2lab/occ_lib.py): F03 rows on all-finite inputs (allfin
RowCaches faith3F03 x0.3 + runstart x1.0), np / id production columns + the 13 occupancy columns, LightGBM 255 leaves, lr 0.05,
min_data_in_leaf 40, 500 trees, seed 0, held-out folds 0 and 1 (existing arms occ_np255 / occ_id255, cache/occ/oof/<arm>/model_f<k>.txt).
Candidates = the SAME rows, labels, weights and parameters plus side-car columns computed only from data <= T: the 12-slot history
T-60..T-5 (all-finite masked layer), the slot-T masked-layer value (training windows: the production draw rng(len(wins)), 62 % of the
eligible cells visible, ineligible-valued cells always; pseudo-March / official windows: the released masked layer at T) and static
link data (cut = 0.6 vf, cap, lanes, length).  Nothing after T is read.
  (f) occF  robinamirbahar cell [7]: fill links masked at T from v(T-5) plus the neighbours' change, and flow at T (8 columns)
        v_b = forward-filled history speed at T-5; dT = v(T) - v_b where visible; nbd = mean of dT over the visible links l - 1, l + 1
        vT_fill  = v(T) / cut if visible, else clip(v_b + clip(nbd, +-15 km/h; 0 if none), 5, 130) / cut
        qT_fill  = vT_fill <= 1 (NaN -> free);  dvT_nb = nbd / cut (NaN if no visible neighbour)
        fr_T     = flow(T) / cap (NaN where masked);  dfr_T = fr_T - forward-filled flow ratio at T-5
        share_qTf = corridor share of qT_fill;  dqf_hi / dqf_lo = production dist_to_queued on the qT_fill state
  (g) occG  plarmuseau 'physics of a synthetic jam' [17], [19]: vehicle count / predicted jam length (12 columns)
        state 'now' = the production 'last' state (speed at T if visible else v_b; flow at T if visible else the T-5 flow);
        density k = q / max(v, 5) veh/km (cross-section); 15 / 30 min ago = history slots -4 / -7 (production q15 / q30 convention);
        links without a density take kc = cap / vf (neutral); kj = 100 x lanes; Nc = sum kc L (corridor at critical density)
        corridor (broadcast): c_N = sum k L / Nc, c_dN15 / c_dN30 = sum (k - k15 / k30) L / Nc, c_jl_now = km of queued links (production
          qlast), c_jl_star = max(0, sum (k - kc) L) / mean(kj - kc) km, c_jl_gap = c_jl_star - c_jl_now
        per queue cluster of qlast (1-link gaps bridged; downstream = index +1 on E/N panels, -1 on W/S, xfer_shock_lib): segment =
          cluster + U = 5 links upstream of the tail; N_seg, N0_seg = sum kc L; jl_star = max(0, N_seg - N0_seg) / mean_seg(kj - kc)
        per link, reference cluster = its own cluster, else the nearest cluster downstream within R = 10 links (stopping at the
          previous cluster): g_cum_km = km from the head end to the link midpoint, g_jl_now = cluster km, g_jl_star, g_margin =
          g_jl_star - g_cum_km, g_dN15 / g_dN30 = (N_seg - N_seg 15 / 30 min ago) / N0_seg; NaN without a reference cluster
Arms: occF_{np,id}255 = base + (f); occG_{np,id}255 = base + (g); folds 0, 1 (fold-k occupancy calibration and fold-k side-cars).
GATE (pre-registered, binding; same rule as EXP_T2_RAMPSCEN): FM-PM = footprint-matched pseudo-March forecast
  (analysis/t2lab/results/xfer/scripts/fpmatch_lib.forecast: panel-exact, +-2 h, same sign and |dn - dn_w| <= max(3, 0.5 |dn_w|),
  pooled fallback for n < 5, all-cell decoded counts, mean over the 40 windows x 0.15);
  lab pairs = pseudo-March offlike_m + runstart_m windows of the masked validation month, decisions of the np + id mean of the
  fold-0 and fold-1 models (base vs candidate); official footprint = the same lab models decoded on the 40 official March ongoing
  windows (validation) and the 40 April ones (private).
  PASS iff FM-PM validation >= +0.0015 AND FM-PM private >= +0.0005 (LB-equivalent).  Reported, not binding: train OOF (official-40,
  offlike, pseudo; folds 0 + 1; np and np + id), pseudo-March offlike_m / runstart_m paired deltas (bootstrap CI, panels positive).
  If both arms pass: deploy the one with the larger FM-PM validation; the combination (f + g) is not tested unless one passes.
AMENDMENT (2026-10-30 15:40 +06, written after a DATA check and the train-OOF log lines of fold 0/1, BEFORE any pseudo-March,
  official-footprint or FM-PM number of any arm existed): the head of a queue (the stable end over 30 min, matched clusters of the
  t36-filled train state) sits at the HIGH link index on ALL 8 panels (P(high end moved) 0.01-0.13 vs low end 0.02-0.78), so the
  xfer_shock_lib rule (-1 on W/S panels) is reversed on D7_I10_W, D7_I210_W, D7_I405_S, D12_I5_S. occG is kept exactly as registered.
  New arm occGd = occG with downstream = +1 on every panel (same 12 columns, same U / R), same rows / params / folds / gate.
  Three arms now share the gate; if several pass, the largest FM-PM validation is deployed (rule above).
RESULT (16:45 +06, logs/ogphys_report.log, results/ogphys/deltas.csv + fm.csv): ALL FAIL the binding FM-PM gate (np + id decisions):
  occF  val +0.00059 / pri +0.00015 (26 / 23 windows changed); pseudo-March offlike_m +0.0033 [-0.0030, +0.0107], runstart_m +0.0015;
        train OOF offlike +0.0001 (vT_fill takes 34 % of the gain, the model mostly re-reads the slot-T state)
  occG  val -0.00043 / pri +0.00094 (28 / 31 windows, dn -70 / -36: shrinking); pseudo-March offlike_m +0.0094 [+0.0017, +0.0174],
        runstart_m +0.0052 [+0.0008, +0.0097]; train OOF offlike +0.0088 [+0.0054, +0.0120] 7/8
  occGd val +0.00151 / pri -0.00002 (27 / 30 windows, dn -89 / -39); pseudo-March offlike_m +0.0102 [+0.0029, +0.0179] 6/8,
        runstart_m +0.0079 [+0.0036, +0.0125] 7/8; train OOF offlike +0.0076 [+0.0039, +0.0113] 7/8
  np-only decisions (info): occF +0.00024 / -0.00034, occG -0.00035 / +0.00071, occGd +0.00101 / +0.00037 (FM-PM noise ~ +-0.0005).
  No kernel. Side finding: xfer_shock_lib.downstream_sign is reversed on the 4 W/S panels (EXP_T2_SHOCK used it).
Kernel rule: a passing arm needs an ongoing model kernel (adapt kaggle_jobs/t2_occ_v26.py for the March rows and
kaggle_jobs/t2_ongoing_pvogf.py for pick A's April rows), never pushed or submitted from here.
Caches: data/t2local/cache/ogphys/ (side-cars, arms), results analysis/t2lab/results/ogphys/.
"""
import gc, json, os, time, warnings
import numpy as np
import pandas as pd
import lab
import occ_lib as oc
import xfer_shock_lib as xs

H, F, ONGOING = lab.H, lab.F, lab.ONGOING
PC = f'{lab.CACHE}/ogphys'
RES = f'{lab.ROOT}/analysis/t2lab/results/ogphys'
for _d in (PC, RES, f'{PC}/side', f'{PC}/side_d', f'{PC}/arms', f'{PC}/pm', f'{PC}/off'):
    os.makedirs(_d, exist_ok=True)
F_COLS = ['vT_fill', 'qT_fill', 'dvT_nb', 'fr_T', 'dfr_T', 'share_qTf', 'dqf_hi', 'dqf_lo']
G_COLS = ['c_N', 'c_dN15', 'c_dN30', 'c_jl_now', 'c_jl_star', 'c_jl_gap', 'g_cum_km', 'g_jl_now', 'g_jl_star', 'g_margin', 'g_dN15', 'g_dN30']
ALL = F_COLS + G_COLS
ARM_COLS = {'occF': F_COLS, 'occG': G_COLS, 'occGd': G_COLS}
ARM_VAR = {'occF': 'xs', 'occG': 'xs', 'occGd': 'd'}       # side-car variant: 'xs' = xfer_shock_lib direction, 'd' = +1 everywhere
VAR_SIGN = {'xs': None, 'd': 1}
U_SEG, R_REF = 5, 10
TRAIN_SETS = [n for n, _ in oc.TRAIN]                     # faith3F03, runstart
EVAL_SETS = ('official', 'offlike', 'pseudo')
SIDE_SETS = ('faith3', 'runstart') + EVAL_SETS
ffill_time, shift_links, dist_to_queued = lab.prod.ffill_time, lab.prod.shift_links, lab.prod.dist_to_queued


# ============================================================================================ kernel-portable block
def phys_feats(P, Vh, Fh, VT, FT, panel, sign=None):
    """(f) + (g) columns {name: [W, L] float32} and the production qlast [W, L] from the history Vh, Fh [W, 12, L] (km/h, vph, NaN =
    missing) and the slot-T masked-layer values VT, FT [W, L] (NaN = not visible)."""
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        cut = P['cut']; cap = P['cap']; W, _, L = Vh.shape
        vr = Vh / cut[None, None, :]; fr = Fh / cap[None, None, :]
        vrf = ffill_time(vr); frf = ffill_time(fr)
        vT = VT / cut[None, :]
        last = np.where(np.isfinite(vT), vT, vrf[:, -1]); qlast = np.nan_to_num(last, nan=9) <= 1.0     # = production window_features
        bc = lambda a: np.repeat(np.asarray(a, np.float32)[:, None], L, 1)
        X = {}
        # ---------------- (f) fill at T from v(T-5) + the neighbours' change, flow at T
        vb = vrf[:, -1]
        dT = vT - vb
        nb = np.stack([shift_links(dT, -1), shift_links(dT, 1)])
        cnt = np.isfinite(nb).sum(0)
        nbd = np.where(cnt > 0, np.nansum(nb, 0) / np.maximum(cnt, 1), np.nan)
        lim = (15.0 / cut)[None, :]
        filled = np.clip(vb + np.nan_to_num(np.clip(nbd, -lim, lim), nan=0.0), (5.0 / cut)[None, :], (130.0 / cut)[None, :])
        fill = np.where(np.isfinite(vT), vT, filled)
        qf = np.nan_to_num(fill, nan=9.0) <= 1.0
        frT = FT / cap[None, :]
        X['vT_fill'] = fill; X['qT_fill'] = qf.astype(np.float32); X['dvT_nb'] = nbd
        X['fr_T'] = frT; X['dfr_T'] = frT - frf[:, -1]
        X['share_qTf'] = bc(qf.mean(1)); X['dqf_hi'] = dist_to_queued(qf, 1); X['dqf_lo'] = dist_to_queued(qf, -1)
        # ---------------- (g) vehicle count / predicted jam length
        Ls = P['length'].astype(np.float64); vf = P['vf'].astype(np.float64); lanes = P['lanes'].astype(np.float64)
        kc = cap / np.maximum(vf, 1.0); kj = 100.0 * lanes
        vnow = last * cut[None, :]
        qnow = np.where(np.isfinite(vT) & np.isfinite(FT), FT, frf[:, -1] * cap[None, :])

        def dens(v, q):
            k = np.where(np.isfinite(v) & np.isfinite(q), q / np.maximum(v, 5.0), np.nan)
            return np.where(np.isfinite(k), k, kc[None, :])
        K0 = dens(vnow, qnow); K15 = dens(vrf[:, -4] * cut[None, :], frf[:, -4] * cap[None, :]); K30 = dens(vrf[:, -7] * cut[None, :], frf[:, -7] * cap[None, :])
        Nc = float((kc * Ls).sum())
        X['c_N'] = bc((K0 * Ls).sum(1) / Nc)
        X['c_dN15'] = bc(((K0 - K15) * Ls).sum(1) / Nc); X['c_dN30'] = bc(((K0 - K30) * Ls).sum(1) / Nc)
        jl_now = (qlast * Ls[None, :]).sum(1); jl_star = np.maximum(0.0, ((K0 - kc[None, :]) * Ls).sum(1)) / float(np.mean(kj - kc))
        X['c_jl_now'] = bc(jl_now); X['c_jl_star'] = bc(jl_star); X['c_jl_gap'] = bc(jl_star - jl_now)
        s = xs.downstream_sign(panel) if sign is None else sign
        G = {c: np.full((W, L), np.nan, np.float64) for c in ('g_cum_km', 'g_jl_now', 'g_jl_star', 'g_margin', 'g_dN15', 'g_dN30')}
        for i in range(W):
            cl = xs.clusters(qlast[i])
            for j, (lo, hi) in enumerate(cl):
                if s == 1:                                       # downstream = higher index: head hi, tail lo, upstream = lower index
                    seg = np.arange(max(0, lo - U_SEG), hi + 1)
                    stop = cl[j - 1][1] + 1 if j > 0 else 0
                    path = np.arange(hi, max(stop, lo - R_REF) - 1, -1)
                else:                                            # downstream = lower index: head lo, tail hi
                    seg = np.arange(lo, min(L, hi + U_SEG + 1))
                    stop = cl[j + 1][0] - 1 if j + 1 < len(cl) else L - 1
                    path = np.arange(lo, min(stop, hi + R_REF) + 1)
                n0 = float((kc[seg] * Ls[seg]).sum()); ns = float((K0[i, seg] * Ls[seg]).sum())
                jls = max(0.0, ns - n0) / float(np.mean(kj[seg] - kc[seg]))
                cum = np.cumsum(Ls[path]) - Ls[path] / 2
                G['g_cum_km'][i, path] = cum; G['g_jl_now'][i, path] = float(Ls[lo:hi + 1].sum()); G['g_jl_star'][i, path] = jls
                G['g_margin'][i, path] = jls - cum
                G['g_dN15'][i, path] = (ns - float((K15[i, seg] * Ls[seg]).sum())) / n0
                G['g_dN30'][i, path] = (ns - float((K30[i, seg] * Ls[seg]).sum())) / n0
        X.update(G)
    assert list(X) == ALL
    return {k: np.asarray(v, np.float32) for k, v in X.items()}, qlast
# ============================================================================================ end of portable block


def stack(X):
    return np.stack([X[c] for c in ALL], -1).astype(np.float32)


def chunked(P, Vh, Fh, VT, FT, panel, n=2048, sign=None):
    parts, ql = [], []
    for a in range(0, len(Vh), n):
        X, q = phys_feats(P, Vh[a:a + n], Fh[a:a + n], VT[a:a + n], FT[a:a + n], panel, sign); parts.append(stack(X)); ql.append(q)
    L = Vh.shape[2]
    return (np.concatenate(parts) if parts else np.zeros((0, L, len(ALL)), np.float32)), (np.concatenate(ql) if ql else np.zeros((0, L), bool))


# ------------------------------------------------------------------ train-split side-cars (window order / slot-T draw = occ build_side)
def side_path(panel, name, var='xs'):
    return f'{PC}/{"side" if var == "xs" else "side_" + var}/{panel}__{oc.af_name(name)}__{ONGOING}.npy'


def build_side(P, name, var='xs'):
    """[W, L, 20] side-car of the allfin RowCache (panel, set, ongoing); the slot-T draw is the production draw of the FULL window
    list (rng(len(wins)), 62 %), restricted afterwards for faith3 -- exactly occ_lib.build_side; parity: my qlast == the cached
    rows' qlast and my VT reproduces has_T / vr_T / vr_last of the cached rows."""
    panel = P['panel']
    win = lab.window_set(panel, name); win = win[win.cond == ONGOING].reset_index(drop=True)
    rc = oc.rows_af(panel, name, ONGOING)
    st = dict(panel=panel, set=oc.af_name(name))
    L = len(P['links'])
    if not len(win):
        np.save(side_path(panel, name, var), np.zeros((0, L, len(ALL)), np.float32)); return dict(st, windows=0, par_ok=True)
    Va, Fa, _ = oc.load_cubes(P)
    d = win.d.to_numpy(); t = win.t.to_numpy()
    rng = np.random.default_rng(len(win))
    draw = rng.random((len(d), L)) < oc.P_VIS
    if name == 'faith3':
        sub = oc.f03_sub(win); d, t, draw = d[sub], t[sub], draw[sub]
    assert np.array_equal(rc.win.d.to_numpy(), d) and np.array_equal(rc.win.t.to_numpy(), t), 'window order differs from the allfin rows'
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Va[d[:, None], idx]; Fh = Fa[d[:, None], idx]
    VT = np.where(draw, P['V'][d, t], np.nan)
    VT = np.where(P['E'][d, t], VT, P['Vx'][d, t])                 # ineligible-valued cells: always visible
    FT = np.where(np.isfinite(VT), Fa[d, t], np.nan).astype(np.float32)
    S, ql = chunked(P, Vh, Fh, VT, FT, panel, sign=VAR_SIGN[var])
    st.update(var=var, windows=len(d), rows=int(rc.n), T_vis=float(np.isfinite(VT).mean()), FT_vis=float(np.isfinite(FT).mean()))
    st['par_qlast'] = bool(np.array_equal(ql, rc.qlast))
    if rc.n:
        cols = rc.cols; wi, li = rc.wi, rc.li
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            vT = VT / P['cut'][None, :]; vrf = ffill_time(Vh / P['cut'][None, None, :]); last = np.where(np.isfinite(vT), vT, vrf[:, -1])
        for cn, ref in (('has_T', np.isfinite(vT).astype(np.float32)), ('vr_T', vT.astype(np.float32)), ('vr_last', last.astype(np.float32))):
            st[f'par_{cn}'] = bool(np.array_equal(np.asarray(rc.X[:, cols.index(cn)]), ref[wi, li], equal_nan=True))
    st['par_ok'] = bool(all(v for k, v in st.items() if k.startswith('par_')))
    st['fill_share_masked'] = float(np.isfinite(S[..., ALL.index('dvT_nb')])[~np.isfinite(VT)].mean()) if (~np.isfinite(VT)).any() else 0.0
    st['g_ref_share'] = float(np.isfinite(S[..., ALL.index('g_cum_km')]).mean())
    tmp = f'{side_path(panel, name, var)}.{os.getpid()}.tmp.npy'
    np.save(tmp, S); os.replace(tmp, side_path(panel, name, var))
    return st


def ensure_side(panels=None, var='xs'):
    out = []
    jd = f'{PC}/{"side" if var == "xs" else "side_" + var}'
    for panel in panels or lab.PANELS:
        if all(os.path.exists(side_path(panel, n, var)) for n in SIDE_SETS) and os.path.exists(f'{jd}/{panel}.json'):
            out += json.load(open(f'{jd}/{panel}.json')); continue
        P = lab.load_panel(panel, cache=False); oc.load_cubes(P)
        sts = []
        for n in SIDE_SETS:
            t1 = time.time(); st = build_side(P, n, var); st['secs'] = round(time.time() - t1, 1); sts.append(st)
            lab.log(f'ogphys side {panel} {n}: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}))
        json.dump(sts, open(f'{jd}/{panel}.json', 'w')); out += sts
        del P; gc.collect()
    return pd.DataFrame(out)


class Side:
    def __init__(self, panel, name, var='xs'):
        self.S = np.load(side_path(panel, name, var), mmap_mode='r')

    def rows(self, wi, li, cols):
        ix = [ALL.index(c) for c in cols]
        return np.asarray(self.S[np.asarray(wi)[:, None], np.asarray(li)[:, None], np.asarray(ix)[None, :]], np.float32)


# ------------------------------------------------------------------ training / prediction (rows and order = occ_lib.train_matrix)
def train_matrix(k, cols, extra, var='xs'):
    parts = []; n = 0
    for panel in lab.PANELS:
        for name, sw in oc.TRAIN:
            rc = oc.rows_af(panel, name, ONGOING)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((panel, name, rc, m, ww, c)); n += c
    nc = len(cols) + len(oc.OCC_COLS) + len(extra)
    X = np.empty((n, nc), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for panel, name, rc, m, ww, c in parts:
        r = np.where(m)[0]
        Xb = oc.matrix(rc, oc.Side(panel, name, ONGOING), m, cols, k, True)
        X[o:o + c, :Xb.shape[1]] = Xb
        if extra:
            X[o:o + c, Xb.shape[1]:] = Side(panel, name, var).rows(rc.wi[r], rc.li[r], extra)
        y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w


def eval_matrix(rc, panel, name, m, cols, k, extra, var='xs'):
    Xb = oc.matrix(rc, oc.Side(panel, name, ONGOING), m, cols, k, True)
    if not extra:
        return Xb
    r = np.where(m)[0]
    return np.hstack([Xb, Side(panel, name, var).rows(rc.wi[r], rc.li[r], extra)])


def arm_dir(arm):
    return f'{PC}/arms/{arm}'
