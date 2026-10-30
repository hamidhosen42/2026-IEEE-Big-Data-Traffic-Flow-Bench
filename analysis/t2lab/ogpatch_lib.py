"""TRACK ogpatch (EXP_T2_PATCH): a RAW SPACE-TIME PATCH feature block on the production-like lab model (T2 lab round 6).

PRE-REGISTERED 2026-10-30 09:34 +06 (machine clock, `date`), written before any ogpatch feature, model or number was computed.

Idea source: research/public_kernels_2026-10-29/plarmuseau_trafficflowbench-the-physics-of-a-synthetic-jam, markdown section 7 (cell [28]):
on their own generated ongoing windows (train Jul-Dec, test February) LightGBM on the raw 9 x 9 space-time patch (speed around the cell,
links x last time steps, no engineering; 162 inputs) scored 0.86 IoU and the raw 21 x 9 patch (378 inputs) 0.864, against 0.85 for 52
engineered features.  Their windows are not the official selection and their base model is weaker than ours.

BASE (identical to EXP_T2_PHYS / ogphys_lib.py): the occ track = production fs4occ at lab scale: F03 rows on all-finite inputs (allfin
RowCaches faith3F03 x 0.3 + runstart x 1.0), np / id production columns + the 13 occupancy columns (fold-k calibration), LightGBM 255 leaves,
lr 0.05, min_data_in_leaf 40, 500 trees, seed 0 (lab.lgb_params), held-out folds 0 and 1.  Base models = the existing arms occ_np255 /
occ_id255 (cache/occ/oof/<arm>/model_f<k>.txt), reused, not refitted.
CANDIDATES = the SAME rows, labels, weights, parameters, folds and np / id arms, plus patch columns appended after the occupancy columns,
computed only from released data stamped <= T with the ogphys history / slot-T conventions:
  strip per window [9, L]: slots T-40, T-35, ..., T-5 = the last 8 slots of the all-finite masked 12-slot history T-60..T-5 (eligible values
    + ineligible-valued cells, NaN = missing) and slot T = the slot-T value: training windows the production draw rng(len(wins)) of the FULL
    window list (62 % of the eligible cells visible, ineligible-valued cells always; faith3 restricted to F03 afterwards = occ_lib / ogphys
    build_side); pseudo-March windows the masked validation layer at T (eligible + ineligible-valued, = ogphys run_pm); official windows
    the released masked layer at T + ineligible values (= ogphys official_inputs).
  channels: pv = v / cut (cut = 0.6 vf of the link the value belongs to); pq = q / cap (cap of that link); the slot-T flow is visible exactly
    where the slot-T speed is; history flows as the production flow features use them.
  row (window w, link l, step h) gets strip[w, j, l + k] for slot j and link offset k, the same for every step h.  Orientation: link index +1
    = one link DOWNSTREAM on EVERY panel = the native lab / production link order (round-5 amendment: the stable queue head sits at the HIGH
    index on all 8 panels, so no panel is flipped and xfer_shock_lib.downstream_sign is NOT used); l + k outside the corridor -> NaN;
    missing -> NaN; no fill, no engineering.
  column names pv{m:02d}_n{k:+d} / pq{m:02d}_n{k:+d} (m = minutes before T: 40, 35, ..., 5, 0; order m outer 40 -> 0, k inner -R..+R).
ARMS (three share the gate; occP9q is the ONE optional variant, fixed here):
  occP9   pv on links l-4..l+4 x the 9 slots T-40..T                                     81 columns
  occP21  pv on links l-10..l+10 x the 9 slots T-40..T                                  189 columns
  occP9q  occP9 + pq (flow / capacity) on the same 9 x 9 cells                          162 columns
          (the notebook's 9 x 9 model had 162 = 2 x 81 inputs; flow is the other released channel; forward fill is NOT tested)
  Each arm: np and id models on folds 0 and 1 (4 fits per arm).  No other arm, no combination arm, no tuning.
GATE (PRE-REGISTERED, binding, unchanged from round 5 / EXP_T2_PHYS): FM-PM = footprint-matched pseudo-March forecast
  (analysis/t2lab/results/xfer/scripts/fpmatch_lib.forecast with its defaults: panel-exact, +-2 h, same sign and |dn - dn_w| <=
  max(3, 0.5 |dn_w|), pooled fallback for n < 5, all-cell decoded counts; 0.15 x mean over the 40 windows); lab pairs = pseudo-March
  offlike_m + runstart_m windows of the masked validation month (the ogphys run_pm selection), decisions of the np + id mean of the fold-0
  and fold-1 models (4 models, base vs candidate); official footprint = the same lab models decoded on the 40 official March ongoing windows
  (validation) and the 40 April ones (private).
  PASS iff FM-PM validation >= +0.0015 AND FM-PM private >= +0.0005 (LB-equivalent).  If several arms pass: deploy the one with the largest
  FM-PM validation.
  Multiplicity: three arms against one gate (as round 5).  FM-PM noise is about +-0.0005 per footprint (round 5: np vs np + id decisions);
  the arms are correlated (occP9 is nested in occP21 and in occP9q), so the family-wise false-pass chance is above the single-arm chance; a
  pass is reported with that caveat.
NOT BINDING (reported): train OOF (official-40, offlike, pseudo; folds 0 + 1; np and np + id decisions); pseudo-March offlike_m / runstart_m
  paired deltas (lab.paired_bootstrap CI, panels positive); official windows changed, cells changed, delta predicted cells dn, cells added
  outside the queue at T; LightGBM gain share of the patch columns.  Info: FM-PM with np-only decisions.
INTEGRITY (asserted, not tuned): (i) the strips reproduce the cached production columns bit-for-bit where defined (pv00_n+0 = vr_T,
  pv05_n+0 = vr_raw_last; finite pv00_n+-1..3 = vr_n+-1..3; finite pv10 / 15 / 20 / 35_n+0 = vr_l1 / l2 / l3 / l6; finite pq05_n+0 = fr_last)
  on every train / eval set and on every pseudo-March and official block; (ii) base parity: this code path with ZERO patch columns (same
  matrix builder and fit) retrains occ_np255 fold 0 and must reproduce cache/occ/oof/occ_np255/model_f0.txt (trees identical); if it does
  not, the plain exp_firstcross_lib.fit path is run on the same matrix: if that equals the new fit, the difference is environmental (reported)
  and this path stands, otherwise the candidates are fitted with the plain path; (iii) the base probabilities of the pm / off jobs equal the
  ogphys pm / off caches (same windows, same models).
  Memory (shared 25 GB machine): the matrix of a fit is built under a file lock (one build at a time across these jobs), the LightGBM Dataset
  is constructed from it and the float matrix is freed before boosting (bit-identity with the plain path: checked by (ii)).
Kernel rule: a passing arm gets kernel changes prepared, never pushed, run on Kaggle or submitted from here: kaggle_jobs/t2_occ_v26.py for the
  March rows and kaggle_jobs/t2_ongoing_pvogf.py for pick A's April rows (and the hedge's April rows if applicable), plus a push order.
Rule (organizer 745700): only released data stamped <= T (the row at T included) enters a feature, fit or selection; nothing after T, no ramp
  values inside the horizon; the leakdiag_* scripts are never imported.
Caches: data/t2local/cache/ogpatch/ (strips, arms, pm, off); results analysis/t2lab/results/ogpatch/; logs analysis/t2lab/logs/ogpatch_*.log.
RESULT (2026-10-30 10:33 +06, appended after all numbers; the pre-registered text above is unchanged: sha1 e0c149b4 of this file at 09:36:46 +06
  = the code that ran; logs/ogpatch_report.log, results/ogpatch/deltas.csv + fm.csv + summary.csv + official_footprint_*.csv):
  ALL THREE FAIL the binding FM-PM gate (np + id decisions):
  occP9   val +0.00057 / pri +0.00008 (22 / 21 windows changed, 69 / 86 cells, dn -11 / -10); pseudo-March offlike_m +0.0042 [-0.0005, +0.0100] 6/8,
          runstart_m +0.0032 [+0.0007, +0.0061] 5/8; train OOF official-40 -0.0013 [-0.0062, +0.0037], offlike -0.0019 [-0.0036, -0.0003] 1/8,
          pseudo +0.0021 [+0.0000, +0.0041] 6/8
  occP21  val +0.00048 / pri -0.00024 (23 / 28 windows, 111 / 138 cells, dn -11 / -10); pseudo-March offlike_m +0.0045 [-0.0013, +0.0114] 6/8,
          runstart_m +0.0044 [+0.0014, +0.0079] 8/8; train OOF official-40 -0.0016 [-0.0077, +0.0037], offlike -0.0017 [-0.0040, +0.0005] 2/8,
          pseudo +0.0115 [+0.0009, +0.0221] 6/8
  occP9q  val +0.00093 / pri -0.00006 (24 / 22 windows, 93 / 93 cells, dn -15 / -15); pseudo-March offlike_m +0.0059 [+0.0002, +0.0129] 6/8,
          runstart_m +0.0045 [+0.0012, +0.0082] 8/8; train OOF official-40 -0.0051 [-0.0143, +0.0029], offlike -0.0005 [-0.0026, +0.0015] 3/8,
          pseudo +0.0016 [-0.0009, +0.0044] 4/8
  np-only decisions (info): occP9 +0.00010 / -0.00029, occP21 +0.00112 / -0.00020, occP9q +0.00079 / +0.00007 (FM-PM noise ~ +-0.0005).
  Gain share of the patch columns (np, fold 0 / 1): occP9 2.6 / 2.7 %, occP21 4.9 / 5.0 %, occP9q 4.0 / 4.1 % (o_last 0.48-0.61 and or_last 0.11-0.15
  of the gain; the top patch columns are the oldest slot T-40 on upstream links n-3 / n-4 / n-10).  Official edits are small and mostly removals
  (|dn| <= 12 per window, 6-18 cells at most; removed 40-74 vs added outside the queue at T 8-25 per split).
  Integrity: (i) strips == production columns on all 40 train / eval sets and on every pm / off block; (ii) the zero-patch path reproduces the
  stored occ_np255 fold 0 model string exactly; (iii) base probabilities == the ogphys caches (pm 64/64, off 64/64).  An early pm / off / report
  run on occP9 + occP21 (logs/ogpatch_*_early.log, before the occP9q fits ended) gave the same numbers as the final run.
  Reading: in-distribution the raw patch adds nothing the 58 / 64 engineered + 13 occupancy columns do not already carry (train OOF offlike
  -0.0005..-0.0019); the pseudo-March gains (+0.003..+0.006) come with small, mostly shrinking official edits that FM-PM prices at ~0 on the
  private footprint.  The notebook's 0.86 vs 0.85 was against a weaker 52-feature set.  Caveat: pseudo-March history visibility is 0.63 vs 0.84
  on train / official windows, which hits a NaN-heavy raw patch harder than forward-filled columns.
  No kernel (no arm passed); nothing pushed, run on Kaggle or submitted.
"""
import fcntl, gc, json, os, time
import numpy as np
import pandas as pd
import lab
import occ_lib as oc

H, F, ONGOING = lab.H, lab.F, lab.ONGOING
PC = f'{lab.CACHE}/ogpatch'
RES = f'{lab.ROOT}/analysis/t2lab/results/ogpatch'
for _d in (PC, RES, f'{PC}/strip', f'{PC}/arms', f'{PC}/pm', f'{PC}/off'):
    os.makedirs(_d, exist_ok=True)
MINS = (40, 35, 30, 25, 20, 15, 10, 5, 0)            # minutes before T of the strip slots j = 0..8 (j = 8: slot T)
NS, RMAX = len(MINS), 10
FOLDS = (0, 1)
N_TREES = 500
EVAL_SETS = ('official', 'offlike', 'pseudo')
SIDE_SETS = ('faith3', 'runstart') + EVAL_SETS
LOCK = f'{PC}/build.lock'


def patch_cols(ch, R):
    return [f'{ch}{m:02d}_n{k:+d}' for m in MINS for k in range(-R, R + 1)]


ARM_COLS = {'occP9': patch_cols('pv', 4), 'occP21': patch_cols('pv', 10), 'occP9q': patch_cols('pv', 4) + patch_cols('pq', 4)}
ARMS = tuple(ARM_COLS)
ALLP = patch_cols('pv', RMAX) + patch_cols('pq', 4)  # union of the arms' columns (pm / off blocks are built once)
PAR_V = {'pv00_n+0': 'vr_T', 'pv05_n+0': 'vr_raw_last'}                                   # exact incl. NaN
PAR_F = dict({f'pv00_n{k:+d}': f'vr_n{k:+d}' for k in (-3, -2, -1, 1, 2, 3)},             # equal where the patch value is finite
             **{'pv10_n+0': 'vr_l1', 'pv15_n+0': 'vr_l2', 'pv20_n+0': 'vr_l3', 'pv35_n+0': 'vr_l6', 'pq05_n+0': 'fr_last'})


# ============================================================================================ kernel-portable block
def patch_strips(P, Vh, Fh, VT, FT):
    """speed / flow ratio strips [W, 9, L] (float32) of the slots T-40..T-5 (the last 8 slots of the 12-slot history Vh, Fh [W, 12, L];
    km/h, vph, NaN = missing) and slot T (VT, FT [W, L]; NaN = not visible).  pv = v / cut, pq = q / cap of the link the value belongs
    to (float64 division like the production ratios, stored as float32).  Uses nothing after T."""
    with np.errstate(all='ignore'):
        Sv = np.concatenate([Vh[:, H - (NS - 1):], VT[:, None]], 1) / P['cut'][None, None, :]
        Sq = np.concatenate([Fh[:, H - (NS - 1):], FT[:, None]], 1) / P['cap'][None, None, :]
    return Sv.astype(np.float32), Sq.astype(np.float32)


def pad_strip(S):
    """[W, 9, L] -> [W, 9, L + 2 RMAX]: NaN outside the corridor."""
    W, J, L = S.shape
    out = np.full((W, J, L + 2 * RMAX), np.nan, np.float32)
    out[:, :, RMAX:RMAX + L] = S
    return out


def col_index(cols):
    """(channel, strip slot j, link offset k) of every patch column name."""
    ch = np.array([c[:2] for c in cols]); j = np.array([MINS.index(int(c[2:4])) for c in cols], np.int64)
    k = np.array([int(c.split('_n')[1]) for c in cols], np.int64)
    return ch, j, k


def patch_rows(Svp, Sqp, wi, li, cols, chunk=65536):
    """patch columns `cols` [n, len(cols)] for rows (window wi, link li) from the PADDED strips: strip[wi, j, li + k] (link index + 1 =
    downstream; NaN outside the corridor or where missing)."""
    ch, j, k = col_index(cols)
    wi = np.asarray(wi, np.int64); li = np.asarray(li, np.int64)
    out = np.empty((len(wi), len(cols)), np.float32)
    for name, Sp in (('pv', Svp), ('pq', Sqp)):
        c = np.where(ch == name)[0]
        if not len(c):
            continue
        jj, kk = j[c][None, :], k[c][None, :] + RMAX
        for a in range(0, len(wi), chunk):
            out[a:a + chunk, c] = Sp[wi[a:a + chunk, None], jj, li[a:a + chunk, None] + kk]
    return out
# ============================================================================================ end of portable block


def parity(X, cols, Sv, Sq, wi, li):
    """INTEGRITY (i): patch values vs the production columns of the same rows (X [n, cols], rows (wi, li))."""
    names = list(PAR_V) + list(PAR_F)
    Pm = patch_rows(pad_strip(Sv), pad_strip(Sq), wi, li, names)
    out = {}
    for j, c in enumerate(names):
        ref = np.asarray(X[:, cols.index(PAR_V.get(c) or PAR_F[c])], np.float32); p = Pm[:, j]
        if c in PAR_V:
            out[f'par_{c}'] = bool(np.array_equal(p, ref, equal_nan=True))
        else:
            f = np.isfinite(p)
            out[f'par_{c}'] = bool(np.array_equal(p[f], ref[f]))
    out['par_ok'] = bool(all(out.values()))
    return out


# ------------------------------------------------------------------ train-split strips (window order / slot-T draw = occ / ogphys build_side)
def strip_path(panel, name):
    return f'{PC}/strip/{panel}__{oc.af_name(name)}__{ONGOING}.npz'


def build_strip(P, name):
    """[W, 9, L] strips of the allfin RowCache (panel, set, ongoing); slot T = the production draw of the FULL window list (rng(len(wins)),
    62 %), restricted afterwards for faith3 (exactly occ_lib.build_side / ogphys_lib.build_side) + INTEGRITY (i) vs the cached rows."""
    panel = P['panel']
    win = lab.window_set(panel, name); win = win[win.cond == ONGOING].reset_index(drop=True)
    rc = oc.rows_af(panel, name, ONGOING)
    st = dict(panel=panel, set=oc.af_name(name)); L = len(P['links'])
    if not len(win):
        z = np.zeros((0, NS, L), np.float32); np.savez(strip_path(panel, name), Sv=z, Sq=z)
        return dict(st, windows=0, rows=0, par_ok=True)
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
    Sv, Sq = patch_strips(P, Vh, Fh, VT, FT)
    st.update(windows=len(d), rows=int(rc.n), T_vis=float(np.isfinite(Sv[:, -1]).mean()), hist_vis=float(np.isfinite(Sv[:, :-1]).mean()),
              q_vis=float(np.isfinite(Sq).mean()))
    if rc.n:
        st.update(parity(rc.X, rc.cols, Sv, Sq, rc.wi, rc.li))
    else:
        st['par_ok'] = True
    tmp = f'{strip_path(panel, name)}.{os.getpid()}.tmp.npz'
    np.savez(tmp, Sv=Sv, Sq=Sq); os.replace(tmp, strip_path(panel, name))
    return st


def ensure_strips(panels=None):
    out = []
    for panel in panels or lab.PANELS:
        jf = f'{PC}/strip/{panel}.json'
        if all(os.path.exists(strip_path(panel, n)) for n in SIDE_SETS) and os.path.exists(jf):
            out += json.load(open(jf)); continue
        P = lab.load_panel(panel, cache=False); oc.load_cubes(P)
        sts = []
        for n in SIDE_SETS:
            t1 = time.time(); st = build_strip(P, n); st['secs'] = round(time.time() - t1, 1); sts.append(st)
            lab.log(f'ogpatch strip {panel} {n}: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}))
        json.dump(sts, open(jf, 'w')); out += sts
        del P; gc.collect()
    return pd.DataFrame(out)


class Strip:
    def __init__(self, panel, name):
        z = np.load(strip_path(panel, name))
        self.Svp = pad_strip(z['Sv']); self.Sqp = pad_strip(z['Sq'])

    def rows(self, wi, li, cols):
        return patch_rows(self.Svp, self.Sqp, wi, li, cols)


# ------------------------------------------------------------------ training / prediction (rows and order = occ_lib / ogphys train_matrix)
def train_matrix(k, cols, pcols):
    """rows / order exactly as occ_lib.train_matrix (panel outer, set inner; e & keep5[:, k] & fold != k & weight > 0); columns = cols +
    OCC_COLS (fold-k calibration) + pcols."""
    parts = []; n = 0
    for panel in lab.PANELS:
        for name, sw in oc.TRAIN:
            rc = oc.rows_af(panel, name, ONGOING)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32)
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((panel, name, rc, m, ww, c)); n += c
    nb = len(cols) + len(oc.OCC_COLS)
    X = np.empty((n, nb + len(pcols)), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for panel, name, rc, m, ww, c in parts:
        X[o:o + c, :nb] = oc.matrix(rc, oc.Side(panel, name, ONGOING), m, cols, k, True)
        if pcols:
            r = np.where(m)[0]
            X[o:o + c, nb:] = Strip(panel, name).rows(rc.wi[r], rc.li[r], pcols)
        y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w


def params():
    return lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=0)


def build_dataset(k, cols, pcols, feat):
    """matrix built under the file lock, LightGBM Dataset constructed, float matrix freed (INTEGRITY (ii) checks bit-identity with
    exp_firstcross_lib.fit).  Returns (Dataset, n rows, positive rate)."""
    import lightgbm as lgb
    with open(LOCK, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t1 = time.time()
        X, y, w = train_matrix(k, cols, pcols)
        assert X.shape[1] == len(feat)
        n, pos = len(y), float(y.mean())
        ds = lgb.Dataset(X, y, weight=None if np.all(w == 1) else w, feature_name=list(feat), free_raw_data=True, params=params())
        ds.construct()
        del X; gc.collect()
        lab.log(f'matrix + Dataset {n} x {len(feat)} built in {time.time() - t1:.0f}s (lock released)')
    return ds, n, pos


def eval_matrix(rc, panel, name, m, cols, k, pcols):
    Xb = oc.matrix(rc, oc.Side(panel, name, ONGOING), m, cols, k, True)
    if not pcols:
        return Xb
    r = np.where(m)[0]
    return np.hstack([Xb, Strip(panel, name).rows(rc.wi[r], rc.li[r], pcols)])


def arm_dir(arm):
    return f'{PC}/arms/{arm}'
