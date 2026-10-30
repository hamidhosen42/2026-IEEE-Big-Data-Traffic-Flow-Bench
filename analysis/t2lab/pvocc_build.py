"""pvocc BUILD (run through run_heavy): occupancy SIDE-CARS aligned with the pvogf lab caches (design + PRE-REGISTERED rule: pvocc_lib.py).
Every block is rebuilt with the kernel row builders and occupancy on; its 172 non-occupancy columns must equal the cached pvogf rows
bit-for-bit (INTEGRITY (ii)); only the 13 occupancy columns are stored:
  data/t2local/cache/pvocc/side/<P>.npz      f03 (F03 rows, all train days), march (all March days, = pvogf march_<P> rows),
                                             ev__offlike_m / ev__runstart_m (pseudo-March days >= 15), priv (official private rows)
  data/t2local/cache/pvocc/oof0/side_<P>.npz (mode oof) fold-0 rows of pvogf_oof.py: f03, march, ev__official / ev__offlike / ev__runstart
                                             (fold-0 prior, calibration on the train days outside fold 0)
  results/pvocc/build_<mode>.json            parity flags and counts per panel
INTEGRITY (iii) on the first two panels (mode md): kernel occ_calib == occ_lib.calibrate (k = 4); kernel occ_features == occ_lib raw_occ +
occ_feats on the pseudo-March windows (same calibration).
Usage: pvocc_build.py [mode=md|oof] [panels=P1,P2] [refresh=0]"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G
import pvocc_lib as V

kw = dict(a.split('=') for a in sys.argv[1:])
MODE = kw.get('mode', 'md')
PANELS = kw['panels'].split(',') if 'panels' in kw else lab.PANELS
REFRESH = bool(int(kw.get('refresh', 0)))
H, F, ONGOING = V.H, V.F, V.ONGOING
GC = V.GC
T0 = time.time()
cols = G.load_cols(); assert len(cols) == 172
V.check_prereg()


def chunk_same(A, B, n=400000):
    """bit-identity (NaN == NaN) of two [n, c] arrays (B may be a memmap), in chunks."""
    if A.shape != B.shape:
        return False
    for s in range(0, len(A), n):
        if not np.array_equal(A[s:s + n], np.asarray(B[s:s + n]), equal_nan=True):
            return False
    return True


def atomic_savez(path, **kw_):
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, **kw_); os.replace(tmp, path)


def march_windows(M, PK):
    cand, Qm, Emv = G.K2.march_faithful(M, PK['cut'])
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    wl = sorted([(int(a), int(b), G.K2.W_RS) for a, b in zip(*np.where(rs))] + [(int(a), int(b), G.K2.W_DENSE) for a, b in zip(*np.where(rest))])
    d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl])
    return cand, Qm, Emv, rs, d, t


def integrity_iii(PK, M, Om, d, t):
    """kernel occ_calib / occ_features vs the EXP_T2_OCC lab code (occ_lib), on this panel."""
    import occ_lib as oc
    P = lab.load_panel(PK['panel'], cache=False)
    oc5, c5 = oc.calibrate(P)                                          # cached calibration of the occ lab (read only)
    st = dict(calib_ocrit_equal=bool(np.array_equal(oc5[4], PK['ocrit'])), calib_c_equal=bool(np.array_equal(c5[4], PK['occ_c'])),
              calib_ocrit_maxdiff=float(np.max(np.abs(oc5[4].astype(np.float64) - PK['ocrit']))),
              calib_c_maxdiff=float(np.max(np.abs(c5[4].astype(np.float64) - PK['occ_c']))))
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = M['Vf'][d[:, None], idx]; Fh = M['Ff'][d[:, None], idx]; VT = M['Vf'][d, t]
    Oh = Om[d[:, None], idx]; OT = np.where(np.isfinite(VT), Om[d, t], np.nan).astype(np.float32)
    lab_cube = oc.occ_feats(oc.raw_occ(Oh, OT, Vh, Fh, PK['lanes']), PK['ocrit'], PK['occ_c'])
    st['features_equal'] = bool(V.same(lab_cube, V.occ_cube(PK, Oh, OT, Vh, Fh)))
    del P; gc.collect()
    return st


stats = []
for pi, panel in enumerate(PANELS):
    out_f = V.side_path(panel, 'md' if MODE == 'md' else 'oof')
    if not REFRESH and os.path.exists(out_f):
        lab.log(panel, 'side-car cached, skipped'); continue
    t1 = time.time(); st = dict(panel=panel, mode=MODE); side = {}
    if MODE == 'md':
        PK = V.panel_pko(panel); prior = PK['prior_all']; k = 4; od = GC
    else:
        dm = lab.load_panel(panel)['block'] != 0
        PK = V.panel_pko(panel, day_mask=dm); k = 0; od = f'{GC}/oof0'
        prior = G.K.prior_table(PK, PK['block'] != 0); PK['prior_all'] = prior            # = pvogf_oof.py (fold-0 train prior everywhere)
    st.update(n_est=int(PK['n_est']), ocrit_q50=float(np.median(PK['ocrit'])))
    # ---- F03 training rows (kernel make_rows with occupancy)
    Xb_, Xo_, ys = [], [], []
    for name, wins, ww in G.f03_lab_windows(panel, k=k):
        X, y, e, meta, R = V.train_rows_occ(PK, wins, prior, cols)
        Xb_.append(X[e][:, :172]); Xo_.append(X[e][:, 172:]); ys.append(y[e].astype(np.float32))
        del X, R; gc.collect()
    Xb_ = np.concatenate(Xb_); side['f03'] = np.concatenate(Xo_); del Xo_
    ref = np.load(f'{od}/f03_{panel}.npy', mmap_mode='r'); yref = np.load(f'{od}/f03_{panel}.npz')['y']
    st['par_f03'] = bool(chunk_same(Xb_, ref) and np.array_equal(np.concatenate(ys), yref)); st['f03_rows'] = len(Xb_)
    del Xb_, ref; gc.collect()
    # ---- released March month (all days): faithful training windows, kernel scenario / march_faithful / pvogf masked_rows
    M, dpos, cQ, cE = G.K2.scenario(PK)
    Om = V.month_occ(panel, 'validation', M)
    cand, Qm, Emv, rs, d, t = march_windows(M, PK)
    R, X, y, e, meta = G.masked_rows(PK, M, Qm, Emv, d, t, cols)
    jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
    Xb, _ = G.K2.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, cols)
    st['par_march'] = bool(V.same(X[e], np.load(f'{od}/march_{panel}.npy')) and V.same(Xb[e], np.load(f'{od}/marchb_{panel}.npy')))
    oc_ = V.masked_occ(PK, M, Om, d, t); side['march'] = oc_[meta[e, 0], meta[e, 1]]; st['march_rows'] = int(e.sum())
    if MODE == 'md':
        zm = np.load(f'{od}/march_{panel}.npz'); st['par_march'] &= bool(np.array_equal(zm['d'], d[meta[:, 0]][e].astype(np.int16)))
    del R, X, Xb; gc.collect()
    if MODE == 'md':
        # ---- pseudo-March evaluation windows (days >= CUT)
        on = G.onset_map(M, Qm)
        sets = {'offlike_m': [w for w in G.offlike_m(on, cand) if w[0] >= G.CUT],
                'runstart_m': sorted((int(a), int(b)) for a, b in zip(*np.where(rs)) if a >= G.CUT)}
        for sname, wl2 in sets.items():
            d2 = np.array([w[0] for w in wl2], int); t2 = np.array([w[1] for w in wl2], int)
            R, X, y, e, meta = G.masked_rows(PK, M, Qm, Emv, d2, t2, cols)
            jd = np.array([dpos[M['days'][x]] for x in d2]); cls = (M['dow'][d2] >= 5).astype(int)
            Xb, _ = G.K2.blend_prior_og(X, meta, PK, jd, cls, t2, cQ, cE, cols)
            z = np.load(f'{od}/ev_{panel}__{sname}.npz')
            st[f'par_ev_{sname}'] = bool(V.same(X, z['X']) and V.same(Xb, z['Xb']) and np.array_equal(meta.astype(np.int32), z['meta'])
                                         and np.array_equal(d2, z['d']) and np.array_equal(t2, z['t']) and np.array_equal(R['qlast'], z['qlast']))
            oc_ = V.masked_occ(PK, M, Om, d2, t2); side[f'ev__{sname}'] = oc_[meta[:, 0], meta[:, 1]]; st[f'{sname}_n'] = len(wl2)
            if pi < 2 and sname == 'runstart_m':
                st.update(integrity_iii(PK, M, Om, d2, t2))
            del R, X, Xb; gc.collect()
        # ---- official private windows (April): blended prior over March + April days < d
        wi, R, X, meta, org = G.split_rows(PK, panel, 'private', cols)
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
        jd = np.array([dpos[pd.Timestamp(x).strftime('%Y-%m-%d')] for x in org.to_numpy()])
        Xb, _ = G.K2.blend_prior_og(X, meta, PK, jd, (dow >= 5).astype(int), tod, cQ, cE, cols)
        z = np.load(f'{od}/priv_{panel}.npz')
        st['par_priv'] = bool(V.same(X, z['X']) and V.same(Xb, z['Xb']) and np.array_equal(meta.astype(np.int32), z['meta'])
                              and np.array_equal(R['qlast'], z['qlast']) and wi.window_id.tolist() == [str(a) for a in z['window_id']])
        oc_ = V.split_occ(PK, panel, 'private'); side['priv'] = oc_[meta[:, 0], meta[:, 1]]; st['priv_rows'] = len(X)
        st['priv_occ_finite_or_last'] = float(np.isfinite(side['priv'][:, V.OCC_COLS.index('or_last')]).mean())
        del R, X, Xb
    else:
        # ---- fold-0 train evaluation windows (pvogf_oof.py EVS)
        for s in ('official', 'offlike', 'runstart'):
            f = f'{od}/ev_{panel}__{s}.npz'
            if not os.path.exists(f):
                continue
            win = lab.window_set(panel, s); win = win[win.cond == ONGOING].reset_index(drop=True)
            widx = np.where(win.fold.to_numpy() == 0)[0]
            wins = list(zip(win.d.to_numpy()[widx].tolist(), win.t.to_numpy()[widx].tolist(), [ONGOING] * len(widx)))
            X, y, e, meta, R = V.train_rows_occ(PK, wins, prior, cols)
            z = np.load(f)
            st[f'par_ev_{s}'] = bool(V.same(X[:, :172], z['X']) and np.array_equal(meta.astype(np.int32), z['meta']) and np.array_equal(R['qlast'], z['qlast']))
            side[f'ev__{s}'] = X[:, 172:]; st[f'{s}_n'] = len(wins)
            del X, R
    st['par_ok'] = bool(all(v for kk, v in st.items() if kk.startswith('par_')))
    st['occ_finite_f03_or_last'] = float(np.isfinite(side['f03'][:, V.OCC_COLS.index('or_last')]).mean())
    st['secs'] = round(time.time() - t1, 1)
    assert st['par_ok'], f'{panel}: row parity FAILED {st}'
    atomic_savez(out_f, **side)
    stats.append(st)
    lab.log(json.dumps(st))
    del M, cQ, cE, PK, Om, side; lab.drop_panel(panel); gc.collect()
if stats:
    f = f'{V.RES}/build_{MODE}.json'
    old = json.load(open(f)) if os.path.exists(f) else []
    old = [o for o in old if o['panel'] not in {s['panel'] for s in stats}] + stats
    json.dump(old, open(f, 'w'), indent=1)
lab.log('PVOCC BUILD DONE', MODE, f'{time.time() - T0:.0f}s')
