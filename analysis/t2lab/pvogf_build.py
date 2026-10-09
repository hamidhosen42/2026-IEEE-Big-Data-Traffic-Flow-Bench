"""pvogf BUILD (run through run_heavy): row caches of the pvogf lab (pvogf_lib.py holds the design and the PRE-REGISTERED rule).
Per panel, in data/t2local/cache/pvogf/:
  f03_<P>.npy / .npz        F03 lab training rows of all train days (k = 4): kernel make_rows (all-finite inputs, production slot-T
                            simulation, ogfeat columns), all-train prior, production keep region; y, w (faith3 rest 0.3 / run-starts 1.0)
  march_<P>.npy, marchb_<P>.npy, march_<P>.npz
                            faithful windows of the masked March month, ALL days (run-starts 1.0, t%3 rest 0.3; x WM at training),
                            built as the kernel's March rows + ogfeat; marchb = full matrix with the blended same-scenario prior
                            (March days < d); y, w, d (training subsets are cut by day later)
  ev_<P>__<set>.npz         pseudo-March evaluation windows of March days >= CUT (offlike_m, runstart_m): X, Xb, meta, truth, ev, qlast, d, t
  priv_<P>.npz              the 10 official private windows: X (ongoing rows, keep region), Xb (prior blended over March + April days < d),
                            meta, qlast, window ids, condition, origin hour
  cols.json                 kernel column order (69 pvaf columns + 103 ogfeat columns)
Checks: base columns == lab.all_cols(); kernel March candidates == pvog_lib.select; ogfeat state at T (c_n0) == qlast count.
Usage: pvogf_build.py [panels=P1,P2] [refresh=0]"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvog_lib as pv
import pvogf_lib as G

kw = dict(a.split('=') for a in sys.argv[1:])
PANELS = kw['panels'].split(',') if 'panels' in kw else lab.PANELS
REFRESH = bool(int(kw.get('refresh', 0)))
H, F, ONGOING, C = G.H, G.F, G.ONGOING, G.CACHE
T0 = time.time()
cols = json.load(open(f'{C}/cols.json')) if os.path.exists(f'{C}/cols.json') else None


def check_cn0(X, meta, qlast, cols, tag):
    j = cols.index('c_n0')
    ok = np.array_equal(X[:, j], qlast.sum(1).astype(np.float32)[meta[:, 0]])
    assert ok, f'{tag}: ogfeat c_n0 != qlast count'


def atomic_save_npy(path, a):
    tmp = f'{path}.tmp{os.getpid()}.npy'
    np.save(tmp, a); os.replace(tmp, path)


def atomic_savez(path, **kw_):
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, **kw_); os.replace(tmp, path)


stats = []
for panel in PANELS:
    if not REFRESH and os.path.exists(f'{C}/priv_{panel}.npz'):
        lab.log(panel, 'cached, skipped'); continue
    t1 = time.time()
    PK = G.panel_pk(panel)
    st = dict(panel=panel)
    # ---- F03 training rows (train split, all days)
    Xs, ys, ws, ss = [], [], [], []
    for si, (name, wins, ww) in enumerate(G.f03_lab_windows(panel, k=4)):
        X, y, e, meta, cols_, R = G.train_rows(PK, wins, PK['prior_all'], cols)
        if cols is None:
            cols = list(cols_)
            base = [c for c in cols if c not in G.NEW]
            assert base == lab.all_cols(), 'base columns differ from the lab production columns'
            assert [c for c in cols if c in G.NEW] == [c for c in cols if c in set(G.NEW)] and len(cols) == 69 + 103
            json.dump(cols, open(f'{C}/cols.json', 'w'))
        assert list(cols_) == cols
        check_cn0(X, meta, R['qlast'], cols, f'{panel} {name}')
        Xs.append(X[e]); ys.append(y[e].astype(np.float32)); ws.append(ww[meta[:, 0]][e].astype(np.float32)); ss.append(np.full(int(e.sum()), si, np.int8))
        st[f'f03_{name}_windows'] = len(wins); st[f'f03_{name}_rows'] = int(e.sum())
        del X, R; gc.collect()
    X = np.concatenate(Xs); del Xs; gc.collect()
    atomic_save_npy(f'{C}/f03_{panel}.npy', X)
    atomic_savez(f'{C}/f03_{panel}.npz', y=np.concatenate(ys), w=np.concatenate(ws), s=np.concatenate(ss))
    st['f03_rows'] = len(X); st['f03_pos'] = float(np.concatenate(ys).mean()); del X; gc.collect()
    # ---- masked March month: kernel scenario (March + April cumulative prior tables) and faithful candidates
    M, dpos, cQ, cE = G.K2.scenario(PK)
    cand, Qm, Emv = G.K2.march_faithful(M, PK['cut'])
    Pv = dict(V=M['V'], E=np.isfinite(M['V']), Q=Qm, Em=M['Em'], cut=PK['cut'])
    on_ref, og_ref = pv.select(Pv)
    assert np.array_equal(og_ref, cand), f'{panel}: kernel March candidates differ from pvog_lib.select'
    on = G.onset_map(M, Qm)
    assert np.array_equal(on, on_ref), f'{panel}: onset map differs'
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    wl = sorted([(int(a), int(b), G.K2.W_RS) for a, b in zip(*np.where(rs))] + [(int(a), int(b), G.K2.W_DENSE) for a, b in zip(*np.where(rest))])
    d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl]); ww = np.array([w[2] for w in wl], np.float32)
    R, X, y, e, meta = G.masked_rows(PK, M, Qm, Emv, d, t, cols)
    check_cn0(X, meta, R['qlast'], cols, f'{panel} March')
    jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
    Xb, a0 = G.K2.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, cols)
    atomic_save_npy(f'{C}/march_{panel}.npy', X[e]); atomic_save_npy(f'{C}/marchb_{panel}.npy', Xb[e])
    atomic_savez(f'{C}/march_{panel}.npz', y=y[e].astype(np.float32), w=ww[meta[:, 0]][e], d=d[meta[:, 0]][e].astype(np.int16))
    st.update(march_windows=len(wl), march_rs=int(rs.sum()), march_rows=int(e.sum()), march_pos=float(y[e].mean()), march_mean_a=round(a0, 3),
              march_rows_lt_cut=int((d[meta[:, 0]][e] < G.CUT).sum()))
    del R, X, Xb; gc.collect()
    # ---- pseudo-March evaluation windows of days >= CUT
    sets = {'offlike_m': [w for w in G.offlike_m(on, cand) if w[0] >= G.CUT],
            'runstart_m': sorted((int(a), int(b)) for a, b in zip(*np.where(rs)) if a >= G.CUT)}
    for sname, wl2 in sets.items():
        d = np.array([w[0] for w in wl2], int); t = np.array([w[1] for w in wl2], int)
        R, X, y, e, meta = G.masked_rows(PK, M, Qm, Emv, d, t, cols)
        jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
        Xb, _ = G.K2.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, cols)
        truth = np.stack([Qm[d, t + h] for h in range(1, F + 1)], 1); ev = np.stack([Emv[d, t + h] for h in range(1, F + 1)], 1)
        atomic_savez(f'{C}/ev_{panel}__{sname}.npz', X=X, Xb=Xb, meta=meta.astype(np.int32), truth=truth, ev=ev, qlast=R['qlast'], d=d, t=t)
        st[f'{sname}_n'] = len(wl2)
        del R, X, Xb; gc.collect()
    # ---- official private windows (April): blended prior over all March + April days < d
    wi, R, X, meta, org = G.split_rows(PK, panel, 'private', cols)
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
    jd = np.array([dpos[pd.Timestamp(x).strftime('%Y-%m-%d')] for x in org.to_numpy()])
    Xb, a0 = G.K2.blend_prior_og(X, meta, PK, jd, (dow >= 5).astype(int), tod, cQ, cE, cols)
    check_cn0(X, meta, R['qlast'], cols, f'{panel} private')
    atomic_savez(f'{C}/priv_{panel}.npz', X=X, Xb=Xb, meta=meta.astype(np.int32), qlast=R['qlast'], window_id=wi.window_id.to_numpy().astype(str),
                 cond=wi.condition.to_numpy().astype(str), tod=(org.dt.hour + org.dt.minute / 60).to_numpy(), jd=jd)
    st.update(priv_windows=len(wi), priv_ongoing=int((wi.condition == ONGOING).sum()), priv_rows=len(X), priv_mean_a=round(a0, 3),
              priv_jd_min=int(jd[(wi.condition == ONGOING).to_numpy()].min()), secs=round(time.time() - t1, 1))
    stats.append(st)
    lab.log(json.dumps(st))
    del R, X, Xb, M, cQ, cE, PK; lab.drop_panel(panel); gc.collect()
if stats:
    f = f'{G.RES}/build_stats.csv'
    old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    pd.concat([old[~old.panel.isin([s['panel'] for s in stats])] if len(old) else old, pd.DataFrame(stats)], ignore_index=True).to_csv(f, index=False)
lab.log('PVOGF BUILD DONE', f'{time.time() - T0:.0f}s')
