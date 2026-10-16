"""extcal BUILD (run through run_heavy; design + PRE-REGISTERED rule in extcal_lib.py).  Per panel, in data/t2local/cache/extcal/:
  fit_<P>.npz   fitting windows of ALL 31 March days = faithful March windows (run-starts 1.0 / every-3rd rest 0.3) U offlike_m:
                X (keep-region rows, kernel March rows + ogfeat), Xb (blended same-scenario prior), meta, truth, ev, qlast, d, t,
                wt (faithful weight, 0 for offlike-only windows), off (in offlike_m), size features SZ [W, 6, len(SIZE_COLS)]
  szf03_<P>.npz size-model rows of the F03 lab windows (train split): SZ [W, 6, k], target n [W, 6], weight w [W]
  szev_<P>__<set>.npz / szpriv_<P>.npz   size features of the pvogf evaluation windows (days >= 15) and of the private windows
Usage: extcal_build.py [panels=P1,P2]"""
import gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G
import extcal_lib as E

kw = dict(a.split('=') for a in sys.argv[1:])
PANELS = kw['panels'].split(',') if 'panels' in kw else lab.PANELS
F, ONGOING = E.F, E.ONGOING
cols = G.load_cols(); CI = {c: i for i, c in enumerate(cols)}
T0 = time.time()


def size_feats(X, Xb, meta, W, t, weekend, pid):
    """window-level size features [W, 6, k] from keep-region rows (X: c_* columns; Xb: blended prior)."""
    SZ = np.full((W, F, len(E.SIZE_COLS)), np.nan, np.float32)
    SZ[:, :, 0] = np.arange(1, F + 1)[None]; SZ[:, :, 1] = pid; SZ[:, :, 2] = np.asarray(t)[:, None]; SZ[:, :, 3] = np.asarray(weekend)[:, None]
    if len(meta):
        w0 = meta[:, 0]; h = meta[:, 2]
        first = np.full(W, -1); r1 = np.where(h == 1)[0]
        first[w0[r1][::-1]] = r1[::-1]                                    # first h == 1 row of each window
        ok = first >= 0
        for j, c in enumerate(E.SIZE_STATIC):
            SZ[ok, :, 4 + j] = X[first[ok], CI[c]][:, None]
        nk = np.bincount(w0[r1], minlength=W).astype(np.float32); SZ[:, :, 4 + len(E.SIZE_STATIC)] = nk[:, None]
        for hh in range(1, F + 1):
            r = np.where(h == hh)[0]
            SZ[:, hh - 1, -2] = np.bincount(w0[r], weights=np.nan_to_num(Xb[r, CI['prior']]), minlength=W)
        SZ[:, :, -1] = np.bincount(w0[r1], weights=np.nan_to_num(Xb[r1, CI['prior_now']]), minlength=W)[:, None]
    else:
        SZ[:, :, 4 + len(E.SIZE_STATIC):] = 0
    return SZ


for panel in PANELS:
    if os.path.exists(f'{E.CACHE}/szpriv_{panel}.npz') and 'nvis' in np.load(f'{E.CACHE}/szf03_{panel}.npz').files:
        lab.log(panel, 'cached, skipped'); continue
    t1 = time.time(); pid = lab.PANELS.index(panel)
    PK = G.panel_pk(panel)
    # ---------------- size-model rows of the F03 lab windows (train split; train prior, lab weights)
    SZs, Ns, Ws, NVs = [], [], [], []
    for name, wins, ww in G.f03_lab_windows(panel, k=4):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            R = G.K.make_rows(PK, wins, PK['prior_all'])
        keep = G.K.ongoing_keep(R); f1 = R['rows'][0][1]; W = R['W']
        SZ = np.full((W, F, len(E.SIZE_COLS)), np.nan, np.float32)
        SZ[:, :, 0] = np.arange(1, F + 1)[None]; SZ[:, :, 1] = pid; SZ[:, :, 2] = R['t'][:, None]
        SZ[:, :, 3] = (PK['dow'][R['d']] >= 5)[:, None]
        for j, c in enumerate(E.SIZE_STATIC):
            SZ[:, :, 4 + j] = np.asarray(f1[c])[:, 0][:, None]
        SZ[:, :, 4 + len(E.SIZE_STATIC)] = keep.sum(1)[:, None]
        n = np.zeros((W, F), np.float32); nv = np.zeros((W, F), np.float32)
        for hh, feats, labl, ev in R['rows']:
            SZ[:, hh - 1, -2] = np.where(keep, np.nan_to_num(feats['prior']), 0).sum(1)
            n[:, hh - 1] = (labl & ev).sum(1); nv[:, hh - 1] = ev.sum(1)
        SZ[:, :, -1] = np.where(keep, np.nan_to_num(f1['prior_now']), 0).sum(1)[:, None]
        # check: the c_n0 column equals the qlast count (as in pvogf_build)
        assert np.array_equal(SZ[:, 0, 4 + E.SIZE_STATIC.index('c_n0')], R['qlast'].sum(1).astype(np.float32))
        SZs.append(SZ); Ns.append(n); NVs.append(nv); Ws.append(np.asarray(ww, np.float32)); del R; gc.collect()
    np.savez(f'{E.CACHE}/szf03_{panel}.npz', SZ=np.concatenate(SZs), n=np.concatenate(Ns), w=np.concatenate(Ws), nvis=np.concatenate(NVs))
    if kw.get('only') == 'szf03':                      # b2 amendment: refresh the F03 size rows only (fit / ev / priv caches unchanged)
        lab.log(panel, 'szf03 refreshed'); lab.drop_panel(panel); continue
    # ---------------- masked March month: fitting windows of all days
    M, dpos, cQ, cE = G.K2.scenario(PK)
    cand, Qm, Emv = G.K2.march_faithful(M, PK['cut'])
    on = G.onset_map(M, Qm)
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    wt = {}
    for a, b in zip(*np.where(rs)):
        wt[(int(a), int(b))] = 1.0
    for a, b in zip(*np.where(rest)):
        wt[(int(a), int(b))] = 0.3
    off = set((int(a), int(b)) for a, b in G.offlike_m(on, cand))
    wl = sorted(set(wt) | off)
    d = np.array([w[0] for w in wl], int); t = np.array([w[1] for w in wl], int)
    R, X, y, e, meta = G.masked_rows(PK, M, Qm, Emv, d, t, cols)
    jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
    Xb, _ = G.K2.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, cols)
    truth = np.stack([Qm[d, t + h] for h in range(1, F + 1)], 1); ev = np.stack([Emv[d, t + h] for h in range(1, F + 1)], 1)
    W = len(wl)
    SZ = size_feats(X, Xb, meta, W, t, M['dow'][d] >= 5, pid)
    assert np.array_equal(SZ[:, 0, 4 + E.SIZE_STATIC.index('c_n0')][SZ[:, 0, 4 + len(E.SIZE_STATIC)] > 0],
                          R['qlast'].sum(1).astype(np.float32)[SZ[:, 0, 4 + len(E.SIZE_STATIC)] > 0])
    np.savez(f'{E.CACHE}/fit_{panel}.npz', X=X, Xb=Xb, meta=meta.astype(np.int32), truth=truth, ev=ev, qlast=R['qlast'], d=d, t=t,
             wt=np.array([wt.get(w, 0.0) for w in wl], np.float32), off=np.array([w in off for w in wl]), SZ=SZ, n=(truth & ev).sum(2).astype(np.float32))
    st = dict(panel=panel, fit_windows=W, faithful=len(wt), offlike_all=len(off), off_only=len(off - set(wt)), rows=len(X),
              lt15=int((d < E.CUT).sum()), f03_windows=int(sum(len(s) for s in SZs)))
    del R, X, Xb; gc.collect()
    # ---------------- size features of the pvogf evaluation windows (days >= 15) and of the private windows
    for s in G.PM_SETS:
        z = np.load(f'{G.CACHE}/ev_{panel}__{s}.npz')
        dd = z['d']; wk = (M['dow'][dd] >= 5)
        # the pvogf ev windows are exactly offlike_m / run-starts of days >= 15
        SZe = size_feats(z['X'], z['Xb'], z['meta'], len(dd), z['t'], wk, pid)
        np.savez(f'{E.CACHE}/szev_{panel}__{s}.npz', SZ=SZe, n=(z['truth'] & z['ev']).sum(2).astype(np.float32))
    z = np.load(f'{G.CACHE}/priv_{panel}.npz')
    wi = lab.window_index(panel, 'private'); org = pd.to_datetime(wi.forecast_origin, utc=True)
    assert wi.window_id.tolist() == [str(x) for x in z['window_id']]
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); wk = (org.dt.dayofweek.to_numpy() >= 5)
    np.savez(f'{E.CACHE}/szpriv_{panel}.npz', SZ=size_feats(z['X'], z['Xb'], z['meta'], len(wi), tod, wk, pid))
    st['secs'] = round(time.time() - t1, 1); lab.log(json.dumps(st))
    del M, cQ, cE, PK; lab.drop_panel(panel); gc.collect()
lab.log('EXTCAL BUILD DONE', f'{time.time() - T0:.0f}s')
