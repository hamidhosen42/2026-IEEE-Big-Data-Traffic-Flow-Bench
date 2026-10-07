"""verify_occ: INDEPENDENT occupancy feature builder (adversarial review of track occ).  Not importing occ_lib.
Builds the 13 features straight from the cubes: history slots T-12..T-1 only, slot T visible only where the production slot-T
speed draw is visible (eligible: rng(len(full ongoing list)) < 0.62; ineligible-valued: always).  Own calibration code."""
import os, json, warnings
import numpy as np
import pandas as pd
import lab

H = lab.H
XDIR = f'{lab.DATA}/x'
ODIR = f'{lab.DATA}/occ'
AF = f'{lab.CACHE}/allfin'
VOUT = f'{lab.CACHE}/verify_occ'
os.makedirs(VOUT, exist_ok=True)
NB = (-3, -2, -1, 1, 2, 3)
NAMES = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in NB] + ['dens_dev']
_CUBES = {}


def cubes(panel):
    if panel in _CUBES:
        return _CUBES[panel]
    P = lab.load_panel(panel, cache=False)
    x = np.load(f'{XDIR}/t2x_{panel}.npz', allow_pickle=False)
    o = np.load(f'{ODIR}/t2occ_{panel}.npz', allow_pickle=False)
    assert [str(a) for a in x['links']] == P['links'] == [str(a) for a in o['links']]
    assert [str(a) for a in x['train_days']] == P['days'] == [str(a) for a in o['train_days']]
    Vx = x['train_Vx'].astype(np.float32); Fx = x['train_Fx'].astype(np.float32)
    E = P['E']
    C = dict(P=P, E=E, V=P['V'], Vx=Vx, Vall=np.where(E, P['V'], Vx), Fall=np.where(E, P['Fq'], Fx),
             O=o['train_O'].astype(np.float32))
    _CUBES.clear(); _CUBES[panel] = C
    return C


def my_calib(C, k):
    """ocrit (best misclassification threshold of queued by o > thr) and c (free-flow median log density ratio); k<4 drops block k."""
    P = C['P']; L = C['O'].shape[2]
    dm = (P['block'] != k) if k < 4 else np.ones(len(P['days']), bool)
    oc = np.full(L, np.nan); cc = np.full(L, np.nan); est = np.zeros(L, bool)
    for l in range(L):
        v = C['Vall'][dm, :, l].ravel(); o = C['O'][dm, :, l].ravel(); q = C['Fall'][dm, :, l].ravel()
        m = np.isfinite(v) & np.isfinite(o)
        v, o, q = v[m], o[m], q[m]
        qd = v <= P['cut'][l]
        nq, nf = int(qd.sum()), int((~qd).sum())
        if nq >= 20 and nf >= 20:
            s = np.argsort(o, kind='mergesort'); os_, qs = o[s], qd[s]
            # threshold between position i and i+1: cells 0..i called free, i+1.. called queued
            fn = np.cumsum(qs)                       # queued called free
            fp = nf - np.cumsum(~qs)                 # free called queued
            i = int(np.argmin(fn + fp))
            oc[l] = 0.5 * (os_[i] + os_[min(i + 1, len(os_) - 1)]); est[l] = True
        ff = (v > 0.85 * P['vf'][l]) & (q > 0) & (o > 0)
        if ff.sum() >= 100:
            cc[l] = np.median(np.log(o[ff] * P['lanes'][l] * v[ff] / q[ff]))
    oc[~est] = np.nanmedian(oc[est]); cc[~np.isfinite(cc)] = np.nanmedian(cc)
    return oc.astype(np.float32), cc.astype(np.float32)


def calib_all(panel):
    f = f'{VOUT}/calib_{panel}.npz'
    if os.path.exists(f):
        z = np.load(f); return z['oc'], z['cc']
    C = cubes(panel); r = [my_calib(C, k) for k in range(5)]
    oc = np.stack([a for a, _ in r]); cc = np.stack([b for _, b in r])
    tmp = f'{f}.{os.getpid()}.tmp.npz'; np.savez(tmp, oc=oc, cc=cc); os.replace(tmp, f)
    return oc, cc


def slotT_visible(C, name, cond, d, t):
    """production slot-T draw of the FULL (set, cond) window list, rows matched by (d, t)."""
    full = lab.window_set(C['P']['panel'], 'faith3' if name == 'faith3F03' else name)
    full = full[full.cond == cond].reset_index(drop=True)
    L = C['O'].shape[2]
    draw = np.random.default_rng(len(full)).random((len(full), L)) < 0.62
    pos = {(a, b): i for i, (a, b) in enumerate(zip(full.d.to_numpy(), full.t.to_numpy()))}
    ix = np.array([pos[(a, b)] for a, b in zip(d, t)], int)
    Et = C['E'][d, t]
    return np.where(Et, draw[ix], np.isfinite(C['Vx'][d, t]))


def raw_arrays(O, Vall, Fall, d, t, vis, lanes):
    """[W, L] arrays from history T-12..T-1 (+ visible slot T)."""
    idx = t[:, None] + np.arange(-H, 0)[None]
    Oh = O[d[:, None], idx]; Vh = Vall[d[:, None], idx]; Fh = Fall[d[:, None], idx]
    OT = np.where(vis, O[d, t], np.nan).astype(np.float32)
    return raw_from(Oh, OT, Vh, Fh, lanes)


def raw_from(Oh, OT, Vh, Fh, lanes):
    fin = np.isfinite(Oh)
    rev = fin[:, ::-1, :]
    li_ = (Oh.shape[1] - 1) - np.argmax(rev, axis=1)
    hl = np.take_along_axis(Oh, li_[:, None, :], 1)[:, 0]
    hl = np.where(fin.any(1), hl, np.nan)
    olast = np.where(np.isfinite(OT), OT, hl)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        good = fin & np.isfinite(Vh) & np.isfinite(Fh) & (Oh > 0) & (Vh > 0) & (Fh > 0)
        dens = np.full(Oh.shape, np.nan, np.float64)
        dens[good] = np.log(Oh[good].astype(np.float64) * np.broadcast_to(lanes[None, None, :], Oh.shape)[good] * Vh[good] / Fh[good])
        out = dict(OT=OT, olast=olast, m3=np.nanmean(Oh[:, -3:], 1), f3=np.nanmean(Oh[:, :3], 1), mx=np.nanmax(Oh, 1),
                   lr3=np.nanmean(dens[:, -3:], 1))
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def feats_rows(R, oc, cc, wi, li):
    """[n, 13] rows (wi, li) in the NAMES order."""
    L = R['olast'].shape[1]
    with np.errstate(all='ignore'):
        orl = R['olast'] / oc[None, :]
        cols = [R['olast'][wi, li], orl[wi, li], (R['OT'] / oc[None, :])[wi, li], (R['m3'] / oc[None, :])[wi, li],
                ((R['m3'] - R['f3']) / oc[None, :])[wi, li], (R['mx'] / oc[None, :])[wi, li]]
        for k in NB:
            lk = li + k; ok = (lk >= 0) & (lk < L)
            cols.append(np.where(ok, orl[wi, np.clip(lk, 0, L - 1)], np.nan))
        cols.append((R['lr3'] - cc[None, :])[wi, li])
    return np.stack(cols, 1).astype(np.float32)


def rows_of(panel, name, cond=lab.ONGOING):
    return lab.RowCache(f'{AF}/rows/{panel}__{name}__{cond}')


def raw_for_rc(panel, name, rc, cond=lab.ONGOING):
    f = f'{VOUT}/raw_{panel}__{name}__{cond}.npz'
    if os.path.exists(f):
        z = np.load(f); return {k: z[k] for k in z.files if k not in ('d', 't')}
    C = cubes(panel)
    d = rc.win.d.to_numpy(); t = rc.win.t.to_numpy()
    vis = slotT_visible(C, name, cond, d, t)
    R = raw_arrays(C['O'], C['Vall'], C['Fall'], d, t, vis, C['P']['lanes'])
    # independent parity: my slot-T visibility vs the baseline rows' has_T
    if rc.n:
        hT = np.asarray(rc.X[:, rc.cols.index('has_T')])
        R['_par_hasT'] = np.array(bool(np.array_equal(hT.astype(bool), vis[rc.wi, rc.li])))
    tmp = f'{f}.{os.getpid()}.tmp.npz'; np.savez(tmp, d=d, t=t, **R); os.replace(tmp, f)
    return R
