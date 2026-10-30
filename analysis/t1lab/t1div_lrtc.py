"""t1div round 4 (EXP_T1_DIV4): 'month-itself' estimators as T1 diversity members (plarmuseau 'eleven anomalies' section 12).
Design + PRE-REGISTERED gate: t1div_lib.py ROUND 4. No training: every estimator is computed per (panel, month, fold) on the
fold's EVALUATION VIEW (released cells minus the evaluation cells HO | EB, ineligible-valued cells kept, exactly the view the
LightGBM evaluation features see), per-link normalised by the link's own mean over that view.
  lrtc   EM low-rank tensor completion of (days x 288 x links): init per-(tod, link) mean over days (fallback link mean, 1);
         ITERS times Z = mean over the 3 unfoldings of the rank-r_i truncated-SVD projection, X = released value else Z;
         prediction = Z (ranks per channel: RANKS, chosen once by `pilot`).
  pw41   41-link position-window profile: mean of the normalised value over the days of the same day type (Mon-Fri / Sat-Sun)
         and the links within +-20 positions; prediction = link mean x profile.
Outputs analysis/t2lab/results/t1div/<PANEL>_f<FOLD>_<arm>.npz with the keys / cell order of the other t1div members.
Usage: t1div_lrtc.py pilot                                    (self-validation on 2% released NON-evaluation cells, D12_I5_S
                                                               validation fold 0; touches no evaluation cell)
       t1div_lrtc.py run <PANEL[,PANEL]> [folds=0,1] [arms=lrtc,pw41]
       t1div_lrtc.py run <PANEL[,PANEL]> arms=lrtc name=lrtc_r4 qranks=4,40,10   (POST-HOC info: rank sensitivity, saved as <name>)
"""
import itertools, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1ramp_exp import load_panel, month_cube, SPLITS  # noqa: E402
import t1div_lib as L  # noqa: E402

ITERS = 60
RANKS = {'v': None, 'q': None}          # filled from the pilot result file (see load_ranks)
PILOT = os.path.join(L.RES, 'lrtc_pilot.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def unfold(X, i):
    return np.moveaxis(X, i, 0).reshape(X.shape[i], -1)


def refold(M, i, shape):
    rest = [shape[j] for j in range(3) if j != i]
    return np.moveaxis(M.reshape([shape[i]] + rest), 0, i)


def proj(M, r):
    """rank-r truncated-SVD projection via the smaller Gram matrix."""
    m, n = M.shape
    r = min(r, m, n)
    if m <= n:
        _, U = np.linalg.eigh(M @ M.T); U = U[:, -r:]
        return U @ (U.T @ M)
    _, V = np.linalg.eigh(M.T @ M); V = V[:, -r:]
    return (M @ V) @ V.T


def init_fill(Y):
    ok = np.isfinite(Y)
    with np.errstate(invalid='ignore', divide='ignore'):
        m_ts = np.nansum(Y, 0) / ok.sum(0)                       # (288, S)
        m_s = np.nansum(Y, (0, 1)) / ok.sum((0, 1))              # (S,)
    m_s = np.where(np.isfinite(m_s), m_s, 1.0)
    m_ts = np.where(np.isfinite(m_ts), m_ts, m_s[None, :])
    return np.where(ok, Y, m_ts[None, :, :])


def lrtc(Y, ranks, iters=ITERS, track=None):
    """Y (D, 288, S) normalised, NaN = missing. Returns Z (low-rank estimate everywhere). track: (mask, truth) -> rmse list."""
    ok = np.isfinite(Y); X = init_fill(Y).astype(np.float64); Y0 = np.where(ok, Y, 0.0)
    hist = []
    for it in range(iters):
        Z = sum(refold(proj(unfold(X, i), ranks[i]), i, X.shape) for i in range(3)) / 3.0
        X = np.where(ok, Y0, Z)
        if track is not None and (it + 1) in track[2]:
            hist.append(float(np.sqrt(np.mean((Z[track[0]] - track[1]) ** 2))))
    return Z, hist


def pw41(Y, wkend, half=20):
    """position-window profile (D, 288, S) of the normalised view Y (NaN = missing)."""
    D, T, S = Y.shape
    ok = np.isfinite(Y); Y0 = np.where(ok, Y, 0.0)
    lo = np.clip(np.arange(S) - half, 0, S); hi = np.clip(np.arange(S) + half + 1, 0, S)

    def prof(sel):
        sm = np.pad(Y0[sel].sum(0), ((0, 0), (1, 0))).cumsum(1); ct = np.pad(ok[sel].sum(0), ((0, 0), (1, 0))).cumsum(1)
        n = ct[:, hi] - ct[:, lo]
        return np.where(n > 0, (sm[:, hi] - sm[:, lo]) / np.maximum(n, 1), np.nan)
    allp = prof(np.ones(D, bool))
    out = np.empty((D, T, S))
    for c in (False, True):
        sel = wkend == c
        if sel.any():
            p = prof(sel); p = np.where(np.isfinite(p), p, allp); out[sel] = np.where(np.isfinite(p), p, 1.0)[None]
    return out


def norm_view(mv, mq):
    """per-link means over the view (fallback: panel mean) -> normalised cubes."""
    with np.errstate(invalid='ignore'):
        mu_v = np.nanmean(mv, (0, 1)); mu_q = np.nanmean(mq, (0, 1))
    mu_v = np.where(np.isfinite(mu_v), mu_v, np.nanmean(mu_v)); mu_q = np.where(np.isfinite(mu_q), mu_q, np.nanmean(mu_q))
    return mv / mu_v[None, None, :], mq / mu_q[None, None, :], mu_v, mu_q


def views(panel, fold, split, z, x, meta):
    M = month_cube(panel, z, x, split); obs = M['obs']
    EM = L.eval_masks(M, meta, split, fold)
    gone = EM['HO'] | EM['EB']
    return M, obs, EM, np.where(gone, np.nan, M['mv']), np.where(gone, np.nan, M['mq'])


def load_ranks():
    P = json.load(open(PILOT))
    for ch in ('v', 'q'):
        RANKS[ch] = tuple(P['chosen'][ch])
    return RANKS


def pilot():
    panel, fold, split = 'D12_I5_S', 0, 'validation'
    z, x, st = load_panel(panel); meta = L.load_meta(panel)
    M, obs, EM, mv, mq = views(panel, fold, split, z, x, meta)
    rel = obs & ~(EM['HO'] | EM['EB'])                                      # released eligible, non-evaluation
    sv = rel & (np.random.default_rng(41).random(rel.shape) < 0.02)         # self-validation cells (never evaluation cells)
    assert not (sv & (EM['HO'] | EM['EB'])).any()
    mvp, mqp = np.where(sv, np.nan, mv), np.where(sv, np.nan, mq)
    u, f, mu_v, mu_q = norm_view(mvp, mqp)
    ln = st.lanes.to_numpy()
    dd, tt, ss = np.nonzero(sv)
    truth = {'v': M['mv'][sv].astype(float), 'q': M['mq'][sv].astype(float)}
    res = []
    for ch, Y, mu in (('v', u, mu_v), ('q', f, mu_q)):
        tgt = (truth[ch] / mu[ss])
        for rk in itertools.product((2, 4, 8), (5, 10, 20, 40), (5, 10, 20, 40)):
            t = time.time()
            Z, hist = lrtc(Y, rk, track=(sv, tgt, (30, 60)))
            pred = Z[sv] * mu[ss]
            err = pred - truth[ch]
            rm = float(np.sqrt(np.mean(err ** 2))) if ch == 'v' else float(np.sqrt(np.mean((err / ln[ss]) ** 2)))
            res.append(dict(ch=ch, r_day=rk[0], r_tod=rk[1], r_link=rk[2], rmse=rm, norm_rmse_it30=hist[0], norm_rmse_it60=hist[1],
                            sec=round(time.time() - t, 1)))
            log(ch, rk, f'rmse {rm:.4f} (normalised it30 {hist[0]:.5f} it60 {hist[1]:.5f}) {time.time() - t:.1f}s')
    # reference: pw41 and the init fill on the same cells
    wk = M['dow'] >= 5
    for ch, Y, mu in (('v', u, mu_v), ('q', f, mu_q)):
        for nm, Zr in (('pw41', pw41(Y, wk)), ('init', init_fill(Y))):
            err = Zr[sv] * mu[ss] - truth[ch]
            rm = float(np.sqrt(np.mean(err ** 2))) if ch == 'v' else float(np.sqrt(np.mean((err / ln[ss]) ** 2)))
            log('reference', ch, nm, f'rmse {rm:.4f}')
            res.append(dict(ch=ch, ref=nm, rmse=rm))
    best = {}
    for ch in ('v', 'q'):
        cand = [r for r in res if r['ch'] == ch and 'ref' not in r]
        b = min(cand, key=lambda r: r['rmse']); best[ch] = [b['r_day'], b['r_tod'], b['r_link']]
        log('CHOSEN', ch, best[ch], f'rmse {b["rmse"]:.4f}')
    json.dump(dict(panel=panel, fold=fold, split=split, n_sv=int(sv.sum()), iters=ITERS, results=res, chosen=best), open(PILOT, 'w'), indent=1)
    log('pilot ->', PILOT)


def run(panels, folds, arms, name=None, qranks=None):
    load_ranks()
    if qranks:                                                   # POST-HOC info only (rank sensitivity); never the gate arm
        RANKS['q'] = tuple(int(v) for v in qranks.split(','))
    log('ranks', RANKS, 'iters', ITERS)
    for panel in panels:
        z, x, st = load_panel(panel); meta = L.load_meta(panel)
        vf = st.vf.to_numpy()
        for fold in folds:
            outs = {a: f'{L.RES}/{panel}_f{fold}_{name if (name and a == "lrtc") else a}.npz' for a in arms}
            if all(os.path.exists(f) for f in outs.values()):
                log('exists', panel, fold); continue
            saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
            store = {a: {} for a in arms}
            for split in SPLITS:
                t = time.time()
                M, obs, EM, mv, mq = views(panel, fold, split, z, x, meta)
                u, f, mu_v, mu_q = norm_view(mv, mq)
                est = {}
                if 'lrtc' in arms:
                    est['lrtc'] = (lrtc(u, RANKS['v'])[0] * mu_v, lrtc(f, RANKS['q'])[0] * mu_q)
                if 'pw41' in arms:
                    wk = M['dow'] >= 5
                    est['pw41'] = (pw41(u, wk) * mu_v, pw41(f, wk) * mu_q)
                for kind, m in L.eval_cells(EM, obs):
                    k = f'{split}_{kind}'
                    d_, t_, s_ = np.nonzero(m)
                    assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(t_, saved[f'{k}|t']) and np.array_equal(s_, saved[f'{k}|s']), k
                    for a in arms:
                        V, Q = est[a]
                        store[a][f'{k}|v'] = np.clip(V[m], 3, 1.05 * vf[s_]).astype(np.float32)
                        store[a][f'{k}|q'] = np.maximum(Q[m], 20).astype(np.float32)
                        vt, qt = saved[f'{k}|vt'].astype(float), saved[f'{k}|qt'].astype(float)
                        ln = st.lanes.to_numpy()[s_]
                        log(f'  {panel} f{fold} {k:16s} {a:5s} rmse v {np.sqrt(np.mean((store[a][f"{k}|v"] - vt) ** 2)):.3f} '
                            f'q/lane {np.sqrt(np.mean(((store[a][f"{k}|q"] - qt) / ln) ** 2)):.2f}')
                log(panel, f'fold {fold}', split, f'{time.time() - t:.0f}s')
            for a in arms:
                info = dict(arm=a, iters=ITERS, ranks=RANKS if a == 'lrtc' else None)
                np.savez_compressed(outs[a], **store[a], info=json.dumps(info))
                log('saved', os.path.basename(outs[a]))


if __name__ == '__main__':
    if sys.argv[1] == 'pilot':
        pilot()
    else:
        assert sys.argv[1] == 'run'
        kw = dict(a.split('=') for a in sys.argv[3:] if '=' in a)
        run(sys.argv[2].split(','), [int(f) for f in kw.get('folds', '0,1').split(',')], kw.get('arms', 'lrtc,pw41').split(','),
            name=kw.get('name'), qranks=kw.get('qranks'))
    log('done')
