"""t1div round 5 (EXP_T1_DIV5 / EXP_T1_W5): LRTC follow-up members. Design + PRE-REGISTERED gates: t1div_lib.py ROUND 5.
Every estimator is computed per (panel, month, fold) on the fold's EVALUATION VIEW (released cells minus HO | EB, ineligible-valued
cells kept), exactly as t1div_lrtc.py (round 4), whose functions are reused.
  lrtc2r  flow LRTC, ranks = best pilot triple of the extended grid r_day {12, 16, 24} x r_tod {20, 40, 60} x r_link {5, 10, 20}
  lrtc2i  flow LRTC, ranks (8, 40, 10), initialised with the pw41 day-type position-window profile
  lrtcK   density LRTC (k = q / max(v, 1), per-link mean normalisation); stored as k_hat (veh/km), the report converts q = k_hat v_S4
  lrtcL   log-flow LRTC (log max(q, 1) minus the link mean); q = exp(Z + mean)
  stkQ    NNLS stack of {lrtc (8, 40, 10), lrtc2r, lrtcL, pw41, per-(tod, link) mean} flow, fitted on 2 % self-validation cells of the
          view (rng 4300 + 10 split + fold, never evaluation cells), estimators computed on the SV-hidden view
Outputs analysis/t2lab/results/t1div/<PANEL>_f<FOLD>_<arm>.npz (keys <split>_<kind>|q, lrtcK: |k) in the cell order of the saved
t1bspec results (asserted).
Usage: t1div5_lrtc.py pilot        (self-validation on the round-4 pilot cells; touches no evaluation cell -> lrtc5_pilot.json)
       t1div5_lrtc.py run [panels=...] [folds=0,1]
"""
import itertools, json, os, sys, time
import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1ramp_exp import load_panel, SPLITS  # noqa: E402
import t1div_lib as L  # noqa: E402
import t1div_lrtc as R4  # noqa: E402

ITERS = 60
Q_RANKS = (8, 40, 10)                       # round-4 flow ranks (production tfb-k7)
PILOT5 = os.path.join(L.RES, 'lrtc5_pilot.json')
GRID4 = list(itertools.product((2, 4, 8), (5, 10, 20, 40), (5, 10, 20, 40)))
GRID_EXT = list(itertools.product((12, 16, 24), (20, 40, 60), (5, 10, 20)))
ARMS = ('lrtc2r', 'lrtc2i', 'lrtcK', 'lrtcL', 'stkQ')
STK_NAMES = ('lrtc', 'lrtc2r', 'lrtcL', 'pw41', 'tlmean')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def lrtc_x0(Y, ranks, X0, iters=ITERS):
    """EM LRTC (= t1div_lrtc.lrtc) from an explicit initial fill X0 (observed cells of Y are imposed)."""
    ok = np.isfinite(Y); Y0 = np.where(ok, Y, 0.0); X = np.where(ok, Y0, X0).astype(np.float64)
    Z = X
    for _ in range(iters):
        Z = sum(R4.refold(R4.proj(R4.unfold(X, i), ranks[i]), i, X.shape) for i in range(3)) / 3.0
        X = np.where(ok, Y0, Z)
    return Z


def init_fill0(Y, fallback):
    """t1div_lrtc.init_fill with a configurable fallback (0 for centred logs)."""
    ok = np.isfinite(Y)
    with np.errstate(invalid='ignore', divide='ignore'):
        m_ts = np.nansum(Y, 0) / ok.sum(0)
        m_s = np.nansum(Y, (0, 1)) / ok.sum((0, 1))
    m_s = np.where(np.isfinite(m_s), m_s, fallback)
    m_ts = np.where(np.isfinite(m_ts), m_ts, m_s[None, :])
    return np.where(ok, Y, m_ts[None, :, :])


def density_view(mv, mq):
    """k = q / max(v, 1) where both are present; per-link mean normalisation (fallback panel mean)."""
    k = np.where(np.isfinite(mv) & np.isfinite(mq), mq / np.maximum(mv, 1.0), np.nan)
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(k, (0, 1))
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu))
    return k / mu[None, None, :], mu


def logflow_view(mq):
    y = np.where(np.isfinite(mq), np.log(np.maximum(mq, 1.0)), np.nan)
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(y, (0, 1))
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu))
    return y - mu[None, None, :], mu


def est_lrtc2r(f, mu_q, ranks):
    return R4.lrtc(f, ranks)[0] * mu_q


def est_lrtc2i(f, mu_q, wk):
    return lrtc_x0(f, Q_RANKS, R4.pw41(f, wk)) * mu_q


def est_lrtcL(mq, ranks):
    Y, mu = logflow_view(mq)
    return np.exp(lrtc_x0(Y, ranks, init_fill0(Y, 0.0)) + mu[None, None, :])


def est_lrtcK(mv, mq, ranks):
    Y, mu = density_view(mv, mq)
    return R4.lrtc(Y, ranks)[0] * mu[None, None, :]


# ------------------------------------------------------------------ pilot (round-4 pilot cells; no evaluation cell)
def pilot():
    panel, fold, split = 'D12_I5_S', 0, 'validation'
    z, x, st = load_panel(panel); meta = L.load_meta(panel)
    M, obs, EM, mv, mq = R4.views(panel, fold, split, z, x, meta)
    rel = obs & ~(EM['HO'] | EM['EB'])
    sv = rel & (np.random.default_rng(41).random(rel.shape) < 0.02)          # = round-4 pilot cells
    assert not (sv & (EM['HO'] | EM['EB'])).any()
    mvp, mqp = np.where(sv, np.nan, mv), np.where(sv, np.nan, mq)
    u, f, mu_v, mu_q = R4.norm_view(mvp, mqp)
    ln = st.lanes.to_numpy(); _, _, ss = np.nonzero(sv)
    qt = M['mq'][sv].astype(float); vt = M['mv'][sv].astype(float); kt = qt / np.maximum(vt, 1.0)
    rq = lambda pred: float(np.sqrt(np.mean(((pred - qt) / ln[ss]) ** 2)))
    res = []
    for rk in GRID_EXT:
        t = time.time(); rm = rq(R4.lrtc(f, rk)[0][sv] * mu_q[ss])
        res.append(dict(arm='lrtc2r', r_day=rk[0], r_tod=rk[1], r_link=rk[2], rmse=rm)); log('lrtc2r', rk, f'q/lane rmse {rm:.3f} {time.time() - t:.1f}s')
    Yk, muk = density_view(mvp, mqp)
    for rk in GRID4:
        Z = R4.lrtc(Yk, rk)[0]; rm = float(np.sqrt(np.mean(((Z[sv] * muk[ss] - kt) / ln[ss]) ** 2)))
        res.append(dict(arm='lrtcK', r_day=rk[0], r_tod=rk[1], r_link=rk[2], rmse=rm)); log('lrtcK', rk, f'k/lane rmse {rm:.4f}')
    for rk in GRID4:
        rm = rq(est_lrtcL(mqp, rk)[sv])
        res.append(dict(arm='lrtcL', r_day=rk[0], r_tod=rk[1], r_link=rk[2], rmse=rm)); log('lrtcL', rk, f'q/lane rmse {rm:.3f}')
    wk = M['dow'] >= 5
    info = dict(lrtc_844010=rq(R4.lrtc(f, Q_RANKS)[0][sv] * mu_q[ss]), lrtc2i_844010=rq(est_lrtc2i(f, mu_q, wk)[sv]))
    chosen = {}
    for a in ('lrtc2r', 'lrtcK', 'lrtcL'):
        b = min((r for r in res if r['arm'] == a), key=lambda r: r['rmse']); chosen[a] = [b['r_day'], b['r_tod'], b['r_link']]
        log('CHOSEN', a, chosen[a], f'rmse {b["rmse"]:.4f}')
    # info: density-member flow with the TRUE speed on the pilot cells (upper bound of the conversion), never used for choices
    Zk = R4.lrtc(Yk, chosen['lrtcK'])[0][sv] * muk[ss]
    info['lrtcK_flow_true_speed'] = rq(Zk * np.maximum(vt, 1.0))
    log('info', json.dumps({k: round(v, 3) for k, v in info.items()}))
    json.dump(dict(panel=panel, fold=fold, split=split, n_sv=int(sv.sum()), iters=ITERS, results=res, chosen=chosen, info=info),
              open(PILOT5, 'w'), indent=1)
    log('pilot ->', PILOT5)


# ------------------------------------------------------------------ members
def run(panels, folds):
    P5 = json.load(open(PILOT5))['chosen']
    rk2, rkK, rkL = tuple(P5['lrtc2r']), tuple(P5['lrtcK']), tuple(P5['lrtcL'])
    log('ranks lrtc2r', rk2, 'lrtcK', rkK, 'lrtcL', rkL, '| flow', Q_RANKS)
    for panel in panels:
        z, x, st = load_panel(panel); meta = L.load_meta(panel)
        vf = st.vf.to_numpy(); ln = st.lanes.to_numpy()
        for fold in folds:
            outs = {a: f'{L.RES}/{panel}_f{fold}_{a}.npz' for a in ARMS}
            if all(os.path.exists(f) for f in outs.values()):
                log('exists', panel, fold); continue
            saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
            store = {a: {} for a in ARMS}; betas = {}
            for si, split in enumerate(SPLITS):
                t = time.time()
                M, obs, EM, mv, mq = R4.views(panel, fold, split, z, x, meta)
                u, f, mu_v, mu_q = R4.norm_view(mv, mq)
                wk = M['dow'] >= 5
                est = {'lrtc2r': est_lrtc2r(f, mu_q, rk2), 'lrtc2i': est_lrtc2i(f, mu_q, wk), 'lrtcL': est_lrtcL(mq, rkL)}
                Kh = est_lrtcK(mv, mq, rkK)
                # stack: SV cells = 2 % of the released non-evaluation cells, estimators on the SV-hidden view
                rel = obs & ~(EM['HO'] | EM['EB'])
                sv = rel & (np.random.default_rng(4300 + 10 * si + fold).random(rel.shape) < 0.02)
                assert not (sv & (EM['HO'] | EM['EB'])).any()
                mvs, mqs = np.where(sv, np.nan, mv), np.where(sv, np.nan, mq)
                _, fs, _, mus = R4.norm_view(mvs, mqs)
                comp = {'lrtc': R4.lrtc(fs, Q_RANKS)[0], 'lrtc2r': R4.lrtc(fs, rk2)[0], 'lrtcL': est_lrtcL(mqs, rkL) / mus[None, None, :],
                        'pw41': R4.pw41(fs, wk), 'tlmean': R4.init_fill(fs)}
                A = np.stack([comp[c][sv] for c in STK_NAMES], 1); b = M['mq'][sv] / mus[np.nonzero(sv)[2]]
                beta, _ = nnls(A, b); betas[split] = dict(zip(STK_NAMES, beta.round(4).tolist()))
                stk = sum(beta[j] * comp[c] for j, c in enumerate(STK_NAMES)) * mus[None, None, :]
                est['stkQ'] = stk
                rm_sv = {c: float(np.sqrt(np.mean(((A[:, j] - b) * mus[np.nonzero(sv)[2]] / ln[np.nonzero(sv)[2]]) ** 2))) for j, c in enumerate(STK_NAMES)}
                rm_sv['stack'] = float(np.sqrt(np.mean(((A @ beta - b) * mus[np.nonzero(sv)[2]] / ln[np.nonzero(sv)[2]]) ** 2)))
                log(f'  {panel} f{fold} {split} stack beta {betas[split]} | SV q/lane rmse ' + ' '.join(f'{k} {v:.2f}' for k, v in rm_sv.items()))
                for kind, m in L.eval_cells(EM, obs):
                    k = f'{split}_{kind}'
                    d_, t_, s_ = np.nonzero(m)
                    assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(t_, saved[f'{k}|t']) and np.array_equal(s_, saved[f'{k}|s']), k
                    qt = saved[f'{k}|qt'].astype(float); vt = saved[f'{k}|vt'].astype(float)
                    for a in ('lrtc2r', 'lrtc2i', 'lrtcL', 'stkQ'):
                        store[a][f'{k}|q'] = np.maximum(est[a][m], 20).astype(np.float32)
                    store['lrtcK'][f'{k}|k'] = np.maximum(Kh[m], 0.01).astype(np.float32)
                    msg = ' '.join(f'{a} {np.sqrt(np.mean(((store[a][f"{k}|q"] - qt) / ln[s_]) ** 2)):.2f}' for a in ('lrtc2r', 'lrtc2i', 'lrtcL', 'stkQ'))
                    kq = store['lrtcK'][f'{k}|k'] * np.maximum(vt, 1.0)          # info only: conversion with the TRUE speed
                    log(f'  {panel} f{fold} {k:16s} q/lane rmse {msg} | lrtcK(true v, info) {np.sqrt(np.mean(((kq - qt) / ln[s_]) ** 2)):.2f}')
                log(panel, f'fold {fold}', split, f'{time.time() - t:.0f}s')
            for a in ARMS:
                info = dict(arm=a, iters=ITERS, ranks=dict(lrtc2r=rk2, lrtc2i=Q_RANKS, lrtcK=rkK, lrtcL=rkL).get(a), betas=betas if a == 'stkQ' else None)
                np.savez_compressed(outs[a], **store[a], info=json.dumps(info))
            log('saved', panel, fold)


if __name__ == '__main__':
    if sys.argv[1] == 'pilot':
        pilot()
    else:
        assert sys.argv[1] == 'run'
        kw = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
        run(kw.get('panels', ','.join(L.PANELS)).split(','), [int(v) for v in kw.get('folds', '0,1').split(',')])
    log('done')
