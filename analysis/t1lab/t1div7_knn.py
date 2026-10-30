"""t1div round 7 (EXP_T1_R7): two new 'month-itself' Task 1 members on the k9 stand-in. Design + PRE-REGISTERED gate: t1div_lib.py
ROUND 7. No training: every estimator is computed per (panel, month, fold) on the fold's EVALUATION VIEW (released cells minus HO | EB,
ineligible-valued cells kept = the view of t1div_lrtc.views; evaluation cells never enter it, asserted), exactly as rounds 4-5.
  knn    kNN-day analog: for a target cell (d, t, l) the K most similar OTHER days d' of the same month, similarity = robust distance
         (mean over the cells released on BOTH days in the box t +- HT slots x l +- HL links (corridor order = order_index) of
         min(|y_d - y_d'|, KNN_CAP), y = value / link mean over the view; valid only with >= KNN_NMIN co-released cells); candidate days
         must have a released value at (t, l); prediction = weighted mean (uniform or 1 / (dist + KNN_EPS)) of their values at (t, l)
         ('ratio' level option: each analog value x the box-sum ratio sum y_d / sum y_d' over the co-released cells, clipped to
         KNN_RCLIP); fallback (no valid analog day): per-(tod, link) mean over the view days, then the link mean. Per channel; the
         config (HT, HL, K, weights, level) is fixed ONCE per channel by `pilot` (self-validation on 2 % released NON-evaluation cells).
  knn0   the same with the plain spec (level 'none'), its own best pilot config (info arms).
  lrtc3  flow LRTC ranks (16, 40, 20) (t1div_lrtc.lrtc, 60 iterations) for the step-M rank ensemble mean((8,40,10), (24,40,10), (16,40,20)).
Vectorised: per target day one 2-D integral image (cumsum over slots then links) of the masked |difference| / count / level sums for all
other days at once, box sums gathered at the target cells only.
Outputs analysis/t2lab/results/t1div/<PANEL>_f<FOLD>_{knn,knn0,lrtc3}.npz (keys <split>_<kind>|v / |q; lrtc3 |q only) in the cell
order of the saved t1bspec results (asserted). Clips v in [3, 1.05 vf], q >= 20 (as every lab member).
Usage: t1div7_knn.py selftest                 (synthetic cube: vectorised core == naive per-cell loop; no lab data)
       t1div7_knn.py timing                   (runtime of one config on one real view; computes NO error on any cell)
       t1div7_knn.py pilot                    (self-validation grid -> knn7_pilot.json; touches no evaluation cell)
       t1div7_knn.py run [panels=...] [folds=0,1]
"""
import itertools, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KNN_CAP, KNN_EPS, KNN_NMIN, KNN_RCLIP = 0.5, 0.01, 5, (0.5, 2.0)
GRID_HT, GRID_HL, GRID_K = (6, 12, 24, 48), (0, 1, 2, 4), (3, 5, 10, 20)
GRID_W, GRID_LEV = ('uni', 'inv'), ('none', 'ratio')
LRTC3_RANKS = (16, 40, 20)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ kNN-day analog (functions copied verbatim into the kernel)
def knn_csum(A):
    """2-D integral image over (slot, link) of A (D, T, S) -> (D, T + 1, S + 1), zero first row / column."""
    C = np.zeros((A.shape[0], A.shape[1] + 1, A.shape[2] + 1))
    C[:, 1:, 1:] = A.cumsum(1).cumsum(2)
    return C


def knn_box(C, t0, t1, l0, l1):
    """box sums of the integral image C over slots [t0, t1) x links [l0, l1) at every target -> (D, n)."""
    return C[:, t1, l1] - C[:, t0, l1] - C[:, t1, l0] + C[:, t0, l0]


def knn_core(Y0, ok, td, tt, ts, ht, hl, kmax, ratio):
    """Y0 (D, T, S) normalised view (0 where not released), ok its release mask, targets (td, tt, ts) -> the kmax nearest analog days of
    every target, sorted by distance (stable: ties by day index): distances dk (kmax, n) (inf = no valid analog), analog values vk
    (kmax, n), level ratios rk (kmax, n) (None unless ratio)."""
    D, T, S = Y0.shape
    n = len(td); kk = min(kmax, D)
    dk = np.full((kmax, n), np.inf); vk = np.zeros((kmax, n)); rk = np.ones((kmax, n)) if ratio else None
    t0 = np.clip(tt - ht, 0, T); t1 = np.clip(tt + ht + 1, 0, T); l0 = np.clip(ts - hl, 0, S); l1 = np.clip(ts + hl + 1, 0, S)
    for d in np.unique(td):
        sel = np.flatnonzero(td == d)
        a, b, c, e = t0[sel], t1[sel], l0[sel], l1[sel]
        cof = (ok & ok[d][None]).astype(np.float64)                        # co-released with day d (row d: d's own cells)
        bA = knn_box(knn_csum(np.minimum(np.abs(Y0 - Y0[d][None]), KNN_CAP) * cof), a, b, c, e)
        bN = knn_box(knn_csum(cof), a, b, c, e)
        dist = np.where(bN >= KNN_NMIN, bA / np.maximum(bN, 1.0), np.inf)
        dist[d] = np.inf                                                   # OTHER days only
        dist[~ok[:, tt[sel], ts[sel]]] = np.inf                            # the analog day needs a released value at (t, l)
        o = np.argsort(dist, axis=0, kind='stable')[:kk]
        dk[:kk, sel] = np.take_along_axis(dist, o, 0)
        vk[:kk, sel] = Y0[o, tt[sel][None, :], ts[sel][None, :]]
        if ratio:
            bD = knn_box(knn_csum(Y0[d][None] * cof), a, b, c, e)
            bO = knn_box(knn_csum(Y0 * cof), a, b, c, e)
            r = np.where(bO > 0, bD / np.where(bO > 0, bO, 1.0), 1.0)
            rk[:kk, sel] = np.clip(np.take_along_axis(r, o, 0), KNN_RCLIP[0], KNN_RCLIP[1])
    return dk, vk, rk


def knn_combine(dk, vk, rk, k, wmode, lev, fb):
    """weighted mean of the k nearest analog values (level-corrected if lev == 'ratio'); fb where no valid analog exists."""
    d_ = dk[:k]; v_ = vk[:k] * rk[:k] if lev == 'ratio' else vk[:k]
    fin = np.isfinite(d_)
    w = np.where(fin, 1.0 / (np.where(fin, d_, 0.0) + KNN_EPS), 0.0) if wmode == 'inv' else fin.astype(np.float64)
    sw = w.sum(0)
    return np.where(sw > 0, (w * v_).sum(0) / np.where(sw > 0, sw, 1.0), fb)


def knn_prep(C):
    """released cube C (D, T, S) in corridor order, NaN = not released -> (Y0, ok, mu, fallback (T, S)); each link normalised by its
    own mean over the released cells (fallback: the mean of the link means), as t1div_lrtc.norm_view."""
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(C, (0, 1))
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu))
    with np.errstate(all='ignore'):
        Y = (C / mu[None, None, :]).astype(np.float64)
    ok = np.isfinite(Y); Y0 = np.where(ok, Y, 0.0)
    n_ts = ok.sum(0); n_s = ok.sum((0, 1))
    m_s = np.where(n_s > 0, Y0.sum((0, 1)) / np.maximum(n_s, 1), 1.0)
    fb = np.where(n_ts > 0, Y0.sum(0) / np.maximum(n_ts, 1), m_s[None, :])
    return Y0, ok, mu, fb


def knn_member(C, tgt, cfg):
    """kNN-day analog prediction (C's units) at the target cells tgt = (day, slot, link) of the released cube C; cfg = (HT, HL, K,
    weights 'uni' | 'inv', level 'none' | 'ratio')."""
    ht, hl, k, wmode, lev = cfg
    Y0, ok, mu, fb = knn_prep(C)
    td, tt, ts = (np.asarray(a, np.int64) for a in tgt)
    dk, vk, rk = knn_core(Y0, ok, td, tt, ts, int(ht), int(hl), int(k), lev == 'ratio')
    return knn_combine(dk, vk, rk, int(k), wmode, lev, fb[tt, ts]) * mu[ts]


# ------------------------------------------------------------------ naive reference (selftest only)
def knn_naive(C, tgt, cfg):
    ht, hl, k, wmode, lev = cfg
    Y0, ok, mu, fb = knn_prep(C)
    D, T, S = C.shape
    out = []
    for d, t, l in zip(*tgt):
        ta, tb, la, lb = max(t - ht, 0), min(t + ht + 1, T), max(l - hl, 0), min(l + hl + 1, S)
        cand = []
        for e in range(D):
            if e == d or not ok[e, t, l]:
                continue
            co = ok[d, ta:tb, la:lb] & ok[e, ta:tb, la:lb]
            if co.sum() < KNN_NMIN:
                continue
            dist = np.minimum(np.abs(Y0[d, ta:tb, la:lb] - Y0[e, ta:tb, la:lb]), KNN_CAP)[co].mean()
            sO = Y0[e, ta:tb, la:lb][co].sum()
            r = float(np.clip(Y0[d, ta:tb, la:lb][co].sum() / sO if sO > 0 else 1.0, *KNN_RCLIP))
            cand.append((dist, e, Y0[e, t, l] * (r if lev == 'ratio' else 1.0)))
        cand.sort(key=lambda c: (c[0], c[1]))
        cand = cand[:k]
        if not cand:
            out.append(fb[t, l] * mu[l]); continue
        w = np.array([1.0 if wmode == 'uni' else 1.0 / (c[0] + KNN_EPS) for c in cand])
        out.append(float((w * np.array([c[2] for c in cand])).sum() / w.sum()) * mu[l])
    return np.array(out)


def selftest():
    rng = np.random.default_rng(3)
    D, T, S = 9, 40, 7
    base = 1 + 0.5 * np.sin(np.arange(T) / 6.0)[None, :, None] * rng.uniform(0.5, 1.5, (D, 1, S)) + 0.1 * rng.normal(size=(D, T, S))
    C = base * rng.uniform(50, 150, S)[None, None, :]
    C[rng.random(C.shape) < 0.45] = np.nan
    C[2, 10:25, :] = np.nan                                                  # a blackout-like block
    C[:, :, 3][rng.random((D, T)) < 0.3] = np.nan
    tg = np.nonzero(rng.random(C.shape) < 0.25)
    worst = 0.0
    cfgs = list(itertools.product((1, 3, 12), (0, 1, 2), (1, 3, 20), GRID_W, GRID_LEV))
    for cfg in cfgs:
        a = knn_member(C, tg, cfg); b = knn_naive(C, tg, cfg)
        assert np.isfinite(a).all(), cfg
        err = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
        worst = max(worst, err)
        assert err < 1e-9, (cfg, err)
    # pilot path (shared core, kmax 20, all (k, w, lev)) == member path
    Y0, ok, mu, fb = knn_prep(C); td, tt, ts = (np.asarray(x, np.int64) for x in tg)
    dk, vk, rk = knn_core(Y0, ok, td, tt, ts, 3, 1, 20, True)
    for k, wm, lev in itertools.product((1, 3, 20), GRID_W, GRID_LEV):
        p = knn_combine(dk, vk, rk, k, wm, lev, fb[tt, ts]) * mu[ts]
        assert np.array_equal(p, knn_member(C, tg, (3, 1, k, wm, lev))), (k, wm, lev)
    print(f'SELFTEST PASSED: vectorised knn_member == naive per-cell loop on {len(tg[0])} cells x {len(cfgs)} configs (max rel diff {worst:.1e}); '
          f'pilot path (kmax 20) == member path bit-identical')


# ------------------------------------------------------------------ lab views
def lab():
    from t1ramp_exp import load_panel, SPLITS  # noqa: E402
    import t1div_lib as L  # noqa: E402
    import t1div_lrtc as R4  # noqa: E402
    return load_panel, SPLITS, L, R4


def timing():
    load_panel, SPLITS, L, R4 = lab()
    panel = 'D7_I10_E'
    z, x, st = load_panel(panel); meta = L.load_meta(panel)
    M, obs, EM, mv, mq = R4.views(panel, 0, 'validation', z, x, meta)
    tg = np.nonzero(EM['HO'] | (EM['EB'] & obs))
    for cfg in ((12, 2, 10, 'inv', 'ratio'), (48, 4, 20, 'inv', 'ratio'), (6, 0, 3, 'uni', 'none')):
        t = time.time(); p = knn_member(mq, tg, cfg)
        log('timing', panel, mq.shape, 'targets', len(tg[0]), cfg, f'{time.time() - t:.2f}s', '| finite', bool(np.isfinite(p).all()))
    Y0, ok, mu, fb = knn_prep(mq)
    allm = np.nonzero(~ok)
    t = time.time(); knn_member(mq, allm, (12, 2, 10, 'inv', 'ratio'))
    log('timing all non-released cells', len(allm[0]), f'{time.time() - t:.2f}s')


PILOT7 = None


def pilot_path(L):
    return os.path.join(L.RES, 'knn7_pilot.json')


def pilot():
    """self-validation: fold-0 views of all 4 panels x 2 months; SV = 2 % of the released eligible NON-evaluation cells
    (rng 41 + 100 panel + 10 split; = the round-4 pilot cells on D12_I5_S validation), hidden from the view; every grid config is scored
    by RMSE (speed km/h, flow vph/lane) at the SV cells; per channel the config with the lowest mean RMSE over the 8 views is chosen
    (knn), and the best config with level 'none' (knn0)."""
    load_panel, SPLITS, L, R4 = lab()
    res = []
    for pi, panel in enumerate(L.PANELS):
        z, x, st = load_panel(panel); meta = L.load_meta(panel); ln = st.lanes.to_numpy()
        for si, split in enumerate(SPLITS):
            t = time.time()
            M, obs, EM, mv, mq = R4.views(panel, 0, split, z, x, meta)
            rel = obs & ~(EM['HO'] | EM['EB'])
            sv = rel & (np.random.default_rng(41 + 100 * pi + 10 * si).random(rel.shape) < 0.02)
            assert not (sv & (EM['HO'] | EM['EB'])).any() and np.isfinite(mv[sv]).all() and np.isfinite(mq[sv]).all()
            mvp, mqp = np.where(sv, np.nan, mv), np.where(sv, np.nan, mq)
            td, tt, ts = (np.asarray(a, np.int64) for a in np.nonzero(sv))
            for ch, C, truth in (('v', mvp, M['mv'][sv].astype(float)), ('q', mqp, M['mq'][sv].astype(float))):
                Y0, ok, mu, fb = knn_prep(C)
                assert not ok[sv].any()
                for ht, hl in itertools.product(GRID_HT, GRID_HL):
                    dk, vk, rk = knn_core(Y0, ok, td, tt, ts, ht, hl, max(GRID_K), True)
                    nfb = int((~np.isfinite(dk[0])).sum())
                    for k, wm, lev in itertools.product(GRID_K, GRID_W, GRID_LEV):
                        p = knn_combine(dk, vk, rk, k, wm, lev, fb[tt, ts]) * mu[ts]
                        p = np.clip(p, 3, 1.05 * st.vf.to_numpy()[ts]) if ch == 'v' else np.maximum(p, 20)
                        e = p - truth
                        rm = float(np.sqrt(np.mean(e ** 2))) if ch == 'v' else float(np.sqrt(np.mean((e / ln[ts]) ** 2)))
                        res.append(dict(panel=panel, split=split, ch=ch, ht=ht, hl=hl, k=k, w=wm, lev=lev, rmse=rm, n=int(len(td)), n_fallback=nfb))
            log(panel, split, 'SV cells', int(sv.sum()), f'{time.time() - t:.0f}s')
    import pandas as pd
    R = pd.DataFrame(res)
    g = R.groupby(['ch', 'ht', 'hl', 'k', 'w', 'lev']).rmse.mean().reset_index()
    chosen, chosen0 = {}, {}
    for ch in ('v', 'q'):
        gc = g[g.ch == ch].sort_values('rmse', kind='mergesort')
        b = gc.iloc[0]; chosen[ch] = [int(b.ht), int(b.hl), int(b.k), str(b.w), str(b.lev)]
        b0 = gc[gc.lev == 'none'].iloc[0]; chosen0[ch] = [int(b0.ht), int(b0.hl), int(b0.k), str(b0.w), str(b0.lev)]
        log('CHOSEN', ch, chosen[ch], f'mean SV rmse {b.rmse:.4f}', '| plain (level none)', chosen0[ch], f'{b0.rmse:.4f}')
        log('  top 8:\n' + gc.head(8).to_string(index=False))
        log('  best per (ht, hl):\n' + gc.groupby(['ht', 'hl']).rmse.min().unstack().round(3).to_string())
    json.dump(dict(design='fold-0 views, 4 panels x 2 months, SV 2% released eligible non-evaluation cells, rng 41+100p+10s',
                   cap=KNN_CAP, eps=KNN_EPS, nmin=KNN_NMIN, rclip=KNN_RCLIP, grid=dict(ht=GRID_HT, hl=GRID_HL, k=GRID_K, w=GRID_W, lev=GRID_LEV),
                   chosen=chosen, chosen0=chosen0, results=res), open(pilot_path(L), 'w'), indent=0)
    log('pilot ->', pilot_path(L))


# ------------------------------------------------------------------ members at the evaluation cells
def run(panels, folds):
    load_panel, SPLITS, L, R4 = lab()
    P7 = json.load(open(pilot_path(L)))
    cfg = {ch: tuple(P7['chosen'][ch]) for ch in ('v', 'q')}; cfg0 = {ch: tuple(P7['chosen0'][ch]) for ch in ('v', 'q')}
    log('knn configs', cfg, '| knn0 (plain)', cfg0, '| lrtc3 ranks', LRTC3_RANKS)
    for panel in panels:
        z, x, st = load_panel(panel); meta = L.load_meta(panel)
        vf = st.vf.to_numpy(); ln = st.lanes.to_numpy()
        for fold in folds:
            outs = {a: f'{L.RES}/{panel}_f{fold}_{a}.npz' for a in ('knn', 'knn0', 'lrtc3')}
            if all(os.path.exists(f) for f in outs.values()):
                log('exists', panel, fold); continue
            saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
            store = {a: {} for a in outs}
            for split in SPLITS:
                t = time.time()
                M, obs, EM, mv, mq = R4.views(panel, fold, split, z, x, meta)
                gone = EM['HO'] | EM['EB']
                assert not np.isfinite(mv[gone]).any() and not np.isfinite(mq[gone]).any(), 'evaluation cells inside the view'
                u, f, mu_v, mu_q = R4.norm_view(mv, mq)
                Zq3 = R4.lrtc(f, LRTC3_RANKS)[0] * mu_q
                for kind, m in L.eval_cells(EM, obs):
                    k = f'{split}_{kind}'
                    d_, t_, s_ = np.nonzero(m)
                    assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(t_, saved[f'{k}|t']) and np.array_equal(s_, saved[f'{k}|s']), k
                    vt, qt = saved[f'{k}|vt'].astype(float), saved[f'{k}|qt'].astype(float)
                    for a, cf in (('knn', cfg), ('knn0', cfg0)):
                        pv = knn_member(mv, (d_, t_, s_), cf['v']); pq = knn_member(mq, (d_, t_, s_), cf['q'])
                        store[a][f'{k}|v'] = np.clip(pv, 3, 1.05 * vf[s_]).astype(np.float32)
                        store[a][f'{k}|q'] = np.maximum(pq, 20).astype(np.float32)
                    store['lrtc3'][f'{k}|q'] = np.maximum(Zq3[m], 20).astype(np.float32)
                    log(f'  {panel} f{fold} {k:16s} n {len(d_):6d} | rmse v knn {np.sqrt(np.mean((store["knn"][k + "|v"] - vt) ** 2)):.3f} '
                        f'knn0 {np.sqrt(np.mean((store["knn0"][k + "|v"] - vt) ** 2)):.3f} | q/lane knn '
                        f'{np.sqrt(np.mean(((store["knn"][k + "|q"] - qt) / ln[s_]) ** 2)):.2f} knn0 '
                        f'{np.sqrt(np.mean(((store["knn0"][k + "|q"] - qt) / ln[s_]) ** 2)):.2f} lrtc3 '
                        f'{np.sqrt(np.mean(((store["lrtc3"][k + "|q"] - qt) / ln[s_]) ** 2)):.2f}')
                log(panel, f'fold {fold}', split, f'{time.time() - t:.0f}s')
            for a in outs:
                info = dict(arm=a, cfg=(cfg if a == 'knn' else cfg0) if a != 'lrtc3' else None, ranks=LRTC3_RANKS if a == 'lrtc3' else None,
                            cap=KNN_CAP, eps=KNN_EPS, nmin=KNN_NMIN, rclip=KNN_RCLIP)
                np.savez_compressed(outs[a], **store[a], info=json.dumps(info))
            log('saved', panel, fold)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selftest':
        selftest()
    elif cmd == 'timing':
        timing()
    elif cmd == 'pilot':
        pilot()
    else:
        assert cmd == 'run'
        import t1div_lib as _L
        kw = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
        run(kw.get('panels', ','.join(_L.PANELS)).split(','), [int(v) for v in kw.get('folds', '0,1').split(',')])
    log('done')
