"""Production snippet for selection-faithful ongoing training (to paste into a NEW Task 2 kernel derived from
kaggle_jobs/t2_model_v20.py) + a local test that it reproduces the lab window sets exactly.

The three functions below use only numpy and the kernel's globals H, F, established.  They read TRAIN days only
(offline window selection / label construction; the filled state is never a feature).

Kernel changes (see the bottom of this file for the exact replacement lines):
  1. after load_panel(): og_ok = faithful_ongoing(P)
  2. P['trainw'], P['trainw_w'] = faithful_train_windows(P, og_ok, on_windows, max_dense=MAX_OG, w_dense=W_DENSE)
  3. the ongoing training matrix gets per-row weights = window weight (to_matrix meta[:, 0] is the window index)
"""
import numpy as np

H, F = 12, 6


def established(Qh):
    return Qh.sum() >= 2 and (Qh.sum(0) >= 2).any()


# ----------------------------------------------------------------------------------------------- snippet starts
def fill_time36(V, maxgap=36):
    """TRAIN-ONLY, SELECTION-ONLY: linear interpolation along time within each day for interior gaps <= maxgap slots."""
    out = V.copy(); D, T, L = V.shape; idx = np.arange(T)
    for d in range(D):
        okd = np.isfinite(V[d])
        for l in np.where(okd.any(0) & ~okd.all(0))[0]:
            ok = okd[:, l]
            if ok.sum() < 2:
                continue
            yi = np.interp(idx, idx[ok], V[d, ok, l])
            prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
            m = (~ok) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < T)
            out[d, m, l] = yi[m]
    return out


def faithful_ongoing(P, pers_max=0.9):
    """bool [D, 288]: origins passing the official ongoing rule evaluated on the time-filled TRAIN state:
    history coverage >= 0.7, established queue in the observed history, a queued filled cell in T+1..T+6 and
    IoU(filled state at T repeated, filled horizon) <= pers_max over all filled cells (empty union counts as 1.0)."""
    Q, E = P['Q'], P['E']; D = Q.shape[0]
    VF = fill_time36(P['V']); MF = np.isfinite(VF); QF = MF & (VF <= P['cut'][None, None, :])
    ok = np.zeros((D, 288), bool)
    for d in range(D):
        for t in range(H, 288 - F):
            if E[d, t - H:t].mean() < 0.7 or not established(Q[d, t - H:t]):
                continue
            Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
            if not Tf.any():
                continue
            S = np.repeat(QF[d, t][None], F, 0)
            u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
            ok[d, t] = (1.0 if u == 0 else i / u) <= pers_max
    return ok


def faithful_train_windows(P, ok, onset_windows, max_dense=3000, w_dense=0.3, seed=0):
    """ongoing training windows = every run-start (first qualifying origin of a run, weight 1.0) + faithful origins on the
    t % 3 == 0 grid that are not run-starts (at most max_dense per panel, weight w_dense).  Returns (windows, weights)."""
    d, t = np.where(ok)
    rs = ~ok[d, t - 1]                                            # t >= H, so t - 1 is always a valid slot
    run = [(int(a), int(b), 'queue_ongoing') for a, b in zip(d[rs], t[rs])]
    k = (~rs) & (t % 3 == 0)
    dense = [(int(a), int(b), 'queue_ongoing') for a, b in zip(d[k], t[k])]
    if len(dense) > max_dense:
        pick = sorted(np.random.default_rng(seed).choice(len(dense), max_dense, replace=False)); dense = [dense[i] for i in pick]
    wins = list(onset_windows) + dense + run
    w = np.r_[np.ones(len(onset_windows)), np.full(len(dense), w_dense), np.ones(len(run))].astype(np.float32)
    return wins, w
# ----------------------------------------------------------------------------------------------- snippet ends


KERNEL_PATCH = r'''
# (a) main loop over panels: replace the ongoing part of P['trainw']
    tw = build_windows(P, global_spacing=False, dense_ongoing=False)          # onset windows as before (ongoing part unused)
    on = [w for w in tw if w[2] == 'queue_onset']
    P['trainw'], P['trainw_w'] = faithful_train_windows(P, faithful_ongoing(P), on, max_dense=MAX_OG, w_dense=0.3)

# (b) final training matrices: keep the per-row window weight for the ongoing rows
    R = make_rows(P, P['trainw'], P['prior_all'])
    for c in Xtr:
        keep = ongoing_keep(R) if c == 'queue_ongoing' else None
        X, y, e, meta, cols = to_matrix(R, c, None, keep)
        Xtr[c].append(X[e]); ytr[c].append(y[e]); wtr[c].append(P['trainw_w'][meta[e, 0]])
    ...
    _wg = np.concatenate(wtr['queue_ongoing'])
    lgb.Dataset(_Xg, _yg, weight=_wg)            # every ongoing member (np511r / id511r, all seeds)
'''

if __name__ == '__main__':
    import time
    import lab
    tot = 0
    for panel in lab.PANELS:
        t0 = time.time(); P = lab.load_panel(panel)
        ok = faithful_ongoing(P)
        _, og = lab.candidates(P)
        assert np.array_equal(ok, og), (panel, 'candidate map differs', int(ok.sum()), int(og.sum()))
        wins, w = faithful_train_windows(P, ok, [], max_dense=10 ** 9)
        rs = lab.window_set(panel, 'runstart'); f3 = lab.window_set(panel, 'faith3')
        mine_rs = sorted((a, b) for (a, b, _), x in zip(wins, w) if x == 1.0)
        mine_dn = sorted((a, b) for (a, b, _), x in zip(wins, w) if x != 1.0)
        assert mine_rs == sorted(zip(rs.d.tolist(), rs.t.tolist())), (panel, 'run-starts differ')
        assert mine_dn == sorted(zip(f3.d[~f3.runstart].tolist(), f3.t[~f3.runstart].tolist())), (panel, 'dense faithful differ')
        # capped version = the local arm's subsample (rng(0) over the non-run-start faith3 windows in (d, t) order)
        wins2, w2 = faithful_train_windows(P, ok, [], max_dense=1500)
        import exp_firstcross_lib as fc
        sub = fc.wfn_F(0.3)(f3, 'faith3', panel) > 0
        assert sorted((a, b) for (a, b, _), x in zip(wins2, w2) if x != 1.0) == sorted(zip(f3.d[sub].tolist(), f3.t[sub].tolist())), (panel, 'cap differs')
        tot += len(wins)
        print(f'{panel}: candidates {int(ok.sum())} == lab, run-starts {len(mine_rs)} == lab, dense faithful {len(mine_dn)} == lab, '
              f'capped-1500 subsample == local arm, {time.time() - t0:.0f}s', flush=True)
        lab.drop_panel(panel)
    print('PROD SNIPPET CHECK PASSED', tot)
