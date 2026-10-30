"""POST-HOC info for EXP_T1_R7 (written 2026-10-30 14:45 +06, AFTER the round-7 gate result 'all FAIL' in logs/t1div7_report.log;
NOTHING here is gate-eligible). Questions: (1) the pre-registered flow pilot optimum HT 6 is the EDGE of its grid -- does a smaller box
help? (2) does knnQ help at weights below the pre-registered grid? (3) do the two small positives (knnQ@0.15, lrtc3m) add up?
  pilotx  flow only, same SV cells / scoring as t1div7_knn.pilot, HT {2, 3, 4} x HL {1, 2, 4} x K {3, 5, 10, 20} x weights {uni, inv},
          level 'ratio' (level 'none' was far worse in the pilot) -> best over the union with the pre-registered grid = knnP
          (knn7_pilot_posthoc.json)
  runx    knnP flow at the evaluation cells -> <PANEL>_f<FOLD>_knnP.npz (|q)
  report  arms on S5 (t1div7_report machinery, functions imported; jproxy_r7posthoc.csv): S4 (parity), base3Q@{0.05, 0.1, 0.15, 0.25},
          knnQ@{0.05, 0.1}, knnPQ@{0.05, 0.1, 0.15, 0.25}, lrtc3m+knnQ@0.15
Usage: t1div7_posthoc.py pilotx | runx | report
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CMD = sys.argv[1]
sys.argv = [sys.argv[0]] + [a for a in sys.argv[2:]]
import t1div7_knn as K7  # noqa: E402
T0 = time.time()
XHT, XHL = (2, 3, 4), (1, 2, 4)


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def pilotx():
    load_panel, SPLITS, L, R4 = K7.lab()
    res = []
    for pi, panel in enumerate(L.PANELS):
        z, x, st = load_panel(panel); meta = L.load_meta(panel); ln = st.lanes.to_numpy()
        for si, split in enumerate(SPLITS):
            M, obs, EM, mv, mq = R4.views(panel, 0, split, z, x, meta)
            rel = obs & ~(EM['HO'] | EM['EB'])
            sv = rel & (np.random.default_rng(41 + 100 * pi + 10 * si).random(rel.shape) < 0.02)
            assert not (sv & (EM['HO'] | EM['EB'])).any()
            mqp = np.where(sv, np.nan, mq); truth = M['mq'][sv].astype(float)
            td, tt, ts = (np.asarray(a, np.int64) for a in np.nonzero(sv))
            Y0, ok, mu, fb = K7.knn_prep(mqp)
            for ht, hl in itertools.product(XHT, XHL):
                dk, vk, rk = K7.knn_core(Y0, ok, td, tt, ts, ht, hl, max(K7.GRID_K), True)
                for k, wm in itertools.product(K7.GRID_K, K7.GRID_W):
                    p = np.maximum(K7.knn_combine(dk, vk, rk, k, wm, 'ratio', fb[tt, ts]) * mu[ts], 20)
                    res.append(dict(panel=panel, split=split, ch='q', ht=ht, hl=hl, k=k, w=wm, lev='ratio',
                                    rmse=float(np.sqrt(np.mean(((p - truth) / ln[ts]) ** 2))), n=int(len(td))))
            log(panel, split, 'SV cells', int(sv.sum()))
    P7 = json.load(open(K7.pilot_path(L)))
    allr = pd.DataFrame(P7['results'] + res)
    g = allr[allr.ch == 'q'].groupby(['ht', 'hl', 'k', 'w', 'lev']).rmse.mean().reset_index().sort_values('rmse', kind='mergesort')
    b = g.iloc[0]; cfg = [int(b.ht), int(b.hl), int(b.k), str(b.w), str(b.lev)]
    log('POST-HOC best flow config over the union of grids', cfg, f'mean SV rmse {b.rmse:.4f} (pre-registered choice',
        P7['chosen']['q'], f'{g[(g.ht == 6) & (g.hl == 2) & (g.k == 10) & (g.w == "inv") & (g.lev == "ratio")].rmse.iloc[0]:.4f})')
    log('  best per (ht, hl), level ratio:\n' + g[g.lev == 'ratio'].groupby(['ht', 'hl']).rmse.min().unstack().round(3).to_string())
    json.dump(dict(note='POST-HOC (after the round-7 gate result); not gate-eligible', chosen_q=cfg, results=res),
              open(os.path.join(L.RES, 'knn7_pilot_posthoc.json'), 'w'), indent=0)


def runx():
    load_panel, SPLITS, L, R4 = K7.lab()
    cfg = tuple(json.load(open(os.path.join(L.RES, 'knn7_pilot_posthoc.json')))['chosen_q'])
    log('knnP flow config (POST-HOC)', cfg)
    for panel in L.PANELS:
        z, x, st = load_panel(panel); meta = L.load_meta(panel); ln = st.lanes.to_numpy()
        for fold in (0, 1):
            saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True); store = {}
            for split in SPLITS:
                M, obs, EM, mv, mq = R4.views(panel, fold, split, z, x, meta)
                assert not np.isfinite(mq[EM['HO'] | EM['EB']]).any()
                for kind, m in L.eval_cells(EM, obs):
                    k = f'{split}_{kind}'; d_, t_, s_ = np.nonzero(m)
                    assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(t_, saved[f'{k}|t']) and np.array_equal(s_, saved[f'{k}|s']), k
                    store[f'{k}|q'] = np.maximum(K7.knn_member(mq, (d_, t_, s_), cfg), 20).astype(np.float32)
                    qt = saved[f'{k}|qt'].astype(float)
                    log(f'  {panel} f{fold} {k:16s} knnP q/lane rmse {np.sqrt(np.mean(((store[k + "|q"] - qt) / ln[s_]) ** 2)):.2f}')
            np.savez_compressed(f'{L.RES}/{panel}_f{fold}_knnP.npz', **store, info=json.dumps(dict(arm='knnP', cfg=cfg, posthoc=True)))


def report():
    sys.argv = [sys.argv[0], 'out=_r7posthoc']
    import t1div7_report as R7  # noqa: E402
    L, R = R7.L, R7.R
    base_load = R7.load_P

    def load_P(panel, fold, split):
        s0, keys, P = base_load(panel, fold, split)
        zc = np.load(f'{L.RES}/{panel}_f{fold}_knnP.npz', allow_pickle=True)
        P['knnP'] = np.concatenate([zc[f'{k}|q'] for k in keys]).astype(float)
        return s0, keys, P
    WS = (0.05, 0.1, 0.15, 0.25)
    ARMS = ['S4'] + [f'base3Q@{w}' for w in WS] + ['knnQ@0.05', 'knnQ@0.1'] + [f'knnPQ@{w}' for w in WS] + ['lrtc3m+knnQ@0.15']
    CONTROL = {**{f'knnQ@{w}': f'base3Q@{w}' for w in (0.05, 0.1)}, **{f'knnPQ@{w}': f'base3Q@{w}' for w in WS}}
    pre = R7.arms_of7

    def arms_post(P, is_blk, vf):
        a7 = pre(P, is_blk, vf)
        v5, q5 = a7['A']; o = ~is_blk
        out = {'A': a7['A'], 'S4': a7['S4']}
        for w in WS:
            out[f'base3Q@{w}'] = (v5, np.where(o, np.maximum((1 - w) * q5 + w * P['base3'][1], 20.0), q5))
            out[f'knnPQ@{w}'] = (v5, np.where(o, np.maximum((1 - w) * q5 + w * P['knnP'], 20.0), q5))
        for w in (0.05, 0.1):
            out[f'knnQ@{w}'] = (v5, np.where(o, np.maximum((1 - w) * q5 + w * P['knn'][1], 20.0), q5))
        e3 = a7['lrtc3m'][1]
        out['lrtc3m+knnQ@0.15'] = (v5, np.where(o, np.maximum(0.85 * e3 + 0.15 * P['knn'][1], 20.0), e3))
        assert set(out) == {'A', *ARMS}
        return out
    R7.arms_of7, R7.load_P = arms_post, load_P
    data = {p: R7.panel_tables(p)[0] for p in L.PANELS}
    keys = [(p, sp) for p in data for sp in R7.SPLITS]
    an = [f'{a}|post' for a in ['A'] + ARMS]
    X = {k: R.prep_arrays(data[k[0]][k[1]], an) for k in keys}
    rng = np.random.default_rng(0); BW = []
    for b in range(R7.B // 4):
        row = {}
        for k in keys:
            nb = len(X[k]['nb']); wb = np.bincount(rng.integers(0, nb, nb), minlength=nb)
            ud = X[k]['ob']; uu = np.unique(ud); cnt = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))
            row[k] = (wb, cnt[np.searchsorted(uu, ud)])
        BW.append(row)
    A = 'A|post'
    base0 = np.array([R.comp(X[k], A, A) for k in keys])
    baseB = np.array([[R.comp(X[k], A, A, *BW[b][k]) for k in keys] for b in range(len(BW))])
    iv = [i for i, k in enumerate(keys) if k[1] == 'validation']; ip = [i for i, k in enumerate(keys) if k[1] == 'private']
    out, JJ = [], {}
    for a in ARMS:
        nm = f'{a}|post'
        d0 = np.array([R.comp(X[k], nm, nm) for k in keys]) - base0; J0 = d0 @ R7.W3
        bJ = ((np.array([[R.comp(X[k], nm, nm, *BW[b][k]) for k in keys] for b in range(len(BW))]) - baseB) @ R7.W3).mean(1)
        JJ[a] = (J0, bJ)
        r = dict(arm=a, scope='deploy', role='POST-HOC', dS_state=d0[:, 0].mean(), dLWR=d0[:, 1].mean(), dS_FD=d0[:, 2].mean(), dJ=J0.mean(),
                 lo=np.percentile(bJ, 2.5), hi=np.percentile(bJ, 97.5), up=int((J0 > 0).sum()), n=len(J0), dJ_val=J0[iv].mean(), dJ_pri=J0[ip].mean(),
                 per=' '.join(f'{v:+.5f}' for v in J0))
        if a in CONTROL:
            Jc, bJc = JJ[CONTROL[a]]
            r.update(ctrl=CONTROL[a], dJ_div=J0.mean() - Jc.mean(), div_lo=np.percentile(bJ - bJc, 2.5), div_up=int((J0 > Jc).sum()))
        out.append(r)
        print(f'POST-HOC {a:17s} deploy dS_state {r["dS_state"]:+.5f} dLWR {r["dLWR"]:+.5f} dS_FD {r["dS_FD"]:+.5f} | dJ {r["dJ"]:+.6f} '
              f'[{r["lo"]:+.6f}, {r["hi"]:+.6f}] up {r["up"]}/{r["n"]} (val {r["dJ_val"]:+.6f} pri {r["dJ_pri"]:+.6f})'
              + (f' | vs {r["ctrl"]} {r["dJ_div"]:+.6f} [lo {r["div_lo"]:+.6f}] {r["div_up"]}/{r["n"]}' if 'ctrl' in r else ''), flush=True)
    T = pd.DataFrame(out); T.to_csv(f'{L.RES}/jproxy_r7posthoc.csv', index=False)
    ps = float(T[T.arm == 'S4'].dJ.iloc[0])
    assert abs(ps + R7.POSTHOC_S5) < 1e-12, 'S5 parity'
    print(f'parity: dJ(S4 - S5) {ps:+.7f} (= -round-5 post-hoc); POST-HOC numbers only, no gate')


if __name__ == '__main__':
    {'pilotx': pilotx, 'runx': runx, 'report': report}[CMD]()
    log('done')
