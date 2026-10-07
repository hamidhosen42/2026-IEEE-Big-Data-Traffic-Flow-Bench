"""verify_occ_pm.py seed=2 : pseudo-March (masked validation month) check of the occ candidate with a NEW model seed.
Final models (all 273 train days, k = 4) base / occ x np / id trained here with seed `seed` on the all-finite RowCaches + my own
occupancy features (verify_occ_feat).  March windows / rows: the allfin/occ pmarch pipeline (val_panel, select, val_rows),
my own occupancy features from validation_Om (history T-12..T-1 + masked slot T), calibration k = 4 (mine)."""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import verify_occ_feat as vf

A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
SEED = int(A.get('seed', 2))
sys.argv = [sys.argv[0], 'none']                    # import occ_pmarch without running its main
import occ_pmarch as opm
import exp_firstcross_lib as fc
ONG = lab.ONGOING; F = lab.F; H = lab.H
CID = lab.colset('id'); CNP = lab.colset('np'); IX_NP = [CID.index(c) for c in CNP]
COLS = lab.all_cols(); IXI_ALL = [COLS.index(c) for c in CID]
TRAIN = [('faith3F03', 0.3), ('runstart', 1.0)]
MD = f'{vf.VOUT}/pm_models_s{SEED}'; os.makedirs(MD, exist_ok=True)


def train_final(cal):
    arms = ('base_np', 'occ_np', 'base_id', 'occ_id')
    if all(os.path.exists(f'{MD}/{a}.txt') for a in arms):
        return {a: lgb.Booster(model_file=f'{MD}/{a}.txt') for a in arms}
    parts = []; n = 0
    for panel in lab.PANELS:
        for name, sw in TRAIN:
            rc = vf.rows_of(panel, name)
            if rc.n:
                m = rc.e & rc.keep5[:, 4]; parts.append((panel, name, rc, m, sw)); n += int(m.sum())
    X = np.empty((n, len(CID) + 13), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for panel, name, rc, m, sw in parts:
        oc5, cc5 = cal[panel]; r = np.where(m)[0]; c = len(r)
        raw = vf.raw_for_rc(panel, name, rc)
        X[o:o + c, :len(CID)] = rc.matrix(m, CID, 4); X[o:o + c, len(CID):] = vf.feats_rows(raw, oc5[4], cc5[4], rc.wi[r], rc.li[r])
        y[o:o + c] = rc.y[m]; w[o:o + c] = sw; o += c
    lab.log(f'final train rows {n} pos {y.mean():.4f}')
    occn = [f'occ_{c}' for c in vf.NAMES]; M = {}
    for arm, ix, cols in (('base_np', IX_NP, CNP), ('occ_np', IX_NP + list(range(len(CID), len(CID) + 13)), CNP + occn),
                          ('base_id', list(range(len(CID))), CID), ('occ_id', list(range(len(CID) + 13)), CID + occn)):
        t1 = time.time()
        Xa = np.ascontiguousarray(X[:, ix]) if len(ix) != X.shape[1] else X
        p = lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=SEED)
        mdl = lgb.train(p, lgb.Dataset(Xa, y, weight=w, feature_name=list(cols), free_raw_data=True), 500)
        mdl.save_model(f'{MD}/{arm}.txt'); M[arm] = mdl
        del Xa; gc.collect()
        lab.log(f'final {arm} trained {time.time() - t1:.0f}s')
    del X; gc.collect()
    return M


def main():
    cal = {p: vf.calib_all(p) for p in lab.PANELS}
    M = train_final(cal)
    res = []
    for panel in lab.PANELS:
        t1 = time.time()
        P, Pv = opm.val_panel(panel)
        oc4, cc4 = cal[panel][0][4], cal[panel][1][4]
        on, og, stable, pers = opm.select(Pv)
        D = len(Pv['days'])
        g_on, g_og = lab._glist(on), lab._glist(og); s_og = set()
        for d0 in range(D):
            a, b = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s_og.update(b)
        prior = lab.prior_table(P, np.ones(len(P['days']), bool))
        dd, tt = np.where(og); rs = ~og[dd, tt - 1]
        rsset = set(zip(dd[rs].tolist(), tt[rs].tolist()))
        sets = {'offlike_m': sorted(s_og), 'runstart_m': sorted(rsset),
                'faith3_m': sorted((a, b) for a, b in zip(dd.tolist(), tt.tolist()) if b % 3 == 0)}
        for sname, wl in sets.items():
            if not wl:
                continue
            wins = [(a, b, ONG) for a, b in wl]
            R = opm.val_rows(P, Pv, wins, prior, ONG)
            keep = lab.ongoing_keep(R)
            X, y, e, meta, call = lab.to_matrix(R, ONG, None, keep)
            assert list(call) == COLS
            d = R['d']; t = R['t']; idx = t[:, None] + np.arange(-H, 0)[None]
            raw = vf.raw_from(Pv['Om'][d[:, None], idx], Pv['Om'][d, t].astype(np.float32), Pv['V_all'][d[:, None], idx],
                              Pv['F_all'][d[:, None], idx], P['lanes'])
            # slot-T occupancy visible exactly where the masked slot-T speed is visible?
            vt = np.isfinite(Pv['V_all'][d, t]); assert np.array_equal(np.isfinite(raw['OT']), vt)
            Xo = vf.feats_rows(raw, oc4, cc4, meta[:, 0], meta[:, 1])
            Xid = X[:, IXI_ALL]; Xf = np.hstack([Xid, Xo])
            rc = opm.rc_of(Pv, R, sname, panel, ONG, extra=dict(runstart=np.array([(a, b) in rsset for a, b in wl], bool)))
            P6 = {}
            for arm, ix in (('base_np', IX_NP), ('occ_np', IX_NP + list(range(len(CID), len(CID) + 13))),
                            ('base_id', list(range(len(CID)))), ('occ_id', list(range(len(CID) + 13)))):
                p6 = np.zeros((R['W'], F, R['L']), np.float32)
                p6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = M[arm].predict(Xf[:, ix], num_threads=lab.THREADS)
                P6[arm] = p6
            P6['base_npid'] = 0.5 * (P6['base_np'] + P6['base_id']); P6['occ_npid'] = 0.5 * (P6['occ_np'] + P6['occ_id'])
            for arm, p6 in P6.items():
                res.append(lab.score_probs(rc, p6, arm=arm, set_name=sname))
            del X, Xf, Xid; gc.collect()
        lab.log(panel, 'pmarch done', f'{time.time() - t1:.0f}s', {k: len(v) for k, v in sets.items()})
        lab.drop_panel(panel); gc.collect()
    df = pd.concat(res, ignore_index=True)
    df.to_pickle(f'{vf.VOUT}/pm_s{SEED}.pkl')
    for s in ('offlike_m', 'runstart_m', 'faith3_m'):
        g = df[df.set == s]
        for c in ('np', 'id', 'npid'):
            a = g[g.arm == f'base_{c}']; b = g[g.arm == f'occ_{c}']
            bt = lab.paired_bootstrap(a, b)
            lab.log(f'PMARCH seed {SEED} [{s}] {c}: base {lab.aggregate(a):.4f} occ {lab.aggregate(b):.4f} diff {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] n={bt["n_windows"]}')
            if s == 'offlike_m' and c == 'npid':
                a2, b2 = fc.paired(a, b)
                lab.log('per panel', (b2.groupby('panel').iou.mean() - a2.groupby('panel').iou.mean()).round(4).to_dict())


main()
lab.log('VERIFY PM DONE', SEED)
