"""Review (1) feature parity and online rule, (2) fold safety, (3c) selection-only quantities never reach a feature.
The production functions are compiled a SECOND way here (source text truncated before the main block, executed as a
module with the real lightgbm / catboost imports), independent of lab's ast extraction. Usage: review_parity.py [PANEL ...]"""
import sys, types, numpy as np, pandas as pd, lab

panels = sys.argv[1:] or ['D7_I10_W', 'D12_I5_S']
src = open(lab.PROD_FILE).read()
main_at = src.index('# ------------------------------------------------------------------ main')
head = src[:main_at].replace("REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]", "REL = '/nonexistent'")
assert 'panels = {}' not in head
px = types.ModuleType('prod_direct'); exec(compile(head, 'prod_direct', 'exec'), px.__dict__)
assert px.PRIOR_COLS == lab.PRIOR_COLS and px.PARAMS == dict(lab.prod.PARAMS)     # ID_DROP is defined inside the main block
rng = np.random.default_rng(123)
fails = 0


def chk(ok, msg):
    global fails
    print(('PASS ' if ok else 'FAIL ') + msg, flush=True)
    fails += (not ok)


for panel in panels:
    P = lab.load_panel(panel); D = len(P['days']); L = len(P['links'])
    chk(np.array_equal(P['block'], np.minimum(np.arange(D) * 4 // D, 3)) and np.all(np.diff(P['block']) >= 0), f'{panel}: fold blocks = production formula, monotone in day')
    for name in ('offlike', 'official', 'dense1500'):
        rc = lab.rows(panel, name)
        wins = list(zip(rc.win.d.tolist(), rc.win.t.tolist(), rc.win.cond.tolist()))
        chk(np.array_equal(rc.fold, P['block'][rc.win.d.to_numpy()[rc.wi]]), f'{panel}/{name}: rc.fold == block[day]')
        for k in (0, 2, 4):
            prior = px.prior_table(P, P['block'] != k if k < 4 else np.ones(D, bool))
            # (2) prior of fold k ignores the fold-k days: zeroing those days changes nothing
            if k < 4:
                P2 = dict(P); P2['Q'] = P['Q'].copy(); P2['E'] = P['E'].copy(); P2['Q'][P['block'] == k] = False; P2['E'][P['block'] == k] = False
                chk(np.array_equal(prior, px.prior_table(P2, P['block'] != k)), f'{panel}/{name}: prior_table(block != {k}) independent of fold-{k} days')
            with np.errstate(all='ignore'):
                import warnings; warnings.simplefilter('ignore')
                R = px.make_rows(P, wins, prior)                        # same full list -> same slot-T mask seed
            Xd, yd, ed, md, cd = px.to_matrix(R, rc.cond, None, px.ongoing_keep(R))
            Xc = rc.matrix(rc.keep5[:, k], rc.cols, k)
            same_cols = list(cd) == rc.cols
            same = same_cols and Xc.shape == Xd.shape and np.array_equal(Xc, Xd, equal_nan=True)
            chk(same and np.array_equal(rc.y[rc.keep5[:, k]], yd) and np.array_equal(rc.e[rc.keep5[:, k]], ed)
                and np.array_equal(np.stack([rc.wi, rc.li, rc.h], 1)[rc.keep5[:, k]], md),
                f'{panel}/{name}: direct production rows (prior {k}) == cache: {Xd.shape} cols_same={same_cols}')
            if name == 'official':
                # matrix() through colset('np') / ('id') is a pure column selection
                for cs in ('np', 'id'):
                    cols = lab.colset(cs); ix = [rc.cols.index(c) for c in cols]
                    chk(np.array_equal(lab.rows(panel, name).matrix(rc.keep5[:, k], cols, k), Xd[:, ix], equal_nan=True), f'{panel}: colset {cs} == column subset ({len(cols)} cols)')
        del rc
    # (1b) online rule: perturb data after T (and before T-60) and check the features; one window at a time
    rc = lab.rows(panel, 'offlike'); wi_ = rc.win
    pick = rng.choice(len(wi_), 12, replace=False)
    ramp_cols = [c for c in rc.cols if c.startswith('ramp_')]
    allowed_T = {'vr_last', 'q_last', 'vr_T', 'has_T', 'dq_hi', 'dq_lo', 'share_q', 'vr_min3'} | {c for c in rc.cols if c.startswith(('vr_n', 'q_n', 'vr_min_'))}
    res = {'future': set(), 'before_hist': set(), 'ramp_at_T_or_later': set(), 'ramp_T-12..T-4': set(), 'ramp_T-3..T-1': set(), 'slot_T': set()}
    for i in pick:
        d, t = int(wi_.d[i]), int(wi_.t[i]); w1 = [(d, t, lab.ONGOING)]
        prior = px.prior_table(P, np.ones(D, bool))

        def feats(Pm):
            with np.errstate(all='ignore'):
                R = px.make_rows(Pm, w1, prior)
            X, _, _, _, cols = px.to_matrix(R, lab.ONGOING, None, None)
            return X, list(cols)

        X0, cols = feats(P)

        def perturb(**mods):
            Pm = dict(P)
            for key, (sl, val) in mods.items():
                a = P[key].copy(); a[d][sl] = val; Pm[key] = a
            return Pm

        def diff(Pm):
            X1, _ = feats(Pm)
            return {cols[j] for j in range(len(cols)) if not np.array_equal(X0[:, j], X1[:, j], equal_nan=True)}
        res['future'] |= diff(perturb(V=(slice(t + 1, None), 1.0), Fq=(slice(t + 1, None), 1.0), Q=(slice(t + 1, None), True), E=(slice(t + 1, None), True)))
        res['before_hist'] |= diff(perturb(V=(slice(0, t - 12), 1.0), Fq=(slice(0, t - 12), 1.0), Q=(slice(0, t - 12), True)))
        res['ramp_at_T_or_later'] |= diff(perturb(ON=(slice(t, None), 5000.0), OFF=(slice(t, None), 5000.0)))
        res['ramp_T-12..T-4'] |= diff(perturb(ON=(slice(t - 12, t - 3), 5000.0), OFF=(slice(t - 12, t - 3), 5000.0)))
        res['ramp_T-3..T-1'] |= diff(perturb(ON=(slice(t - 3, t), 5000.0), OFF=(slice(t - 3, t), 5000.0)))
        res['slot_T'] |= diff(perturb(V=(slice(t, t + 1), 1.0)))
    chk(not res['future'], f'{panel}: no feature depends on data after T (changed: {sorted(res["future"])})')
    chk(not res['before_hist'], f'{panel}: no feature depends on data before T-60 (changed: {sorted(res["before_hist"])})')
    chk(not res['ramp_at_T_or_later'], f'{panel}: ramp features ignore ramp flows at T or later (changed: {sorted(res["ramp_at_T_or_later"])})')
    chk(res['ramp_T-12..T-4'] <= {'ramp_on_surge'}, f'{panel}: ramp flows in T-12..T-4 reach only ramp_on_surge (changed: {sorted(res["ramp_T-12..T-4"])})')
    chk(res['ramp_T-3..T-1'] <= set(ramp_cols), f'{panel}: ramp flows in T-3..T-1 reach only ramp_* (changed: {sorted(res["ramp_T-3..T-1"])})')
    chk(res['slot_T'] <= allowed_T, f'{panel}: slot-T value reaches only the masked-slot features (changed: {sorted(res["slot_T"])})')
    # (3c) selection-only quantities: not a column; and the row cache X is identical whether or not the fill is loaded (by construction)
    bad = [c for c in rc.cols if any(s in c for s in ('pers', 'faith', 'runstart', 'fill', 'QF', 'VF'))]
    chk(not bad, f'{panel}: no selection-side column in X ({len(rc.cols)} cols): {bad}')
    # (2b) training rows for fold k come from windows outside fold k; evaluation windows are inside; day sets disjoint
    for k in (0, 3):
        X, y, w = lab.train_matrix('offlike', k, lab.colset('np'), panels=[panel])
        m = rc.e & rc.keep5[:, k] & (rc.fold != k)
        chk(len(X) == int(m.sum()) and np.array_equal(X, rc.matrix(m, lab.colset('np'), k), equal_nan=True) and np.array_equal(y.astype(bool), rc.y[m]),
            f'{panel}: train_matrix(offlike, k={k}) == rows with fold != {k} ({len(X)} rows)')
        tr_days = set(rc.win.d[rc.win.fold != k]); ev_days = set(rc.win.d[rc.win.fold == k])
        chk(not (tr_days & ev_days) and ev_days == set(np.where(P['block'] == k)[0]) & set(rc.win.d), f'{panel}: fold {k} train days and eval days disjoint ({len(tr_days)} vs {len(ev_days)} days)')
    # (2c) predict_set uses only fold-k windows and the fold-k keep region
    class Const:
        def predict(self, X, num_threads=1):
            return np.full(len(X), 0.5, np.float32)
    P6 = lab.predict_set(Const(), rc, 1, lab.colset('np'))
    f1 = rc.win.fold.to_numpy() == 1
    chk(np.all(P6[~f1] == 0) and np.all((P6[f1] > 0).reshape(f1.sum(), -1).any(1)), f'{panel}: predict_set(k=1) fills only fold-1 windows')
    keep_img = np.zeros_like(P6, bool); keep_img[rc.wi[rc.keep5[:, 1]], rc.h[rc.keep5[:, 1]] - 1, rc.li[rc.keep5[:, 1]]] = True
    chk(np.array_equal(P6 > 0, keep_img & f1[:, None, None]), f'{panel}: predict_set(k=1) nonzero exactly on keep5[:, 1]')
    lab.drop_panel(panel)
print('PARITY REVIEW DONE, failures:', fails)
