"""TRACK ogclock: parity + online-rule checks of the own row builder.
1. make_rows_ck = copy of production make_rows (t2_model_v20.py) that appends the ck columns after the production columns.
   With the ck columns cut off, X / y / e / meta must equal the cached production rows lab.rows(panel, set) exactly.
2. The ck columns of make_rows_ck equal the FE cache (build_fe) gathered at (wi, li).
3. Online rule: changing data after T, or train cells before T-12 that the emulated masked layer hides, changes no ck value;
   changing visible cells before T-12 does change some.
Usage: ogclock_parity.py PANEL SET[,SET]"""
import sys, warnings
import numpy as np
import lab
import ogclock_lib as ck

H, F = lab.H, lab.F


def make_rows_ck(P, wins, prior, THIN):
    """production make_rows + ck columns (appended after the production columns of every step)."""
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins]); cond = [w[2] for w in wins]
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P['V'][d[:, None], idx]; Fh = P['Fq'][d[:, None], idx]
    rng = np.random.default_rng(len(wins))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
        lab.add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    R_ = ck.ck_compose(P['V'][d], t, Vh, VT, P['cut'], THIN[d])
    CKF = ck.ck_features(R_, t)
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, P['dow'][d], t, h)
        for k in ck.CK_ALL:
            feats[k] = CKF[k]
        rows.append((h, feats, P['Q'][d, np.minimum(t + h, 287)], P['E'][d, np.minimum(t + h, 287)]))
    return dict(d=d, t=t, cond=np.array(cond), rows=rows, qlast=qlast, W=W, L=L)


panel = sys.argv[1]
sets = [x for x in sys.argv[2].split(',') if x != 'none']
P = lab.load_panel(panel)
THIN, rates = ck.thin_cube(P)
print(f'{panel}: emulated per-day visible share mean {rates.mean():.3f}, cells visible {THIN[P["E"]].mean():.3f} of eligible')
D = len(P['days'])
priors = [lab.prior_table(P, P['block'] != k) for k in range(4)] + [lab.prior_table(P, np.ones(D, bool))]
for s in sets:
    rc = lab.rows(panel, s, lab.ONGOING)
    win = rc.win; wins = list(zip(win.d.tolist(), win.t.tolist(), win.cond.tolist()))
    R = make_rows_ck(P, wins, priors[4], THIN)
    d, t = R['d'], R['t']; dw = P['dow'][d]
    keeps = []
    for k in range(5):
        f0 = dict(R['rows'][0][1]); f0['prior_now'] = lab.prior_feats(priors[k], dw, t, 1)[1]
        keeps.append(lab.ongoing_keep(dict(R, rows=[(1, f0, None, None)])))
    keepU = np.logical_or.reduce(keeps)
    X, y, e, meta, cols = lab.to_matrix(R, lab.ONGOING, None, keepU)
    npc = len(rc.cols)
    assert list(cols[:npc]) == list(rc.cols) and list(cols[npc:]) == ck.CK_ALL
    same = np.array_equal(X[:, :npc], np.asarray(rc.X), equal_nan=True)
    maxdiff = float(np.nanmax(np.abs(X[:, :npc] - np.asarray(rc.X)))) if X.shape == rc.X.shape else float('nan')
    print(f'{s}: rows {len(X)} vs cache {rc.n} | production columns identical: {same} (max abs diff {maxdiff:.3g}) | y {np.array_equal(y.astype(bool), rc.y)} '
          f'e {np.array_equal(e.astype(bool), rc.e)} meta {np.array_equal(meta, np.stack([rc.wi, rc.li, rc.h], 1))}')
    FE = ck.build_fe(panel, s)
    Xe = np.asarray(FE)[rc.wi, rc.li]
    print(f'{s}: ck columns == FE cache gathered at (wi, li): {np.array_equal(X[:, npc:], Xe, equal_nan=True)}; '
          f'ck NaN share per column: ' + ', '.join(f'{c} {np.isnan(Xe[:, j]).mean():.2f}' for j, c in enumerate(ck.CK_ALL)))
    assert same and np.array_equal(X[:, npc:], Xe, equal_nan=True)

# online rule on 80 offlike windows
rc = lab.rows(panel, 'offlike', lab.ONGOING); win = rc.win.iloc[:80]
Vh, Fh, VT, d, t = ck.train_inputs(P, rc.win)
Vh, VT, d, t = Vh[:80], VT[:80], d[:80], t[:80]
base = ck.fe_train(P, THIN, d, t, Vh, VT)
V0 = P['V'].copy()
rng = np.random.default_rng(1)
s_ = np.arange(288)[None, :, None]
for name, sel in (('after T', lambda dd, tt: s_ > tt), ('hidden cells before T-12', lambda dd, tt: (s_ < tt - H) & ~THIN[dd][None]),
                  ('visible cells before T-12', lambda dd, tt: (s_ < tt - H) & THIN[dd][None])):
    changed = 0
    for i in range(len(d)):
        P['V'] = V0.copy()
        m = np.broadcast_to(sel(d[i], t[i])[0], P['V'].shape[1:])
        P['V'][d[i]][m] = (rng.random(m.sum()) * 120).astype(np.float32)
        x = ck.fe_train(P, THIN, d[i:i + 1], t[i:i + 1], Vh[i:i + 1], VT[i:i + 1])
        changed += int(not np.array_equal(x, base[i:i + 1], equal_nan=True))
    print(f'online rule: perturbing {name}: windows with any ck change {changed}/{len(d)}')
P['V'] = V0
lab.log('DONE')
