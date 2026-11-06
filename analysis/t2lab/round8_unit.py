"""round8_unit.py -- ROUND 8 unit checks BEFORE use (PREREG results/round8/PREREG.md asserted via round8_lib):
 1. exact expected-IoU (Poisson-binomial) vs brute-force enumeration and vs Monte-Carlo on random probability vectors;
 2. FEAT columns: no slot > T (asserted inside feat_cols), truncation check (slots > T replaced by random values -> identical
    columns) on train and pseudo-March windows, production slot-T mask reproduction (has_T) on dense1500 / offlike rows."""
import itertools, json
import numpy as np
import lab, ogfeat_lib as og
import round8_lib as r8

rng = np.random.default_rng(123); rep = {}
# ---------------- 1a brute force
worst = 0.0
for trial in range(30):
    n = int(rng.integers(2, 13)); p = rng.beta(0.5, 2.0, n); p[rng.random(n) < 0.2] = 0.0
    order, curve = r8.eiou_curve(p); ps = p[order]
    outs = np.array(list(itertools.product([0, 1], repeat=n)), bool)               # all 2^n truths (in sorted order)
    pr = np.prod(np.where(outs, ps[None], 1 - ps[None]), 1)
    for m in range(1, len(curve) + 1):
        I = outs[:, :m].sum(1); O = outs[:, m:].sum(1)
        exact = float((pr * I / (m + O)).sum())
        worst = max(worst, abs(exact - curve[m - 1]))
rep['brute_force_max_abs_err'] = worst
assert worst < 1e-12, worst
# ---------------- 1b Monte-Carlo
mc = []
for trial in range(6):
    n = int(rng.choice([20, 60, 146])); p = rng.beta(0.3, 3.0, n)
    order, curve = r8.eiou_curve(p); ps = p[order]
    X = rng.random((200_000, n)) < ps[None]
    for m in sorted({1, 2, 3, len(curve) // 2, int(np.argmax(curve)) + 1, len(curve)}):
        if m < 1 or m > len(curve):
            continue
        I = X[:, :m].sum(1); O = X[:, m:].sum(1); v = I / (m + O)
        z = (v.mean() - curve[m - 1]) / (v.std() / np.sqrt(len(v)))
        mc.append(dict(n=n, m=m, exact=round(float(curve[m - 1]), 5), mc=round(float(v.mean()), 5), z=round(float(z), 2)))
rep['monte_carlo'] = mc
assert max(abs(r['z']) for r in mc) < 4.5, mc
# decoder agrees with argmax of the curve; production decoder for reference
p = rng.beta(0.3, 3.0, 6 * 40); p[:200] = 0
assert r8.decode_exact(p).sum() == int(np.argmax(r8.eiou_curve(p)[1])) + 1
print('DEC unit tests OK', json.dumps(rep)[:400])

# ---------------- 2 FEAT checks
tr = {}
for panel in ('D7_I10_E', 'D12_I5_S', 'D7_I405_N'):
    for name in ('dense1500', 'offlike'):
        rc, Z = r8.train_feat(panel, name)                    # asserts mask == has_T
        tr[f'{panel}/{name}'] = dict(windows=len(rc.win), nan_share=[round(float(np.isnan(Z[..., j]).mean()), 3) for j in range(Z.shape[-1])])
    P = lab.load_panel(panel); st = r8.panel_static(P['npz'])
    win = lab.rows(panel, 'dense1500', lab.ONSET).win
    mask = r8.prod_mask(len(win), len(P['links']))
    d, t = win.d.to_numpy(), win.t.to_numpy()
    full = r8.feat_cols(P['V'], P['Fq'], d, t, mask, st)
    for i in rng.choice(len(win), 40, replace=False):
        V2 = P['V'].copy(); F2 = P['Fq'].copy()
        V2[:, t[i] + 1:] = rng.uniform(0, 130, V2[:, t[i] + 1:].shape); F2[:, t[i] + 1:] = rng.uniform(0, 9000, F2[:, t[i] + 1:].shape)
        z = r8.feat_cols(V2, F2, d[i:i + 1], t[i:i + 1], mask[i:i + 1], st)
        assert np.array_equal(z[0], full[i], equal_nan=True), (panel, i)
    # E1 (masked March layer, mask None)
    _, Pv = og.val_panel(panel)
    e1 = r8.e1_panel(panel)
    d1, t1 = e1['win'].d.to_numpy(), e1['win'].t.to_numpy()
    full1 = r8.feat_cols(Pv['V'], Pv['Fq'], d1, t1, None, st)
    assert np.array_equal(full1[e1['wi'], e1['li']], e1['Z'], equal_nan=True)
    for i in range(len(d1)):
        V2 = Pv['V'].copy(); F2 = Pv['Fq'].copy()
        V2[:, t1[i] + 1:] = rng.uniform(0, 130, V2[:, t1[i] + 1:].shape); F2[:, t1[i] + 1:] = rng.uniform(0, 9000, F2[:, t1[i] + 1:].shape)
        z = r8.feat_cols(V2, F2, d1[i:i + 1], t1[i:i + 1], None, st)
        assert np.array_equal(z[0], full1[i], equal_nan=True), (panel, 'e1', i)
    tr[f'{panel}/truncation'] = 'train 40 windows + E1 %d windows identical' % len(d1)
    lab.drop_panel(panel)
print('FEAT checks OK', json.dumps(tr))
json.dump(dict(dec=rep, feat=tr), open(f'{r8.RES}/unit_checks.json', 'w'), indent=1)
