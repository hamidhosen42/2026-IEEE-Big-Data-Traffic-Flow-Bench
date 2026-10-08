"""ogfeat v27: parity + smoke of the generated kernel kaggle_jobs/t2_ogfeat_v27a.py (functions compiled with ast; the Kaggle main
never runs, no Kaggle path is touched).
A (lab parity, plain inputs): kernel make_rows + to_matrix on the lab panel dict (no all-finite / occupancy cubes) == lab row cache
  production columns + lab ogfeat extras (og_*_all2 columns) for offlike and runstart windows: same rows, bit-identical values.
B (production inputs, smoke): kernel P with all-finite cubes (Vf / Ff) + occupancy (O, ocrit, occ_c) from the local exports.
  (1) every v26 column of the v27 rows == v26's own make_rows (compiled from kaggle_jobs/t2_occ_v26.py) on the same windows;
  (2) every ogfeat column == ogfeat_lib.og_extra / og_extra2 / og_step applied independently to the all-finite inputs v26 feeds
      window_features (history Vf / Ff, slot-T draw with the ineligible-valued override);
  (3) shapes, column sets for np / id, finite shares.
Usage: ogfeat_kernel_parity_v27.py [PANEL] [max_windows]"""
import ast, glob, gc, json, os, sys, time, types, warnings
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import verify_occ_feat as vf

KF = f'{lab.ROOT}/kaggle_jobs/t2_ogfeat_v27a.py'
K26 = f'{lab.ROOT}/kaggle_jobs/t2_occ_v26.py'
panel = sys.argv[1] if len(sys.argv) > 1 else 'D7_I10_W'
MAXW = int(sys.argv[2]) if len(sys.argv) > 2 else 400
CONSTS = {'H', 'F', 'SPACING', 'PARAMS', 'PRIOR_COLS', 'ID_DROP', 'OG_STATIC', 'OG_STEP', 'OG_ADD_PROD', 'CAPD', 'BIG', 'OG_STEP_ALL',
          'ONGOING', 'VARIANTS', 'NB', 'OCC_COLS', 'ALLFIN', 'OCC', 'PART', 'TAG', 'W_RS', 'W_DENSE', 'FILL_GAP', 'PERS_MAX', 'MAX_OG',
          'N_BIG', 'TIME_LIMIT_S'}


def compile_kernel(path):
    tree = ast.parse(open(path).read()); body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            body.append(node)
        elif isinstance(node, ast.Assign):
            tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if tg and tg <= CONSTS:
                body.append(node)
    K = types.ModuleType(os.path.basename(path)[:-3])
    K.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, warnings=warnings, T0=time.time(), SMOKE=False)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), K.__dict__)
    return K


K = compile_kernel(KF); K26 = compile_kernel(K26)
assert K.PART == 'a' and len(K.OG_STATIC) == 98 and K.OG_STEP == ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3'] and K.OG_ADD_PROD == []
assert K.OG_STATIC == og.static_names() + og.static_names2()
print('kernel compiled | OG_STATIC', len(K.OG_STATIC), '| OG_STEP', K.OG_STEP, '| ALLFIN', K.ALLFIN, 'OCC', K.OCC, '| VARIANTS', list(K.VARIANTS))
NEW = list(K.OG_STATIC) + list(K.OG_STEP)

# ------------------------------------------------------------------ A: lab parity on plain inputs
P = lab.load_panel(panel)
prior = lab.prior_table(P, np.ones(len(P['days']), bool))
for s in ('offlike', 'runstart'):
    rc = lab.rows(panel, s)
    FE = og.FE2(og.build_fe(panel, s), og.build_fe2(panel, s))
    wins = list(zip(rc.win.d.tolist(), rc.win.t.tolist(), rc.win.cond.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        R = K.make_rows(P, wins, prior)
        X, y, e, meta, cols = K.to_matrix(R, lab.ONGOING, None, K.ongoing_keep(R))
    m = rc.keep5[:, 4]
    assert np.array_equal(meta, np.stack([rc.wi[m], rc.li[m], rc.h[m]], 1)), 'row order differs'
    pc = lab.all_cols()
    Xl = og.full_matrix(rc, FE, m, pc, list(K.OG_STATIC), list(K.OG_STEP), 4)
    lc = pc + NEW
    assert set(cols) == set(lc), (set(cols) ^ set(lc))
    ix = [cols.index(c) for c in lc]
    bad = [c for j, c in enumerate(lc) if not np.array_equal(X[:, ix[j]], Xl[:, j], equal_nan=True)]
    print(f'A {panel} {s}: kernel {X.shape} lab {Xl.shape} | identical {not bad} | differing columns {bad}')
    assert not bad
    del R, X, Xl; gc.collect()

# ------------------------------------------------------------------ B: production (all-finite + occupancy) inputs
C = vf.cubes(panel)
PK = dict(P)
PK['Vf'], PK['Ff'] = C['Vall'], C['Fall']
O = np.where(np.isfinite(PK['Vf']), C['O'], np.nan).astype(np.float32)
PK['O'] = O
PK['ocrit'], PK['occ_c'], n_est = K.occ_calib(PK['Vf'], PK['Ff'], O, P['cut'], P['vf'], P['lanes'])
rc = lab.rows(panel, 'offlike')
wins = list(zip(rc.win.d.tolist(), rc.win.t.tolist(), rc.win.cond.tolist()))[:MAXW]
t1 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    R27 = K.make_rows(PK, wins, prior)
    X27, y27, e27, m27, c27 = K.to_matrix(R27, lab.ONGOING, None, K.ongoing_keep(R27))
    R26 = K26.make_rows(PK, wins, prior)
    X26, y26, e26, m26, c26 = K26.to_matrix(R26, lab.ONGOING, None, K26.ongoing_keep(R26))
print(f'B {panel}: {len(wins)} offlike windows | v27 rows {X27.shape} in {time.time() - t1:.1f}s (incl. v26 rows) | v26 rows {X26.shape}')
assert np.array_equal(m27, m26) and np.array_equal(y27, y26) and np.array_equal(e27, e26), 'rows / labels differ from v26'
assert set(c26) <= set(c27) and set(c27) - set(c26) == set(NEW), (set(c27) - set(c26)) ^ set(NEW)
bad = [c for c in c26 if not np.array_equal(X27[:, c27.index(c)], X26[:, c26.index(c)], equal_nan=True)]
print(f'B (1) v26 columns unchanged in v27: {not bad} | differing {bad}')
assert not bad
# (2) independent recomputation of the ogfeat columns on the all-finite inputs (v26 make_rows draw, lab module functions)
d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
idx = t[:, None] + np.arange(-12, 0)[None]
Vh = PK['Vf'][d[:, None], idx]; Fh = PK['Ff'][d[:, None], idx]
rng = np.random.default_rng(len(wins))
VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
VT = np.where(P['E'][d, t], VT, PK['Vf'][d, t])
Fe = {}
K26.add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    X1 = og.og_extra(P, Vh, Fh, VT, Fe); X1.update(og.og_extra2(P, Vh, Fh, VT, Fe))
wi, li, hh = m27[:, 0], m27[:, 1], m27[:, 2]
bad = []
for c in K.OG_STATIC:
    if not np.array_equal(X27[:, c27.index(c)], X1[c][wi, li].astype(np.float32), equal_nan=True):
        bad.append(c)
for h in range(1, 7):
    S = og.og_step(X1, h); r = hh == h
    for c in K.OG_STEP:
        if not np.array_equal(X27[r, c27.index(c)], np.asarray(S[c], np.float32)[wi[r], li[r]], equal_nan=True):
            bad.append(f'{c}@h{h}')
print(f'B (2) ogfeat columns == lab functions on the all-finite inputs: {not bad} | differing {bad[:10]}')
assert not bad
# (3) column sets and finite shares
cn = [x for x in c27 if x not in K.PRIOR_COLS or x in K.OG_ADD_PROD]; ci = [x for x in c27 if x not in K.ID_DROP or x in K.OG_ADD_PROD]
assert all(c in cn and c in ci for c in NEW + K.OCC_COLS) and 'tod' not in cn and 'dow' not in ci
fin = {c: round(float(np.isfinite(X27[:, c27.index(c)]).mean()), 3) for c in NEW}
low = {c: v for c, v in fin.items() if v < 0.5}
print(f'B (3) all {len(c27)} cols | np {len(cn)} | id {len(ci)} | ogfeat finite share min {min(fin.values())} median {np.median(list(fin.values())):.3f} | '
      f'below 0.5: {low} | inf anywhere {bool(np.isinf(X27).any())} | row MB/1M rows {X27.shape[1] * 4:.0f}')
print('KERNEL PARITY + SMOKE OK')
