"""ogfeat: function-level parity of the generated kernel with the lab (no Kaggle paths are touched: only the kernel's function
definitions and constants are executed, as lab.py does for the production file).
Checks on one panel: kernel make_rows + to_matrix (train rows) == lab row cache production columns + lab extra columns."""
import ast, sys, types, glob, os, time, gc, json, warnings
import numpy as np, pandas as pd
import lab, ogfeat_lib as og

KF = f'{lab.ROOT}/kaggle_jobs/t2_ogfeat_v23.py'
panel = sys.argv[1] if len(sys.argv) > 1 else 'D7_I10_W'
src = open(KF).read(); tree = ast.parse(src)
body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name != 'main':
        body.append(node)
    elif isinstance(node, ast.Assign):
        tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        if tg and tg <= {'H', 'F', 'SPACING', 'PARAMS', 'PRIOR_COLS', 'ID_DROP', 'OG_STATIC', 'OG_STEP', 'OG_ADD_PROD', 'CAPD', 'BIG', 'OG_STEP_ALL', 'ONGOING', 'VARIANTS'}:
            body.append(node)
K = types.ModuleType('kern'); K.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, T0=time.time(), SMOKE=False)
exec(compile(ast.Module(body=body, type_ignores=[]), KF, 'exec'), K.__dict__)
print('kernel functions:', len(body), '| OG_STATIC', len(K.OG_STATIC), 'OG_STEP', K.OG_STEP, 'OG_ADD_PROD', K.OG_ADD_PROD)
P = lab.load_panel(panel)
for s in ('offlike', 'runstart'):
    rc = lab.rows(panel, s); FE = og.build_fe(panel, s)
    wins = list(zip(rc.win.d.tolist(), rc.win.t.tolist(), rc.win.cond.tolist()))
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        R = K.make_rows(P, wins, prior)
        X, y, e, meta, cols = K.to_matrix(R, lab.ONGOING, None, K.ongoing_keep(R))
    m = rc.keep5[:, 4]
    assert np.array_equal(meta, np.stack([rc.wi[m], rc.li[m], rc.h[m]], 1)), 'row order differs'
    pc = lab.all_cols()
    Xl = og.full_matrix(rc, FE, m, pc, list(K.OG_STATIC), list(K.OG_STEP), 4)
    lc = pc + list(K.OG_STATIC) + list(K.OG_STEP)
    assert set(cols) == set(lc), (set(cols) ^ set(lc))
    ix = [cols.index(c) for c in lc]
    same = np.array_equal(X[:, ix], Xl, equal_nan=True)
    bad = [c for j, c in enumerate(lc) if not np.array_equal(X[:, ix[j]], Xl[:, j], equal_nan=True)]
    print(f'{panel} {s}: kernel matrix {X.shape}, lab matrix {Xl.shape}, identical {same}, differing columns {bad}')
    assert same
cn = [x for x in cols if x not in K.PRIOR_COLS or x in K.OG_ADD_PROD]; ci = [x for x in cols if x not in K.ID_DROP or x in K.OG_ADD_PROD]
print('kernel np columns', len(cn), 'id columns', len(ci))
print('KERNEL PARITY OK')
