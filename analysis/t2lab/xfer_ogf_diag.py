"""TRANSFER analysis 2026-10-08: ogfeat (all2) vs base on PRODUCTION inputs (all-finite + occupancy), with probabilities saved.
Same rows / weights / params as ogfeat_overlap_v27.py (kernel functions compiled from kaggle_jobs/t2_ogfeat_v27a.py, F03 recipe,
held-out fold k), but the per-window probability fields of both arms are stored so that the cell-level footprint (added / removed
cells, inside / outside the queue at T, by step h) can be compared with the March footprint of tfb-h4 vs tfb-h1.
Evaluation sets: official-40 + offlike + runstart windows of fold k.
Usage (run_heavy): xfer_ogf_diag.py [fold=0] [trees=500] [leaves=255]
Output: results/xfer/ogf_diag_f<k>_L<leaves>.pkl  {'df': per-window frame (both arms), 'P': {(arm, panel, set): (widx, P6 float16)},
                                          'aux': {(panel, set): (widx, truth, ev, qlast)}}"""
import ast, gc, glob, json, os, sys, time, types, warnings
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import exp_firstcross_lib as fc
import verify_occ_feat as vf

CONSTS = {'H', 'F', 'SPACING', 'PARAMS', 'PRIOR_COLS', 'ID_DROP', 'OG_STATIC', 'OG_STEP', 'OG_ADD_PROD', 'CAPD', 'BIG', 'OG_STEP_ALL',
          'ONGOING', 'VARIANTS', 'NB', 'OCC_COLS', 'ALLFIN', 'OCC', 'PART', 'TAG', 'W_RS', 'W_DENSE', 'FILL_GAP', 'PERS_MAX', 'MAX_OG',
          'N_BIG', 'TIME_LIMIT_S'}
RES = f'{lab.ROOT}/analysis/t2lab/results/xfer'
os.makedirs(RES, exist_ok=True)


def compile_kernel(path):                                  # = ogfeat_overlap_v27.compile_kernel
    tree = ast.parse(open(path).read()); body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            body.append(node)
        elif isinstance(node, ast.Assign):
            tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if tg and tg <= CONSTS:
                body.append(node)
    Km = types.ModuleType(os.path.basename(path)[:-3])
    Km.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, warnings=warnings, T0=time.time(), SMOKE=False)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), Km.__dict__)
    return Km


kw = dict(a.split('=') for a in sys.argv[1:])
k = int(kw.get('fold', 0)); n_trees = int(kw.get('trees', 500)); leaves = int(kw.get('leaves', 255))
EVS = ('official', 'offlike', 'runstart')
K = compile_kernel(f'{lab.ROOT}/kaggle_jobs/t2_ogfeat_v27a.py')
NEW = list(K.OG_STATIC) + list(K.OG_STEP)
spec = fc.ARMS['F03']
T0 = time.time()
TR = []; EV = {}; AUX = {}
cols = None
for panel in lab.PANELS:
    P = lab.load_panel(panel); C = vf.cubes(panel)
    PK = dict(P); PK['Vf'], PK['Ff'] = C['Vall'], C['Fall']
    PK['O'] = np.where(np.isfinite(PK['Vf']), C['O'], np.nan).astype(np.float32)
    PK['ocrit'], PK['occ_c'], _ = K.occ_calib(PK['Vf'], PK['Ff'], PK['O'], P['cut'], P['vf'], P['lanes'])
    prior = lab.prior_table(P, P['block'] != k)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, sw in spec['train']:
            rc = lab.rows(panel, name)
            if not rc.n:
                continue
            ww = np.full(len(rc.win), float(sw), np.float32) * np.asarray(spec['wfn'](rc.win, name, panel), np.float32)
            sel = np.where((rc.win.fold.to_numpy() != k) & (ww > 0))[0]
            if not len(sel):
                continue
            wins = list(zip(rc.win.d.to_numpy()[sel].tolist(), rc.win.t.to_numpy()[sel].tolist(), [lab.ONGOING] * len(sel)))
            R = K.make_rows(PK, wins, prior)
            X, y, e, meta, cols = K.to_matrix(R, lab.ONGOING, cols, K.ongoing_keep(R))
            TR.append((X[e], y[e].astype(np.float32), ww[sel][meta[:, 0]][e])); del R, X; gc.collect()
        for s in EVS:
            rc = lab.rows(panel, s)
            widx = np.where(rc.win.fold.to_numpy() == k)[0]
            if not len(widx):
                continue
            wins = list(zip(rc.win.d.to_numpy()[widx].tolist(), rc.win.t.to_numpy()[widx].tolist(), [lab.ONGOING] * len(widx)))
            R = K.make_rows(PK, wins, prior)
            X, _, _, meta, _ = K.to_matrix(R, lab.ONGOING, cols, K.ongoing_keep(R))
            EV[(panel, s)] = (rc, widx, X, meta)
            AUX[(panel, s)] = (widx, rc.truth[widx], rc.ev[widx], rc.qlast[widx], rc.win.iloc[widx].reset_index(drop=True))
            del R; gc.collect()
    lab.log(panel, 'rows built', sum(len(t[1]) for t in TR), f'{time.time() - T0:.0f}s')
    lab.drop_panel(panel)
X = np.concatenate([t[0] for t in TR]); y = np.concatenate([t[1] for t in TR]); w = np.concatenate([t[2] for t in TR]); del TR; gc.collect()
base_cols = [c for c in cols if c not in K.PRIOR_COLS and c not in NEW]
new_cols = base_cols + NEW
lab.log(f'train rows {len(y)} pos {y.mean():.4f} | base {len(base_cols)} cols | new {len(new_cols)} cols')
params = lab.lgb_params(num_leaves=leaves, learning_rate=0.05, min_data_in_leaf=40)
res = []; PS = {}
for arm, cs in (('base_afocc', base_cols), ('all2_afocc', new_cols)):
    ix = [cols.index(c) for c in cs]
    t1 = time.time(); mdl = fc.fit(X[:, ix], y, w, params, n_trees, cs); lab.log(arm, f'trained in {time.time() - t1:.0f}s')
    mdl.save_model(f'{RES}/model_{arm}_f{k}_L{leaves}.txt')
    for (panel, s), (rc, widx, Xe, meta) in EV.items():
        P6 = np.zeros((len(rc.win), lab.F, rc.qlast.shape[1]), np.float32)
        P6[widx[meta[:, 0]], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(Xe[:, ix], num_threads=lab.THREADS)
        res.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
        PS[(arm, panel, s)] = (widx, P6[widx].astype(np.float16))
    del mdl; gc.collect()
D = pd.concat(res, ignore_index=True)
for s in EVS:
    A = D[(D.arm == 'base_afocc') & (D.set == s)]; B = D[(D.arm == 'all2_afocc') & (D.set == s)]
    print(f'{s}: n {len(A)} | base {lab.aggregate(A):.4f} | new {lab.aggregate(B):.4f} | paired', lab.paired_bootstrap(A, B), flush=True)
pd.to_pickle(dict(df=D, P=PS, aux=AUX, base_cols=base_cols, new_cols=new_cols), f'{RES}/ogf_diag_f{k}_L{leaves}.pkl')
lab.log('XFER OGF DIAG DONE')
