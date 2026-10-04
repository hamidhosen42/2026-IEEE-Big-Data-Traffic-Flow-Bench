"""ogfeat runner: OOF folds (F03 recipe, lab scale) + pseudo-March scoring for one or several feature arms.

Usage (through run_heavy):
    ogfeat_run.py <base: np|id> <arm=group+group[,arm=...]> [folds=01] [final=0|1|load] [leaves=255] [lr=0.05] [trees=500] [seed=0]
      arm 'base' (no '=') = production columns only.   groups: g1a (tod, dow), tod, g1b, g2, g2s, g3, cols:a+b+c
      final=1    also train the k=4 model (all train days) and score pseudo-March with it
      final=load score pseudo-March with results/firstcross/final/model_F03_<base>255_500.txt (baseline arm only)
Per arm the cache dir data/t2local/cache/ogfeat/arms/og_<base>_<arm>[_s<seed>]/ holds fold<k>.pkl (per-window frame + probabilities),
model_f<k>.txt, pm_f<k>.pkl (pseudo-March probabilities of the fold-k / final model) so later reports can pair any two arms.
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import ogfeat_lib as og

base = sys.argv[1]
arm_specs = sys.argv[2].split(',')
kw = dict(a.split('=', 1) for a in sys.argv[3:])
folds = tuple(int(c) for c in kw.get('folds', '01'))
final = kw.get('final', '0')
leaves = int(kw.get('leaves', 255)); lr = float(kw.get('lr', 0.05)); n_trees = int(kw.get('trees', 500)); seed = int(kw.get('seed', 0))
EVALS = ('official', 'offlike', 'pseudo')
SMOKE = bool(os.environ.get('OGF_SMOKE'))
if SMOKE:                                            # plumbing test: 2 small panels, 20 trees, separate arm names
    lab.PANELS[:] = ['D7_I10_W', 'D7_I210_E']; n_trees = 20
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)


def arm_dir(name):
    d = f'{og.ODIR}/arms/{name}'; os.makedirs(d, exist_ok=True)
    return d


def pm_probs(model, pc, st, sp):
    """pseudo-March probabilities of one model: {(panel, set): P6 float16}."""
    out = {}; allc = lab.all_cols(); ix = [allc.index(c) for c in pc]
    for panel in lab.PANELS:
        pm = og.build_pm(panel)
        for s in og.PM_SETS:
            z = pm[s]; meta = z['meta']
            if not len(z['win']):
                continue
            X = z['X'][:, ix]
            if st or sp:
                X = np.concatenate([X, og.extra_matrix(og.pm_fe(panel, s, z, st), meta[:, 0], meta[:, 1], meta[:, 2], st, sp)], 1)
            P6 = np.zeros((len(z['win']), lab.F, z['qlast'].shape[1]), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = model.predict(X, num_threads=lab.THREADS)
            out[(panel, s)] = P6.astype(np.float16)
    return out


for spec in arm_specs:
    aname, groups = (spec.split('=', 1) + [''])[:2] if '=' in spec else (spec, '')
    groups = [g for g in groups.split('+') if g] if not groups.startswith('cols:') else [groups]
    pc, st, sp = og.spec_cols(base, groups)
    cols = pc + st + sp
    name = ('smoke_' if SMOKE else '') + f'og_{base}_{aname}' + (f'_s{seed}' if seed else '') + (f'_L{leaves}' if leaves != 255 else '')
    odir = arm_dir(name)
    json.dump(dict(base=base, groups=groups, prod_cols=pc, static=st, step=sp, leaves=leaves, lr=lr, n_trees=n_trees, seed=seed),
              open(f'{odir}/config.json', 'w'), indent=1)
    lab.log(f'=== {name}: {len(pc)} production + {len(st)} static + {len(sp)} step columns; folds {folds}, final {final}')
    params = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=seed)
    for k in list(folds) + ([4] if final != '0' else []):
        ff = f'{odir}/fold{k}.pkl'; mf = f'{odir}/model_f{k}.txt'; pf = f'{odir}/pm_f{k}.pkl'
        if os.path.exists(pf) and (k == 4 or os.path.exists(ff)):
            lab.log(f'{name} fold {k}: found on disk, skipped'); continue
        t1 = time.time()
        stored = f'{fc.RES}/final/model_F03_{base}255_500.txt'
        if (k == 4 and aname == 'base' and seed == 0 and leaves == 255 and n_trees == 500 and lr == 0.05 and os.path.exists(stored)
                and not SMOKE):
            mdl = lgb.Booster(model_file=stored)                        # baseline final model of exp_firstcross_final.py (same rows / params)
            assert mdl.feature_name() == cols, 'column order of the stored final model differs'
            lab.log(f'{name} final: loaded the stored F03 {base} final model')
        elif os.path.exists(mf):
            mdl = lgb.Booster(model_file=mf)
        else:
            X, y, w = og.train_matrix(k, pc, st, sp)
            nrow = len(X)
            mdl = fc.fit(X, y, w, params, n_trees, cols)
            del X, y, w; gc.collect()
            mdl.save_model(mf)
            lab.log(f'{name} fold {k}: {nrow} rows x {len(cols)} cols trained in {time.time() - t1:.0f}s')
        if k != 4:
            res = []; probs = {}
            for s in EVALS:
                for panel in lab.PANELS:
                    rc = lab.rows(panel, s, lab.ONGOING)
                    widx = np.where(rc.win.fold.to_numpy() == k)[0]
                    if not len(widx):
                        continue
                    rc, P6 = og.predict_set(mdl, panel, s, k, pc, st, sp)
                    res.append(lab.score_probs(rc, P6, widx, arm=name, set_name=s))
                    probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
            df = pd.concat(res, ignore_index=True)
            pd.to_pickle(dict(df=df, probs=probs), ff)
            lab.log(f'{name} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in EVALS))
        pd.to_pickle(pm_probs(mdl, pc, st, sp), pf)
        if k == 4 or True:
            imp = pd.Series(mdl.feature_importance('gain'), index=cols).sort_values(ascending=False)
            (imp / imp.sum()).to_csv(f'{odir}/gain_f{k}.csv')
        del mdl; gc.collect()
    lab.log(f'{name} done')
lab.log('DONE')
