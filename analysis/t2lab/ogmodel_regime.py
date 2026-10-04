"""TRACK ogmodel S6: WINDOW-REGIME structure for the ongoing model (mid-run vs run-start picks) driven by the pick-spacing
flag g72 (see ogmodel_g72.py): one extra feature column `g72`, everything else = the F03 recipe (same rows, weights,
LightGBM parameters as fc_F03_<cols>255).

Training flag (train split, offline window construction): 1 for the non-run-start faithful windows (mid-run), and 1 for a
random share FLIP of the run-start windows (in the train replays 105 of 2,285 run-start picks = 4.6 % sit exactly at the
end of a spacing block, so that P(run-start | g72) = 13 % as in deployment); 0 for the other run-starts.
Inference flag (deployable): g72 = the previous official window of the same panel (any condition) is exactly 360 min
earlier - from the forecast origins of EARLIER windows in window_index.csv only (timestamps < T).
Evaluation flags: offlike -> g72 of the train replays (results/ogmodel/g72_offlike.pkl); official 40 -> g72 from the
official train window index; 'oracle' -> the lab's non-run-start flag (upper reference, not deployable); 'zero' -> 0.
Pseudo-March: offlike_m with the replay g72 of the March masked layer; runstart_m with flag 0 (they are run-starts).

Usage (through run_heavy): ogmodel_regime.py <cols=np> [wd=0.3] [flip=0.046] [folds=01234] [seed=0] [leaves=255] [lr=0.05] [trees=500] [tag]
       ogmodel_regime.py pmarch <arm> [<arm> ...]          (light: pseudo-March evaluation of saved final models)"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import ogmodel_lib as og

F = lab.F
EV = ('official', 'offlike')
_G = {}
STRUCT = os.environ.get('STRUCT', 'joint')          # 'joint' (one model, default) or any ogmodel_lib.STRUCTS key, e.g. 'qsplit'
SPEC_S = dict(og.STRUCTS[STRUCT])


def eval_flags(panel, set_name, kind):
    """per-window flag aligned with lab.rows(panel, set_name).win."""
    win = lab.rows(panel, set_name, lab.ONGOING).win
    if kind == 'zero':
        return np.zeros(len(win), np.float32)
    if kind == 'oracle':
        return (~win.runstart.astype(bool)).to_numpy().astype(np.float32)
    if set_name not in _G:
        _G[set_name] = pd.read_pickle(f'{og.RES}/g72_{set_name}.pkl')
    g = _G[set_name]; g = g[g.panel == panel]
    col = 'g72_any' if 'g72_any' in g else 'g72'
    mp = {(int(d), int(t)): bool(v) for d, t, v in zip(g.d, g.t, g[col])}
    return np.array([mp[(int(d), int(t))] for d, t in zip(win.d, win.t)], np.float32)


def train_flags(panel, name, win, flip, seed):
    if name == 'faith3':
        return np.ones(len(win), np.float32)
    rng = np.random.default_rng(1000 + 17 * lab.PANELS.index(panel) + seed)
    return (rng.random(len(win)) < flip).astype(np.float32)


def train_matrix(k, cols, wd, flip, seed=0):
    wfn = fc.wfn_F(wd); parts = []; n = 0
    for panel in lab.PANELS:
        for name, sw in og.SPEC['train']:
            rc = lab.rows(panel, name, lab.ONGOING)
            if not rc.n:
                continue
            ww = np.asarray(wfn(rc.win, name, panel), np.float32) * sw
            m = rc.e & rc.keep5[:, k] & (rc.fold != k) & (ww[rc.wi] > 0)
            c = int(m.sum()); parts.append((rc, m, ww, train_flags(panel, name, rc.win, flip, seed), c)); n += c
    X = np.empty((n, len(cols) + 1), np.float32); y = np.empty(n, np.float32); w = np.empty(n, np.float32); o = 0
    for rc, m, ww, fl, c in parts:
        X[o:o + c, :-1] = rc.matrix(m, cols, k); X[o:o + c, -1] = fl[rc.wi[m]]; y[o:o + c] = rc.y[m]; w[o:o + c] = ww[rc.wi[m]]; o += c
    return X, y, w


def predict(mdl, rc, k, cols, flagw):
    P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
    wmask = rc.win.fold.to_numpy() == k
    m = rc.keep5[:, k] & wmask[rc.wi]
    if m.any():
        X = np.empty((int(m.sum()), len(cols) + 1), np.float32)
        X[:, :-1] = rc.matrix(m, cols, k); X[:, -1] = flagw[rc.wi[m]]
        P6[rc.wi[m], rc.h[m] - 1, rc.li[m]] = og.predict_struct(SPEC_S, mdl, X, cols + ['g72'])
    return P6


def run(cols='np', wd=0.3, flip=0.046, folds=(0, 1, 2, 3, 4), seed=0, leaves=255, lr=0.05, n_trees=500, tag=''):
    cols_ = lab.colset(cols)
    base = f'og_g72feat_{cols}{leaves}' + (f'_{STRUCT}' if STRUCT != 'joint' else '') + (f'_wd{wd:g}' if wd != 0.3 else '') + (f'_flip{flip:g}' if flip != 0.046 else '') + (f'_s{seed}' if seed else '') + tag
    kinds = ('g72', 'oracle', 'zero')
    params = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=seed)
    for k in folds:
        if k == 4 and os.path.exists(f'{og.MODELS}/{base}.json'):
            continue
        if k < 4 and all(os.path.exists(f'{og.arm_dir(base + "__" + kd)}/fold{k}.pkl') for kd in kinds):
            lab.log(f'{base} fold {k}: found, skipped'); continue
        t1 = time.time()
        X, y, w = train_matrix(k, cols_, wd, flip, seed)
        nrow = len(X); fsh = float(X[:, -1].mean()); fw = float((X[:, -1] * w).sum() / w.sum())
        mdl = og.fit_struct(SPEC_S, X, y, w, cols_ + ['g72'], params, n_trees)
        del X, y, w; gc.collect()
        imp = dict(zip(cols_ + ['g72'], sum(m_.feature_importance('gain') for _, m_ in mdl))); tot = sum(imp.values())
        rank = sorted(imp, key=imp.get, reverse=True).index('g72') + 1
        lab.log(f'{base} fold {k}: {nrow} rows, flag share {fsh:.3f} (weighted {fw:.3f}), {time.time() - t1:.0f}s | g72 gain share {imp["g72"] / tot:.4f} (rank {rank}/{len(imp)})')
        if k == 4:
            og.save_models(base, dict(SPEC_S, extra='g72'), mdl, cols_ + ['g72'], dict(leaves=leaves, lr=lr, n_trees=n_trees, seed=seed, wd=wd, flip=flip, nrow=nrow)); continue
        for kd in kinds:
            res = []; probs = {}
            for s in EV:
                for panel in lab.PANELS:
                    rc = lab.rows(panel, s, lab.ONGOING)
                    widx = np.where(rc.win.fold.to_numpy() == k)[0]
                    if not len(widx):
                        continue
                    P6 = predict(mdl, rc, k, cols_, eval_flags(panel, s, kd))
                    res.append(lab.score_probs(rc, P6, widx, arm=f'{base}__{kd}', set_name=s))
                    probs[(s, panel)] = (widx, P6[widx].astype(np.float16))
            df = pd.concat(res, ignore_index=True)
            os.makedirs(og.arm_dir(f'{base}__{kd}'), exist_ok=True)
            pd.to_pickle(dict(df=df, probs=probs, nrow=nrow), f'{og.arm_dir(base + "__" + kd)}/fold{k}.pkl')
            lab.log(f'{base}__{kd} fold {k}: ' + ', '.join(f'{s} {lab.aggregate(df[df.set == s]):.4f} (n={int((df.set == s).sum())})' for s in EV))
        del mdl; gc.collect()
    ref = lab.load_arm(f'fc_F03_{cols}255')
    rows = []
    for kd in kinds:
        df = og.merge(f'{base}__{kd}')
        if not len(df):
            continue
        for s in EV:
            rows.append(og.delta_row(ref[ref.set == s], df[df.set == s], f'{base}__{kd}', s))
            if s == 'offlike':
                fl = pd.concat([pd.DataFrame(dict(panel=p, d=lab.rows(p, s).win.d, t=lab.rows(p, s).win.t, g=eval_flags(p, s, 'g72'))) for p in lab.PANELS])
                a = ref[ref.set == s].merge(fl, on=['panel', 'd', 't']); b = df[df.set == s].merge(fl, on=['panel', 'd', 't'])
                for nm, m in (('g72=1', 1.0), ('g72=0', 0.0)):
                    r = og.delta_row(a[a.g == m], b[b.g == m], f'{base}__{kd}', f'{s} [{nm}]'); rows.append(r)
    T = pd.DataFrame(rows)
    print(f'\n== {base}: paired vs fc_F03_{cols}255 (same windows, OOF) =='); print(T.round(4).to_string(index=False), flush=True)
    T.to_csv(f'{og.RES}/s6_{base}_deltas.csv', index=False)


PMS = og.PM_SETS + ('midrun3_m',)


def pmarch(arms):
    """pseudo-March evaluation of saved g72 models against the reference final F03 model(s).
    offlike_m: replay g72 flags (deployable); runstart_m: flag 0; midrun3_m (mid-run windows): flag 1 (regime known by construction)."""
    import lightgbm as lgb
    M = pd.read_pickle(f'{og.RES}/g72_pmarch.pkl')
    res = []
    for a in arms:
        spec, models, cols = og.load_models(a)
        ckind = 'id' if '_id' in a else 'np'
        ref = lgb.Booster(model_file=f'{fc.RES}/final/model_F03_{ckind}255_500.txt'); rcols = lab.colset(ckind)
        for panel in lab.PANELS:
            for sname in PMS:
                pm = og.pm_load(panel, sname, 'T')
                if pm is None:
                    continue
                if sname == 'offlike_m':
                    g = M[M.panel == panel]; mp = {(int(d), int(t)): bool(v) for d, t, v in zip(g.d, g.t, g.g72_any)}
                    fw = np.array([mp[(int(d), int(t))] for d, t in zip(pm.rc.win.d, pm.rc.win.t)], np.float32)
                elif sname == 'midrun3_m':                    # mid-run windows: the regime the flag stands for
                    fw = np.ones(pm.W, np.float32)
                else:
                    fw = np.zeros(pm.W, np.float32)
                ix = [pm.cols.index(c) for c in cols[:-1]]
                for kd, f in (('g72', fw), ('zero', np.zeros(pm.W, np.float32))):
                    X = np.empty((len(pm.X), len(cols)), np.float32); X[:, :-1] = pm.X[:, ix]; X[:, -1] = f[pm.meta[:, 0]]
                    P6 = np.zeros((pm.W, F, pm.L), np.float32); P6[pm.meta[:, 0], pm.meta[:, 2] - 1, pm.meta[:, 1]] = og.predict_struct(spec, models, X, cols)
                    df = lab.score_probs(pm.rc, P6, arm=f'{a}__{kd}', set_name=sname); df['g72'] = fw; res.append(df)
                P6 = og.pm_predict(pm, lambda X: ref.predict(X, num_threads=lab.THREADS), rcols)
                df = lab.score_probs(pm.rc, P6, arm=f'ref_{ckind}|{a}', set_name=sname); df['g72'] = fw; res.append(df)
    R = pd.concat(res, ignore_index=True); R.to_pickle(f'{og.RES}/s6_pmarch_{"_".join(arms)[:100]}.pkl')
    rows = []
    for a in arms:
        ckind = 'id' if '_id' in a else 'np'
        for sname in PMS:
            A = R[(R.set == sname) & (R.arm == f'ref_{ckind}|{a}')]
            for kd in ('g72', 'zero'):
                B = R[(R.set == sname) & (R.arm == f'{a}__{kd}')]
                rows.append(og.delta_row(A, B, f'{a}__{kd}', sname))
                if sname == 'offlike_m' and kd == 'g72':
                    for nm, m in (('g72=1', 1.0), ('g72=0', 0.0)):
                        rows.append(og.delta_row(A[A.g72 == m], B[B.g72 == m], f'{a}__{kd}', f'{sname} [{nm}]'))
    T = pd.DataFrame(rows); print('== PSEUDO-MARCH (masked histories, visible labels; train-only models) =='); print(T.round(4).to_string(index=False))
    T.to_csv(f'{og.RES}/s6_pmarch_{"_".join(arms)[:100]}_deltas.csv', index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
    a = sys.argv[1:]
    if a and a[0] == 'pmarch':
        pmarch(a[1:])
    else:
        run(cols=a[0] if a else 'np', wd=float(a[1]) if len(a) > 1 else 0.3, flip=float(a[2]) if len(a) > 2 else 0.046,
            folds=tuple(int(c) for c in a[3]) if len(a) > 3 else (0, 1, 2, 3, 4), seed=int(a[4]) if len(a) > 4 else 0,
            leaves=int(a[5]) if len(a) > 5 else 255, lr=float(a[6]) if len(a) > 6 else 0.05, n_trees=int(a[7]) if len(a) > 7 else 500,
            tag=('_' + a[8]) if len(a) > 8 else '')
    lab.log('DONE')
