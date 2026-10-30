"""DIAGNOSTIC - NOT FOR SUBMISSION (analysis/LEAK_DIAG_2026-10-29.md).  Uses post-T data on purpose (see leakdiag_lib).

IoU of the official-like train sets (offlike, official) with and without the post-T leak, from the cell tables written by
leakdiag_build.py.  Variants (same windows, same production decode):
  base        production-like OOF probabilities (onset CatBoost NORAMP, ongoing F03)
  R_self      simple leak rule: p = max(p, 0.9) where the link's own on-ramp is switched off at T+h
  R_spread    R_self + p = max(p, 0.6) where a switched-off on-ramp link is within +-1 link, max(p, 0.4) within +-2
  R_map       p = max(p, P(Q_k | switched-off ramp a)) with the ramp -> link table learned on the training folds
  S_ctrl      LightGBM stacker on (p, h, qlast) only: the no-leak control for any stacking / recalibration effect
  S_ramp      stacker + ramp leak features             (cross-fitted by day-block fold; official uses the fold-0 model)
  S_pb        stacker + post-buffer mainline features (T+95..T+105, thinned to the val/private visibility)
  S_both      stacker + both
Run:  .venv/bin/python analysis/t2lab/run_heavy.py -- .venv/bin/python analysis/t2lab/leakdiag_eval.py
"""
import gc, json, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

import lab
import leakdiag_lib as ld

F, ONSET, ONGOING = lab.F, lab.ONSET, lab.ONGOING
CELLS = f'{lab.CACHE}/leakdiag'
RES = f'{lab.ROOT}/analysis/t2lab/results/leakdiag'
os.makedirs(RES, exist_ok=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)
RI = {n: i for i, n in enumerate(ld.RAMP_FEATS)}
STK = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=lab.THREADS, seed=0, deterministic=True, force_row_wise=True)
N_TREES = 300
VARIANTS = {'S_ctrl': (False, False), 'S_ramp': (True, False), 'S_pb': (False, True), 'S_both': (True, True)}
# v2 stackers boost a RESIDUAL on the base-model logit (init_score): where the extra features are silent the base ranking is kept.
# (v1 = plain trees on p, logs/leakdiag_*_v1_plainstack.log: piecewise-constant p created ties that the top-m decode broke
#  arbitrarily, costing onset 0.012 even in the no-leak control.)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_stacker(X, y, names):
    return lgb.train(STK, lgb.Dataset(X, y.astype(np.float32), init_score=_logit(X[:, 0]), feature_name=names, free_raw_data=True), N_TREES)


def apply_stacker(model, X):
    z = model.predict(X, raw_score=True, num_threads=lab.THREADS) + _logit(X[:, 0])
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def load(set_name, cond):
    out = []
    for panel in lab.PANELS:
        f = f'{CELLS}/cells_{panel}__{set_name}__{cond}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f)
        it = {k: z[k] for k in z.files}
        it['FR'] = it['FR'].astype(np.float32); it['FPB'] = it['FPB'].astype(np.float32)
        it['panel'] = panel; it['win'] = pd.read_pickle(f.replace('.npz', '.pkl'))
        out.append(it)
    return out


def steps_of(cond):
    return [F - 1] if cond == ONSET else list(range(F))


def design(it, cond, ramp, pb):
    """per-cell design matrix over the predicted steps: columns p, h, qlast (+ ramp features) (+ post-buffer features)."""
    st = steps_of(cond); W, _, L = it['P6'].shape
    p = it['P6'][:, st]; S = len(st)
    cols = [p[..., None], np.broadcast_to(np.array(st, np.float32)[None, :, None, None] + 1, (W, S, L, 1)),
            np.broadcast_to(it['qlast'][:, None, :, None].astype(np.float32), (W, S, L, 1))]
    names = ['p', 'h', 'qlast']
    if ramp:
        cols.append(it['FR'][:, st]); names += ld.RAMP_FEATS
    if pb:
        cols.append(it['FPB'][:, st]); names += ld.PB_FEATS
    X = np.concatenate(cols, -1).reshape(-1, len(names)).astype(np.float32)
    y = it['truth'][:, st].reshape(-1); e = it['ev'][:, st].reshape(-1)
    fold = np.broadcast_to(it['fold'][:, None, None], (W, S, L)).reshape(-1)
    return X, y, e, fold, names


def put(it, cond, pflat):
    st = steps_of(cond); W, _, L = it['P6'].shape
    P6 = np.zeros_like(it['P6']); P6[:, st] = pflat.reshape(W, len(st), L)
    return P6


def score(it, P6, cond, set_name, arm):
    return ld.score_frame(it['truth'], it['ev'], it['qlast'], it['win'], P6, it['panel'], cond, set_name, arm)


def rule_variants(it, cond):
    st = steps_of(cond); P6 = it['P6']; FR = it['FR']
    out = {}
    sf = FR[..., RI['self_flag']] > 0
    m = np.zeros_like(P6, bool); m[:, st] = sf[:, st]
    out['R_self'] = np.where(m, np.maximum(P6, 0.9), P6)
    n1 = np.zeros_like(m); n1[:, st] = (FR[:, st][..., RI['nb1']] > 0)
    n2 = np.zeros_like(m); n2[:, st] = (FR[:, st][..., RI['nb2']] > 0)
    q = np.where(n2, np.maximum(P6, 0.4), P6); q = np.where(n1, np.maximum(q, 0.6), q)
    out['R_spread'] = np.where(m, np.maximum(q, 0.9), q)
    lp = np.zeros_like(P6); lp[:, st] = FR[:, st][..., RI['lk_pos']]
    out['R_map'] = np.maximum(P6, lp)
    return out


def run_condition(cond, train_set='offlike', eval_sets=('offlike', 'official')):
    data = {s: load(s, cond) for s in set(eval_sets) | {train_set}}
    res = []
    for s in eval_sets:
        for it in data[s]:
            res.append(score(it, it['P6'], cond, s, 'base'))
            for nm, P6 in rule_variants(it, cond).items():
                res.append(score(it, P6, cond, s, nm))
    imp = {}
    for var, (ramp, pb) in VARIANTS.items():
        t1 = time.time()
        D = [design(it, cond, ramp, pb) for it in data[train_set]]
        names = D[0][4]
        X = np.concatenate([d[0] for d in D]); y = np.concatenate([d[1] for d in D]); e = np.concatenate([d[2] for d in D])
        fo = np.concatenate([d[3] for d in D]); del D
        models = {}
        for k in range(4):
            m = e & (fo != k)
            models[k] = fit_stacker(X[m], y[m], names)
        del X, y, e, fo; gc.collect()
        imp[var] = pd.Series(models[0].feature_importance('gain'), index=names)
        for s in eval_sets:
            for it in data[s]:
                Xs, _, _, fos, _ = design(it, cond, ramp, pb)
                p = np.zeros(len(Xs), np.float32)
                for k in np.unique(fos):
                    mk = fos == k
                    p[mk] = apply_stacker(models[int(k)], Xs[mk])
                res.append(score(it, put(it, cond, p), cond, s, var))
        lab.log(f'{cond} {var}: {time.time() - t1:.0f}s')
    return pd.concat(res, ignore_index=True), imp, data


def reveal_onset(data, s):
    """does the ramp drop (or the post-buffer state) reveal the T+30 onset location?"""
    rows = []
    for it in data[s]:
        A = set(it['A'].tolist())
        T6 = it['truth'][:, F - 1] & it['ev'][:, F - 1]; FR = it['FR'][:, F - 1]; PB = it['FPB'][:, F - 1]
        L = T6.shape[1]; isA = np.zeros(L, bool); isA[list(A)] = True
        for w in range(len(T6)):
            tr = T6[w]
            if not tr.any():
                continue
            sf = FR[w, :, RI['self_flag']] > 0; ok_ = FR[w, :, RI['self_ok']] > 0
            nb = (FR[w, :, RI['nb2']] > 0) | sf; lp = FR[w, :, RI['lk_pos']]
            pbq = PB[w, :, ld.PB_FEATS.index('pb_any')] > 0
            rows.append(dict(panel=it['panel'], n_true=int(tr.sum()), on_ramp_link=bool((tr & isA).any()), ramp_usable=bool((tr & ok_).any()),
                             self_flag=bool((tr & sf).any()), flag_within2=bool((tr & nb).any()), map_reveal=bool((tr & (lp > 0.5)).any()),
                             cells_self=float((tr & sf).sum() / tr.sum()), cells_within2=float((tr & nb).sum() / tr.sum()),
                             n_flag=int(sf.sum()), n_flag_true=int((sf & tr).sum()),
                             n_flag_ev=int((sf & it['ev'][w, F - 1]).sum()),
                             pb_reveal=bool((tr & pbq).any()), cells_pb=float((tr & pbq).sum() / tr.sum())))
    return pd.DataFrame(rows)


def reveal_ongoing(data, s):
    rows = []
    for it in data[s]:
        tr = it['truth'] & it['ev']; ql = it['qlast'][:, None, :]
        sf = it['FR'][..., RI['self_flag']] > 0
        new = tr & ~ql
        for h in range(F):
            rows.append(dict(panel=it['panel'], h=h + 1, true=int(tr[:, h].sum()), true_flag=int((tr[:, h] & sf[:, h]).sum()),
                             new=int(new[:, h].sum()), new_flag=int((new[:, h] & sf[:, h]).sum()),
                             flag_ev=int((sf[:, h] & it['ev'][:, h]).sum()), flag_true=int((sf[:, h] & tr[:, h]).sum())))
    return pd.DataFrame(rows)


def summarize(df, cond):
    out = []
    for s in dict.fromkeys(df.set):
        g = df[(df.set == s)]
        base = g[g.arm == 'base']
        for arm in dict.fromkeys(g.arm):
            a = g[g.arm == arm]
            r = dict(cond=cond, set=s, arm=arm, n=len(a), iou=lab.aggregate(a), d_base=lab.aggregate(a) - lab.aggregate(base))
            if arm != 'base':
                bt = lab.paired_bootstrap(base[['panel', 'd', 't', 'cond', 'family', 'iou']], a[['panel', 'd', 't', 'cond', 'family', 'iou']], n=500)
                r.update(lo=bt['lo'], hi=bt['hi'])
            for h in range(1, F + 1):
                r[f's{h}'] = lab.aggregate(a, f's{h}')
            out.append(r)
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    allres, sums, imps = [], [], {}
    for cond in (ONSET, ONGOING):
        df, imp, data = run_condition(cond)
        df.to_pickle(f'{RES}/eval_{cond}.pkl'); allres.append(df); imps[cond] = imp
        S = summarize(df, cond); sums.append(S)
        print(f'\n== {cond}: IoU (window -> panel -> family -> mean), delta vs base with paired-bootstrap 95% CI ==')
        print(S.round(4).to_string(index=False))
        print(f'\n-- {cond}: stacker gain importance (fold-0 model) --')
        print(pd.DataFrame(imp).round(0).to_string())
        if cond == ONSET:
            for s in ('offlike', 'official'):
                R = reveal_onset(data, s); R.to_csv(f'{RES}/reveal_onset_{s}.csv', index=False)
                print(f'\n-- onset location reveal [{s}] ({len(R)} windows with a visible T+30 queue) --')
                print('share of windows:', R[['on_ramp_link', 'ramp_usable', 'self_flag', 'flag_within2', 'map_reveal', 'pb_reveal']].mean().round(3).to_dict())
                print('mean share of true T+30 cells: own ramp switched off', round(R.cells_self.mean(), 3), '| switched-off ramp within +-2', round(R.cells_within2.mean(), 3),
                      '| post-buffer queued', round(R.cells_pb.mean(), 3))
                print('precision of own-ramp flags at T+30 in onset windows:', round(R.n_flag_true.sum() / max(R.n_flag_ev.sum(), 1), 3),
                      f'({R.n_flag_true.sum()} / {R.n_flag_ev.sum()}); mean true cells per window {R.n_true.mean():.2f}')
                print('by panel:'); print(R.groupby('panel')[['on_ramp_link', 'ramp_usable', 'self_flag', 'flag_within2', 'map_reveal', 'pb_reveal']].mean().round(2).to_string())
        else:
            for s in ('offlike',):
                R = reveal_ongoing(data, s); R.to_csv(f'{RES}/reveal_ongoing_{s}.csv', index=False)
                g = R.groupby('h')[['true', 'true_flag', 'new', 'new_flag', 'flag_ev', 'flag_true']].sum()
                g['share_true_flagged'] = g.true_flag / g.true; g['share_new_flagged'] = g.new_flag / g.new.clip(lower=1); g['flag_precision'] = g.flag_true / g.flag_ev.clip(lower=1)
                print(f'\n-- ongoing [{s}]: share of true queued cells whose own on-ramp is switched off, by step --'); print(g.round(3).to_string())
        del data; gc.collect()
    S = pd.concat(sums, ignore_index=True); S.to_csv(f'{RES}/summary.csv', index=False)
    print('\n== implied S_total gain = 0.15 x dIoU_onset + 0.15 x dIoU_ongoing (offlike, train in-scenario) ==')
    rows = []
    for arm in dict.fromkeys(S.arm):
        a = S[(S.set == 'offlike') & (S.arm == arm)].set_index('cond')
        if len(a) < 2:
            continue
        don, dog = a.loc[ONSET, 'd_base'], a.loc[ONGOING, 'd_base']
        ctrl = S[(S.set == 'offlike') & (S.arm == 'S_ctrl')].set_index('cond')
        rows.append(dict(arm=arm, d_onset=don, d_ongoing=dog, S_total_gain=0.15 * don + 0.15 * dog,
                         S_total_gain_vs_ctrl=0.15 * (don - ctrl.loc[ONSET, 'd_base']) + 0.15 * (dog - ctrl.loc[ONGOING, 'd_base'])))
    G = pd.DataFrame(rows); G.to_csv(f'{RES}/s_total_gain.csv', index=False)
    print(G.round(4).to_string(index=False))
    lab.log(f'eval done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
