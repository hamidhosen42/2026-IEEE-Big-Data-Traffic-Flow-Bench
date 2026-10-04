"""Production dry run (label-free): train arms on ALL train days (k = 4) and predict the released validation / private
official ongoing windows (features from data <= T only: lab.split_rows).  No labels of those windows exist locally and
none are used; the script only measures how much the submission would change and what the models themselves expect:
  * agreement (IoU) of the decoded masks of each arm with arm A and with last-valid persistence (qlast repeated),
  * predicted queue size per step relative to the queue at T,
  * the decoder's expected-IoU surrogate (model's own confidence).
Usage (through run_heavy): exp_firstcross_final.py <ARM[,ARM...]> [cols=np] [leaves=255] [lr=0.05] [n_trees=500]"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

arms = sys.argv[1].split(',')
cols = sys.argv[2] if len(sys.argv) > 2 else 'np'
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 255
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
n_trees = int(sys.argv[5]) if len(sys.argv) > 5 else 500
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
cols_ = lab.colset(cols)
MD = f'{fc.RES}/final'; os.makedirs(MD, exist_ok=True)


def surrogate(p):
    """value of the decoder objective at its maximum (the model's expected-IoU proxy) and the number of chosen cells."""
    ps = np.sort(p)[::-1]; cs = np.cumsum(ps); m = np.arange(1, len(p) + 1)
    v = cs / (m + (cs[-1] - cs)); j = int(np.argmax(v))
    return float(v[j]), j + 1


import lightgbm as lgb
models = {}
for a in arms:
    f = f'{MD}/model_{a}_{cols}{leaves}_{n_trees}.txt'
    if os.path.exists(f):
        models[a] = lgb.Booster(model_file=f); lab.log('loaded', f); continue
    t1 = time.time(); sp = fc.ARMS[a]
    X, y, w, info = fc.train_matrix(sp['train'], 4, cols_, sp['wfn'])
    p = lab.lgb_params(num_leaves=leaves, learning_rate=lr, min_data_in_leaf=40, seed=0)
    models[a] = fc.fit(X, y, w, p, n_trees, cols_)
    lab.log(f'final {a}: {len(X)} rows, pos {y.mean():.3f}, {time.time() - t1:.0f}s | {json.dumps(info)}')
    del X, y, w; gc.collect()
    models[a].save_model(f)

rows = []
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    for split in ('validation', 'private'):
        R = lab.split_rows(P, split)
        X, _, _, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        ix = [list(call).index(c) for c in cols_]
        og = np.where(R['cond'] == lab.ONGOING)[0]
        masks = {}; P6s = {}
        for a in arms:
            P6 = np.zeros((R['W'], lab.F, R['L']), np.float32)
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = models[a].predict(X[:, ix], num_threads=lab.THREADS)
            P6s[a] = P6; masks[a] = {i: lab.decode(P6[i].ravel()).reshape(lab.F, -1) for i in og}
        np.savez_compressed(f'{MD}/probs_{panel}_{split}_{cols}{leaves}.npz', og=og, **{a: P6s[a][og].astype(np.float16) for a in arms})
        for i in og:
            ql = np.repeat(R['qlast'][i][None], lab.F, 0); allm = np.ones_like(ql, bool)
            r = dict(panel=panel, family=panel.rsplit('_', 1)[0], split=split, w=int(i), window_id=str(R['wi'].window_id.iloc[i]), tod=int(R['t'][i]),
                     nq_T=int(R['qlast'][i].sum()), L=int(R['L']))
            for a in arms:
                m = masks[a][i]; sv, sm = surrogate(P6s[a][i].ravel())
                r[f'{a}_n'] = int(m.sum()); r[f'{a}_surr'] = sv
                r[f'{a}_vs_pers'] = lab.iou(m, ql, allm)
                r[f'{a}_vs_{arms[0]}'] = lab.iou(m, masks[arms[0]][i], allm)
                for h in range(lab.F):
                    r[f'{a}_n{h + 1}'] = int(m[h].sum())
            rows.append(r)
    lab.drop_panel(panel)
D = pd.DataFrame(rows); D.to_pickle(f'{MD}/dryrun_{cols}{leaves}_{"_".join(arms)}.pkl')


def pm(g, c):
    return g.groupby(['family', 'panel'])[c].mean().groupby(level=0).mean().mean()


out = []
for split, g in D.groupby('split'):
    for a in arms:
        r = dict(split=split, arm=a, n=len(g), surrogate=pm(g, f'{a}_surr'), iou_vs_persistence=pm(g, f'{a}_vs_pers'), iou_vs_base=pm(g, f'{a}_vs_{arms[0]}'),
                 cells_pred=g[f'{a}_n'].sum(), cells_qT_x6=6 * g.nq_T.sum())
        for h in range(lab.F):
            r[f'n{h + 1}/nqT'] = g[f'{a}_n{h + 1}'].sum() / max(g.nq_T.sum(), 1)
        out.append(r)
T = pd.DataFrame(out)
print('== dry run on the released official ongoing windows (label-free) =='); print(T.round(4).to_string(index=False))
T.to_csv(f'{MD}/dryrun_summary_{cols}{leaves}_{"_".join(arms)}.csv', index=False)
for a in arms[1:]:
    t = D.groupby(['split', 'panel'])[[f'{a}_vs_{arms[0]}', f'{a}_surr', f'{arms[0]}_surr', f'{a}_n', f'{arms[0]}_n']].mean().round(3)
    print(f'\nper panel: {a} vs {arms[0]}'); print(t.to_string())
lab.log('DONE')
