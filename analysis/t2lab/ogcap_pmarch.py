"""TRACK ogcap: pseudo-March evaluation of the final (all 273 train days, k = 4) models at the proxy capacity.
Regime: NEW scenario (validation month), train-only models, MASKED histories (about 50 % of history cells visible; the real
official windows have about 76 %), labels = VISIBLE masked-layer cells only (the real scorer uses the full noise-free state).
Rows: the cached pseudo-March rows of track ogmodel (read-only: data/t2local/cache/ogmodel/pmarch/<panel>__<set>__T.npz,
sets offlike_m = faithful greedy multi-start, runstart_m = faithful run-starts).  Nothing is trained or tuned on March labels.
Usage: ogcap_pmarch.py [leaves=511] [lr=0.05] [n_trees=900] [cols=np]"""
import os, sys
import numpy as np
import pandas as pd
import lab
import ogcap_lib as oc

leaves = int(sys.argv[1]) if len(sys.argv) > 1 else 511
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
nt = int(sys.argv[3]) if len(sys.argv) > 3 else 900
cols = sys.argv[4] if len(sys.argv) > 4 else 'np'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
A = lambda st, seed=0: oc.arm_name(st, cols, leaves, lr, nt, seed)
J, J1, PS, QS = A('joint'), A('joint', 1), A('perstep'), A('qsplit')
one = [1.0] * 5
ENTRIES = {   # name -> (arms, per-arm step weights or None = equal weights)
    'joint': ([J], None), 'joint_s1': ([J1], None), 'perstep': ([PS], None), 'qsplit': ([QS], None),
    'h6spec_a1': ([J, PS], [one + [0.0], [0.0] * 5 + [1.0]]), 'h6spec_a05': ([J, PS], [one + [0.5], [0.0] * 5 + [0.5]]),
    'joint2': ([J, J1], None), 'joint+perstep': ([J, PS], None), 'joint+qsplit': ([J, QS], None), 'perstep+qsplit': ([PS, QS], None),
    'joint+perstep+qsplit': ([J, PS, QS], None), 'joint2+perstep': ([J, J1, PS], None), 'joint2+qsplit': ([J, J1, QS], None),
}
COMPARE = [('joint', x) for x in ('joint_s1', 'perstep', 'qsplit', 'h6spec_a1', 'h6spec_a05', 'joint2', 'joint+perstep', 'joint+qsplit',
                                  'perstep+qsplit', 'joint+perstep+qsplit')]
COMPARE += [('joint2', x) for x in ('joint+perstep', 'joint+qsplit', 'perstep+qsplit', 'joint2+perstep', 'joint2+qsplit', 'joint+perstep+qsplit')]
# lab-capacity final models (255 leaves, lr 0.05, 500 trees, same F03 rows; read-only): firstcross joint, ogmodel qsplit / perstep
R255 = dict(joint255=('fc', f'{lab.ROOT}/analysis/t2lab/results/firstcross/final/model_F03_{cols}255_500.txt'),
            qsplit255=('og', f'og_qsplit_{cols}255'), perstep255=('og', f'og_perstep_{cols}255'))
ENTRIES.update({'joint255': (['joint255'], None), 'qsplit255': (['qsplit255'], None), 'perstep255': (['perstep255'], None),
                'joint+joint255': ([J, 'joint255'], None), 'joint+qsplit255': ([J, 'qsplit255'], None), 'joint+perstep255': ([J, 'perstep255'], None),
                'h6spec255_a05': ([J, 'perstep255'], [one + [0.5], [0.0] * 5 + [0.5]])})
COMPARE += [('joint255', 'joint'), ('joint255', 'qsplit255'), ('joint255', 'perstep255'), ('qsplit255', 'joint'), ('qsplit255', 'qsplit')]
COMPARE += [('joint', x) for x in ('joint+joint255', 'joint+qsplit255', 'joint+perstep255', 'h6spec255_a05')]
COMPARE += [('joint+joint255', x) for x in ('joint+qsplit255', 'joint+perstep255')]


def load_255(name):
    import json, lightgbm as lgb
    kind, ref = R255[name]
    if kind == 'fc':
        return {}, [(0, lgb.Booster(model_file=ref))], lab.colset(cols)
    d = f'{lab.CACHE}/ogmodel/models'; cfg = json.load(open(f'{d}/{ref}.json'))
    return cfg['spec'], [(g, lgb.Booster(model_file=f'{d}/{ref}__g{g}.txt')) for g in cfg['groups']], cfg['cols']


avail = {a for a in (J, J1, PS, QS) if os.path.exists(f'{oc.MODELS}/{a}.json')}
loaded = {a: oc.load_models(a) for a in avail}
for nm, (kind, ref) in R255.items():
    if os.path.exists(ref if kind == 'fc' else f'{lab.CACHE}/ogmodel/models/{ref}.json'):
        loaded[nm] = load_255(nm); avail.add(nm)
lab.log('final models available:', sorted(avail))
ok = [n for n, (arms, _) in ENTRIES.items() if set(arms) <= avail]
res = []
for panel in lab.PANELS:
    for sname in oc.PM_SETS:
        pm = oc.pm_load(panel, sname, 'T')
        if pm is None:
            continue
        P6 = {}
        for a in avail:
            spec, models, cs = loaded[a]
            P6[a] = oc.pm_predict(pm, lambda X, spec=spec, models=models, cs=cs: oc.predict_struct(spec, models, X, cs), cs)
        for n in ok:
            arms, sw = ENTRIES[n]
            if sw is None:
                p = sum(P6[a] for a in arms) / len(arms)
            else:
                ws = [np.asarray(w, np.float32)[None, :, None] for w in sw]
                p = sum(w * P6[a] for w, a in zip(ws, arms)) / np.maximum(sum(ws), 1e-12)
            df = lab.score_probs(pm.rc, p, arm=n, set_name=sname)
            df['runstart'] = pm.rc.win.runstart.to_numpy()
            res.append(df)
    lab.log(panel, 'done')
R = pd.concat(res, ignore_index=True)
tag = f'{cols}{leaves}_l{int(round(lr * 100)):02d}_t{nt}'
R.to_pickle(f'{oc.RES}/pmarch_{tag}.pkl')
S = pd.DataFrame([dict(set=s, entry=n, n=int(((R.set == s) & (R.arm == n)).sum()), iou=lab.aggregate(R[(R.set == s) & (R.arm == n)]),
                       **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(R[(R.set == s) & (R.arm == n)]))}) for s in oc.PM_SETS for n in ok])
S.to_csv(f'{oc.RES}/pmarch_summary_{tag}.csv', index=False)
print('== PSEUDO-MARCH (train-only k=4 models; masked histories; visible labels): aggregated IoU ==')
print(S.round(4).to_string(index=False))
rows = []
for a, b in COMPARE:
    if a in ok and b in ok:
        for s in oc.PM_SETS:
            rows.append(dict(cmp=f'{b} - {a}', **oc.delta_row(R[(R.set == s) & (R.arm == a)], R[(R.set == s) & (R.arm == b)], b, s)))
D = pd.DataFrame(rows); D.to_csv(f'{oc.RES}/pmarch_deltas_{tag}.csv', index=False)
print('\n== paired deltas B - A (lab.paired_bootstrap, windows resampled within panel) ==')
print(D[['cmp', 'set', 'n', 'base', 'iou', 'diff', 'lo', 'hi', 'p_pos', 'panels_pos', 'd_h6']].round(4).to_string(index=False))
lab.log('DONE', tag)
