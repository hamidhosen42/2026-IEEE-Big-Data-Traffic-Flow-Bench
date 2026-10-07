"""TRACK ogcap: paired deltas of the structures against the joint model at the SAME capacity (identical F03 rows / weights).
Usage: ogcap_report.py <folds e.g. 02 | 0123> [leaves=511] [lr=0.05] [n_trees=900] [cols=np]
Every entry (single arm or blend) is scored from the stored float16 probabilities with the production decode, on the
windows scored by all of its parts inside <folds>.  Writes results/ogcap/deltas_<cols><leaves>_l<lr>_t<trees>_f<folds>.csv
(pooled over the folds and per fold).  Fold 0 additionally: proxy vs FULL production setting (lr 0.03, 1800 trees):
  fc_F03_np511 (joint, full, lab arm) and og_qsplit_np511 (qsplit, full, ogmodel arm)."""
import os, sys
import numpy as np
import pandas as pd
import lab
import ogcap_lib as oc

folds = [int(c) for c in sys.argv[1]]
leaves = int(sys.argv[2]) if len(sys.argv) > 2 else 511
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
nt = int(sys.argv[4]) if len(sys.argv) > 4 else 900
cols = sys.argv[5] if len(sys.argv) > 5 else 'np'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
A = lambda st, seed=0: oc.arm_name(st, cols, leaves, lr, nt, seed)
J, J1, PS, QS = A('joint'), A('joint', 1), A('perstep'), A('qsplit')


def have(arm):
    ks = oc._my_folds(arm)
    if ks:
        return all(k in ks for k in folds)
    try:
        return set(folds) <= set(oc.load_df(arm).fold.unique().tolist())
    except Exception:
        return False


one = [1, 1, 1, 1, 1]
ENTRIES = {   # name -> (parts, step_w)
    'joint': ([(J, 1)], None),
    'joint_s1': ([(J1, 1)], None),
    'perstep': ([(PS, 1)], None),
    'qsplit': ([(QS, 1)], None),
    'h6spec_a1': ([(J, 1), (PS, 1)], {J: one + [0.0], PS: [0] * 5 + [1.0]}),
    'h6spec_a05': ([(J, 1), (PS, 1)], {J: one + [0.5], PS: [0] * 5 + [0.5]}),
    'joint2': ([(J, 1), (J1, 1)], None),
    'joint+perstep': ([(J, 1), (PS, 1)], None),
    'joint+qsplit': ([(J, 1), (QS, 1)], None),
    'perstep+qsplit': ([(PS, 1), (QS, 1)], None),
    'joint+perstep+qsplit': ([(J, 1), (PS, 1), (QS, 1)], None),
    'joint2+perstep': ([(J, 1), (J1, 1), (PS, 1)], None),
    'joint2+qsplit': ([(J, 1), (J1, 1), (QS, 1)], None),
}
# secondary: cross-capacity blends with the lab-capacity arms of track ogmodel / firstcross (255 leaves, lr 0.05, 500 trees;
# 4-fold OOF on identical rows): is a cheap 255-leaf structure member useful next to the 511 joint model?
ENTRIES.update({
    'joint+perstep255': ([(J, 1), ('og_perstep_np255', 1)], None),
    'joint+qsplit255': ([(J, 1), ('og_qsplit_np255', 1)], None),
    'joint+joint255': ([(J, 1), ('fc_F03_np255', 1)], None),
    'h6spec255_a05': ([(J, 1), ('og_perstep_np255', 1)], {J: one + [0.5], 'og_perstep_np255': [0] * 5 + [0.5]}),
})
COMPARE = [('joint', x) for x in ('joint_s1', 'perstep', 'qsplit', 'h6spec_a1', 'h6spec_a05', 'joint2', 'joint+perstep', 'joint+qsplit',
                                  'perstep+qsplit', 'joint+perstep+qsplit', 'joint+perstep255', 'joint+qsplit255', 'joint+joint255',
                                  'h6spec255_a05')]
COMPARE += [('joint2', x) for x in ('joint+perstep', 'joint+qsplit', 'perstep+qsplit', 'joint2+perstep', 'joint2+qsplit',
                                    'joint+perstep+qsplit', 'joint+perstep255', 'joint+qsplit255', 'joint+joint255')]
COMPARE += [('joint+joint255', x) for x in ('joint+perstep255', 'joint+qsplit255')]
# context: the capacity step itself and the ogmodel 255 structures on the same folds / windows
ENTRIES.update({'joint255': ([('fc_F03_np255', 1)], None), 'qsplit255': ([('og_qsplit_np255', 1)], None),
                'perstep255': ([('og_perstep_np255', 1)], None)})
COMPARE += [('joint255', 'joint'), ('joint255', 'qsplit255'), ('joint255', 'perstep255'), ('qsplit255', 'joint'), ('perstep255', 'joint'),
            ('qsplit255', 'qsplit'), ('perstep255', 'perstep')]
if 0 in folds:   # full production setting (fold 0 only)
    ENTRIES.update({'FULL_joint': ([('fc_F03_np511', 1)], None), 'FULL_qsplit': ([('og_qsplit_np511', 1)], None),
                    'FULL_joint+qsplit': ([('fc_F03_np511', 1), ('og_qsplit_np511', 1)], None),
                    'FULLjoint+PROXYperstep': ([('fc_F03_np511', 1), (PS, 1)], None),
                    'FULLjoint+PROXYjoint': ([('fc_F03_np511', 1), (J, 1)], None),
                    'FULLjoint+perstep255': ([('fc_F03_np511', 1), ('og_perstep_np255', 1)], None),
                    'FULLjoint+qsplit255': ([('fc_F03_np511', 1), ('og_qsplit_np255', 1)], None),
                    'FULLjoint+joint255': ([('fc_F03_np511', 1), ('fc_F03_np255', 1)], None)})
    COMPARE += [('FULL_joint', 'joint'), ('FULL_qsplit', 'qsplit'), ('FULL_joint', 'FULL_qsplit'), ('FULL_joint', 'FULL_joint+qsplit'),
                ('FULL_joint', 'FULLjoint+PROXYperstep'), ('FULLjoint+PROXYjoint', 'FULLjoint+PROXYperstep'),
                ('FULLjoint+PROXYjoint', 'FULL_joint+qsplit'), ('FULL_joint', 'FULLjoint+perstep255'), ('FULL_joint', 'FULLjoint+qsplit255'),
                ('FULL_joint', 'FULLjoint+joint255'), ('FULLjoint+PROXYjoint', 'FULLjoint+perstep255'),
                ('FULLjoint+PROXYjoint', 'FULLjoint+qsplit255'), ('FULLjoint+joint255', 'FULLjoint+perstep255'),
                ('FULLjoint+joint255', 'FULLjoint+qsplit255')]

ok = {n for n, (parts, _) in ENTRIES.items() if all(have(a) for a, _ in parts)}
lab.log('entries available:', sorted(ok))
fr = {}
for n in sorted(ok):
    parts, sw = ENTRIES[n]
    fr[n] = {s: oc.score_blend(parts, s, n, folds=folds, step_w=sw) for s in oc.EVALS}
rows = []
for a, b in COMPARE:
    if a not in ok or b not in ok:
        continue
    for s in oc.EVALS:
        fa, fb = fr[a][s], fr[b][s]
        if not len(fa) or not len(fb):
            continue
        rows.append(dict(cmp=f'{b} - {a}', fold='pooled', **oc.delta_row(fa, fb, b, s)))
        if len(folds) > 1:
            for k in folds:
                ga, gb = fa[fa.fold == k], fb[fb.fold == k]
                if len(ga) and len(gb):
                    rows.append(dict(cmp=f'{b} - {a}', fold=str(k), **oc.delta_row(ga, gb, b, s)))
D = pd.DataFrame(rows)
tag = f'{cols}{leaves}_l{int(round(lr * 100)):02d}_t{nt}_f{"".join(map(str, folds))}'
D.to_csv(f'{oc.RES}/deltas_{tag}.csv', index=False)
S = pd.DataFrame([dict(entry=n, set=s, n=len(fr[n][s]), iou=lab.aggregate(fr[n][s]), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(fr[n][s]))})
                  for n in sorted(ok) for s in oc.EVALS if len(fr[n][s])])
S.to_csv(f'{oc.RES}/summary_{tag}.csv', index=False)
print('== summary (aggregated IoU) ==')
print(S[S.set != 'pseudo'].round(4).to_string(index=False))
print('\n== paired deltas B - A (pooled over folds', folds, ') ==')
cols_show = ['cmp', 'set', 'fold', 'n', 'diff', 'lo', 'hi', 'p_pos', 'panels_pos', 'd_h6']
print(D[D.fold == 'pooled'][cols_show].round(4).to_string(index=False))
if len(folds) > 1:
    print('\n== per fold (offlike) ==')
    print(D[(D.fold != 'pooled') & (D.set == 'offlike')][cols_show].round(4).to_string(index=False))
lab.log('DONE', tag)
