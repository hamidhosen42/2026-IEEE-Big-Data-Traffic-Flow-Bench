"""TRACK ogmodel: consolidated tables for the structure arms (light, no training).
  1. every arm found on disk vs the reference arm (paired, common windows, lab.paired_bootstrap, panels positive)
  2. probability blends of cached arms (step-6 specialist, per-step for late steps, structure + joint) vs the single joint
     model AND vs the 2-seed joint bag (the fair reference for any 2-model blend)
Usage: ogmodel_report.py [cols=np] [folds e.g. 0123]"""
import os, sys
import numpy as np
import pandas as pd
import lab
import ogmodel_lib as og

cols = sys.argv[1] if len(sys.argv) > 1 else 'np'
folds = [int(c) for c in sys.argv[2]] if len(sys.argv) > 2 else [0, 1, 2, 3]
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
REF = f'fc_F03_{cols}255'
ref = lab.load_arm(REF); ref = ref[ref.fold.isin(folds)]


def have(arm, nf=4):
    return all(os.path.exists(f'{og.arm_dir(arm)}/fold{k}.pkl') for k in folds)


arms = [a for a in sorted(os.listdir(og.OOF)) if a.startswith('og_') and f'_{cols}' in a and have(a)]
print(f'reference {REF}, folds {folds}; arms on disk: {arms}\n')
rows = []; summ = []
g = lab.summary(ref); g.insert(0, 'arm', REF); summ.append(g)
for a in arms:
    df = og.merge(a); df = df[df.fold.isin(folds)]
    g = lab.summary(df); g.insert(0, 'arm', a); summ.append(g)
    for s in og.EVALS:
        if (df.set == s).any():
            rows.append(og.delta_row(ref[ref.set == s], df[df.set == s], a, s))
S = pd.concat(summ, ignore_index=True)
print('== SUMMARY (aggregated IoU, per-step IoU, error split) ==')
for s in og.EVALS:
    print(S[S.set == s].round(4).to_string(index=False)); print()
D = pd.DataFrame(rows)
print(f'== PAIRED DELTAS vs {REF} (single model, seed 0) ==')
for s in og.EVALS:
    print(D[D.set == s].round(4).to_string(index=False)); print()
S.to_csv(f'{og.RES}/report_summary_{cols}.csv', index=False); D.to_csv(f'{og.RES}/report_deltas_{cols}.csv', index=False)

# ---- blends
J, J1, PS = REF, f'og_joint_{cols}255_s1', f'og_perstep_{cols}255'
B = {}
if have(J1):
    B['joint2 (seed 0 + seed 1)'] = dict(parts=[(J, 1), (J1, 1)])
if have(PS):
    B['h6spec a=1.0 (joint 1-5, per-step model at 6)'] = dict(parts=[(J, 1), (PS, 1)], step_w={J: [1, 1, 1, 1, 1, 0], PS: [0, 0, 0, 0, 0, 1]})
    B['h6spec a=0.5 (step 6: 0.5 joint + 0.5 specialist)'] = dict(parts=[(J, 1), (PS, 1)], step_w={J: [1, 1, 1, 1, 1, .5], PS: [0, 0, 0, 0, 0, .5]})
    B['h56spec a=0.5'] = dict(parts=[(J, 1), (PS, 1)], step_w={J: [1, 1, 1, 1, .5, .5], PS: [0, 0, 0, 0, .5, .5]})
    B['perstep 4-6, joint 1-3'] = dict(parts=[(J, 1), (PS, 1)], step_w={J: [1, 1, 1, 0, 0, 0], PS: [0, 0, 0, 1, 1, 1]})
    B['joint + perstep (0.5 / 0.5)'] = dict(parts=[(J, 1), (PS, 1)])
    if have(J1):
        B['joint2 + perstep (1/3 each)'] = dict(parts=[(J, 1), (J1, 1), (PS, 1)])
for st in ('hg33', 'hg42', 'qsplit', 'diff', 'hwA', 'hwB', 'hwC', 'hg33q'):
    a = f'og_{st}_{cols}255'
    if have(a):
        B[f'joint + {st} (0.5 / 0.5)'] = dict(parts=[(J, 1), (a, 1)])
rows = []; frames = {}
for name, b in B.items():
    for s in og.EVALS:
        df = og.score_blend(b['parts'], s, name, step_w=b.get('step_w'), folds=folds)
        frames[(name, s)] = df
        r = og.delta_row(ref[ref.set == s], df, name, s); rows.append(r)
BD = pd.DataFrame(rows)
print(f'== BLENDS vs {REF} (single joint model) ==')
for s in og.EVALS:
    print(BD[BD.set == s].round(4).to_string(index=False)); print()
BD.to_csv(f'{og.RES}/report_blends_{cols}.csv', index=False)
j2 = 'joint2 (seed 0 + seed 1)'
if j2 in B:
    rows = []
    for name in B:
        if name == j2 or len(B[name]['parts']) < 2 or 'step_w' in B[name]:
            continue
        for s in og.EVALS:
            rows.append(og.delta_row(frames[(j2, s)], frames[(name, s)], name, s))
    B2 = pd.DataFrame(rows)
    print('== 2-model BLENDS vs the 2-seed joint bag (fair reference: same number of models) ==')
    for s in og.EVALS:
        print(B2[B2.set == s].round(4).to_string(index=False)); print()
    B2.to_csv(f'{og.RES}/report_blends_vs_joint2_{cols}.csv', index=False)
