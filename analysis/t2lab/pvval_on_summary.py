"""pvval ONSET: consolidation of the pvon results (light: reads results/pvon/*.pkl, no training).
1. masking-adaptation control on pseudo-March days >= 15 (n = 131 windows), 3 mask seeds pooled per window
   (seed 1 from step2_cut15_s1_S.pkl, seeds 2 / 3 from step2_cut15_s2-3_S.pkl; CatBoost v13 columns = NORAMP)
2. in-train raw-deployment harm test (step3, 4 folds, raw targets R / masked targets M)
3. synthetic scenario shift (pvon_shift f2-3, halves A / B): kernel-style masked source rows vs raw source rows."""
import numpy as np
import pandas as pd
import lab
import pvval_lib as V

KEY = ['panel', 'd', 't', 'cond', 'set', 'family', 'fold']
pd.set_option('display.width', 250)


def seedmean(frames, name):
    """per-window mean IoU over the given (frame, arm) pairs -> one frame with arm = name."""
    out = None
    for i, (R, a) in enumerate(frames):
        g = R[R.arm == a][KEY + ['iou']].rename(columns={'iou': f'iou{i}'})
        out = g if out is None else out.merge(g, on=KEY)
    out['iou'] = out[[c for c in out.columns if c.startswith('iou')]].mean(1)
    out['arm'] = name
    return out[KEY + ['iou', 'arm']]


R1 = pd.read_pickle('results/pvon/step2_cut15_s1_S.pkl'); R23 = pd.read_pickle('results/pvon/step2_cut15_s2-3_S.pkl')
G = pd.concat([R1[R1.arm.isin(['A', 'A_s4', 'B_bag', 'B_bagL', 'B_w5', 'B_w10', 'B_w20', 'B_w40'])],
               seedmean([(R1, 'AmS_all'), (R23, 'AmS_all#2'), (R23, 'AmS_all#3')], 'AmS_all3'),
               seedmean([(R1, 'BmS_all_bag'), (R23, 'BmS_all#2_bag'), (R23, 'BmS_all#3_bag')], 'BmS_all3_bag'),
               seedmean([(R1, 'AmS_mix'), (R23, 'AmS_mix#2'), (R23, 'AmS_mix#3')], 'AmS_mix3'),
               seedmean([(R1, 'BmS_mix_bag'), (R23, 'BmS_mix#2_bag'), (R23, 'BmS_mix#3_bag')], 'BmS_mix3_bag')], ignore_index=True)
G['set'] = 'march_on'
rows = []
for a, b, note in [('A', 'A_s4', 'prior blend (production model + sp4)'),
                   ('A_s4', 'B_bag', 'B - A_s4: March rows, raw-train regime (kernel tfb-t2opv)'),
                   ('A_s4', 'B_bagL', 'bag at half weights 5/10/20'),
                   ('B_bag', 'B_bagL', 'bagL vs kernel bag'),
                   ('A_s4', 'AmS_all3', 'masking adaptation alone (A_mask - A_s4), 3 seeds'),
                   ('AmS_all3', 'BmS_all3_bag', 'B_mask - A_mask: March rows in the matched regime, 3 seeds'),
                   ('AmS_mix3', 'BmS_mix3_bag', 'same, raw+masked train mix, 3 seeds'),
                   ('B_bag', 'BmS_all3_bag', 'masked-train bag vs kernel bag'),
                   ('B_bag', 'BmS_mix3_bag', 'mixed-train bag vs kernel bag')]:
    c = V.pair(G, a, b, 'march_on'); c['note'] = note; rows.append(c)
print('== 1. onset masking control, pseudo-March days >= 15 ==')
print('  '.join(f'{a} {lab.aggregate(G[G.arm == a]):.4f}' for a in dict.fromkeys(G.arm)))
for c in rows:
    print(V.fmt(c), '|', c['note'])

S = pd.concat([pd.read_pickle('results/pvon/step3_f0-1_s1.pkl'), pd.read_pickle('results/pvon/step3_f2-3_s1.pkl')], ignore_index=True)
print('\n== 2. onset in-train harm test (4 folds; i = base, ii = + source month RAW, iv = + source month re-selected MASKED, bag 10/20/40) ==')
for s in ('R', 'M'):
    for a, b in [('i', 'ii_bag'), ('i', 'iv_bag'), ('ii_bag', 'iv_bag'), ('i', 'ii_w10'), ('i', 'iv_w10'), ('i', 'ii_w1')]:
        c = V.pair(S, a, b, s); rows.append(dict(c, note='harm')); print(V.fmt(c))

Z = pd.read_pickle('results/pvon/shift_f2-3_s1_dA-B.pkl')
print('\n== 3. synthetic scenario shift (roll by L/3 inside one half of the corridor; folds 2-3) ==')
for s in sorted(Z.set.unique()):
    for a, b in [('a_s4', 'c_bag'), ('b_bag', 'c_bag'), ('a_s4', 'b_bag'), ('c_bag', 'c_bagL')]:
        if {a, b} <= set(Z[Z.set == s].arm):
            c = V.pair(Z, a, b, s); rows.append(dict(c, note='shift')); print(V.fmt(c))
pd.DataFrame(rows).to_csv(f'{V.RES}/onset_summary.csv', index=False)
