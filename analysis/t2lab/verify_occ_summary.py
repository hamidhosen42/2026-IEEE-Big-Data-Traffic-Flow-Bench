"""verify_occ_summary.py : pool my seed-2 OOF frames (verify_occ_train) over the folds present and report occ vs base
(np, id, np+id; all / run-start / mid-run; drop D12_I5_S; per panel; per fold), plus seed noise vs the occ track's seed-0 arms."""
import glob
import numpy as np
import pandas as pd
import lab
import occ_lib as oc
import exp_firstcross_lib as fc
import verify_occ_feat as vf

SEED = 2
fs = sorted(glob.glob(f'{vf.VOUT}/train_f*_s{SEED}.pkl'))
df = pd.concat([pd.read_pickle(f)['df'] for f in fs], ignore_index=True)
folds = sorted(df.fold.unique())
lab.log('folds present', folds)


def bt(a, b):
    a, b = fc.paired(a, b)
    r = lab.paired_bootstrap(a, b)
    pp = (b.groupby('panel').iou.mean() - a.groupby('panel').iou.mean())
    return f'{r["diff"]:+.4f} [{r["lo"]:+.4f}, {r["hi"]:+.4f}] n={r["n_windows"]} panels +{int((pp > 1e-9).sum())}/-{int((pp < -1e-9).sum())}'


for s in ('offlike', 'pseudo', 'official'):
    g = df[df.set == s]
    for c in ('np', 'id', 'npid'):
        a = g[g.arm == f'base_{c}']; b = g[g.arm == f'occ_{c}']
        line = f'[{s}] seed {SEED} {c}: base {lab.aggregate(a):.4f} occ {lab.aggregate(b):.4f} diff {bt(a, b)}'
        if s == 'offlike':
            rs = a.runstart.astype(bool); rsb = b.runstart.astype(bool)
            line += f' | runstart {bt(a[rs], b[rsb])} | midrun {bt(a[~rs], b[~rsb])}'
            line += f' | drop D12_I5_S {bt(a[a.panel != "D12_I5_S"], b[b.panel != "D12_I5_S"])}'
            line += f' | runstart & drop D12_I5_S {bt(a[rs & (a.panel != "D12_I5_S")], b[rsb & (b.panel != "D12_I5_S")])}'
        lab.log(line)
        if s == 'offlike':
            a2, b2 = fc.paired(a, b)
            lab.log(f'   per panel {c}:', (b2.groupby("panel").iou.mean() - a2.groupby("panel").iou.mean()).round(4).to_dict())
            lab.log(f'   per fold {c}:', {int(k): round(lab.aggregate(b2[b2.fold == k]) - lab.aggregate(a2[a2.fold == k]), 4) for k in folds})
            best = (b2.groupby('panel').iou.mean() - a2.groupby('panel').iou.mean()).idxmax()
            lab.log(f'   drop my best panel {best} ({c}): {bt(a[a.panel != best], b[b.panel != best])}')
# seed noise: my seed-2 arms vs the occ track's seed-0 arms (same rows, same features), offlike, same folds
for mine, theirs in (('base_np', 'af:af_all_np255'), ('occ_np', 'occ_np255'), ('base_id', 'af:af_all_id255'), ('occ_id', 'occ_id255')):
    t = oc.load_arm(theirs); t = t[(t.set == 'offlike') & t.fold.isin(folds)]
    m = df[(df.set == 'offlike') & (df.arm == mine)]
    lab.log(f'NOISE offlike {mine} seed {SEED} vs {theirs} seed 0: {bt(t, m)}')
lab.log('SUMMARY DONE')
