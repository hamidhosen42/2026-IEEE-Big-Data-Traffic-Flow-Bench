"""TRACK ogcap: paired deltas of arbitrary entries against a base entry on given folds (light: scoring of stored probabilities).
Usage: ogcap_cmp.py <folds e.g. 02> <base> <entry> [<entry> ...] [--out name]
An entry is an arm name or an equal-weight blend 'armA+armB[+...]' (arms: ogcap / ogmodel / lab caches)."""
import sys
import pandas as pd
import lab
import ogcap_lib as oc

args = sys.argv[1:]
out = None
if '--out' in args:
    i = args.index('--out'); out = args[i + 1]; args = args[:i] + args[i + 2:]
folds = [int(c) for c in args[0]]
base, entries = args[1], args[2:]
pd.set_option('display.width', 250)
fr = {e: {s: oc.score_blend([(a, 1.0) for a in e.split('+')], s, e, folds=folds) for s in oc.EVALS} for e in [base] + entries}
rows = []
for e in entries:
    for s in oc.EVALS:
        if len(fr[base][s]) and len(fr[e][s]):
            rows.append(dict(cmp=f'{e} - {base}', fold='pooled', **oc.delta_row(fr[base][s], fr[e][s], e, s)))
            for k in folds if len(folds) > 1 else []:
                a, b = fr[base][s], fr[e][s]
                if (a.fold == k).any() and (b.fold == k).any():
                    rows.append(dict(cmp=f'{e} - {base}', fold=str(k), **oc.delta_row(a[a.fold == k], b[b.fold == k], e, s)))
D = pd.DataFrame(rows)
if out:
    D.to_csv(f'{oc.RES}/{out}.csv', index=False)
print(D[['cmp', 'set', 'fold', 'n', 'base', 'iou', 'diff', 'lo', 'hi', 'p_pos', 'panels_pos', 'd_h6']].round(4).to_string(index=False))
