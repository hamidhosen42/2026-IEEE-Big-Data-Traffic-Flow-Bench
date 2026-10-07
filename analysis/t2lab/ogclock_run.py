"""TRACK ogclock: OOF F03 arm (lab scale: 255 leaves, lr 0.05, 500 trees, min_data_in_leaf 40) with ck columns appended.
Usage: ogclock_run.py ARM KIND GROUP FOLDS
  KIND   production column set: np | id | np+tod | id+tod | np+tod+dow ...
  GROUP  ck columns: all | link | corr | none | comma-free names joined by ':'
  FOLDS  e.g. 0,1
Reference arms trained on identical rows / weights / params: fc_F03_np255, fc_F03_id255 (cache/oof)."""
import sys
import lab
import ogclock_lib as ck

arm, kind, group, folds = sys.argv[1], sys.argv[2], sys.argv[3], tuple(int(x) for x in sys.argv[4].split(','))
xc = {'all': ck.CK_ALL, 'link': ck.CK_LINK, 'corr': ck.CK_CORR, 'none': []}.get(group) or group.split(':')
df = ck.run_arm(arm, kind, xc, folds=folds)
for s in ('official', 'offlike', 'pseudo'):
    g = df[df.set == s]
    lab.log(f'{arm} {s}: {lab.aggregate(g):.4f} n={len(g)}')
lab.log('DONE')
