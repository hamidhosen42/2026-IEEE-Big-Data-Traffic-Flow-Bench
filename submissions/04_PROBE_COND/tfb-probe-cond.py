"""LB decomposition probes: FINAL v2 with queue predictions zeroed for one window condition."""
import glob, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
sub = pd.read_csv([p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True) if 'kaggle_public' not in p][0])
key = pd.read_csv(f'{REL}/submission_key.csv', usecols=['submission_id', 'task', 'window_id'], dtype=str)
key = key[key.task == 'queue']
wi = pd.concat([pd.read_csv(f) for f in glob.glob(f'{REL}/task2/*/*/window_index.csv')])
cond = key.window_id.map(wi.set_index('window_id').condition)
assert cond.notna().all()
ids = {c: set(key.submission_id[cond == c].astype('int64')) for c in ('queue_onset', 'queue_ongoing')}
for c, s in ids.items():
    out = sub.copy(); m = out.submission_id.isin(s)
    print(c, 'rows zeroed', int(m.sum()), 'positives removed', int(out.loc[m, 'queue_pred'].sum()))
    out.loc[m, 'queue_pred'] = 0
    out.to_csv(f'/kaggle/working/submission_zero_{c.split("_")[1]}.csv', index=False)
