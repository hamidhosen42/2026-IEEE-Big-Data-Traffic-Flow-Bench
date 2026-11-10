"""FINAL v2 submission with ONLY the Task 2 rows replaced by each Task 2 v5 variant (one file per variant)."""
import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv([p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True) if 'kaggle_public' not in p][0])
key = pd.read_csv(f'{REL}/submission_key.csv', usecols=['submission_id', 'task', 'window_id', 'timestamp', 'link_id'], dtype=str)
key = key[key.task == 'queue'].copy(); key['timestamp'] = pd.to_datetime(key.timestamp, utc=True)
isq = (base.task == 'queue').to_numpy()
for f in sorted(glob.glob('/kaggle/input/**/queue_submission*.csv', recursive=True)):
    name = f.split('/')[-1].replace('queue_submission', '').replace('.csv', '').strip('_') or 'main'
    q = pd.read_csv(f, dtype={'window_id': str, 'link_id': str}); q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = key.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
    assert m.queue_pred.notna().all()
    new = pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy())
    vals = new.reindex(base.loc[isq, 'submission_id'].to_numpy()).to_numpy()
    same = float((vals == base.loc[isq, 'queue_pred'].to_numpy()).mean())
    out = base.copy(); out.loc[isq, 'queue_pred'] = vals; out['queue_pred'] = out.queue_pred.astype(int)
    assert len(out) == len(base) and out.notna().all().all() and out.loc[isq, 'queue_pred'].isin([0, 1]).all()
    out.to_csv(f'/kaggle/working/submission_{name}.csv', index=False)
    print(f'{name:12s} positive share {vals.mean():.4f}  agreement with FINAL v2 queue rows {same:.4f}', flush=True)
