"""Controlled LB probe: EXP_000 submission with ONLY the Task 2 rows replaced (Task 2 v2 blend_70)."""
import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base_p = [p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True) if 'kaggle_public' not in p]
q_p = glob.glob('/kaggle/input/**/queue_submission_blend70.csv', recursive=True)
assert len(base_p) == 1 and len(q_p) == 1, (base_p, q_p)
sub = pd.read_csv(base_p[0])
key = pd.read_csv(f'{REL}/submission_key.csv', usecols=['submission_id', 'task', 'window_id', 'timestamp', 'link_id'], dtype=str)
key = key[key.task == 'queue']
q = pd.read_csv(q_p[0], dtype={'window_id': str, 'link_id': str})
q['timestamp'] = pd.to_datetime(q.timestamp, utc=True); key['timestamp'] = pd.to_datetime(key.timestamp, utc=True)
m = key.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
assert m.queue_pred.notna().all(), f'{m.queue_pred.isna().sum()} queue keys unmatched'
new = pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy())
isq = sub.task == 'queue'
before = sub.loc[isq, 'queue_pred'].mean()
sub.loc[isq, 'queue_pred'] = new.reindex(sub.loc[isq, 'submission_id'].to_numpy()).to_numpy()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(sub) == len(sample) and (sub.submission_id.to_numpy() == sample).all()
assert sub[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].notna().all().all()
assert sub.loc[isq, 'queue_pred'].isin([0, 1]).all()
sub['queue_pred'] = sub.queue_pred.astype(int)
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('queue rows', int(isq.sum()), 'positive share before', round(before, 4), 'after', round(sub.loc[isq, 'queue_pred'].mean(), 4))
print('rows', len(sub), 'OK')
