"""COMBO_8: start from the exact COMBO7 file (LB 0.86509); replace ONLY queue_ongoing window rows with v14 'id511'
(identity-kept prior-free ongoing: drops prior/tod/dow only; pseudo-March gate +0.0036 vs np511). Onset rows, Task 1 and Task 4 untouched.
CONTROL: v14 main (np511 ongoing) vs COMBO7 ongoing rows agreement (v13 main) is printed."""
import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob('/kaggle/input/**/tfb-combo7/**/submission_COMBO7.csv', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
kq['timestamp'] = pd.to_datetime(kq.timestamp, utc=True)
cond = pd.concat([pd.read_csv(f, dtype=str)[['window_id', 'condition']]
                  for f in glob.glob(f'{REL}/task2/*/*/window_index.csv') if '/train/' not in f]).drop_duplicates('window_id')
kq = kq.merge(cond, on='window_id', how='left', validate='many_to_one')
assert kq.condition.notna().all(), 'window without condition'


def load(name):
    q = pd.read_csv(glob.glob(f'/kaggle/input/**/tfb-t2v14/**/{name}', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
    q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
    assert m.queue_pred.notna().all()
    return pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy())


isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
og_ids = set(kq.loc[kq.condition == 'queue_ongoing', 'submission_id'].astype('int64'))
is_og = np.array([s in og_ids for s in sid])
cur = base.loc[isq, 'queue_pred'].to_numpy()
main14 = load('queue_submission.csv').reindex(sid).to_numpy()
print('CONTROL: v14 main (np511) vs COMBO7 ongoing rows agreement', round(float((main14[is_og] == cur[is_og]).mean()), 5),
      '| onset rows agreement (info)', round(float((main14[~is_og] == cur[~is_og]).mean()), 5))
idv = load('queue_submission_id511.csv').reindex(sid).to_numpy()
new = np.where(is_og, idv, cur)
print('ongoing rows changed share', round(float((new[is_og] != cur[is_og]).mean()), 5), 'onset rows changed', int((new[~is_og] != cur[~is_og]).sum()))
out = base.copy(); out.loc[isq, 'queue_pred'] = new
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv('/kaggle/working/submission_COMBO8.csv', index=False)
print('COMBO8 OK', len(out))
