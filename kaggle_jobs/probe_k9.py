"""LB decomposition probes on tfb-k9 (diagnostic only, never a final pick; 2026-11-04).

Writes three files from the exact tfb-k9 output:
  probe_noonset.csv   - queue_pred = 0 on queue_onset rows   -> LB drop = 0.30 * 0.5 * S_onset
  probe_noongoing.csv - queue_pred = 0 on queue_ongoing rows -> LB drop = 0.30 * 0.5 * S_ongoing
  probe_noodme.csv    - path_flow = 0 on odme rows            -> LB drop = 0.20 * S_ODME
Everything else is byte-for-byte the k9 value. Prints the changed-row counts as a check."""
import glob
import numpy as np, pandas as pd

REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
k9 = pd.read_csv(glob.glob('/kaggle/input/**/tfb-k9/**/submission_FDRTV.csv', recursive=True)[0])
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id'])
assert k9.submission_id.equals(sample.submission_id), 'k9 rows not in sample order'
K = pd.read_csv(f'{REL}/submission_key.csv', dtype=str, usecols=['submission_id', 'task', 'window_id'])
K['submission_id'] = K.submission_id.astype('int64')
K = K.set_index('submission_id').loc[k9.submission_id].reset_index()
assert (K.task.to_numpy() == k9.task.to_numpy()).all(), 'task labels differ from the key'
W = pd.concat([pd.read_csv(p, dtype=str)[['window_id', 'condition']] for p in glob.glob(f'{REL}/task2/*/*/window_index.csv')]).drop_duplicates()
assert W.window_id.is_unique, 'window_id with conflicting condition'
cond = K.window_id.map(W.set_index('window_id').condition)
q = (k9.task == 'queue').to_numpy()
assert cond[q].notna().all() and cond[q].isin(['queue_onset', 'queue_ongoing']).all(), 'unmapped queue rows'
for name, mask, col in (('noonset', q & (cond == 'queue_onset').to_numpy(), 'queue_pred'),
                        ('noongoing', q & (cond == 'queue_ongoing').to_numpy(), 'queue_pred'),
                        ('noodme', (k9.task == 'odme').to_numpy(), 'path_flow')):
    out = k9.copy()
    out.loc[mask, col] = 0
    out.to_csv(f'/kaggle/working/probe_{name}.csv', index=False)
    print(f'probe_{name}: rows set to 0 = {int(mask.sum())} | nonzero before = {int((k9.loc[mask, col] != 0).sum())}', flush=True)
print('PROBES OK', len(k9))
