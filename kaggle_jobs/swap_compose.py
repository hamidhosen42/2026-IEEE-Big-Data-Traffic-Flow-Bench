"""COMPOSE swap: every row from the base file; Task 1 state rows (speed_kmh, flow_vph) from T1_KERNEL / T1_FILE.
Used when a Task 1 variant was rebuilt from the raw mix (whose queue rows may be older than the chain base)."""
import glob, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE = 'tfb-c20', 'submission_TV.csv'
T1_KERNEL, T1_FILE, OUT_NAME = 'tfb-c17', 'submission_FDRTV.csv', 'CMP'
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0])
t1 = pd.read_csv(glob.glob(f'/kaggle/input/**/{T1_KERNEL}/**/{T1_FILE}', recursive=True)[0], usecols=['submission_id', 'task', 'speed_kmh', 'flow_vph'])
assert len(base) == len(t1) and (base.submission_id.to_numpy() == t1.submission_id.to_numpy()).all(), 'row order differs'
iss = (base.task == 'state').to_numpy()
assert ((t1.task == 'state').to_numpy() == iss).all()
out = base.copy()
out.loc[iss, 'speed_kmh'] = t1.loc[iss, 'speed_kmh'].to_numpy(); out.loc[iss, 'flow_vph'] = t1.loc[iss, 'flow_vph'].to_numpy()
dv = out.loc[iss, 'speed_kmh'].to_numpy() - base.loc[iss, 'speed_kmh'].to_numpy()
dq = out.loc[iss, 'flow_vph'].to_numpy() - base.loc[iss, 'flow_vph'].to_numpy()
print('state rows changed share', round(float(((dv != 0) | (dq != 0)).mean()), 5), '| rms dv', round(float(np.sqrt(np.mean(dv ** 2))), 4),
      '| rms dq', round(float(np.sqrt(np.mean(dq ** 2))), 3))
isq = (base.task == 'queue').to_numpy()
assert np.array_equal(out.queue_pred.to_numpy(), base.queue_pred.to_numpy(), equal_nan=True), 'queue rows changed'
assert np.array_equal(out.path_flow.to_numpy(), base.path_flow.to_numpy(), equal_nan=True), 'path_flow changed'
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out))
