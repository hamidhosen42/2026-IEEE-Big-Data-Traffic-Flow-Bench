"""Generic queue swap: start from the exact base file; replace queue rows of one condition (COND) with SRC_FILE from SRC_KERNEL.
Everything else untouched. Edit the four constants below before pushing."""
import glob, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE = 'tfb-e1', 'submission_FDRTV.csv'
SRC_KERNEL, SRC_FILE, COND, OUT_NAME = 'tfb-t2v20', 'queue_submission_og8.csv', 'queue_ongoing', 'E2'
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
kq['timestamp'] = pd.to_datetime(kq.timestamp, utc=True)
cond = pd.concat([pd.read_csv(f, dtype=str)[['window_id', 'condition']]
                  for f in glob.glob(f'{REL}/task2/*/*/window_index.csv') if '/train/' not in f]).drop_duplicates('window_id')
kq = kq.merge(cond, on='window_id', how='left', validate='many_to_one'); assert kq.condition.notna().all()
q = pd.read_csv(glob.glob(f'/kaggle/input/**/{SRC_KERNEL}/**/{SRC_FILE}', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
m = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert m.queue_pred.notna().all()
isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
src = pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
cmask = pd.Series((m.condition == COND).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
cur = base.loc[isq, 'queue_pred'].to_numpy()
print('info: other-condition agreement (not used)', round(float((src[~cmask] == cur[~cmask]).mean()), 5))
new = np.where(cmask, src, cur)
print(COND, 'rows changed share', round(float((new[cmask] != cur[cmask]).mean()), 5), '| positives before/after', int(cur[cmask].sum()), int(new[cmask].sum()))
out = base.copy(); out.loc[isq, 'queue_pred'] = new
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out))
