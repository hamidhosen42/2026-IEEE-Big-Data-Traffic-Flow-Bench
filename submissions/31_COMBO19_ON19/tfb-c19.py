"""Onset swap (v19): start from the exact current best file (COMBO15, LB 0.86944); replace ONLY the queue_onset rows with the
tfb-t2onv19 onset file (ABC_st03 = 0.3 * stage-2 + 0.7 * mean(A0, B, C); fallback A0_st03). Ongoing, state and ODME rows untouched
(asserted). Derived from swap_combo13.py. Gates before submitting: v19 log CONTROL PASS (A0 vs tfb-t2v13 >= 0.999) + CV table."""
import glob, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE = 'tfb-c18', 'submission_COMBO18.csv'
SRC_KERNEL, SRC_FILE, SRC_FALLBACK, COND, OUT_NAME = 'tfb-t2onv19', 'queue_submission_ABC_st03.csv', 'queue_submission_ABC_st03.csv', 'queue_onset', 'COMBO19'
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
kq['timestamp'] = pd.to_datetime(kq.timestamp, utc=True)
cond = pd.concat([pd.read_csv(f, dtype=str)[['window_id', 'condition']]
                  for f in glob.glob(f'{REL}/task2/*/*/window_index.csv') if '/train/' not in f]).drop_duplicates('window_id')
kq = kq.merge(cond, on='window_id', how='left', validate='many_to_one'); assert kq.condition.notna().all()
isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
cmask = pd.Series((kq.condition == COND).to_numpy(), index=kq.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
assert not pd.isna(cmask).any(); cmask = cmask.astype(bool)
cur = base.loc[isq, 'queue_pred'].to_numpy()


def load_src(fname):
    fs = glob.glob(f'/kaggle/input/**/{SRC_KERNEL}/**/{fname}', recursive=True)
    if not fs:
        return None
    q = pd.read_csv(fs[0], dtype={'window_id': str, 'link_id': str}); q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert m.queue_pred.notna().all(), fname
    return pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()


src = load_src(SRC_FILE); used = SRC_FILE
if src is None:
    src = load_src(SRC_FALLBACK); used = SRC_FALLBACK
assert src is not None, 'no v19 onset file found'
print('source file', used)
a0 = load_src('queue_submission_A0.csv')
if a0 is not None:      # info: does plain A0 (v13 replica) reproduce the base onset rows?
    print('info: A0 vs base onset rows agreement', round(float((a0[cmask] == cur[cmask]).mean()), 6),
          '| positives A0 / base', int(a0[cmask].sum()), int(cur[cmask].sum()))
print('info: other-condition agreement (not used; v19 writes 0 there)', round(float((src[~cmask] == cur[~cmask]).mean()), 5))
new = np.where(cmask, src, cur)
print(COND, 'rows changed', int((new[cmask] != cur[cmask]).sum()), 'of', int(cmask.sum()), '| share', round(float((new[cmask] != cur[cmask]).mean()), 5),
      '| positives before/after', int(cur[cmask].sum()), int(new[cmask].sum()))
out = base.copy(); out.loc[isq, 'queue_pred'] = new
# ---- changed onset rows (per window, then the first rows)
ch = kq.set_index(kq.submission_id.astype('int64')).reindex(sid)[cmask & (new != cur)].copy()
ch['old'] = cur[cmask & (new != cur)]; ch['new'] = new[cmask & (new != cur)]
print('onset windows changed', ch.window_id.nunique(), 'of', kq[kq.condition == COND].window_id.nunique(),
      '| cells 0->1', int((ch.new == 1).sum()), '| 1->0', int((ch.new == 0).sum()))
pd.set_option('display.width', 200)
print(ch.groupby('window_id').agg(n=('new', 'size'), add=('new', 'sum')).sort_values('n', ascending=False).head(40).to_string())
print(ch[['window_id', 'timestamp', 'link_id', 'old', 'new']].head(60).to_string())
# ---- hard gates: ongoing / state / ODME rows identical, file shape
og = ~cmask
assert (out.loc[isq, 'queue_pred'].to_numpy()[og] == base.loc[isq, 'queue_pred'].to_numpy()[og]).all(), 'ongoing rows changed'
for c in ('speed_kmh', 'flow_vph', 'path_flow'):
    assert np.array_equal(out.loc[~isq, c].to_numpy(), base.loc[~isq, c].to_numpy(), equal_nan=True), f'non-queue column {c} changed'
assert out.loc[isq, ['speed_kmh', 'flow_vph', 'path_flow']].equals(base.loc[isq, ['speed_kmh', 'flow_vph', 'path_flow']])
assert list(out.columns) == list(base.columns) and (out.task.to_numpy() == base.task.to_numpy()).all()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out), '| onset source', SRC_KERNEL, used)
