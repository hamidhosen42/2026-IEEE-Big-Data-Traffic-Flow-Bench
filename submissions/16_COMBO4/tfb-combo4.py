"""COMBO_4 (T1 v7 E big model): start from the exact 0.85639 file (v9); swap Task 1 rows (T1 v6 D: extended low-rank) and Task 2 rows (v11: v9 + CatBoost onset)."""
import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob('/kaggle/input/**/tfb-swap-v9/**/submission_main.csv', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
# ---- Task 1 rows
st = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t1v7-full/**/state_submission_E.csv', recursive=True)[0], dtype={'station_id': str, 'link_id': str})
ks = key[key.task == 'state'][['submission_id', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
for d in (st, ks):
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True)
m = ks.merge(st, on=['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'], how='left', validate='one_to_one')
assert m.speed_kmh.notna().all() and m.flow_vph.notna().all(), 'unmatched state keys'
sv = pd.Series(m.speed_kmh.to_numpy(), index=m.submission_id.astype('int64').to_numpy())
sq = pd.Series(m.flow_vph.to_numpy(), index=m.submission_id.astype('int64').to_numpy())
iss = (base.task == 'state').to_numpy(); sid = base.loc[iss, 'submission_id'].to_numpy()
t1 = base.copy()
t1.loc[iss, 'speed_kmh'] = sv.reindex(sid).to_numpy(); t1.loc[iss, 'flow_vph'] = sq.reindex(sid).to_numpy()
low = (t1.loc[iss, 'flow_vph'] < 50).mean()
print('T1 swap: rms speed change', round(float(np.sqrt(np.mean((t1.loc[iss, 'speed_kmh'] - base.loc[iss, 'speed_kmh']) ** 2))), 3),
      'rms flow change', round(float(np.sqrt(np.mean((t1.loc[iss, 'flow_vph'] - base.loc[iss, 'flow_vph']) ** 2))), 2), 'flow<50 share', round(float(low), 5))
# ---- Task 2 rows from v10 main
q = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t2v12/**/queue_submission.csv', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
for d in (q, kq):
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True)
mq = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert mq.queue_pred.notna().all()
qv = pd.Series(mq.queue_pred.astype(int).to_numpy(), index=mq.submission_id.astype('int64').to_numpy())
isq = (base.task == 'queue').to_numpy()
both = t1.copy(); both.loc[isq, 'queue_pred'] = qv.reindex(base.loc[isq, 'submission_id'].to_numpy()).to_numpy()
qc = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t2v12/**/queue_submission_v11.csv', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
qc['timestamp'] = pd.to_datetime(qc.timestamp, utc=True)
mc = kq.merge(qc, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
ctrl = pd.Series(mc.queue_pred.astype(int).to_numpy(), index=mc.submission_id.astype('int64').to_numpy()).reindex(base.loc[isq, 'submission_id'].to_numpy()).to_numpy()
q11 = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t2v11/**/queue_submission.csv', recursive=True)[0], dtype={'window_id': str, 'link_id': str}); q11['timestamp'] = pd.to_datetime(q11.timestamp, utc=True)
m11 = kq.merge(q11, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
ref = pd.Series(m11.queue_pred.astype(int).to_numpy(), index=m11.submission_id.astype('int64').to_numpy()).reindex(base.loc[isq, 'submission_id'].to_numpy()).to_numpy()
print('CONTROL: v12 v11-variant vs v11 main agreement', round(float((ctrl == ref).mean()), 5))
print('info: v12 v11-variant vs 0.85639 queue rows agreement', round(float((ctrl == base.loc[isq, 'queue_pred'].to_numpy()).mean()), 5))
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
for name, df in (('COMBO4', both),):
    assert len(df) == len(sample) and (df.submission_id.to_numpy() == sample).all()
    assert np.isfinite(df[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
    assert (df[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all() and df.loc[isq, 'queue_pred'].isin([0, 1]).all()
    df['queue_pred'] = df.queue_pred.astype(int)
    df.to_csv(f'/kaggle/working/submission_{name}.csv', index=False)
    print(name, 'OK', len(df), 'queue agreement with 0.85639', round(float((df.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).mean()), 4))
