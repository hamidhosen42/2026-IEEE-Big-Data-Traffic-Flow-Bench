"""COMBO_9: start from the exact COMBO8 file (LB 0.86537); swap ONLY Task 1 rows with T1 v9 FT (transductive rows; val-month holdout 0.9406 vs F 0.9386, 10/10 panels)."""
import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob('/kaggle/input/**/tfb-combo8/**/submission_COMBO8.csv', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
# ---- Task 1 rows
st = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t1v9-full/**/state_submission_FT.csv', recursive=True)[0], dtype={'station_id': str, 'link_id': str})
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
# CONTROL: the v9 'F' variant must reproduce the COMBO8 Task 1 rows (= v8 F) closely
stF = pd.read_csv(glob.glob('/kaggle/input/**/tfb-t1v9-full/**/state_submission_F.csv', recursive=True)[0], dtype={'station_id': str, 'link_id': str})
stF['timestamp'] = pd.to_datetime(stF.timestamp, utc=True)
mF = ks.merge(stF, on=['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'], how='left', validate='one_to_one')
fv = pd.Series(mF.speed_kmh.to_numpy(), index=mF.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
print('CONTROL: rms speed diff v9-F vs COMBO8 T1 rows', round(float(np.sqrt(np.nanmean((fv - base.loc[iss, 'speed_kmh'].to_numpy()) ** 2))), 4))
both = t1
isq = (base.task == 'queue').to_numpy()
assert (both.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(both) == len(sample) and (both.submission_id.to_numpy() == sample).all()
assert np.isfinite(both[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert (both[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
both['queue_pred'] = both.queue_pred.astype(int)
both.to_csv('/kaggle/working/submission_COMBO9.csv', index=False)
print('COMBO9 OK', len(both))
