"""T1 BAG swap: start from the exact base file (BASE_KERNEL/BASE_FILE); Task 1 rows = mean of T1 v11 FTA (seed 0) and
T1 v12 FTA (seed 1, different transductive hidden rounds and row subsample). Queue and Task 4 rows untouched."""
import glob, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE, OUT_NAME = 'tfb-combo11', 'submission_COMBO11.csv', 'COMBO12'
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
ks = key[key.task == 'state'][['submission_id', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
ks['timestamp'] = pd.to_datetime(ks.timestamp, utc=True)
iss = (base.task == 'state').to_numpy(); sid = base.loc[iss, 'submission_id'].to_numpy()


def load(kernels, fname):
    st = pd.concat([pd.read_csv(glob.glob(f'/kaggle/input/**/{k}/**/{fname}', recursive=True)[0], dtype={'station_id': str, 'link_id': str})
                    for k in kernels], ignore_index=True)
    st['timestamp'] = pd.to_datetime(st.timestamp, utc=True)
    m = ks.merge(st, on=['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'], how='left', validate='one_to_one')
    assert m.speed_kmh.notna().all() and m.flow_vph.notna().all(), 'unmatched state keys'
    idx = m.submission_id.astype('int64').to_numpy()
    return (pd.Series(m.speed_kmh.to_numpy(), index=idx).reindex(sid).to_numpy(),
            pd.Series(m.flow_vph.to_numpy(), index=idx).reindex(sid).to_numpy())


v0, q0 = load(('tfb-t1v11a', 'tfb-t1v11b'), 'state_submission_FTA.csv')
v1, q1 = load(('tfb-t1v12a', 'tfb-t1v12b'), 'state_submission_FTA.csv')
print('CONTROL: base T1 rows vs v11 FTA rms speed', round(float(np.sqrt(np.mean((base.loc[iss, 'speed_kmh'].to_numpy() - v0) ** 2))), 4))
print('member disagreement rms speed', round(float(np.sqrt(np.mean((v0 - v1) ** 2))), 3), 'flow', round(float(np.sqrt(np.mean((q0 - q1) ** 2))), 2))
out = base.copy()
out.loc[iss, 'speed_kmh'] = 0.5 * (v0 + v1); out.loc[iss, 'flow_vph'] = 0.5 * (q0 + q1)
isq = (base.task == 'queue').to_numpy()
assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out))
