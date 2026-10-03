"""T1 FD swap (derived from swap_combo15.py). Start from the exact base file (BASE_KERNEL/BASE_FILE, default COMBO15 = LB 0.86944);
only Task 1 rows change. Queue and Task 4 rows untouched.
  MODE 'probe': T1 rows = 0.5 * (base T1 rows = current bag mean) + 0.5 * (mean of the FD members listed; normally one: v13 seed 7).
  MODE 'fdbag': T1 rows = mean of the FD members listed (use only once >= 3 FD members exist).
FDR (swap_fdrecon, if it was kept) is NOT applied here: re-apply it on top of this output as a separate step.
LB KEEP: dLB >= +0.0002 for the probe, >= +0.0005 for the FD-only bag."""
import glob, numpy as np, pandas as pd
MODE = 'probe'                                       # 'probe' | 'fdbag'
BASE_KERNEL, BASE_FILE = 'tfb-combo15', 'submission_COMBO15.csv'
FD_KERNELS = (('tfb-t1v13a', 'tfb-t1v13b'),)         # one (half a, half b) pair per FD member; add ('tfb-t1v13c', 'tfb-t1v13d') etc. as they finish
OUT_NAME = 'T1FD_PROBE' if MODE == 'probe' else 'T1FD_BAG'
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


M = [load(pair, 'state_submission_FTA.csv') for pair in FD_KERNELS]
assert MODE in ('probe', 'fdbag')
assert MODE == 'probe' or len(M) >= 3, 'FD-only bag needs >= 3 FD members'
vfd, qfd = np.mean([m[0] for m in M], 0), np.mean([m[1] for m in M], 0)
vb, qb = base.loc[iss, 'speed_kmh'].to_numpy(), base.loc[iss, 'flow_vph'].to_numpy()
print('FD members', FD_KERNELS, '| base vs FD-mean rms speed', round(float(np.sqrt(np.mean((vb - vfd) ** 2))), 3),
      'flow', round(float(np.sqrt(np.mean((qb - qfd) ** 2))), 2))
for i in range(len(M)):
    for j in range(i + 1, len(M)):
        print('FD member disagreement', FD_KERNELS[i], FD_KERNELS[j], 'rms speed', round(float(np.sqrt(np.mean((M[i][0] - M[j][0]) ** 2))), 3))
if MODE == 'probe':
    vn, qn = 0.5 * vb + 0.5 * vfd, 0.5 * qb + 0.5 * qfd
else:
    vn, qn = vfd, qfd
out = base.copy()
out.loc[iss, 'speed_kmh'] = vn; out.loc[iss, 'flow_vph'] = qn
isq = (base.task == 'queue').to_numpy()
assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
iso = ~(iss | isq)
assert np.allclose(out.loc[iso, 'path_flow'].to_numpy(), base.loc[iso, 'path_flow'].to_numpy(), equal_nan=True)
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, MODE, 'OK', len(out), '| new vs base rms speed', round(float(np.sqrt(np.mean((vn - vb) ** 2))), 3))
