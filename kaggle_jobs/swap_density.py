"""Task 3 density blend (Hisernberg TV/reconciliation idea; local prototype blend 0.25 = best, +0.0005 score):
at Task 1 target cells, k_new = (1-A)*k_pred + A*k_interp, where k_interp is the per-station temporal interpolation of the RELEASED
density q/v of the same split (Task 1 is offline, any released value allowed). Speed kept, flow = k_new * speed. Queue / Task 4 untouched."""
import glob, re, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE, OUT_NAME, A = 'tfb-combo13', 'submission_COMBO13.csv', 'COMBO_DENS', 0.25
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
ks = key[key.task == 'state'][['submission_id', 'panel', 'timestamp', 'station_id']].copy()
ks['timestamp'] = pd.to_datetime(ks.timestamp, utc=True); ks['submission_id'] = ks.submission_id.astype('int64')
KI = []
for panel in sorted(ks.panel.unique()):
    for split in ('validation', 'private'):
        fs = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'))
        for f in fs:
            d = pd.read_parquet(f, columns=['timestamp', 'station_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
            d['station_id'] = d.station_id.astype(str); d['timestamp'] = pd.to_datetime(d.timestamp, utc=True)
            d['k'] = np.where(d.speed_kmh.notna() & d.flow_vph.notna(), d.flow_vph / d.speed_kmh.clip(lower=1), np.nan)
            pv = d.pivot_table(index='timestamp', columns='station_id', values='k', aggfunc='mean', dropna=False).sort_index()
            pv = pv.interpolate(axis=0, limit=3, limit_direction='both', limit_area='inside')   # short gaps only (no blackouts)
            s = pv.stack(future_stack=True).dropna().rename('ki').reset_index(); s['panel'] = panel
            KI.append(s)
KI = pd.concat(KI, ignore_index=True)
m = ks.merge(KI, on=['panel', 'timestamp', 'station_id'], how='left')
iss = (base.task == 'state').to_numpy(); sid = base.loc[iss, 'submission_id'].to_numpy()
ki = pd.Series(m.ki.to_numpy(), index=m.submission_id.to_numpy()).reindex(sid).to_numpy()
v = base.loc[iss, 'speed_kmh'].to_numpy(); q = base.loc[iss, 'flow_vph'].to_numpy()
kp = q / np.clip(v, 1, None)
ok = np.isfinite(ki)
kn = np.where(ok, (1 - A) * kp + A * ki, kp)
qn = np.maximum(kn * v, 20.0)
print('k_interp coverage', round(float(ok.mean()), 4), '| rms flow change', round(float(np.sqrt(np.mean((qn - q) ** 2))), 2),
      '| median |dk|/k', round(float(np.nanmedian(np.abs(kn - kp) / np.maximum(kp, 1e-6))), 4))
out = base.copy(); out.loc[iss, 'flow_vph'] = qn
isq = (base.task == 'queue').to_numpy()
assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert float((out.loc[iss, 'flow_vph'] < 50).mean()) < 0.01
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out))
