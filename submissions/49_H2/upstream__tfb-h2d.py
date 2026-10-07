"""t1dark swap = T1 MIX swap + BLACKOUT-ROW override (raw, BEFORE FD recon/TV).
T1 rows = W_FD * mean(FD-feature members v13) + (1-W_FD) * mean(old FTA bag) (= swap_t1mix.py); then on the blackout rows
(state rows whose timestamp lies in (origin, origin + 90 min] of an official Task 2 window of that panel, validation or
private; about 2% of the Task 1 targets) T1 = (1 - W_DARK) * T1 + W_DARK * t1dark member (kernels <DARK_MEMBER>a/b =
kaggle_jobs/t1_t1dark_member_a/b.py, trained with simulated-blackout transductive rows). Regular rows are unchanged.
Previous notes (swap_t1mix.py):
T1 MIX swap (raw, BEFORE FD recon/TV): T1 rows = W_FD * mean(FD-feature members v13) + (1-W_FD) * mean(old FTA bag).
Queue and Task 4 rows come from the base file unchanged. The output is meant to be passed through swap_fdrecon (DO_RECON+DO_TV)."""
import glob, os, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE, OUT_NAME = 'tfb-h1', 'submission_FDRTV.csv', 'T1DARK'
W_FD = 0.5
W_DARK = 0.5
DARK_MEMBER = 'tfb-t1ramp'
FD_MEMBERS = ('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11')
MEMBERS = ('tfb-t1v11', 'tfb-t1v12', 'tfb-t1v12s2', 'tfb-t1v12s3', 'tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7')
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


M = [load((f'{k}a', f'{k}b'), 'state_submission_FTA.csv') for k in MEMBERS]
MF = [load((f'{k}a', f'{k}b'), 'state_submission_FTA.csv') for k in FD_MEMBERS]
print('old members', len(M), '| FD members', len(MF), FD_MEMBERS, '| W_FD', W_FD)
vo, qo = np.mean([m[0] for m in M], 0), np.mean([m[1] for m in M], 0)
vf_, qf_ = np.mean([m[0] for m in MF], 0), np.mean([m[1] for m in MF], 0)
print('rms speed old-bag vs FD-bag', round(float(np.sqrt(np.mean((vo - vf_) ** 2))), 3), '| flow', round(float(np.sqrt(np.mean((qo - qf_) ** 2))), 2))
print('info: base (post FDR/TV) vs old bag rms speed', round(float(np.sqrt(np.mean((base.loc[iss, 'speed_kmh'].to_numpy() - vo) ** 2))), 3))
out = base.copy()
out.loc[iss, 'speed_kmh'] = W_FD * vf_ + (1 - W_FD) * vo; out.loc[iss, 'flow_vph'] = W_FD * qf_ + (1 - W_FD) * qo
# ---- t1dark: blackout rows (timestamp in (origin, origin + 90 min] of an official window of the panel) <- dark member
vd, qd = load((f'{DARK_MEMBER}a', f'{DARK_MEMBER}b'), 'state_submission_FTA.csv')
kk = ks.set_index(ks.submission_id.astype('int64')).reindex(sid)
assert kk.panel.notna().all(), 'state rows without a key'
inb = np.zeros(len(sid), bool)
for panel in kk.panel.unique():
    pm = (kk.panel == panel).to_numpy(); tsp = kk.timestamp[pm]; hit = np.zeros(int(pm.sum()), bool)
    for split in ('validation', 'private'):
        wi_p = f'{REL}/task2/{panel}/{split}/window_index.csv'
        if os.path.exists(wi_p):
            for o in pd.to_datetime(pd.read_csv(wi_p).forecast_origin, utc=True):
                hit |= ((tsp > o) & (tsp <= o + pd.Timedelta(minutes=90))).to_numpy()
    inb[np.flatnonzero(pm)[hit]] = True
assert 0.005 < inb.mean() < 0.05, f'blackout share {inb.mean():.4f} outside the expected range'
v_mix = out.loc[iss, 'speed_kmh'].to_numpy(); q_mix = out.loc[iss, 'flow_vph'].to_numpy()
print('t1dark blackout rows', int(inb.sum()), 'share of state rows', round(float(inb.mean()), 4), '| W_DARK', W_DARK,
      '| rms dark vs mix on blackout rows: speed', round(float(np.sqrt(np.mean((vd[inb] - v_mix[inb]) ** 2))), 3),
      'flow', round(float(np.sqrt(np.mean((qd[inb] - q_mix[inb]) ** 2))), 2),
      '| regular rows: speed', round(float(np.sqrt(np.mean((vd[~inb] - v_mix[~inb]) ** 2))), 3))
out.loc[iss, 'speed_kmh'] = np.where(inb, (1 - W_DARK) * v_mix + W_DARK * vd, v_mix)
out.loc[iss, 'flow_vph'] = np.where(inb, (1 - W_DARK) * q_mix + W_DARK * qd, q_mix)
isq = (base.task == 'queue').to_numpy()
assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out))
