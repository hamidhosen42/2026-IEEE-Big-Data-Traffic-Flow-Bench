"""Function test of kaggle_jobs/swap_lrtc_k9.py (EXP_T1_K9_PREP, tfb-k9) on a synthetic /kaggle/input tree (argv[1]) built around the
REAL local release of D12_I405_S private (data/kaggle_public: network + mainline_states_masked, symlinked), exactly as
t1lrtc_kernel_fntest.py (K7): state rows = every released cell with a missing flow (a superset of the Task 1 targets), 5 queue rows,
3 path rows, one synthetic Task 2 window index, base tfb-k5d/submission_K5D.csv = per-station time interpolation of the released
cells (+ noise); every 3rd blackout row starts at 1.12 vf, so the step-B speed clip at 1.05 vf binds there.
Checks:
  1. steps M + B == an INDEPENDENT rebuild: speed / flow cubes pivoted by separate code, lab t1div_lrtc.lrtc (60 iterations), each
     station normalised by its own mean; ordinary rows q = max(0.65 q_base + 0.35 max(q_lrtc(24, 40, 10), 20), 20), speed unchanged;
     blackout rows v = clip(0.85 v_base + 0.15 clip(v_lrtc(8, 20, 20), 3, 1.05 vf), 3, 1.05 vf),
     q = max(0.85 q_base + 0.15 max(q_lrtc(8, 40, 10), 20), 20) (vf of fd_parameters.csv read here independently).
  2. steps R + T == the unchanged kaggle_jobs/swap_fdrecon_k5.py run on submission_LRTC.csv as its base (to CSV round-off).
  3. queue / path rows byte-identical to the base (also asserted inside the kernel)."""
import os, sys, glob, shutil, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
TMP = os.path.abspath(sys.argv[1]); shutil.rmtree(TMP, ignore_errors=True)
IN = os.path.join(TMP, 'input'); OUTD = os.path.join(TMP, 'working')
REL = f'{IN}/data/kaggle_public'
PANEL, SPLIT = 'D12_I405_S', 'private'
SRC_REL = os.path.join(ROOT, 'data', 'kaggle_public')
os.makedirs(f'{REL}/corridors/{PANEL}/{SPLIT}', exist_ok=True); os.makedirs(OUTD, exist_ok=True)
os.symlink(f'{SRC_REL}/corridors/{PANEL}/network', f'{REL}/corridors/{PANEL}/network')
os.symlink(f'{SRC_REL}/corridors/{PANEL}/{SPLIT}/mainline_states_masked', f'{REL}/corridors/{PANEL}/{SPLIT}/mainline_states_masked')
rng = np.random.default_rng(0)

# ---------------- released month (independent pivot) and synthetic key / base
files = sorted(glob.glob(f'{SRC_REL}/corridors/{PANEL}/{SPLIT}/mainline_states_masked/*/*.parquet'))
d = pd.concat([pd.read_parquet(f, columns=['date', 'timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'mask_regime']) for f in files],
              ignore_index=True)
d['station_id'] = d.station_id.astype(str); d['link_id'] = d.link_id.astype(str)
d['ts'] = pd.to_datetime(d.timestamp, utc=True)
days = sorted(d.date.astype(str).unique()); stations = sorted(d.station_id.unique())
d['di'] = d.date.astype(str).map({x: i for i, x in enumerate(days)}); d['slot'] = d.ts.dt.hour * 12 + d.ts.dt.minute // 5
d['si'] = d.station_id.map({x: i for i, x in enumerate(stations)})
Qc = np.full((len(days), 288, len(stations)), np.nan); Vc = Qc.copy()
fin = d.flow_vph.notna().to_numpy()
Qc[d.di[fin], d.slot[fin], d.si[fin]] = d.flow_vph[fin]
fv = d.speed_kmh.notna().to_numpy(); Vc[d.di[fv], d.slot[fv], d.si[fv]] = d.speed_kmh[fv]
tgt = d[d.flow_vph.isna()].sort_values(['ts', 'station_id']).reset_index(drop=True)
print('released month', PANEL, SPLIT, '| days', len(days), 'stations', len(stations), '| released share flow', round(float(np.isfinite(Qc).mean()), 3),
      'speed', round(float(np.isfinite(Vc).mean()), 3), '| synthetic state rows', len(tgt))
st = tgt[['timestamp', 'station_id', 'link_id', 'mask_regime']].assign(task='state', panel=PANEL)
st['mask_regime'] = st.mask_regime.astype(str)
q = pd.DataFrame({'task': 'queue', 'panel': PANEL, 'timestamp': '2031-04-02T10:00:00Z', 'station_id': '', 'link_id': 'x', 'mask_regime': ''}, index=range(5))
pf = pd.DataFrame({'task': 'path_flow', 'panel': PANEL, 'timestamp': '', 'station_id': '', 'link_id': '', 'mask_regime': ''}, index=range(3))
key = pd.concat([st, q, pf], ignore_index=True)[['task', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']]
key.insert(0, 'submission_id', np.arange(len(key)))
key.to_csv(f'{REL}/submission_key.csv', index=False)
key[['submission_id']].to_csv(f'{REL}/sample_submission.csv', index=False)
origins = ['2031-04-03T07:00:00Z', '2031-04-08T16:30:00Z', '2031-04-15T17:00:00Z', '2031-04-22T07:30:00Z', '2031-04-28T16:00:00Z']
os.makedirs(f'{REL}/task2/{PANEL}/{SPLIT}', exist_ok=True)
pd.DataFrame({'window_id': [f'w{i}' for i in range(5)], 'forecast_origin': origins}).to_csv(f'{REL}/task2/{PANEL}/{SPLIT}/window_index.csv', index=False)


def tinterp(C):
    out = C.copy()
    for di in range(C.shape[0]):
        for s in range(C.shape[2]):
            y = C[di, :, s]; ok = np.isfinite(y)
            if ok.any():
                out[di, :, s] = np.interp(np.arange(288), np.flatnonzero(ok), y[ok])
    return out


Vi, Qi = tinterp(Vc), tinterp(Qc)
td, tt, ts_ = tgt.di.to_numpy(), tgt.slot.to_numpy(), tgt.si.to_numpy()
vb = np.clip(np.nan_to_num(Vi[td, tt, ts_], nan=100.0) * rng.normal(1, 0.01, len(tgt)), 5, 130)
fdp = pd.read_csv(f'{SRC_REL}/corridors/{PANEL}/network/fd_parameters.csv', dtype={'link_id': str}).groupby('link_id').free_speed_kmh.mean()
vfl = tgt.link_id.map(fdp).to_numpy(float); assert np.isfinite(vfl).all()
tsr = pd.to_datetime(tgt.timestamp, utc=True)
blk = np.zeros(len(tgt), bool)
for o in pd.to_datetime(pd.Series(origins), utc=True):
    blk |= ((tsr > o) & (tsr <= o + pd.Timedelta(minutes=90))).to_numpy()
hi = np.flatnonzero(blk)[::3]; vb[hi] = 1.12 * vfl[hi]                # step-B clip test rows (0.85 * 1.12 vf + 0.15 v_lrtc > 1.05 vf)
qb = np.maximum(np.nan_to_num(Qi[td, tt, ts_], nan=3000.0) * rng.normal(1, 0.02, len(tgt)), 60)
n = len(key)
base = pd.DataFrame({'submission_id': key.submission_id, 'task': key.task, 'speed_kmh': 0.0, 'flow_vph': 0.0,
                     'queue_pred': np.where(key.task == 'queue', 1, 0), 'path_flow': np.where(key.task == 'path_flow', 7.5, 0.0)})
iss = (base.task == 'state').to_numpy()
base.loc[iss, 'speed_kmh'] = vb; base.loc[iss, 'flow_vph'] = qb
os.makedirs(f'{IN}/tfb-k5d', exist_ok=True); base.to_csv(f'{IN}/tfb-k5d/submission_K5D.csv', index=False)


def run_kernel(path, inp, outd):
    src = open(path).read().replace('/kaggle/input', inp).replace('/kaggle/working', outd)
    g = {'__name__': '__main__'}
    exec(compile(src, os.path.basename(path), 'exec'), g)
    return g


# ---------------- run tfb-k9
print('\n=== swap_lrtc_k9.py ===', flush=True)
t0 = time.time()
G = run_kernel(os.path.join(ROOT, 'kaggle_jobs', 'swap_lrtc_k9.py'), IN, OUTD)
print(f'kernel wall time {time.time() - t0:.1f}s')
lr = pd.read_csv(f'{OUTD}/submission_LRTC.csv'); fr = pd.read_csv(f'{OUTD}/submission_FDRTV.csv')

# ---------------- check 1: independent rebuild of steps M + B with the lab functions
import t1div_lrtc as T  # noqa: E402


def lab_member(C, ranks):
    with np.errstate(all='ignore'):
        mu = np.nanmean(C, (0, 1)); mu = np.where(np.isfinite(mu), mu, np.nanmean(mu))
        return T.lrtc(C / mu[None, None, :], ranks, iters=60)[0] * mu[None, None, :]


q2lab = np.maximum(lab_member(Qc, (24, 40, 10))[td, tt, ts_], 20.0)
qlab = np.maximum(lab_member(Qc, (8, 40, 10))[td, tt, ts_], 20.0)
vlab = np.clip(lab_member(Vc, (8, 20, 20))[td, tt, ts_], 3, 1.05 * vfl)
okq = np.isfinite(Qc).sum((0, 1))[ts_] >= 200; okv = np.isfinite(Vc).sum((0, 1))[ts_] >= 200
useM = ~blk & okq; useB = blk & okq & okv
exp_q = np.where(useM, np.maximum(0.65 * qb + 0.35 * q2lab, 20.0), qb)
exp_q = np.where(useB, np.maximum(0.85 * qb + 0.15 * qlab, 20.0), exp_q)
exp_v = np.where(useB, np.clip(0.85 * vb + 0.15 * vlab, 3, 1.05 * vfl), vb)
got_q = lr.loc[iss, 'flow_vph'].to_numpy(); got_v = lr.loc[iss, 'speed_kmh'].to_numpy()
rel_q = np.max(np.abs(got_q - exp_q) / np.maximum(exp_q, 1)); rel_v = np.max(np.abs(got_v - exp_v) / np.maximum(exp_v, 1))
print(f'\ncheck 1: blackout rows {int(blk.sum())} (blended {int(useB.sum())}, speed at the 1.05 vf clip {int((useB & (exp_v >= 1.05 * vfl - 1e-9)).sum())}) '
      f'| ordinary rows blended {int(useM.sum())} | max rel |kernel - lab rebuild|: q {rel_q:.2e} v {rel_v:.2e} | max abs: ordinary q '
      f'{np.abs(got_q - exp_q)[~blk].max():.2e} vph, blackout q {np.abs(got_q - exp_q)[blk].max():.2e} vph v {np.abs(got_v - exp_v)[blk].max():.2e} km/h '
      f'| ordinary speed max |dv| vs base {np.abs(got_v - vb)[~blk].max():.2e}')
assert rel_q < 1e-6 and rel_v < 1e-6, 'steps M + B differ from the lab rebuild'
assert np.abs(got_v - vb)[~blk].max() < 1e-6, 'step M / B changed an ordinary speed'
assert useB.sum() > 0 and (np.abs(got_v - vb)[useB] > 1e-6).mean() > 0.5, 'step B did not act on the blackout rows'
assert (useB & (exp_v >= 1.05 * vfl - 1e-9)).sum() >= 0.2 * useB.sum() and (got_v <= 1.05 * vfl + 1e-9)[useB].all(), 'clip not exercised / violated'
print('check 1 OK (steps M + B == lab lrtc on independently pivoted cubes; ordinary speed unchanged)')

# ---------------- check 2: R + T == unchanged swap_fdrecon_k5.py on submission_LRTC.csv
IN2 = os.path.join(TMP, 'input2'); OUT2 = os.path.join(TMP, 'working2'); os.makedirs(OUT2, exist_ok=True)
os.makedirs(f'{IN2}/tfb-k5d', exist_ok=True); os.symlink(f'{IN}/data', f'{IN2}/data')
shutil.copy(f'{OUTD}/submission_LRTC.csv', f'{IN2}/tfb-k5d/submission_K5D.csv')
print('\n=== swap_fdrecon_k5.py on submission_LRTC.csv ===', flush=True)
run_kernel(os.path.join(ROOT, 'kaggle_jobs', 'swap_fdrecon_k5.py'), IN2, OUT2)
ref = pd.read_csv(f'{OUT2}/submission_FDRTV.csv')
dv = np.abs(ref.speed_kmh.to_numpy() - fr.speed_kmh.to_numpy()).max(); dq = np.abs(ref.flow_vph.to_numpy() - fr.flow_vph.to_numpy()).max()
print(f'\ncheck 2: max |dv| {dv:.2e} max |dq| {dq:.2e} | rows changed by R+T vs LRTC: '
      f'{int(((fr.speed_kmh != lr.speed_kmh) | (fr.flow_vph != lr.flow_vph)).sum())}')
assert dv < 1e-9 and dq < 1e-6, 'R + T of tfb-k9 differ from swap_fdrecon_k5'   # the reference re-reads the LRTC file from CSV (round-off)
print('check 2 OK (R + T identical to swap_fdrecon_k5.py up to CSV round-off)')
# ---------------- check 3
assert (fr.queue_pred.to_numpy() == base.queue_pred.to_numpy()).all() and np.array_equal(fr.path_flow.to_numpy(), base.path_flow.to_numpy())
assert (fr.submission_id.to_numpy() == base.submission_id.to_numpy()).all()
print('check 3 OK (queue / path rows and id order unchanged)')
print('\nK9 FUNCTION TEST PASSED')
