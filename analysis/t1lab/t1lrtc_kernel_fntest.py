"""Function test of kaggle_jobs/swap_lrtc_k7.py (EXP_T1_DIV4, tfb-k7) on a synthetic /kaggle/input tree (argv[1]) built around the
REAL local release of D12_I405_S private (data/kaggle_public: network + mainline_states_masked, symlinked): state rows = every
released cell with a missing flow (a superset of the Task 1 targets), 5 queue rows, 3 path rows, one synthetic Task 2 window index,
base tfb-k5d/submission_K5D.csv = per-station time interpolation of the released cells (+ noise).
Checks:
  1. step M == an INDEPENDENT rebuild: flow cube pivoted by separate code, lab t1div_lrtc.lrtc / norm_view (ranks (8, 40, 10),
     60 iterations); ordinary rows q = 0.65 q_base + 0.35 max(q_lrtc, 20); speed and blackout rows unchanged.
  2. steps R + T == the unchanged kaggle_jobs/swap_fdrecon_k5.py run on submission_LRTC.csv as its base (to CSV round-off).
  3. queue / path rows byte-identical to the base (also asserted inside the kernel)."""
import os, sys, glob, shutil
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
print('released month', PANEL, SPLIT, '| days', len(days), 'stations', len(stations), '| flow released share', round(float(np.isfinite(Qc).mean()), 3),
      '| synthetic state rows', len(tgt))
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


# ---------------- run tfb-k7
print('\n=== swap_lrtc_k7.py ===', flush=True)
G = run_kernel(os.path.join(ROOT, 'kaggle_jobs', 'swap_lrtc_k7.py'), IN, OUTD)
lr = pd.read_csv(f'{OUTD}/submission_LRTC.csv'); fr = pd.read_csv(f'{OUTD}/submission_FDRTV.csv')

# ---------------- check 1: independent rebuild of step M with the lab functions
import t1div_lrtc as T  # noqa: E402
with np.errstate(all='ignore'):
    mu = np.nanmean(Qc, (0, 1)); mu = np.where(np.isfinite(mu), mu, np.nanmean(mu))
    Zl, _ = T.lrtc(Qc / mu[None, None, :], (8, 40, 10), iters=60)
qlab = np.maximum(Zl[td, tt, ts_] * mu[ts_], 20.0)
tsr = pd.to_datetime(tgt.timestamp, utc=True)
blk = np.zeros(len(tgt), bool)
for o in pd.to_datetime(pd.Series(origins), utc=True):
    blk |= ((tsr > o) & (tsr <= o + pd.Timedelta(minutes=90))).to_numpy()
nobs = np.isfinite(Qc).sum((0, 1)); okst = nobs[ts_] >= 200
exp_q = np.where(~blk & okst, 0.65 * qb + 0.35 * qlab, qb)
got_q = lr.loc[iss, 'flow_vph'].to_numpy(); got_v = lr.loc[iss, 'speed_kmh'].to_numpy()
rel_err = np.max(np.abs(got_q - exp_q) / np.maximum(exp_q, 1))
print(f'\ncheck 1: blackout rows {int(blk.sum())} | max rel |q_kernel - q_lab-rebuild| {rel_err:.2e} | speed max |dv| {np.abs(got_v - vb).max():.2e} '
      f'| blackout-row flow max |dq| {np.abs(got_q[blk] - qb[blk]).max():.2e} | rows blended {int((~blk & okst).sum())}')
assert rel_err < 1e-6, 'step M differs from the lab rebuild'
assert np.abs(got_v - vb).max() < 1e-6 and np.abs(got_q[blk] - qb[blk]).max() < 1e-6
print('check 1 OK (step M == lab lrtc on an independently pivoted cube; speed and blackout rows unchanged)')

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
assert dv < 1e-9 and dq < 1e-6, 'R + T of tfb-k7 differ from swap_fdrecon_k5'   # the reference re-reads the LRTC file from CSV (round-off)
print('check 2 OK (R + T identical to swap_fdrecon_k5.py up to CSV round-off)')
# ---------------- check 3
assert (fr.queue_pred.to_numpy() == base.queue_pred.to_numpy()).all() and np.array_equal(fr.path_flow.to_numpy(), base.path_flow.to_numpy())
assert (fr.submission_id.to_numpy() == base.submission_id.to_numpy()).all()
print('check 3 OK (queue / path rows and id order unchanged)')
print('\nK7 FUNCTION TEST PASSED')
