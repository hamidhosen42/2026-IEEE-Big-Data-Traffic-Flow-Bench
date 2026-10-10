"""Function test of kaggle_jobs/t1_catb_swap.py on a synthetic /kaggle/input tree (argv[1]): 2 panels x 2 days of state rows,
queue + path rows, all member outputs (old bag 8, FD bag 6, t1bspec a/b/c, t1catb a/b/c/d with panels split over the parts),
one Task 2 window per panel. Checks: ordinary rows = (1 - W_CATB) * mix + W_CATB * catb, blackout rows =
(1 - W_DARK) * that + W_DARK * bspec, queue / path rows untouched, output written."""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
TMP = os.path.abspath(sys.argv[1]); IN = os.path.join(TMP, 'input'); OUTD = os.path.join(TMP, 'working')
rng = np.random.default_rng(0)
REL = f'{IN}/data/kaggle_public'
os.makedirs(REL, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
panels = ['D7_I10_E', 'D12_I5_S']
rows = []
for p in panels:
    for day in ('2031-03-02', '2031-03-03'):
        for t in range(288):
            for s in range(3):
                rows.append((p, f'{day}T{t // 12:02d}:{(t % 12) * 5:02d}:00Z', f'{p}_s{s}', f'{p}_l{s}', 'R2'))
st = pd.DataFrame(rows, columns=['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'])
st.insert(0, 'task', 'state')
q = pd.DataFrame({'task': 'queue', 'panel': panels[0], 'timestamp': '2031-03-02T10:00:00Z', 'station_id': '', 'link_id': 'x', 'mask_regime': ''}, index=range(5))
pf = pd.DataFrame({'task': 'path_flow', 'panel': panels[0], 'timestamp': '', 'station_id': '', 'link_id': '', 'mask_regime': ''}, index=range(3))
key = pd.concat([st, q, pf], ignore_index=True); key.insert(0, 'submission_id', np.arange(len(key)))
key.to_csv(f'{REL}/submission_key.csv', index=False)
key[['submission_id']].to_csv(f'{REL}/sample_submission.csv', index=False)
for p in panels:
    os.makedirs(f'{REL}/task2/{p}/validation', exist_ok=True)
    pd.DataFrame({'forecast_origin': ['2031-03-02T08:00:00Z']}).to_csv(f'{REL}/task2/{p}/validation/window_index.csv', index=False)
base = pd.DataFrame({'submission_id': key.submission_id, 'task': key.task, 'speed_kmh': 50.0, 'flow_vph': 1000.0,
                     'queue_pred': np.where(key.task == 'queue', 1, 0), 'path_flow': np.where(key.task == 'path_flow', 7.5, 0.0)})
os.makedirs(f'{IN}/tfb-j4', exist_ok=True); base.to_csv(f'{IN}/tfb-j4/submission_J4.csv', index=False)
src = open(os.path.join(ROOT, 'kaggle_jobs', 't1_catb_swap.py')).read()
ns0 = {}
exec(compile('\n'.join(l for l in src.split('\n') if l.startswith(('BASE_KERNEL', 'CATB_MEMBER', 'W_FD', 'W_DARK', 'DARK_MEMBER', 'FD_MEMBERS', 'MEMBERS'))), 'c', 'exec'), ns0)
K = ns0
STV = st.copy()


def member(slug, part_panels, v, qv):
    os.makedirs(f'{IN}/{slug}', exist_ok=True)
    m = STV[STV.panel.isin(part_panels)].drop(columns='task').assign(speed_kmh=v, flow_vph=qv)
    m.to_csv(f'{IN}/{slug}/state_submission_FTA.csv', index=False)


vals = {}
for k in K['MEMBERS'] + K['FD_MEMBERS']:
    vals[k] = (rng.uniform(40, 110), rng.uniform(500, 1500))
    member(f'{k}a', [panels[0]], *vals[k]); member(f'{k}b', [panels[1]], *vals[k])
vals['bs'] = (20.0, 900.0); member(f'{K["DARK_MEMBER"]}a', [panels[0]], 20.0, 900.0); member(f'{K["DARK_MEMBER"]}b', [panels[1]], 20.0, 900.0)
vmix = K['W_FD'] * np.mean([vals[k][0] for k in K['FD_MEMBERS']]) + (1 - K['W_FD']) * np.mean([vals[k][0] for k in K['MEMBERS']])
CB = vmix + 2.0                                            # within the swap's sanity bound (rms < 5 km/h)
for i, part in enumerate(K['CATB_PARTS']):                 # catb parts: panel 0 in a, panel 1 in c, b and d empty-but-present
    member(f'{K["CATB_MEMBER"]}{part}', [panels[0]] if part == 'a' else ([panels[1]] if part == 'c' else []), CB, 1100.0)
src = src.replace('/kaggle/input', IN).replace('/kaggle/working', OUTD)
exec(compile(src, 't1_catb_swap[test]', 'exec'), {'__name__': '__main__'})
out = pd.read_csv(f'{OUTD}/submission_K1D.csv')
vm = (1 - K['W_CATB']) * vmix + K['W_CATB'] * CB
ts = pd.to_datetime(key.timestamp.where(key.task == 'state'), utc=True)
o = pd.Timestamp('2031-03-02T08:00:00Z')
inb = (key.task == 'state') & (ts > o) & (ts <= o + pd.Timedelta(minutes=90))
ordn = (key.task == 'state') & ~inb
assert np.allclose(out.speed_kmh[ordn], vm), (out.speed_kmh[ordn].unique()[:3], vm)
assert np.allclose(out.speed_kmh[inb], (1 - K['W_DARK']) * vm + K['W_DARK'] * 20.0)
assert (out.queue_pred[key.task == 'queue'] == 1).all() and np.allclose(out.path_flow[key.task == 'path_flow'], 7.5)
print('SWAP FUNCTION TEST OK | ordinary', round(vm, 4), '| blackout rows', int(inb.sum()), '| W_CATB', K['W_CATB'])
