"""Function test of kaggle_jobs/t1_div2_swap.py (EXP_T1_XGB, K5D) on a synthetic /kaggle/input tree (argv[1]): 2 panels x 2 days
of state rows, queue + path rows, base tfb-k3/submission_FDRTV.csv, all member outputs (old bag 8, FD bag 6, t1bspec a/b,
CatBoost seeds tfb-t1catb / tfb-t1catb2 a..d and the candidate tfb-t1xgb a..d with panels split over the parts), one Task 2
window per panel. Checks (speed AND flow): ordinary rows = (1 - W_CAND) * ((1 - W_CATB) * mix + W_CATB * mean(catb seeds))
+ W_CAND * xgb; blackout rows = (1 - W_DARK) * that + W_DARK * bspec; queue / path rows untouched; output written.
Same synthetic layout as t1catb_swap_fntest.py."""
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
os.makedirs(f'{IN}/tfb-k3', exist_ok=True); base.to_csv(f'{IN}/tfb-k3/submission_FDRTV.csv', index=False)
src = open(os.path.join(ROOT, 'kaggle_jobs', 't1_div2_swap.py')).read()
K = {}
exec(compile('\n'.join(l for l in src.split('\n') if l.startswith(('BASE_KERNEL', 'CATB_MEMBER', 'CAND_MEMBER', 'W_FD', 'W_DARK',
                                                                     'DARK_MEMBER', 'FD_MEMBERS', 'MEMBERS'))), 'c', 'exec'), K)
assert K['BASE_KERNEL'] == 'tfb-k3' and K['CATB_MEMBER'] == 'tfb-t1catb+tfb-t1catb2' and K['CAND_MEMBER'] == 'tfb-t1xgb', K
STV = st.copy()


def member(slug, part_panels, v, qv):
    os.makedirs(f'{IN}/{slug}', exist_ok=True)
    m = STV[STV.panel.isin(part_panels)].drop(columns='task').assign(speed_kmh=v, flow_vph=qv)
    m.to_csv(f'{IN}/{slug}/state_submission_FTA.csv', index=False)


def parts4(slug, v, qv):                                   # panel 0 in part a, panel 1 in part c, b and d empty-but-present
    for part in 'abcd':
        member(f'{slug}{part}', [panels[0]] if part == 'a' else ([panels[1]] if part == 'c' else []), v, qv)


vals = {}
for k in K['MEMBERS'] + K['FD_MEMBERS']:
    vals[k] = (rng.uniform(40, 110), rng.uniform(500, 1500))
    member(f'{k}a', [panels[0]], *vals[k]); member(f'{k}b', [panels[1]], *vals[k])
member(f'{K["DARK_MEMBER"]}a', [panels[0]], 20.0, 900.0); member(f'{K["DARK_MEMBER"]}b', [panels[1]], 20.0, 900.0)
mix = [K['W_FD'] * np.mean([vals[k][i] for k in K['FD_MEMBERS']]) + (1 - K['W_FD']) * np.mean([vals[k][i] for k in K['MEMBERS']]) for i in (0, 1)]
C1, C2, XG = (mix[0] + 2.0, mix[1] + 40.0), (mix[0] + 1.0, mix[1] - 20.0), (mix[0] - 1.5, mix[1] + 30.0)   # within the rms < 5 km/h bounds
parts4('tfb-t1catb', *C1); parts4('tfb-t1catb2', *C2); parts4('tfb-t1xgb', *XG)
src = src.replace('/kaggle/input', IN).replace('/kaggle/working', OUTD)
exec(compile(src, 't1_div2_swap[test]', 'exec'), {'__name__': '__main__'})
out = pd.read_csv(f'{OUTD}/submission_K5D.csv')
W_CATB, W_CAND = 0.35, 0.35
m1 = [(1 - W_CATB) * mix[i] + W_CATB * 0.5 * (C1[i] + C2[i]) for i in (0, 1)]
m2 = [(1 - W_CAND) * m1[i] + W_CAND * XG[i] for i in (0, 1)]
ts = pd.to_datetime(key.timestamp.where(key.task == 'state'), utc=True)
o = pd.Timestamp('2031-03-02T08:00:00Z')
inb = (key.task == 'state') & (ts > o) & (ts <= o + pd.Timedelta(minutes=90))
ordn = (key.task == 'state') & ~inb
for i, c, bs in ((0, 'speed_kmh', 20.0), (1, 'flow_vph', 900.0)):
    assert np.allclose(out[c][ordn], m2[i]), (c, out[c][ordn].unique()[:3], m2[i])
    assert np.allclose(out[c][inb], (1 - K['W_DARK']) * m2[i] + K['W_DARK'] * bs), c
assert (out.queue_pred[key.task == 'queue'] == 1).all() and np.allclose(out.path_flow[key.task == 'path_flow'], 7.5)
assert (out.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
print('K5D SWAP FUNCTION TEST OK | ordinary speed', round(m2[0], 4), 'flow', round(m2[1], 3), '| blackout rows', int(inb.sum()),
      '| W_CATB', W_CATB, 'W_CAND', W_CAND)
