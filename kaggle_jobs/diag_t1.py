"""Compare base (EXP_000) and our (SUB_A) Task 1 predictions on the validation+private template."""
import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
subs = {p.split('/')[-2]: p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True) if 'kaggle_public' not in p}
print(subs)
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
key = key[key.task == 'state'][['submission_id', 'split', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']]
key['submission_id'] = key.submission_id.astype('int64')
cols = ['submission_id', 'speed_kmh', 'flow_vph']
A = None
for name, p in subs.items():
    s = pd.read_csv(p, usecols=cols + ['task']); s = s[s.task == 'state'][cols]
    s.columns = ['submission_id', f'v_{name}', f'q_{name}']
    A = s if A is None else A.merge(s, on='submission_id')
A = A.merge(key, on='submission_id')
ts = pd.to_datetime(A.timestamp, utc=True)
A['gap'] = False
for panel in A.panel.unique():
    for split in ('validation', 'private'):
        f = f'{REL}/task2/{panel}/{split}/window_index.csv'
        try: wi = pd.read_csv(f)
        except FileNotFoundError: continue
        m = (A.panel == panel) & (A.split == split)
        for o in pd.to_datetime(wi.forecast_origin, utc=True):
            A.loc[m & (ts > o) & (ts <= o + pd.Timedelta(minutes=90)), 'gap'] = True
names = list(subs)
b, o = names[0], names[1]
fd = pd.concat([pd.read_csv(f'{REL}/corridors/{p}/network/fd_parameters.csv').assign(panel=p) for p in A.panel.unique()])
A = A.merge(fd[['panel', 'link_id', 'lanes', 'free_speed_kmh']], on=['panel', 'link_id'], how='left')
A['dv'] = A[f'v_{o}'] - A[f'v_{b}']; A['dq'] = (A[f'q_{o}'] - A[f'q_{b}']) / A.lanes
print('gap share of state rows', A.gap.mean().round(4))
print(A.groupby('gap')[['dv', 'dq']].agg(lambda x: np.sqrt(np.mean(x ** 2))).round(3).rename(columns={'dv': 'rms speed diff', 'dq': 'rms flow/lane diff'}))
for n in names:
    r = A[f'v_{n}'] / A.free_speed_kmh
    print(n, 'speed > 1.05 vf share', round(float((r > 1.05).mean()), 5), 'max ratio', round(float(r.max()), 3),
          'speed<0.6vf share', round(float((r < 0.6).mean()), 4))
print(A.groupby(['split', 'gap'])[[f'v_{b}', f'v_{o}']].mean().round(2))
