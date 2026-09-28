"""Assemble the upload file: Task 1 + Task 2 from upstream kernels, Task 4 solved here, joined on submission_key.

Task 4 = the official prior-regularised NNLS on each split's OWN released counts and OWN weak prior
(lambda = 0.05), which is what the base notebook ships.
Hard gates (the kernel fails rather than writing a bad file): exact row count and id order versus
sample_submission.csv, zero unmatched keys in every task, no NaN/inf, queue in {0,1}, non-negative
values, and fewer than 20% of state flows below 50 vph in every panel x regime.
"""
import glob, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working/submission.csv'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
KEYS = {'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
        'queue': ['window_id', 'timestamp', 'link_id'], 'odme': ['panel', 'departure_time', 'path_id']}
VALUES = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def find(name):
    hits = [h for h in glob.glob(f'/kaggle/input/**/{name}', recursive=True) if 'kaggle_public' not in h]
    assert len(hits) == 1, f'expected exactly one upstream {name}, found {hits}'
    return hits[0]


# ---------------- Task 4
rows = []
for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    paths = pd.read_csv(f'{net}/path_set.csv', dtype={'path_id': str})
    inc = pd.read_csv(f'{net}/path_link_incidence.csv', dtype=str)
    pids = paths.path_id.tolist(); lids = inc.link_id.drop_duplicates().tolist()
    A = np.zeros((len(lids), len(pids)))
    A[inc.link_id.map({l: i for i, l in enumerate(lids)}), inc.path_id.map({p: i for i, p in enumerate(pids)})] = 1
    for split in ('validation', 'private'):
        c = pd.read_csv(f'{REL}/task4/{panel}/{split}/synthetic_link_counts.csv', dtype={'link_id': str})
        pr = pd.read_csv(f'{REL}/task4/{panel}/{split}/synthetic_weak_prior.csv', dtype={'path_id': str})
        b = pr.set_index('path_id').reindex(pids).path_flow.fillna(0).to_numpy()
        meas = np.array([l in set(c.link_id) for l in lids])
        cnt = c.set_index('link_id').reindex(np.array(lids)[meas])['count'].to_numpy()
        lam = 0.05
        f, _ = nnls(np.vstack([A[meas], np.sqrt(lam) * np.eye(len(pids))]), np.r_[cnt, np.sqrt(lam) * b])
        s_link = 1 - np.abs(A[meas] @ f - cnt).sum() / cnt.sum()
        dep = str(pr.departure_time.iloc[0])
        rows.append(pd.DataFrame({'panel': panel, 'departure_time': dep, 'path_id': pids, 'path_flow': f}))
        log(panel, split, 'paths', len(pids), 'S_link(own counts)', round(s_link, 6), 'dev/prior', round(np.abs(f - b).sum() / b.sum(), 4))
odme = pd.concat(rows, ignore_index=True)

tables = {'state': pd.read_csv(find('state_submission.csv')), 'queue': pd.read_csv(find('queue_submission_blend70.csv')), 'odme': odme}
for task, fr in tables.items():
    fr = fr[KEYS[task] + VALUES[task]].copy()
    for col in KEYS[task]:
        fr[col] = pd.to_datetime(fr[col], utc=True) if col == 'timestamp' else fr[col].astype(str)
    assert not fr.duplicated(KEYS[task]).any(), f'{task}: duplicate keys'
    tables[task] = fr.set_index(KEYS[task])
    log(task, 'rows', len(fr))

# ---------------- merge on submission_key (streamed, template order)
sample_ids = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
parts, gaps = [], {t: 0 for t in tables}
for chunk in pd.read_csv(f'{REL}/submission_key.csv', chunksize=1_000_000, dtype=str):
    chunk['timestamp'] = pd.to_datetime(chunk.timestamp, utc=True, errors='coerce')
    out = pd.DataFrame({'submission_id': chunk.submission_id.astype('int64').to_numpy(), 'task': chunk.task.to_numpy()})
    for col in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'):
        out[col] = 0.0
    for task, table in tables.items():
        m = (out.task == task).to_numpy()
        if not m.any():
            continue
        found = table.reindex(pd.MultiIndex.from_frame(chunk.loc[m, KEYS[task]]))
        for col in VALUES[task]:
            v = found[col].to_numpy(); gaps[task] += int(np.isnan(v).sum())
            out.loc[m, col] = np.nan_to_num(v)
    parts.append(out)
sub = pd.concat(parts, ignore_index=True)
log('unmatched keys per task', gaps)

# ---------------- hard gates
assert all(v == 0 for v in gaps.values()), f'unmatched keys: {gaps}'
assert len(sub) == len(sample_ids) and (sub.submission_id.to_numpy() == sample_ids).all(), 'row set/order differs from sample_submission'
num = sub[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']]
assert np.isfinite(num.to_numpy()).all(), 'NaN/inf present'
assert (num >= 0).all().all(), 'negative values'
assert sub.loc[sub.task == 'queue', 'queue_pred'].isin([0, 1]).all(), 'non-binary queue_pred'
st = tables['state'].reset_index()
low = st.assign(low=st.flow_vph < 50).groupby(['panel', 'mask_regime']).low.mean()
assert (low <= 0.2).all(), f'S_FD empty-road guard would fire: {low[low > 0.2]}'
sub['queue_pred'] = sub.queue_pred.astype(int)
sub.to_csv(OUT, index=False)
log('WROTE', OUT, len(sub), 'rows')
print(sub.groupby('task')[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].describe().T.round(3).to_string())
