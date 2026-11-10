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
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-final.py (notebook) cell 7; patched lines end in [TFB]
SMOKE = False  # [TFB] cell 1 of tfb-final; strict gates always (the skeleton must match every odme key row)
# [TFB] In the entry script this stage writes the SKELETON submission: odme rows = the NNLS below (verbatim), state and queue
# [TFB] rows 0. t1_mix fills the state rows, assemble fills the queue rows and re-applies the state hard gates.

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = f'{TFB.out}/submission.csv'  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
KEYS = {'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
        'queue': ['window_id', 'timestamp', 'link_id'], 'odme': ['panel', 'departure_time', 'path_id']}
VALUES = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def find(name):
    return TFB.kfile('tfb-final', name)  # [TFB] unused in the entry script (no upstream state / queue tables)


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
odme.to_csv(f'{TFB.out}/odme_path_flow.csv', index=False)  # [TFB] the Task 4 table itself (raw NNLS output; parity / reuse)
# [TFB] pandas' default CSV float parser is not correctly rounded, so one to_csv + read_csv cycle g() is not idempotent:
# [TFB] some values move by an ulp on each of the first cycles. In the Kaggle chain path_flow went through 26+ cycles
# [TFB] (tfb-final -> ... -> tfb-j4) and ended on fixed points of g (tfb-i4 asserts g(x) == x). Writing that fixed point
# [TFB] here makes every later cycle of the entry script a no-op and reproduces the chain's values.
import io  # [TFB]


def _csv_cycle(v):  # [TFB] one write + read with pandas defaults, exactly as every swap kernel did
    return pd.read_csv(io.StringIO(pd.DataFrame({'path_flow': v}).to_csv(index=False)))['path_flow'].to_numpy()  # [TFB]


_pf = odme.path_flow.to_numpy(np.float64)  # [TFB]
_first = _csv_cycle(_pf)  # [TFB] = what tfb-final's own submission.csv parses to
for _it in range(1, 101):  # [TFB]
    _nx = _csv_cycle(_pf)  # [TFB]
    if np.array_equal(_nx, _pf):  # [TFB]
        break  # [TFB]
    _pf = _nx  # [TFB]
assert np.array_equal(_csv_cycle(_pf), _pf), f'path_flow has no CSV fixed point after {_it} cycles ({int((_csv_cycle(_pf) != _pf).sum())} values)'  # [TFB]
log('path_flow CSV fixed point after', _it - 1, 'cycles | values moved vs the first parse (tfb-final file):', int((_pf != _first).sum()),  # [TFB]
    '| max |d|', float(np.abs(_pf - _first).max()))  # [TFB]
odme['path_flow'] = _pf  # [TFB]

tables = {'odme': odme}  # [TFB] state / queue tables come from later stages; their rows stay 0 in the skeleton
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
if SMOKE:
    print('SMOKE: unmatched keys expected (panel/day subsets):', gaps)
else:
    assert all(v == 0 for v in gaps.values()), f'unmatched keys: {gaps}'
assert len(sub) == len(sample_ids) and (sub.submission_id.to_numpy() == sample_ids).all(), 'row set/order differs from sample_submission'
num = sub[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']]
assert np.isfinite(num.to_numpy()).all(), 'NaN/inf present'
assert (num >= 0).all().all(), 'negative values'
assert sub.loc[sub.task == 'queue', 'queue_pred'].isin([0, 1]).all(), 'non-binary queue_pred'
# [TFB] the empty-road guard on state flows (< 50 vph share <= 20 % per panel x regime) runs in the assemble stage
sub['queue_pred'] = sub.queue_pred.astype(int)
sub.to_csv(OUT, index=False)
log('WROTE', OUT, len(sub), 'rows')
print(sub.groupby('task')[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].describe().T.round(3).to_string())
