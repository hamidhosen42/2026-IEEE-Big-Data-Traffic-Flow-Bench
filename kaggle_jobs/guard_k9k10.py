"""Cross-generation guard for the k9 chain (adapted 2026-11-03 from an external review script, tfb_submit_guard.py): reads the
exact output CSVs of tfb-k7 / tfb-k8 / tfb-k9 / tfb-k10 (kernel sources) and checks that k9 changes only state rows, that
k10 = k9 + exactly the April queue delta of k8 - k7, and that March queue rows and ODME rows are unchanged everywhere.
Prints 'FINAL: CROSS-GENERATION SUBMISSION GUARD PASSED' or stops at the first failing gate."""
import glob
import numpy as np, pandas as pd

FILES = {'k7': ('tfb-k7', 'submission_FDRTV.csv'), 'k8': ('tfb-k8', 'submission_K8.csv'),
         'k9': ('tfb-k9', 'submission_FDRTV.csv'), 'k10': ('tfb-k10', 'submission_K10.csv')}
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
cols = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']
S = {}
for n, (kern, fn) in FILES.items():
    d = pd.read_csv(glob.glob(f'/kaggle/input/**/{kern}/**/{fn}', recursive=True)[0], usecols=cols)
    assert d.submission_id.is_unique, f'{n}: duplicate submission ID'
    assert np.isfinite(d[cols[2:]].to_numpy(dtype=float)).all(), f'{n}: nonfinite'
    S[n] = d
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id'])
for n, d in S.items():
    assert len(d) == len(sample) and d.submission_id.equals(sample.submission_id), f'{n}: wrong row order/count'
    assert d.task.equals(S['k7'].task), f'{n}: task labels changed'
K = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
assert K.submission_id.is_unique, 'submission_key has duplicate ids'
key = K[['submission_id', 'task']].assign(submission_id=lambda x: x.submission_id.astype('int64'))
assert key.set_index('submission_id').loc[sample.submission_id].task.reset_index(drop=True).equals(S['k7'].task.reset_index(drop=True)), 'key task differs'
win = []
for p in sorted(glob.glob(f'{REL}/task2/*/*/window_index.csv')):
    split = p.split('/')[-2]
    if split not in ('validation', 'private'):
        continue
    z = pd.read_csv(p, dtype=str)
    win.append(z[['window_id', 'condition']].assign(split=split))
W = pd.concat(win, ignore_index=True).drop_duplicates()
assert W.window_id.is_unique, 'window_id collision across panel/split'
Q = K[K.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
Q = Q.merge(W, on='window_id', how='left', validate='many_to_one')
assert Q[['condition', 'split']].notna().all().all(), 'unknown queue window'
assert Q.condition.isin(['queue_onset', 'queue_ongoing']).all(), 'unknown condition'
Q['submission_id'] = Q.submission_id.astype('int64')
Q = Q.set_index('submission_id').loc[S['k7'].loc[S['k7'].task == 'queue', 'submission_id']].reset_index()
mon = pd.to_datetime(Q.timestamp, utc=True).dt.strftime('%Y-%m').to_numpy()
assert ((Q.split == 'validation').to_numpy() == (mon == '2031-03')).all(), 'March split/date mismatch'
assert ((Q.split == 'private').to_numpy() == (mon == '2031-04')).all(), 'April split/date mismatch'
priv = (Q.split == 'private').to_numpy()
qmask = (S['k7'].task == 'queue').to_numpy(); smask = (S['k7'].task == 'state').to_numpy(); omask = (S['k7'].task == 'odme').to_numpy()


def equal_num(x, y, mask, tol=1e-9):
    return np.allclose(x[mask].to_numpy(float), y[mask].to_numpy(float), atol=tol, rtol=0)


def gate(c, description):
    assert bool(c), 'FAIL: ' + description
    print('PASS:', description)


for c in ('speed_kmh', 'flow_vph'):
    gate(equal_num(S['k7'][c], S['k8'][c], smask), f'k8 state {c} equals k7')
    gate(equal_num(S['k9'][c], S['k10'][c], smask), f'k10 state {c} equals k9')
    for n in ('k9', 'k10'):
        gate(equal_num(S['k7'][c], S[n][c], ~smask), f'{n}: no non-state {c} edits')
for n in ('k8', 'k9', 'k10'):
    gate(equal_num(S['k7'].path_flow, S[n].path_flow, omask), f'{n} ODME path_flow unchanged')
    gate(np.array_equal(S['k7'].queue_pred.to_numpy()[~qmask], S[n].queue_pred.to_numpy()[~qmask]), f'{n} non-queue queue_pred unchanged')
q = {n: d.loc[qmask, 'queue_pred'].to_numpy(dtype=int) for n, d in S.items()}
for n in q:
    gate(np.isin(q[n], [0, 1]).all(), f'{n} binary queue')
gate(np.array_equal(q['k9'], q['k7']), 'k9 queue identical to k7')
gate(np.array_equal(q['k10'], q['k8']), 'k10 queue identical to k8')
for other, base in (('k8', 'k7'), ('k10', 'k9')):
    ch = q[other] != q[base]
    gate(not (ch & ~priv).any(), f'{other}: March queue unchanged')
    print(f'{other} vs {base}: changed {int(ch.sum())} April queue rows (onset {int((ch & (Q.condition == "queue_onset").to_numpy()).sum())}, '
          f'ongoing {int((ch & (Q.condition == "queue_ongoing").to_numpy()).sum())})')
gate(np.array_equal(q['k8'] - q['k7'], q['k10'] - q['k9']), 'k10-k9 queue delta equals k8-k7')
st = smask
for c in ('speed_kmh', 'flow_vph'):
    dlt = np.abs(S['k9'][c].to_numpy(float)[st] - S['k7'][c].to_numpy(float)[st])
    print(f'k9 vs k7 state {c}: changed share {float((dlt > 1e-9).mean()):.4f} | mean |d| {dlt.mean():.4f} | max |d| {dlt.max():.3f}')
print('FINAL: CROSS-GENERATION SUBMISSION GUARD PASSED')
