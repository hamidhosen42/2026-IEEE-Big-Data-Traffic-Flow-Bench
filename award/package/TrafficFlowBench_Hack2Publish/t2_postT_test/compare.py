"""Post-T deletion test, step 4: compare the producers' outputs of a truncated case against the full-data reference.

  compare.py REL_FULL REF_WORK CASE_WORK CUT CASE_NAME OUT_CSV

For every queue_submission_*.csv and every *_probs.npz the two runs share, only the windows whose forecast origin T <= CUT are
compared (they must be identical cell by cell); 'tight' = the windows with T == CUT (the organizer's exact per-window test),
'le' = all windows with T <= CUT. Decoded files: changed / missing cells. Probability files: windows whose float arrays differ,
max |diff|. Appends one row per (kernel, file, scope) to OUT_CSV.
"""
import glob, os, sys
import numpy as np
import pandas as pd

REL, REF, CASE, CUT, NAME, OUT = sys.argv[1:7]
KEY = ['window_id', 'timestamp', 'link_id']
wi = pd.concat([pd.read_csv(f, dtype=str)[['window_id', 'forecast_origin', 'condition']].assign(split=f.split('/')[-2])
                for f in glob.glob(f'{REL}/task2/*/*/window_index.csv') if f.split('/')[-2] in ('validation', 'private')])
T = dict(zip(wi.window_id, wi.forecast_origin))
le = {w for w, t in T.items() if t <= CUT}; tight = {w for w, t in T.items() if t == CUT}
rows = []
for fr in sorted(glob.glob(f'{REF}/kernels/*/queue_submission_*.csv')):
    kern, fn = fr.split('/')[-2], os.path.basename(fr)
    fc = f'{CASE}/kernels/{kern}/{fn}'
    a = pd.read_csv(fr, dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    a = a[a.window_id.isin(le)]
    if not os.path.exists(fc):
        rows.append(dict(case=NAME, cut=CUT, kernel=kern, file=fn, scope='le', n_windows=a.window_id.nunique(), n_cells=len(a),
                         n_changed=-1, n_missing=len(a), note='file missing in case run')); continue
    b = pd.read_csv(fc, dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = a.merge(b[KEY + ['queue_pred']].rename(columns={'queue_pred': 'q_case'}), on=KEY, how='left', validate='one_to_one')
    for scope, ws in (('tight', tight), ('le', le)):
        x = m[m.window_id.isin(ws)]
        miss = x.q_case.isna()
        chg = (~miss) & (x.q_case.fillna(-1).astype(int) != x.queue_pred.astype(int))
        rows.append(dict(case=NAME, cut=CUT, kernel=kern, file=fn, scope=scope, n_windows=x.window_id.nunique(), n_cells=len(x),
                         n_changed=int(chg.sum()), n_missing=int(miss.sum()), n_pos_ref=int(x.queue_pred.sum()),
                         windows_changed=int(x[chg | miss].window_id.nunique()), note=''))
for fr in sorted(glob.glob(f'{REF}/kernels/*/*_probs.npz')):
    kern, fn = fr.split('/')[-2], os.path.basename(fr)
    fc = f'{CASE}/kernels/{kern}/{fn}'
    A = np.load(fr, allow_pickle=False)
    if not os.path.exists(fc):
        rows.append(dict(case=NAME, cut=CUT, kernel=kern, file=fn, scope='le', n_changed=-1, note='file missing in case run')); continue
    B = np.load(fc, allow_pickle=False)
    for scope, ws in (('tight', tight), ('le', le)):
        nw = nd = 0; mx = 0.0; nmiss = 0
        for k in A.files:
            if not k.endswith('__window_id'):
                continue
            pre = k[:-len('window_id')]
            wa = A[k].astype(str)
            wb = B[k].astype(str) if k in B.files else np.array([], str)
            pos_b = {w: i for i, w in enumerate(wb)}
            for nm in [f for f in A.files if f.startswith(pre) and not f.endswith('__window_id')]:
                for i, w in enumerate(wa):
                    if w not in ws:
                        continue
                    nw += 1
                    if nm not in B.files or w not in pos_b:
                        nmiss += 1; continue
                    x, y = A[nm][i], B[nm][pos_b[w]]
                    if not np.array_equal(x, y, equal_nan=True) if x.dtype.kind == 'f' else not np.array_equal(x, y):
                        nd += 1; mx = max(mx, float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
        rows.append(dict(case=NAME, cut=CUT, kernel=kern, file=fn, scope=scope, n_windows=nw, n_cells=nw, n_changed=nd,
                         n_missing=nmiss, note=f'array entries (window x member) differing; max|diff|={mx:.3g}'))
df = pd.DataFrame(rows)
df.to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
bad = df[(df.n_changed != 0) | (df.n_missing.fillna(0) != 0)]
print(f'[compare] {NAME} cut {CUT}: {len(df)} checks, {len(bad)} with differences', flush=True)
if len(bad):
    print(bad.to_string(index=False), flush=True)
