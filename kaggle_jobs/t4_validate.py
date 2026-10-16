"""Strict Task 4 validation of the final-pick files (after toolkit PR #1, 2026-10-06: the local evaluator accepted wrong OD zones,
ignored unknown path_ids and crashed on duplicates).  For every submission file in FILES, on the odme rows of submission_key.csv:
  - every key path_id exists in corridors/<panel>/network/path_set.csv; key OD zones (if the key carries them) match path_set;
  - no duplicate (panel, departure_time, path_id); every path_set path present for every (panel, split, departure_time);
  - the file has exactly the sample_submission ids in order; path_flow finite and >= 0 on odme rows, NaN-free;
  - per panel / split: path count, departure times, total flow, share of zero flows.
Prints 'T4 VALIDATION PASSED' or the list of failures."""
import glob, numpy as np, pandas as pd
FILES = [('tfb-k3', 'submission_FDRTV.csv'), ('tfb-k4', 'submission_K4.csv')]
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
print('key columns', list(key.columns), '| tasks', key.task.value_counts().to_dict())
ko = key[key.task == 'odme'].copy()
print('odme key rows', len(ko), '| columns with values', [c for c in ko.columns if ko[c].notna().any()])
fails = []
dup = ko.duplicated(['panel', 'departure_time', 'path_id']).sum()
if dup:
    fails.append(f'key: {dup} duplicate (panel, departure_time, path_id)')
splitcol = 'split' if 'split' in ko.columns and ko.split.notna().any() else None
for panel, g in ko.groupby('panel'):
    ps = pd.read_csv(f'{REL}/corridors/{panel}/network/path_set.csv', dtype=str)
    if ps.path_id.duplicated().any():
        fails.append(f'{panel}: path_set has duplicate path_ids')
    unknown = set(g.path_id) - set(ps.path_id)
    if unknown:
        fails.append(f'{panel}: {len(unknown)} key path_ids not in path_set, e.g. {sorted(unknown)[:3]}')
    for zc in ('origin_zone', 'destination_zone'):
        if zc in g.columns and g[zc].notna().any():
            m = g.merge(ps[['path_id', zc]], on='path_id', how='left', suffixes=('', '_ps'))
            bad = (m[zc] != m[f'{zc}_ps']).sum()
            if bad:
                fails.append(f'{panel}: {bad} rows with {zc} different from path_set')
    for dep, gd in g.groupby('departure_time'):
        miss = set(ps.path_id) - set(gd.path_id)
        if miss:
            fails.append(f'{panel} {dep}: {len(miss)} path_set paths missing from the key')
            break
    print(f'{panel}: key rows {len(g)} | departure times {g.departure_time.nunique()} | paths {g.path_id.nunique()} of path_set {len(ps)}')
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
for kern, fname in FILES:
    f = glob.glob(f'/kaggle/input/**/{kern}/**/{fname}', recursive=True)
    if not f:
        fails.append(f'{kern}/{fname}: not attached'); continue
    s = pd.read_csv(f[0])
    if len(s) != len(sample) or not (s.submission_id.to_numpy() == sample).all():
        fails.append(f'{kern}: submission_id order / count differs from sample_submission')
    if s.submission_id.duplicated().any():
        fails.append(f'{kern}: duplicate submission_id')
    m = ko[['submission_id', 'panel', 'departure_time', 'path_id'] + ([splitcol] if splitcol else [])].assign(
        submission_id=lambda d: d.submission_id.astype('int64')).merge(s[['submission_id', 'task', 'path_flow']], on='submission_id', how='left',
                                                                       validate='one_to_one')
    if m.path_flow.isna().any():
        fails.append(f'{kern}: {int(m.path_flow.isna().sum())} odme rows without path_flow')
    if (m.task != 'odme').any():
        fails.append(f'{kern}: {int((m.task != "odme").sum())} odme key rows with another task label in the file')
    pf = m.path_flow.to_numpy(float)
    if not np.isfinite(pf).all() or (pf < 0).any():
        fails.append(f'{kern}: path_flow not finite or negative ({int((~np.isfinite(pf)).sum())} / {int((pf < 0).sum())})')
    grp = [splitcol, 'panel'] if splitcol else ['panel']
    print(f'== {kern}/{fname}: odme rows {len(m)}')
    print(m.groupby(grp).path_flow.agg(rows='size', total='sum', zero_share=lambda x: float((x == 0).mean())).round(3).to_string())
print('FAILURES:' if fails else 'T4 VALIDATION PASSED', *fails, sep='\n  ')
