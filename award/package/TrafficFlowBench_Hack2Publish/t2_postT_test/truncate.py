"""Post-T deletion test, step 2: copy of a release tree with every Task 2 traffic row stamped after CUT removed.

  truncate.py SRC DST CUT [mode=keep|delete|meta]

CUT = an ISO timestamp 'YYYY-MM-DDTHH:MM:SSZ' (normally the forecast origin T of an official window; rows stamped AT T stay).
Removed (validation = March and private = April only; the train split is never touched, as in the organizer's rerun):
  corridors/<P>/<split>/mainline_states_masked/**.parquet   rows with timestamp > CUT
  corridors/<P>/<split>/ramp_states/**.parquet              rows with timestamp > CUT
  task2/<P>/<split>/window_history.parquet                  rows with timestamp > CUT
mode keep   (default) a file left without rows is kept as an empty parquet with the same schema;
mode delete a file left without rows is deleted (window_history.parquet is always kept);
mode meta   as delete, and the windows with origin > CUT are also dropped from window_index.csv / sample_submission_queue.csv
            (the organizer might hand over only the windows that exist at CUT).
Everything else (network, train split, task2 train files, window_index / templates in keep / delete) is symlinked unchanged.
Days entirely before CUT's day are symlinked (all their rows are <= CUT); days after it are emptied / deleted.
"""
import glob, os, shutil, sys
import pandas as pd

SRC, DST, CUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = dict(a.split('=') for a in sys.argv[4:]).get('mode', 'keep')
assert MODE in ('keep', 'delete', 'meta') and len(CUT) == 20 and CUT.endswith('Z'), (MODE, CUT)
CUT_DAY = CUT[:10]
SPLITS = ('validation', 'private')
stats = dict(files_linked=0, files_filtered=0, files_emptied=0, files_deleted=0, rows_removed=0, rows_kept=0)


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


def file_day(path):
    return os.path.basename(path)[-18:-8].replace('_', '-')


def filter_file(src, dst, keep_empty):
    day = file_day(src)
    if day < CUT_DAY:
        link(src, dst); stats['files_linked'] += 1; return
    d = pd.read_parquet(src)
    m = d.timestamp.astype(str) <= CUT
    stats['rows_removed'] += int((~m).sum()); stats['rows_kept'] += int(m.sum())
    d = d[m].reset_index(drop=True)
    if len(d) == 0 and not keep_empty:
        stats['files_deleted'] += 1; return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    d.to_parquet(dst, index=False)
    stats['files_emptied' if len(d) == 0 else 'files_filtered'] += 1


SR, DR = f'{SRC}/kaggle_public', f'{DST}/kaggle_public'
assert not os.path.exists(DST), f'{DST} exists'
for f in glob.glob(f'{SR}/*'):
    if os.path.basename(f) not in ('corridors', 'task2'):
        link(f, f'{DR}/{os.path.basename(f)}')
for pdir in sorted(glob.glob(f'{SR}/corridors/*')):
    panel = os.path.basename(pdir)
    for sub in sorted(glob.glob(f'{pdir}/*')):
        name = os.path.basename(sub)
        if name not in SPLITS:
            link(sub, f'{DR}/corridors/{panel}/{name}'); continue
        for layer in sorted(glob.glob(f'{sub}/*')):
            lname = os.path.basename(layer)
            assert lname in ('mainline_states_masked', 'ramp_states'), layer
            for f in sorted(glob.glob(f'{layer}/**/*.parquet', recursive=True)):
                filter_file(f, f'{DR}/corridors/{panel}/{name}/{lname}/{os.path.relpath(f, layer)}', keep_empty=(MODE == 'keep'))
for pdir in sorted(glob.glob(f'{SR}/task2/*')):
    panel = os.path.basename(pdir)
    for sub in sorted(glob.glob(f'{pdir}/*')):
        name = os.path.basename(sub)
        if name not in SPLITS:
            link(sub, f'{DR}/task2/{panel}/{name}'); continue
        out = f'{DR}/task2/{panel}/{name}'; os.makedirs(out, exist_ok=True)
        h = pd.read_parquet(f'{sub}/window_history.parquet')
        m = h.timestamp.astype(str) <= CUT
        stats['rows_removed'] += int((~m).sum()); stats['rows_kept'] += int(m.sum())
        wi = pd.read_csv(f'{sub}/window_index.csv', dtype=str)
        alive = set(wi.window_id[wi.forecast_origin <= CUT])
        if MODE == 'meta':
            m &= h.window_id.isin(alive)
            wi[wi.window_id.isin(alive)].to_csv(f'{out}/window_index.csv', index=False)
            t = pd.read_csv(f'{sub}/sample_submission_queue.csv', dtype=str)
            t[t.window_id.isin(alive)].to_csv(f'{out}/sample_submission_queue.csv', index=False)
        else:
            for fn in ('window_index.csv', 'sample_submission_queue.csv'):
                link(f'{sub}/{fn}', f'{out}/{fn}')
        h[m].reset_index(drop=True).to_parquet(f'{out}/window_history.parquet', index=False)
        for f in glob.glob(f'{sub}/*'):
            if os.path.basename(f) not in ('window_history.parquet', 'window_index.csv', 'sample_submission_queue.csv'):
                link(f, f'{out}/{os.path.basename(f)}')
print('TRUNCATED', DST, 'cut', CUT, 'mode', MODE, stats, flush=True)
