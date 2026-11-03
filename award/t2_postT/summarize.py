"""Summarise the post-T deletion test: results_<family>.csv (+ .failures) -> results_summary.csv and a printed table.

  summarize.py REL_FULL   (REL_FULL = the full kaggle_public tree, for the window index)
"""
import glob, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = sys.argv[1]
wi = pd.concat([pd.read_csv(f, dtype=str).assign(split=f.split('/')[-2]) for f in glob.glob(f'{REL}/task2/*/*/window_index.csv')
                if f.split('/')[-2] in ('validation', 'private')])
USED = {  # (kernel, file) -> rows of the final picks that come from it
    ('tfb-t2osp', 'queue_submission_sp4.csv'): 'A+B March onset',
    ('tfb-t2osp', 'queue_submission_base.csv'): 'B April onset',
    ('tfb-t2opv', 'queue_submission_pv.csv'): 'A April onset',
    ('tfb-t2occ', 'queue_submission_fs4occ.csv'): 'A+B March ongoing (+ input of veto10)',
    ('tfb-t2occ', 't2v26occ_probs.npz'): 'qlast for veto10',
    ('tfb-t2ogfa', 't2v27a_probs.npz'): 'input of fs4ogf -> veto10',
    ('tfb-t2ogfb', 't2v27b_probs.npz'): 'input of fs4ogf -> veto10',
    ('tfb-t2ogfc', 'queue_submission_fs4ogf.csv'): 'input of veto10',
    ('tfb-t2ogveto', 'queue_submission_veto10.csv'): 'B April ongoing',
    ('tfb-t2ogpvogf', 'queue_submission_pvogf.csv'): 'A April ongoing',
    ('tfb-t2ogpvogf', 't2pvogf_probs.npz'): 'A April ongoing (probabilities)',
}
rows = []
for f in sorted(glob.glob(f'{HERE}/results_*.csv')):
    fam = os.path.basename(f)[8:-4]
    if fam in ('summary', 'try', 'noise'):
        continue
    d = pd.read_csv(f)
    d = d[d.case != 'case']
    for c in ('n_windows', 'n_cells', 'n_changed', 'n_missing'):
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0).astype(int)
    d['mode'] = d.case.str.split('_').str[1]
    fails = []
    if os.path.exists(f + '.failures'):
        fails = [l.split('\t') for l in open(f + '.failures').read().strip().split('\n') if l.strip()]
    for (kern, fn), g in d.groupby(['kernel', 'file']):
        t, le = g[g.scope == 'tight'], g[g.scope == 'le']
        rows.append(dict(family=fam, kernel=kern, file=fn, used_for=USED.get((kern, fn), ''), cases=g.case.nunique(),
                         modes='/'.join(sorted(g['mode'].unique())), tight_windows=int(t.n_windows.sum()), tight_cells=int(t.n_cells.sum()),
                         le_cells=int(le.n_cells.sum()), changed=int(g.n_changed.sum()), missing=int(g.n_missing.sum()),
                         failed_cases=len(fails)))
S = pd.DataFrame(rows)
S['verdict'] = S.apply(lambda r: 'PASS' if r.changed == 0 and r.missing == 0 and r.failed_cases == 0 else 'FAIL', axis=1)
S.to_csv(f'{HERE}/results_summary.csv', index=False)
pd.set_option('display.width', 250)
print(S.to_string(index=False))
for f in sorted(glob.glob(f'{HERE}/results_*.csv')):
    fam = os.path.basename(f)[8:-4]
    if fam in ('summary', 'try', 'noise'):
        continue
    d = pd.read_csv(f); d = d[d.case != 'case']
    d['mode'] = d.case.str.split('_').str[1]; d['n_windows'] = pd.to_numeric(d.n_windows)
    x = d[(d.kernel == 'tfb-t2osp') & (d.file == 'queue_submission_sp4.csv') & (d.scope == 'tight')]
    x = x.assign(split=x.cut.str[:7].map({'2031-03': 'March', '2031-04': 'April'}))
    print(f'{fam}: {d.case.nunique()} cases; windows tested exactly at their own T, by mode / month:',
          x.groupby(['mode', 'split']).n_windows.sum().to_dict())
