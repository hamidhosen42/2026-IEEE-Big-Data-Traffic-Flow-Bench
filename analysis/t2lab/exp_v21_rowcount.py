"""Measured training-row counts of t2_model_v21.py (full MAX_OG = 3000) on the local npz panels: runtime / memory estimate."""
import json, pickle, gc, resource, numpy as np
import lab
import exp_v21_localtest_helper as hlp
K = hlp.load_kernel(f'{lab.ROOT}/kaggle_jobs/t2_model_v21.py')
K.build_windows = lambda P_, global_spacing=True, dense_ongoing=False: pickle.load(open(f'{lab.CACHE}/windows/{P_["panel"]}__denseall.pkl', 'rb'))
out = {}; cols = None
for panel in lab.PANELS:
    P = dict(lab.load_panel(panel, cache=False)); K.select_windows(P)
    cols, cur, ex = K.panel_rows(P, cols)
    hm = ex['rs'] & ~ex['incur']
    out[panel] = dict(cur=len(cur['y']), cur_pos=float(cur['y'].mean()), f03=len(ex['y']), f03_pos=float(ex['y'].mean()), rs_rows=int(ex['rs'].sum()),
                      rs_pos=float(ex['y'][ex['rs']].mean()), h=len(cur['y']) + int(hm.sum()), L=len(P['links']))
    print(panel, out[panel], flush=True); del P, cur, ex; gc.collect()
    json.dump(out, open(f'{lab.ROOT}/analysis/t2lab/results/v21/rowcount.json', 'w'), indent=1)
tot = {k: sum(v[k] for v in out.values()) for k in ('cur', 'f03', 'rs_rows', 'h')}
print('TOTAL', tot, 'row-models', 2 * tot['cur'] + 2 * tot['f03'] + tot['h'], 'peak RSS GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
