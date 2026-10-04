"""ideas: light plumbing check (no training): March bundles + saved production-like model must reproduce the saved
exp_onset_splitprior numbers (base 0.7140, s4 0.7875, s8 0.8010 on 214 windows)."""
import time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
import lab, ideas_lib as il

cat = CatBoostClassifier(); cat.load_model('results/splitprior/cat_onset_noramp.cbm')
ref = pd.read_pickle('results/splitprior_noramp/march_onset_variants.pkl')
res = []
for panel in lab.PANELS:
    t0 = time.time()
    b = il.march_bundle(panel)
    rc = il.march_rc(b)
    for name, k in (('base', None), ('s4', 4), ('s8', 8)):
        X = b['X'].copy(); X[:, il.JP] = il.march_prior(b, k)
        p = cat.predict_proba(X[:, il.KEEPI], thread_count=2)[:, 1]
        res.append(il.score(rc, b['wi'], b['li'], p, name, 'march_on'))
    print(panel, 'W', b['W'], 'L', b['L'], 'blind', int((b['maxprior'] < 0.05).sum()), 'low', int((b['maxprior'] < 0.2).sum()),
          'extras nan share', np.isnan(b['EX']).mean(0).round(2), f'{time.time() - t0:.1f}s', flush=True)
R = pd.concat(res, ignore_index=True)
for name in ('base', 's4', 's8'):
    g = R[R.arm == name]; r0 = ref[ref.arm == name]
    m = g.merge(r0[['panel', 'd', 't', 'iou']], on=['panel', 'd', 't'], suffixes=('', '_ref'))
    print(name, 'n', len(g), 'agg', round(lab.aggregate(g), 4), 'ref', round(lab.aggregate(r0), 4), 'max abs window diff', float((m.iou - m.iou_ref).abs().max()), 'matched', len(m))
