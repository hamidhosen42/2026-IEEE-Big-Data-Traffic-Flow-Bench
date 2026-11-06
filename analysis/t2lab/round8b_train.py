"""round8b_train.py -- ROUND 8b pv models (PREREG results/round8/PREREG_8b.md asserted via round8b_lib).
  prep                          build / cache the train rows (recipes prod, prodnoperm, lab) and the March rows (cut 15, 20)
  fit <recipe> <cut> <seed...>  one CatBoost per March weight w in (10, 20, 40) and seed -> cache/round8b/models"""
import os, sys, time, warnings
import numpy as np
import lab
import round8b_lib as rb

warnings.filterwarnings('ignore')
if sys.argv[1] == 'prep':
    for rec in ('prod', 'prodnoperm', 'lab'):
        X, y = rb.train_rows(rec); lab.log('train rows', rec, X.shape, 'pos', round(float(y.mean()), 4))
    for c in (15, 20):
        X, y = rb.march_rows(c); lab.log('March rows cut', c, X.shape)
    for c in (15, 20):
        ev = rb.eval_set(c); lab.log('eval cut', c, {p: len(s['widx']) for p, s in ev.items()}, 'total', sum(len(s['widx']) for s in ev.values()))
else:
    rec, cut = sys.argv[2], int(sys.argv[3])
    Xt, yt = rb.train_rows(rec); Xm, ym = rb.march_rows(cut)
    X = np.concatenate([Xt, Xm]); y = np.concatenate([yt, ym])
    for s in map(int, sys.argv[4:]):
        for w in rb.WS:
            f = rb.model_path(rec, cut, s, w)
            if os.path.exists(f):
                continue
            t0 = time.time()
            m = rb.fit(X, y, np.r_[np.ones(len(Xt)), np.full(len(Xm), float(w))], s); m.save_model(f)
            lab.log(f'{rec} cut {cut} seed {s} w {w} {time.time() - t0:.0f}s')
lab.log('DONE')
