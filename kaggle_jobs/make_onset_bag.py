"""Round 8 (2026-11-06): generate seed-bagged variants of the two submitted onset kernels.

The sources are the exact submitted kernels in consolidation/lineage_src (tfb-t2osp.py, tfb-t2opv.py). The only change is that every
onset CatBoost (random_seed 0) becomes the mean of seeds 0..K-1 with the same data, rows and settings. Each replacement is an exact
string that must match once. A control block appended at the end compares every written queue file with the original kernel's file
of the same name (attach hosen42/<original kernel> as a kernel source).

  python kaggle_jobs/make_onset_bag.py --seeds 10      -> kaggle_jobs/onset_bag/tfb-t2osp-bag.py, tfb-t2opv-bag.py
"""
import argparse, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f'{ROOT}/consolidation/lineage_src'
OUT = f'{ROOT}/kaggle_jobs/onset_bag'

CAT = "CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=0)"
BAG_CLASS = '''
class SeedBag:  # [R8] mean of the onset CatBoost over seeds 0..N_SEEDS-1 (production used random_seed 0 only)
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], 0)


def fit_bag(X, y, **fit_kw):  # [R8]
    ms = []
    for s in range(N_SEEDS):
        m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=s)
        m.fit(X, y, **fit_kw)
        ms.append(m)
    return SeedBag(ms)
'''

OSP_FIT = (f"cat_on = {CAT}\ncat_on.fit(_Xo[_sub], _yo[_sub]); log('catboost onset trained on', len(_sub), 'rows (v9 features)')",
           "cat_on = fit_bag(_Xo[_sub], _yo[_sub]); log('catboost onset bag of', N_SEEDS, 'seeds trained on', len(_sub), 'rows (v9 features)')  # [R8]")
PV_FIT = (f"    m_ = {CAT}\n    m_.fit(np.concatenate([_Xo_keep, Xm]), np.concatenate([_yo_keep, ym]),\n"
          "           sample_weight=np.r_[np.ones(len(_Xo_keep)), np.full(len(Xm), float(wm))])",
          "    m_ = fit_bag(np.concatenate([_Xo_keep, Xm]), np.concatenate([_yo_keep, ym]),  # [R8]\n"
          "                 sample_weight=np.r_[np.ones(len(_Xo_keep)), np.full(len(Xm), float(wm))])")


def control(orig, files):
    return f'''

# ---- [R8] CONTROL: agreement of every written queue file with the original kernel {orig}
for fn in {files!r}:
    try:
        a = pd.read_csv(f'{{OUT}}/{{fn}}', dtype={{'window_id': str, 'link_id': str}})
        b = pd.read_csv(glob.glob(f'/kaggle/input/**/{orig}/**/{{fn}}', recursive=True)[0], dtype={{'window_id': str, 'link_id': str}})
        m = a.merge(b, on=['window_id', 'timestamp', 'link_id'], how='outer', suffixes=('', '_orig'), indicator=True)
        assert (m._merge == 'both').all(), 'row sets differ'
        x, y = m.queue_pred.to_numpy().astype(bool), m.queue_pred_orig.to_numpy().astype(bool)
        log(f'R8 CONTROL {{fn}}: rows {{len(m)}} changed {{int((x != y).sum())}} pos {{int(x.sum())}} vs orig {{int(y.sum())}} '
            f'jaccard {{(x & y).sum() / max((x | y).sum(), 1):.4f}}')
    except Exception as ex:
        log('R8 CONTROL', fn, 'skipped:', type(ex).__name__, str(ex)[:200])
log('R8 BAG DONE seeds', N_SEEDS)
'''


def patch(src, reps):
    for old, new in reps:
        assert src.count(old) == 1, f'expected exactly one match for: {old[:80]!r} (found {src.count(old)})'
        src = src.replace(old, new)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    head = f'N_SEEDS = {a.seeds}  # [R8]\n'
    for name, reps, files in (('tfb-t2osp', [OSP_FIT], ['queue_submission_base.csv', 'queue_submission_sp4.csv']),
                              ('tfb-t2opv', [OSP_FIT, PV_FIT], ['queue_submission_base.csv', 'queue_submission_sp4.csv',
                                                                'queue_submission_pv.csv', 'queue_submission_pvb.csv'])):
        src = open(f'{SRC}/{name}.py').read()
        anchor = 'from catboost import CatBoostClassifier\n'
        assert src.count(anchor) == 1
        src = src.replace(anchor, anchor + head + BAG_CLASS)
        src = patch(src, reps) + control(name, files)
        assert 'random_seed=0)' not in src.replace('random_seed=s)', ''), 'a seed-0 CatBoost is left'
        compile(src, name, 'exec')
        path = f'{OUT}/{name}-bag.py'
        open(path, 'w').write(src)
        print('wrote', os.path.relpath(path, ROOT), len(src), 'chars')


if __name__ == '__main__':
    main()
