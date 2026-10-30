"""Function test of kaggle_jobs/swap_k8.py (hedge B on tfb-k7) on a synthetic /kaggle/input tree (argv[1]): 1 panel, 2 validation
+ 2 private windows (onset / ongoing), base tfb-k7/submission_FDRTV.csv, sources tfb-t2osp/queue_submission_base.csv and
tfb-t2ogveto/queue_submission_veto10.csv. Checks: only private onset rows take tfb-t2osp, only private ongoing rows take tfb-t2ogveto,
validation rows and all state / path rows unchanged, output submission_K8.csv written."""
import os, sys, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
TMP = os.path.abspath(sys.argv[1]); shutil.rmtree(TMP, ignore_errors=True)
IN = os.path.join(TMP, 'input'); OUTD = os.path.join(TMP, 'working'); REL = f'{IN}/data/kaggle_public'
os.makedirs(OUTD, exist_ok=True)
P = 'D7_I10_E'
W = [('wv0', 'validation', 'queue_onset', '2031-03-02T08:00:00Z'), ('wv1', 'validation', 'queue_ongoing', '2031-03-05T17:00:00Z'),
     ('wp0', 'private', 'queue_onset', '2031-04-02T08:00:00Z'), ('wp1', 'private', 'queue_ongoing', '2031-04-05T17:00:00Z')]
for sp in ('validation', 'private'):
    os.makedirs(f'{REL}/task2/{P}/{sp}', exist_ok=True)
    pd.DataFrame([(w, c, o) for w, s, c, o in W if s == sp], columns=['window_id', 'condition', 'forecast_origin']).to_csv(
        f'{REL}/task2/{P}/{sp}/window_index.csv', index=False)
rows = [('state', '', '2031-03-02T00:00:00Z', 'L1')] * 3
for w, s, c, o in W:
    t0 = pd.Timestamp(o)
    for h in range(1, 7):
        for l in ('L1', 'L2'):
            rows.append(('queue', w, (t0 + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ'), l))
rows += [('path_flow', '', '', '')] * 2
key = pd.DataFrame(rows, columns=['task', 'window_id', 'timestamp', 'link_id']); key.insert(0, 'submission_id', np.arange(len(key)))
os.makedirs(REL, exist_ok=True); key.to_csv(f'{REL}/submission_key.csv', index=False); key[['submission_id']].to_csv(f'{REL}/sample_submission.csv', index=False)
base = pd.DataFrame({'submission_id': key.submission_id, 'task': key.task, 'speed_kmh': np.where(key.task == 'state', 80.0, 0.0),
                     'flow_vph': np.where(key.task == 'state', 1500.0, 0.0), 'queue_pred': 0, 'path_flow': np.where(key.task == 'path_flow', 3.0, 0.0)})
os.makedirs(f'{IN}/tfb-k7', exist_ok=True); base.to_csv(f'{IN}/tfb-k7/submission_FDRTV.csv', index=False)
kq = key[key.task == 'queue'][['window_id', 'timestamp', 'link_id']]
for slug, fn in (('tfb-t2osp', 'queue_submission_base.csv'), ('tfb-t2ogveto', 'queue_submission_veto10.csv')):
    os.makedirs(f'{IN}/{slug}', exist_ok=True); kq.assign(queue_pred=1).to_csv(f'{IN}/{slug}/{fn}', index=False)
src = open(os.path.join(ROOT, 'kaggle_jobs', 'swap_k8.py')).read().replace('/kaggle/input', IN).replace('/kaggle/working', OUTD)
exec(compile(src, 'swap_k8.py', 'exec'), {'__name__': '__main__'})
o = pd.read_csv(f'{OUTD}/submission_K8.csv')
isq = (key.task == 'queue').to_numpy(); priv = key.window_id.str.startswith('wp').to_numpy()
assert (o.queue_pred[isq & priv] == 1).all() and (o.queue_pred[isq & ~priv] == 0).all(), 'wrong rows swapped'
assert np.array_equal(o[['speed_kmh', 'flow_vph', 'path_flow']].to_numpy(), base[['speed_kmh', 'flow_vph', 'path_flow']].to_numpy())
print('K8 SWAP FUNCTION TEST PASSED (private onset + ongoing rows swapped, validation / state / path rows unchanged)')
