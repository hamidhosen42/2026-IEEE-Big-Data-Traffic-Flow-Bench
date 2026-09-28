"""Mix: FINAL v2 base; ONGOING queue rows from Task 2 v5 (og_noprior); ONSET queue rows from Task 2 v3 (context features)."""
import glob, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv([p for p in glob.glob('/kaggle/input/**/tfb-final/**/submission.csv', recursive=True)][0])
key = pd.read_csv(f'{REL}/submission_key.csv', usecols=['submission_id', 'task', 'window_id', 'timestamp', 'link_id'], dtype=str)
key = key[key.task == 'queue'].copy(); key['timestamp'] = pd.to_datetime(key.timestamp, utc=True)
wi = pd.concat([pd.read_csv(f) for f in glob.glob(f'{REL}/task2/*/*/window_index.csv')])
key['cond'] = key.window_id.map(wi.set_index('window_id').condition)
def load(pattern):
    f = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True); assert len(f) == 1, (pattern, f)
    q = pd.read_csv(f[0], dtype={'window_id': str, 'link_id': str}); q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = key.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert m.queue_pred.notna().all()
    return pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy())
og = load('tfb-t2v5/queue_submission_og_noprior.csv')
cond = pd.Series(key.cond.to_numpy(), index=key.submission_id.astype('int64').to_numpy())
isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
for name, pat in (('ctxon70', 'tfb-t2v3-full/queue_submission_blend70.csv'), ('ctxon_main', 'tfb-t2v3-full/queue_submission.csv')):
    on = load(pat)
    vals = pd.Series(og.reindex(sid).to_numpy(), index=sid)
    is_on = (cond.reindex(sid) == 'queue_onset').to_numpy()
    vals[is_on] = on.reindex(sid).to_numpy()[is_on]
    out = base.copy(); out.loc[isq, 'queue_pred'] = vals.to_numpy().astype(int)
    assert out.notna().all().all() and out.loc[isq, 'queue_pred'].isin([0, 1]).all()
    out.to_csv(f'/kaggle/working/submission_{name}.csv', index=False)
    diff = float((vals.to_numpy() != og.reindex(sid).to_numpy())[is_on].mean())
    print(name, 'onset cells changed vs og_noprior', round(diff, 4), 'positive share', round(vals.mean(), 4), flush=True)
