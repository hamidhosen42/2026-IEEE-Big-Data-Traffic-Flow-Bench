"""Multi-part queue swap: start from the exact base file; for every entry of PARTS replace the queue rows of one condition
(queue_onset / queue_ongoing / any) and one split (validation / private / any) with the rows of SRC_FILE from SRC_KERNEL.
Optional 5th element of a part: 'g72' / 'nog72' / 'any' = only windows whose previous official window of the same panel and
split is exactly 360 min earlier (g72) / all other windows (nog72); computed from the window index (origins only).
Everything else is untouched. Edit BASE / PARTS / OUT_NAME before pushing."""
import glob, numpy as np, pandas as pd
BASE_KERNEL, BASE_FILE = 'tfb-i4', 'submission_FDRTV.csv'
PARTS = [('tfb-t2ogpvogf', 'queue_submission_pvogfv.csv', 'queue_ongoing', 'private')]
OUT_NAME = 'J4'
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
base = pd.read_csv(glob.glob(f'/kaggle/input/**/{BASE_KERNEL}/**/{BASE_FILE}', recursive=True)[0])
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
kq['timestamp'] = pd.to_datetime(kq.timestamp, utc=True)
wins = []
for f in glob.glob(f'{REL}/task2/*/*/window_index.csv'):
    sp = f.split('/')[-2]
    if sp == 'train':
        continue
    w = pd.read_csv(f, dtype=str)
    org = pd.to_datetime(w.forecast_origin, utc=True)
    gs = ((org - pd.Timestamp('2030-01-01', tz='UTC')).dt.total_seconds() // 300).astype(np.int64).to_numpy()
    o = np.argsort(gs, kind='stable'); gap = np.full(len(gs), 10 ** 9, np.int64); gap[o[1:]] = np.diff(gs[o])
    wins.append(w[['window_id', 'condition']].assign(split=sp, g72=(gap == 72)))
wins = pd.concat(wins).drop_duplicates('window_id')
print('g72 windows per split and condition:', wins.groupby(['split', 'condition']).g72.sum().to_dict())
kq = kq.merge(wins, on='window_id', how='left', validate='many_to_one'); assert kq.condition.notna().all() and kq.split.notna().all()
isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
cur = base.loc[isq, 'queue_pred'].to_numpy().astype(int); new = cur.copy()
for part in PARTS:
    src_kernel, src_file, cond, split = part[:4]
    gsel = part[4] if len(part) > 4 else 'any'
    q = pd.read_csv(glob.glob(f'/kaggle/input/**/{src_kernel}/**/{src_file}', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
    q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert m.queue_pred.notna().all()
    src = pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
    sel = np.ones(len(m), bool)
    if cond != 'any':
        sel &= (m.condition == cond).to_numpy()
    if split != 'any':
        sel &= (m.split == split).to_numpy()
    if gsel != 'any':
        sel &= (m.g72 if gsel == 'g72' else ~m.g72).to_numpy()
    mask = pd.Series(sel, index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
    before = new.copy(); new = np.where(mask, src, new)
    print(f'PART {src_kernel}/{src_file} cond={cond} split={split} windows={gsel}: rows {int(mask.sum())} | changed {int((new != before).sum())} '
          f'share {float((new[mask] != before[mask]).mean()):.5f} | positives {int(before[mask].sum())} -> {int(new[mask].sum())}')
    for sp in ('validation', 'private'):
        ms = pd.Series((m.split == sp).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
        print(f'  {sp}: changed rows {int(((new != before) & ms).sum())}')
val = pd.Series((kq.split == 'validation').to_numpy(), index=kq.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
print('TOTAL changed rows: validation', int(((new != cur) & val).sum()), '| private', int(((new != cur) & ~val).sum()))
out = base.copy(); out.loc[isq, 'queue_pred'] = new
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'/kaggle/working/submission_{OUT_NAME}.csv', index=False)
print(OUT_NAME, 'OK', len(out))
