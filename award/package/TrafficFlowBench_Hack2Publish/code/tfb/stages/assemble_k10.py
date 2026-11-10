"""K10 = FINAL CANDIDATE B (hedge) on tfb-k9 (EXP_T1_K9_PREP, prepared 2026-10-30, NOT pushed): identical to swap_k8.py (K8 = hedge B on
tfb-k7) except the base, so K10 - K9 has exactly the April queue rows of K8 - K7 = K6 - K5 (April onset = production CatBoost with the
train prior, April ongoing = veto10; March rows unchanged).

(swap_k8.py docstring follows)
K8 = FINAL CANDIDATE B (hedge) on tfb-k7 (prepared 2026-10-30, NOT pushed): identical to swap_k6.py (K6 = hedge B on tfb-k5)
except the base, so K8 - K7 has exactly the April queue rows of K6 - K5 (April onset = production CatBoost with the train prior,
April ongoing = veto10; March rows unchanged).

(swap_k6.py docstring follows)
Multi-part queue swap: start from the exact base file; for every entry of PARTS replace the queue rows of one condition
(queue_onset / queue_ongoing / any) and one split (validation / private / any) with the rows of SRC_FILE from SRC_KERNEL.
Optional 5th element of a part: 'g72' / 'nog72' / 'any' = only windows whose previous official window of the same panel and
split is exactly 360 min earlier (g72) / all other windows (nog72); computed from the window index (origins only).
Everything else is untouched. Edit BASE / PARTS / OUT_NAME before pushing."""
import glob, numpy as np, pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-k10.py (= kaggle_jobs/swap_k10.py, swap_queue2 + gates); patched lines end in [TFB]
CFG = TFB.cfg  # [TFB] tfb/config.py ASSEMBLE_BASE[pick] + QUEUE_PARTS[pick]: one part per (condition, split) block
BASE_KERNEL, BASE_FILE = CFG['base'][0], CFG['base'][1]  # [TFB] original base tfb-k9/submission_FDRTV.csv
PARTS = [tuple(p) for p in CFG['parts']]  # [TFB]
OUT_NAME = CFG['out_name']  # [TFB]
REL = TFB.rel  # [TFB]
base = pd.read_csv(TFB.kfile(BASE_KERNEL, BASE_FILE))  # [TFB]
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
wins = pd.concat(wins).drop_duplicates(); assert wins.window_id.is_unique, 'window_id with conflicting condition/split'
print('g72 windows per split and condition:', wins.groupby(['split', 'condition']).g72.sum().to_dict())
kq = kq.merge(wins, on='window_id', how='left', validate='many_to_one'); assert kq.condition.notna().all() and kq.split.notna().all()
isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
cur = base.loc[isq, 'queue_pred'].to_numpy().astype(int); new = cur.copy()
covered = np.zeros(len(sid), bool)  # [TFB] queue rows replaced by some part (the T4 skeleton's queue rows are 0)
for part in PARTS:
    src_kernel, src_file, cond, split = part[:4]
    gsel = part[4] if len(part) > 4 else 'any'
    q = pd.read_csv(TFB.kfile(src_kernel, src_file), dtype={'window_id': str, 'link_id': str})  # [TFB]
    q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert m.queue_pred.notna().all()
    assert set(m.submission_id.astype('int64')) == set(sid) and m.submission_id.is_unique, 'queue ids of key and base differ'
    src = pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
    sel = np.ones(len(m), bool)
    if cond != 'any':
        sel &= (m.condition == cond).to_numpy()
    if split != 'any':
        sel &= (m.split == split).to_numpy()
    if gsel != 'any':
        sel &= (m.g72 if gsel == 'g72' else ~m.g72).to_numpy()
    mask = pd.Series(sel, index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
    assert not pd.isna(src).any() and not pd.isna(mask).any(), 'unmapped queue rows after reindex'
    before = new.copy(); new = np.where(mask, src, new)
    covered |= mask  # [TFB]
    print(f'PART {src_kernel}/{src_file} cond={cond} split={split} windows={gsel}: rows {int(mask.sum())} | changed {int((new != before).sum())} '
          f'share {float((new[mask] != before[mask]).mean()):.5f} | positives {int(before[mask].sum())} -> {int(new[mask].sum())}')
    for sp in ('validation', 'private'):
        ms = pd.Series((m.split == sp).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
        print(f'  {sp}: changed rows {int(((new != before) & ms).sum())}')
val = pd.Series((kq.split == 'validation').to_numpy(), index=kq.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
print('TOTAL changed rows: validation', int(((new != cur) & val).sum()), '| private', int(((new != cur) & ~val).sum()))
assert CFG['march_from_parts'] or not ((new != cur) & val).any(), 'K10 changed March validation queue rows'  # [TFB] the skeleton base has no March queue rows; validation parts fill them
out = base.copy(); out.loc[isq, 'queue_pred'] = new
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
# [TFB] added final gates (tfb-final cell 7 hard gates + full queue coverage), then the writer of the kernel
print('queue rows covered by a part:', int(covered.sum()), 'of', len(covered), '| positives per split:', {sp: int(new[(val if sp == 'validation' else ~val)].sum()) for sp in ('validation', 'private')})  # [TFB]
assert covered.all() or not CFG.get('require_full_cover'), f'{int((~covered).sum())} queue rows not covered by any part'  # [TFB]
assert (out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']] >= 0).all().all(), 'negative values'  # [TFB]
_ks = key[key.task == 'state'][['submission_id', 'panel', 'mask_regime']].assign(submission_id=lambda d: d.submission_id.astype('int64'))  # [TFB]
_st = _ks.merge(out.loc[out.task == 'state', ['submission_id', 'flow_vph']], on='submission_id', how='left', validate='one_to_one')  # [TFB]
assert _st.flow_vph.notna().all(), 'state key rows missing from the file'  # [TFB]
_low = _st.assign(low=_st.flow_vph < 50).groupby(['panel', 'mask_regime']).low.mean()  # [TFB]
assert (_low <= 0.2).all(), f'S_FD empty-road guard would fire: {_low[_low > 0.2]}'  # [TFB]
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'{TFB.out}/submission_{OUT_NAME}.csv', index=False)  # [TFB]
print(OUT_NAME, 'OK', len(out))
