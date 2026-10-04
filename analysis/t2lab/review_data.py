"""Review (data facts): family ids vs aggregate(), official window slot range, real validation / private window visibility
(history coverage, slot-T visible share) versus the train-side simulation, split_rows smoke test, export dtypes.
Light: loads one panel at a time, no training."""
import numpy as np, pandas as pd, lab

pd.set_option('display.width', 250)
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False)
    fam = set(z['train_wi_family_id'].tolist())
    assert fam == {panel.rsplit('_', 1)[0]}, (panel, fam)                      # aggregate() family key == official family_id
    wi = lab.window_index(panel, 'train')
    ts = pd.to_datetime(wi.forecast_origin, utc=True); slot = ts.dt.hour * 12 + ts.dt.minute // 5
    cnt = wi.condition.value_counts().to_dict()
    out = dict(panel=panel, L=len(z['links']), D=len(z['train_days']), onset=cnt.get('queue_onset'), ongoing=cnt.get('queue_ongoing'),
               slot_min=int(slot.min()), slot_max=int(slot.max()), train_obs_share=round(float(np.isfinite(z['train_V']).mean()), 4),
               train_ON_dtype=str(z['train_ON'].dtype), train_V_dtype=str(z['train_V'].dtype), train_F_dtype=str(z['train_F'].dtype))
    for split in ('validation', 'private'):
        Vh = z[f'{split}_Vh']; VT = z[f'{split}_VT']; wis = lab.window_index(panel, split)
        out[f'{split}_W'] = len(wis)
        out[f'{split}_hist_cov'] = round(float(np.isfinite(Vh).mean()), 4)
        out[f'{split}_VT_visible'] = round(float(np.isfinite(VT).mean()), 4)
        out[f'{split}_fut_cov'] = sorted(set(wis.future_coverage.tolist()))[:3] if 'future_coverage' in wis else None
    print(out, flush=True)

# train-side simulated slot-T visibility and history coverage on an ongoing set
P = lab.load_panel('D7_I10_W')
rc = lab.rows('D7_I10_W', 'offlike')
hasT = rc.X[:, rc.cols.index('has_T')]
print('train simulation (D7_I10_W offlike rows): has_T share', round(float(hasT.mean()), 4), '| official hist coverage of these windows',
      round(float(np.mean([P['E'][d, t - 12:t].mean() for d, t in zip(rc.win.d, rc.win.t)])), 4))
R = lab.split_rows(P, 'validation')
X, _, _, meta, cols = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
assert list(cols) == rc.cols, 'split_rows column order differs from the cache'
print('split_rows validation: windows', R['W'], 'ongoing rows', len(X), 'cols identical to cache:', list(cols) == rc.cols,
      'has_T share', round(float(X[:, cols.index('has_T')].mean()), 4), 'ramp_on3 finite', round(float(np.isfinite(X[:, cols.index('ramp_on3')]).mean()), 4))
print('official windows (validation) parsed:', len(lab.official_windows(P, 'validation')))
print('DATA REVIEW DONE')
