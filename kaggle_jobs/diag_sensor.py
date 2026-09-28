import glob, pandas as pd, numpy as np
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
qs = [p for p in glob.glob('/kaggle/input/**/queue_submission_blend70.csv', recursive=True)]
Q = pd.read_csv(qs[0]) if qs else None
for panel in ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']:
    topo = pd.read_csv(f'{REL}/corridors/{panel}/network/lwr_mainline_topology.csv', dtype={'link_id': str})
    uns = set(topo.link_id[topo.has_sensor.astype(int) == 0])
    f = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))[100]
    tr = pd.read_parquet(f, columns=['link_id', 'station_id', 'speed_kmh'])
    h = pd.read_parquet(f'{REL}/task2/{panel}/validation/window_history.parquet')
    obs_tr = tr.groupby('link_id').speed_kmh.apply(lambda s: s.notna().mean())
    obs_h = h.groupby('link_id').speed_kmh.apply(lambda s: s.notna().mean())
    in_train = len(uns & set(tr.link_id)); in_hist = len(uns & set(h.link_id))
    msg = f'{panel}: links {len(topo)}, flagged unsensored {len(uns)}, of which in train rows {in_train} (obs share {obs_tr.reindex(list(uns)).mean():.3f}), ' \
          f'in val history {in_hist} (obs share {obs_h.reindex(list(uns)).mean():.3f}); sensored obs share train {obs_tr.drop(list(uns), errors="ignore").mean():.3f}'
    if Q is not None:
        qq = Q[Q.window_id.str.contains(f'{panel}_validation')]
        msg += f'; our positive rate unsensored {qq[qq.link_id.isin(uns)].queue_pred.mean():.3f} vs sensored {qq[~qq.link_id.isin(uns)].queue_pred.mean():.3f}'
    print(msg, flush=True)
