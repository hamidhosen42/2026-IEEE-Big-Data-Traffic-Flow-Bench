import glob, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
for panel in ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']:
    topo = pd.read_csv(f'{REL}/corridors/{panel}/network/lwr_mainline_topology.csv', dtype={'link_id': str})
    f = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))[0]
    mp = pd.read_parquet(f, columns=['link_id', 'milepost']).groupby('link_id').milepost.mean()
    t = topo.set_index('link_id').join(mp)
    a = t.order_index.rank(); b = t.milepost.rank()
    by_oi = t.sort_values('order_index').index.tolist(); by_mp = t.sort_values(['milepost']).index.tolist()
    pos_mp = {l: i for i, l in enumerate(by_mp)}
    # share of order_index neighbours that are also milepost neighbours
    adj = np.mean([abs(pos_mp[x] - pos_mp[y]) == 1 for x, y in zip(by_oi[:-1], by_oi[1:])])
    print(f'{panel}: links {len(t)} | spearman(order_index, milepost) {a.corr(b):+.3f} | order_index neighbours adjacent in milepost order: {adj:.3f} | milepost missing {int(t.milepost.isna().sum())}', flush=True)
