"""How different are the candidate free-speed sources, and how much would the queue labels change?"""
import glob, os, numpy as np, pandas as pd
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
for panel in ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']:
    net = f'{REL}/corridors/{panel}/network'
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).set_index('link_id')
    tp = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).set_index('link_id')
    fd = pd.read_csv(f'{net}/fd_parameters.csv', dtype={'link_id': str}).set_index('link_id')
    vl = lk.free_speed_kmh; vt = tp.free_flow_speed_kmh.reindex(vl.index); vt2 = tp.free_speed_kmh.reindex(vl.index)
    fs = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))[:60]
    x = pd.concat([pd.read_parquet(f, columns=['link_id', 'speed_kmh', 'is_score_eligible']) for f in fs])
    x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
    cl = 0.6 * x.link_id.map(vl); ct = 0.6 * x.link_id.map(vt)
    qa = x.speed_kmh <= cl; qb = x.speed_kmh <= ct
    inter = (qa & qb).sum(); union = (qa | qb).sum()
    # observed free-flow speed: 85th percentile of speeds by link
    p85 = x.groupby('link_id').speed_kmh.quantile(0.85)
    print(f'{panel}: vf links.csv mean {vl.mean():.1f} | topo free_flow {vt.mean():.1f} / free_speed {vt2.mean():.1f} | fd {fd.free_speed_kmh.mean():.1f} '
          f'| observed p85 {p85.mean():.1f} | queued share links-cut {qa.mean():.4f} topo-cut {qb.mean():.4f} | label IoU {inter / max(union, 1):.3f}', flush=True)
