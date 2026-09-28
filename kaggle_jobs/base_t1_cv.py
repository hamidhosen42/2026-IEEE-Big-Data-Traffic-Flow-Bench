"""EXP_000 local CV: the base notebook's own Task 1 code (unchanged src/) run on the train split."""
import glob, os, sys, shutil, time, numpy as np, pandas as pd
src = [p for p in glob.glob('/kaggle/input/**/src/task1_state.py', recursive=True)][0]
shutil.copytree(os.path.dirname(src), '/kaggle/working/src', dirs_exist_ok=True)
sys.path.insert(0, '/kaggle/working/src')
from task1_state import reconstruct_panel_split, build_historical_profiles
from network_utils import load_lane_counts, load_fd_params, load_ordered_links
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
from pathlib import Path
rows = []
for panel in ['D7_I10_W', 'D7_I210_W', 'D12_I405_S']:
    t0 = time.time(); pdir = Path(REL) / 'corridors' / panel
    sp, fl = build_historical_profiles(pdir)
    pred = reconstruct_panel_split(panel, pdir, 'train', sp, fl, load_lane_counts(pdir), load_fd_params(pdir), load_ordered_links(pdir))
    full = pd.concat([pd.read_parquet(f, columns=['timestamp', 'station_id', 'speed_kmh', 'flow_vph']) for f in sorted(glob.glob(f'{pdir}/train/mainline_states/*/*.parquet'))])
    tpl = pd.read_csv(f'{REL}/task1/{panel}/train/sample_submission_state.csv', dtype=str)
    for d in (pred, full, tpl):
        d['timestamp'] = pd.to_datetime(d.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ'); d['station_id'] = d.station_id.astype(str)
    m = tpl[['timestamp', 'station_id', 'mask_regime']].merge(pred[['timestamp', 'station_id', 'speed_kmh', 'flow_vph']], on=['timestamp', 'station_id'], how='left') \
        .merge(full.rename(columns={'speed_kmh': 'tv', 'flow_vph': 'tq'}), on=['timestamp', 'station_id'], how='left')
    lanes = pd.read_csv(f'{pdir}/network/fd_parameters.csv').set_index('link_id').lanes
    st2link = pd.read_parquet(sorted(glob.glob(f'{pdir}/train/mainline_states/*/*.parquet'))[0], columns=['station_id', 'link_id']).drop_duplicates().set_index('station_id').link_id
    m['lanes'] = m.station_id.map(st2link).map(lanes)
    m = m.dropna(subset=['tv', 'tq'])
    miss = int(m.speed_kmh.isna().sum()); m = m.fillna({'speed_kmh': 0, 'flow_vph': 0})
    sc = {}
    for r, g in m.groupby('mask_regime'):
        rv = np.sqrt(np.mean((g.speed_kmh - g.tv) ** 2)); rq = np.sqrt(np.mean(((g.flow_vph - g.tq) / g.lanes) ** 2))
        sc[r] = (0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600), rv, rq)
    rows.append((panel, np.mean([v[0] for v in sc.values()]), sc, miss))
    print(panel, f'S_state={np.mean([v[0] for v in sc.values()]):.4f}', {k: tuple(round(x, 3) for x in v) for k, v in sc.items()}, 'missing preds', miss, f'{time.time() - t0:.0f}s', flush=True)
