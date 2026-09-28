"""Load one panel's train mainline (full + masked) into a tidy frame with the Task 1 target flag."""
import glob, re
import pandas as pd

REL = 'data/kaggle_public'

def load_panel(panel, split='train'):
    C = f'{REL}/corridors/{panel}/{split}'
    full = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{C}/mainline_states/*/*.parquet'))], ignore_index=True) \
        if split == 'train' else None
    parts = []
    for f in sorted(glob.glob(f'{C}/mainline_states_masked/*/*.parquet')):
        d = pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        d['regime'] = re.search(r'mask_regime=(R\d)', f).group(1)
        parts.append(d)
    masked = pd.concat(parts, ignore_index=True).rename(columns={'speed_kmh': 'm_speed', 'flow_vph': 'm_flow', 'is_score_eligible': 'm_elig'})
    if full is None:
        return masked
    d = full.merge(masked, on=['timestamp', 'station_id', 'link_id'], how='inner')
    d['target'] = (d.is_score_eligible == 1) & d.m_speed.isna() & d.speed_kmh.notna()
    d['ts'] = pd.to_datetime(d.timestamp, utc=True)
    d['tod'] = d.ts.dt.hour * 12 + d.ts.dt.minute // 5
    d['dow'] = d.ts.dt.dayofweek
    return d
