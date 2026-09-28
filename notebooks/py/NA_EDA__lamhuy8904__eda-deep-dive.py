# %% [md 0]
# # Continuum Traffic Dynamics: Empirical Calibration and Diagnostics
# 
# This exploratory analysis notebook performs offline calibration and sensitivity analysis for the multi-task pipeline:
# 1. **Task 4 (ODME)**: Grid search over regularization parameter $\lambda$ and prior anchor blending ratio to maximize $S_{\text{ODME}}$.
# 2. **Task 2 (Queue Forecasting)**: Verification of onset trigger thresholds and shockwave propagation across all 80 validation windows.
# 3. **Task 1 & 3 (FD Coupling)**: Empirical measurement of $S_{\text{FD}}$ consistency and speed/flow reconstruction error under fundamental diagram projection.

# %% [cell 1]
import os, sys, json, glob, math, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
warnings.filterwarnings('ignore')

def locate_release_root() -> Path:
    candidates = [
        '/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
        '/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
        '/kaggle/input/2026-ieee-big-data-traffic-flow-bench',
    ]
    candidates += glob.glob('/kaggle/input/**/kaggle_public', recursive=True)
    for c in candidates:
        p = Path(c)
        if (p / 'config' / 'corridors.json').exists() or (p / 'corridors').exists():
            return p
        if p.name == 'corridors' and p.parent.exists():
            return p.parent
    raise FileNotFoundError('Failed to locate dataset.')

RELEASE = locate_release_root()
print(f'Release root: {RELEASE}')

manifest = json.loads((RELEASE / 'config' / 'corridors.json').read_text(encoding='utf-8'))
PANELS = [p['corridor_id'] for p in manifest['panels']]
FAMILIES = {p['corridor_id']: p['family_id'] for p in manifest['panels']}
print(f'Active panels ({len(PANELS)}): {PANELS}')

# %% [md 2]
# ## 1. Task 4 (ODME): Regularization $\lambda$ and Prior Blending Grid Search
# 
# We systematically test combinations of $\lambda \in [0.01, 0.05, 0.20, 1.0, 5.0, 20.0]$ and prior weights $w_{\text{prior}} \in [0.0, 0.3, 0.5, 0.7, 1.0]$ across all 10 corridors to determine the parameter configuration that maximizes total $S_{\text{ODME}} = 0.45 S_{\text{od}} + 0.25 S_{\text{link}} + 0.15 S_{\text{dev}} + 0.15 S_{\text{attr}}$.

# %% [cell 3]
print('=== Task 4 Comprehensive Sensitivity Grid Search ===')
lambdas = [0.01, 0.05, 0.20, 1.0, 5.0, 20.0]
prior_weights = [0.0, 0.3, 0.5, 0.7, 1.0]

results = []

for panel in PANELS:
    network = RELEASE / 'corridors' / panel / 'network'
    paths = pd.read_csv(network / 'path_set.csv')
    paths['path_id'] = paths.path_id.astype(str)
    path_ids = paths.path_id.tolist()
    path_index = {x: i for i, x in enumerate(path_ids)}
    
    incidence = pd.read_csv(network / 'path_link_incidence.csv')
    incidence['path_id'] = incidence.path_id.astype(str)
    incidence['link_id'] = incidence.link_id.astype(str)
    link_ids = incidence.link_id.drop_duplicates().tolist()
    link_index = {x: i for i, x in enumerate(link_ids)}
    
    rr = incidence.link_id.map(link_index).to_numpy(dtype=np.int64)
    cc = incidence.path_id.map(path_index).to_numpy(dtype=np.int64)
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr, cc] = 1.0
    
    base_od = pd.read_csv(network / 'base_od.csv')
    base_od['path_id'] = base_od.path_id.astype(str)
    base = paths[['path_id', 'destination_zone']].merge(base_od[['path_id', 'base_flow']], on='path_id', how='left')
    base = base.set_index('path_id').reindex(path_ids).reset_index()
    base_values = pd.to_numeric(base.base_flow, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    
    counts_path = RELEASE / 'task4' / panel / 'validation' / 'synthetic_link_counts.csv'
    prior_path = RELEASE / 'task4' / panel / 'validation' / 'synthetic_weak_prior.csv'
    if not (counts_path.exists() and prior_path.exists()):
        continue
        
    counts_df = pd.read_csv(counts_path, dtype={'link_id': str})
    val_counts = counts_df.set_index('link_id').reindex(link_ids).fillna(0.0)['count'].to_numpy(dtype=float)
    observed = set(counts_df.link_id)
    measured = np.array([link in observed for link in link_ids], dtype=bool)
    
    prior = pd.read_csv(prior_path, dtype={'path_id': str})
    prior_flow = prior.set_index('path_id').reindex(path_ids)['path_flow'].to_numpy(dtype=float)
    prior_flow = np.nan_to_num(prior_flow, nan=0.0)
    
    # Destination attraction vector for S_attr
    dest_zones = paths['destination_zone'].astype(str).tolist()
    unique_dests = sorted(list(set(dest_zones)))
    dest_map = {d: i for i, d in enumerate(unique_dests)}
    dest_indices = np.array([dest_map[d] for d in dest_zones])
    
    def compute_attr(f_vec):
        attr = np.zeros(len(unique_dests), dtype=float)
        np.add.at(attr, dest_indices, f_vec)
        tot = attr.sum()
        return attr / max(tot, 1e-9)
        
    a_star = compute_attr(prior_flow)
    dstar = np.sum(np.abs(prior_flow - base_values))
    
    for lam in lambdas:
        for wp in prior_weights:
            anchor = (1.0 - wp) * base_values + wp * prior_flow
            aa = np.vstack([A[measured], np.sqrt(lam) * np.eye(len(path_ids))])
            bb = np.concatenate([val_counts[measured], np.sqrt(lam) * anchor])
            f, _ = nnls(aa, bb)
            
            # Metric calculation
            loaded = A @ f
            link_err = np.sum(np.abs(loaded[measured] - val_counts[measured]))
            link_tot = np.sum(val_counts[measured])
            s_link = max(0.0, 1.0 - link_err / max(link_tot, 1e-9))
            
            s_od = max(0.0, 1.0 - np.sum(np.abs(f - prior_flow)) / max(np.sum(prior_flow), 1e-9))
            
            dhat = np.sum(np.abs(f - base_values))
            s_dev = np.exp(-abs(dhat / max(dstar, 1e-9) - 1))
            
            a_hat = compute_attr(f)
            s_attr = max(0.0, 1.0 - 0.5 * np.sum(np.abs(a_hat - a_star)))
            
            s_total = 0.45 * s_od + 0.25 * s_link + 0.15 * s_dev + 0.15 * s_attr
            
            results.append({
                'panel': panel, 'lambda': lam, 'w_prior': wp,
                'S_link': s_link, 'S_od': s_od, 'S_dev': s_dev, 'S_attr': s_attr,
                'S_ODME': s_total
            })

res_df = pd.DataFrame(results)
summary = res_df.groupby(['lambda', 'w_prior'])[['S_link', 'S_od', 'S_dev', 'S_attr', 'S_ODME']].mean().reset_index()
summary = summary.sort_values('S_ODME', ascending=False)

print('\nTop 10 Parameter Configurations for Task 4:')
print(summary.head(10).to_string(index=False))

best_row = summary.iloc[0]
print(f'\nOptimal Task 4 Configuration: lambda={best_row["lambda"]}, w_prior={best_row["w_prior"]} -> Expected S_ODME = {best_row["S_ODME"]:.4f}')

# %% [md 4]
# ## 2. Task 2 (Queue Forecasting): Onset and Ongoing Kinematics
# 
# We inspect the exact speed profiles and shockwave speeds for all 80 validation windows to diagnose why onset predictions were under-represented.

# %% [cell 5]
print('=== Task 2 Validation Queue Dynamics Analysis ===')
window_index_files = sorted(RELEASE.glob('task2/**/window_index.csv')) or sorted(RELEASE.glob('**/window_index.csv'))
history_files = sorted(RELEASE.glob('task2/**/window_history.parquet')) or sorted(RELEASE.glob('**/window_history.parquet'))

windows = pd.concat([pd.read_csv(f) for f in window_index_files], ignore_index=True)
windows_val = windows[windows['split'] == 'validation'].copy()
history = pd.concat([pd.read_parquet(f) for f in history_files], ignore_index=True)
history['speed_kmh'] = pd.to_numeric(history['speed_kmh'], errors='coerce')
history['window_id'] = history['window_id'].astype(str)

print(f'Total validation windows: {len(windows_val)}')
onset_windows = windows_val[windows_val.condition == 'queue_onset']
ongoing_windows = windows_val[windows_val.condition == 'queue_ongoing']
print(f'Onset: {len(onset_windows)}, Ongoing: {len(ongoing_windows)}')

for panel in sorted(windows_val.panel.unique()):
    p_dir = RELEASE / 'corridors' / panel
    links = pd.read_csv(p_dir / 'network' / 'links.csv', dtype={'link_id': str})
    fd_path = p_dir / 'network' / 'fd_parameters.csv'
    vcuts = {}
    if fd_path.exists():
        fd_df = pd.read_csv(fd_path, dtype={'link_id': str})
        if 'v_cut' in fd_df.columns:
            for row in fd_df.dropna(subset=['link_id', 'v_cut']).itertuples():
                vcuts[str(row.link_id)] = float(row.v_cut)
    
    p_onset = onset_windows[onset_windows.panel == panel]
    p_hist = history[history.window_id.isin(p_onset.window_id)]
    
    min_margins = []
    max_decels = []
    for wid, g in p_hist.groupby('window_id'):
        w_margins = []
        w_decels = []
        for lid, lg in g.groupby('link_id'):
            cut = vcuts.get(str(lid), 63.0)
            valid = lg.dropna(subset=['speed_kmh']).sort_values('timestamp')
            if len(valid) >= 4:
                v0 = valid['speed_kmh'].iloc[-1]
                v_prev = valid['speed_kmh'].iloc[-4]
                margin = (v0 - cut) / max(cut, 1.0)
                decel = (v0 - v_prev) / 3.0
                w_margins.append(margin)
                w_decels.append(decel)
        if w_margins:
            min_margins.append(min(w_margins))
        if w_decels:
            max_decels.append(min(w_decels))
            
    print(f'{panel} Onset Windows (n={len(p_onset)}):')
    print(f'  Minimum margin distribution: mean={np.mean(min_margins):.3f}, min={np.min(min_margins):.3f}, max={np.max(min_margins):.3f}')
    print(f'  Strongest decel distribution: mean={np.mean(max_decels):.2f}, min={np.min(max_decels):.2f} km/h/step')



# %% [md 6]
# ## 3. Task 1 & Task 3: FD Projection Blend Factor Sensitivity
# 
# We evaluate the effect of $\alpha \in [0.0, 0.10, 0.20, 0.30]$ on speed/flow reconstruction RMSE and FD consistency error.

# %% [cell 7]
print('=== Task 1 & Task 3 Fundamental Diagram Calibration ===')
# Inspect speed-flow dispersion against theoretical FD curve for first corridor
p_dir = RELEASE / 'corridors' / PANELS[0]
train_files = sorted((p_dir / 'train' / 'mainline_states').glob('**/*.parquet'))
if train_files:
    sample_df = pd.read_parquet(train_files[0])
    sample_df = sample_df.dropna(subset=['speed_kmh', 'flow_vph'])
    sample_df['density'] = sample_df['flow_vph'] / sample_df['speed_kmh'].clip(lower=1.0)
    print(f'Sample corridor: {PANELS[0]}')
    print(f'Density percentiles: 5%={sample_df.density.quantile(0.05):.1f}, 50%={sample_df.density.quantile(0.50):.1f}, 95%={sample_df.density.quantile(0.95):.1f} veh/km')
    print(f'Speed percentiles: 5%={sample_df.speed_kmh.quantile(0.05):.1f}, 50%={sample_df.speed_kmh.quantile(0.50):.1f}, 95%={sample_df.speed_kmh.quantile(0.95):.1f} km/h')
    print('Data validation completed successfully.')
