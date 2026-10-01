"""Topology audit for all 10 panels (organizer contract, discussion 743819):
longitudinal order = milepost (then link_id); conservation connectivity = incoming/outgoing_link_ids;
ramps = ramp_attachment_map.csv. Checks what our pipelines assume:
  * Task 1 sorts stations by milepost (per panel model, symmetric neighbour features)
  * Task 2 sorts links by order_index and pools all panels (features named up/dn by array direction)
Outputs topology_audit_links.csv (link -> sensored upstream / downstream, ramps, position, spacing) and a summary."""
import glob, json
import numpy as np, pandas as pd

import os
REL = os.environ.get('TFB_REL') or glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
PANELS = os.environ['TFB_PANELS'].split(',') if os.environ.get('TFB_PANELS') else ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
T2 = set(p.split('/')[-1] for p in glob.glob(f'{REL}/task2/*') if '.' not in p.split('/')[-1])
rows, summ = [], []
for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype=str)
    topo['link_id'] = topo.link_id.astype(str)
    ram = pd.read_csv(f'{net}/ramp_attachment_map.csv', dtype=str)
    fd = pd.read_csv(f'{net}/fd_parameters.csv', dtype={'link_id': str})
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).drop_duplicates('link_id').set_index('link_id')
    f0 = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))[0]
    m = pd.read_parquet(f0, columns=['station_id', 'link_id', 'milepost', 'direction']).drop_duplicates('station_id')
    m['link_id'] = m.link_id.astype(str)
    st_per_link = m.groupby('link_id').station_id.nunique()
    mp = m.groupby('link_id').milepost.mean()
    sens = set(m.link_id)
    T = topo.drop_duplicates('link_id').set_index('link_id')
    inc = {l: [x for x in str(T.loc[l, 'incoming_link_ids']).split(';') if x and x != 'nan'] for l in T.index}
    out = {l: [x for x in str(T.loc[l, 'outgoing_link_ids']).split(';') if x and x != 'nan'] for l in T.index}

    def walk(l, nxt, k=3):
        """first k sensored links reached by following nxt through latent (unsensored, non-connector) links."""
        res, frontier, seen, hops = [], [l], {l}, 0
        while frontier and len(res) < k and hops < 60:
            new = []
            for a in frontier:
                for b in nxt.get(a, []):
                    if b in seen or b.startswith('CONN'):
                        continue
                    seen.add(b)
                    if b in sens:
                        res.append(b)
                    else:
                        new.append(b)
            frontier = new; hops += 1
        return res[:k]

    order_mp = m.sort_values(['milepost', 'link_id'])  # scorer order
    order_mp = order_mp.drop_duplicates('link_id').link_id.tolist()
    oi = pd.to_numeric(T.order_index, errors='coerce').reindex(order_mp)
    rho = pd.Series(np.arange(len(order_mp))).corr(pd.Series(oi.to_numpy()), method='spearman')
    signs = []
    for l in order_mp:
        up = walk(l, inc, 1)
        if up:
            signs.append(np.sign(mp[up[0]] - mp[l]))
    signs = np.array(signs)
    # adjacency: is the next link by milepost the first sensored neighbour by topology (either direction)?
    adj_ok = 0; adj_n = 0
    for a, b in zip(order_mp[:-1], order_mp[1:]):
        adj_n += 1
        if b in walk(a, out, 1) or b in walk(a, inc, 1):
            adj_ok += 1
    ramps_on = ram[ram.ramp_type.str.upper().isin(['OR', 'ON'])] if 'ramp_type' in ram else ram.iloc[:0]
    lanes = lk.lanes.reindex(order_mp).to_numpy(float)
    for i, l in enumerate(order_mp):
        rows.append(dict(panel=panel, link_id=l, pos_milepost_rank=i, milepost=mp[l], order_index=oi.get(l),
                         stations=int(st_per_link.get(l, 0)), sensored_up=';'.join(walk(l, inc, 3)),
                         sensored_dn=';'.join(walk(l, out, 3)),
                         on_ramps=str(T.loc[l, 'on_ramp_link_ids']) if l in T.index else '',
                         off_ramps=str(T.loc[l, 'off_ramp_link_ids']) if l in T.index else '',
                         lanes=lanes[i], lane_change_next=(lanes[i + 1] - lanes[i]) if i + 1 < len(lanes) else np.nan,
                         spacing_next_km=(mp[order_mp[i + 1]] - mp[l]) if i + 1 < len(order_mp) else np.nan,
                         length_km=lk.length_km.get(l), vf_links=lk.free_speed_kmh.get(l)))
    summ.append(dict(panel=panel, task2=panel in T2, direction=str(m.direction.iloc[0]), sensored_links=len(order_mp),
                     stations=len(m), links_with_multiple_stations=int((st_per_link > 1).sum()),
                     topo_links=len(T), spearman_orderindex_vs_milepost=round(float(rho), 4),
                     upstream_has_higher_milepost_share=round(float((signs > 0).mean()), 3) if len(signs) else None,
                     upstream_has_lower_milepost_share=round(float((signs < 0).mean()), 3) if len(signs) else None,
                     n_upstream_checks=len(signs), milepost_neighbours_topologically_adjacent=f'{adj_ok}/{adj_n}',
                     milepost_ties=int(m.milepost.duplicated().sum()), n_ramps=len(ram),
                     lane_changes=int(np.nansum(np.abs(np.diff(lanes)) > 0)),
                     vf_links_unique=int(lk.free_speed_kmh.reindex(order_mp).nunique()),
                     vf_links_equals_fd=bool(np.allclose(lk.free_speed_kmh.reindex(order_mp).to_numpy(float),
                                                          fd.groupby('link_id').free_speed_kmh.mean().reindex(order_mp).to_numpy(float), equal_nan=True))))
    print(summ[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/topology_audit_links.csv', index=False)
pd.DataFrame(summ).to_csv(f'{OUT}/topology_audit_summary.csv', index=False)
print(pd.DataFrame(summ).to_string())
