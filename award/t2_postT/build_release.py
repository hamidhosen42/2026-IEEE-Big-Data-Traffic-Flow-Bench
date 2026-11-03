"""Post-T deletion test, step 1: rebuild a release-shaped Task 2 input tree from the local exports.

The raw competition release is NOT on this machine (data/kaggle_public only holds network files and D12_I405_S). The lab exports
data/t2local/t2data_<P>.npz (eligible speed / flow cubes, masked months, official windows with history + slot T),
data/t2local/x/t2x_<P>.npz (ineligible-but-valued cells) and data/t2local/occ/t2occ_<P>.npz (occupancy) were written on Kaggle
from the real release. This script turns them back into parquet / csv files with the layout and column names the producers
read (same approach as analysis/t2lab/pvogf_fakerel.py, + occupancy, + train window_index). Values are the exported ones
(speed float16-rounded); ramps are re-filed as one ON and one OFF ramp per mainline link (sums = the exported cubes).
Fidelity is NOT the point: the test compares the producers' outputs on this tree against the same tree with post-cut rows removed.

Usage: build_release.py OUT [panels=P1,P2,...] [ndays=90]
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = f'{ROOT}/data/t2local'
OUT = sys.argv[1]
kw = dict(a.split('=') for a in sys.argv[2:])
PANELS = kw.get('panels', 'D7_I10_E,D7_I10_W,D7_I210_E,D7_I210_W,D7_I405_N,D7_I405_S,D12_I5_N,D12_I5_S').split(',')
NDAYS = int(kw.get('ndays', 90))
REL = f'{OUT}/kaggle_public'
H, F = 12, 6
TSV = None


def ts_list(day):
    return np.array([f'{day}T{t // 12:02d}:{(t % 12) * 5:02d}:00Z' for t in range(288)])


def day_rows(day, V, Fq, O, elig, Vx, Fx, links):
    """one day: eligible rows (speed NaN where masked) + ineligible-valued rows; occupancy where exported."""
    t, l = np.where(elig)
    a = pd.DataFrame(dict(t=t, l=l, speed_kmh=V[t, l].astype(np.float64), flow_vph=Fq[t, l].astype(np.float64),
                          occupancy=O[t, l].astype(np.float64), is_score_eligible=1))
    t, l = np.where(np.isfinite(Vx) & ~elig)
    b = pd.DataFrame(dict(t=t, l=l, speed_kmh=Vx[t, l].astype(np.float64), flow_vph=Fx[t, l].astype(np.float64),
                          occupancy=O[t, l].astype(np.float64), is_score_eligible=0))
    g = pd.concat([a, b], ignore_index=True).sort_values(['t', 'l'], kind='stable')
    g['timestamp'] = ts_list(day)[g.t.to_numpy()]; g['link_id'] = np.asarray(links)[g.l.to_numpy()]
    return g[['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible']].reset_index(drop=True)


def ramp_day(day, ON, OFF, links):
    parts = []
    for typ, A in (('ON', ON), ('OFF', OFF)):
        t, l = np.where(np.isfinite(A))
        parts.append(pd.DataFrame(dict(t=t, ramp_link_id=[f'{typ}_{links[x]}' for x in l], ramp_type=typ,
                                       flow_vph=A[t, l].astype(np.float64), pct_observed=100.0, is_score_eligible=1)))
    g = pd.concat(parts, ignore_index=True).sort_values(['t', 'ramp_link_id'], kind='stable')
    g['timestamp'] = ts_list(day)[g.t.to_numpy()]
    return g[['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'pct_observed', 'is_score_eligible']].reset_index(drop=True)


def by_day(days, A):
    """reorder a [D, ...] cube to sorted day order; returns (sorted days, cube)."""
    days = np.array([str(x) for x in days]); o = np.argsort(days)
    return days[o].tolist(), A[o]


def window_index(z, split):
    return pd.DataFrame({str(c): z[f'{split}_wi_{c}'].astype(str) for c in z[f'{split}_wi_columns']})


for panel in PANELS:
    z = np.load(f'{DATA}/t2data_{panel}.npz', allow_pickle=False)
    x = np.load(f'{DATA}/x/t2x_{panel}.npz', allow_pickle=False)
    o = np.load(f'{DATA}/occ/t2occ_{panel}.npz', allow_pickle=False)
    links = [str(a) for a in z['links']]
    assert [str(a) for a in x['links']] == links and [str(a) for a in o['links']] == links
    net = f'{REL}/corridors/{panel}/network'; os.makedirs(net, exist_ok=True)
    pd.DataFrame(dict(link_id=links, order_index=z['order_index'])).to_csv(f'{net}/lwr_mainline_topology.csv', index=False)
    pd.DataFrame(dict(link_id=links, free_speed_kmh=z['vf'], capacity_vph=z['cap'], lanes=z['lanes'], length_km=z['length'])).to_csv(
        f'{net}/links.csv', index=False)
    pd.DataFrame(dict(ramp_link_id=[f'{t}_{l}' for t in ('ON', 'OFF') for l in links], nearest_mainline_link_id=links + links)).to_csv(
        f'{net}/ramp_attachment_map.csv', index=False)
    # ---- train split (first NDAYS days; train is never truncated)
    days = [str(a) for a in z['train_days']]
    assert days == sorted(days) and [str(a) for a in x['train_days']] == days and [str(a) for a in o['train_days']] == days
    n = min(NDAYS, len(days))
    d0 = f'{REL}/corridors/{panel}/train/mainline_states/year_month=R0'; os.makedirs(d0, exist_ok=True)
    d1 = f'{REL}/corridors/{panel}/train/ramp_states'; os.makedirs(d1, exist_ok=True)
    for i in range(n):
        V = z['train_V'][i].astype(np.float32)
        day_rows(days[i], V, z['train_F'][i].astype(np.float32), o['train_O'][i].astype(np.float32), np.isfinite(V),
                 x['train_Vx'][i].astype(np.float32), x['train_Fx'][i].astype(np.float32), links).to_parquet(
            f'{d0}/part_{days[i].replace("-", "_")}.parquet', index=False)
        ramp_day(days[i], z['train_ON'][i].astype(np.float32), z['train_OFF'][i].astype(np.float32), links).to_parquet(
            f'{d1}/part_{days[i].replace("-", "_")}.parquet', index=False)
    tw = window_index(z, 'train'); dt = f'{REL}/task2/{panel}/train'; os.makedirs(dt, exist_ok=True)
    tw.to_csv(f'{dt}/window_index.csv', index=False)
    # ---- released months (masked) + official windows
    for split in ('validation', 'private'):
        sd, Vm = by_day(z[f'{split}_days'], z[f'{split}_Vm']); _, Fm = by_day(z[f'{split}_days'], z[f'{split}_Fm'])
        _, Em = by_day(z[f'{split}_days'], z[f'{split}_Em'])
        sx, Vmx = by_day(x[f'{split}_days'], x[f'{split}_Vmx']); _, Fmx = by_day(x[f'{split}_days'], x[f'{split}_Fmx'])
        so, Om = by_day(o[f'{split}_days'], o[f'{split}_Om'])
        assert sd == sx == so, (panel, split)
        for k, day in enumerate(sd):
            dm = f'{REL}/corridors/{panel}/{split}/mainline_states_masked/mask_regime=R{k % 3 + 1}'; os.makedirs(dm, exist_ok=True)
            day_rows(day, Vm[k].astype(np.float32), Fm[k].astype(np.float32), Om[k].astype(np.float32), Em[k],
                     Vmx[k].astype(np.float32), Fmx[k].astype(np.float32), links).to_parquet(
                f'{dm}/synthetic_mainline_{day.replace("-", "_")}.parquet', index=False)
        sr, ON = by_day(z[f'{split}_days'], z[f'{split}_ON']); _, OFF = by_day(z[f'{split}_days'], z[f'{split}_OFF'])
        dr = f'{REL}/corridors/{panel}/{split}/ramp_states'; os.makedirs(dr, exist_ok=True)
        for k, day in enumerate(sr):
            ramp_day(day, ON[k].astype(np.float32), OFF[k].astype(np.float32), links).to_parquet(
                f'{dr}/part_{day.replace("-", "_")}.parquet', index=False)
        wi = window_index(z, split)
        dt = f'{REL}/task2/{panel}/{split}'; os.makedirs(dt, exist_ok=True)
        wi.to_csv(f'{dt}/window_index.csv', index=False)
        assert [str(a) for a in x[f'{split}_window_id']] == wi.window_id.tolist() == [str(a) for a in o[f'{split}_window_id']]
        Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32); Oh = o[f'{split}_Oh'].astype(np.float32)
        Vhx = x[f'{split}_Vhx'].astype(np.float32); Fhx = x[f'{split}_Fhx'].astype(np.float32)
        hist = []; tpl = []
        for i, r in enumerate(wi.itertuples()):
            org = pd.Timestamp(r.forecast_origin)
            for s in range(H):
                tss = (org + pd.Timedelta(minutes=5 * (s - H))).strftime('%Y-%m-%dT%H:%M:%SZ')
                for el, VV, FF in ((1, Vh, Fh), (0, Vhx, Fhx)):
                    l = np.where(np.isfinite(VV[i, s]) & (np.isnan(Vh[i, s]) if el == 0 else True))[0]
                    if len(l):
                        hist.append(pd.DataFrame(dict(window_id=r.window_id, timestamp=tss, link_id=np.asarray(links)[l],
                                                      speed_kmh=VV[i, s, l].astype(np.float64), flow_vph=FF[i, s, l].astype(np.float64),
                                                      occupancy=Oh[i, s, l].astype(np.float64), is_score_eligible=el)))
            for h in range(1, F + 1):
                tss = (org + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ')
                tpl.append(pd.DataFrame(dict(window_id=r.window_id, timestamp=tss, link_id=links)))
        pd.concat(hist, ignore_index=True).to_parquet(f'{dt}/window_history.parquet', index=False)
        T = pd.concat(tpl, ignore_index=True); assert len(T) == int(z[f'{split}_tpl_rows'][0])
        T.assign(queue_pred=0).to_csv(f'{dt}/sample_submission_queue.csv', index=False)
    print(panel, 'release written', n, 'train days', flush=True)
print('RELEASE DONE', OUT, PANELS, NDAYS)
