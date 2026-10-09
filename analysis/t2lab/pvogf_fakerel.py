"""pvogf: build a FAKE Kaggle input tree from the local exports, so that the generated pvogf kernel can run end-to-end locally
(SMOKE) and its load_split can be parity-tested against the lab caches.  Nothing is downloaded; nothing in the project is modified.
Layout under OUT (the kernel's /kaggle/input):
  kaggle_public/corridors/<P>/network/{lwr_mainline_topology,links,ramp_attachment_map}.csv
  kaggle_public/corridors/<P>/train/mainline_states/R0/part_<YYYY_MM_DD>.parquet         first NDAYS train days (eligible + ineligible-valued cells)
  kaggle_public/corridors/<P>/<split>/mainline_states_masked/R<k>/part_<date>.parquet    validation / private masked months (eligible rows incl.
                                                                                         masked ones with NaN speed, + ineligible-valued rows)
  kaggle_public/corridors/<P>/<split>/ramp_states/part_<date>.parquet                    one on- and one off-ramp per mainline link (sums = export cubes)
  kaggle_public/task2/<P>/<split>/{window_index.csv, window_history.parquet, sample_submission_queue.csv}
  tfb-t2v22af/queue_submission_fs4af.csv, tfb-t2ogpvaf/queue_submission_pvo.csv          fake reference queue files (all template rows)
Usage: pvogf_fakerel.py OUT [panels=D7_I10_E,D12_I5_S] [ndays=90]"""
import os, sys
import numpy as np
import pandas as pd
import lab

OUT = sys.argv[1]
kw = dict(a.split('=') for a in sys.argv[2:])
PANELS = kw.get('panels', 'D7_I10_E,D12_I5_S').split(',')
NDAYS = int(kw.get('ndays', 90))
REL = f'{OUT}/kaggle_public'
XDIR = f'{lab.DATA}/x'
H, F = 12, 6


def ts_str(day, t):
    return f'{day}T{t // 12:02d}:{(t % 12) * 5:02d}:00Z'


def cube_rows(days, V, Fq, E, links, Vx=None, Fx=None, elig_rows=None):
    """long rows (timestamp, link_id, speed_kmh, flow_vph, is_score_eligible) of a [D, 288, L] cube, one frame per day."""
    out = []
    for d, day in enumerate(days):
        parts = []
        m = E[d] if elig_rows is None else elig_rows[d]                      # eligible rows (speed may be NaN when masked)
        t, l = np.where(m)
        parts.append(pd.DataFrame(dict(t=t, l=l, speed_kmh=V[d][t, l].astype(np.float64), flow_vph=Fq[d][t, l].astype(np.float64), is_score_eligible=1)))
        if Vx is not None:
            t, l = np.where(np.isfinite(Vx[d]))
            parts.append(pd.DataFrame(dict(t=t, l=l, speed_kmh=Vx[d][t, l].astype(np.float64), flow_vph=Fx[d][t, l].astype(np.float64), is_score_eligible=0)))
        g = pd.concat(parts, ignore_index=True)
        tsv = np.array([ts_str(day, x) for x in range(288)])
        g['timestamp'] = tsv[g.t.to_numpy()]; g['link_id'] = np.asarray(links)[g.l.to_numpy()]
        out.append((day, g[['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible']]))
    return out


def ramp_rows(days, ON, OFF, links):
    out = []
    for d, day in enumerate(days):
        parts = []
        for typ, A in (('ON', ON), ('OFF', OFF)):
            t, l = np.where(np.isfinite(A[d]))
            parts.append(pd.DataFrame(dict(t=t, ramp_link_id=[f'{typ}_{links[x]}' for x in l], ramp_type=typ, flow_vph=A[d][t, l].astype(np.float64),
                                           pct_observed=100.0, is_score_eligible=1)))
        g = pd.concat(parts, ignore_index=True)
        tsv = np.array([ts_str(day, x) for x in range(288)])
        g['timestamp'] = tsv[g.t.to_numpy()]
        out.append((day, g[['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'pct_observed', 'is_score_eligible']]))
    return out


refs = []
for panel in PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False); x = np.load(f'{XDIR}/t2x_{panel}.npz', allow_pickle=False)
    links = [str(a) for a in z['links']]
    net = f'{REL}/corridors/{panel}/network'; os.makedirs(net, exist_ok=True)
    pd.DataFrame(dict(link_id=links, order_index=z['order_index'])).to_csv(f'{net}/lwr_mainline_topology.csv', index=False)
    pd.DataFrame(dict(link_id=links, free_speed_kmh=z['vf'], capacity_vph=z['cap'], lanes=z['lanes'], length_km=z['length'])).to_csv(f'{net}/links.csv', index=False)
    pd.DataFrame(dict(ramp_link_id=[f'{t}_{l}' for t in ('ON', 'OFF') for l in links], nearest_mainline_link_id=links + links)).to_csv(
        f'{net}/ramp_attachment_map.csv', index=False)
    # ---- train split (first NDAYS days)
    days = [str(a) for a in z['train_days']]; assert days == sorted(days) and [str(a) for a in x['train_days']] == days
    n = min(NDAYS, len(days))
    V = z['train_V'][:n].astype(np.float32); Fq = z['train_F'][:n].astype(np.float32)
    Vx = x['train_Vx'][:n].astype(np.float32); Fx = x['train_Fx'][:n].astype(np.float32)
    d0 = f'{REL}/corridors/{panel}/train/mainline_states/R0'; os.makedirs(d0, exist_ok=True)
    for day, g in cube_rows(days[:n], V, Fq, np.isfinite(V), links, Vx, Fx):
        g.to_parquet(f'{d0}/part_{day.replace("-", "_")}.parquet', index=False)
    d1 = f'{REL}/corridors/{panel}/train/ramp_states'; os.makedirs(d1, exist_ok=True)
    for day, g in ramp_rows(days[:n], z['train_ON'][:n].astype(np.float32), z['train_OFF'][:n].astype(np.float32), links):
        g.to_parquet(f'{d1}/part_{day.replace("-", "_")}.parquet', index=False)
    # ---- released months (masked) + official windows
    for split in ('validation', 'private'):
        sd = np.array([str(a) for a in z[f'{split}_days']]); o = np.argsort(sd)
        xd = np.array([str(a) for a in x[f'{split}_days']]); ox = np.argsort(xd); assert xd[ox].tolist() == sd[o].tolist()
        Vm = z[f'{split}_Vm'][o].astype(np.float32); Fm = z[f'{split}_Fm'][o].astype(np.float32); Em = z[f'{split}_Em'][o]
        Vmx = x[f'{split}_Vmx'][ox].astype(np.float32); Fmx = x[f'{split}_Fmx'][ox].astype(np.float32)
        for k, (day, g) in enumerate(cube_rows(list(sd[o]), Vm, Fm, None, links, Vmx, Fmx, elig_rows=Em)):
            dm = f'{REL}/corridors/{panel}/{split}/mainline_states_masked/R{k % 3}'; os.makedirs(dm, exist_ok=True)
            g.to_parquet(f'{dm}/part_{day.replace("-", "_")}.parquet', index=False)
        dr = f'{REL}/corridors/{panel}/{split}/ramp_states'; os.makedirs(dr, exist_ok=True)
        for day, g in ramp_rows(list(sd[o]), z[f'{split}_ON'][o].astype(np.float32), z[f'{split}_OFF'][o].astype(np.float32), links):
            g.to_parquet(f'{dr}/part_{day.replace("-", "_")}.parquet', index=False)
        wi = lab.window_index(panel, split)
        dt = f'{REL}/task2/{panel}/{split}'; os.makedirs(dt, exist_ok=True)
        wi.to_csv(f'{dt}/window_index.csv', index=False)
        Vh = z[f'{split}_Vh'].astype(np.float32); Fh = z[f'{split}_Fh'].astype(np.float32)
        Vhx = x[f'{split}_Vhx'].astype(np.float32); Fhx = x[f'{split}_Fhx'].astype(np.float32)
        assert [str(a) for a in x[f'{split}_window_id']] == wi.window_id.tolist()
        hist = []; tpl = []
        for i, r in enumerate(wi.itertuples()):
            org = pd.Timestamp(r.forecast_origin)
            for s in range(H):
                tss = (org + pd.Timedelta(minutes=5 * (s - H))).strftime('%Y-%m-%dT%H:%M:%SZ')
                for el, VV, FF in ((1, Vh, Fh), (0, Vhx, Fhx)):
                    l = np.where(np.isfinite(VV[i, s]))[0]
                    if len(l):
                        hist.append(pd.DataFrame(dict(window_id=r.window_id, timestamp=tss, link_id=np.asarray(links)[l], speed_kmh=VV[i, s, l].astype(np.float64),
                                                      flow_vph=FF[i, s, l].astype(np.float64), is_score_eligible=el)))
            for h in range(1, F + 1):
                tss = (org + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ')
                tpl.append(pd.DataFrame(dict(window_id=r.window_id, timestamp=tss, link_id=links)))
        pd.concat(hist, ignore_index=True).to_parquet(f'{dt}/window_history.parquet', index=False)
        T = pd.concat(tpl, ignore_index=True); assert len(T) == int(z[f'{split}_tpl_rows'][0])
        T.assign(queue_pred=0).to_csv(f'{dt}/sample_submission_queue.csv', index=False)
        refs.append(T)
    print(panel, 'fake release written', flush=True)
R = pd.concat(refs, ignore_index=True)
rng = np.random.default_rng(0)
for k, f in (('tfb-t2v22af', 'queue_submission_fs4af.csv'), ('tfb-t2ogpvaf', 'queue_submission_pvo.csv')):
    os.makedirs(f'{OUT}/{k}', exist_ok=True)
    R.assign(queue_pred=(rng.random(len(R)) < 0.05).astype(int)).to_csv(f'{OUT}/{k}/{f}', index=False)
print('FAKE RELEASE DONE', OUT, PANELS, NDAYS)
