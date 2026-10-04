"""Export compact Task 2 arrays for LOCAL experiments (not a submission; nothing here is redistributed).
Per Task 2 panel one npz: train cubes (speed, flow; eligible observed cells, NaN elsewhere), ramp cubes, network vectors,
official window_index of every split, validation/private window histories + slot-T values, and the masked validation/private
month cubes (speed, flow, eligibility) with their ramp cubes. Same cell semantics as t2_model_v20.load_panel."""
import glob, os, time
import numpy as np, pandas as pd

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
OUT = '/kaggle/working'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
H = 12


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def day_of(f):
    return os.path.basename(f)[-18:-8].replace('_', '-')


def state_cube(files, li, with_elig=False):
    D, L = len(files), len(li)
    V = np.full((D, 288, L), np.nan, np.float32); Fq = V.copy()
    El = np.zeros((D, 288, L), bool) if with_elig else None
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        if with_elig:
            e = x[x.is_score_eligible == 1]
            El[d, e.t.to_numpy(), e.l.astype(int).to_numpy()] = True
        x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
        g = x.groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return V, Fq, El


def ramp_cube(panel, split, li, days):
    net = f'{REL}/corridors/{panel}/network'
    D, L = len(days), len(li)
    ON = np.full((D, 288, L), np.nan, np.float32); OFF = ON.copy()
    amap = pd.read_csv(f'{net}/ramp_attachment_map.csv', dtype=str)
    amap = amap[amap.nearest_mainline_link_id.isin(li)]
    rl = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id.map(li)))
    dpos = {dd: i for i, dd in enumerate(days)}
    for f in sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/ramp_states/**/*.parquet', recursive=True)):
        r = pd.read_parquet(f, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'pct_observed', 'is_score_eligible'])
        r = r[(r.is_score_eligible == 1) & r.flow_vph.notna() & (r.pct_observed >= 75) & r.ramp_link_id.isin(rl)]
        if r.empty:
            continue
        ts = pd.to_datetime(r.timestamp, utc=True)
        r = r.assign(d=ts.dt.strftime('%Y-%m-%d').map(dpos), t=(ts.dt.hour * 12 + ts.dt.minute // 5), l=r.ramp_link_id.map(rl),
                     on=r.ramp_type.str.upper().isin(['OR', 'ON'])).dropna(subset=['d'])
        for flag, A in ((True, ON), (False, OFF)):
            g = r[r.on == flag].groupby(['d', 't', 'l']).flow_vph.sum().reset_index()
            A[g.d.astype(int).to_numpy(), g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return ON, OFF


for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    links = topo.link_id.tolist(); li = {l: i for i, l in enumerate(links)}
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
    fd = pd.read_csv(f'{net}/fd_parameters.csv', dtype={'link_id': str}).groupby('link_id').mean(numeric_only=True).reindex(links)
    out = dict(links=np.array(links), vf=lk.free_speed_kmh.fillna(105).to_numpy(), cap=lk.capacity_vph.fillna(7200).to_numpy(),
               lanes=lk.lanes.fillna(4).to_numpy(), length=lk.length_km.fillna(0.5).to_numpy(),
               milepost=pd.to_numeric(topo.get('milepost', pd.Series(np.nan, index=topo.index)), errors='coerce').to_numpy(),
               order_index=pd.to_numeric(topo.order_index, errors='coerce').to_numpy(),
               fd_kjam=fd.k_jam.to_numpy() if 'k_jam' in fd else np.full(len(links), np.nan),
               fd_kcrit=fd.critical_density.to_numpy() if 'critical_density' in fd else np.full(len(links), np.nan))
    # ---- train
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    days = [day_of(f) for f in files]
    V, Fq, _ = state_cube(files, li)
    ON, OFF = ramp_cube(panel, 'train', li, days)
    out.update(train_days=np.array(days), train_V=V.astype(np.float16), train_F=Fq.astype(np.float32),
               train_ON=ON.astype(np.float16), train_OFF=OFF.astype(np.float16))
    log(panel, 'train', V.shape, 'observed share', round(float(np.isfinite(V).mean()), 3))
    for split in ('train', 'validation', 'private'):
        wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv', dtype=str)
        for c in wi.columns:
            out[f'{split}_wi_{c}'] = wi[c].fillna('').to_numpy().astype(str)
        out[f'{split}_wi_columns'] = np.array(list(wi.columns))
        if split == 'train':
            continue
        W, L = len(wi), len(links)
        hist = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
        Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy()
        wpos = {w: i for i, w in enumerate(wi.window_id)}
        hist = hist[(hist.is_score_eligible == 1) & hist.speed_kmh.notna()]
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        ts = pd.to_datetime(hist.timestamp, utc=True)
        step = ((ts - hist.window_id.map(org)).dt.total_seconds() // 300).astype(int) + H
        ok = (step >= 0) & (step < H) & hist.link_id.astype(str).isin(li)
        hh = hist[ok].assign(s=step[ok].values, w=hist[ok].window_id.map(wpos).values, l=hist[ok].link_id.astype(str).map(li).values)
        g = hh.groupby(['w', 's', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        Vh[g.w, g.s, g.l] = g.speed_kmh; Fh[g.w, g.s, g.l] = g.flow_vph
        mfiles = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'))
        mdays = [day_of(f) for f in mfiles]
        Vm, Fm, Em = state_cube(mfiles, li, with_elig=True)
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
        dix = np.array([mdays.index(d) if d in mdays else -1 for d in org.dt.strftime('%Y-%m-%d')])
        VT = np.full((W, L), np.nan, np.float32); FT = VT.copy()
        okd = dix >= 0
        VT[okd] = Vm[dix[okd], tod[okd]]; FT[okd] = Fm[dix[okd], tod[okd]]
        ONs, OFFs = ramp_cube(panel, split, li, mdays)
        tpl = pd.read_csv(f'{REL}/task2/{panel}/{split}/sample_submission_queue.csv', dtype=str, usecols=['window_id', 'timestamp', 'link_id'])
        out.update({f'{split}_Vh': Vh.astype(np.float16), f'{split}_Fh': Fh, f'{split}_VT': VT.astype(np.float16), f'{split}_FT': FT,
                    f'{split}_days': np.array(mdays), f'{split}_Vm': Vm.astype(np.float16), f'{split}_Fm': Fm, f'{split}_Em': Em,
                    f'{split}_ON': ONs.astype(np.float16), f'{split}_OFF': OFFs.astype(np.float16),
                    f'{split}_tpl_rows': np.array([len(tpl)]), f'{split}_tpl_links': np.array(sorted(tpl.link_id.unique()))})
        log(panel, split, 'windows', W, 'history filled', round(float(np.isfinite(Vh).mean()), 3), 'slot-T visible', round(float(np.isfinite(VT).mean()), 3),
            'masked month days', len(mdays), 'observed share', round(float(np.isfinite(Vm).mean()), 3), 'eligible share', round(float(Em.mean()), 3))
    np.savez_compressed(f'{OUT}/t2data_{panel}.npz', **out)
    log(panel, 'saved', round(os.path.getsize(f'{OUT}/t2data_{panel}.npz') / 1e6, 1), 'MB')
log('EXPORT DONE')
