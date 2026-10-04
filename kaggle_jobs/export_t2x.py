"""Export, for LOCAL experiments only, the released values of INELIGIBLE cells (is_score_eligible != 1 but speed present,
pct_observed 1-74: about 7 percent of all cells) that the Task 2 pipeline drops.  Same cube layout as export_t2data.py
(t2data_<PANEL>.npz): per Task 2 panel t2x_<PANEL>.npz with
  train_Vx, train_Fx            [273, 288, L]  ineligible-valued speed / flow (NaN elsewhere), train_days
  <split>_Vmx, <split>_Fmx      masked month layer, ineligible-valued cells, <split>_days
  <split>_Vhx, <split>_Fhx      [W, 12, L] official window histories, ineligible-valued cells
  <split>_VTx, <split>_FTx      [W, L] masked-layer value at slot T, ineligible-valued cells
Nothing here is redistributed; not a submission."""
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


def cube_x(files, li):
    """speed / flow of cells that are NOT score-eligible but carry a speed value."""
    D, L = len(files), len(li)
    V = np.full((D, 288, L), np.nan, np.float32); Fq = V.copy(); n_el = 0; n_x = 0
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        n_el += int(((x.is_score_eligible == 1) & x.speed_kmh.notna()).sum())
        x = x[(x.is_score_eligible != 1) & x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        n_x += len(x)
        g = x.groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return V, Fq, n_el, n_x


for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    links = topo.link_id.tolist(); li = {l: i for i, l in enumerate(links)}
    out = dict(links=np.array(links))
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    V, Fq, n_el, n_x = cube_x(files, li)
    out.update(train_days=np.array([day_of(f) for f in files]), train_Vx=V.astype(np.float16), train_Fx=Fq)
    log(panel, 'train', V.shape, 'eligible valued', n_el, 'ineligible valued', n_x, 'share of cube', round(float(np.isfinite(V).mean()), 4))
    for split in ('validation', 'private'):
        mfiles = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'), key=lambda f: os.path.basename(f)[-18:-8])
        mdays = [day_of(f) for f in mfiles]
        Vm, Fm, n_el, n_x = cube_x(mfiles, li)
        wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv', dtype=str)
        W, L = len(wi), len(links)
        hist = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
        hist = hist[(hist.is_score_eligible != 1) & hist.speed_kmh.notna()]
        Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy()
        wpos = {w: i for i, w in enumerate(wi.window_id)}
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        ts = pd.to_datetime(hist.timestamp, utc=True)
        step = ((ts - hist.window_id.map(org)).dt.total_seconds() // 300).astype(int) + H
        ok = (step >= 0) & (step < H) & hist.link_id.astype(str).isin(li)
        hh = hist[ok].assign(s=step[ok].values, w=hist[ok].window_id.map(wpos).values, l=hist[ok].link_id.astype(str).map(li).values)
        g = hh.groupby(['w', 's', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        Vh[g.w, g.s, g.l] = g.speed_kmh; Fh[g.w, g.s, g.l] = g.flow_vph
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
        dix = np.array([mdays.index(d) if d in mdays else -1 for d in org.dt.strftime('%Y-%m-%d')])
        VT = np.full((W, L), np.nan, np.float32); FT = VT.copy(); okd = dix >= 0
        VT[okd] = Vm[dix[okd], tod[okd]]; FT[okd] = Fm[dix[okd], tod[okd]]
        out.update({f'{split}_days': np.array(mdays), f'{split}_Vmx': Vm.astype(np.float16), f'{split}_Fmx': Fm,
                    f'{split}_Vhx': Vh.astype(np.float16), f'{split}_Fhx': Fh, f'{split}_VTx': VT.astype(np.float16), f'{split}_FTx': FT,
                    f'{split}_window_id': wi.window_id.to_numpy().astype(str)})
        log(panel, split, 'masked layer ineligible valued', n_x, 'share', round(float(np.isfinite(Vm).mean()), 4),
            '| history share', round(float(np.isfinite(Vh).mean()), 4), '| slot-T share', round(float(np.isfinite(VT).mean()), 4))
    np.savez_compressed(f'{OUT}/t2x_{panel}.npz', **out)
    log(panel, 'saved', round(os.path.getsize(f'{OUT}/t2x_{panel}.npz') / 1e6, 1), 'MB')
log('EXPORT X DONE')
