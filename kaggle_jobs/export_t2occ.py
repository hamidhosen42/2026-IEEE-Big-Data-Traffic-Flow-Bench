"""Export, for LOCAL experiments only, the released OCCUPANCY of every cell that carries a speed value (eligible or not); Task 2
never used occupancy.  Same cube layout as export_t2data.py (t2data_<PANEL>.npz): per Task 2 panel t2occ_<PANEL>.npz with
  train_O                       [273, 288, L]  occupancy (NaN where no speed value), train_days
  <split>_Om                    masked month layer, <split>_days (sorted)
  <split>_Oh                    [W, 12, L] official window histories
  <split>_OT                    [W, L] masked-layer occupancy at slot T
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


def cube_o(files, li):
    """occupancy of every cell with a speed value (eligible or not)."""
    D, L = len(files), len(li)
    O = np.full((D, 288, L), np.nan, np.float32); n = 0
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'occupancy'])
        x = x[x.speed_kmh.notna() & x.occupancy.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        n += len(x)
        g = x.groupby(['t', 'l']).occupancy.mean().reset_index()
        O[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.occupancy.to_numpy()
    return O, n


for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    links = topo.link_id.tolist(); li = {l: i for i, l in enumerate(links)}
    out = dict(links=np.array(links))
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    O, n = cube_o(files, li)
    out.update(train_days=np.array([day_of(f) for f in files]), train_O=O.astype(np.float16))
    log(panel, 'train occupancy cells', n, 'share', round(float(np.isfinite(O).mean()), 4), 'quantiles', np.nanquantile(O, [0.01, 0.5, 0.99]).round(3).tolist())
    for split in ('validation', 'private'):
        mfiles = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'), key=lambda f: os.path.basename(f)[-18:-8])
        mdays = [day_of(f) for f in mfiles]
        Om, n = cube_o(mfiles, li)
        wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv', dtype=str)
        W, L = len(wi), len(links)
        hist = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
        hist = hist[hist.speed_kmh.notna() & hist.occupancy.notna()]
        Oh = np.full((W, H, L), np.nan, np.float32)
        wpos = {w: i for i, w in enumerate(wi.window_id)}
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        ts = pd.to_datetime(hist.timestamp, utc=True)
        step = ((ts - hist.window_id.map(org)).dt.total_seconds() // 300).astype(int) + H
        ok = (step >= 0) & (step < H) & hist.link_id.astype(str).isin(li)
        hh = hist[ok].assign(s=step[ok].values, w=hist[ok].window_id.map(wpos).values, l=hist[ok].link_id.astype(str).map(li).values)
        g = hh.groupby(['w', 's', 'l']).occupancy.mean().reset_index()
        Oh[g.w, g.s, g.l] = g.occupancy
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
        dix = np.array([mdays.index(d) if d in mdays else -1 for d in org.dt.strftime('%Y-%m-%d')])
        OT = np.full((W, L), np.nan, np.float32); okd = dix >= 0
        OT[okd] = Om[dix[okd], tod[okd]]
        out.update({f'{split}_days': np.array(mdays), f'{split}_Om': Om.astype(np.float16), f'{split}_Oh': Oh.astype(np.float16),
                    f'{split}_OT': OT.astype(np.float16), f'{split}_window_id': wi.window_id.to_numpy().astype(str)})
        log(panel, split, 'masked occupancy share', round(float(np.isfinite(Om).mean()), 4), '| history share', round(float(np.isfinite(Oh).mean()), 4))
    np.savez_compressed(f'{OUT}/t2occ_{panel}.npz', **out)
    log(panel, 'saved', round(os.path.getsize(f'{OUT}/t2occ_{panel}.npz') / 1e6, 1), 'MB')
log('EXPORT OCC DONE')
