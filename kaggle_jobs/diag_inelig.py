"""Diagnostic: what do the INELIGIBLE cells (is_score_eligible == 0, pct_observed < 75) carry?  Our pipelines drop them everywhere.
For 3 panels: (1) cell counts by eligibility x value presence x pct_observed bin in train / validation-masked / private-masked /
window_history; (2) accuracy of ineligible-but-valued speeds and flows: |value - mean of the eligible temporal neighbours t-1, t+1|
compared with the same leave-one-out statistic on eligible cells; (3) queue-state agreement with the neighbours; (4) how many
Task-2 history cells (60 min before official origins) would be filled by ineligible-but-valued cells.  No submission output."""
import glob, os, time
import numpy as np, pandas as pd

T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
PANELS = ['D7_I10_W', 'D12_I5_S', 'D7_I405_N']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def load(files, n=None):
    files = sorted(files, key=lambda f: os.path.basename(f)[-18:-8])
    if n:
        files = files[:: max(1, len(files) // n)][:n]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), len(files)


def counts(x, name):
    x = x.copy()
    x['has_v'] = x.speed_kmh.notna(); x['has_q'] = x.flow_vph.notna()
    x['pbin'] = pd.cut(x.pct_observed, [-1, 0, 20, 50, 74.999, 99.999, 100], labels=['0', '1-20', '21-50', '51-74', '75-99', '100'])
    t = x.groupby(['is_score_eligible', 'pbin', 'has_v'], observed=True).size().unstack(fill_value=0)
    t['share'] = (t.sum(1) / len(x)).round(4)
    log(f'--- {name}: {len(x)} cells; columns {list(x.columns)[:14]}')
    print(t.to_string(), flush=True)
    return x


def neighbour_stats(x, links, cut, name):
    """value vs mean of ELIGIBLE temporal neighbours (t-1, t+1) on the same link."""
    ts = pd.to_datetime(x.timestamp, utc=True)
    x = x.assign(g=(ts.astype('int64') // (300 * 10 ** 9)).to_numpy(), l=x.link_id.astype(str).map(links)).dropna(subset=['l'])
    x['l'] = x.l.astype(int)
    g0 = x.g.min(); G = int(x.g.max() - g0 + 1); L = len(links)
    V = np.full((G, L), np.nan, np.float32); Q = V.copy(); EL = np.zeros((G, L), bool); PO = np.full((G, L), np.nan, np.float32)
    gi = (x.g - g0).to_numpy()
    V[gi, x.l.to_numpy()] = x.speed_kmh.to_numpy(); Q[gi, x.l.to_numpy()] = x.flow_vph.to_numpy()
    EL[gi, x.l.to_numpy()] = (x.is_score_eligible == 1).to_numpy() & x.speed_kmh.notna().to_numpy()
    PO[gi, x.l.to_numpy()] = x.pct_observed.to_numpy()
    Ve = np.where(EL, V, np.nan); Qe = np.where(EL, Q, np.nan)
    nbv = 0.5 * (Ve[:-2] + Ve[2:]); nbq = 0.5 * (Qe[:-2] + Qe[2:])            # needs both neighbours eligible
    ctr_v, ctr_q, ctr_el, ctr_po = V[1:-1], Q[1:-1], EL[1:-1], PO[1:-1]
    ok = np.isfinite(nbv) & np.isfinite(ctr_v)
    rows = []
    for nm, m in (('eligible (leave-one-out)', ok & ctr_el), ('ineligible valued, pct 51-74', ok & ~ctr_el & (ctr_po > 50)),
                  ('ineligible valued, pct 21-50', ok & ~ctr_el & (ctr_po > 20) & (ctr_po <= 50)),
                  ('ineligible valued, pct 1-20', ok & ~ctr_el & (ctr_po > 0) & (ctr_po <= 20)),
                  ('ineligible valued, pct 0', ok & ~ctr_el & (ctr_po <= 0))):
        if m.sum() < 50:
            rows.append(dict(group=nm, n=int(m.sum()))); continue
        dv = (ctr_v - nbv)[m]; dq = (ctr_q - nbq)[m & np.isfinite(ctr_q) & np.isfinite(nbq)]
        cutm = np.broadcast_to(cut[None, :], ctr_v.shape)[m]
        qc = ctr_v[m] <= cutm; qn = nbv[m] <= cutm
        rows.append(dict(group=nm, n=int(m.sum()), speed_mae=float(np.abs(dv).mean()), speed_bias=float(dv.mean()), speed_rmse=float(np.sqrt((dv ** 2).mean())),
                         flow_mae=float(np.abs(dq).mean()) if len(dq) else np.nan, flow_bias=float(dq.mean()) if len(dq) else np.nan,
                         queued_share=float(qc.mean()), queue_agree=float((qc == qn).mean()),
                         queue_agree_when_nb_queued=float((qc[qn]).mean()) if qn.any() else np.nan))
    log(f'--- {name}: value vs eligible temporal neighbours')
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    # how much of the "missing" state is recoverable: share of non-eligible cells that carry a value
    ne = ~EL
    log(f'{name}: non-eligible cells {int(ne.sum())} of {EL.size} ({ne.mean():.3f}); with a speed value {float(np.isfinite(V[ne]).mean()):.3f}')
    return V, EL


for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    links = {l: i for i, l in enumerate(topo.link_id.tolist())}
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).set_index('link_id').reindex(topo.link_id.tolist())
    cut = 0.6 * lk.free_speed_kmh.fillna(105).to_numpy()
    x, n = load(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'), 16)
    log(panel, 'train days', n)
    x = counts(x, f'{panel} train mainline_states')
    neighbour_stats(x, links, cut, f'{panel} train')
    for split in ('validation', 'private'):
        m, n = load(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'), 16)
        m = counts(m, f'{panel} {split} mainline_states_masked ({n} days)')
        neighbour_stats(m, links, cut, f'{panel} {split} masked')
        h = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
        h = counts(h, f'{panel} {split} window_history')
        hl = h[h.link_id.astype(str).isin(links)]
        tot = len(hl); el = int(((hl.is_score_eligible == 1) & hl.speed_kmh.notna()).sum()); iv = int(((hl.is_score_eligible != 1) & hl.speed_kmh.notna()).sum())
        log(f'{panel} {split} window_history: cells {tot}, eligible valued {el / tot:.3f}, ineligible valued {iv / tot:.3f}, no value {(tot - el - iv) / tot:.3f}')
log('DIAG DONE')
