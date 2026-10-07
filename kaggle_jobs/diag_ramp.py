"""Diagnostic: ramp_states coverage and quality. (1) cells by eligibility x pct_observed bin x flow present (train / validation /
private); (2) accuracy of ineligible-but-valued ramp flows vs the mean of eligible temporal neighbours (t-1, t+1) on the same ramp,
compared with the same leave-one-out statistic on eligible cells; (3) inside the 90-min mainline blackouts after official Task 2
origins (validation / private): share of ramp cells with a flow value.  No submission output."""
import glob, os, time
import numpy as np, pandas as pd
T0 = time.time()
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]
PANELS = ['D7_I10_W', 'D12_I5_S', 'D7_I405_N']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def load(panel, split, n=None):
    fs = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/ramp_states/**/*.parquet', recursive=True))
    if n:
        fs = fs[:: max(1, len(fs) // n)][:n]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True), len(fs)


for panel in PANELS:
    for split in ('train', 'validation', 'private'):
        if split == 'train':
            continue
        x, nf = load(panel, split, None)
        x['has_q'] = x.flow_vph.notna()
        x['pbin'] = pd.cut(x.pct_observed, [-1, 0, 20, 50, 74.999, 99.999, 100], labels=['0', '1-20', '21-50', '51-74', '75-99', '100'])
        t = x.groupby(['is_score_eligible', 'pbin', 'has_q'], observed=True).size().unstack(fill_value=0)
        t['share'] = (t.sum(1) / len(x)).round(4)
        log(f'--- {panel} {split}: {nf} files, {len(x)} ramp cells, ramps {x.ramp_link_id.nunique()} | columns {list(x.columns)[:12]}')
        print(t.to_string(), flush=True)
        # accuracy vs eligible temporal neighbours on the same ramp
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(g=(ts.astype('int64') // (300 * 10 ** 9)))
        el = (x.is_score_eligible == 1) & x.has_q & (x.pct_observed >= 75)
        piv = x.pivot_table(index='g', columns='ramp_link_id', values='flow_vph', aggfunc='mean')
        pe = x[el].pivot_table(index='g', columns='ramp_link_id', values='flow_vph', aggfunc='mean').reindex(index=piv.index, columns=piv.columns)
        po = x.pivot_table(index='g', columns='ramp_link_id', values='pct_observed', aggfunc='mean').reindex(index=piv.index, columns=piv.columns)
        pe = pe.reindex(np.arange(piv.index.min(), piv.index.max() + 1)); piv = piv.reindex(pe.index); po = po.reindex(pe.index)
        E = pe.to_numpy(); V = piv.to_numpy(); PO = po.to_numpy()
        nb = 0.5 * (E[:-2] + E[2:]); c = V[1:-1]; ce = np.isfinite(E[1:-1]); pc = PO[1:-1]
        ok = np.isfinite(nb) & np.isfinite(c)
        rows = []
        for nm, m in (('eligible (loo)', ok & ce), ('inelig 51-74', ok & ~ce & (pc > 50) & (pc < 75)), ('inelig 21-50', ok & ~ce & (pc > 20) & (pc <= 50)),
                      ('inelig 1-20', ok & ~ce & (pc > 0) & (pc <= 20))):
            d = (c - nb)[m]
            rows.append(dict(group=nm, n=int(m.sum()), mae=float(np.abs(d).mean()) if m.any() else np.nan, bias=float(d.mean()) if m.any() else np.nan,
                             rel_mae=float((np.abs(d) / np.maximum(nb[m], 50)).mean()) if m.any() else np.nan))
        print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
        if split != 'train':
            wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv')
            inb = np.zeros(len(x), bool)
            for o in pd.to_datetime(wi.forecast_origin, utc=True):
                inb |= ((ts > o) & (ts <= o + pd.Timedelta(minutes=90))).to_numpy()
            log(f'{panel} {split}: ramp cells inside mainline blackouts {int(inb.sum())}, with flow {x.has_q[inb].mean():.3f} (outside {x.has_q[~inb].mean():.3f}),'
                f' eligible {(el[inb]).mean():.3f} (outside {el[~inb].mean():.3f})')
log('DIAG RAMP DONE')
