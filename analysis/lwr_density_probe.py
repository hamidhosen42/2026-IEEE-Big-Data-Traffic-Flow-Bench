"""Local prototype (D12_I405_S train, 60 days): residual HGB model (like T1, smaller), then density post-processing.
Train days 0-39, test 40-59. Truth = full layer; targets = masked eligible cells. Score part = 0.35*S_state + 0.10*S_LWR_proxy."""
import glob, re, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
R = 'data/kaggle_public/corridors/D12_I405_S'
fd = pd.read_csv(f'{R}/network/fd_parameters.csv', dtype={'link_id': str}).drop_duplicates('link_id').set_index('link_id')
def cube(f):
    d = pd.read_parquet(f, columns=['timestamp', 'link_id', 'milepost', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
    d = d.groupby(['timestamp', 'link_id'], as_index=False).agg(milepost=('milepost', 'first'), speed_kmh=('speed_kmh', 'mean'),
                                                                flow_vph=('flow_vph', 'mean'), e=('is_score_eligible', 'max'))
    links = d.drop_duplicates('link_id').sort_values('milepost').link_id.tolist()
    P = lambda c: d.pivot(index='timestamp', columns='link_id', values=c).reindex(columns=links).to_numpy()
    return links, P('speed_kmh'), P('flow_vph'), P('e') == 1
def interp(a, axis):
    b = np.moveaxis(a, axis, 0).copy(); x = np.arange(b.shape[0])
    for j in range(b.shape[1]):
        ok = np.isfinite(b[:, j])
        if 1 < ok.sum() < len(ok): b[:, j] = np.where(ok, b[:, j], np.interp(x, x[ok], b[ok, j]))
    return np.moveaxis(b, 0, axis)
def sh(a, dt, ds):
    o = np.full_like(a, np.nan); T, S = a.shape
    o[max(-dt, 0):T + min(-dt, 0), max(-ds, 0):S + min(-ds, 0)] = a[max(dt, 0):T + min(dt, 0), max(ds, 0):S + min(ds, 0)]; return o
days = []
for f in sorted(glob.glob(f'{R}/train/mainline_states/*/*.parquet'))[:60]:
    date = re.search(r'(\d{4}_\d{2}_\d{2})', f).group(1); mf = glob.glob(f'{R}/train/mainline_states_masked/*/*{date}.parquet')
    if not mf: continue
    links, V, Q, E = cube(f); _, MV, MQ, _ = cube(mf[0]); reg = re.search(r'mask_regime=(R\d)', mf[0]).group(1)
    vf = fd.free_speed_kmh.reindex(links).to_numpy(); ln = fd.lanes.reindex(links).to_numpy(); L = fd.length_km.reindex(links).to_numpy()
    tgt = E & np.isnan(MV) & np.isfinite(V)
    nv = MV / vf; nq = MQ / ln
    tv = interp(nv, 0); tq = interp(nq, 0); sv = interp(nv, 1); sq = interp(nq, 1)
    tv = np.where(np.isfinite(tv), tv, 1.0); tq = np.where(np.isfinite(tq), tq, np.nanmean(nq))
    F = {'tv': tv, 'tq': tq, 'sv': sv, 'sq': sq}
    for ds in (-2, -1, 1, 2):
        for dt in (-1, 0, 1): F[f'v{ds}{dt}'] = sh(nv, dt, ds); F[f'q{ds}{dt}'] = sh(nq, dt, ds)
    for dt in (-2, -1, 1, 2): F[f'v0{dt}'] = sh(nv, dt, 0); F[f'q0{dt}'] = sh(nq, dt, 0)
    X = np.column_stack([F[k][tgt] for k in F])
    kint = interp(np.where(np.isfinite(MV), MQ / np.clip(MV, 1, None), np.nan), 0)
    days.append(dict(V=V, Q=Q, E=E, tgt=tgt, X=X, tv=tv, tq=tq, vf=vf, ln=ln, L=L, reg=reg, kint=kint,
                     yv=(V / vf - tv)[tgt], yq=(Q / ln - tq)[tgt]))
tr, te = days[:40], days[40:]
Xtr = np.concatenate([d['X'] for d in tr])
mv = HGB(max_iter=300, learning_rate=0.1, max_leaf_nodes=63).fit(Xtr, np.concatenate([d['yv'] for d in tr]))
mq = HGB(max_iter=300, learning_rate=0.1, max_leaf_nodes=63).fit(Xtr, np.concatenate([d['yq'] for d in tr]))
res = {}
for d in te:
    V, Q, E, tgt, vf, ln, L = d['V'], d['Q'], d['E'], d['tgt'], d['vf'], d['ln'], d['L']
    pv = np.full(V.shape, np.nan); pq = pv.copy(); vfb = np.broadcast_to(vf, V.shape)
    pv[tgt] = (d['tv'][tgt] + mv.predict(d['X'])) * vfb[tgt]; pq[tgt] = (d['tq'][tgt] + mq.predict(d['X'])) * np.broadcast_to(ln, V.shape)[tgt]
    km = pq / np.clip(pv, 1, None); ki = d['kint']; ki = np.where(np.isfinite(ki), ki, km)
    var = {'model': (pv, pq)}
    for a in (0.15, 0.25, 0.35): var[f'blend{a}'] = (pv, (a * ki + (1 - a) * km) * pv)
    for thr in (0.6, 0.8):
        for a in (0.25, 0.5, 0.75):
            g = pv < thr * vfb; var[f'gated{thr}_{a}'] = (pv, np.where(g, a * ki + (1 - a) * km, km) * pv)
    N = Q / np.clip(V, 1, None) * L; ok = E[1:] & E[:-1] & np.isfinite(N[1:]) & np.isfinite(N[:-1]); dT = N[1:] - N[:-1]
    for k, (a, b) in var.items():
        vv = np.where(tgt, a, V); qq = np.where(tgt, b, Q); Np = qq / np.clip(vv, 1, None) * L
        r = res.setdefault((k, d['reg']), [0, 0, 0, 0, 0])
        r[0] += np.nansum(np.abs(Np[1:] - Np[:-1] - dT)[ok]); r[1] += np.nansum(np.abs(dT)[ok])
        r[2] += np.nansum((a - V)[tgt] ** 2); r[3] += np.nansum(((b - Q) / ln)[tgt] ** 2); r[4] += tgt.sum()
rows = []
for (k, reg), (a, b, s_v, s_q, n) in res.items():
    rv = np.sqrt(s_v / n); rq = np.sqrt(s_q / n)
    rows.append(dict(var=k, reg=reg, S_LWR=1 - a / b, S_state=0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)))
g = pd.DataFrame(rows).groupby('var')[['S_LWR', 'S_state']].mean()
g['score_part'] = 0.35 * g.S_state + 0.10 * g.S_LWR; g['delta'] = g.score_part - g.loc['model', 'score_part']
print(g.sort_values('score_part', ascending=False).round(4).to_string())
