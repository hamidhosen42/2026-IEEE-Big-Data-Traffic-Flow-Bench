"""Experiment 1b: Kalman, neighbour ratio transfer, and a stacked gradient-boosting model (day-grouped CV)."""
import sys, time, json
import numpy as np
sys.path.insert(0, 'analysis')
from t1lab import *
from sklearn.ensemble import HistGradientBoostingRegressor

PANEL = sys.argv[1] if len(sys.argv) > 1 else 'D12_I405_S'
OCC = len(sys.argv) > 2 and sys.argv[2] == 'occ'
t0 = time.time()
X, st, days, reg, dow = load_cube(PANEL)
D, T, S = X['v'].shape
lanes = st.lanes.to_numpy()[None, None, :]
vf = st.free_speed_kmh.to_numpy()[None, None, :]
hv, hq = hist_profile(X, dow)
hv = np.where(np.isfinite(hv), hv, np.nanmean(hv)); hq = np.where(np.isfinite(hq), hq, np.nanmean(hq))
mql = X['mq'] / lanes                                      # work in per-lane flow

tv = interp_axis(X['mv'], 1); tq = interp_axis(mql, 1) * lanes
tv = np.where(np.isfinite(tv), tv, hv); tq = np.where(np.isfinite(tq), tq, hq)
sv = interp_axis(X['mv'], 2); sq = interp_axis(mql, 2) * lanes
sqt = interp_axis(X['mq'], 2)
sqt = np.where(np.isfinite(sqt), sqt, tq)
sv = np.where(np.isfinite(sv), sv, tv); sq = np.where(np.isfinite(sq), sq, tq)
res = []

kv = kf_channel(X['mv'], hv); kq = kf_channel(X['mq'], hq)

# neighbour ratio transfer (up/downstream station, ratio interpolated in time)
rv = [ratio_transfer(X['mv'], o) for o in (-1, 1)]
rq = [ratio_transfer(X['mq'], o) for o in (-1, 1)]
with np.errstate(all='ignore'):
    rvm = np.nanmean(np.stack(rv), 0); rqm = np.nanmean(np.stack(rq), 0)
rvm = np.where(np.isfinite(rvm), rvm, tv); rqm = np.where(np.isfinite(rqm), rqm, tq)


# ------------------------------------------------ stacked model
def gap_dist(obs):
    """Steps to the previous / next released value along time, per (day, station)."""
    fin = np.isfinite(obs)
    idx = np.where(fin, np.arange(T)[None, :, None], -10 ** 6)
    prev = np.maximum.accumulate(idx, axis=1)
    idx2 = np.where(fin, np.arange(T)[None, :, None], 10 ** 6)
    nxt = np.minimum.accumulate(idx2[:, ::-1], axis=1)[:, ::-1]
    t = np.arange(T)[None, :, None]
    return np.minimum(t - prev, 99), np.minimum(nxt - t, 99)


gp, gn = gap_dist(X['mv'])
nb_obs = np.zeros_like(X['mv'])
for o in (-1, 1):
    nb_obs += np.isfinite(np.roll(X['mv'], o, axis=2))
tod = np.broadcast_to(np.arange(T)[None, :, None], X['v'].shape)
feat = {
    'tv': tv / vf, 'sv': sv / vf, 'kv': kv / vf, 'rvm': rvm / vf, 'hv': hv / vf,
    'tq': tq / lanes, 'sq': sq / lanes, 'kq': kq / lanes, 'rqm': rqm / lanes, 'hq': hq / lanes,
    'sqt': sqt / lanes, 'tk': tq / np.maximum(tv, 1), 'sk': sqt / np.maximum(sv, 1),
    **{f'{c}{o:+d}{dt:+d}': np.roll(np.roll(X[m], -o, axis=2), -dt, axis=1) / sc
       for c, m, sc in (('v', 'mv', vf), ('q', 'mq', lanes)) for o in (-2, -1, 1, 2) for dt in (-1, 0, 1)},
    **({f'o{o:+d}{dt:+d}': np.roll(np.roll(X['mo'], -o, axis=2), -dt, axis=1)
        for o in (-2, -1, 0, 1, 2) for dt in (-1, 0, 1) if (o, dt) != (0, 0)} if OCC else {}),
    **({'to': interp_axis(X['mo'], 1), 'so': interp_axis(X['mo'], 2)} if OCC else {}),
    'rv_up': rv[0] / vf, 'rv_dn': rv[1] / vf, 'rq_up': rq[0] / lanes, 'rq_dn': rq[1] / lanes,
    'gap_prev': gp, 'gap_next': gn, 'nb_obs': nb_obs, 'tod': tod,
    'lanes': np.broadcast_to(lanes, X['v'].shape), 'vf': np.broadcast_to(vf, X['v'].shape),
    'regime': np.broadcast_to(np.array([{'R1': .2, 'R2': .3, 'R3': .5}[r] for r in reg])[:, None, None], X['v'].shape),
}
tgt = X['target']
F = np.column_stack([np.asarray(f)[tgt] for f in feat.values()]).astype(np.float32)
yv = (X['v'] / vf)[tgt]; yq = (X['q'] / lanes)[tgt]
day_of = np.broadcast_to(np.arange(D)[:, None, None], X['v'].shape)[tgt]
fold = day_of % 3
pv = np.zeros(len(F)); pq = np.zeros(len(F))
for k in range(3):
    tr, te = fold != k, fold == k
    for y, out in ((yv, pv), (yq, pq)):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                          min_samples_leaf=100, l2_regularization=1.0, random_state=k)
        m.fit(F[tr], y[tr]); out[te] = m.predict(F[te])
    print(f'  fold {k} done {time.time() - t0:.0f}s', flush=True)
gv = tv.copy(); gq = tq.copy()
gv[tgt] = pv * np.broadcast_to(vf, X['v'].shape)[tgt]
gq[tgt] = pq * np.broadcast_to(lanes, X['v'].shape)[tgt]
res.append(report('stacked HGB v2 (more features)', X, st, reg, gv, gq))
# density model with L1 loss (LWR is an L1 metric on N = k*L)
yk = np.log((X['q'] / np.maximum(X['v'], 1))[tgt] / np.broadcast_to(lanes, X['v'].shape)[tgt])
pk = np.zeros(len(F))
for k in range(3):
    tr, te = fold != k, fold == k
    m = HistGradientBoostingRegressor(loss='absolute_error', max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                      min_samples_leaf=100, l2_regularization=1.0, random_state=k)
    m.fit(F[tr], yk[tr]); pk[te] = m.predict(F[te])
khat = np.exp(pk) * np.broadcast_to(lanes, X['v'].shape)[tgt]
k0 = gq[tgt] / np.maximum(gv[tgt], 1)
for w in (0.25, 0.5, 1.0):
    kk = np.exp((1 - w) * np.log(k0) + w * np.log(khat))
    for a_ in (0.0, 0.5, 1.0):          # a_ = share of the density correction taken by speed
        r = kk / k0
        v2 = gv.copy(); q2 = gq.copy()
        v2[tgt] = gv[tgt] / r ** a_; q2[tgt] = gq[tgt] * r ** (1 - a_)
        res.append(report(f'L1-density w={w} speed_share={a_}', X, st, reg, v2, q2))
