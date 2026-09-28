"""Experiment 1: public-notebook Task 1 recipes vs simple baselines, scored on S_state and Task 3 proxies."""
import sys, time
import numpy as np
sys.path.insert(0, 'analysis')
from t1lab import *

PANEL = sys.argv[1] if len(sys.argv) > 1 else 'D12_I405_S'
t0 = time.time()
X, st, days, reg, dow = load_cube(PANEL)
print(PANEL, 'cube', X['v'].shape, 'targets', int(X['target'].sum()), f'{time.time() - t0:.0f}s', flush=True)
lanes = st.lanes.to_numpy()[None, None, :]
vf = st.free_speed_kmh.to_numpy()[None, None, :]
cap = st.capacity_vph.to_numpy()[None, None, :]
L = st.length_km.to_numpy()[None, None, :]
hv, hq = hist_profile(X, dow)
res = []

# reference points for calibrating the proxies
res.append(report('perfect (released truth)', X, st, reg, X['v'], X['q']))
res.append(report('historical mean (official baseline)', X, st, reg, hv, hq))

# temporal / spatial interpolation (spatial = physical milepost order)
tv = interp_axis(X['mv'], 1); tq = interp_axis(X['mq'], 1)
sv = interp_axis(X['mv'], 2); sq = interp_axis(X['mq'], 2)
tv = np.where(np.isfinite(tv), tv, hv); tq = np.where(np.isfinite(tq), tq, hq)
sv = np.where(np.isfinite(sv), sv, tv); sq = np.where(np.isfinite(sq), sq, tq)
res.append(report('temporal interp', X, st, reg, tv, tq))
res.append(report('spatial interp (milepost)', X, st, reg, sv, sq))
for a in (0.85, 0.7, 0.5):
    res.append(report(f'blend speed T, flow {a:.2f}T+{1 - a:.2f}S', X, st, reg, tv, a * tq + (1 - a) * sq))

# ---- soukeaizenz v2..v13 recipe (limit 12 temporal, limit 8 spatial, 85/15, then physics steps)
t12v = interp_axis(X['mv'], 1, 12); t12q = interp_axis(X['mq'] / lanes, 1, 12)
s8v = interp_axis(t12v, 2, 8); s8q = interp_axis(t12q, 2, 8)
pv = np.where(np.isfinite(t12v), 0.85 * t12v + 0.15 * np.where(np.isfinite(s8v), s8v, t12v),
              np.where(np.isfinite(s8v), 0.9 * s8v + 0.1 * hv, hv))
pq = np.where(np.isfinite(t12q), 0.85 * t12q + 0.15 * np.where(np.isfinite(s8q), s8q, t12q),
              np.where(np.isfinite(s8q), 0.9 * s8q + 0.1 * hq / lanes, hq / lanes)) * lanes
pv = np.clip(pv, 5, vf * 1.05); pq = np.maximum(pq, 50)
res.append(report('souk: 85/15 blend', X, st, reg, pv, pq))
k = pq / np.maximum(pv, 1)
kp = np.pad(k, ((0, 0), (1, 1), (0, 0)), mode='edge')
ks = (kp[:, :-2] + kp[:, 1:-1] + kp[:, 2:]) / 3
pq2 = np.clip(0.85 * pq + 0.15 * ks * pv, 50, cap * 1.15)
res.append(report('souk: + density rolling 85/15', X, st, reg, pv, pq2))
kc = cap / vf; kj = st.k_jam.to_numpy()[None, None, :]; w = cap / np.maximum(kj - kc, 1)
k = pq2 / np.maximum(pv, 1)
qfd = np.where(k <= kc, vf * k, w * np.maximum(kj - k, 0))
ok = (qfd > 0) & (qfd < cap * 1.3)
pq3 = np.where(ok, 0.85 * pq2 + 0.15 * qfd, pq2)
res.append(report('souk: + soft FD 0.15', X, st, reg, pv, pq3))
# conservation correction between consecutive links, cascading in link order, 5 passes
def conservation(q, alpha=0.4, passes=5):
    q = q.copy()
    for _ in range(passes):
        for j in range(q.shape[2] - 1):
            c = alpha * 0.5 * (q[:, :, j + 1] - q[:, :, j])
            q[:, :, j] += c; q[:, :, j + 1] -= c
    return np.clip(q, 50, cap * 1.15)


res.append(report('souk: + conservation 0.40 x5 (full recipe)', X, st, reg, pv, conservation(pq3)))
res.append(report('souk: conservation only on top of 85/15', X, st, reg, pv, conservation(pq)))

# ---- shuangranse: weekday x slot profile + local-level Kalman/RTS on same-day deviations
def kalman_rts(y, q, r):
    """y: (..., T) with NaNs; q, r broadcast over leading dims."""
    T = y.shape[-1]; fin = np.isfinite(y)
    x = np.where(fin[..., 0], y[..., 0], 0.0); p = np.where(fin[..., 0], r, 1e12)
    xf = np.zeros_like(y); pf = np.zeros_like(y); xp = np.zeros_like(y); pp = np.zeros_like(y)
    for t in range(T):
        if t: p = p + q
        xp[..., t], pp[..., t] = x, p
        o = fin[..., t]; g = p / (p + r)
        x = np.where(o, x + g * (np.nan_to_num(y[..., t]) - x), x); p = np.where(o, (1 - g) * p, p)
        xf[..., t], pf[..., t] = x, p
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        xs[..., t] = xf[..., t] + pf[..., t] / pp[..., t + 1] * (xs[..., t + 1] - xp[..., t + 1])
    return xs


def kf_channel(obs, base, qs=0.35, rs=0.8):
    dev = np.moveaxis(obs - base, 1, -1)                    # (D, S, T)
    with np.errstate(all='ignore'):
        sd = np.nanstd(dev, axis=-1, keepdims=True)
    sd = np.where(np.isfinite(sd), np.maximum(sd, 1e-3), 1e-3)
    sm = kalman_rts(dev, (qs * sd) ** 2, (rs * sd) ** 2)
    return np.clip(base + np.moveaxis(sm, -1, 1), 0, None)


hv0 = np.where(np.isfinite(hv), hv, np.nanmean(hv)); hq0 = np.where(np.isfinite(hq), hq, np.nanmean(hq))
kv = kf_channel(X['mv'], hv0); kq = kf_channel(X['mq'], hq0)
res.append(report('shuangranse: profile + Kalman/RTS', X, st, reg, kv, kq))
res.append(report('KF speed + blend flow 0.5T+0.5S', X, st, reg, kv, 0.5 * tq + 0.5 * sq))

import json
json.dump(res, open(f'analysis/exp_task1_{PANEL}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s')
