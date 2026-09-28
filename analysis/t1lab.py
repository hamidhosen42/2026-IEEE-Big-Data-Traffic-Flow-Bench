"""Task 1 / Task 3 lab: day x time x station matrices, candidate reconstructions, and scores.

Every method returns (speed, flow) arrays shaped like the truth cube (D, 288, S). Only target
cells are scored. Task 3 is approximated locally with two LWR proxies, because the organizer
boundary flows are withheld:
  proxy_obs    rhs = dN of the released truth (perfect answer -> 1.0)
  proxy_smooth rhs = dN of the truth smoothed over 3 steps (a stand-in for the noise-free state)
"""
import glob, re
import numpy as np
import pandas as pd

REL = 'data/kaggle_public'


def load_cube(panel):
    C = f'{REL}/corridors/{panel}'
    full = pd.concat([pd.read_parquet(f, columns=['date', 'timestamp', 'station_id', 'link_id', 'milepost',
                                                   'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
                      for f in sorted(glob.glob(f'{C}/train/mainline_states/*/*.parquet'))], ignore_index=True)
    parts = []
    for f in sorted(glob.glob(f'{C}/train/mainline_states_masked/*/*.parquet')):
        d = pd.read_parquet(f, columns=['timestamp', 'station_id', 'speed_kmh', 'flow_vph', 'occupancy'])
        d['regime'] = re.search(r'mask_regime=(R\d)', f).group(1)
        parts.append(d)
    masked = pd.concat(parts, ignore_index=True).rename(columns={'speed_kmh': 'm_speed', 'flow_vph': 'm_flow', 'occupancy': 'm_occ'})
    d = full.merge(masked, on=['timestamp', 'station_id'], how='inner')
    fd = pd.read_csv(f'{C}/network/fd_parameters.csv').set_index('link_id')
    st = d.drop_duplicates('station_id').set_index('station_id')[['link_id', 'milepost']]
    st = st.join(fd[['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph', 'critical_density', 'k_jam']], on='link_id')
    st = st.sort_values('milepost')                       # physical order along the corridor
    days = sorted(d.date.unique())
    ts = pd.to_datetime(d.timestamp, utc=True)
    di = pd.Index(days).get_indexer(d.date)
    ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    si = st.index.get_indexer(d.station_id)
    shape = (len(days), 288, len(st))

    def cube(col, fill=np.nan, dtype=float):
        a = np.full(shape, fill, dtype=dtype)
        a[di, ti, si] = d[col].to_numpy()
        return a
    X = dict(v=cube('speed_kmh'), q=cube('flow_vph'), mv=cube('m_speed'), mq=cube('m_flow'), mo=cube('m_occ'),
             elig=cube('is_score_eligible', 0, np.int8).astype(bool))
    X['target'] = X['elig'] & np.isnan(X['mv']) & np.isfinite(X['v'])
    reg = d.drop_duplicates('date').set_index('date').regime.reindex(days).to_numpy()
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    return X, st, np.asarray(days), reg, dow


def interp_axis(a, axis, limit=None):
    """Linear interpolation of NaNs along one axis of a 3-D cube (edges held constant)."""
    b = np.moveaxis(a, axis, -1)
    sh = b.shape
    flat = b.reshape(-1, sh[-1])
    out = flat.copy()
    x = np.arange(sh[-1])
    for i in range(flat.shape[0]):
        y = flat[i]
        ok = np.isfinite(y)
        if ok.sum() == 0 or ok.all():
            continue
        f = np.interp(x, x[ok], y[ok])
        if limit is not None:
            idx = np.where(ok)[0]
            pos = np.searchsorted(idx, x)
            dl = np.abs(x - idx[np.clip(pos - 1, 0, len(idx) - 1)])
            dr = np.abs(idx[np.clip(pos, 0, len(idx) - 1)] - x)
            f = np.where(np.minimum(dl, dr) <= limit, f, np.nan)
        out[i] = np.where(ok, y, f)
    return np.moveaxis(out.reshape(sh), -1, axis)


def hist_profile(X, dow):
    """Weekday x time-of-day x station mean over released (non-NaN) masked cells."""
    prof_v = np.full((7, 288, X['v'].shape[2]), np.nan); prof_q = prof_v.copy()
    for w in range(7):
        sel = dow == w
        with np.errstate(all='ignore'):
            prof_v[w] = np.nanmean(X['mv'][sel], axis=0); prof_q[w] = np.nanmean(X['mq'][sel], axis=0)
    return prof_v[dow], prof_q[dow]


# ---------------------------------------------------------------- scoring
def s_state(X, st, reg, v, q):
    out = {}
    lanes = st.lanes.to_numpy()[None, None, :]
    for r in ('R1', 'R2', 'R3'):
        m = X['target'] & (reg == r)[:, None, None]
        rv = np.sqrt(np.mean((v[m] - X['v'][m]) ** 2))
        rq = np.sqrt(np.mean(((q - X['q']) / lanes)[m] ** 2))
        out[r] = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
    return np.mean(list(out.values())), out


def _N(v, q, st):
    return q / np.clip(v, 1, None) * st.length_km.to_numpy()[None, None, :]


def lwr_proxy(X, st, reg, v, q, smooth=False):
    tgt = X['target']
    vv = np.where(tgt, v, X['v']); qq = np.where(tgt, q, X['q'])
    N = _N(vv, qq, st); Nt = _N(X['v'], X['q'], st)
    if smooth:
        pad = np.pad(Nt, ((0, 0), (1, 1), (0, 0)), mode='edge')
        Nt = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3
    e = X['elig']
    ok = e[:, :-1] & e[:, 1:]
    touch = tgt[:, :-1] | tgt[:, 1:]            # transitions untouched by targets add equal error for everyone
    dN = N[:, 1:] - N[:, :-1]; dNt = Nt[:, 1:] - Nt[:, :-1]
    out = {}
    for r in ('R1', 'R2', 'R3'):
        m = ok & (reg == r)[:, None, None]
        out[r] = max(0, 1 - np.abs(dN - dNt)[m].sum() / np.abs(dNt)[m].sum())
    return np.mean(list(out.values())), out


def s_fd(X, st, v, q):
    tgt = X['target']
    L = st.lanes.to_numpy()[None, None, :]
    k = q / np.clip(v, 1, None) / L
    ql = q / L
    vf = st.free_speed_kmh.to_numpy()[None, None, :]
    cap = st.capacity_vph.to_numpy()[None, None, :] / L
    kc = st.critical_density.to_numpy()[None, None, :] / L
    kj = st.k_jam.to_numpy()[None, None, :] / L
    qfd = np.where(k <= kc, vf * k, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - k, 0))
    return max(0, 1 - np.abs(ql - qfd)[tgt].sum() / np.abs(ql)[tgt].sum())


def report(name, X, st, reg, v, q):
    s, sr = s_state(X, st, reg, v, q)
    lo, lor = lwr_proxy(X, st, reg, v, q)
    fd = s_fd(X, st, v, q)
    phys = (fd + 2 * lo) / 3
    print(f'{name:44s} S_state={s:.4f} [R3 {sr["R3"]:.4f}]  S_FD={fd:.3f}  '
          f'LWR={lo:.3f} [R3 {lor["R3"]:.3f}]  '
          f'~T1+T3={0.35 * s + 0.15 * phys:.4f}', flush=True)
    return dict(name=name, s_state=s, s_fd=fd, lwr=lo, t13=0.35 * s + 0.15 * phys)


def kalman_rts(y, q, r):
    """Local-level Kalman filter + RTS smoother along the last axis; q, r shaped like y[..., 0]."""
    T = y.shape[-1]; fin = np.isfinite(y)
    x = np.where(fin[..., 0], np.nan_to_num(y[..., 0]), 0.0); p = np.where(fin[..., 0], r, 1e12)
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
    """shuangranse recipe: smooth same-day deviations from the profile with a Kalman/RTS pass per station-day."""
    dev = np.moveaxis(obs - base, 1, -1)                    # (D, S, T)
    with np.errstate(all='ignore'):
        sd = np.nanstd(dev, axis=-1)
    sd = np.where(np.isfinite(sd), np.maximum(sd, 1e-3), 1e-3)
    sm = kalman_rts(dev, (qs * sd) ** 2, (rs * sd) ** 2)
    return np.clip(base + np.moveaxis(sm, -1, 1), 0, None)


def ratio_transfer(obs, offset, log=True):
    """Estimate a station from its neighbour at `offset` (+1 = next station by milepost):
    neighbour value now x the temporally interpolated ratio between the two stations."""
    nb = np.roll(obs, -offset, axis=2)
    if offset > 0: nb[:, :, -offset:] = np.nan
    else: nb[:, :, :-offset] = np.nan
    with np.errstate(all='ignore'):
        r = np.log(obs) - np.log(nb) if log else obs - nb
    ri = interp_axis(r, 1)
    est = nb * np.exp(ri) if log else nb + ri
    return est
