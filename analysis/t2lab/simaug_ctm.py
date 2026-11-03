"""simaug: per-corridor CTM (cell transmission model, triangular FD) driven by train boundary data.

Static network per panel from data/t2local/t2data_<P>.npz (vf, cap, kcrit, kjam, lanes, length; index +1 = downstream).
Links shorter than LMIN are merged into cells (cap = min cap of the members, vf = length-harmonic mean, kj = length mean)
so that dt = DT s satisfies CFL.  Boundary conditions from the observed (masked, time-filled) train data of the same day:
  upstream ghost cell sending  = min(vf0 * k0_obs, cap0),  k0_obs = F0 / V0 (observed link 0)
  downstream ghost receiving   = min(capL, w (kjL - kL_obs))
Ramps at link l attach at the upstream interface of the cell containing l: off-ramp share beta = OFF / F_obs(upstream link),
on-ramp demand = ON (vph) with a point queue, proportional-priority merge.
Multipliers m_on[L], m_off[L] (per link) rescale the ramp demands (synthetic scenario).
Output: link speed / flow cubes [D, 288, L] (5-min means), speed = cell speed * vf_link / vf_cell, clipped to vf_link."""
import numpy as np

DT = 6.0            # s
LMIN = 0.25         # km (vf <= 130 km/h -> L/vf >= 6.9 s)


def fill_time(X):
    """linear interpolation along the slot axis within each day (edges: nearest); all-NaN columns stay NaN."""
    D, T, L = X.shape
    out = X.astype(np.float32).copy()
    t = np.arange(T, dtype=np.float32)
    for d in range(D):
        A = out[d]
        ok = np.isfinite(A)
        for l in np.where((~ok).any(0) & ok.any(0))[0]:
            m = ok[:, l]
            A[~m, l] = np.interp(t[~m], t[m], A[m, l])
    return out


def build_cells(net, lmin=LMIN):
    L = len(net['length']); cells = []; cur = []; acc = 0.0
    for l in range(L):
        cur.append(l); acc += net['length'][l]
        if acc >= lmin:
            cells.append(cur); cur = []; acc = 0.0
    if cur:
        if cells:
            cells[-1] += cur
        else:
            cells.append(cur)
    C = len(cells); cell_of = np.zeros(L, int)
    for c, mem in enumerate(cells):
        cell_of[mem] = c
    Lc = np.array([net['length'][m].sum() for m in cells])
    vfc = np.array([net['length'][m].sum() / (net['length'][m] / net['vf'][m]).sum() for m in cells])
    capc = np.array([net['cap'][m].min() for m in cells])
    kjc = np.array([(net['kjam'][m] * net['length'][m]).sum() / net['length'][m].sum() for m in cells])
    kcc = capc / vfc
    kjc = np.maximum(kjc, kcc * 1.5)
    wc = capc / (kjc - kcc)
    return dict(cells=cells, cell_of=cell_of, C=C, L=Lc, vf=vfc, cap=capc, kj=kjc, kc=kcc, w=wc)


def simulate(net, Vobs, Fobs, ON, OFF, m_on=None, m_off=None, dt=DT, lmin=LMIN, cap_mult=None):
    """Vobs, Fobs, ON, OFF: [D, 288, L] (NaN = missing; ramps all-NaN on links without a ramp). Returns Vs, Fs [D, 288, L]."""
    D, T, L = Vobs.shape
    cl = build_cells(net, lmin)
    C = cl['C']; co = cl['cell_of']
    cap = cl['cap'].copy() if cap_mult is None else cl['cap'] * cap_mult
    vf, kj, Lc = cl['vf'], cl['kj'], cl['L']
    kc = cap / vf; w = cap / (kj - kc)
    Vf = fill_time(Vobs); Ff = fill_time(Fobs)
    for A, dflt in ((Vf, net['vf']), (Ff, None)):
        bad = ~np.isfinite(A)
        if bad.any():
            if dflt is not None:
                A[bad] = np.broadcast_to(dflt[None, None, :], A.shape)[bad]
            else:
                mu = np.nanmean(A, (0, 1)); A[bad] = np.broadcast_to(mu[None, None, :], A.shape)[bad]
    ONf = np.nan_to_num(fill_time(ON), nan=0.0); OFFf = np.nan_to_num(fill_time(OFF), nan=0.0)
    if m_on is not None:
        ONf = ONf * (m_on[None, None, :] if m_on.ndim == 1 else m_on[:, None, :])
    if m_off is not None:
        OFFf = OFFf * (m_off[None, None, :] if m_off.ndim == 1 else m_off[:, None, :])
    # per cell ramp demands [D, T, C]
    onc = np.zeros((D, T, C), np.float32); offc = np.zeros((D, T, C), np.float32)
    np.add.at(onc, (slice(None), slice(None), co), ONf); np.add.at(offc, (slice(None), slice(None), co), OFFf)
    # off-ramp share at cell c's upstream interface: OFF / observed flow at the last link of cell c-1 (cell 0: link 0)
    up_link = np.array([cl['cells'][c - 1][-1] if c > 0 else 0 for c in range(C)])
    beta = np.clip(offc / np.maximum(Ff[:, :, up_link], 100.0), 0.0, 0.6).astype(np.float32)
    # boundary densities
    k0 = Ff[:, :, 0] / np.maximum(Vf[:, :, 0], 1.0)
    kL = Ff[:, :, -1] / np.maximum(Vf[:, :, -1], 1.0)
    S0 = np.minimum(net['vf'][0] * k0, net['cap'][0]).astype(np.float32)
    RL = np.minimum(net['cap'][-1], (net['cap'][-1] / max(net['kjam'][-1] - net['cap'][-1] / net['vf'][-1], 1e-3)) * np.maximum(net['kjam'][-1] - kL, 0)).astype(np.float32)
    # init density from slot 0 observed
    kl0 = Ff[:, 0, :] / np.maximum(Vf[:, 0, :], 1.0)
    k = np.zeros((D, C), np.float32)
    for c, mem in enumerate(cl['cells']):
        k[:, c] = np.minimum((kl0[:, mem] * net['length'][mem]).sum(1) / Lc[c], kc[c])
    nr = np.zeros((D, C), np.float32)   # on-ramp point queues (veh)
    steps = int(round(300 / dt)); h = dt / 3600.0
    vf32, cap32, kj32, w32, L32 = (x.astype(np.float32) for x in (vf, cap, kj, w, Lc))
    vsum = np.zeros((D, T, C), np.float32); qsum = np.zeros((D, T, C), np.float32)
    for t in range(T):
        b = beta[:, t]; r_dem = onc[:, t]; s0 = S0[:, t]; rl = RL[:, t]
        for _ in range(steps):
            S = np.minimum(vf32 * k, cap32)                              # sending [D, C]
            R = np.minimum(cap32, w32 * (kj32 - k))                      # receiving
            Sup = np.concatenate([s0[:, None], S], 1)                    # sending into interfaces 0..C  [D, C+1]
            # interface c (into cell c), c = 0..C-1: mainline after off-ramp share b[:, c]
            ms = Sup[:, :C] * (1.0 - b)
            Dr = np.minimum(r_dem + nr / h, 3000.0)
            tot = ms + Dr
            over = tot > R
            pm = np.where(tot > 0, ms / np.maximum(tot, 1e-6), 1.0)
            ym = np.where(over, np.minimum(ms, np.maximum(pm * R, R - Dr)), ms)
            ri = np.where(over, np.minimum(Dr, np.maximum(R - ym, 0.0)), Dr)
            # total flow leaving the upstream cell through interface c = ym / (1 - b)
            yout = ym / np.maximum(1.0 - b, 1e-3)
            out_last = np.minimum(S[:, -1], rl)
            outflow = np.concatenate([yout[:, 1:], out_last[:, None]], 1)
            k = np.maximum(k + h / L32 * (ym + ri - outflow), 0.0)
            nr = np.maximum(nr + (r_dem - ri) * h, 0.0)
            q = np.minimum(vf32 * k, w32 * (kj32 - k)); q = np.maximum(q, 0.0)
            v = np.where(k > 1e-3, q / np.maximum(k, 1e-3), vf32)
            vsum[:, t] += v; qsum[:, t] += q
    vsum /= steps; qsum /= steps
    Vs = np.minimum(vsum[:, :, co] * (net['vf'] / vf[co])[None, None, :], net['vf'][None, None, :]).astype(np.float32)
    Fs = qsum[:, :, co].astype(np.float32)
    return Vs, Fs, cl


def load_net(panel, data=None):
    import os; data = data or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 't2local')
    z = np.load(f'{data}/t2data_{panel}.npz', allow_pickle=False)
    return dict(vf=z['vf'].astype(np.float64), cap=z['cap'].astype(np.float64), kjam=z['fd_kjam'].astype(np.float64),
                kcrit=z['fd_kcrit'].astype(np.float64), lanes=z['lanes'].astype(np.float64), length=z['length'].astype(np.float64))
