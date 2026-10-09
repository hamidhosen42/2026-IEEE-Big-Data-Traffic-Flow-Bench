"""X2 physics shock gate (pre-registered 2026-10-08, see analysis/TRANSFER_ANALYSIS_2026-10-08.md section 5).

Allowed region for NEW cells (cells outside the visible queue at T, qlast) of an ongoing window, from data before T only:
  * clusters = runs of qlast links (gaps of 1 link bridged); downstream direction: index +1 for E/N panels, -1 for W/S panels;
  * tail = most upstream link of a cluster, head = most downstream link;
  * recent state per link = last finite value in slots T-3..T-1 of the history (speed V km/h, total flow F vph),
    per-lane flow q = F / lanes, density k = q / v (veh/km/lane);
  * upstream state u = mean over the (up to) 3 non-queued links upstream of the tail; queue state c = mean over the tail link and
    the next link inside the cluster;
  * shock speed w = (q_u - q_c) / (k_u - k_c) [km/h] when both are finite and k_c - k_u > 5, clipped to [-30, 30]; else no growth;
  * at step h the tail has moved d_h = max(0, -w) * h * 5 / 60 km upstream: upstream link j (j = 1, 2, ...) is reachable when the
    summed length of links 1..j-1 upstream of the tail is < d_h; plus a margin of M = 1 link beyond the reachable links;
  * head side: M links downstream of the head.
Gate: G = D & (qlast | allowed); windows with an empty qlast are not gated; if G would be empty, D is kept."""
import numpy as np

M = 1


def downstream_sign(panel):
    return 1 if panel.split('_')[-1] in ('E', 'N') else -1


def _last_finite(a):
    """a [W, S, L] -> last finite value along S (nan if none)."""
    out = np.full(a.shape[::2], np.nan, np.float32)
    for s in range(a.shape[1]):
        v = a[:, s]; m = np.isfinite(v); out[m] = v[m]
    return out


def clusters(q):
    idx = np.where(q)[0]
    if not len(idx):
        return []
    out = []; a = idx[0]; b = idx[0]
    for i in idx[1:]:
        if i - b <= 2:
            b = i
        else:
            out.append((a, b)); a = b = i
    out.append((a, b))
    return out


def allowed_region(Vh, Fh, qlast, lanes, length, panel, F=6, margin=M):
    """Vh, Fh [W, >=3, L] history (last axis-1 slot = T-1); qlast [W, L] -> allowed [W, F, L] bool, w [W, n_clusters] list."""
    W, L = qlast.shape; s = downstream_sign(panel)
    v = _last_finite(Vh[:, -3:]); f = _last_finite(Fh[:, -3:])
    q = f / lanes[None]; k = q / np.where(v > 1, v, np.nan)
    allowed = np.zeros((W, F, L), bool); ws = []
    for i in range(W):
        wl = []
        for lo, hi in clusters(qlast[i]):
            tail, head = (lo, hi) if s == 1 else (hi, lo)
            up = [tail - s * j for j in range(1, 4) if 0 <= tail - s * j < L and not qlast[i, tail - s * j]]
            cq = [tail] + ([tail + s] if lo <= tail + s <= hi else [])
            qu = np.nanmean(q[i, up]) if up else np.nan; ku = np.nanmean(k[i, up]) if up else np.nan
            qc = np.nanmean(q[i, cq]); kc = np.nanmean(k[i, cq])
            w = np.nan
            if np.isfinite(qu) and np.isfinite(ku) and np.isfinite(qc) and np.isfinite(kc) and kc - ku > 5:
                w = float(np.clip((qu - qc) / (kc - ku), -30, 30))
            wl.append(w)
            grow = 0.0 if not np.isfinite(w) else max(0.0, -w)
            # cells of the cluster span (bridged gaps) are allowed
            allowed[i, :, lo:hi + 1] = True
            for h in range(1, F + 1):
                d = grow * h * 5 / 60.0
                j = 1; cum = 0.0; reach = 0
                while 0 <= tail - s * j < L:
                    if cum < d:
                        reach = j; cum += float(length[tail - s * j]); j += 1
                    else:
                        break
                for jj in range(1, reach + margin + 1):
                    x = tail - s * jj
                    if 0 <= x < L:
                        allowed[i, h - 1, x] = True
                for jj in range(1, margin + 1):
                    x = head + s * jj
                    if 0 <= x < L:
                        allowed[i, h - 1, x] = True
        ws.append(wl)
    return allowed, ws


def gate(D, qlast, allowed):
    """D [F, L] decision, qlast [L], allowed [F, L] -> gated decision."""
    if not qlast.any():
        return D
    G = D & (qlast[None] | allowed)
    return G if G.any() else D
