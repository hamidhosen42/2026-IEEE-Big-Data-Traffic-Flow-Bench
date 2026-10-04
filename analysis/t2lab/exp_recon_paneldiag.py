"""EXPERIMENT recon, side check: the kernel's panel-exclusion diagnostic at wider gates, on the RELEASED cells of the
validation + private months of the 8 Task 2 panels (data/t2local) and of D12_I405_S (private month only, local copy).
swap_fdrecon.py excludes a panel from R/T when, on released eligible cells with v < GATE*vf,
rel = (q/lanes/v - kFD(v)) / kFD(v) has |median| > 2 %, or an IQR outside [-5 %, +5 %], or fewer than 200 cells; the automated
chain SKIPS a recon probe when any panel is excluded. Task 1 is offline, so released observations of any time may be used.
FD parameters: links.csv vf / capacity / lanes + fd_parameters k_jam, k_crit as exported (checked: k_crit == cap / vf)."""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', 't1lab'))
OUT = os.path.join(HERE, 'results', 'recon'); os.makedirs(OUT, exist_ok=True)
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
GATES = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
BANDS = [(0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 9)]
rows, brows = [], []


def diag(panel, v, q, lanes, vf, cap, kjam):
    """v, q: [cells, L] released eligible observations (NaN elsewhere); per-link vectors."""
    kc = cap / np.clip(vf, 1, None)
    kj = np.maximum(np.where(np.isfinite(kjam), kjam, 100.0 * lanes), 1.05 * kc)
    capl, kcl, kjl = cap / lanes, kc / lanes, kj / lanes
    w = capl / np.maximum(kjl - kcl, 1e-6)
    bad = (w < 8) | (w > 40) | ~np.isfinite(w) | (kjl < 60) | (kjl > 200) | ~np.isfinite(kjl)
    ok = np.isfinite(v) & np.isfinite(q) & ~bad[None, :]
    k = q / lanes[None, :] / np.clip(v, 1, None)
    kfd = w[None, :] * kjl[None, :] / (np.clip(v, 1, None) + w[None, :])
    rel = (k - kfd) / kfd; r = v / vf[None, :]
    n_all = int(ok.sum())
    for g in GATES:
        x = rel[ok & (r < g)]
        if len(x) < 200:
            rows.append(dict(panel=panel, gate=g, n_gate=len(x), share=len(x) / max(n_all, 1), excluded=True)); continue
        med, q25, q75 = np.median(x), np.percentile(x, 25), np.percentile(x, 75)
        rows.append(dict(panel=panel, gate=g, n_gate=len(x), share=len(x) / n_all, median=med, q25=q25, q75=q75,
                         excluded=bool(abs(med) > 0.02 or q25 < -0.05 or q75 > 0.05)))
    for a, b in BANDS:
        m = ok & (r >= a) & (r < b); x = rel[m]
        brows.append(dict(panel=panel, band=f'{a}-{b}', share=m.sum() / max(n_all, 1), n=int(m.sum()),
                          median=np.median(x) if len(x) else np.nan, q25=np.percentile(x, 25) if len(x) else np.nan,
                          q75=np.percentile(x, 75) if len(x) else np.nan,
                          on_congested_branch=float((np.abs(x) < 0.10).mean()) if len(x) else np.nan))
    return int(bad.sum()), len(bad), n_all


for panel in PANELS:
    z = np.load(os.path.join(ROOT, 'data', 't2local', f't2data_{panel}.npz'), allow_pickle=True)
    lanes, vf, cap, kjam, kcrit = (z[k].astype(float) for k in ('lanes', 'vf', 'cap', 'fd_kjam', 'fd_kcrit'))
    v = np.concatenate([z[f'{s}_Vm'].astype(np.float64).reshape(-1, len(lanes)) for s in ('validation', 'private')])
    q = np.concatenate([z[f'{s}_Fm'].astype(np.float64).reshape(-1, len(lanes)) for s in ('validation', 'private')])
    nb, nl, n = diag(panel, v, q, lanes, vf, cap, kjam)
    print(f'{panel:12s} links {nl:3d} implausible-FD links {nb} | released cells {n} | max |k_crit - cap/vf| / k_crit '
          f'{np.nanmax(np.abs(kcrit - cap / vf) / kcrit):.2e}', flush=True)
import t1lab as L  # noqa: E402
Z = L.Data(); st = Z.st
P = Z.priv; pe = P['melig'] & np.isfinite(P['mv']) & np.isfinite(P['mq'])
v = np.where(pe, P['mv'], np.nan).astype(np.float64).reshape(-1, Z.S); q = np.where(pe, P['mq'], np.nan).astype(np.float64).reshape(-1, Z.S)
nb, nl, n = diag('D12_I405_S', v, q, st.lanes.to_numpy(float), st.vf.to_numpy(float), st.cap.to_numpy(float), st.k_jam.to_numpy(float))
print(f'D12_I405_S   links {nl:3d} implausible-FD links {nb} | released cells {n} (private month only)')
G = pd.DataFrame(rows); B = pd.DataFrame(brows)
G.to_csv(os.path.join(OUT, 'paneldiag_gate.csv'), index=False); B.to_csv(os.path.join(OUT, 'paneldiag_band.csv'), index=False)
pd.set_option('display.width', 250)
print('\nkernel exclusion diagnostic per gate (released validation + private cells):')
print(G.round(4).to_string(index=False))
print('\nexcluded panels per gate:', {g: G[(G.gate == g) & G.excluded].panel.tolist() for g in GATES})
print('\nper speed band (share of released cells; rel = (k - kFD(v)) / kFD(v); on_congested_branch = share with |rel| < 10 %):')
print(B.round(4).to_string(index=False))
