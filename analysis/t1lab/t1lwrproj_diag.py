"""t1lwrproj step 1 (no model): is the per-link LWR residual dN(t) explainable by station-to-station CONSERVATION with released
ramp flows?  B1 would push dN_i(t) = N_i(t+1) - N_i(t) (N = q/max(v,1)*L) towards c_i(t) = dt*(q_in(t) - q_out(t) + on_i(t) - off_i(t)).
On TRUTH (train months, 8 Task 2 panels of data/t2local, all eligible observed cells) we measure for each flux variant:
  sig    = mean |dN_true|  (the LWR signal; the scorer's rhs equals it by construction: organizer flux projected on the release)
  E_cons = sum |dN_true - c| / sum |dN_true|   (= 1 - S_LWR the TRUTH would get if the flux were this conservation estimate)
  r      = corr(dN_true, c), and corr(dN_true, dt*(on - off)) (ramps alone)
Variants: up = q of the upstream sensor - own q; dn = own q - downstream q; ctr = (q_up - q_dn)/2; both travel orientations
(index-1 upstream / index+1 upstream) are reported, the better one is the physical one.
Usage: t1lwrproj_diag.py"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANELS = ('D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S')
DT = 5.0 / 60.0


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def main():
    print(f'{"panel":10s} {"orient":6s} {"var":4s} {"n":>9s} {"sig":>7s} {"|c|":>7s} {"E_cons":>7s} {"r":>7s} {"r_ramp":>7s}')
    for p in PANELS:
        z = np.load(os.path.join(ROOT, 'data', 't2local', f't2data_{p}.npz'), allow_pickle=True)
        V = z['train_V'].astype(np.float64); F = z['train_F'].astype(np.float64)
        F = np.where(np.isfinite(V), F, np.nan)
        L = z['length'].astype(np.float64)
        ON = np.nan_to_num(z['train_ON'].astype(np.float64), nan=0.0, posinf=0.0)
        OFF = np.nan_to_num(z['train_OFF'].astype(np.float64), nan=0.0, posinf=0.0)
        N = F / np.clip(V, 1, None) * L[None, None, :]
        dN = N[:, 1:] - N[:, :-1]                                 # forward interval, flux at the interval start
        Ft = F[:, :-1]; R = DT * (ON - OFF)[:, :-1]
        for orient, sgn in (('i-1up', 1), ('i+1up', -1)):
            up = np.full_like(Ft, np.nan); dn = np.full_like(Ft, np.nan)
            if sgn == 1:
                up[..., 1:] = Ft[..., :-1]; dn[..., :-1] = Ft[..., 1:]
            else:
                up[..., :-1] = Ft[..., 1:]; dn[..., 1:] = Ft[..., :-1]
            for var, c in (('up', DT * (up - Ft) + R), ('dn', DT * (Ft - dn) + R), ('ctr', DT * (up - dn) / 2 + R)):
                ok = np.isfinite(dN) & np.isfinite(c)
                a, b, rr = dN[ok], c[ok], R[ok]
                e = np.abs(a - b).sum() / np.abs(a).sum()
                print(f'{p:10s} {orient:6s} {var:4s} {ok.sum():9d} {np.abs(a).mean():7.3f} {np.abs(b).mean():7.3f} {e:7.2f} {corr(a, b):+7.3f} '
                      f'{corr(a, rr):+7.3f}', flush=True)


if __name__ == '__main__':
    main()
