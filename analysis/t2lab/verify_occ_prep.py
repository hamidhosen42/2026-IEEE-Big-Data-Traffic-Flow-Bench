"""verify_occ_prep.py : build my calibration + raw arrays (all panels, train + eval sets) and compare them with the occ side-cars;
time-alignment sanity of the occupancy cube (same-slot vs lagged association with speed)."""
import numpy as np
import lab
import verify_occ_feat as vf

SETS = ['faith3F03', 'runstart', 'offlike', 'official', 'pseudo']
MAP = dict(OT='oT', olast='olast', m3='omean3', f3='ofirst3', mx='omax12', lr3='lr3')
for panel in lab.PANELS:
    oc5, cc5 = vf.calib_all(panel)
    z = np.load(f'{lab.CACHE}/occ/calib/{panel}.npz')
    msg = [f'calib max|d ocrit| {np.max(np.abs(z["ocrit5"] - oc5)):.2e} max|d c| {np.max(np.abs(z["c5"] - cc5)):.2e}']
    for name in SETS:
        rc = vf.rows_of(panel, name)
        R = vf.raw_for_rc(panel, name, rc)
        s = np.load(f'{lab.CACHE}/occ/side/{panel}__{name}__{lab.ONGOING}.npz')
        assert np.array_equal(s['d'], rc.win.d.to_numpy()) and np.array_equal(s['t'], rc.win.t.to_numpy())
        bad = {k: float(np.max(np.abs(np.nan_to_num(R[k], nan=-99) - np.nan_to_num(s[v], nan=-99)))) for k, v in MAP.items()}
        msg.append(f'{name}: W {len(rc.win)} hasT-parity {bool(R.get("_par_hasT", True))} raw max|diff| {max(bad.values()):.1e} T-vis {np.isfinite(R["OT"]).mean():.3f}')
    C = vf.cubes(panel); O = C['O']; V = C['Vall']
    # alignment: corr(O[t], V[t]) vs corr(O[t], V[t+1]) vs corr(O[t+1], V[t]) over finite pairs (within day)
    def cr(a, b):
        m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])
    c0 = cr(O[:, 1:-1], V[:, 1:-1]); cf = cr(O[:, 1:-1], V[:, 2:]); cb = cr(O[:, 1:-1], V[:, :-2])
    nan_switch = float((np.isfinite(V[:, 1:]) != np.isfinite(V[:, :-1])).mean())
    msg.append(f'align corr(O_t,V_t) {c0:.4f} corr(O_t,V_t+1) {cf:.4f} corr(O_t,V_t-1) {cb:.4f} | within-day NaN switch rate {nan_switch:.4f}')
    lab.log(panel, ' | '.join(msg))
lab.log('PREP DONE')
