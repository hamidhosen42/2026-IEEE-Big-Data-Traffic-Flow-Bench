"""simaug: are released train ramp flows conservative with the mainline flow?  dF_l = F[l] - F[l-1] vs net ramp at link l
(ON - OFF at link l, also at l-1) on free-flow cells (both links observed, free).  Also corridor totals."""
import sys, os, numpy as np, warnings
sys.path.insert(0, os.path.dirname(__file__)); import lab
warnings.filterwarnings('ignore')
for p in lab.PANELS:
    P = lab.load_panel(p); F = P['Fq']; V = P['V']; cut = P['cut']
    ON = np.nan_to_num(P['ON']); OFF = np.nan_to_num(P['OFF'])
    hasr = np.isfinite(P['ON']).any((0, 1)) | np.isfinite(P['OFF']).any((0, 1))
    ok = np.isfinite(F[:, :, 1:]) & np.isfinite(F[:, :, :-1]) & (V[:, :, 1:] > cut[1:]) & (V[:, :, :-1] > cut[:-1])
    okr = ok & np.isfinite(P['ON'][:, :, 1:] if True else 0) | ok & ~np.isfinite(P['ON'][:, :, 1:])
    dF = (F[:, :, 1:] - F[:, :, :-1])
    net_same = (ON - OFF)[:, :, 1:]; net_prev = (ON - OFF)[:, :, :-1]
    m = ok & hasr[None, None, 1:]
    a = dF[m]; b = net_same[m]; c = net_prev[m]
    sl = np.polyfit(b, a, 1)[0]; slp = np.polyfit(c, a, 1)[0]
    # daily totals: sum over the day of F_last - F_first vs sum of net ramps (observed-cell means)
    mF = np.nanmean(F, (0, 1)); mon = np.nanmean(np.where(np.isfinite(P['ON']), P['ON'], np.nan), (0, 1)); moff = np.nanmean(np.where(np.isfinite(P['OFF']), P['OFF'], np.nan), (0, 1))
    print(f"{p}: free-flow dF vs net ramp same link: corr {np.corrcoef(a, b)[0, 1]:+.3f} slope {sl:+.3f} | prev link corr {np.corrcoef(a, c)[0, 1]:+.3f} slope {slp:+.3f} | "
          f"sd dF {a.std():.0f} sd net {b.std():.0f} | corridor F last-first {mF[-1] - mF[0]:+.0f} vs sum(ON)-sum(OFF) {np.nansum(mon) - np.nansum(moff):+.0f} vph; mean F {np.nanmean(mF):.0f}; max ramp mean {np.nanmax(mon):.0f}")
    lab.drop_panel(p)
