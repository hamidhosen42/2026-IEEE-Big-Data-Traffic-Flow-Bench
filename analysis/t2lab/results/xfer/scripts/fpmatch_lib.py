import numpy as np, pandas as pd

def forecast(L, off, tod_w=2.0, panel_exact=True, min_n=5):
    Lc = L[(L.dn != 0) | (L.dI != 0)]; pred = []
    for _, r in off.iterrows():
        if r.changed == 0:
            pred.append(0.0); continue
        tol = max(3, 0.5 * abs(r.dn)); sg = np.sign(r.dn)
        m = (np.abs(Lc.dn - r.dn) <= tol) & (np.sign(Lc.dn) == sg) & (np.abs(Lc.tod - r.tod) <= tod_w)
        c = Lc[m & (Lc.panel == r.panel)] if panel_exact else Lc[m]
        if len(c) < min_n:
            c = Lc[m]
        if len(c) < min_n:
            c = Lc[(np.abs(Lc.dn - r.dn) <= tol) & (np.sign(Lc.dn) == sg)]
        pred.append(c.dI.mean() if len(c) else 0.0)
    return np.array(pred)


