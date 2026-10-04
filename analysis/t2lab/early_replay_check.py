"""early track (light): brute-force check of the replay evidence (blanks, day ranges) and of the mixture formula for a few
(start, window) items, written independently of early_lib.item_stats / design_priors (plain loops, no cumulative sums)."""
import numpy as np
from scipy.special import betaln
import lab, early_lib as el, early_engine as en

F = lab.F
worst = 0.0; nchk = 0
for panel in ('D12_I5_S', 'D7_I405_S', 'D7_I210_W'):
    b = el.march_bundle(panel)
    for d0 in (7, 14, 21):
        items, meta = en.items_replay(b, starts=[d0], hide=18)
        S = el.item_stats(b, items, (-6, 6))
        base3 = b['X'][[it[0] for it in items]][..., el.JP]
        new = el.design_priors(base3, S, (-6, 6), form='bma', pi=0.9, mu=0.05, nu=1.0, cw=1.0)
        on, og = b['picks'][d0]
        hidden = np.zeros((b['D'], 288), bool)
        for (d, t) in list(on) + list(og):
            if d < b['D']:
                hidden[d, t + 1:t + 19] = True
        for k, (i, days, evd) in enumerate(items):
            d, t = int(b['d'][i]), int(b['t'][i])
            assert list(days) == list(range(d0, d)) and (d, t) in on
            for col, s in ((0, min(t + 6, 287)), (1, t), (2, min(t + 6, 287))):
                for l in range(0, b['L'], 7):                         # every 7th link
                    n = Fs = R = 0.0; ll0 = 0.0
                    for dd in range(d0, d):
                        c = int(b['wk'][dd]); O = e = X = 0.0
                        for ss in range(max(s - 6, 0), min(s + 6, 287) + 1):
                            if hidden[dd, ss] or not b['Ev'][dd, ss, l]:
                                continue
                            e += 1; O += float(b['Qv'][dd, ss, l]); X += float(b['prior_tr'][c, ss, l])
                        if e > 0:
                            f = O / e; r = X / e; rc = min(max(r, 0.02), 0.98)
                            n += 1; Fs += f; R += r; ll0 += f * np.log(rc) + (1 - f) * np.log(1 - rc)
                    if n > 0:
                        al, be = 0.05, 0.95
                        z = np.log(0.1 / 0.9) + betaln(al + Fs, be + n - Fs) - betaln(al, be) - ll0
                        dev = (1 / (1 + np.exp(-z))) * ((al + Fs) / (al + be + n) - R / n)
                    else:
                        dev = 0.0
                    want = min(max(float(base3[k, l, col]) + dev, 0.0), 1.0)
                    worst = max(worst, abs(want - float(new[k, l, col]))); nchk += 1
print('checked', nchk, 'cells; max abs difference brute force vs engine:', f'{worst:.2e}')
assert worst < 1e-4
