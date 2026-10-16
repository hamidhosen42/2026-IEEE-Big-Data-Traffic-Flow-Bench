"""capscen_selbias.py -- how much public-LB selection bias can N submissions produce?
(1) best-of-N: E[max] of N equicorrelated public-noise draws; selection part = sigma * sqrt(1 - rho) * E[max_N iid].
(2) greedy hill-climb: N sequential probes, each a change with true effect ~ N(mu, sd_true) and public difference noise sd_d
    (mix of onset-like 0.004 and ongoing/T1-like 0.0015 LB, the probe noise measured in TRANSFER_ANALYSIS section 1);
    keep if the observed public delta > 0.0001 (our KEEP rule). Bias = public gain kept - true gain kept."""
import numpy as np
rng = np.random.default_rng(0)
print('(1) best-of-N, E[max] iid:', {N: round(float(np.sort(rng.standard_normal((20000, N)), 1)[:, -1].mean()), 2) for N in (10, 48, 70, 100)})
for s in (0.004, 0.006):
    for rho in (0.0, 0.5, 0.8, 0.9):
        em = {N: np.sort(rng.standard_normal((20000, N)), 1)[:, -1].mean() for N in (48, 100)}
        print(f'   sigma {s} rho {rho}: selection bias N=48 {s * np.sqrt(1 - rho) * em[48]:.4f}  N=100 {s * np.sqrt(1 - rho) * em[100]:.4f}')
print('(2) greedy hill-climb (keep if public delta > 0.0001)')
for mu, sdt in ((0.0, 0.0005), (0.0003, 0.001), (-0.0003, 0.001)):
    for N in (48, 70, 100):
        R = 20000; sdd = np.where(rng.random((R, N)) < 0.5, 0.004, 0.0015)
        tr = rng.normal(mu, sdt, (R, N)); ob = tr + rng.normal(0, 1, (R, N)) * sdd; k = ob > 0.0001
        pub = (ob * k).sum(1); tru = (tr * k).sum(1)
        print(f'   true effects ~N({mu}, {sdt}) N={N}: public gain {pub.mean():+.4f}, true gain {tru.mean():+.4f}, bias {np.mean(pub - tru):+.4f} (q10-q90 {np.quantile(pub - tru, .1):+.4f}..{np.quantile(pub - tru, .9):+.4f}), kept {k.sum(1).mean():.0f}')

# (3) window-level hill-climb with bounded IoU (the additive model (2) ignores that the 80 public windows are FIXED: a window
#     already at IoU ~1 cannot gain).  40 onset windows: outcome of any alternative decision ~ 0 w.p. 0.22 else Beta(18, 2) (mean
#     ~0.70 ~ our onset level); 40 ongoing windows: alternative = current + N(0, 0.04), clipped.  A probe is zero-mean in truth:
#     onset probe re-draws 3-7 windows, ongoing probe perturbs 25 windows.  Keep if public S gain > 0.0001.  1 window-IoU = 0.00375 LB.
def draw_on(n): return np.where(rng.random(n) < 0.22, 0.0, rng.beta(18, 2, n))
print('(3) window-level hill-climb, zero true effect (bias = public gain)')
for N in (48, 70, 100):
    out = []
    for rep in range(2000):
        on = draw_on(40); og = np.clip(rng.normal(0.86, 0.12, 40), 0, 1); g = 0.0
        for p in range(N):
            if rng.random() < 0.5:
                ix = rng.choice(40, rng.integers(3, 8), replace=False); new = draw_on(len(ix)); d = (new - on[ix]).sum() * 0.00375
                if d > 0.0001: on[ix] = new; g += d
            else:
                ix = rng.choice(40, 25, replace=False); new = np.clip(og[ix] + rng.normal(0, 0.04, 25), 0, 1); d = (new - og[ix]).sum() * 0.00375
                if d > 0.0001: og[ix] = new; g += d
        out.append(g)
    out = np.array(out); print(f'   N={N}: public-only gain mean {out.mean():+.4f} (q10-q90 {np.quantile(out, .1):+.4f}..{np.quantile(out, .9):+.4f})')
