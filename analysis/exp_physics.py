"""Task 3 post-processing on top of the stacked model (D12_I405_S train, day-grouped CV preds)."""
import sys, numpy as np
sys.path.insert(0, 'analysis')
exec(open('analysis/exp_task1b.py').read().split("res.append(report('stacked HGB v2")[0])   # rebuild gv, gq
tgt = X['target']
base = report('stacked HGB (reference)', X, st, reg, gv, gq)
L = st.length_km.to_numpy()[None, None, :]
# density implied by observed neighbours in time (log-space interpolation of k = q/v)
with np.errstate(all='ignore'):
    lk_obs = np.log(X['mq'] / np.clip(X['mv'], 1, None))
lk_int = interp_axis(lk_obs, 1)
lk_hat = np.log(gq / np.clip(gv, 1, None))
for w in (0.25, 0.5, 0.75, 1.0):
    for a in (0.0, 0.25, 0.5):
        lr = np.clip(np.where(tgt, lk_hat - np.where(np.isfinite(lk_int), lk_int, lk_hat), 0), -0.5, 0.5) * w
        v2 = gv * np.exp(a * lr); q2 = gq * np.exp(-(1 - a) * lr)
        report(f'density pull w={w} a(speed share)={a}', X, st, reg, v2, q2)
