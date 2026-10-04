"""Seed-shift size on the real Task 2 panels: train months (one scenario seed) vs the masked March month (new seed).
Uses March observations of the whole month, so results may only steer PRIVATE-file decisions (all March data precede April origins)."""
import glob, numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob('data/t2local/t2data_*.npz')):
    panel = f.split('t2data_')[1][:-4]
    z = np.load(f)
    cut = 0.6 * z['vf'].astype(np.float32)
    V = z['train_V'].astype(np.float32); E = np.isfinite(V); Q = E & (V <= cut)
    days = pd.to_datetime(pd.Series(z['train_days'])); wk = (days.dt.dayofweek >= 5).to_numpy()
    Vm = z['validation_Vm'].astype(np.float32); Em = np.isfinite(Vm); Qm = Em & (Vm <= cut)
    mdays = pd.to_datetime(pd.Series(z['validation_days'])); mwk = (mdays.dt.dayofweek >= 5).to_numpy()
    month = days.dt.strftime('%Y-%m').to_numpy()
    out = dict(panel=panel, L=V.shape[2])
    # queued share overall
    out['q_share_train'] = Q.sum() / E.sum(); out['q_share_march'] = Qm.sum() / Em.sum()
    # per-month train queued share spread
    ms = [Q[month == m].sum() / E[month == m].sum() for m in sorted(set(month))]
    out['q_share_train_month_min'] = min(ms); out['q_share_train_month_max'] = max(ms)
    res = {}
    for cls, name in ((False, 'wd'), (True, 'we')):
        with np.errstate(all='ignore'):
            pt = Q[wk == cls].sum(0) / np.maximum(E[wk == cls].sum(0), 1)          # [288, L] train prior
            pm_n = Em[mwk == cls].sum(0); pm = Qm[mwk == cls].sum(0) / np.maximum(pm_n, 1)
        # hour-level aggregation to reduce the noise of a single month: [24, L]
        pth = pt.reshape(24, 12, -1).mean(1); pmh = pm.reshape(24, 12, -1).mean(1)
        ok = pm_n.reshape(24, 12, -1).sum(1) > 0
        out[f'corr_hourlink_{name}'] = np.corrcoef(pth[ok], pmh[ok])[0, 1]
        # link-level share
        lt = Q[wk == cls].sum((0, 1)) / np.maximum(E[wk == cls].sum((0, 1)), 1); lm = Qm[mwk == cls].sum((0, 1)) / np.maximum(Em[mwk == cls].sum((0, 1)), 1)
        out[f'corr_link_{name}'] = np.corrcoef(lt, lm)[0, 1]
        # March queued cells where the train prior (same class, slot, link) is low
        qm_cells = Qm[mwk == cls]                                               # [d, 288, L]
        pr = np.broadcast_to(pt[None], qm_cells.shape)
        n = qm_cells.sum()
        out[f'march_q_prior_lt05_{name}'] = (qm_cells & (pr < 0.05)).sum() / max(n, 1)
        out[f'march_q_prior_lt20_{name}'] = (qm_cells & (pr < 0.20)).sum() / max(n, 1)
        # train-recurrent cells (prior >= 0.8) that are free in March (observed)
        em_cells = Em[mwk == cls]; hi = np.broadcast_to(pt[None] >= 0.8, em_cells.shape)
        out[f'prior_ge80_free_in_march_{name}'] = (em_cells & hi & ~qm_cells).sum() / max((em_cells & hi).sum(), 1)
        # same check inside train, leave-one-month-out (how reliable is the prior within the train seed?)
        vals = []
        for m in sorted(set(month)):
            a = (month == m) & (wk == cls); b = (month != m) & (wk == cls)
            if a.sum() == 0: continue
            with np.errstate(all='ignore'):
                pb = Q[b].sum(0) / np.maximum(E[b].sum(0), 1)
            qa = Q[a]; vals.append((qa & (np.broadcast_to(pb[None], qa.shape) < 0.05)).sum() / max(qa.sum(), 1))
        out[f'train_q_prior_lt05_{name}'] = float(np.mean(vals))
    # onset timing: first queued slot per (day, link) in the AM (slots 48..144 = 04:00-12:00), weekdays
    def first_slot(Qx, Ex, sel):
        q = Qx[sel][:, 48:144]; has = q.any(1); fs = np.where(has, q.argmax(1), -1) + 48
        return fs, has
    fs_t, has_t = first_slot(Q, E, ~wk); fs_m, has_m = first_slot(Qm, Em, ~mwk)
    freq_t = has_t.mean(0); freq_m = has_m.mean(0)
    rec = freq_t >= 0.6                                         # links with a recurrent weekday AM queue in train
    if rec.any():
        med_t = np.array([np.median(fs_t[has_t[:, j], j]) for j in np.where(rec)[0]])
        iqr_t = np.array([np.subtract(*np.percentile(fs_t[has_t[:, j], j], [75, 25])) for j in np.where(rec)[0]])
        med_m = np.array([np.median(fs_m[has_m[:, j], j]) if has_m[:, j].any() else np.nan for j in np.where(rec)[0]])
        out['am_recurrent_links'] = int(rec.sum())
        out['am_train_onset_iqr_slots'] = float(np.median(iqr_t))
        out['am_march_minus_train_onset_slots_med'] = float(np.nanmedian(med_m - med_t))
        out['am_march_minus_train_onset_slots_absmed'] = float(np.nanmedian(np.abs(med_m - med_t)))
        out['am_recurrent_links_active_in_march'] = float(np.mean(freq_m[rec] >= 0.3))
    out['am_links_new_in_march'] = int(((freq_t < 0.1) & (freq_m >= 0.5)).sum())
    rows.append(out); print(panel, 'done', flush=True)
df = pd.DataFrame(rows).set_index('panel')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.T.round(3).to_string())
df.to_csv('analysis/seedshift/shift_train_vs_march.csv')
