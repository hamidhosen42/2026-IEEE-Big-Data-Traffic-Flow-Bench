"""Onset under the month shift: is an IN-SPLIT location prior (same month, earlier days only, i.e. data <= T) more informative
than the train prior about WHERE the T+30 queue appears?  Model-free signal check.

Windows: onset pseudo-windows of the masked March month (official-like greedy, multi-start, blank-touching windows dropped; the
same selection as exp_firstcross_pmarch.py), and, as an in-distribution reference, the official-like onset windows of the train
months (lab 'offlike' set).  Labels: visible cells at T+30 (speed <= 0.6 v_free).  Scores per window: ranking AUC of each prior
over the visible links at T+30, and the IoU of the top-m links (m = true queued count, i.e. size oracle) = pure location skill.
Priors: train  = P(queued at slot T+6 +- 2 | link, weekday class) from the 9 train months (all train days);
        insplit = share of the earlier days of the same month (days < d) with a visible queued cell at T+6 +- 2 (Task-1-like 33%
                  random hiding emulated on the train months; March is already masked).
March results may steer private-file decisions or motivate a probe only (decision support)."""
import json, sys, time
import numpy as np, pandas as pd
import lab
from exp_firstcross_prod_snippet import fill_time36

H, F = lab.H, lab.F
rng = np.random.default_rng(0)


# ---- copied from exp_firstcross_pmarch.py (that module runs its evaluation at import time)
def val_panel(panel):
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    V = z['validation_Vm'][o].astype(np.float32); Fq = z['validation_Fm'][o].astype(np.float32); Em = z['validation_Em'][o]
    E = np.isfinite(V); Q = E & (V <= P['cut'][None, None, :])
    Pv = dict(panel=panel, links=P['links'], li=P['li'], cut=P['cut'], vf=P['vf'], cap=P['cap'], lanes=P['lanes'], length=P['length'],
              days=list(days[o]), dow=pd.to_datetime(pd.Series(days[o])).dt.dayofweek.to_numpy(), V=V, Fq=Fq, E=E, Q=Q, Em=Em,
              ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    return P, Pv


def select(Pv):
    """faithful ongoing candidates / production-rule onset candidates on the masked validation month; blank-touching windows dropped."""
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy(); pers = np.full((D, 288), np.nan); stable = og.copy()
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if lab.established(Q[d, t - H:t]):
                Tf = QF[d, t + 1:t + 1 + F]; Mf = MF[d, t + 1:t + 1 + F]
                if not Tf.any():
                    continue
                S = np.repeat(QF[d, t][None], F, 0)
                u = ((Tf | S) & Mf).sum(); i = (Tf & S & Mf).sum()
                pers[d, t] = 1.0 if u == 0 else i / u
                og[d, t] = pers[d, t] <= 0.9; stable[d, t] = pers[d, t] > 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og, stable, pers, blank




def window_scores(Q, E, d, t, prior_tr, insplit, n_days):
    tr = Q[d, t + 6]; ev = E[d, t + 6]
    if not tr[ev].any() or ev.sum() < 5:
        return None
    out = dict(n_days=n_days, m=int(tr[ev].sum()))
    for nm, sc in (('train', prior_tr), ('insplit', insplit), ('mix', 0.5 * prior_tr + 0.5 * insplit)):
        s = sc[ev]; y = tr[ev]
        pos, neg = s[y], s[~y]
        auc = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])) if len(neg) else np.nan
        m = int(y.sum()); top = np.argsort(-s, kind='stable')[:m]; pred = np.zeros(len(s), bool); pred[top] = True
        out[f'auc_{nm}'] = float(auc); out[f'iou_{nm}'] = float((pred & y).sum() / max((pred | y).sum(), 1))
    return out


def insplit_prior(Qm, d, t, hide=None):
    if d == 0:
        return np.zeros(Qm.shape[2], np.float32), 0
    lo, hi = max(t + 6 - 2, 0), min(t + 6 + 3, 288)
    q = Qm[:d, lo:hi]
    if hide is not None:
        q = q & hide[:d, lo:hi]
    return q.any(1).mean(0).astype(np.float32), d


rows = []
for panel in lab.PANELS:
    t0 = time.time()
    P, Pv = val_panel(panel)
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))          # [3, 288, L]: weekday, weekend, all
    # ---- March onset pseudo-windows
    on, og, stable, pers, blank = select(Pv)
    g_on, g_og = lab._glist(on), lab._glist(og); s = set()
    for d0 in range(len(Pv['days'])):
        a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(a)
    for d, t in sorted(s):
        cls = int(Pv['dow'][d] >= 5)
        ptr = prior[cls, min(t + 6, 287)]
        ins, nd = insplit_prior(Pv['Q'], d, t)
        r = window_scores(Pv['Q'], Pv['E'], d, t, ptr, ins, nd)
        if r:
            rows.append(dict(panel=panel, set='march', d=d, t=t, wk=cls, **r))
    # ---- train official-like onset windows (in-split = same calendar month, earlier days; 33% hiding emulated)
    W = lab.window_set(panel, 'offlike'); W = W[W.cond == 'queue_onset']
    days = pd.to_datetime(pd.Series(P['days'])); month = days.dt.strftime('%Y-%m').to_numpy()
    hide = rng.random(P['Q'].shape) > 0.33
    for d, t in zip(W.d.to_numpy(), W.t.to_numpy()):
        m0 = np.where(month == month[d])[0][0]
        cls = int(P['dow'][d] >= 5)
        # train prior without the window's own month (leave-month-out) to avoid self-information
        keep = month != month[d]
        ptr = lab.prior_table(P, keep)[cls, min(t + 6, 287)] if False else prior[cls, min(t + 6, 287)]
        ins, nd = insplit_prior(P['Q'][m0:], d - m0, t, hide[m0:])
        r = window_scores(P['Q'], P['E'], d, t, ptr, ins, nd)
        if r:
            rows.append(dict(panel=panel, set='train', d=d, t=t, wk=cls, **r))
    lab.log(panel, f'{time.time() - t0:.0f}s', 'march windows', sum(1 for x in rows if x['panel'] == panel and x['set'] == 'march'))
    lab.drop_panel(panel)

df = pd.DataFrame(rows)
df.to_csv('results/onset_insplit_signal.csv', index=False)
pd.set_option('display.width', 220)
df['nd_bin'] = pd.cut(df.n_days, [-1, 0, 2, 5, 10, 40], labels=['0', '1-2', '3-5', '6-10', '11+'])
for st in ('march', 'train'):
    x = df[df.set == st]
    print(f'\n===== {st}: {len(x)} onset windows =====')
    print(x.groupby('nd_bin', observed=True)[['auc_train', 'auc_insplit', 'auc_mix', 'iou_train', 'iou_insplit', 'iou_mix']].mean().round(3).to_string())
    print(x.groupby('nd_bin', observed=True).size().to_string())
    print('per panel (n_days >= 1):')
    print(x[x.n_days >= 1].groupby('panel')[['iou_train', 'iou_insplit', 'iou_mix']].mean().round(3).to_string())

# ================= early-month first-crossing onset candidates (official windows sit in days 0-5) =================
rows2 = []
for panel in lab.PANELS:
    P, Pv = val_panel(panel)
    prior = lab.prior_table(P, np.ones(len(P['days']), bool))
    on, og, stable, pers, blank = select(Pv)
    fc_on = on.copy(); fc_on[:, 1:] &= ~on[:, :-1]                       # first-crossing: queue first appears at T+30
    for d, t in zip(*np.where(fc_on)):
        if d > 6:
            continue
        cls = int(Pv['dow'][d] >= 5)
        ptr = prior[cls, min(t + 6, 287)]; ins, nd = insplit_prior(Pv['Q'], d, t)
        for k in (1, 2, 4, 8):
            a_ = nd / (nd + k)
            r = window_scores(Pv['Q'], Pv['E'], d, t, ptr, (1 - a_) * ptr + a_ * ins, nd)
            if r:
                rows2.append(dict(panel=panel, set='march', k=k, d=d, t=t, **r))
    days = pd.to_datetime(pd.Series(P['days'])); month = days.dt.strftime('%Y-%m').to_numpy()
    hide = rng.random(P['Q'].shape) > 0.33
    on_t = np.zeros_like(fc_on[:0])
    Qt, Et = P['Q'], P['E']
    D = Qt.shape[0]
    est = np.zeros((D, 288), bool); hq = np.zeros((D, 288), bool)
    for d in range(D):
        mi = np.where(month == month[d])[0][0]
        if d - mi > 6:
            continue
        for t in range(H, 288 - F):
            if Et[d, t - H:t].mean() < 0.7 or lab.established(Qt[d, t - H:t]):
                continue
            hq[d, t] = Qt[d, t + 1:t + 1 + F].any()
        fcx = hq[d].copy(); fcx[1:] &= ~hq[d][:-1]
        for t in np.where(fcx)[0]:
            cls = int(P['dow'][d] >= 5)
            ptr = prior[cls, min(t + 6, 287)]; ins, nd = insplit_prior(Qt[mi:], d - mi, t, hide[mi:])
            for k in (1, 2, 4, 8):
                a_ = nd / (nd + k)
                r = window_scores(Qt, Et, d, t, ptr, (1 - a_) * ptr + a_ * ins, nd)
                if r:
                    rows2.append(dict(panel=panel, set='train', k=k, d=d, t=t, **r))
    lab.drop_panel(panel)
d2 = pd.DataFrame(rows2); d2.to_csv('results/onset_insplit_early.csv', index=False)
print('\n\n######## EARLY-MONTH first-crossing onset candidates (days 0-6); insplit column = blended prior (1-a) train + a insplit, a = n/(n+k) ########')
for st in ('march', 'train'):
    x = d2[d2.set == st]
    print(f'\n== {st}: windows per k = {len(x) // 4}')
    t_ = x.groupby(['k', 'n_days'])[['iou_train', 'iou_insplit', 'auc_train', 'auc_insplit']].mean().round(3)
    print(t_.to_string())
    g = x.groupby('k')[['iou_train', 'iou_insplit']].mean(); g['delta'] = g.iou_insplit - g.iou_train
    print('pooled days 0-6:'); print(g.round(4).to_string())
    pp = x[x.k == 2].groupby('panel')[['iou_train', 'iou_insplit']].mean(); pp['delta'] = pp.iou_insplit - pp.iou_train
    print('per panel (k=2):'); print(pp.round(3).to_string())
