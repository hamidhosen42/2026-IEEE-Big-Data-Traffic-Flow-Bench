"""TRACK ogmodel S3: why is step 6 weak?  Light analysis of cached OOF probabilities (no training).
  T1 per-step error split (shares of the step union): missed growth / dropped persisting / kept cleared / false growth
  T2 growth vs dissipation: cells changing state between T and T+h, recall and precision of the decoded mask
  T3 run-start vs non-run-start windows
  T4 calibration by step (mean predicted vs observed queued share, Platt slope / intercept), by regime (queued / free at T)
  T5 reliability bins for steps 1, 3, 6
  T6 expected (model's own) vs realised step IoU
  T7 per-step recalibration fitted cross-fold on official-like windows (isotonic / Platt, per step or per step x regime),
     decode unchanged (expected-IoU top-m over the 6 x L cells): paired deltas on official / offlike / pseudo
Usage: ogmodel_s3.py <name>=<arm[+arm...]> [...]      e.g.  np=fc_F03_np255 npid=fc_F03_np255+fc_F03_id255"""
import json, sys, warnings
import numpy as np
import pandas as pd
import lab
import ogmodel_lib as og

F = lab.F
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)
BLENDS = dict(a.split('=') for a in sys.argv[1:]) or {'np': 'fc_F03_np255', 'id': 'fc_F03_id255', 'npid': 'fc_F03_np255+fc_F03_id255'}
BLENDS = {k: v.split('+') for k, v in BLENDS.items()}
OUT = og.RES


def load_set(set_name, arms):
    """per panel: rc, P6 (mean of arms), done, keepW [W, L] (keep region under the window's own fold prior)."""
    out = {}
    for panel in lab.PANELS:
        rc = lab.rows(panel, set_name, lab.ONGOING)
        if not len(rc.win):
            continue
        P6 = 0; done = np.ones(len(rc.win), bool)
        for a in arms:
            p, dn = og.load_probs(a, set_name, panel); P6 = P6 + p / len(arms); done &= dn
        fw = rc.win.fold.to_numpy()[rc.wi]
        keep = rc.keep5[np.arange(rc.n), fw]
        keepW = np.zeros((len(rc.win), rc.L), bool); keepW[rc.wi[keep], rc.li[keep]] = True
        out[panel] = dict(rc=rc, P6=P6.astype(np.float32), done=done, keep=keep, keepW=keepW)
    return out


def cell_frame(S):
    """one row per (window, link, step) inside the keep region with an observed label."""
    fr = []
    for panel, v in S.items():
        rc = v['rc']; m = v['keep'] & rc.e & v['done'][rc.wi]
        fr.append(pd.DataFrame(dict(panel=panel, family=panel.rsplit('_', 1)[0], w=rc.wi[m], fold=rc.win.fold.to_numpy()[rc.wi[m]], h=rc.h[m], li=rc.li[m],
                                    p=v['P6'][rc.wi[m], rc.h[m] - 1, rc.li[m]], y=rc.y[m], q=rc.qlast[rc.wi[m], rc.li[m]],
                                    rs=rc.win.runstart.to_numpy()[rc.wi[m]])))
    return pd.concat(fr, ignore_index=True)


def step_counts(S):
    """one row per (window, step): counts of the decoded mask on observed cells."""
    rows = []
    for panel, v in S.items():
        rc = v['rc']
        for i in np.where(v['done'])[0]:
            truth, ev, ql = rc.truth[i], rc.ev[i], rc.qlast[i]
            p6 = v['P6'][i]
            pred = lab.decode(p6.ravel()).reshape(F, -1)
            w = rc.win.iloc[i]
            for h in range(F):
                T = truth[h] & ev[h]; Pp = pred[h] & ev[h]; q = ql; e = ev[h]
                pk = p6[h]
                r = dict(panel=panel, family=panel.rsplit('_', 1)[0], d=int(w.d), t=int(w.t), fold=int(w.fold), runstart=bool(w.runstart), h=h + 1,
                         tp=int((T & Pp).sum()), fn_new=int((T & ~Pp & ~q).sum()), fn_old=int((T & ~Pp & q).sum()), fp_old=int((Pp & ~T & q).sum()),
                         fp_new=int((Pp & ~T & ~q).sum()), n_true=int(T.sum()), n_pred=int(Pp.sum()), n_qT=int((q & e).sum()),
                         growth=int((T & ~q).sum()), diss=int((~truth[h] & q & e).sum()), fn_outkeep=int((T & ~v['keepW'][i]).sum()),
                         exp_inter=float(pk[pred[h] & e].sum()), exp_union=float(Pp.sum() + pk[~pred[h] & e].sum()))
                r['union'] = r['tp'] + r['fn_new'] + r['fn_old'] + r['fp_old'] + r['fp_new']
                rows.append(r)
    return pd.DataFrame(rows)


def fam_mean(df, num, den):
    """micro ratio per panel (sum num / sum den) -> family mean -> mean (same nesting as the competition aggregation)."""
    g = df.groupby(['family', 'panel'])[[*num, den]].sum()
    r = g[list(num)].div(g[den].clip(lower=1), axis=0)
    return r.groupby(level=0).mean().mean()


def platt_fit(p, y):
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogisticRegression(C=1e6, max_iter=200).fit(x, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def platt_apply(ab, p):
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(ab[0] * x + ab[1])))


def iso_fit(p, y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p, y)


def fit_cal(C, kind, by):
    """calibrators per group (h or (h, q)) from the cell frame C."""
    cal = {}
    for key, g in C.groupby(by):
        if len(g) < 200 or g.y.nunique() < 2:
            continue
        cal[key] = platt_fit(g.p.to_numpy(), g.y.to_numpy()) if kind == 'platt' else iso_fit(g.p.to_numpy(), g.y.to_numpy())
    return cal


def apply_cal(cal, kind, by, v, widx):
    """calibrated copy of P6 for windows widx (cells outside the keep region stay 0)."""
    rc = v['rc']; P6 = v['P6'].copy()
    wm = np.zeros(len(rc.win), bool); wm[widx] = True
    m = v['keep'] & wm[rc.wi]
    w_, h_, l_ = rc.wi[m], rc.h[m], rc.li[m]
    p = P6[w_, h_ - 1, l_]; q = rc.qlast[w_, l_]
    out = p.copy()
    for key, c in cal.items():
        k = (h_ == key) if by == ['h'] else ((h_ == key[0]) & (q == key[1]))
        if k.any():
            out[k] = platt_apply(c, p[k]) if kind == 'platt' else c.predict(p[k])
    P6[w_, h_ - 1, l_] = out
    return P6


for name, arms in BLENDS.items():
    print(f'\n######################## {name} = mean({arms}) ########################')
    S = load_set('offlike', arms)
    SC = step_counts(S)
    # ---- T1 / T2
    t1 = []
    for h in range(1, F + 1):
        g = SC[SC.h == h]
        sh = fam_mean(g, ['fn_new', 'fn_old', 'fp_old', 'fp_new', 'tp'], 'union')
        gr = fam_mean(g, ['fn_new'], 'growth'); di = fam_mean(g, ['fp_old'], 'diss')
        pg = g.groupby('panel')[['growth', 'diss', 'n_true', 'n_qT', 'fn_outkeep', 'n_pred', 'union']].sum()
        nw = g.groupby('panel').size()
        t1.append(dict(h=h, step_iou_micro=sh['tp'], missed_growth=sh['fn_new'], dropped_persisting=sh['fn_old'], kept_cleared=sh['fp_old'], false_growth=sh['fp_new'],
                       growth_cells_per_win=(pg.growth / nw).mean(), diss_cells_per_win=(pg.diss / nw).mean(), qT_cells_per_win=(pg.n_qT / nw).mean(),
                       true_cells_per_win=(pg.n_true / nw).mean(), growth_recall=1 - gr['fn_new'], diss_recall=1 - di['fp_old'],
                       pred_growth_precision=1 - (g.fp_new.sum() / max((g.fp_new + g.growth - g.fn_new).sum(), 1)),
                       pred_clear_precision=1 - (g.fn_old.sum() / max((g.fn_old + g.diss - g.fp_old).sum(), 1)),
                       fn_outside_keep_share_of_union=(pg.fn_outkeep / pg.union.clip(lower=1)).mean(),
                       exp_step_iou=fam_mean(g, ['exp_inter'], 'exp_union')['exp_inter']))
    T1 = pd.DataFrame(t1)
    print('\n== T1/T2/T6  offlike OOF, per step (shares of the step union; micro per panel -> family -> mean) ==')
    print(T1.round(4).to_string(index=False)); T1.to_csv(f'{OUT}/s3_step_errors_{name}.csv', index=False)
    tot = SC.groupby(['family', 'panel'])[['fn_new', 'fn_old', 'fp_old', 'fp_new', 'union']].sum()
    e6 = SC[SC.h == 6].groupby(['family', 'panel'])[['fn_new', 'fn_old', 'fp_old', 'fp_new', 'union']].sum()
    err_all = tot[['fn_new', 'fn_old', 'fp_old', 'fp_new']].sum(1); err6 = e6[['fn_new', 'fn_old', 'fp_old', 'fp_new']].sum(1)
    print(f'step-6 share of all error cells: {float((err6 / err_all).groupby(level=0).mean().mean()):.4f} | step-6 errors: growth side (missed growth + false growth) '
          f'{float(((e6.fn_new + e6.fp_new) / err6).groupby(level=0).mean().mean()):.4f}, dissipation side (kept cleared + dropped persisting) '
          f'{float(((e6.fp_old + e6.fn_old) / err6).groupby(level=0).mean().mean()):.4f}')
    # ---- T3 run-start vs not
    W = og.score_blend([(a, 1.0) for a in arms], 'offlike', name)
    t3 = []
    for nm, m in (('run-start', W.runstart.astype(bool)), ('not run-start', ~W.runstart.astype(bool)), ('all', np.ones(len(W), bool))):
        g = W[m.to_numpy() if hasattr(m, 'to_numpy') else m]
        t3.append(dict(subset=nm, n=len(g), iou=lab.aggregate(g), **{f'h{h}': lab.aggregate(g, f's{h}') for h in range(1, F + 1)}, pers_qlast=lab.aggregate(g, 'pers_qlast'),
                       fn_new=lab.aggregate(g, 'fn_new'), fp_old=lab.aggregate(g, 'fp_old'), fn_old=lab.aggregate(g, 'fn_old'), fp_new=lab.aggregate(g, 'fp_new')))
    T3 = pd.DataFrame(t3); print('\n== T3  offlike OOF by window type =='); print(T3.round(4).to_string(index=False)); T3.to_csv(f'{OUT}/s3_runstart_{name}.csv', index=False)
    sc6 = SC[SC.h == 6]
    for nm, m in (('run-start', sc6.runstart), ('not run-start', ~sc6.runstart)):
        g = sc6[m]; sh = fam_mean(g, ['fn_new', 'fn_old', 'fp_old', 'fp_new', 'tp'], 'union')
        print(f'  step 6, {nm}: micro IoU {sh["tp"]:.4f} missed growth {sh["fn_new"]:.4f} kept cleared {sh["fp_old"]:.4f} dropped persisting {sh["fn_old"]:.4f} false growth {sh["fp_new"]:.4f}'
              f' | growth cells/window {g.growth.mean():.2f} dissipating cells/window {g.diss.mean():.2f}')
    # ---- T4 calibration by step
    C = cell_frame(S)
    t4 = []
    for h in range(1, F + 1):
        for qn, m in (('all', np.ones(len(C), bool)), ('queued at T', C.q.to_numpy()), ('free at T', ~C.q.to_numpy())):
            g = C[(C.h == h).to_numpy() & m]
            a, b = platt_fit(g.p.to_numpy(), g.y.to_numpy())
            bins = np.clip((g.p.to_numpy() * 10).astype(int), 0, 9)
            ece = sum(abs(g.y.to_numpy()[bins == j].mean() - g.p.to_numpy()[bins == j].mean()) * (bins == j).mean() for j in range(10) if (bins == j).any())
            pm = g.groupby('panel').apply(lambda x: x.y.mean() - x.p.mean(), include_groups=False)
            t4.append(dict(h=h, regime=qn, cells=len(g), mean_pred=g.p.mean(), observed=g.y.mean(), obs_minus_pred=g.y.mean() - g.p.mean(), platt_slope=a, platt_icpt=b, ece10=ece,
                           panels_under=int((pm > 0).sum()), brier=float(((g.p - g.y) ** 2).mean())))
    T4 = pd.DataFrame(t4); print('\n== T4  calibration by step, offlike OOF cells (keep region, observed labels) ==')
    print(T4.round(4).to_string(index=False)); T4.to_csv(f'{OUT}/s3_calibration_{name}.csv', index=False)
    t5 = []
    for h in (1, 3, 6):
        g = C[C.h == h]; bins = np.clip((g.p.to_numpy() * 10).astype(int), 0, 9)
        for j in range(10):
            k = bins == j
            if k.any():
                t5.append(dict(h=h, bin=f'{j / 10:.1f}-{(j + 1) / 10:.1f}', cells=int(k.sum()), mean_pred=g.p.to_numpy()[k].mean(), observed=g.y.to_numpy()[k].mean()))
    T5 = pd.DataFrame(t5); print('\n== T5  reliability (steps 1, 3, 6) =='); print(T5.round(4).to_string(index=False)); T5.to_csv(f'{OUT}/s3_reliability_{name}.csv', index=False)
    # ---- T7 recalibration, cross-fitted over folds on offlike cells
    base = {s: og.score_blend([(a, 1.0) for a in arms], s, name) for s in og.EVALS}
    SS = {'offlike': S, 'official': load_set('official', arms), 'pseudo': load_set('pseudo', arms)}
    t7 = []
    for kind, by in (('platt', ['h']), ('iso', ['h']), ('platt', ['h', 'q']), ('iso', ['h', 'q'])):
        tag = f'{kind}_{"x".join(by)}'
        res = {s: [] for s in og.EVALS}
        for k in range(4):
            cal = fit_cal(C[C.fold != k], kind, by)
            for s in og.EVALS:
                for panel, v in SS[s].items():
                    widx = np.where(v['done'] & (v['rc'].win.fold.to_numpy() == k))[0]
                    if len(widx):
                        res[s].append(lab.score_probs(v['rc'], apply_cal(cal, kind, by, v, widx), widx, arm=tag, set_name=s))
        for s in og.EVALS:
            df = pd.concat(res[s], ignore_index=True)
            df.to_pickle(f'{OUT}/s3_cal_{name}_{tag}_{s}.pkl')
            t7.append(og.delta_row(base[s], df, tag, s))
    T7 = pd.DataFrame(t7); print('\n== T7  per-step recalibration fitted on offlike OOF cells of the other folds, paired vs uncalibrated ==')
    print(T7.round(4).to_string(index=False)); T7.to_csv(f'{OUT}/s3_recal_{name}.csv', index=False)
    # full-fit calibrators (all folds) for the pseudo-March transfer test
    for kind, by in (('platt', ['h']), ('platt', ['h', 'q'])):
        cal = fit_cal(C, kind, by)
        json.dump({str(k): v for k, v in cal.items()}, open(f'{OUT}/s3_calfull_{name}_{kind}_{"x".join(by)}.json', 'w'), indent=1)
lab.log('DONE')
