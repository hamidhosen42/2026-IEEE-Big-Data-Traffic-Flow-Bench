"""DIAGNOSTIC - NOT FOR SUBMISSION (analysis/LEAK_DIAG_2026-10-29.md).  Uses post-T data on purpose (see leakdiag_lib).

The same leak on PSEUDO-MARCH: the masked validation month (new scenario seed), where the production-like models sit near
their LB level (onset s4 ~0.79, LB onset ~0.75-0.77).  Windows, features and labels as the existing lab scripts:
  onset    exp_onset_splitprior.py (NORAMP CatBoost trained on all train days, production s4 same-scenario prior blend),
           pseudo-onset windows of the masked March month, labels = visible masked-layer cells
  ongoing  ogfeat pseudo-March 'offlike_m' windows (cache/ogfeat/pm_*), probabilities of the F03 np base model
           (cache/ogfeat/arms/og_np_base/pm_f0.pkl)
Leak: March on-ramp flows at T+5..T+30 (profile from the March month itself, ramp -> link tables learned on train) and the
March masked layer at T+95..T+105.  Stackers / rules are trained on the TRAIN offlike cells (leakdiag_eval design) and applied
to March unchanged.
Run:  .venv/bin/python analysis/t2lab/run_heavy.py -- .venv/bin/python analysis/t2lab/leakdiag_pmarch.py
"""
import gc, os, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

import lab
import leakdiag_lib as ld
import leakdiag_eval as le
from exp_firstcross_prod_snippet import fill_time36

H, F, ONSET, ONGOING = lab.H, lab.F, lab.ONSET, lab.ONGOING
RES = le.RES
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)


# ---------------------------------------------------------------- copied from exp_onset_splitprior.py (pseudo-March onset)
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
    V, E, Q, Em = Pv['V'], Pv['E'], Pv['Q'], Pv['Em']; D, _, L = V.shape
    VF = fill_time36(V); MF = np.isfinite(VF); QF = MF & (VF <= Pv['cut'][None, None, :])
    blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    og = np.zeros((D, 288), bool); on = og.copy()
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
                og[d, t] = (1.0 if u == 0 else i / u) <= 0.9
            else:
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on, og


def onset_rows(P, Pv, wins, prior):
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([ONSET] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


def cum_tables(Pv, smooth):
    Q = Pv['Q'].astype(np.float32); E = Pv['E'].astype(np.float32)
    if smooth:
        def box(a):
            c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
            return c[:, 5:] - c[:, :-5]
        Q, E = box(Q), box(E)
    D = Q.shape[0]; wk = Pv['dow'] >= 5
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
    return cQ, cE


# ---------------------------------------------------------------- leak inputs of the March month
def march_leak(P, Pv):
    R, ok, flag, A_m, prof = ld.ramp_cubes(Pv['ON'], Pv['dow'])                 # profile from the March month itself
    z = np.load(f'{le.CELLS}/maps_{P["panel"]}.npz'); A_t = z['A']; L = len(P['links'])
    pos = {int(a): i for i, a in enumerate(A_t)}
    Mp = np.zeros((len(A_m), L), np.float32); Rc = Mp.copy(); Mn = np.ones_like(Mp)
    for i, a in enumerate(A_m):                                                  # train-learned rows; unseen ramps carry no mapping
        if int(a) in pos:
            Mp[i] = z['Mpos4'][pos[int(a)]]; Rc[i] = z['Rc4'][pos[int(a)]]; Mn[i] = z['Mneg4'][pos[int(a)]]
    return R, ok, flag, A_m, (Mp, Rc, Mn), len(set(map(int, A_m)) & set(pos)) / max(len(A_m), 1)


def cells_item(panel, d, t, P6, truth, ev, qlast, P, Pv, leak):
    R, ok, flag, A_m, maps, _ = leak; L = len(P['links'])
    FR = ld.ramp_feats(d, t, flag, ok, R, A_m, *maps, L)
    FPB = ld.pb_feats(d, t, Pv['V'], Pv['E'], P['cut'], P['vf'], L)            # March masked layer: already at release visibility
    win = pd.DataFrame(dict(d=d, t=t, fold=0))
    return dict(panel=panel, win=win, d=d, t=t, fold=np.zeros(len(d), int), truth=truth, ev=ev, qlast=qlast, P6=P6.astype(np.float32),
                FR=FR, FPB=FPB, A=A_m)


def build_march():
    cols = lab.all_cols(); KEEPI = [i for i, c in enumerate(cols) if not c.startswith('ramp_')]
    JP = [cols.index(c) for c in PRIOR3]
    cat = CatBoostClassifier(); cat.load_model(f'{lab.ROOT}/analysis/t2lab/results/splitprior/cat_onset_noramp.cbm')
    pm = pd.read_pickle(f'{lab.CACHE}/ogfeat/arms/og_np_base/pm_f0.pkl')
    on_items, on_base_items, og_items, info = [], [], [], []
    for panel in lab.PANELS:
        t1 = time.time()
        P, Pv = val_panel(panel)
        leak = march_leak(P, Pv)
        # ---- onset
        on, og = select(Pv); D = len(Pv['days'])
        g_on, g_og = lab._glist(on), lab._glist(og); s = set()
        for d0 in range(D):
            a, _ = lab.greedy(g_on, g_og, 'interleaved', start_day=d0); s.update(a)
        wins = sorted(s)
        if wins:
            prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
            Rr = onset_rows(P, Pv, [(a, b, ONSET) for a, b in wins], prior_tr)
            X, y, e, meta, call = lab.to_matrix(Rr, ONSET, None, None)
            assert list(call) == cols
            wi, li = meta[:, 0], meta[:, 1]
            d, t = Rr['d'], Rr['t']; cls = (Pv['dow'][d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
            c_r, d_r, s6_r, t_r = cls[wi], d[wi], s6[wi], t[wi]
            trv = [prior_tr[c_r, s6_r, li], prior_tr[c_r, t_r, li], prior_tr[2, s6_r, li]]
            cQ, cE = cum_tables(Pv, True)
            Xv = X.copy()
            for j, (cc, ss) in enumerate(((c_r, s6_r), (c_r, t_r), (np.full_like(c_r, 2), s6_r))):
                q = cQ[cc, d_r, ss, li]; ee = cE[cc, d_r, ss, li]
                sp = np.where(ee > 0, q / np.maximum(ee, 1e-9), 0.0); n = ee / 5.0
                a_ = n / (n + 4.0)
                Xv[:, JP[j]] = (1 - a_) * trv[j] + a_ * sp
            del cQ, cE
            truth = np.stack([Pv['Q'][d, t + h] for h in range(1, F + 1)], 1); ev = np.stack([Pv['E'][d, t + h] for h in range(1, F + 1)], 1)
            for Xm, dst in ((Xv, on_items), (X, on_base_items)):
                P6 = np.zeros((Rr['W'], F, Rr['L']), np.float32); P6[wi, F - 1, li] = cat.predict_proba(Xm[:, KEEPI])[:, 1]
                dst.append(cells_item(panel, d, t, P6, truth, ev, Rr['qlast'], P, Pv, leak))
            del X, Xv, Rr
        # ---- ongoing
        f = f'{lab.CACHE}/ogfeat/pm_{panel}__offlike_m.npz'
        z = np.load(f); win = pd.read_pickle(f.replace('.npz', '.pkl'))
        P6 = pm[(panel, 'offlike_m')].astype(np.float32)
        assert P6.shape[0] == len(win) == z['truth'].shape[0], panel
        if len(win):
            og_items.append(cells_item(panel, win.d.to_numpy(), win.t.to_numpy(), P6, z['truth'], z['ev'], z['qlast'], P, Pv, leak))
        info.append(dict(panel=panel, onset_windows=len(wins), ongoing_windows=len(win), march_ramp_links=len(leak[3]),
                         share_with_train_table=round(leak[5], 2), sec=round(time.time() - t1)))
        lab.log(info[-1])
        del P, Pv, leak; lab.drop_panel(panel); gc.collect()
    return on_items, on_base_items, og_items, pd.DataFrame(info)


def final_stackers(cond):
    """one stacker per variant on ALL train offlike cells (all folds) of the condition."""
    data = le.load('offlike', cond); models = {}
    for var, (ramp, pb) in le.VARIANTS.items():
        D = [le.design(it, cond, ramp, pb) for it in data]
        X = np.concatenate([d_[0] for d_ in D]); y = np.concatenate([d_[1] for d_ in D]); e = np.concatenate([d_[2] for d_ in D])
        models[var] = le.fit_stacker(X[e], y[e], D[0][4])
        del D, X, y, e; gc.collect()
    return models


def evaluate(items, cond, set_name, models, base_label='base'):
    res = []
    for it in items:
        res.append(le.score(it, it['P6'], cond, set_name, base_label))
        if models is None:
            continue
        for nm, P6 in le.rule_variants(it, cond).items():
            res.append(le.score(it, P6, cond, set_name, nm))
        for var, (ramp, pb) in le.VARIANTS.items():
            X, _, _, _, _ = le.design(it, cond, ramp, pb)
            res.append(le.score(it, le.put(it, cond, le.apply_stacker(models[var], X)), cond, set_name, var))
    return pd.concat(res, ignore_index=True)


def main():
    t0 = time.time()
    on_items, on_base_items, og_items, info = build_march()
    print(info.to_string(index=False))
    sums = []
    for cond, items in ((ONSET, on_items), (ONGOING, og_items)):
        models = final_stackers(cond)
        df = evaluate(items, cond, 'pmarch', models)
        if cond == ONSET:
            df = pd.concat([df, evaluate(on_base_items, cond, 'pmarch', None, base_label='base_trainprior')], ignore_index=True)
        df.to_pickle(f'{RES}/pmarch_{cond}.pkl')
        S = le.summarize(df, cond); sums.append(S)
        print(f'\n== pseudo-March {cond}: IoU and delta vs base (production-like) ==')
        print(S.round(4).to_string(index=False))
        if cond == ONSET:
            R = le.reveal_onset({'pmarch': items}, 'pmarch'); R.to_csv(f'{RES}/reveal_onset_pmarch.csv', index=False)
            print(f'\n-- pseudo-March onset location reveal ({len(R)} windows) --')
            print('share of windows:', R[['on_ramp_link', 'ramp_usable', 'self_flag', 'flag_within2', 'map_reveal', 'pb_reveal']].mean().round(3).to_dict())
            print('mean share of true T+30 cells: own ramp switched off', round(R.cells_self.mean(), 3), '| within +-2', round(R.cells_within2.mean(), 3),
                  '| post-buffer queued', round(R.cells_pb.mean(), 3), '| flag precision', round(R.n_flag_true.sum() / max(R.n_flag_ev.sum(), 1), 3))
    S = pd.concat(sums, ignore_index=True); S.to_csv(f'{RES}/pmarch_summary.csv', index=False)
    rows = []
    for arm in dict.fromkeys(S.arm):
        a = S[S.arm == arm].set_index('cond')
        if ONSET in a.index and ONGOING in a.index:
            don, dog = a.loc[ONSET, 'd_base'], a.loc[ONGOING, 'd_base']
            rows.append(dict(arm=arm, iou_onset=a.loc[ONSET, 'iou'], iou_ongoing=a.loc[ONGOING, 'iou'], d_onset=don, d_ongoing=dog,
                             S_total_gain=0.15 * don + 0.15 * dog))
    G = pd.DataFrame(rows); G.to_csv(f'{RES}/pmarch_s_total_gain.csv', index=False)
    print('\n== pseudo-March: implied S_total gain = 0.15 x dIoU_onset + 0.15 x dIoU_ongoing ==')
    print(G.round(4).to_string(index=False))
    lab.log(f'pmarch done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
