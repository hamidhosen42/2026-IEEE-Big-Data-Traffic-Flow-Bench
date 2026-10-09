"""pv2 lab runs (design + PRE-REGISTERED rule in pv2_lib.py).  Run through run_heavy.py.
  stage A = MANY-DAY pseudo-March (binding gate 1): pvon_step2 plain family (raw lab dense1500 onset rows + March onset windows of days
            < 15 at weights 10 / 20 / 40, blended prior), evaluated on the 131 onset windows of March days >= 15; arms pv, a1, a2, a3,
            b1, b2 (+ A_s4 reference), seeds 0 and 1.
  stage B = OFFICIAL-40 train analog (binding gate 2): pv recipe inside the train scenario (pvon_step3 design, fold 0, source month =
            31 train days after fold 0 on the emulated masked layer, seed 1), evaluated on the 40 official train onset windows
            (+ offlike fold-0 onset windows, not binding).
Outputs results/pv2/stage{A,B}.pkl (one row per window x arm x seed).
Usage: pv2_run.py A|B"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvon_lib as L
import pv2_lib as V2

STAGE = sys.argv[1] if len(sys.argv) > 1 else 'A'
CUT = 15
SEEDS = (0, 1)
WS = (10.0, 20.0, 40.0)
WB1 = (40.0, 80.0, 160.0)
KC = L.KEEPC
CN = L.COLS_NR
NB = len(CN)
CS = {'pv': NB, 'a1': NB + 3, 'a2': NB + 7, 'a3': NB + 10}
OUT = 'results/pv2'; os.makedirs(OUT, exist_ok=True)
PRIOR3 = ('prior', 'prior_now', 'prior_alldays')
JP = [CN.index(c) for c in PRIOR3]
H, F = lab.H, lab.F


def set_prior(X, prior, d, t, l, dow):
    """prior columns of onset rows (h = 6) from the prior table `prior` [3, 288, L]."""
    cls = (dow[d] >= 5).astype(int); s6 = np.minimum(t + 6, 287)
    X[:, JP[0]] = prior[cls, s6, l]; X[:, JP[1]] = prior[cls, t, l]; X[:, JP[2]] = prior[2, s6, l]
    return X


def rowkeys(d, t, dow):
    return (dow[d] >= 5).astype(int), np.minimum(t + 6, 287), t


def fit_models(Xt, yt, Xm, ym, seed, tag):
    """all models of one seed: A (train only), B_<cs>_w (train + March x w), b1 extra weights, March-only."""
    M = {}
    t0 = time.time()
    M['A'] = L.fit_cat(Xt[:, :NB], yt, None, seed=seed)
    for cs, nc in CS.items():
        for w in WS:
            M[f'{cs}_w{w:g}'] = L.fit_cat(np.concatenate([Xt[:, :nc], Xm[:, :nc]]), np.r_[yt, ym],
                                          np.r_[np.ones(len(Xt)), np.full(len(Xm), w)], seed=seed)
        lab.log(f'{tag} seed {seed} {cs} trained ({time.time() - t0:.0f}s)')
    for w in WB1[1:]:
        M[f'pv_w{w:g}'] = L.fit_cat(np.concatenate([Xt[:, :NB], Xm[:, :NB]]), np.r_[yt, ym],
                                    np.r_[np.ones(len(Xt)), np.full(len(Xm), w)], seed=seed)
    M['monly'] = L.fit_cat(Xm[:, :NB], ym, None, seed=seed)
    lab.log(f'{tag} seed {seed} all models trained ({time.time() - t0:.0f}s)')
    return M


def arm_probs(M, X, Xraw=None):
    pr = {}
    if Xraw is not None:
        pr['A'] = M['A'].predict_proba(Xraw[:, :NB])[:, 1]
    pr['A_s4'] = M['A'].predict_proba(X[:, :NB])[:, 1]
    p = {n: m.predict_proba(X[:, :CS[n.split('_w')[0]]])[:, 1] for n, m in M.items() if '_w' in n}
    for cs in CS:
        pr[cs] = np.mean([p[f'{cs}_w{w:g}'] for w in WS], 0)
    pr['b1'] = np.mean([p[f'pv_w{w:g}'] for w in WB1], 0)
    pr['monly'] = M['monly'].predict_proba(X[:, :NB])[:, 1]
    pr['b2'] = 0.5 * pr['pv'] + 0.5 * pr['monly']
    return pr


T00 = time.time()
if STAGE == 'A':
    Xt, yt, Xm, ym, EV, info = [], [], [], [], [], []
    for panel in L.PANELS:
        t1 = time.time()
        P, Pv = L.val_panel(panel)
        on, og = L.select(Pv)
        D = len(Pv['days'])
        prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
        cQ, cE = L.cum_tables(Pv)
        newT = V2.pv2_events(P['Q'], P['E'])
        dayT = V2.pv2_day_tables(P['V'], P['Q'], P['E'], P['cut'], newT)
        totT = V2.pv2_class_tot(dayT, P['dow'], np.ones(len(P['days']), bool))
        newM = V2.pv2_events(Pv['Q'], Pv['E'])
        cumM = V2.pv2_class_cum(V2.pv2_day_tables(Pv['V'], Pv['Q'], Pv['E'], Pv['cut'], newM), Pv['dow'])
        # ---- March training rows (days < CUT)
        picks = L.spaced_picks(on, 0, CUT)
        R = L.onset_rows(P, Pv, picks, prior_tr); X, y, e, meta = L.matrix(R)
        Xk = L.blend_rows(X, R, meta, Pv['dow'], prior_tr, cQ, cE)[:, KC]
        wi, li = meta[:, 0], meta[:, 1]; d = R['d'][wi]
        c, s6, tT = rowkeys(d, R['t'][wi], Pv['dow'])
        Xk = V2.pv2_columns(Xk, CN, wi, li, c, s6, tT, totT, cum=cumM, jj=d)
        Xm.append(Xk[e]); ym.append(y[e].astype(int))
        # ---- evaluation windows (days >= CUT, official-like multi-start greedy)
        wins = [w for w in L.multistart_onset(on, og, D) if w[0] >= CUT]
        R = L.onset_rows(P, Pv, wins, prior_tr); X, y, e, meta = L.matrix(R)
        Xk = L.blend_rows(X, R, meta, Pv['dow'], prior_tr, cQ, cE)[:, KC]
        wi, li = meta[:, 0], meta[:, 1]; d = R['d'][wi]
        c, s6, tT = rowkeys(d, R['t'][wi], Pv['dow'])
        Xk = V2.pv2_columns(Xk, CN, wi, li, c, s6, tT, totT, cum=cumM, jj=d)
        d_, t_ = R['d'], R['t']
        rc = L.namespace(panel, 'march_on', d_, t_, np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                         np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1), R['qlast'], R['L'])
        EV.append((rc, Xk, X[:, KC], meta))
        # ---- raw train rows (lab dense1500 onset rows: all-train prior, all links of every window)
        rt = lab.rows(panel, 'dense1500', lab.ONSET)
        XA = np.asarray(rt.X)[:, KC]
        wi = rt.wi.astype(int); li = rt.li.astype(int); d = rt.win.d.to_numpy()[wi]
        c, s6, tT = rowkeys(d, rt.win.t.to_numpy()[wi], P['dow'])
        XA = V2.pv2_columns(XA, CN, wi, li, c, s6, tT, totT, day=dayT, dd=d)
        Xt.append(XA[rt.e]); yt.append(rt.y[rt.e].astype(int))
        info.append(dict(panel=panel, march_w=len(picks), march_rows=int(e.sum()) if False else len(Xm[-1]), eval_w=len(wins),
                         train_rows=int(rt.e.sum()), ev_train=int(newT.sum()), ev_march=int(newM.sum()),
                         ev_train_per_day=round(float(newT.sum()) / len(P['days']), 2), ev_march_per_day=round(float(newM.sum()) / D, 2)))
        lab.log(info[-1], f'{time.time() - t1:.0f}s')
        del dayT, cumM, rt, XA; lab.drop_panel(panel); gc.collect()
    print(pd.DataFrame(info).to_string(index=False), flush=True)
    Xt = np.concatenate(Xt); yt = np.concatenate(yt); Xm = np.concatenate(Xm); ym = np.concatenate(ym)
    lab.log('train rows', Xt.shape, 'March rows', Xm.shape, 'pos', round(float(yt.mean()), 4), round(float(ym.mean()), 4))
    res = []
    for seed in SEEDS:
        M = fit_models(Xt, yt, Xm, ym, seed, 'A')
        for rc, Xk, Xraw, meta in EV:
            for arm, p in arm_probs(M, Xk, Xraw).items():
                df = L.score(rc, meta, p, arm); df['seed'] = seed
                res.append(df)
        del M; gc.collect()
    Rr = pd.concat(res, ignore_index=True)
    Rr.to_pickle(f'{OUT}/stageA.pkl')
else:
    K = 0; NSRC = 31; MSEED = 1
    PT = {n: ([], []) for n in ('base', 'src')}
    EVO, info = [], []
    for panel in L.PANELS:
        t1 = time.time()
        P = lab.load_panel(panel); blk = P['block']; dow = P['dow']; D = len(P['days'])
        tgt = np.where(blk == K)[0]
        src = np.arange(tgt[-1] + 1, tgt[-1] + 1 + NSRC)
        old = blk != K; old[src] = False
        prior_old = lab.prior_table(P, old)
        Pm = L.mask_train(P, MSEED)
        order = np.r_[src, tgt]; pos = np.full(D, -1); pos[order] = np.arange(len(order))
        So = dict(Q=Pm['Q'][order], E=Pm['E'][order], dow=dow[order])
        cQ, cE = L.cum_tables(So)
        newS = V2.pv2_events(So['Q'], So['E'])
        cumS = V2.pv2_class_cum(V2.pv2_day_tables(Pm['V'][order], So['Q'], So['E'], P['cut'], newS), So['dow'])
        newT = V2.pv2_events(P['Q'], P['E'])
        dayT = V2.pv2_day_tables(P['V'], P['Q'], P['E'], P['cut'], newT)
        totT = V2.pv2_class_tot(dayT, dow, old)
        # ---- base rows: dense1500 onset rows of the old-scenario days, old-scenario prior, own day left out of the pv2 tables
        rt = lab.rows(panel, 'dense1500', lab.ONSET)
        rd_all = rt.win.d.to_numpy()[rt.wi]
        mb = old[rd_all]
        r = np.where(mb)[0]
        XB = np.asarray(rt.X[r])[:, KC]
        wi = rt.wi[r].astype(int); li = rt.li[r].astype(int); d = rd_all[r]; t = rt.win.t.to_numpy()[wi]
        XB = set_prior(XB, prior_old, d, t, li, dow)
        c, s6, tT = rowkeys(d, t, dow)
        XB = V2.pv2_columns(XB, CN, wi, li, c, s6, tT, totT, day=dayT, dd=d)
        eb = rt.e[r]
        PT['base'][0].append(XB[eb]); PT['base'][1].append(rt.y[r][eb].astype(int))
        # ---- source-month rows: windows re-selected on the masked layer (production onset rule, 72-slot spacing), visible labels
        on_m, _ = L.select(Pm, need_og=False)
        pk = [(int(src[a]), b) for a, b in L.spaced_picks(on_m[src])]
        R = L.onset_rows(P, Pm, pk, prior_old); X, y, e, meta = L.matrix(R)
        jw = pos[R['d']]
        Xk = L.blend_rows(X, R, meta, dow, prior_old, cQ, cE, day=jw)[:, KC]
        wi, li = meta[:, 0], meta[:, 1]
        c, s6, tT = rowkeys(R['d'][wi], R['t'][wi], dow)
        Xk = V2.pv2_columns(Xk, CN, wi, li, c, s6, tT, totT, cum=cumS, jj=jw[wi])
        PT['src'][0].append(Xk[e]); PT['src'][1].append(y[e].astype(int))
        # ---- targets: official-40 (fold 0) and offlike fold-0 onset windows, raw history (lab cache), blended old prior + scenario
        for sname in ('official', 'offlike'):
            ro = lab.rows(panel, sname, lab.ONSET)
            widx = np.where(ro.win.fold.to_numpy() == K)[0]
            if not len(widx):
                continue
            me = np.isin(ro.wi, widx); r = np.where(me)[0]
            remap = np.full(len(ro.win), -1); remap[widx] = np.arange(len(widx))
            od = ro.win.d.to_numpy()[widx]; ot = ro.win.t.to_numpy()[widx]
            meta = np.stack([remap[ro.wi[r]], ro.li[r].astype(int)], 1)
            X = np.asarray(ro.X[r])[:, KC]
            X = set_prior(X, prior_old, od[meta[:, 0]], ot[meta[:, 0]], meta[:, 1], dow)
            Rw = dict(d=od, t=ot)
            Xf = np.zeros((len(X), len(lab.all_cols())), np.float32); Xf[:, KC] = X          # blend_rows indexes the full column set
            jw = pos[od]
            Xk = L.blend_rows(Xf, Rw, meta, dow, prior_old, cQ, cE, day=jw)[:, KC]
            wi, li = meta[:, 0], meta[:, 1]
            c, s6, tT = rowkeys(od[wi], ot[wi], dow)
            Xk = V2.pv2_columns(Xk, CN, wi, li, c, s6, tT, totT, cum=cumS, jj=jw[wi])
            rc = L.namespace(panel, sname, od, ot, ro.truth[widx], ro.ev[widx], ro.qlast[widx], ro.L, fold=K)
            EVO.append((rc, Xk, meta))
        info.append(dict(panel=panel, src=f'{P["days"][src[0]]}..{P["days"][src[-1]]}', base_rows=int(eb.sum()), src_w=len(pk),
                         src_rows=int(e.sum()), ev_src_per_day=round(float(newS[:NSRC].sum()) / NSRC, 2)))
        lab.log(info[-1], f'{time.time() - t1:.0f}s')
        del Pm, dayT, cumS, rt; lab.drop_panel(panel); gc.collect()
    print(pd.DataFrame(info).to_string(index=False), flush=True)
    Xt = np.concatenate(PT['base'][0]); yt = np.concatenate(PT['base'][1])
    Xm = np.concatenate(PT['src'][0]); ym = np.concatenate(PT['src'][1])
    lab.log('base rows', Xt.shape, 'source rows', Xm.shape, 'pos', round(float(yt.mean()), 4), round(float(ym.mean()), 4))
    res = []
    for seed in SEEDS:
        M = fit_models(Xt, yt, Xm, ym, seed, 'B')
        for rc, Xk, meta in EVO:
            for arm, p in arm_probs(M, Xk).items():
                df = L.score(rc, meta, p, arm); df['seed'] = seed
                res.append(df)
        del M; gc.collect()
    Rr = pd.concat(res, ignore_index=True)
    Rr.to_pickle(f'{OUT}/stageB.pkl')
lab.log(f'stage {STAGE} DONE {time.time() - T00:.0f}s')
