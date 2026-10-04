"""pvon step 3 (+ fixes of step 4): RAW-DEPLOYMENT HARM TEST inside the train split (no scenario shift: mechanics only).

For a target fold k (held-out day block): source month = the 31 train days right before fold k (k = 0: right after it), i.e. the
train-split stand-in of "March before April".  Base = production onset rows (lab dense1500, RAW histories) of the other folds
WITHOUT the source month; fold-safe prior (train days outside fold k) in every row.
Arms (CatBoost onset as production, no ramp columns; weights 10 / 20 / 40 and their bag, as kaggle_jobs/t2_onset_pv.py):
  i        base only
  ii       base + source-month onset windows, RAW features                       -> (ii) - (i)   = over-weighting one month
  iii      base + the same windows, MASKED histories + masked slot T, labels only on the emulated-visible cells
  iv       base + source-month windows RE-SELECTED on the emulated masked layer (exactly how the March rows are built)
                                                                                -> (iii/iv) - (ii) = cost of masked source rows
  i_mS     base rows all masked (windows re-selected on the masked cube)        (matched regime for the masked targets)
  i_mix    raw base rows (half the train weight) + masked base rows (half)
  iv_mix   i_mix + masked source month ;  iv_mS = i_mS + masked source month
Targets of fold k:
  R   official-like onset windows with RAW histories (lab 'offlike', production slot-T simulation)        = deployment regime
  M   pseudo windows selected on the emulated masked layer, masked features, visible labels              = the pseudo-March regime
Test-time masking (TTM, R only): the target history is re-masked K times (regimes 0.5 / 0.7 / 0.8, slot T untouched) and the
probabilities are averaged.
Usage: pvon_step3.py FOLDS(comma) MASKSEED [K_TTM]"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvon_lib as L

H, F = lab.H, lab.F
FOLDS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
KTTM = int(sys.argv[3]) if len(sys.argv) > 3 else 4
WS = (10.0, 20.0, 40.0)
KC = L.KEEPC
NSRC = 31
TAG = f'f{"-".join(map(str, FOLDS))}_s{SEED}' + L.STAG
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def sub_state(S, days):
    return dict(S, V=S['V'][days], Fq=S['Fq'][days], E=S['E'][days], Q=S['Q'][days], Em=S['Em'][days], dow=S['dow'][days],
                ON=S['ON'][days], OFF=S['OFF'][days], days=[S['days'][i] for i in days])


res = []
for k in FOLDS:
    parts = {n: ([], []) for n in ('base', 'src_raw', 'src_F', 'src_S', 'base_S')}
    EVR, EVM, EVT = [], [], []
    info = []
    for panel in L.PANELS:
        t1 = time.time()
        P = lab.load_panel(panel); blk = P['block']; D = len(P['days'])
        tgt = np.where(blk == k)[0]
        src = np.arange(tgt[0] - NSRC, tgt[0]) if k >= 1 else np.arange(tgt[-1] + 1, tgt[-1] + 1 + NSRC)
        srcset = set(src.tolist())
        prior_k = lab.prior_table(P, blk != k)
        rt = lab.rows(panel, 'dense1500', lab.ONSET)
        wd = rt.win.d.to_numpy(); wt_ = rt.win.t.to_numpy()
        row_src = np.isin(wd, src)[rt.wi]
        mb = rt.e & (rt.fold != k) & ~row_src
        parts['base'][0].append(rt.matrix(mb, L.COLS, k)[:, KC]); parts['base'][1].append(rt.y[mb].astype(int))
        ms = rt.e & row_src
        parts['src_raw'][0].append(rt.matrix(ms, L.COLS, k)[:, KC]); parts['src_raw'][1].append(rt.y[ms].astype(int))
        Pm = L.mask_train(P, SEED)
        # iii: the production windows of the source month, masked features, visible labels
        wsrc = [(int(a), int(b)) for a, b in zip(wd, wt_) if a in srcset]
        R = L.onset_rows(P, Pm, wsrc, prior_k); X, y, e, meta = L.matrix(R)
        parts['src_F'][0].append(X[e][:, KC]); parts['src_F'][1].append(y[e].astype(int))
        # iv: source-month windows re-selected on the masked layer (scan starts at the first source day, as the kernel in March)
        on_m, _ = L.select(Pm, need_og=False)
        pk = [(int(src[a]), b) for a, b in L.spaced_picks(on_m[src])]
        R = L.onset_rows(P, Pm, pk, prior_k); X, y, e, meta = L.matrix(R)
        parts['src_S'][0].append(X[e][:, KC]); parts['src_S'][1].append(y[e].astype(int))
        # masked base rows: windows re-selected on the masked cube, days outside fold k and outside the source month
        pkb = [w for w in L.spaced_picks(on_m) if blk[w[0]] != k and w[0] not in srcset]
        R = L.onset_rows(P, Pm, pkb, prior_k); X, y, e, meta = L.matrix(R)
        parts['base_S'][0].append(X[e][:, KC]); parts['base_S'][1].append(y[e].astype(int))
        # ---- target R: official-like raw windows of fold k
        ro = lab.rows(panel, 'offlike', lab.ONSET)
        widx = np.where(ro.win.fold.to_numpy() == k)[0]
        me = np.isin(ro.wi, widx)
        remap = {w: i for i, w in enumerate(widx)}
        metaR = np.stack([np.array([remap[w] for w in ro.wi[me]]), ro.li[me].astype(int)], 1)
        od = ro.win.d.to_numpy(); ot = ro.win.t.to_numpy()
        rcR = L.namespace(panel, f'R', od[widx], ot[widx], ro.truth[widx], ro.ev[widx], ro.qlast[widx], ro.L, fold=k)
        XR = ro.matrix(me, L.COLS, k)[:, KC]
        EVR.append((rcR, XR, metaR))
        # ---- test-time masking of the raw targets: history masked K times, slot T kept as in the raw rows
        if KTTM:
            VT = L.prod_VT(P, od, ot, len(od))[widx]                      # the production slot-T simulation of the cached rows
            wl = list(zip(od[widx].tolist(), ot[widx].tolist()))
            R0 = L.onset_rows(P, L.raw_state(P), wl, prior_k, VT=VT); X0, _, _, m0 = L.matrix(R0)
            assert np.array_equal(m0[:, :2], metaR) and np.array_equal(X0[:, KC], XR, equal_nan=True), 'raw target rebuild != cache'
            xs = []
            for j in range(KTTM):
                Pj = L.mask_train(P, 100 + 10 * SEED + j)
                Rj = L.onset_rows(P, Pj, wl, prior_k, VT=VT); Xj, _, _, mj = L.matrix(Rj); xs.append(Xj[:, KC]); del Pj
            EVT.append(xs)
        # ---- target M: pseudo windows of fold k selected on the masked layer
        sub = sub_state(Pm, tgt)
        on_t, og_t = L.select(sub)
        wm = [(int(tgt[a]), b) for a, b in L.multistart_onset(on_t, og_t, len(tgt))]
        nM = len(wm)
        if wm:
            R = L.onset_rows(P, Pm, wm, prior_k); X, y, e, metaM = L.matrix(R)
            d_, t_ = R['d'], R['t']
            rcM = L.namespace(panel, 'M', d_, t_, np.stack([Pm['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                              np.stack([Pm['E'][d_, t_ + h] for h in range(1, F + 1)], 1), R['qlast'], R['L'], fold=k)
            # diagnostic: is the queue already established in the RAW history (hidden by the mask)?
            rcM.hidden_est = np.array([bool(lab.established(P['Q'][a, b - H:b])) for a, b in zip(d_, t_)])
            rcM.share = Pm['share'][d_]
            EVM.append((rcM, X[:, KC], metaM))
        info.append(dict(fold=k, panel=panel, src=f'{P["days"][src[0]]}..{P["days"][src[-1]]}', base_rows=int(mb.sum()), src_w=len(wsrc),
                         src_rows_raw=int(ms.sum()), srcF_rows=len(parts['src_F'][0][-1]), srcS_w=len(pk), srcS_rows=len(parts['src_S'][0][-1]),
                         baseS_w=len(pkb), tgtR_w=len(widx), tgtM_w=nM))
        lab.log(info[-1], f'{time.time() - t1:.0f}s')
        del Pm; lab.drop_panel(panel); gc.collect()
    print(pd.DataFrame(info).to_string(index=False), flush=True)
    C = {n: (np.concatenate(a), np.concatenate(b)) for n, (a, b) in parts.items()}
    N = len(C['base'][0])
    lab.log({n: (len(v[0]), round(float(v[1].mean()), 4)) for n, v in C.items()})
    models = {}

    def fit(name, train, src=None, w=None):
        Xs = [C[n][0] for n, _ in train]; ys = [C[n][1] for n, _ in train]
        ws = [np.full(len(C[n][0]), sh * N / len(C[n][0])) for n, sh in train]
        if src is not None:
            Xs.append(C[src][0]); ys.append(C[src][1]); ws.append(np.full(len(C[src][0]), float(w)))
        t0 = time.time()
        models[name] = L.fit_cat(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))
        lab.log(f'fold {k} {name}: {sum(len(x) for x in Xs)} rows, {time.time() - t0:.0f}s')

    RAW = [('base', 1.0)]; MS = [('base_S', 1.0)]; MIX = [('base', 0.5), ('base_S', 0.5)]
    fit('i', RAW); fit('i_mS', MS); fit('i_mix', MIX)
    for w in WS:
        fit(f'ii_w{w:g}', RAW, 'src_raw', w); fit(f'iii_w{w:g}', RAW, 'src_F', w); fit(f'iv_w{w:g}', RAW, 'src_S', w)
        fit(f'iv_mix_w{w:g}', MIX, 'src_S', w); fit(f'iv_mS_w{w:g}', MS, 'src_S', w)
    fit('ii_w1', RAW, 'src_raw', 1.0)                                    # = the production recipe (the month at its natural weight)

    def preds(X):
        pr = {n: m.predict_proba(X)[:, 1] for n, m in models.items()}
        for f_ in ('ii', 'iii', 'iv', 'iv_mix', 'iv_mS'):
            pr[f'{f_}_bag'] = np.mean([pr[f'{f_}_w{w:g}'] for w in WS], 0)
        return pr

    for i, (rc, X, meta) in enumerate(EVR):
        pr = preds(X)
        for n, p in pr.items():
            res.append(L.score(rc, meta, p, n))
        if KTTM:
            pt = [preds(Xj) for Xj in EVT[i]]
            for n in ('i', 'i_mS', 'i_mix', 'ii_bag', 'iii_bag', 'iv_bag', 'iv_mix_bag', 'iv_mS_bag'):
                res.append(L.score(rc, meta, np.mean([q[n] for q in pt], 0), f'{n}@ttm'))
                res.append(L.score(rc, meta, pt[0][n], f'{n}@ttm1'))
                res.append(L.score(rc, meta, 0.5 * pr[n] + 0.5 * np.mean([q[n] for q in pt], 0), f'{n}@ttmr'))
    for rc, X, meta in EVM:
        pr = preds(X)
        for n, p in pr.items():
            df = L.score(rc, meta, p, n); df['hidden_est'] = rc.hidden_est; df['share'] = rc.share
            res.append(df)
    del models, C; gc.collect()
    Rr = pd.concat(res, ignore_index=True)
    Rr.to_pickle(f'{L.OUT}/step3_{TAG}.pkl')

Rr = pd.concat(res, ignore_index=True)
Rr.to_pickle(f'{L.OUT}/step3_{TAG}.pkl')
for s in ('R', 'M'):
    G = Rr[Rr.set == s]
    arms = list(dict.fromkeys(G.arm))
    print(f'\n== target {s}: folds {FOLDS}, n = {int((G.arm == "i").sum())} windows ==')
    print(pd.DataFrame([dict(arm=a, iou=lab.aggregate(G[G.arm == a])) for a in arms]).round(4).to_string(index=False))
    pairs = [('i', a) for a in arms if a != 'i'] + [('ii_bag', 'iii_bag'), ('ii_bag', 'iv_bag'), ('ii_bag', 'iv_mix_bag'), ('i_mS', 'iv_mS_bag'),
                                                     ('i_mix', 'iv_mix_bag')]
    if s == 'R' and KTTM:
        pairs += [('ii_bag', 'iv_bag@ttm'), ('iv_bag', 'iv_bag@ttm'), ('iv_bag', 'iv_bag@ttmr')]
    for a, b in pairs:
        print(L.fmt(L.compare(G, a, b)))
lab.log('DONE')
