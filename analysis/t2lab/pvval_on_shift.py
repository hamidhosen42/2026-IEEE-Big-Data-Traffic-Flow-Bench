"""pvval copy of pvon_shift.py (not edited; outputs redirected to results/pvval): synthetic scenario shift replication on
folds 0-1 for (1) the kernel bag 10/20/40 vs the half-weight bag 5/10/20 and (2) masked (c) vs raw (b) source rows on RAW
targets.  Original docstring follows.
pvon: SYNTHETIC SCENARIO SHIFT inside the train split = the only local emulation of the April deployment regime
(new scenario + 31 same-scenario days + RAW target histories) that has labels.

For a target fold k: the "new-scenario period" = source month (31 days next to fold k) + fold k.  In that period the traffic state
is rolled by DELTA links along the corridor (speed / v_cut and flow / capacity are rolled, so the queue state is rolled exactly;
the static link attributes stay in place): the queue locations no longer match the old-scenario prior nor what the base model
learned about pos / static attributes, the local dynamics are unchanged.  The released "masked layer" of the new scenario is
emulated on the rolled cube (regimes 0.5 / 0.7 / 0.8, i.i.d. cells).
  old-scenario prior   = train days outside fold k and outside the source month (unshifted)
  scenario prior       = masked source month (31 days; visible cells; +-2 slots), blended a = n/(n+4) as kernel tfb-t2opv
Arms (CatBoost onset as production, no ramp columns; March-row weights 10 / 20 / 40, bag):
  a_tr    base model (raw old-scenario rows), old prior at evaluation                      = production 'base'
  a_s4    base model, blended prior at evaluation                                          = 'sp4'
  b       base + source-month onset windows with RAW histories (ideal, not available in reality)
  c       base + source-month onset windows built from the MASKED layer (selection, history, slot T, visible labels) = 'pv'
  d       (raw base rows + masked copies of the base rows) + masked source rows            = candidate fix (train-time masking)
  amS, m  base trained on masked rows only (amS) + masked source rows (m): the matched-regime pair for the M targets
  c@ttm   arm c with test-time masking of the raw target history (K maskings averaged, slot T untouched)
Targets of fold k: R = official-like windows, RAW histories (deployment regime) ; M = pseudo windows selected / featured on the
masked layer (the regime of the pseudo-March evaluation).
Several shift sizes share the base models of a fold (the old-scenario rows do not depend on the shift).
Usage: pvon_shift.py FOLDS MASKSEED DELTAS(comma) [K_TTM] [ARMS]   (delta >= 1: links; 0 < delta < 1: fraction of L; 0 = no shift; A / B = half of the panels by L/3;
                                                                   ARMS default bcd, e.g. bc skips the train-time-masking arm)"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lab
import pvon_lib as L
L.OUT = 'results/pvval'

H, F = lab.H, lab.F
FOLDS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DELTAS = [x if x in ('A', 'B', 'C', 'D') else float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.33]
KTTM = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ARMS = sys.argv[5] if len(sys.argv) > 5 else 'bcd'
WS = (10.0, 20.0, 40.0)
KC = L.KEEPC
NSRC = 31
TAG = f'f{"-".join(map(str, FOLDS))}_s{SEED}_d{"-".join(str(x) for x in DELTAS)}' + L.STAG
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def sub_state(S, days):
    return dict(S, V=S['V'][days], Fq=S['Fq'][days], E=S['E'][days], Q=S['Q'][days], Em=S['Em'][days], dow=S['dow'][days],
                ON=S['ON'][days], OFF=S['OFF'][days], days=[S['days'][i] for i in days])


def shift_panel(P, days, delta, seg=None):
    """new-scenario panel: on `days` the normalised traffic state is rolled by `delta` links (static attributes stay).
    seg: roll only inside the first `seg` links (partial relocation: the bottlenecks of the rest of the corridor stay)."""
    if delta == 0:
        return P
    cut, cap = P['cut'], P['cap']; n = len(cut) if seg is None else seg
    V = P['V'].copy(); Fq = P['Fq'].copy(); Q = P['Q'].copy(); ON = P['ON'].copy(); OFF = P['OFF'].copy()
    V[days, :, :n] = (np.roll(P['V'][days][:, :, :n] / cut[:n], delta, 2) * cut[:n]).astype(np.float32)
    Fq[days, :, :n] = (np.roll(P['Fq'][days][:, :, :n] / cap[:n], delta, 2) * cap[:n]).astype(np.float32)
    Q[days, :, :n] = np.roll(P['Q'][days][:, :, :n], delta, 2)
    ON[days, :, :n] = np.roll(P['ON'][days][:, :, :n], delta, 2); OFF[days, :, :n] = np.roll(P['OFF'][days][:, :, :n], delta, 2)
    out = {k_: v for k_, v in P.items() if not (k_.startswith('sel_') or k_.startswith('fill_'))}
    out.update(V=V, Fq=Fq, E=np.isfinite(V), Q=Q, ON=ON, OFF=OFF)
    return out


res = []
for k in FOLDS:
    parts = {n: ([], []) for n in ['base', 'base_S'] + [f'{a}@{dl}' for a in ('src_raw', 'src_S') for dl in DELTAS]}
    EVR = {dl: [] for dl in DELTAS}; EVM = {dl: [] for dl in DELTAS}
    info = []
    for panel in L.PANELS:
        t1 = time.time()
        P = lab.load_panel(panel); blk = P['block']; D = len(P['days']); Lk = len(P['links'])
        tgt = np.where(blk == k)[0]
        src = np.arange(tgt[0] - NSRC, tgt[0]) if k >= 1 else np.arange(tgt[-1] + 1, tgt[-1] + 1 + NSRC)
        new = np.concatenate([src, tgt]); src0 = int(src[0]); srcset = set(src.tolist())
        old_mask = np.ones(D, bool); old_mask[new] = False
        prior_old = lab.prior_table(P, old_mask)
        # ---- base rows (raw, old scenario) with the old prior in the prior columns
        rt = lab.rows(panel, 'dense1500', lab.ONSET)
        wd = rt.win.d.to_numpy(); wt_ = rt.win.t.to_numpy()
        mb = rt.e & old_mask[wd][rt.wi]
        Xb = np.asarray(rt.X[np.where(mb)[0]]).copy()
        pf = lab.prior_feats(prior_old, P['dow'][wd], wt_, 6)
        for j in range(3):
            Xb[:, L.JP[j]] = pf[j][rt.wi[mb], rt.li[mb]]
        parts['base'][0].append(Xb[:, KC]); parts['base'][1].append(rt.y[mb].astype(int))
        ro = lab.rows(panel, 'offlike', lab.ONSET)
        widx = np.where(ro.win.fold.to_numpy() == k)[0]
        od = ro.win.d.to_numpy(); ot = ro.win.t.to_numpy()
        wl = list(zip(od[widx].tolist(), ot[widx].tolist()))
        wsrc = [(int(a), int(b)) for a, b in zip(wd, wt_) if a in srcset]
        for di, dl in enumerate(DELTAS):
            seg = None
            if dl in ('A', 'B'):                                      # half of the panels shifted by L/3 (A: E/N directions, B: W/S)
                delta = int(round(Lk / 3)) if (lab.PANELS.index(panel) % 2 == 0) == (dl == 'A') else 0
            elif dl in ('C', 'D'):                                    # half of the panels, only the first half of the corridor rolled
                delta = int(round(Lk / 6)) if (lab.PANELS.index(panel) % 2 == 0) == (dl == 'C') else 0; seg = Lk // 2
            else:
                delta = int(round(dl if dl >= 1 else Lk * dl))
            Ps = shift_panel(P, new, delta, seg)
            Pm = L.mask_train(Ps, SEED)
            cQ, cE = L.cum_tables(sub_state(Pm, src))                 # scenario tables over the masked source month (timeline = src days)
            on_m, _ = L.select(Pm, need_og=False)
            if di == 0:
                # masked copies of the base rows (old scenario, windows re-selected on the masked cube; identical for every shift)
                pkb = [w for w in L.spaced_picks(on_m) if old_mask[w[0]]]
                R = L.onset_rows(P, Pm, pkb, prior_old); X, y, e, meta = L.matrix(R)
                parts['base_S'][0].append(X[e][:, KC]); parts['base_S'][1].append(y[e].astype(int))
            # ---- b: source-month production windows, RAW histories of the new scenario, blended prior (scenario days < own day)
            VTs = L.prod_VT(Ps, np.array([w[0] for w in wsrc]), np.array([w[1] for w in wsrc]), len(wsrc))
            R = L.onset_rows(P, L.raw_state(Ps), wsrc, prior_old, VT=VTs); X, y, e, meta = L.matrix(R)
            Xk = L.blend_rows(X, R, meta, P['dow'], prior_old, cQ, cE, day=R['d'] - src0)
            parts[f'src_raw@{dl}'][0].append(Xk[e][:, KC]); parts[f'src_raw@{dl}'][1].append(y[e].astype(int))
            # ---- c: source-month windows built from the masked layer (the kernel recipe)
            pk = [(int(src[a]), b) for a, b in L.spaced_picks(on_m[src])]
            R = L.onset_rows(P, Pm, pk, prior_old); X, y, e, meta = L.matrix(R)
            Xk = L.blend_rows(X, R, meta, P['dow'], prior_old, cQ, cE, day=R['d'] - src0)
            parts[f'src_S@{dl}'][0].append(Xk[e][:, KC]); parts[f'src_S@{dl}'][1].append(y[e].astype(int))
            # ---- target R: official-like windows of fold k, RAW histories of the new scenario
            VT = L.prod_VT(Ps, od, ot, len(od))[widx]
            R = L.onset_rows(P, L.raw_state(Ps), wl, prior_old, VT=VT); X, y, e, metaR = L.matrix(R)
            allsrc = np.full(R['W'], NSRC)                            # every target window sees the whole source month, no target day
            Xk = L.blend_rows(X, R, metaR, P['dow'], prior_old, cQ, cE, day=allsrc)
            d_, t_ = R['d'], R['t']
            rcR = L.namespace(panel, f'R@{dl}', d_, t_, np.stack([Ps['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                              np.stack([Ps['E'][d_, t_ + h] for h in range(1, F + 1)], 1), R['qlast'], R['L'], fold=k)
            xs = []
            for j in range(KTTM):
                Pj = L.mask_train(Ps, 100 + 10 * SEED + j)
                Rj = L.onset_rows(P, Pj, wl, prior_old, VT=VT); Xj, _, _, mj = L.matrix(Rj)
                xs.append(L.blend_rows(Xj, Rj, mj, P['dow'], prior_old, cQ, cE, day=allsrc)[:, KC]); del Pj
            EVR[dl].append((rcR, X[:, KC], Xk[:, KC], metaR, xs))
            # ---- target M: pseudo windows of fold k selected on the masked layer
            on_t, og_t = L.select(sub_state(Pm, tgt))
            wm = [(int(tgt[a]), b) for a, b in L.multistart_onset(on_t, og_t, len(tgt))]
            if wm:
                R = L.onset_rows(P, Pm, wm, prior_old); X, y, e, metaM = L.matrix(R)
                Xk = L.blend_rows(X, R, metaM, P['dow'], prior_old, cQ, cE, day=np.full(R['W'], NSRC))
                d_, t_ = R['d'], R['t']
                rcM = L.namespace(panel, f'M@{dl}', d_, t_, np.stack([Pm['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                                  np.stack([Pm['E'][d_, t_ + h] for h in range(1, F + 1)], 1), R['qlast'], R['L'], fold=k)
                rcM.hidden_est = np.array([bool(lab.established(Ps['Q'][a, b - H:b])) for a, b in zip(d_, t_)])
                rcM.share = Pm['share'][d_]
                EVM[dl].append((rcM, X[:, KC], Xk[:, KC], metaM))
            info.append(dict(fold=k, panel=panel, delta=delta, src=f'{P["days"][src[0]]}..{P["days"][src[-1]]}', base_rows=int(mb.sum()),
                             src_w=len(wsrc), src_rows=len(parts[f'src_raw@{dl}'][0][-1]), srcS_w=len(pk),
                             srcS_rows=len(parts[f'src_S@{dl}'][0][-1]), tgtR_w=len(wl), tgtM_w=len(wm)))
            del Pm, cQ, cE
            if delta:
                del Ps
        lab.log(panel, f'{time.time() - t1:.0f}s')
        lab.drop_panel(panel); gc.collect()
    print(pd.DataFrame(info).to_string(index=False), flush=True)
    C = {n: (np.concatenate(a), np.concatenate(b)) for n, (a, b) in parts.items()}
    N = len(C['base'][0])
    lab.log({n: (len(v[0]), round(float(v[1].mean()), 4)) for n, v in C.items()})
    models = {}

    def fit(name, train, src=None, w=None, seed=0):
        Xs = [C[n][0] for n, _ in train]; ys = [C[n][1] for n, _ in train]
        ws = [np.full(len(C[n][0]), sh * N / len(C[n][0])) for n, sh in train]
        if src is not None:
            Xs.append(C[src][0]); ys.append(C[src][1]); ws.append(np.full(len(C[src][0]), float(w)))
        t0 = time.time()
        models[name] = L.fit_cat(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), seed=seed)
        lab.log(f'fold {k} {name}: {sum(len(x) for x in Xs)} rows, {time.time() - t0:.0f}s')

    RAW = [('base', 1.0)]; MIX = [('base', 0.5), ('base_S', 0.5)]; MS = [('base_S', 1.0)]
    if 'm' in ARMS:
        fit('amS', MS)                                               # base trained on masked rows only (matched regime for M targets)
    fit('a', RAW); fit('aseed', RAW, seed=1)                          # aseed: retraining-noise floor of the paired comparisons
    if 'd' in ARMS:
        fit('amix', MIX)
    FAM = [f_ for f_ in ('b', 'c', 'd', 'm') if f_ in ARMS]
    for dl in DELTAS:
        for w in WS:
            if 'b' in ARMS:
                fit(f'b_w{w:g}', RAW, f'src_raw@{dl}', w)
            if 'c' in ARMS:
                fit(f'c_w{w:g}', RAW, f'src_S@{dl}', w)
                if w == WS[0]:
                    fit('c_w5', RAW, f'src_S@{dl}', 5.0)          # lower weight: response curve (c_bagL = 5 / 10 / 20)
            if 'd' in ARMS:
                fit(f'd_w{w:g}', MIX, f'src_S@{dl}', w)
            if 'm' in ARMS:
                fit(f'm_w{w:g}', MS, f'src_S@{dl}', w)

        def preds(X):
            pr = {n: m.predict_proba(X)[:, 1] for n, m in models.items()}
            for f_ in FAM:
                pr[f'{f_}_bag'] = np.mean([pr[f'{f_}_w{w:g}'] for w in WS], 0)
            if 'c' in ARMS:
                pr['c_bagL'] = np.mean([pr[f'c_w{w:g}'] for w in (5, 10, 20)], 0)
            return pr

        ren = {'a': 'a_s4', 'amix': 'amix_s4', 'aseed': 'aseed_s4', 'amS': 'amS_s4'}
        for rc, X, Xk, meta, xs in EVR[dl]:
            res.append(L.score(rc, meta, models['a'].predict_proba(X)[:, 1], 'a_tr'))
            pr = preds(Xk)
            for n, p in pr.items():
                res.append(L.score(rc, meta, p, ren.get(n, n)))
            if xs:
                pt = [preds(Xj) for Xj in xs]
                for n in ['a'] + (['amix'] if 'd' in ARMS else []) + (['amS'] if 'm' in ARMS else []) + [f'{f_}_bag' for f_ in FAM]:
                    res.append(L.score(rc, meta, np.mean([q[n] for q in pt], 0), f'{ren.get(n, n)}@ttm'))
                    res.append(L.score(rc, meta, 0.5 * pr[n] + 0.5 * np.mean([q[n] for q in pt], 0), f'{ren.get(n, n)}@ttmr'))
        for rc, X, Xk, meta in EVM[dl]:
            out = [('a_tr', models['a'].predict_proba(X)[:, 1])]
            pr = preds(Xk)
            out += [(ren.get(n, n), p) for n, p in pr.items()]
            for n, p in out:
                df = L.score(rc, meta, p, n); df['hidden_est'] = rc.hidden_est; df['share'] = rc.share
                res.append(df)
        pd.concat(res, ignore_index=True).to_pickle(f'{L.OUT}/shift_{TAG}.pkl')
    del models, C; gc.collect()

Rr = pd.concat(res, ignore_index=True)
Rr.to_pickle(f'{L.OUT}/shift_{TAG}.pkl')
for dl in DELTAS:
    for s in (f'R@{dl}', f'M@{dl}'):
        G = Rr[Rr.set == s]
        arms = list(dict.fromkeys(G.arm))
        print(f'\n== SHIFT target {s}: folds {FOLDS}, n = {int((G.arm == "a_s4").sum())} windows ==')
        print(pd.DataFrame([dict(arm=a, iou=lab.aggregate(G[G.arm == a])) for a in arms]).round(4).to_string(index=False))
        pairs = [('a_s4', a) for a in arms if a != 'a_s4'] + [('b_bag', 'c_bag'), ('b_bag', 'd_bag'), ('c_bag', 'd_bag'), ('amix_s4', 'd_bag')]
        if s[0] == 'R' and KTTM:
            pairs += [('c_bag', 'c_bag@ttm'), ('c_bag', 'c_bag@ttmr')]
        for a, b in pairs:
            if a in arms and b in arms:
                print(L.fmt(L.compare(G, a, b)))
lab.log('DONE')
