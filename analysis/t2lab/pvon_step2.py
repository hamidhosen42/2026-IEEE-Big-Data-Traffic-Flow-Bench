"""pvon steps 1 + 2: baseline reproduction and the MASKING-ADAPTATION control on pseudo-March.

Evaluation: pseudo-March onset windows of days >= CUT (official-like multi-start greedy on the masked layer, visible labels),
            evaluation features = masked-layer history + masked slot T, s4 scenario-prior blend (March days < d).
Same-scenario training rows: March onset windows of days < CUT (production onset rule, 72-slot spacing, visible labels, blended prior).
Arms (CatBoost 600 it / depth 8 / lr 0.08, no ramp columns = production v13 onset):
  A, A_s4        raw train rows (lab dense1500 onset rows)                          [A: train prior at evaluation]
  B<w>, Bbag     raw train rows + March rows x w (10 / 20 / 40), bag = mean         = kaggle_jobs/t2_onset_pv.py
  AmF_all        train rows with EMULATED MASKED histories, same windows as A (labels: all raw observed cells)
  AmS_all        train windows RE-SELECTED on the emulated masked train cube (the rule that builds the March rows), masked
                 features, visible labels                                             = full emulation of the March rows
  AmF_mix/AmS_mix raw rows + masked rows (each half of the train weight)
  BmF_all ... BmS_mix   the same + March rows x w (bag of 10 / 20 / 40)
  The real same-scenario training effect is Bm* - Am* (both sides trained in the evaluation's feature regime).
Usage: pvon_step2.py CUT MASKSEEDS(comma)     Decision support only (March labels -> private file)."""
import gc, json, os, re, sys, time
import numpy as np
import pandas as pd
import lab
import pvon_lib as L

H, F = lab.H, lab.F
CUT = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEEDS = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1]
WS = (10.0, 20.0, 40.0)
FAMS = sys.argv[3] if len(sys.argv) > 3 else 'FS'             # which masked-train emulations: F (same windows), S (re-selected)
TAG = f'cut{CUT}_s{"-".join(map(str, SEEDS))}' + ('' if FAMS == 'FS' else f'_{FAMS}') + L.STAG
KC = L.KEEPC
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

Xm, ym, Xt, yt, EV = [], [], [], [], []
XF = {s: [] for s in SEEDS}; yF = {s: [] for s in SEEDS}; XS = {s: [] for s in SEEDS}; yS = {s: [] for s in SEEDS}
info = []
for panel in L.PANELS:
    t1 = time.time()
    P, Pv = L.val_panel(panel)
    on, og = L.select(Pv)
    D = len(Pv['days'])
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    cQ, cE = L.cum_tables(Pv)
    picks = L.spaced_picks(on, 0, CUT)
    nm = 0
    if picks:
        R = L.onset_rows(P, Pv, picks, prior_tr); X, y, e, meta = L.matrix(R)
        Xk = L.blend_rows(X, R, meta, Pv['dow'], prior_tr, cQ, cE)
        Xm.append(Xk[e][:, KC]); ym.append(y[e].astype(int)); nm = int(e.sum())
    wins = [w for w in L.multistart_onset(on, og, D) if w[0] >= CUT]
    if wins:
        R = L.onset_rows(P, Pv, wins, prior_tr); X, y, e, meta = L.matrix(R)
        Xk = L.blend_rows(X, R, meta, Pv['dow'], prior_tr, cQ, cE)
        d_, t_ = R['d'], R['t']
        rc = L.namespace(panel, 'march_on', d_, t_, np.stack([Pv['Q'][d_, t_ + h] for h in range(1, F + 1)], 1),
                         np.stack([Pv['E'][d_, t_ + h] for h in range(1, F + 1)], 1), R['qlast'], R['L'])
        rc.share = Pv['share'][d_]; rc.hist_cov = R['hist_cov']
        EV.append((rc, X[:, KC], Xk[:, KC], meta))
    # ---- raw train rows (production rows of the lab cache)
    rt = lab.rows(panel, 'dense1500', lab.ONSET); r = np.where(rt.e)[0]
    Xt.append(np.asarray(rt.X[r])[:, KC]); yt.append(rt.y[r].astype(int))
    wt = list(zip(rt.win.d.tolist(), rt.win.t.tolist()))
    # sanity: the raw-state selection reproduces the production onset training windows exactly
    on_raw, _ = L.select(L.raw_state(P), need_og=False)
    assert L.spaced_picks(on_raw) == wt, (panel, 'raw-state spaced picks != dense1500 onset windows')
    nS = {}
    for s in SEEDS:
        Pm = L.mask_train(P, s)
        # F-mask: the production windows, masked history + masked slot T, labels on all raw observed cells
        R = L.onset_rows(P, Pm, wt, prior_tr)
        s6 = np.minimum(R['t'] + 6, 287); R['rows'] = [(6, R['rows'][0][1], P['Q'][R['d'], s6], P['E'][R['d'], s6])]
        X, y, e, meta = L.matrix(R)
        assert int(e.sum()) == len(r)
        XF[s].append(X[e][:, KC]); yF[s].append(y[e].astype(int))
        # S-mask: windows re-selected on the masked cube (the March-row recipe), visible labels
        on_m, _ = L.select(Pm, need_og=False)
        pk = L.spaced_picks(on_m)
        R = L.onset_rows(P, Pm, pk, prior_tr); X, y, e, meta = L.matrix(R)
        XS[s].append(X[e][:, KC]); yS[s].append(y[e].astype(int)); nS[s] = (len(pk), int(e.sum()), int(y[e].sum()))
        del Pm
    info.append(dict(panel=panel, march_train_w=len(picks), march_rows=nm, eval_w=len(wins), raw_w=len(wt), raw_rows=len(r),
                     raw_pos=int(rt.y[r].sum()), **{f'S{s}': str(nS[s]) for s in SEEDS}))
    lab.log(panel, info[-1], f'{time.time() - t1:.0f}s')
    del cQ, cE; lab.drop_panel(panel); gc.collect()

print(pd.DataFrame(info).to_string(index=False))
Xm = np.concatenate(Xm); ym = np.concatenate(ym); Xt = np.concatenate(Xt); yt = np.concatenate(yt)
N = len(Xt)
lab.log('March training rows', len(Xm), 'pos rate', round(float(ym.mean()), 4), '| raw train rows', N, 'pos rate', round(float(yt.mean()), 4))

models = {}


def fit(name, parts, march_w=None):
    """parts: list of (X, y, total_weight_share); train weight sums to N (the raw row count); March rows get weight march_w each."""
    Xs = [p[0] for p in parts]; ys = [p[1] for p in parts]
    ws = [np.full(len(p[0]), p[2] * N / len(p[0])) for p in parts]
    if march_w is not None:
        Xs.append(Xm); ys.append(ym); ws.append(np.full(len(Xm), float(march_w)))
    t0 = time.time()
    models[name] = L.fit_cat(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))
    lab.log(f'{name} trained on {sum(len(x) for x in Xs)} rows in {time.time() - t0:.0f}s')


WX = (2.0, 5.0)                                                 # extra (lower) weights, plain family only: weight response


def family(prefix, parts):
    fit(f'A{prefix}', parts)
    for w in WS + (WX if prefix == '' else ()):
        fit(f'B{prefix}_w{w:g}', parts, w)


family('', [(Xt, yt, 1.0)])
for s in SEEDS:
    xf = np.concatenate(XF[s]); yf = np.concatenate(yF[s]); xs = np.concatenate(XS[s]); ys_ = np.concatenate(yS[s])
    lab.log(f'mask seed {s}: F rows {len(xf)} pos {yf.mean():.4f} | S rows {len(xs)} pos {ys_.mean():.4f}')
    if 'F' in FAMS:
        family(f'mF_all#{s}', [(xf, yf, 1.0)])
        family(f'mF_mix#{s}', [(Xt, yt, 0.5), (xf, yf, 0.5)])
    if 'S' in FAMS:
        family(f'mS_all#{s}', [(xs, ys_, 1.0)])
        family(f'mS_mix#{s}', [(Xt, yt, 0.5), (xs, ys_, 0.5)])
    del xf, xs; gc.collect()

res = []
for rc, X, Xk, meta in EV:
    pr = {n: m.predict_proba(Xk)[:, 1] for n, m in models.items()}
    out = {'A': models['A'].predict_proba(X)[:, 1], 'A_s4': pr['A']}
    fams = sorted({n[1:].split('_w')[0] for n in models if n.startswith('B')})
    for f_ in fams:
        if f_:
            out[f'A{f_}'] = pr[f'A{f_}']
        for w in WS:
            out[f'B{f_}_w{w:g}'] = pr[f'B{f_}_w{w:g}']
        out[f'B{f_}_bag'] = np.mean([pr[f'B{f_}_w{w:g}'] for w in WS], 0)
        if not f_:
            for w in WX:
                out[f'B_w{w:g}'] = pr[f'B_w{w:g}']
            out['B_bagL'] = np.mean([pr[f'B_w{w:g}'] for w in (5.0, 10.0, 20.0)], 0)      # the kernel bag at half the weights
    for name, p in out.items():
        df = L.score(rc, meta, p, name); df['share'] = rc.share; df['hist_cov'] = rc.hist_cov
        res.append(df)
Rr = pd.concat(res, ignore_index=True)
# seed-mean arms (per-window mean IoU over the mask seeds): 'AmF_all#1' -> 'AmF_all', 'BmF_all#1_bag' -> 'BmF_all_bag'
key = ['panel', 'd', 't', 'cond']
Rr['base'] = Rr.arm.str.replace(r'#\d+', '', regex=True)
if len(SEEDS) > 1:
    g = Rr[Rr.arm != Rr.base].groupby(['base'] + key + ['family', 'set', 'fold'], as_index=False)[['iou', 'share', 'hist_cov']].mean()
    g['arm'] = g.base
    Rr = pd.concat([Rr, g], ignore_index=True)
else:
    Rr['arm'] = Rr.base
Rr = Rr.drop(columns='base')
Rr.to_pickle(f'{L.OUT}/step2_{TAG}.pkl')
S = pd.DataFrame([dict(arm=a, iou=lab.aggregate(Rr[Rr.arm == a]), n=int((Rr.arm == a).sum())) for a in dict.fromkeys(Rr.arm)])
print(f'\n== pseudo-March onset, days >= {CUT} (masked evaluation regime) =='); print(S.round(4).to_string(index=False))
print('\n== paired bootstrap ==')
cmp_ = []
arms = list(dict.fromkeys(Rr.arm))
for b in arms:
    if b not in ('A_s4',):
        cmp_.append(L.compare(Rr, 'A_s4', b))
for b in arms:
    if b.startswith('B') and b.endswith('_bag') and b != 'B_bag':
        a = 'A' + b[1:-4]
        if a in arms:
            cmp_.append(L.compare(Rr, a, b, 'same-regime effect of the March rows'))
        cmp_.append(L.compare(Rr, 'B_bag', b, 'vs the kernel recipe'))
for c in cmp_:
    print(L.fmt(c))
pd.DataFrame(cmp_).to_csv(f'{L.OUT}/step2_{TAG}_cmp.csv', index=False)
lab.log('DONE')
