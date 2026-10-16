"""extcal REPORT (decode + fits + scoring; no member training).  Design and the PRE-REGISTERED rule: extcal_lib.py.
Inputs: results/extcal/probs/CF15_s{0,1}.pkl (and CF31_s{0,1}.pkl for FM-PM), results/pvogf/probs/{MD_s0, MD_s1, FULL_s0}.pkl,
the extcal_build.py and pvogf_build.py caches.
Sections
  0 calibration diagnostics (reliability per step, inside / outside qlast): CF15 out-of-sample fit windows vs MD evaluation windows
  1 fits on the CF15 windows (March days < 15) -> parameters results/extcal/params_CF15.json (+ size model)
  2 many-day pseudo-March (offlike_m binding, runstart_m reported; days >= 15): each candidate vs B0 = pvogfv lab analog
  3 FM-PM private (candidates passing (1)): CF31 re-fit, lab-emulated April decisions (FULL_s0) -> params_CF31.json
  4 verdict under the pre-registered rule
Usage: extcal_report.py [fmpm=all]  (fmpm=all also prices candidates that fail (1); NOT binding, labelled as such)"""
import json, os, sys
from importlib import util
import numpy as np
import pandas as pd
import lab
import extcal_lib as E

kw = dict(a.split('=', 1) for a in sys.argv[1:])
F, ONGOING, PANELS = E.F, E.ONGOING, lab.PANELS
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
_s = util.spec_from_file_location('fpa', f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts/fpmatch_lib.py')
fpa = util.module_from_spec(_s); _s.loader.exec_module(fpa)
doc = E.__doc__; print(doc[doc.index('PRE-REGISTERED'):])
PM_SETS = ('offlike_m', 'runstart_m')
CFSEEDS = [int(x) for x in kw.get('cfseeds', '0,1').split(',')]      # smoke tests only; the binding run uses seeds 0 and 1
if CFSEEDS != [0, 1]:
    print('*** SMOKE RUN: cross-fit seeds', CFSEEDS, '- NOT the pre-registered configuration ***')


def loadp(path):
    return {k: v.astype(np.float32) for k, v in pd.read_pickle(path).items()} if os.path.exists(path) else None


mix = lambda P, key, arm: 0.25 * P[key + (f'{arm}_np',)] + 0.25 * P[key + (f'{arm}_id',)] + 0.5 * P[key + (f'{arm}_full',)]
MD = {s: loadp(f'{E.PVRES}/probs/MD_s{s}.pkl') for s in (0, 1)}
FULL = loadp(f'{E.PVRES}/probs/FULL_s0.pkl')


def cf_probs(mode):
    """seed-average out-of-sample pvogf probabilities on the fit windows of a cross-fit mode: {panel: (P6, z, sel)}"""
    Ps = [loadp(f'{E.RES}/probs/{mode}_s{s}.pkl') for s in CFSEEDS]
    if any(P is None or (PANELS[-1], 'ogf_full') not in P for P in Ps):
        return None
    out = {}
    for p in PANELS:
        z = dict(np.load(f'{E.CACHE}/fit_{p}.npz'))
        sel = (z['d'] < E.CUT) if mode == 'CF15' else np.ones(len(z['d']), bool)
        P6 = sum(0.25 * P[(p, 'ogf_np')] + 0.25 * P[(p, 'ogf_id')] + 0.5 * P[(p, 'ogf_full')] for P in Ps) / len(Ps)
        keep = np.zeros(P6.shape, bool); m = z['meta']; keep[m[:, 0], m[:, 2] - 1, m[:, 1]] = True
        out[p] = dict(P6=P6[sel], keep=keep[sel], truth=z['truth'][sel], ev=z['ev'][sel], qlast=z['qlast'][sel], d=z['d'][sel],
                      wt=z['wt'][sel], SZ=z['SZ'][sel], n=z['n'][sel])
    return out


def with_nvis(SZ, nvis):
    return np.concatenate([SZ, np.asarray(nvis, np.float32)[..., None]], -1)


def size_model(mode, b2=False):
    """b / b2: LightGBM poisson on the F03 windows + the March faithful fitting windows of the mode (b2: + n_vis_h)."""
    import lightgbm as lgb
    cols = E.SIZE_COLS_B2 if b2 else E.SIZE_COLS
    Xs, ys, ws = [], [], []
    for p in PANELS:
        z = np.load(f'{E.CACHE}/szf03_{p}.npz')
        Xs.append((with_nvis(z['SZ'], z['nvis']) if b2 else z['SZ']).reshape(-1, len(cols))); ys.append(z['n'].reshape(-1)); ws.append(np.repeat(z['w'], F))
        f = np.load(f'{E.CACHE}/fit_{p}.npz'); sel = (f['wt'] > 0) & ((f['d'] < E.CUT) if mode == 'CF15' else True)
        SZf = with_nvis(f['SZ'], f['ev'].sum(2)) if b2 else f['SZ']
        Xs.append(SZf[sel].reshape(-1, len(cols))); ys.append(f['n'][sel].reshape(-1)); ws.append(np.repeat(f['wt'][sel] * E.WM, F))
    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    prm = dict(objective='poisson', num_leaves=63, learning_rate=0.05, min_data_in_leaf=50, seed=0, verbose=-1, num_threads=lab.THREADS)
    mdl = lgb.train(prm, lgb.Dataset(X, y, weight=w, feature_name=cols), 400)
    mdl.save_model(f'{E.CACHE}/models/size_{mode}{"_b2" if b2 else ""}.txt')
    return mdl, len(y)


def fit_all(mode, CFP):
    """all candidate parameters fitted on the cross-fit windows CFP."""
    prm = {}
    cells = []
    for p in PANELS:
        c = CFP[p]; m = c['keep'] & c['ev']; w_, h_, l_ = np.where(m)
        cells.append(pd.DataFrame(dict(panel=p, h=h_ + 1, inq=c['qlast'][w_, l_].astype(int), p=c['P6'][w_, h_, l_], y=c['truth'][w_, h_, l_].astype(float))))
    cells = pd.concat(cells, ignore_index=True)
    for k in ('a1', 'a2', 'a3'):
        prm[k] = E.fit_cells(cells, k)
    prm['b'], nb = size_model(mode); prm['b2'], _ = size_model(mode, b2=True)
    for k, outside in (('c1', False), ('c2', True)):
        tabs = [E.cum_tables(CFP[p]['P6'], E.decode6(CFP[p]['P6']), CFP[p]['truth'], CFP[p]['ev'], CFP[p]['qlast'], outside) for p in PANELS]
        f, tr = E.fit_factors(tabs, PANELS); prm[k] = f
        print(f'  {mode} {k}: f_h = {f.tolist()} | fit-window aggregate IoU {tr[0][1]:.4f} -> {tr[-1][1]:.4f}')
    print(f'  {mode}: cells for a1-a3 {len(cells)} (pos {cells.y.mean():.3f}); size-model rows {nb}; '
          f'a1 groups fitted {sum(v is not None for v in prm["a1"].values())}/{len(prm["a1"])}, a3 {sum(v is not None for v in prm["a3"].values())}/{len(prm["a3"])}')
    js = {k: {'|'.join(map(str, kk)): (None if v is None else (list(v) if k != 'a2' else [v[0].tolist(), v[1].tolist()])) for kk, v in prm[k].items()}
          for k in ('a1', 'a2', 'a3')}
    js.update(c1=prm['c1'].tolist(), c2=prm['c2'].tolist(), b=f'{E.CACHE}/models/size_{mode}.txt', b2=f'{E.CACHE}/models/size_{mode}_b2.txt')
    json.dump(js, open(f'{E.RES}/params_{mode}.json', 'w'))
    return prm, cells


def candidate(k, prm, P6, D6, qlast, panel, SZ, scored=None):
    """candidate pvogf decision (before the veto).  scored: lab ev mask [W, 6, L] (None = every link scored, April)."""
    if k in ('a1', 'a2', 'a3'):
        return E.decode6(E.apply_cal(P6, panel, qlast, k, prm[k]))
    if k == 'b':
        m6 = prm['b'].predict(SZ.reshape(-1, len(E.SIZE_COLS))).reshape(len(P6), F)
        return E.size_decode(P6, m6)
    if k == 'b2':
        nv = scored.sum(2) if scored is not None else np.full((len(P6), F), P6.shape[2])
        m6 = prm['b2'].predict(with_nvis(SZ, nv).reshape(-1, len(E.SIZE_COLS_B2))).reshape(len(P6), F)
        return E.size_decode(P6, m6, scored)
    return E.rescale_steps(P6, D6, prm[k], qlast if k == 'c2' else None)


def boot(dfa, dfb, qs=(0.004, 0.996), n=2000, seed=0):
    """lab.paired_bootstrap with other quantiles (Bonferroni-style 99.2 % CI for 6 candidates)."""
    key = ['panel', 'd', 't', 'cond']
    m = dfa[key + ['family', 'iou']].merge(dfb[key + ['iou']], on=key, suffixes=('_a', '_b'))
    rng = np.random.default_rng(seed)
    groups = [(fam, g['iou_b'].to_numpy() - g['iou_a'].to_numpy()) for (fam, _), g in m.groupby(['family', 'panel'])]
    fams = sorted({f for f, _ in groups}); diffs = np.zeros(n)
    for b in range(n):
        acc = {f: [] for f in fams}
        for f, dv in groups:
            acc[f].append(dv[rng.integers(0, len(dv), len(dv))].mean())
        diffs[b] = np.mean([np.mean(v) for v in acc.values()])
    return float(np.quantile(diffs, qs[0])), float(np.quantile(diffs, qs[1]))


# ================================================================ 0 + 1. CF15 fits
CF15 = cf_probs('CF15')
assert CF15 is not None, 'CF15 cross-fit probabilities missing'
print('\n== 0. RELIABILITY (keep-region cells with a visible label): mean p vs observed rate ==')


def reliab(rows, tag):
    D = pd.concat(rows, ignore_index=True)
    D['bin'] = pd.cut(D.p, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    t = D.groupby(['inq', 'bin'], observed=True).agg(n=('y', 'size'), p=('p', 'mean'), obs=('y', 'mean')).reset_index()
    th = D.groupby(['h', 'inq']).agg(p=('p', 'mean'), obs=('y', 'mean')).unstack('inq')
    print(f'-- {tag}: by p bin (inq = inside qlast)'); print(t.round(3).to_string(index=False))
    print(f'-- {tag}: by step'); print(th.round(3).to_string())


rr = []
for p in PANELS:
    c = CF15[p]; m = c['keep'] & c['ev']; w_, h_, l_ = np.where(m)
    rr.append(pd.DataFrame(dict(h=h_ + 1, inq=c['qlast'][w_, l_].astype(int), p=c['P6'][w_, h_, l_], y=c['truth'][w_, h_, l_].astype(float))))
reliab(rr, 'CF15 out-of-sample fit windows (days < 15)')
EV = {}
for p in PANELS:
    for s in PM_SETS:
        z = np.load(f'{lab.CACHE}/pvogf/ev_{p}__{s}.npz'); sz = np.load(f'{E.CACHE}/szev_{p}__{s}.npz')
        keep = np.zeros(z['truth'].shape, bool); m = z['meta']; keep[m[:, 0], m[:, 2] - 1, m[:, 1]] = True
        EV[(p, s)] = dict(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], d=z['d'], t=z['t'], keep=keep, SZ=sz['SZ'])
rr = []
for p in PANELS:
    e = EV[(p, 'offlike_m')]; P6 = sum(mix(MD[s], ('offlike_m', p), 'ogf') for s in (0, 1)) / 2
    m = e['keep'] & e['ev']; w_, h_, l_ = np.where(m)
    rr.append(pd.DataFrame(dict(h=h_ + 1, inq=e['qlast'][w_, l_].astype(int), p=P6[w_, h_, l_], y=e['truth'][w_, h_, l_].astype(float))))
reliab(rr, 'MD evaluation offlike_m (days >= 15)')

print('\n== 1. FITS on CF15 (March days < 15, out-of-sample cross-fit, seed avg) ==')
PRM15, _ = fit_all('CF15', CF15)

# ================================================================ 2. many-day pseudo-March
cfgs = {'avg': [0, 1], 's0': [0], 's1': [1]}
rows = []
for p in PANELS:
    fam = p.rsplit('_', 1)[0]
    for s in PM_SETS:
        e = EV[(p, s)]
        for cfg, seeds in cfgs.items():
            pA = sum(mix(MD[x], (s, p), 'base') for x in seeds) / len(seeds); pB = sum(mix(MD[x], (s, p), 'ogf') for x in seeds) / len(seeds)
            A = E.decode6(pA); B = E.decode6(pB); B0, _ = E.veto(A, B, e['qlast'])
            arms = {'B0': B0, 'pvo': A, 'pvogf': B}
            for k in E.CANDS:
                arms[k], _ = E.veto(A, candidate(k, PRM15, pB, B, e['qlast'], p, e['SZ'], e['ev']), e['qlast'])
            if cfg == 'avg':
                arms['oracle_step'] = E.size_decode(pB, (e['truth'] & e['ev']).sum(2), e['ev'])
            for arm, dec in arms.items():
                rows.append(pd.DataFrame(dict(cfg=cfg, arm=arm, set=s, panel=p, family=fam, cond=ONGOING, d=e['d'], t=e['t'],
                                              chg=(dec ^ B0).sum((1, 2)), **E.score_dec(dec, e['truth'], e['ev'], e['qlast']))))
D = pd.concat(rows, ignore_index=True); D.to_pickle(f'{E.RES}/pm_frames.pkl')


def cmp(cfg, s, b, a='B0'):
    g = D[(D.cfg == cfg) & (D.set == s)]; A = g[g.arm == a]; B = g[g.arm == b]
    bt = lab.paired_bootstrap(A, B); dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
    lo9, hi9 = boot(A, B)
    return dict(cfg=cfg, set=s, cand=b, n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                lo992=lo9, hi992=hi9, p_pos=bt['p_pos'], npos=int((dl > 1e-9).sum()), nneg=int((dl < -1e-9).sum()),
                win_changed=int((B.chg.to_numpy() > 0).sum()), dn_all=int(B.n_all.sum() - A.n_all.sum()), per_panel={k: round(v, 4) for k, v in dl.items()})


T1 = []
print('\n== 2. MANY-DAY pseudo-March (days >= 15; MD probabilities; calibration fitted on CF15) — candidate vs B0 (pvogfv) ==')
for cfg in ('avg', 's0', 's1'):
    for s in PM_SETS:
        for b in E.CANDS + (('oracle_step', 'pvogf', 'pvo') if cfg == 'avg' else ()):
            r = cmp(cfg, s, b); T1.append(r)
            print(f"{cfg:3s} {s:10s} {b:11s} n={r['n']:4d} {r['iou_B']:.4f} vs {r['iou_A']:.4f} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                  f" (99.2% [{r['lo992']:+.4f}, {r['hi992']:+.4f}]) panels +{r['npos']}/-{r['nneg']} windows changed {r['win_changed']} dcells {r['dn_all']:+d}")
T1 = pd.DataFrame(T1); T1.to_csv(f'{E.RES}/pm_table.csv', index=False)
print('\nper-panel deltas (avg, offlike_m):')
for k in E.CANDS:
    print(f'  {k:3s}', T1[(T1.cfg == 'avg') & (T1.set == 'offlike_m') & (T1.cand == k)].iloc[0]['per_panel'])
g = D[(D.cfg == 'avg') & (D.set == 'offlike_m')]
print('\nper-step IoU profile is not separable; positives (all-cell) per arm on offlike_m avg:',
      {a: int(g[g.arm == a].n_all.sum()) for a in ('B0',) + E.CANDS + ('oracle_step',)}, '| true', int(g[g.arm == 'B0'].n_true.sum()))

# ================================================================ 3. FM-PM private
PASS1 = [k for k in E.CANDS if (lambda r: r['diff'] >= 0.005 and r['lo'] > 0 and r['npos'] >= 6)(T1[(T1.cfg == 'avg') & (T1.set == 'offlike_m') & (T1.cand == k)].iloc[0])]
print('\ncandidates passing gate (1):', PASS1 or 'none')
FMC = list(E.CANDS) if kw.get('fmpm') == 'all' else PASS1
FM = {}
CF31 = cf_probs('CF31') if FMC else None
if FMC and CF31 is None:
    print('FM-PM: CF31 cross-fit probabilities missing -> run extcal_train.py CF31 seeds=0,1')
elif FMC:
    print('\n== 3. FM-PM PRIVATE (CF31 re-fit; lab-emulated April decisions FULL_s0) ==')
    PRM31, _ = fit_all('CF31', CF31)
    emu = pd.read_pickle(f'{E.PVRES}/private_decisions_emulated.pkl')
    off = {k: [] for k in FMC}; DEC = {}
    for p in PANELS:
        z = np.load(f'{lab.CACHE}/pvogf/priv_{p}.npz'); og = z['cond'] == ONGOING; ql = z['qlast'][og]
        SZ = np.load(f'{E.CACHE}/szpriv_{p}.npz')['SZ'][og]
        pA = mix(FULL, ('private', p), 'base')[og]; pB = mix(FULL, ('private', p), 'ogf')[og]
        A = E.decode6(pA); B = E.decode6(pB); B0, _ = E.veto(A, B, ql)
        assert np.array_equal(B0, emu[p]['pvogfv']), f'{p}: B0 differs from the emulated pvogfv'
        DEC[p] = dict(B0=B0, wid=z['window_id'][og], qlast=ql)
        for k in FMC:
            Ck, _ = E.veto(A, candidate(k, PRM31, pB, B, ql, p, SZ, None), ql); DEC[p][k] = Ck
            for i in range(len(B0)):
                off[k].append(dict(window_id=z['window_id'][og][i], panel=p, tod=float(z['tod'][og][i]), nA=int(B0[i].sum()), nB=int(Ck[i].sum()),
                                   changed=int((B0[i] ^ Ck[i]).sum())))
    pd.to_pickle(DEC, f'{E.RES}/private_decisions_emulated.pkl')
    g = D[D.cfg == 'avg']
    for k in FMC:
        o = pd.DataFrame(off[k]); o['dn'] = o.nB - o.nA
        L = g[g.arm == 'B0'][['set', 'panel', 'd', 't', 'iou', 'n_all']].merge(g[g.arm == k][['set', 'panel', 'd', 't', 'iou', 'n_all']], on=['set', 'panel', 'd', 't'], suffixes=('_a', '_b'))
        L['dI'] = L.iou_b - L.iou_a; L['dn'] = L.n_all_b - L.n_all_a; L['tod'] = L.t / 12
        fe = fpa.forecast(L, o, panel_exact=True); fp = fpa.forecast(L, o, panel_exact=False)
        FM[k] = 0.15 * float(fe.mean())
        o.assign(fm=fe).to_csv(f'{E.RES}/fm_private_{k}.csv', index=False)
        print(f"{k:3s}{'' if k in PASS1 else ' (not binding: failed (1))'} | windows changed {int((o.changed > 0).sum())}/40 cells {int(o.changed.sum())} | positives B0 {int(o.nA.sum())} -> {int(o.nB.sum())}"
              f" | FM-PM dIoU {fe.mean():+.4f} -> LB-equiv {FM[k]:+.5f} (pooled matching {0.15 * fp.mean():+.5f})")

# ================================================================ 4. verdict
print('\n== 4. VERDICT (pre-registered rule) ==')
for k in E.CANDS:
    r = T1[(T1.cfg == 'avg') & (T1.set == 'offlike_m') & (T1.cand == k)].iloc[0]
    ok1 = k in PASS1; fm = FM.get(k) if k in PASS1 else None
    ok2 = (fm is not None and fm >= 0.0005)
    print(f"{k:3s}: (1) offlike_m {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] panels +{r['npos']}/8 -> {'PASS' if ok1 else 'FAIL'} | "
          f"(2) FM-PM private {('%+.5f' % fm) if fm is not None else 'n/a'} -> {('PASS' if ok2 else 'FAIL') if ok1 else 'n/a'} | => {'ADOPT' if ok1 and ok2 else 'DO NOT ADOPT'}")
