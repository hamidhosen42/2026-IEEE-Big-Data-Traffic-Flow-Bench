"""verify_pvval_og.py -- ADVERSARIAL VERIFIER for candidate ongoing_april_pvo_as_built (track pvval).
Independent re-implementation of the key pseudo-March comparison B - A (ongoing windows of March days >= CUT).

  * code under test: the March-row functions of kaggle_jobs/t2_ongoing_pv.py are COMPILED FROM THAT FILE (ast, main() never
    runs): fill_time, established, window_features, add_ramp, prior_table, prior_feats, to_matrix, ongoing_keep, decode,
    march_faithful, scenario (its two file readers patched to the local export, April = empty), blend_prior_og.
  * March training rows = the kernel main() recipe (lines 662-687) restricted to March days < CUT.
  * evaluation windows (days >= CUT): runstart_m = kernel run-starts; offlike_m = own interleaved greedy (5 + 5 quota, 72 slots)
    restarted at 00:00 of every March day over an own onset map + the kernel's ongoing candidate map.
  * scoring: kernel decode + own IoU on visible cells + window -> panel -> family mean + own paired bootstrap.
  * models at lab scale (255 leaves, lr 0.05, min_data 40, 500 trees, seed SEED):
      B  = npB / idB / fullB on the F03 lab rows (k = 4, all-train prior) + March rows x WM (fullB: blended prior)
      A1 = np / id on the same F03 rows only (same seed)       A_final = results/firstcross/final F03 models (seed 0)
usage: verify_pvval_og.py MODE CUT SEED [WM]     MODE = parity (no training) | train
"""
import ast, gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc

MODE = sys.argv[1]; CUT = int(sys.argv[2]); SEED = int(sys.argv[3]); WMX = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
PANELS = [p for p in os.environ.get('VPANELS', ','.join(lab.PANELS)).split(',') if p]
TAG = f'{MODE}_c{CUT}_s{SEED}_w{WMX:g}' + ('_smoke' if SMOKE else '')
OUT = f'{lab.ROOT}/analysis/t2lab/results/verify_pvval'
os.makedirs(OUT, exist_ok=True)
KFILE = f'{lab.ROOT}/kaggle_jobs/t2_ongoing_pv.py'
T00 = time.time()

# ------------------------------------------------------------------ compile the kernel's functions from the kernel file
WANT_F = {'log', 'fill_time', 'established', 'iou', 'shift_links', 'ffill_time', 'dist_to_queued', 'window_features', 'add_ramp',
          'prior_table', 'prior_feats', 'to_matrix', 'ongoing_keep', 'decode', 'march_faithful', 'scenario', 'blend_prior_og'}
WANT_A = {'H', 'F', 'SPACING', 'FILL_GAP', 'PERS_MAX', 'W_RS', 'W_DENSE', 'ONGOING', 'PARAMS', 'PRIOR_COLS', 'ID_DROP', 'WM', 'K_BLEND'}
tree = ast.parse(open(KFILE).read())
body = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in WANT_F:
        body.append(n)
    elif isinstance(n, ast.Assign):
        nm = set()
        for tg in n.targets:
            if isinstance(tg, ast.Name):
                nm.add(tg.id)
            elif isinstance(tg, ast.Tuple):
                nm |= {e.id for e in tg.elts if isinstance(e, ast.Name)}
        if nm and nm <= WANT_A:
            body.append(n)
K = {'np': np, 'pd': pd, 'gc': gc, 'time': time, 'json': json, 'os': os, 'T0': time.time()}
exec(compile(ast.Module(body=body, type_ignores=[]), KFILE, 'exec'), K)
got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
assert got == WANT_F, WANT_F - got
H, F, ONGOING = K['H'], K['F'], K['ONGOING']
assert (H, F, K['W_RS'], K['W_DENSE'], K['WM'], K['K_BLEND'], K['FILL_GAP'], K['PERS_MAX']) == (12, 6, 1.0, 0.3, 5.0, 4, 36, 0.9)
assert ONGOING == lab.ONGOING
COLS = {'full': lab.all_cols(), 'np': lab.colset('np'), 'id': lab.colset('id')}
IX = {k: [COLS['full'].index(c) for c in v] for k, v in COLS.items()}
assert COLS['np'] == [c for c in COLS['full'] if c not in K['PRIOR_COLS']]
assert COLS['id'] == [c for c in COLS['full'] if c not in K['ID_DROP']]

CUR = {}
K['masked_month'] = lambda panel, split, li: (CUR[split]['days'], CUR[split]['V'], CUR[split]['Fq'], CUR[split]['Em'])
K['ramp_cube'] = lambda panel, split, li, days: (CUR[split]['ON'], CUR[split]['OFF'])


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


log('kernel functions compiled from', KFILE, sorted(got), '| MODE', MODE, 'CUT', CUT, 'SEED', SEED, 'WM', WMX, 'panels', len(PANELS))


# ------------------------------------------------------------------ pseudo-March data (kernel recipe)
def onset_map(V, Em, Q):
    E = np.isfinite(V); blank = (Em.sum(2) >= 5) & (E.sum(2) == 0)
    D = V.shape[0]; on = np.zeros((D, 288), bool)
    for d in range(D):
        for t in range(H, 288 - F):
            if blank[d, t - H:t + F + 1].any() or Em[d, t - H:t].mean() < 0.7:
                continue
            if not K['established'](Q[d, t - H:t]):
                on[d, t] = Q[d, t + 1:t + 1 + F].any()
    return on


def greedy_restart(on, og, quota=5, spacing=72):
    """own implementation: interleaved chronological scan from 00:00 of every day; a pick needs >= spacing slots to the previous
    pick; candidates of a condition whose quota is full are skipped. Returns the union of ongoing picks [(d, t)]."""
    D = on.shape[0]
    g = np.r_[np.flatnonzero(on.ravel()), np.flatnonzero(og.ravel())]
    isog = np.r_[np.zeros(int(on.sum()), bool), np.ones(int(og.sum()), bool)]
    o = np.argsort(g, kind='stable'); g, isog = g[o], isog[o]
    picks = set()
    for d0 in range(D):
        n_on = n_og = 0; last = -10 ** 9
        for gg, io in zip(g[g >= d0 * 288].tolist(), isog[g >= d0 * 288].tolist()):
            if gg - last < spacing:
                continue
            if io and n_og < quota:
                picks.add(gg); n_og += 1; last = gg
            elif (not io) and n_on < quota:
                n_on += 1; last = gg
            if n_on >= quota and n_og >= quota:
                break
    return sorted((int(x // 288), int(x % 288)) for x in picks)


def kernel_rows(Pk, M, Qm, Emv, dpos, cQ, cE, d, t):
    """kernel main() lines 671-683 for windows (d, t) of the masked March month (masked history + slot T, visible labels)."""
    idx = t[:, None] + np.arange(-H, 0)[None]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = K['window_features'](Pk, M['V'][d[:, None], idx], M['Fq'][d[:, None], idx], t, M['dow'][d], Pk['prior_all'], M['V'][d, t])
        K['add_ramp'](Fe, M['ON'], M['OFF'], d, t, Pk['cap'])
    W_, L_ = qlast.shape
    R = dict(d=d, t=t, cond=np.array([ONGOING] * W_), qlast=qlast, W=W_, L=L_, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W_, L_), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = K['prior_feats'](Pk['prior_all'], M['dow'][d], t, h)
        R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))
    assert set(R['rows'][0][1]) == set(COLS['full']), set(R['rows'][0][1]) ^ set(COLS['full'])
    X, y, e, meta, _ = K['to_matrix'](R, ONGOING, COLS['full'], K['ongoing_keep'](R))
    jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
    Xb, a0 = K['blend_prior_og'](X, meta, Pk, jd, cls, t, cQ, cE, COLS['full'])
    return R, X, Xb, y, e, meta, a0


def build_panel(panel):
    P = lab.load_panel(panel)
    Pk = dict(panel=panel, li=P['li'], cut=P['cut'], cap=P['cap'], lanes=P['lanes'], length=P['length'], dow=P['dow'], E=P['E'], Q=P['Q'])
    Pk['prior_all'] = K['prior_table'](Pk, np.ones(len(P['days']), bool))
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    L = len(P['links'])
    CUR['validation'] = dict(days=[str(x) for x in days[o]], V=z['validation_Vm'][o].astype(np.float32), Fq=z['validation_Fm'][o].astype(np.float32),
                             Em=z['validation_Em'][o], ON=z['validation_ON'][o].astype(np.float32), OFF=z['validation_OFF'][o].astype(np.float32))
    CUR['private'] = dict(days=[], V=np.zeros((0, 288, L), np.float32), Fq=np.zeros((0, 288, L), np.float32), Em=np.zeros((0, 288, L), bool))
    M, dpos, cQ, cE = K['scenario'](Pk)
    cand, Qm, Emv = K['march_faithful'](M, Pk['cut'])
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    wl = sorted([(int(a), int(b), K['W_RS']) for a, b in zip(*np.where(rs))] + [(int(a), int(b), K['W_DENSE']) for a, b in zip(*np.where(rest))])
    wl = [w for w in wl if w[0] < CUT]
    out = dict(panel=panel, n_train_windows=len(wl), n_train_rs=sum(1 for w in wl if w[2] == 1.0))
    if wl:
        d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl]); ww = np.array([w[2] for w in wl], np.float32)
        R, X, Xb, y, e, meta, a0 = kernel_rows(Pk, M, Qm, Emv, dpos, cQ, cE, d, t)
        out['TR'] = dict(X=X[e], Xb=Xb[e], y=y[e].astype(np.float32), w=(ww[meta[:, 0]] * WMX)[e], dmax=int(d.max()))
        out['a_train'] = a0
        del R, X, Xb
    on = onset_map(M['V'], M['Em'], Qm)
    sets = {'offlike_m': [w for w in greedy_restart(on, cand) if w[0] >= CUT],
            'runstart_m': sorted((int(a), int(b)) for a, b in zip(*np.where(rs)) if a >= CUT)}
    out['EV'] = {}
    for sname, wl2 in sets.items():
        if not wl2:
            continue
        d = np.array([w[0] for w in wl2]); t = np.array([w[1] for w in wl2])
        R, X, Xb, y, e, meta, a0 = kernel_rows(Pk, M, Qm, Emv, dpos, cQ, cE, d, t)
        truth = np.stack([Qm[d, t + h] for h in range(1, F + 1)], 1); ev = np.stack([Emv[d, t + h] for h in range(1, F + 1)], 1)
        out['EV'][sname] = dict(X=X, Xb=Xb, meta=meta, W=R['W'], L=R['L'], d=d, t=t, truth=truth, ev=ev, a=a0)
    log(panel, 'March train windows (days <', CUT, ')', out['n_train_windows'], 'rs', out['n_train_rs'],
        'rows', 0 if 'TR' not in out else len(out['TR']['y']), '| eval', {k: len(v['d']) for k, v in out['EV'].items()},
        '| mean a train %.3f' % out.get('a_train', float('nan')))
    lab.drop_panel(panel); CUR.clear(); gc.collect()
    return out


def p6(ev, p):
    o = np.zeros((ev['W'], F, ev['L']), np.float32); o[ev['meta'][:, 0], ev['meta'][:, 2] - 1, ev['meta'][:, 1]] = p
    return o


def predict(mdl, cs, ev):
    X = ev['Xb'] if cs == 'full' else np.ascontiguousarray(ev['X'][:, IX[cs]])
    return p6(ev, mdl.predict(X, num_threads=lab.THREADS))


def score(ev, P6):
    r = np.zeros(ev['W'])
    for i in range(ev['W']):
        pred = K['decode'](P6[i].ravel()).reshape(F, -1)
        T = ev['truth'][i] & ev['ev'][i]; Pp = pred & ev['ev'][i]
        u = (T | Pp).sum(); r[i] = 1.0 if u == 0 else (T & Pp).sum() / u
    return r


def agg(df):
    pm = df.groupby(['family', 'panel']).iou.mean()
    return float(pm.groupby(level=0).mean().mean())


def boot(dfa, dfb, n=2000, seed=0):
    m = dfa.merge(dfb, on=['set', 'panel', 'family', 'd', 't'], suffixes=('_a', '_b'))
    g = [(f, (x.iou_b - x.iou_a).to_numpy()) for (f, p), x in m.groupby(['family', 'panel'])]
    fams = sorted({f for f, _ in g}); rng = np.random.default_rng(seed); ds = np.zeros(n)
    for b in range(n):
        acc = {f: [] for f in fams}
        for f, dv in g:
            acc[f].append(dv[rng.integers(0, len(dv), len(dv))].mean())
        ds[b] = np.mean([np.mean(v) for v in acc.values()])
    obs = np.mean([np.mean([dv.mean() for f, dv in g if f == ff]) for ff in fams])
    per = {p: float((x.iou_b - x.iou_a).mean()) for (f, p), x in m.groupby(['family', 'panel'])}
    return dict(n=len(m), diff=float(obs), lo=float(np.quantile(ds, 0.025)), hi=float(np.quantile(ds, 0.975)),
                pos=sum(v > 1e-9 for v in per.values()), neg=sum(v < -1e-9 for v in per.values()), per=per)


def report(S, pairs, label):
    log(f'== {label} ==')
    for sname in ('offlike_m', 'runstart_m'):
        g = S[S.set == sname]
        arms = list(dict.fromkeys(g.arm))
        log(sname, 'n', int((g.arm == arms[0]).sum()), ' '.join(f'{a} {agg(g[g.arm == a]):.4f}' for a in arms))
        for a, b in pairs:
            if a in arms and b in arms:
                c = boot(g[g.arm == a][['set', 'panel', 'family', 'd', 't', 'iou']], g[g.arm == b][['set', 'panel', 'family', 'd', 't', 'iou']])
                best = max(c['per'], key=c['per'].get)
                gg = g[g.panel != best]
                c2 = boot(gg[gg.arm == a][['set', 'panel', 'family', 'd', 't', 'iou']], gg[gg.arm == b][['set', 'panel', 'family', 'd', 't', 'iou']])
                log(f'  {sname} {b} - {a}: {c["diff"]:+.4f} [{c["lo"]:+.4f}, {c["hi"]:+.4f}] panels +{c["pos"]}/-{c["neg"]} n={c["n"]}'
                    f' | drop best panel {best} ({c["per"][best]:+.4f}): {c2["diff"]:+.4f} [{c2["lo"]:+.4f}, {c2["hi"]:+.4f}]'
                    f' | per panel ' + ' '.join(f'{k}:{v:+.3f}' for k, v in c['per'].items()))


# ------------------------------------------------------------------ main
data = [build_panel(p) for p in PANELS]
log('data built', f'{time.time() - T00:.0f}s')
MF = f'{fc.RES}/final'
A = {cs: lgb.Booster(model_file=f'{MF}/model_F03_{cs}255_500.txt') for cs in ('np', 'id', 'full')}
for cs in ('np', 'id', 'full'):
    assert A[cs].feature_name() == COLS[cs], cs
probs = {}
for z in data:
    for sname, ev in z['EV'].items():
        for cs in ('np', 'id', 'full'):
            probs[(sname, z['panel'], f'Af_{cs}')] = predict(A[cs], cs, ev)


def score_all(arms):
    rows = []
    for z in data:
        fam = z['panel'].rsplit('_', 1)[0]
        for sname, ev in z['EV'].items():
            for an, w in arms.items():
                if not all((sname, z['panel'], m) in probs for m in w):
                    continue
                p = sum(wt * probs[(sname, z['panel'], m)] for m, wt in w.items())
                r = score(ev, p)
                rows.append(pd.DataFrame(dict(arm=an, set=sname, panel=z['panel'], family=fam, d=ev['d'], t=ev['t'], iou=r)))
    return pd.concat(rows, ignore_index=True)


ARMS = {'A': dict(Af_np=0.5, Af_id=0.5), 'A_s4': dict(Af_np=0.25, Af_id=0.25, Af_full=0.5)}
S = score_all(ARMS)
# parity with the claimed per-window table (same final models, independently built eval rows)
try:
    C = pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/pvval/pm_scores.pkl')
    for mine, theirs in (('A', 'R:np+id'), ('A_s4', 'R:npid+full_s4')):
        a = S[S.arm == mine]; b = C[(C.arm == theirs) & (C.d >= CUT) & C.panel.isin(PANELS)]
        m = a.merge(b[['set', 'panel', 'd', 't', 'iou']], on=['set', 'panel', 'd', 't'], how='outer', suffixes=('_v', '_c'), indicator=True)
        both = m[m._merge == 'both']
        log(f'PARITY {mine} vs claimed {theirs}: windows mine {len(a)} claimed {len(b)} matched {len(both)} only-mine {int((m._merge == "left_only").sum())}'
            f' only-claimed {int((m._merge == "right_only").sum())} | max |iou diff| {float((both.iou_v - both.iou_c).abs().max()) if len(both) else float("nan"):.2e}'
            f' | n diff>1e-6 {int(((both.iou_v - both.iou_c).abs() > 1e-6).sum())}')
except Exception as ex:
    log('PARITY skipped', type(ex).__name__, str(ex)[:200])
report(S, [('A', 'A_s4')], 'final models only')
S.to_pickle(f'{OUT}/scores_{TAG}.pkl')
if MODE == 'parity':
    log('DONE parity'); sys.exit(0)

# ------------------------------------------------------------------ training: B (F03 + March x WM) and A1 (F03 only), seed SEED
TRm = [z['TR'] for z in data if 'TR' in z]
Xm = np.concatenate([x['X'] for x in TRm]); Xmb = np.concatenate([x['Xb'] for x in TRm])
ym = np.concatenate([x['y'] for x in TRm]); wm = np.concatenate([x['w'] for x in TRm])
assert max(x['dmax'] for x in TRm) < CUT
for z in data:
    z.pop('TR', None)
gc.collect()
sp = fc.ARMS['F03']
Xtr, ytr, wtr, info = fc.train_matrix(sp['train'], 4, COLS['full'], sp['wfn'], panels=PANELS)
log(f'F03 rows {len(ytr)} (weight {wtr.sum():.0f}, pos {ytr.mean():.4f}) | March rows {len(ym)} (weight {wm.sum():.0f}, pos {ym.mean():.4f})',
    f'{time.time() - T00:.0f}s')
prm = lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40, seed=SEED)
n0 = len(ytr)
for name, cs, plus in (('npB', 'np', True), ('idB', 'id', True), ('fullB', 'full', True), ('npA1', 'np', False), ('idA1', 'id', False)):
    t1 = time.time(); ix = IX[cs]; n = n0 + (len(ym) if plus else 0)
    X = np.empty((n, len(ix)), np.float32)
    X[:n0] = Xtr if cs == 'full' else Xtr[:, ix]
    if plus:
        X[n0:] = Xmb if cs == 'full' else Xm[:, ix]
    y = np.concatenate([ytr, ym]) if plus else ytr; w = np.concatenate([wtr, wm]) if plus else wtr
    ds = lgb.Dataset(X, y, weight=w, feature_name=list(COLS[cs]), free_raw_data=True)
    mdl = lgb.train(prm, ds, NT)
    del X, ds; gc.collect()
    if not SMOKE:
        mdl.save_model(f'{OUT}/model_{TAG}_{name}.txt')
    for z in data:
        for sname, ev in z['EV'].items():
            probs[(sname, z['panel'], name)] = predict(mdl, cs, ev)
    log(f'model {name}: {n} rows, {NT} trees, {time.time() - t1:.0f}s')
    pd.to_pickle({k: v.astype(np.float16) for k, v in probs.items()}, f'{OUT}/probs_{TAG}.pkl')
    if name == 'fullB':
        ARMS.update({'B': dict(npB=0.25, idB=0.25, fullB=0.5), 'B_npid': dict(npB=0.5, idB=0.5), 'Af_npid+fullB': dict(Af_np=0.25, Af_id=0.25, fullB=0.5)})
        S = score_all(ARMS); S.to_pickle(f'{OUT}/scores_{TAG}.pkl')
        report(S, [('A', 'B'), ('A_s4', 'B'), ('A', 'B_npid'), ('A_s4', 'Af_npid+fullB')], f'B (seed {SEED}) vs final models')
ARMS.update({'A1': dict(npA1=0.5, idA1=0.5), 'A1_s4': dict(npA1=0.25, idA1=0.25, Af_full=0.5)})
S = score_all(ARMS); S.to_pickle(f'{OUT}/scores_{TAG}.pkl')
report(S, [('A', 'A1'), ('A1', 'B'), ('A1', 'B_npid'), ('A1_s4', 'B'), ('A', 'B'), ('A_s4', 'B')], f'matched seed {SEED}: B vs A1')
log('DONE', f'{time.time() - T00:.0f}s')
