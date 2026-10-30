"""pvogfo kernel checks (local; no Kaggle path or token is touched; the kernel's /kaggle paths are redirected to a fake input tree built by
pvocc_fakerel.py).  KDIR holds t2_ongoing_pvogfo.py + _smoke.py from pvogfo_make_kernel.py (veto=0 or veto=1).
A  static: ast.parse; occ_calib / occ_features == kaggle_jobs/t2_occ_v26.py (verbatim source); load_panel == t2_ogfeat_v27a.py load_panel (= v26
   OCC lines on the pvogf reader); make_rows / load_split == v27a's up to the position of the occupancy update (after the ogfeat step
   columns); og_* block == pvogf; only the expected functions differ from t2_ongoing_pvogf.py; the smoke copy differs only by SMOKE.
B  parity with the lab (kernel functions compiled with ast from the generated kernel; main() never runs):
   (0) kernel load_panel on the fake release (first 90 train days): O cube, Vf / Ff and the occ_calib calibration == the lab inputs;
   (1) make_rows + to_matrix on the F03 lab windows (all 273 days) == pvogf f03 cache (172 columns) + pvocc side-car (13), bit-identical;
   (2) kernel scenario / masked_month(occ) / march_faithful / march_rows / blend on the fake release == pvogf March cache + side-car (X, Xb);
   (3) kernel load_split (fake release, private) + blend == pvogf private cache + side-car (X, Xb, meta, qlast).
C  end-to-end: the SMOKE kernel (2 panels, 90 days, 60 trees) executed on the fake release; outputs / CONTROL / veto checked.
D  LightGBM path: lgb.train(P_PV, Dataset(X, y, w)) == Dataset(params=P_PV).construct() + free + lgb.train (model strings), on smoke-size rows.
Usage: pvogfo_kernel_test.py FAKE_ROOT KDIR OUT_DIR [panels=D7_I10_E,D12_I5_S] [parts=A,B,C,D]"""
import ast, contextlib, gc, glob, inspect, io, json, os, re, sys, time, types, warnings
import numpy as np, pandas as pd
import lab
import pvogf_lib as G
import pvocc_lib as V

FAKE, KDIR, OUTD = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(a.split('=') for a in sys.argv[4:])
PANELS = kw.get('panels', 'D7_I10_E,D12_I5_S').split(',')
PARTS = kw.get('parts', 'A,B,C,D').split(',')
KF, KS = f'{KDIR}/t2_ongoing_pvogfo.py', f'{KDIR}/t2_ongoing_pvogfo_smoke.py'
os.makedirs(OUTD, exist_ok=True)
F, ONGOING = G.F, G.ONGOING
T0 = time.time()
CONSTS = set(G.CONSTS) | {'OCC', 'NB', 'OCC_COLS', 'VETO_O'}


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def compile_k(path, rel=None):
    tree = ast.parse(open(path).read()); body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            body.append(node)
        elif isinstance(node, ast.Assign):
            tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if tg and tg <= CONSTS:
                body.append(node)
    K = types.ModuleType('pvogfo_k')
    K.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, warnings=warnings, T0=time.time(), SMOKE=False)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), K.__dict__)
    if rel:
        K.REL = rel; K.log = lambda *a: None
    return K


def fsrc(path):
    src = open(path).read()
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


# ------------------------------------------------------------------ A: static
if 'A' in PARTS:
    src = open(KF).read(); ast.parse(src); sk = open(KS).read(); ast.parse(sk)
    assert sk.count('\nSMOKE = True\n') == 1 and sk.replace('\nSMOKE = True\n', '\nSMOKE = False\n') == src, 'smoke copy differs beyond SMOKE'
    k, v26, v27, pvg = fsrc(KF), fsrc(f'{lab.ROOT}/kaggle_jobs/t2_occ_v26.py'), fsrc(f'{lab.ROOT}/kaggle_jobs/t2_ogfeat_v27a.py'), fsrc(f'{lab.ROOT}/kaggle_jobs/t2_ongoing_pvogf.py')
    assert k['occ_calib'] == v26['occ_calib'] and k['occ_features'] == v26['occ_features'], 'occupancy feature block not verbatim'
    assert k['load_panel'] == v27['load_panel'] and v27['load_panel'] == v26['load_panel'], 'load_panel differs from v26 / v27a'
    upd = re.compile(r"\n        if occ is not None:[^\n]*\n            feats\.update\(occ\)")
    for fn in ('make_rows', 'load_split'):
        nc = lambda x: re.sub(r'[ \t]+#[^\n]*', '', x)                     # comments differ ('# v27:' vs '# pvogf:'), code must not
        a_, b_ = nc(upd.sub('', k[fn])), nc(upd.sub('', v27[fn]))
        assert a_ == b_, f'{fn}: differs from v27a beyond the occupancy update position'
        assert len(upd.findall(k[fn])) == 1 and k[fn].index('feats.update(occ)') > k[fn].index('og_step(OGX, h)'), f'{fn}: occupancy not after the step columns'
    for fn in [n for n in pvg if n.startswith('og_')]:
        assert k[fn] == pvg[fn], fn
    changed = sorted(n for n in pvg if n in k and k[n] != pvg[n]); new = sorted(n for n in k if n not in pvg)
    log('A static OK | occ_calib / occ_features verbatim v26 | load_panel == v27a | make_rows / load_split == v27a up to the occupancy position'
        f' | og_* block == pvogf | changed vs pvogf: {changed} | new: {new}')
    assert set(changed) <= {'load_panel', 'make_rows', 'load_split', 'masked_month', 'scenario', 'march_rows', 'main', 'pvo_rows'}, changed
    assert set(new) == {'occ_calib', 'occ_features', 'veto_arrays'}, new
    Kc = compile_k(KF)
    assert Kc.OCC is True and Kc.OCC_COLS == V.OCC_COLS and Kc.P_PV['deterministic'] is True and Kc.P_PV['force_row_wise'] is True
    assert Kc.N_PV == 900 and Kc.P_PV['num_leaves'] == 511 and Kc.WM == 5.0 and Kc.K_BLEND == 4 and Kc.VETO_G == 10
    log('A constants: VETO_O', Kc.VETO_O, '| P_PV', {x: Kc.P_PV[x] for x in ('num_leaves', 'learning_rate', 'min_data_in_leaf', 'num_threads', 'deterministic', 'force_row_wise')})

# ------------------------------------------------------------------ B: parity with the lab
if 'B' in PARTS:
    Kn = compile_k(KF); Kr = compile_k(KF, rel=f'{FAKE}/kaggle_public')
    cols = G.load_cols(); C185 = cols + V.OCC_COLS
    for panel in PANELS:
        PK = V.panel_pko(panel); S = V.Side(panel)
        # (0) kernel train reader on the fake release (first 90 days)
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            Pr = Kr.load_panel(panel)
        nd = len(Pr['days']); assert Pr['days'] == PK['days'][:nd]
        oc90, c90, _ = V.calib(PK, np.arange(len(PK['days'])) < nd)
        ok0 = (V.same(Pr['O'], PK['O'][:nd]) and V.same(Pr['Vf'], PK['Vf'][:nd]) and V.same(Pr['Ff'], PK['Ff'][:nd])
               and np.array_equal(Pr['ocrit'], oc90) and np.array_equal(Pr['occ_c'], c90))
        log(f'B0 {panel} kernel load_panel (fake release, {nd} days): O, Vf, Ff, ocrit, occ_c identical to the lab {ok0}'); assert ok0
        del Pr; gc.collect()
        # (1) F03 rows
        Xs = []
        for name, wins, ww in G.f03_lab_windows(panel, k=4):
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                R = Kn.make_rows(PK, wins, PK['prior_all'])
                X, y, e, meta, c_ = Kn.to_matrix(R, ONGOING, None, Kn.ongoing_keep(R))
            assert list(c_) == C185, 'kernel column order differs from the lab (172 + OCC_COLS)'
            Xs.append(X[e]); del R, X
        Xk = np.concatenate(Xs); del Xs
        ok1 = V.same(Xk[:, :172], np.load(f'{V.GC}/f03_{panel}.npy', mmap_mode='r')) and V.same(Xk[:, 172:], S['f03'])
        log(f'B1 {panel} make_rows: kernel {Xk.shape} == pvogf cache + side-car {ok1}'); assert ok1; del Xk; gc.collect()
        # (2) March rows from the fake release
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            M, dpos, cQ, cE = Kr.scenario(PK)
            cand, Qm, Emv = Kr.march_faithful(M, PK['cut'])
            prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
            rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
            wl = sorted([(int(a), int(b), Kr.W_RS) for a, b in zip(*np.where(rs))] + [(int(a), int(b), Kr.W_DENSE) for a, b in zip(*np.where(rest))])
            d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl])
            R, X, y, e, meta = Kr.march_rows(PK, M, Qm, Emv, d, t, C185)
            jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
            Xb, _ = Kr.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, C185)
        Ml = dict(days=M['days'], Vf=M['Vf'])
        okO = V.same(M['O'], V.month_occ(panel, 'validation', Ml))
        ok2 = okO and V.same(X[e][:, :172], np.load(f'{V.GC}/march_{panel}.npy')) and V.same(Xb[e][:, :172], np.load(f'{V.GC}/marchb_{panel}.npy')) \
            and V.same(X[e][:, 172:], S['march']) and V.same(Xb[e][:, 172:], S['march'])
        log(f'B2 {panel} scenario + masked_month(occ) + march_rows (fake release): rows {int(e.sum())} | March O cube == export {okO} | X, Xb == cache + side-car {ok2}')
        assert ok2; del R, X, Xb
        # (3) official private windows from the fake release
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            wi, tpl, R = Kr.load_split(PK, panel, 'private')
            X, _, _, meta, _ = Kr.to_matrix(R, ONGOING, C185, Kr.ongoing_keep(R))
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
        jd = np.array([dpos[pd.Timestamp(x).strftime('%Y-%m-%d')] for x in org.to_numpy()])
        Xb, _ = Kr.blend_prior_og(X, meta, PK, jd, (dow >= 5).astype(int), tod, cQ, cE, C185)
        z = np.load(f'{V.GC}/priv_{panel}.npz')
        ok3 = np.array_equal(meta, z['meta']) and V.same(X[:, :172], z['X']) and V.same(Xb[:, :172], z['Xb']) and np.array_equal(R['qlast'], z['qlast']) \
            and V.same(X[:, 172:], S['priv']) and V.same(Xb[:, 172:], S['priv'])
        log(f'B3 {panel} load_split (fake release, private): rows {len(X)} | X, Xb, meta, qlast == cache + side-car {ok3}'); assert ok3
        del M, cQ, cE, PK; lab.drop_panel(panel); gc.collect()

# ------------------------------------------------------------------ C: end-to-end SMOKE run on the fake release
if 'C' in PARTS:
    src = open(KS).read()
    assert src.count("'/kaggle/working'") == 1
    src = src.replace('/kaggle/input', FAKE).replace("'/kaggle/working'", repr(OUTD))
    buf = io.StringIO(); t1 = time.time()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exec(compile(src, 'pvogfo_smoke', 'exec'), {'__name__': '__main__'})
    out = buf.getvalue(); open(f'{OUTD}/smoke_log.txt', 'w').write(out)
    log(f'C smoke kernel finished in {time.time() - t1:.0f}s; log lines {len(out.splitlines())}')
    for l in out.splitlines():
        if re.search(r'pvogf features|pvogfo:|occupancy|model .* trained|CONTROL|VETO|REF pick|queue rows|PVOGFO DONE|vs fs4 on', l):
            print('   ', l[:260])
    assert 'PVOGFO DONE' in out
    veto_o = 'VETO_O = True' in src
    tpl_rows = sum(len(pd.read_csv(f'{FAKE}/kaggle_public/task2/{p}/{s}/sample_submission_queue.csv')) for p in PANELS for s in ('validation', 'private'))
    wi = pd.concat([pd.read_csv(f'{FAKE}/kaggle_public/task2/{p}/{s}/window_index.csv') for p in PANELS for s in ('validation', 'private')])
    ref = pd.read_csv(f'{FAKE}/tfb-t2v22af/queue_submission_fs4af.csv', dtype={'window_id': str, 'link_id': str})
    outs = ('pvogfo', 'pvogfo_npid') + (('pvogfvo', 'pvoo') if veto_o else ())
    Q = {}
    for name in outs:
        q = pd.read_csv(f'{OUTD}/queue_submission_{name}.csv', dtype={'window_id': str, 'link_id': str}); Q[name] = q
        assert list(q.columns) == ['window_id', 'timestamp', 'link_id', 'queue_pred'] and len(q) == tpl_rows and q.queue_pred.isin([0, 1]).all()
        m = q.merge(ref, on=['window_id', 'timestamp', 'link_id'], suffixes=('', '_r'))
        cond = m.window_id.map(dict(zip(wi.window_id, wi.condition)))
        val_og = m.window_id.str.contains('_validation_') & (cond == ONGOING)
        ok = bool((m.queue_pred[cond != ONGOING] == 0).all()) and bool((m.queue_pred[val_og] == m.queue_pred_r[val_og]).all())
        log(f'C {name}: rows {len(q)} (= template {tpl_rows}) | onset rows 0 and validation ongoing rows == fs4af reference: {ok}'
            f' | private ongoing positives {int(m.queue_pred[(cond == ONGOING) & m.window_id.str.contains("_private_")].sum())}'); assert ok
    pz = np.load(f'{OUTD}/t2pvogfo_probs.npz')
    log('C probs npz keys', len(pz.files), sorted({k.split('__')[-1] for k in pz.files}))
    if veto_o:
        vw = pd.read_csv(f'{OUTD}/t2pvogfo_veto_windows.csv')
        # vO rows == the lab veto rule applied to the kernel's own pvoo / pvogfo rows
        a = Q['pvoo'].queue_pred.to_numpy(); b = Q['pvogfo'].queue_pred.to_numpy(); v = Q['pvogfvo'].queue_pred.to_numpy()
        wid = Q['pvogfo'].window_id.to_numpy(); vet = set(vw.window_id[vw.veto])
        exp = np.where(np.isin(wid, list(vet)), a, b)
        log(f'C veto file {vw.shape}, vetoed {int(vw.veto.sum())} | pvogfvo == (pvoo on vetoed windows, else pvogfo): {bool(np.array_equal(v, exp))}')
        assert np.array_equal(v, exp)

# ------------------------------------------------------------------ D: LightGBM fit path (memory pattern) leaves the model unchanged
if 'D' in PARTS:
    import lightgbm as lgb
    Kc = compile_k(KF)
    z = np.load(f'{V.GC}/f03_D7_I10_W.npz'); Xd = np.asarray(np.load(f'{V.GC}/f03_D7_I10_W.npy', mmap_mode='r')[:60000]); yd = z['y'][:60000]; wd = z['w'][:60000]
    p = dict(Kc.P_PV, num_threads=4)
    m1 = lgb.train(p, lgb.Dataset(Xd.copy(), yd, weight=wd), 30).model_to_string()
    X2 = Xd.copy(); ds = lgb.Dataset(X2, yd, weight=wd, params=p); ds.construct(); del X2; gc.collect()
    m2 = lgb.train(p, ds, 30).model_to_string()
    log(f'D LightGBM: plain lgb.train vs construct-then-free path (P_PV with deterministic / force_row_wise, 30 trees, 60k rows): model strings identical {m1 == m2}')
    assert m1 == m2
log('PVOGFO KERNEL TEST DONE', PARTS)
