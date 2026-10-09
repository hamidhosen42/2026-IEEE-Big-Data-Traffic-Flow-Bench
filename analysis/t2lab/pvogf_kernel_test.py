"""pvogf kernel checks (local; no Kaggle path or token is touched; the kernel's /kaggle paths are redirected to a fake input tree).
A  static: ast.parse of t2_ongoing_pvogf.py / _smoke.py; the og_* block in the kernel == ogfeat_lib.py portable block (verbatim, only
   the warnings context replaced); OG_STATIC / OG_STEP == the lab all2 columns; the kernel differs from pvaf only by the generator edits.
B  parity with the lab caches (data/t2local/cache/pvogf, built by pvogf_build.py with the lab harness), kernel functions compiled
   with ast from t2_ongoing_pvogf.py (main() never runs), real lab panels (all 273 train days):
   (1) make_rows + to_matrix on the F03 lab windows == f03 cache rows (bit-identical, 172 columns);
   (2) scenario (local readers) + march_faithful + march_rows + blend_prior_og == March cache rows (X and the blended Xb);
   (3) load_split on the FAKE release (task2 window files written from the same exports) + blend == private cache rows (X, Xb, meta).
C  end-to-end: the SMOKE kernel (2 panels, 90 days, 60 trees) executed on the fake release; outputs / CONTROL lines / veto checked.
Usage: pvogf_kernel_test.py FAKE_ROOT OUT_DIR [panels=D7_I10_E,D12_I5_S] [parts=A,B,C]"""
import ast, gc, glob, io, json, os, re, sys, time, types, warnings, contextlib
import numpy as np, pandas as pd
import lab
import pvogf_lib as G

FAKE, OUTD = sys.argv[1], sys.argv[2]
kw = dict(a.split('=') for a in sys.argv[3:])
PANELS = kw.get('panels', 'D7_I10_E,D12_I5_S').split(',')
PARTS = kw.get('parts', 'A,B,C').split(',')
KF = f'{lab.ROOT}/kaggle_jobs/t2_ongoing_pvogf.py'; KS = f'{lab.ROOT}/kaggle_jobs/t2_ongoing_pvogf_smoke.py'
os.makedirs(OUTD, exist_ok=True)
F, ONGOING = G.F, G.ONGOING
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ A: static
if 'A' in PARTS:
    src = open(KF).read(); ast.parse(src); ast.parse(open(KS).read())
    lib = open(f'{lab.ROOT}/analysis/t2lab/ogfeat_lib.py').read()
    a = lib.index('# ============================================================================================ kernel-portable block')
    b = lib.index('# ============================================================================================ end of portable block')
    blk = lib[a:b].replace("    with warnings.catch_warnings(), np.errstate(all='ignore'):\n        warnings.simplefilter('ignore')\n",
                           "    with np.errstate(all='ignore'):\n").replace('OG_STEP = [', 'OG_STEP_ALL = [')
    assert blk in src, 'feature block differs from ogfeat_lib.py'
    K = G.compile_kernel(KF)
    names = json.load(open(f'{lab.CACHE}/ogfeat/names.json')) + json.load(open(f'{lab.CACHE}/ogfeat/names2.json'))
    assert K.OG_STATIC == names and K.OG_STEP == ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3'] and K.VETO_G == 10
    assert K.N_PV == 900 and K.P_PV['num_leaves'] == 511 and K.WM == 5.0 and K.K_BLEND == 4
    sk = open(KS).read(); assert sk.count('\nSMOKE = True\n') == 1 and sk.replace('\nSMOKE = True\n', '\nSMOKE = False\n') == src
    # the og_* functions compiled from the kernel are the lab's (source identity)
    import inspect, ogfeat_lib as og
    for fn in ('og_ffill', 'og_dist', 'og_close', 'og_rsum', 'og_rmean', 'og_edges', 'og_extra', 'og_extra2', 'og_step'):
        s1 = inspect.getsource(getattr(K, fn)); s2 = inspect.getsource(getattr(og, fn))
        s2 = s2.replace("    with warnings.catch_warnings(), np.errstate(all='ignore'):\n        warnings.simplefilter('ignore')\n", "    with np.errstate(all='ignore'):\n")
        assert s1 == s2, fn
    # functions unchanged from pvaf except the edited ones
    Kp = G.compile_kernel(f'{lab.ROOT}/kaggle_jobs/t2_ongoing_pvaf.py')
    changed = [n for n in dir(Kp) if callable(getattr(Kp, n, None)) and hasattr(K, n) and getattr(getattr(Kp, n), '__code__', None) is not None
               and inspect.getsource(getattr(Kp, n)) != inspect.getsource(getattr(K, n))]
    log('A static OK | feature block verbatim | functions changed vs pvaf:', sorted(changed),
        '| new:', sorted(n for n in dir(K) if callable(getattr(K, n, None)) and not hasattr(Kp, n) and n.startswith(('og_', 'march_rows', 'pvo_rows', 'veto'))))
    assert set(changed) <= {'make_rows', 'load_split'}, changed

# ------------------------------------------------------------------ B: parity with the lab caches
if 'B' in PARTS:
    K = G.compile_kernel(KF)                                  # make_rows / to_matrix / blend (no file access)
    Kr = G.compile_kernel(KF); Kr.REL = f'{FAKE}/kaggle_public'; Kr.log = lambda *a: None      # real kernel readers on the fake release
    cols = G.load_cols()
    for panel in PANELS:
        PK = G.panel_pk(panel)
        # (1) train F03 rows
        Xs, ys = [], []
        for name, wins, ww in G.f03_lab_windows(panel, k=4):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                R = K.make_rows(PK, wins, PK['prior_all'])
                X, y, e, meta, c_ = K.to_matrix(R, ONGOING, None, K.ongoing_keep(R))
            assert list(c_) == cols, 'kernel column order differs from the lab cache'
            Xs.append(X[e]); ys.append(y[e]); del R, X
        Xk = np.concatenate(Xs); Xl = np.load(f'{G.CACHE}/f03_{panel}.npy', mmap_mode='r'); yl = np.load(f'{G.CACHE}/f03_{panel}.npz')['y']
        ok1 = Xk.shape == Xl.shape and np.array_equal(Xk, np.asarray(Xl), equal_nan=True) and np.array_equal(np.concatenate(ys).astype(np.float32), yl)
        log(f'B1 {panel} make_rows: kernel {Xk.shape} vs lab cache {Xl.shape} | identical {ok1}'); assert ok1; del Xk, Xs; gc.collect()
        # (2) released-March rows: kernel scenario / masked_month / ramp_cube on the fake release -> march_faithful -> march_rows -> blend
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            M, dpos, cQ, cE = Kr.scenario(PK)
            cand, Qm, Emv = Kr.march_faithful(M, PK['cut'])
            prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
            rs = cand & ~prev; rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
            wl = sorted([(int(a), int(b), Kr.W_RS) for a, b in zip(*np.where(rs))] + [(int(a), int(b), Kr.W_DENSE) for a, b in zip(*np.where(rest))])
            d = np.array([w[0] for w in wl]); t = np.array([w[1] for w in wl])
            R, X, y, e, meta = Kr.march_rows(PK, M, Qm, Emv, d, t, cols)
            jd = np.array([dpos[M['days'][x]] for x in d]); cls = (M['dow'][d] >= 5).astype(int)
            Xb, _ = Kr.blend_prior_og(X, meta, PK, jd, cls, t, cQ, cE, cols)
        ok2 = np.array_equal(X[e], np.load(f'{G.CACHE}/march_{panel}.npy'), equal_nan=True) and np.array_equal(Xb[e], np.load(f'{G.CACHE}/marchb_{panel}.npy'), equal_nan=True)
        log(f'B2 {panel} scenario + march_rows (fake release): rows {int(e.sum())} | X and Xb identical to the lab cache {ok2}'); assert ok2; del R, X, Xb
        # (3) load_split on the fake release (private) + blend
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wi, tpl, R = Kr.load_split(PK, panel, 'private')
            X, _, _, meta, _ = Kr.to_matrix(R, ONGOING, cols, Kr.ongoing_keep(R))
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
        jd = np.array([dpos[pd.Timestamp(x).strftime('%Y-%m-%d')] for x in org.to_numpy()])
        Xb, _ = Kr.blend_prior_og(X, meta, PK, jd, (dow >= 5).astype(int), tod, cQ, cE, cols)
        z = np.load(f'{G.CACHE}/priv_{panel}.npz')
        ok3 = np.array_equal(meta, z['meta']) and np.array_equal(X, z['X'], equal_nan=True) and np.array_equal(Xb, z['Xb'], equal_nan=True) \
            and np.array_equal(R['qlast'], z['qlast'])
        if not ok3:
            bad = [c for j, c in enumerate(cols) if X.shape == z['X'].shape and not np.array_equal(X[:, j], z['X'][:, j], equal_nan=True)]
            log('B3 differing columns:', bad[:20], X.shape, z['X'].shape)
        log(f'B3 {panel} load_split (fake release, private): rows {len(X)} | X, Xb, meta, qlast identical to the lab cache {ok3}'); assert ok3
        del M, cQ, cE, PK; lab.drop_panel(panel); gc.collect()

# ------------------------------------------------------------------ C: end-to-end SMOKE run on the fake release
if 'C' in PARTS:
    src = open(KS).read()
    assert src.count("'/kaggle/working'") == 1
    src = src.replace('/kaggle/input', FAKE).replace("'/kaggle/working'", repr(OUTD))
    buf = io.StringIO()
    t1 = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, 'pvogf_smoke', 'exec'), {'__name__': '__main__'})
    out = buf.getvalue(); open(f'{OUTD}/smoke_log.txt', 'w').write(out)
    log(f'C smoke kernel finished in {time.time() - t1:.0f}s; log lines {len(out.splitlines())}')
    for l in out.splitlines():
        if re.search(r'pvogf features|model .* trained|CONTROL|VETO|REF pvo|queue rows|PVOGF DONE|vs fs4 on', l):
            print('   ', l)
    assert 'PVOGF DONE' in out
    tpl_rows = sum(len(pd.read_csv(f'{FAKE}/kaggle_public/task2/{p}/{s}/sample_submission_queue.csv')) for p in ('D7_I10_E', 'D12_I5_S') for s in ('validation', 'private'))
    for name in ('pvogf', 'pvogf_npid', 'pvogfv'):
        q = pd.read_csv(f'{OUTD}/queue_submission_{name}.csv', dtype={'window_id': str, 'link_id': str})
        assert list(q.columns) == ['window_id', 'timestamp', 'link_id', 'queue_pred'] and len(q) == tpl_rows and q.queue_pred.isin([0, 1]).all()
        ref = pd.read_csv(f'{FAKE}/tfb-t2v22af/queue_submission_fs4af.csv', dtype={'window_id': str, 'link_id': str})
        m = q.merge(ref, on=['window_id', 'timestamp', 'link_id'], suffixes=('', '_r'))
        wi = pd.concat([pd.read_csv(f'{FAKE}/kaggle_public/task2/{p}/{s}/window_index.csv') for p in ('D7_I10_E', 'D12_I5_S') for s in ('validation', 'private')])
        cond = m.window_id.map(dict(zip(wi.window_id, wi.condition)))
        val_og = m.window_id.str.contains('_validation_') & (cond == ONGOING)
        log(f'C {name}: rows {len(q)} (= template {tpl_rows}) | onset rows all 0: {bool((m.queue_pred[cond != ONGOING] == 0).all())}'
            f' | validation ongoing rows == fs4af reference: {bool((m.queue_pred[val_og] == m.queue_pred_r[val_og]).all())}'
            f' | private ongoing positives {int(m.queue_pred[(cond == ONGOING) & m.window_id.str.contains("_private_")].sum())}')
    pz = np.load(f'{OUTD}/t2pvogf_probs.npz'); vw = pd.read_csv(f'{OUTD}/t2pvogf_veto_windows.csv')
    log('C probs npz keys', len(pz.files), '| veto windows file', vw.shape, 'vetoed', int(vw.veto.sum()))
log('PVOGF KERNEL TEST DONE', PARTS)
