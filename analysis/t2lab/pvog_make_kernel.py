"""pvog: derive kaggle_jobs/t2_pvog_ongoing.py from the production kernel kaggle_jobs/t2_ongoing_pv.py by exact, asserted
string replacements (every `old` block must occur exactly once), so the diff is exactly the list below.
Usage: pvog_make_kernel.py '<trio>=<WM>,<trio>=<WM>,...'   e.g.  'tr=0,m1=1'   (WM 0 = train rows only)"""
import ast, difflib, sys

ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
SRC = f'{ROOT}/kaggle_jobs/t2_ongoing_pv.py'; DST = f'{ROOT}/kaggle_jobs/t2_pvog_ongoing.py'
trios = dict((a.split('=')[0], float(a.split('=')[1])) for a in sys.argv[1].split(','))
s = open(SRC).read()
REPL = []


def rep(old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    REPL.append((old, new))


rep('''"""Task 2 ONGOING for the PRIVATE month with same-scenario data (ogpv).''',
    '''"""Task 2 ONGOING for the PRIVATE month, track pvog (derived from t2_ongoing_pv.py; see analysis/t2lab/pvog_*.py).
Several model TRIOS (np, id, full) are trained in one run, one per entry of TRIOS = {name: WM}: WM is the weight multiplier of the
released-March faithful windows (masked-layer rows); WM = 0 trains on the train F03 rows ONLY (raw histories, no March rows).
Every trio is applied to the PRIVATE ongoing windows as 0.25 np + 0.25 id + 0.5 full, the full model fed the same-scenario blended
prior (unchanged).  Output: queue_submission_pvog_<name>.csv per trio (all validation + private queue rows; validation ongoing rows
copied from tfb-t2v22 fs4, onset rows 0).  The text below is the original docstring of t2_ongoing_pv.py.

Task 2 ONGOING for the PRIVATE month with same-scenario data (ogpv).''')
rep('''WM = 5.0                                    # weight multiplier of the March windows (lab: 5)
''', f'''TRIOS = {trios!r}          # trio name -> weight multiplier of the March windows (0 = train rows only)
REF_PVO = ('tfb-t2ogpv', 'queue_submission_pvo.csv')   # the March x5 blend already built (optional footprint CONTROL)
''')
rep('''        MR.append(dict(X=X[e], Xb=Xb[e], y=y[e], w=(ww[meta[:, 0]] * WM)[e]))''',
    '''        MR.append(dict(X=X[e], Xb=Xb[e], y=y[e], w=ww[meta[:, 0]][e]))            # unscaled: the trio's WM is applied below''')
rep('''    log(f'rows: F03 {len(ye)} (pos {ye.mean():.4f}) | March {len(ym)} (pos {ym.mean():.4f}, weight sum {wm.sum():.0f} vs F03 {we.sum():.0f})')
    y_all = np.concatenate([ye, ym]); w_all = np.concatenate([we, wm])
    models = {}
    for name, getX in (('npB', lambda: np.concatenate([Xe[:, ix_np], Xm[:, ix_np]])),
                       ('idB', lambda: np.concatenate([Xe[:, ix_id], Xm[:, ix_id]])),
                       ('fullB', lambda: np.concatenate([Xe, Xmb]))):
        t1 = time.time(); X = getX()
        models[name] = lgb.train(P_PV, lgb.Dataset(X, y_all, weight=w_all), N_PV)
        del X; gc.collect()
        log(f'model {name} trained: {len(y_all)} rows, {N_PV} trees, {(time.time() - t1) / 60:.1f} min')
    del Xe, Xm, Xmb; gc.collect()
''', '''    log(f'rows: F03 {len(ye)} (pos {ye.mean():.4f}) | March {len(ym)} (pos {ym.mean():.4f}, unit weight sum {wm.sum():.0f} vs F03 {we.sum():.0f})')
    models = {}; t_model = 45 * 60.0
    for trio, wmv in TRIOS.items():
        use_m = wmv > 0
        y_all = np.concatenate([ye, ym]) if use_m else ye
        w_all = np.concatenate([we, wm * np.float32(wmv)]) if use_m else we
        for name, ix, Xmar in (('np', ix_np, Xm), ('id', ix_id, Xm), ('full', None, Xmb)):
            if time.time() - T0 + t_model > TIME_LIMIT_S - 20 * 60:
                log(f'TIME: model {trio}/{name} skipped (trio {trio} will not be written)'); continue
            t1 = time.time()
            Xa = Xe if ix is None else Xe[:, ix]
            X = np.concatenate([Xa, Xmar if ix is None else Xmar[:, ix]]) if use_m else np.ascontiguousarray(Xa)
            models[(trio, name)] = lgb.train(P_PV, lgb.Dataset(X, y_all, weight=w_all), N_PV)
            del X, Xa; gc.collect()
            t_model = max(t_model, time.time() - t1)
            log(f'model {trio}/{name} trained: {len(y_all)} rows (March rows {"x%g" % wmv if use_m else "not used"}), {N_PV} trees, {(time.time() - t1) / 60:.1f} min')
    del Xe, Xm, Xmb; gc.collect()
    done = [t for t in TRIOS if all((t, n) in models for n in ('np', 'id', 'full'))]
    log('trios complete:', done)
    NAMES = [f'pvog_{t}' for t in done]
''')
rep('''            preds = {n: np.zeros((W, F, L), bool) for n in ('pvo', 'pvo_npid')}''',
    '''            preds = {n: np.zeros((W, F, L), bool) for n in NAMES}''')
rep('''                pn = models['npB'].predict(X[:, ix_np]); pi = models['idB'].predict(X[:, ix_id]); pf = models['fullB'].predict(Xb)
                for name, p in (('pvo', 0.25 * pn + 0.25 * pi + 0.5 * pf), ('pvo_npid', 0.5 * (pn + pi))):
                    P6 = np.zeros((W, F, L), np.float32); P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = p
                    for i in np.where(is_og)[0]:
                        preds[name][i] = decode(P6[i].ravel()).reshape(F, L)
                stats.append(dict(panel=panel, ongoing_windows=int(is_og.sum()), mean_a=round(a0, 3),
                                  size_T=float(R['qlast'][is_og].sum(1).mean()),
                                  size_pvo=float(preds['pvo'][is_og].sum((1, 2)).mean() / F),
                                  size_npid=float(preds['pvo_npid'][is_og].sum((1, 2)).mean() / F)))
                log(json.dumps(stats[-1]))''',
    '''                st = dict(panel=panel, ongoing_windows=int(is_og.sum()), mean_a=round(a0, 3), size_T=float(R['qlast'][is_og].sum(1).mean()))
                for t in done:
                    p = 0.25 * models[(t, 'np')].predict(X[:, ix_np]) + 0.25 * models[(t, 'id')].predict(X[:, ix_id]) + 0.5 * models[(t, 'full')].predict(Xb)
                    P6 = np.zeros((W, F, L), np.float32); P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = p
                    for i in np.where(is_og)[0]:
                        preds[f'pvog_{t}'][i] = decode(P6[i].ravel()).reshape(F, L)
                    st[f'size_{t}'] = float(preds[f'pvog_{t}'][is_og].sum((1, 2)).mean() / F)
                stats.append(st)
                log(json.dumps(stats[-1]))''')
rep('''            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split', 'q_pvo', 'q_pvo_npid']])''',
    '''            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split'] + [f'q_{n}' for n in NAMES]])''')
rep('''    for name in ('pvo', 'pvo_npid'):
        q = np.where(val, S.q_fs4.astype(int), S[f'q_{name}'])''',
    '''    for name in NAMES:
        q = np.where(val, S.q_fs4.astype(int), S[f'q_{name}'])''')
rep('''        for sp_ in ('validation', 'private'):
            mm = m[m.split == sp_]
            a, b = mm.out_pvo.to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1
            log(f'CONTROL chain-base ongoing vs pvo ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')
    except Exception as ex:
        log('CONTROL base skipped:', type(ex).__name__, str(ex)[:200])
    log('OGPV DONE')''',
    '''        for name in NAMES:
            for sp_ in ('validation', 'private'):
                mm = m[m.split == sp_]
                a, b = mm[f'out_{name}'].to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1
                log(f'CONTROL chain-base ongoing vs {name} ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')
    except Exception as ex:
        log('CONTROL base skipped:', type(ex).__name__, str(ex)[:200])
    # CONTROL (optional): footprint against the March x5 blend already built by tfb-t2ogpv (a trio with WM = 5 must reproduce it)
    try:
        pvo = pd.read_csv(glob.glob(f'/kaggle/input/**/{REF_PVO[0]}/**/{REF_PVO[1]}', recursive=True)[0], dtype={'window_id': str, 'link_id': str})
        pvo['ts'] = pd.to_datetime(pvo.timestamp, utc=True)
        m = S.merge(pvo[['window_id', 'ts', 'link_id', 'queue_pred']].rename(columns={'queue_pred': 'q_pvo_ref'}), on=['window_id', 'ts', 'link_id'], how='left')
        for name in NAMES:
            for sp_ in ('validation', 'private'):
                k = (m.is_og & (m.split == sp_) & m.q_pvo_ref.notna()).to_numpy()
                a, b = m[f'out_{name}'].to_numpy()[k].astype(bool), m.q_pvo_ref.to_numpy()[k] == 1
                log(f'CONTROL tfb-t2ogpv pvo vs {name} ({sp_}): agreement {float((a == b).mean()):.6f} jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.4f} rows {int(k.sum())}')
    except Exception as ex:
        log('CONTROL pvo skipped:', type(ex).__name__, str(ex)[:200])
    log('PVOG DONE')''')
out = s
for old, new in REPL:
    out = out.replace(old, new)
ast.parse(out)
assert 'WM)' not in out.split('def main():')[1].replace('TRIOS', ''), 'a stale WM reference is left in main()'
open(DST, 'w').write(out)
d = list(difflib.unified_diff(s.splitlines(), out.splitlines(), 't2_ongoing_pv.py', 't2_pvog_ongoing.py', lineterm='', n=0))
print(f'written {DST}: {len(out.splitlines())} lines; diff lines +{sum(x.startswith("+") and not x.startswith("+++") for x in d)} -{sum(x.startswith("-") and not x.startswith("---") for x in d)}')
