"""TRACK ogclock: parity of the kernel kaggle_jobs/t2_ogclock_v25.py (functions compiled from the kernel file with ast; its main code
never runs) against the lab implementation on local data.
  a. ck_features / ck_compose of the kernel == lab on the cached offlike rows of one panel
  b. ck_train (train rows, thinned day before T-12) == lab fe_train with the same visibility cube
  c. ck_thin: per-day visible share follows the measured regimes
  d. ck_day_cube: a masked-layer day rebuilt from a parquet-like frame (ISO timestamps) == the released cube
  e. ck_split on the official validation / private windows == lab ck_compose + ck_features on the released cubes; data at or
     after T-12 of the day cube never changes a ck value (the window history and slot-T value are used there)"""
import ast, types, glob, os, time, gc, json, sys
import numpy as np, pandas as pd
import lab
import ogclock_lib as ck

KF = f'{lab.ROOT}/kaggle_jobs/t2_ogclock_v25.py'
WANT = {'H', 'F', 'SPACING', 'PANELS', 'CK_LINK', 'CK_CORR', 'CK_ALL', 'CK_REGIMES', 'CK_REGIME_P', 'ALLFIN', 'CKF'}
tree = ast.parse(open(KF).read()); body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        body.append(node)
    elif isinstance(node, ast.Assign):
        tg = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        if tg and tg <= WANT:
            body.append(node)
K = types.ModuleType('k25'); K.__dict__.update(np=np, pd=pd, glob=glob, os=os, time=time, gc=gc, json=json, T0=time.time(), SMOKE=False, REL='/nonexistent')
exec(compile(ast.Module(body=body, type_ignores=[]), KF, 'exec'), K.__dict__)
assert K.CK_ALL == ck.CK_ALL and K.H == lab.H
panel = sys.argv[1] if len(sys.argv) > 1 else 'D7_I405_S'
P = lab.load_panel(panel)
rc = lab.rows(panel, 'offlike', lab.ONGOING)
Vh, Fh, VT, d, t = ck.train_inputs(P, rc.win)
THIN, rates = ck.thin_cube(P)
# a + b
lab_fe = np.asarray(ck.build_fe(panel, 'offlike'))
P2 = dict(P, THIN=THIN)
kt = K.ck_train(P2, d, t, Vh, VT)
kfe = np.stack([kt[c] for c in ck.CK_ALL], -1)
print(f'a/b {panel} offlike: kernel ck_train == lab FE cache: {np.array_equal(kfe, lab_fe, equal_nan=True)} ({len(d)} windows)')
# c
kth = K.ck_thin(panel, len(P['days']), len(P['links']))
dv = np.array([kth[i][P['E'][i]].mean() for i in range(len(P['days']))])
print(f'c ck_thin: overall visible share of eligible {kth[P["E"]].mean():.3f}; per-day shares around 0.507 / 0.705 / 0.803: '
      f'{np.mean(np.abs(dv - 0.507) < 0.03):.2f} / {np.mean(np.abs(dv - 0.705) < 0.03):.2f} / {np.mean(np.abs(dv - 0.803) < 0.03):.2f}')
# d + e
z = np.load(P['npz'], allow_pickle=False)
ok_all = True
for split in ('validation', 'private'):
    days = [str(x) for x in z[f'{split}_days']]
    Vm = z[f'{split}_Vm'].astype(np.float32)
    wi = lab.window_index(panel, split)
    DAY = {}
    for dt_ in sorted(wi.date.unique()):
        C = Vm[days.index(dt_)]
        s_, l_ = np.where(np.isfinite(C))
        ts = (pd.Timestamp(dt_, tz='UTC') + pd.to_timedelta(s_ * 5, unit='min')).strftime('%Y-%m-%dT%H:%M:%S+00:00')
        m = pd.DataFrame(dict(timestamp=ts, link_id=np.array(P['links'])[l_], speed_kmh=C[s_, l_].astype(np.float64), is_score_eligible=1))
        DAY[dt_] = K.ck_day_cube(m, P['li'], len(P['links']))
        same = np.array_equal(DAY[dt_], C, equal_nan=True)
        ok_all &= same
    org = pd.to_datetime(wi.forecast_origin, utc=True); tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
    Vh_ = z[f'{split}_Vh'].astype(np.float32); VT_ = z[f'{split}_VT'].astype(np.float32)
    kc = K.ck_split(P, wi, DAY, tod, Vh_, VT_)
    ref = ck.ck_features(ck.ck_compose(np.stack([Vm[days.index(x)] for x in wi.date]), tod, Vh_, VT_, P['cut']), tod)
    eq = all(np.array_equal(kc[c], ref[c], equal_nan=True) for c in ck.CK_ALL)
    DAY2 = {k: v.copy() for k, v in DAY.items()}
    for i, (dt_, T) in enumerate(zip(wi.date, tod)):
        DAY2[dt_][T - lab.H:] = 1.0                                   # garbage at and after T-12 (all windows of that day)
    kc2 = K.ck_split(P, wi, DAY2, tod, Vh_, VT_)
    # windows sharing a day with a LATER window see that window's garbage before their own T-12 only if it lies < T-12: restrict
    first = np.array([T == tod[(wi.date == dt_).to_numpy()].min() for dt_, T in zip(wi.date, tod)])
    eq2 = all(np.array_equal(kc2[c][first], kc[c][first], equal_nan=True) for c in ck.CK_ALL)
    print(f'd/e {split}: day cubes rebuilt from parquet-like frames == released cube: {ok_all}; ck_split == lab: {eq}; '
          f'data at/after T-12 ignored (earliest window of each day, n={int(first.sum())}): {eq2}; windows {len(wi)}')
lab.log('DONE')
