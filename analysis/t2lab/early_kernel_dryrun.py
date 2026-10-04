"""early track (light): DRY RUN of the new code paths of kaggle_jobs/t2_early_onset.py on local data.
The kernel's own source is executed (function defs via ast; the inference block between two anchor lines via exec), with
  * masked_cube() replaced by a reader of the local npz (same arrays: visible eligible speeds, days sorted),
  * the feature rows of the official windows taken from lab.split_rows (ramp columns dropped = the kernel's v13 columns),
  * cat_on replaced by a recorder (captures the matrix of every variant).
Checks: (1) sp4 prior columns == lab sp4 design; (2) mix* prior columns == lab mixture design on VALIDATION windows;
(3) mix* == sp4 on PRIVATE windows (PRIVATE_SAME_DESIGN False); (4) base untouched; (5) no evidence day >= the window's day."""
import ast, sys, textwrap
import numpy as np, pandas as pd
from scipy.special import betaln
import lab, early_lib as el, early_engine as en

KF = f'{lab.ROOT}/kaggle_jobs/t2_early_onset.py'
src = open(KF).read(); tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)] + \
       [n for n in tree.body if isinstance(n, ast.Assign) and {getattr(t, 'id', '') for t in n.targets} & {'KS', 'MIX', 'PRIVATE_SAME_DESIGN', 'PRIVATE_K', 'VARS', 'H', 'F', 'SPACING'}]
keep = [n for n in keep if not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple))] + \
       [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and {e.id for e in n.targets[0].elts} == {'H', 'F', 'SPACING'}]
ns = dict(np=np, pd=pd, betaln=betaln, log=lambda *a: None, REL='x', SMOKE=False)
exec(compile(ast.Module(body=keep, type_ignores=[]), KF, 'exec'), ns)
ns['log'] = lambda *a: None                                   # the kernel's log() needs its T0 / time globals
a0 = src.index("        # same-scenario prior per onset row"); a1 = src.index("        ison = R['cond'] == 'queue_onset'")
BLOCK = compile(textwrap.dedent(src[a0:a1]), KF + ':inference-block', 'exec')
print('kernel constants: KS', ns['KS'], 'MIX', ns['MIX'], 'PRIVATE_SAME_DESIGN', ns['PRIVATE_SAME_DESIGN'], 'VARS', ns['VARS'])
NORAMP = [c for c in el.COLS if not c.startswith('ramp_')]
worst = dict(sp=0.0, mix=0.0, priv=0.0, base=0.0); nmoved = {}


class Rec:
    def __init__(self):
        self.X = []

    def predict_proba(self, X):
        self.X.append(X.copy()); return np.zeros((len(X), 2))


for panel in lab.PANELS:
    P = lab.load_panel(panel)
    z = np.load(P['npz'], allow_pickle=False)
    P['prior_all'] = lab.prior_table(P, np.ones(len(P['days']), bool))

    def masked_cube(panel_, split, li, z=z):
        days = np.array([str(x) for x in z[f'{split}_days']]); o = np.argsort(days)
        return list(days[o]), z[f'{split}_Vm'][o].astype(np.float32)
    ns['masked_cube'] = masked_cube
    dpos, cQ, cE, lay = ns['scenario_cums'](P)
    for split in ('validation', 'private'):
        R = lab.split_rows(P, split, P['prior_all'])
        wi = R['wi']
        Xc, _, _, meta, cols = lab.to_matrix(R, lab.ONSET, NORAMP, None)
        org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
        tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy(); dow = org.dt.dayofweek.to_numpy()
        rec = Rec()
        g = dict(ns, org=org, dpos=dpos, dow=dow, tod=tod, wv=meta[:, 0], lv=meta[:, 1], P=P, cQ=cQ, cE=cE, Xc=Xc, meta=meta,
                 JP=[cols.index(c) for c in el.PRIOR3], split=split, lay=lay, W=R['W'], L=R['L'], cat_on=rec, R=R)
        exec(BLOCK, g)
        names = list(g['variants']); cap = dict(zip(names, rec.X)); JPk = g['JP']
        assert names == ns['VARS'] and len(rec.X) == len(names)
        worst['base'] = max(worst['base'], float(np.abs(cap['base'] - Xc).max()))
        # lab reference on the same windows
        on = np.where(R['cond'] == lab.ONSET)[0]; L = R['L']
        dd = np.array([dpos[x] for x in wi.date.to_numpy()[on]]); assert np.array_equal(dd, g['jd'][on])
        days = sorted(dpos, key=dpos.get); wk = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy() >= 5
        Vall = np.concatenate([masked_cube(panel, 'validation', None)[1], masked_cube(panel, 'private', None)[1]])
        E = np.isfinite(Vall); Q = E & (np.nan_to_num(Vall, nan=1e9) <= P['cut'][None, None, :])
        b = dict(panel=panel, L=L, D=len(days), W=len(on), wk=wk, Qv=Q, Ev=E, prior_tr=P['prior_all'], d=dd, t=R['t'][on],
                 X=np.zeros((len(on), L, 1), np.float32))
        items, _ = en.items_recent(b, 999)
        base3 = Xc[:, JPk].reshape(len(on), L, 3)
        ref = {'sp4': el.design_priors(base3, el.item_stats(b, items, (-2, 2)), (-2, 2), ref='base', k=4, cw=0.0, ncount='obs')}
        for nm, cfg in ns['MIX'].items():
            box = (-cfg['W'], cfg['W'])
            ref[nm] = el.design_priors(base3, el.item_stats(b, items, box), box, form='bma', pi=cfg['pi'], mu=cfg['mu'], nu=cfg['nu'], cw=1.0)
        got = {nm: cap[nm][:, JPk].reshape(len(on), L, 3) for nm in names}
        worst['sp'] = max(worst['sp'], float(np.abs(got['sp4'] - ref['sp4']).max()))
        for nm in ns['MIX']:
            if split == 'validation':
                worst['mix'] = max(worst['mix'], float(np.abs(got[nm] - ref[nm]).max()))
            else:
                worst['priv'] = max(worst['priv'], float(np.abs(got[nm] - got['sp4']).max()))
            nmoved[(split, nm)] = nmoved.get((split, nm), 0) + int((np.abs(got[nm][..., 0] - base3[..., 0]) > 0.05).sum())
        nmoved[(split, 'sp4')] = nmoved.get((split, 'sp4'), 0) + int((np.abs(got['sp4'][..., 0] - base3[..., 0]) > 0.05).sum())
        other = [c for c in range(Xc.shape[1]) if c not in JPk]
        assert all(np.array_equal(cap[nm][:, other], Xc[:, other], equal_nan=True) for nm in names), 'non-prior columns changed'
        print(panel, split, 'onset windows', len(on), 'evidence days per window', dd.tolist(), 'kernel moved stats', g['moved'])
    lab.drop_panel(panel)
print('\nmax abs differences: base vs untouched', worst['base'], '| sp4 kernel vs lab', f"{worst['sp']:.2e}", '| mix kernel vs lab (validation)', f"{worst['mix']:.2e}",
      '| mix vs sp4 (private)', worst['priv'])
print('prior cells moved by > 0.05:', nmoved)
assert worst['base'] == 0 and worst['sp'] < 1e-5 and worst['mix'] < 1e-5 and worst['priv'] == 0
print('DRY RUN OK')
