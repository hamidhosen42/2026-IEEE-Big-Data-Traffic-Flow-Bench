"""early track (light): the kernel's mix_dev (extracted from kaggle_jobs/t2_early_onset.py with ast, not re-typed) must give the
same substituted prior columns as the lab design (early_lib.design_priors, form='bma', classes pooled) on the March bundles."""
import ast, sys
import numpy as np
from scipy.special import betaln
import lab, early_lib as el, early_engine as en

KF = f'{lab.ROOT}/kaggle_jobs/t2_early_onset.py'
tree = ast.parse(open(KF).read())
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mix_dev']
cfgs = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'MIX' for t in n.targets)]
ns = dict(np=np, betaln=betaln)
exec(compile(ast.Module(body=fn + cfgs, type_ignores=[]), KF, 'exec'), ns)
mix_dev, MIX = ns['mix_dev'], ns['MIX']
print('kernel variants:', MIX)
worst = 0.0
for panel in lab.PANELS:
    b = el.march_bundle(panel)
    # the kernel's layer construction (scenario_cums): Q, E float32, expected = E * prior_all[class of the day]
    E = b['Ev'].astype(np.float32); Q = b['Qv'].astype(np.float32); wk = b['wk']
    z0 = np.zeros((len(wk), 1, Q.shape[2]), np.float32)
    lay = tuple(np.concatenate([z0, np.cumsum(a, 1, dtype=np.float32)], 1) for a in (Q, E, E * b['prior_tr'][wk.astype(int)]))
    items, _ = en.items_recent(b, 99)
    widx = np.arange(b['W']); base3 = b['X'][widx][..., el.JP]
    for name, cfg in MIX.items():
        box = (-cfg['W'], cfg['W'])
        S = el.item_stats(b, items, box)
        ref = el.design_priors(base3, S, box, form='bma', pi=cfg['pi'], mu=cfg['mu'], nu=cfg['nu'], cw=1.0)
        got = base3.copy()
        for i in widx:
            j, t = int(b['d'][i]), int(b['t'][i])
            if j == 0:
                continue
            for cj, s in (((0, 2), min(t + 6, 287)), ((1,), t)):
                dev = mix_dev(lay, j, s, cfg['W'], cfg['pi'], cfg['mu'], cfg['nu'])
                for c in cj:
                    got[i, :, c] = np.clip(base3[i, :, c] + dev, 0.0, 1.0)
        d = float(np.abs(got - ref).max()); worst = max(worst, d)
        print(panel, name, 'max abs diff', f'{d:.2e}', 'cells moved > 0.05:', int((np.abs(ref - base3) > 0.05).sum()), 'of', ref.size)
print('WORST', f'{worst:.2e}')
assert worst < 1e-4
