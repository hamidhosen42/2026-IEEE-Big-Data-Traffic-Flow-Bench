"""early track: build the March bundles and check parity with exp_onset_splitprior.py (NORAMP=1): base 0.7140, s4 0.7875, s8 0.8010,
and parity of the fast scorer with lab.score_probs."""
import types
import numpy as np, pandas as pd
import lab, early_lib as el

cat = el.load_cat()
ref = pd.read_pickle('results/splitprior_noramp/march_onset_variants.pkl')
res = {k: [] for k in ('base', 's4', 's8', 'b4')}
maxdiff = 0.0
for panel in lab.PANELS:
    b = el.march_bundle(panel)
    evd = el.Evidence(b)
    widx = np.arange(b['W'])
    items = [(i, np.arange(0, b['d'][i]), evd) for i in widx]
    base3 = b['X'][widx][..., el.JP]
    S5 = el.item_stats(b, items, (-2, 2)); S1 = el.item_stats(b, items, (0, 0))
    for name, (S, box, k) in dict(base=(None, None, None), s4=(S5, (-2, 2), 4), s8=(S5, (-2, 2), 8), b4=(S1, (0, 0), 4)).items():
        p3 = None if S is None else el.design_priors(base3, S, box, ref='base', k=k, cw=0.0, ncount='obs', cw_all=1.0)
        p = el.predict_items(cat, b, widx, p3)
        iou, _ = el.iou_items(b, widx, p)
        res[name].append(el.frame(b, widx, iou))
        r = ref[(ref.arm == name) & (ref.panel == panel)].sort_values(['d', 't'])
        assert len(r) == len(iou), (panel, len(r), len(iou))
        maxdiff = max(maxdiff, float(np.abs(r.iou.to_numpy() - iou).max()))
    lab.log(panel, 'windows', b['W'], 'L', b['L'])
for name, v in res.items():
    df = pd.concat(v, ignore_index=True)
    print(name, 'n', len(df), 'IoU', round(lab.aggregate(df), 4))
print('max abs per-window IoU difference vs exp_onset_splitprior.py results:', maxdiff)
