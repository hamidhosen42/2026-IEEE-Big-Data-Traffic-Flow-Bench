"""FM-PM forecast for base/all2 combinations on the official validation footprint (kernel probabilities fs4occ vs fs4ogf)."""
import sys
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
import lab
DL, OUT = sys.argv[1], sys.argv[2]
from importlib import util
spec = util.spec_from_file_location('fpa', __file__.replace('fm_combo.py', 'fpmatch_lib.py'))
fpa = util.module_from_spec(spec); spec.loader.exec_module(fpa)
occ = np.load(f'{DL}/tfb-t2occ/t2v26occ_probs.npz'); ga = np.load(f'{DL}/tfb-t2ogfa/t2v27a_probs.npz'); gb = np.load(f'{DL}/tfb-t2ogfb/t2v27b_probs.npz')
C = pd.read_pickle(f'{OUT}/ogf_combo_pm.pkl')
res = {}
for split in ('validation', 'private'):
    rows = []
    for panel in lab.PANELS:
        k = f'{panel}__{split}__'
        pa = (occ[k + 'fs_np'].astype(np.float32) + occ[k + 'fs_id'].astype(np.float32) + occ[k + 'fs_np1'].astype(np.float32) + occ[k + 'fs_id1'].astype(np.float32)) / 4
        pb = (((ga[k + 'fs_np'] + ga[k + 'fs_id']) + gb[k + 'fs_np1']) + gb[k + 'fs_id1']) / 4
        ql = occ[k + 'qlast']; wids = occ[k + 'window_id']; L = ql.shape[1]
        wi = lab.window_index(panel, split).set_index('window_id')
        for i, w in enumerate(wids):
            A = lab.decode(pa[i].ravel()).reshape(6, L); B = lab.decode(pb[i].ravel()).reshape(6, L); q = np.repeat(ql[i][None], 6, 0)
            addout = int((B & ~A & ~q).sum())
            cand = {'all2': B, 'veto10': A if addout > 10 else B, 'veto3': A if addout > 3 else B, 'blend0.5': lab.decode((0.5 * pa[i] + 0.5 * pb[i]).ravel()).reshape(6, L),
                    'and': A & B}
            o = pd.Timestamp(wi.loc[w, 'forecast_origin'])
            for a, P in cand.items():
                rows.append(dict(arm=a, window_id=w, panel=panel, tod=o.hour + o.minute / 60, nA=int(A.sum()), nB=int(P.sum()), changed=int((A ^ P).sum())))
    res[split] = pd.DataFrame(rows)
for a in ('all2', 'veto10', 'veto3', 'blend0.5', 'and'):
    L_ = C[['set', 'panel', 'd', 't', 'iou_base', f'iou_{a}', 'n_base', f'n_{a}']].rename(columns={f'iou_{a}': 'iou_b', 'iou_base': 'iou_a', 'n_base': 'n_pred_a', f'n_{a}': 'n_pred_b'})
    L_['dI'] = L_.iou_b - L_.iou_a; L_['dn'] = L_.n_pred_b - L_.n_pred_a; L_['tod'] = L_.t / 12
    out = []
    for split in ('validation', 'private'):
        off = res[split][res[split].arm == a].copy(); off['dn'] = off.nB - off.nA
        f = fpa.forecast(L_, off)
        out.append(f'{split}: windows changed {(off.changed > 0).sum()} cells {off.changed.sum()} | FM-PM dIoU {f.mean():+.4f} -> LB-equiv {0.15 * f.mean():+.5f}')
    pmv = {s: np.mean([g.dI.mean() for _, g in L_[L_.set == s].groupby('panel')]) for s in ('offlike_m', 'runstart_m')}
    print(f'{a:9s} | pmarch offlike_m {pmv["offlike_m"]:+.4f} runstart_m {pmv["runstart_m"]:+.4f} | ' + ' | '.join(out))
