"""VERIFY ogclock (light, no training): robustness of the track's saved results.
(a) 4-fold OOF ck vs reference: per-fold deltas, per-panel deltas, drop-the-best-panel aggregate + bootstrap (offlike / official / pseudo);
(b) pseudo-March (k = 4 models, masked validation month): same for np / id / blend, and the overlap with the in-flight g72 F10 switch."""
import numpy as np, pandas as pd
import lab
import exp_firstcross_lib as fc
import ogclock_lib as ck

pd.set_option('display.width', 250)


def pair(A, B):
    a, b = fc.paired(A, B)
    return a, b


def drop_best(a, b, label):
    pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean(); dp = (pb - pa).sort_values()
    best = dp.index[-1]
    bt_all = lab.paired_bootstrap(a, b)
    a2, b2 = a[a.panel != best], b[b.panel != best]
    bt = lab.paired_bootstrap(a2, b2)
    lab.log(f'{label}: all {bt_all["diff"]:+.4f} [{bt_all["lo"]:+.4f}, {bt_all["hi"]:+.4f}] n {bt_all["n_windows"]} | best panel {best} '
            f'{dp.iloc[-1]:+.4f} -> without it {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] | per-panel ' +
            ' '.join(f'{p}:{v:+.3f}' for p, v in dp.items()))


for arm, ref in (('ck_np_all', 'fc_F03_np255'), ('ck_id_all', 'fc_F03_id255')):
    B = ck.load_arm(arm); A = lab.load_arm(ref)
    for s in ('offlike', 'official', 'pseudo'):
        a, b = pair(A[A.set == s], B[B.set == s])
        drop_best(a, b, f'OOF {arm} {s}')
        if s == 'offlike':
            pf = []
            for k in range(4):
                ak, bk = a[a.fold == k], b[b.fold == k]
                pf.append(f'f{k} {lab.aggregate(bk) - lab.aggregate(ak):+.4f} (ref {lab.aggregate(ak):.4f})')
            lab.log(f'OOF {arm} offlike per fold: ' + ' | '.join(pf))

R = pd.read_pickle(f'{ck.RES}/pmblend.pkl')
for s in ck.PM_SETS:
    for base, arm in (('base_np', 'ck_np'), ('base_id', 'ck_id'), ('blend_base', 'blend_ck')):
        A = R[(R.set == s) & (R.arm == base)].drop(columns=['g72', 'wtype']); B = R[(R.set == s) & (R.arm == arm)].drop(columns=['g72', 'wtype'])
        a, b = pair(A, B)
        drop_best(a, b, f'PM {arm} vs {base} {s}')
# overlap with the in-flight g72 switch (F10 np on windows whose previous official-like pick is exactly 360 min earlier)
s = 'offlike_m'
for base, arm in (('sw_np_F10', 'sw_np_ck'), ('base_np', 'sw_np_F10'), ('base_np', 'ck_np')):
    for g in (True, False):
        A = R[(R.set == s) & (R.arm == base) & (R.g72 == g)].drop(columns=['g72', 'wtype'])
        B = R[(R.set == s) & (R.arm == arm) & (R.g72 == g)].drop(columns=['g72', 'wtype'])
        a, b = pair(A, B)
        bt = lab.paired_bootstrap(a, b)
        lab.log(f'PM {arm} vs {base} offlike_m g72={g}: {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] n {bt["n_windows"]}')
lab.log('DONE')
