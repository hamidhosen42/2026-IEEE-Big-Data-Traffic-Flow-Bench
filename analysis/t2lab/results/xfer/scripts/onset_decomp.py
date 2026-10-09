"""Onset IoU loss decomposition: size (|pred| != |true|) vs location, by panel; train OOF (elig_bag CatBoost, offlike/official)
and pseudo-March (base = train prior, s4 = same-scenario prior blend). Window weights = competition aggregation."""
import sys
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
R = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab/results'
A = pd.read_pickle(f'{R}/allfin/onset_cv.pkl'); A = A[A.arm == 'elig_bag']
M = pd.read_pickle(f'{R}/splitprior/march_onset_variants.pkl'); M = M[M.arm.isin(['base', 's4'])].assign(set=lambda d: 'pmarch_' + d.arm)


def wts(df):
    n = df.groupby('panel').panel.transform('size'); nfam = df.groupby('family').panel.transform('nunique')
    return 1.0 / n / nfam / df.family.nunique()


def decomp(df, lab):
    df = df.copy(); w = wts(df)
    nt, npred = df.n_true.to_numpy(float), df.n_pred.to_numpy(float)
    r = np.where(np.maximum(nt, npred) == 0, 1.0, np.minimum(nt, npred) / np.maximum(np.maximum(nt, npred), 1))
    df['loss'] = 1 - df.iou; df['size_loss'] = 1 - r; df['loc_loss'] = r - df.iou
    df['blind'] = df.n_true == 0
    tot = (w * df.loss).sum()
    out = dict(set=lab, n=len(df), iou=1 - tot, loss=tot, size_loss=(w * df.size_loss).sum(), loc_loss=(w * df.loc_loss).sum(),
               over=(w * df.size_loss * (npred > nt)).sum(), under=(w * df.size_loss * (npred < nt)).sum(),
               zero_overlap=(w * (df.iou == 0)).sum(), loss_zero_overlap=(w * df.loss * (df.iou == 0)).sum(),
               mean_ntrue=nt.mean(), mean_npred=npred.mean(), exact_size=np.mean(nt == npred))
    pan = df.assign(w=w).groupby('panel').apply(lambda g: pd.Series(dict(iou=(g.w * g.iou).sum() / g.w.sum(), loss_share=(g.w * g.loss).sum() / tot,
                                                                        size=(g.w * g.size_loss).sum() / g.w.sum(), loc=(g.w * g.loc_loss).sum() / g.w.sum(),
                                                                        zero=(g.iou == 0).mean())))
    return out, pan


rows = []
for s in ('official', 'offlike'):
    o, p = decomp(A[A.set == s], 'trainOOF_' + s); rows.append(o)
    print(f'--- {s} per panel'); print(p.round(3).to_string())
for s in ('pmarch_base', 'pmarch_s4'):
    o, p = decomp(M[M.set == s], s); rows.append(o)
    print(f'--- {s} per panel'); print(p.round(3).to_string())
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))
