"""Label-free window descriptors (features at or before T only) of ongoing windows:
do the released validation (March) / private (April) official ongoing windows look like the official train windows /
official-like faithful windows (young, growing queues) or like the stable windows of the current pseudo / dense sets?
Light (no training).  Nothing after T of any released window is read: lab.split_rows uses window_history, the masked
slot-T value and ramp flows before T only."""
import json, os, sys
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
COLS = ['share_q', 'share_q15', 'n_q_hist_win', 'q_last', 'q_count', 'q_age', 'vr_last', 'vr_slope', 'vr_l6', 'has_T', 'tod']


def descr(X, cols, wi, li, h, L):
    """per-window descriptors from the h == 1 rows of a production-shaped matrix."""
    m = h == 1
    ix = {c: cols.index(c) for c in COLS}
    Xm = np.asarray(X[np.where(m)[0]][:, [ix[c] for c in COLS]], np.float32)
    df = pd.DataFrame(Xm, columns=COLS); df['w'] = wi[m]
    df['q30'] = (np.nan_to_num(df.vr_l6, nan=9) <= 1.0).astype(np.float32)
    ql = df.q_last > 0.5
    df['qc_q'] = np.where(ql, df.q_count, np.nan)                 # queued observations in the history of links queued now
    df['young_q'] = np.where(ql, (df.q_count <= 3).astype(float), np.nan)       # queued now with <= 3 queued observations in 60 min
    df['old_q'] = np.where(ql, (df.q_count >= 9).astype(float), np.nan)
    df['slope_q'] = np.where(ql, df.vr_slope, np.nan)
    g = df.groupby('w')
    out = pd.DataFrame(dict(share_q=g.share_q.first(), share_q15=g.share_q15.first(), n_q_hist=g.n_q_hist_win.first(),
                            nq_now=g.q_last.sum(), nq30=g.q30.sum(), qc_q=g.qc_q.mean(), young_q=g.young_q.mean(), old_q=g.old_q.mean(),
                            slope_q=g.slope_q.mean(), has_T=g.has_T.mean(), tod=g.tod.first()))
    out['L'] = L
    out['growth15'] = (out.share_q - out.share_q15) * L           # links gained in the last 15 min (last-valid-filled state)
    out['growth30'] = out.nq_now - out.nq30
    out['rel_growth30'] = out.growth30 / np.maximum(out.nq_now, 1)
    return out


rows = []
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    for s in ('official', 'offlike', 'runstart', 'pseudo', 'dense1500'):
        rc = lab.rows(panel, s)
        d = descr(rc.X, rc.cols, rc.wi, rc.li, rc.h, rc.L)
        w = rc.win.loc[d.index]
        d['set'] = s; d['panel'] = panel; d['pers_fill'] = w.pers_fill.to_numpy(); d['faithful'] = w.faithful.to_numpy(); d['runstart'] = w.runstart.to_numpy()
        rows.append(d)
    for split in ('validation', 'private'):
        R = lab.split_rows(P, split)
        X, _, _, meta, cols = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        d = descr(X, list(cols), meta[:, 0], meta[:, 1], meta[:, 2], R['L'])
        d['set'] = split; d['panel'] = panel; d['pers_fill'] = np.nan; d['faithful'] = np.nan; d['runstart'] = np.nan
        rows.append(d)
    lab.drop_panel(panel)
D = pd.concat(rows, ignore_index=True)
D.to_pickle(f'{fc.RES}/descr_windows.pkl')
D['grp'] = D.set
extra = [D[(D.set == 'pseudo') & (D.pers_fill > 0.9)].assign(grp='pseudo_stable'), D[(D.set == 'pseudo') & (D.faithful == True)].assign(grp='pseudo_faithful'),
         D[(D.set == 'dense1500') & (D.pers_fill > 0.9)].assign(grp='dense_stable'), D[(D.set == 'dense1500') & (D.faithful == True)].assign(grp='dense_faithful'),
         D[(D.set == 'offlike') & (D.runstart == True)].assign(grp='offlike_runstart'), D[(D.set == 'offlike') & (D.runstart == False)].assign(grp='offlike_midrun')]
DD = pd.concat([D] + extra, ignore_index=True)
ORDER = ['official', 'validation', 'private', 'offlike', 'offlike_runstart', 'offlike_midrun', 'runstart', 'pseudo', 'pseudo_faithful', 'pseudo_stable',
         'dense1500', 'dense_faithful', 'dense_stable']
VARS = ['share_q', 'nq_now', 'n_q_hist', 'qc_q', 'young_q', 'old_q', 'growth15', 'growth30', 'rel_growth30', 'slope_q', 'has_T', 'tod']


def panel_mean(g, v):          # mean over panels of the per-panel mean (each panel weighs the same, like the metric)
    return g.groupby('panel')[v].mean().mean()


tab = pd.DataFrame({grp: {**{'n': len(g)}, **{v: panel_mean(g, v) for v in VARS}} for grp, g in DD.groupby('grp')}).T.reindex(ORDER)
print('== window descriptors: mean over panels of the per-panel mean =='); print(tab.round(3).to_string())
med = pd.DataFrame({grp: {v: g.groupby('panel')[v].median().mean() for v in VARS} for grp, g in DD.groupby('grp')}).T.reindex(ORDER)
print('\n== mean over panels of the per-panel median =='); print(med.round(3).to_string())
tab.to_csv(f'{fc.RES}/descr_means.csv'); med.to_csv(f'{fc.RES}/descr_medians.csv')
# per-panel view of the most telling descriptors
for v in ('rel_growth30', 'young_q', 'qc_q', 'share_q'):
    t = DD[DD.grp.isin(['official', 'validation', 'private', 'offlike', 'pseudo_stable'])].groupby(['panel', 'grp'])[v].mean().unstack()
    print(f'\n== per panel: {v} =='); print(t[['official', 'validation', 'private', 'offlike', 'pseudo_stable']].round(3).to_string())
# tod of official windows (slot of the day)
print('\nslot-of-day quartiles:'); print(DD.groupby('grp').tod.quantile([0.1, 0.5, 0.9]).unstack().reindex(ORDER).round(0).to_string())
