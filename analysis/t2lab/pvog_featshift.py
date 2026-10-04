"""pvog (light): feature regime of the ongoing rows in (i) raw F03 train rows, (ii) emulated masked F03 train rows,
(iii) the kernel-style March training rows (masked layer), (iv) the released official ongoing windows (raw window_history,
slot T through the masked layer), validation and private.  Label-free for the official windows."""
import sys, warnings
import numpy as np
import pandas as pd
import lab
import pvog_lib as pv

panels = sys.argv[1].split(',') if len(sys.argv) > 1 else ['D7_I10_E', 'D7_I10_W']
cols = pv.COLS['all']; c = {k: cols.index(k) for k in cols}
rows = []


def stats(tag, panel, X):
    ql = X[:, c['q_last']] == 1
    r = dict(src=tag, panel=panel, rows=len(X), miss_last=X[:, c['miss_last']].mean(), has_T=X[:, c['has_T']].mean(),
             vr_raw_last_nan=np.isnan(X[:, c['vr_raw_last']]).mean(), vr_last_nan=np.isnan(X[:, c['vr_last']]).mean(),
             q_count_all=X[:, c['q_count']].mean(), q_count_qlast=X[ql, c['q_count']].mean() if ql.any() else np.nan,
             q_age_qlast=X[ql, c['q_age']].mean() if ql.any() else np.nan, n_q_hist_win=X[:, c['n_q_hist_win']].mean(),
             share_q=X[:, c['share_q']].mean(), vr_mean12_nan=np.isnan(X[:, c['vr_mean12']]).mean(), q_last=ql.mean())
    rows.append(r)


TR, EV = pv.march_data(31, 1.0, eval_from=99, panels=panels, verbose=False)
for pi, panel in enumerate(panels):
    P = lab.load_panel(panel)
    rc = lab.rows(panel, 'runstart'); m = rc.e & rc.keep5[:, 4]
    stats('train raw (F03 run-starts)', panel, rc.matrix(m, cols, 4))
    z = pv.masked_f03_rows(panel); stats('train masked emulation (F03)', panel, np.asarray(z['X'][z['rs']]))
    stats('March rows (kernel style)', panel, TR['X'][TR['panel'] == pi])
    for split in ('validation', 'private'):
        R = lab.split_rows(P, split)
        X, _, _, meta, call = lab.to_matrix(R, lab.ONGOING, None, lab.ongoing_keep(R))
        stats(f'official {split} ongoing (raw history)', panel, X)
    lab.drop_panel(panel)
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
g = T.drop(columns=['panel']).groupby('src', sort=False).mean(numeric_only=True)
print('\nmean over panels'); print(g.round(3).to_string())
T.to_csv(f'{pv.RES}/featshift.csv', index=False)
