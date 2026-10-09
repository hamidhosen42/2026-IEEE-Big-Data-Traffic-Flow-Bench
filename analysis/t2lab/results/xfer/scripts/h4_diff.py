"""Compare validation ongoing rows of fs4occ (tfb-h1/h3 base) vs fs4ogf (tfb-h4) on the 40 official March windows.
Uses kernel probability outputs (downloaded) + local window index / history (data <= T only)."""
import sys, os
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
import lab

DL = sys.argv[1]
OUT = sys.argv[2]
occ = np.load(f'{DL}/tfb-t2occ/t2v26occ_probs.npz')
ga = np.load(f'{DL}/tfb-t2ogfa/t2v27a_probs.npz'); gb = np.load(f'{DL}/tfb-t2ogfb/t2v27b_probs.npz')
cA = pd.read_csv(f'{DL}/tfb-t2occ/queue_submission_fs4occ.csv'); cB = pd.read_csv(f'{DL}/tfb-t2ogfc/queue_submission_fs4ogf.csv')
rows = []; cells = []
for panel in lab.PANELS:
    P = lab.load_panel(panel); L = len(P['links'])
    wi = lab.window_index(panel, 'validation')
    R = lab.split_rows(P, 'validation')
    for split in ('validation',):
        k = f'{panel}__{split}__'
        wids = list(occ[k + 'window_id'])
        assert wids == list(ga[k + 'window_id']) == list(gb[k + 'window_id'])
        pa = (occ[k + 'fs_np'].astype(np.float32) + occ[k + 'fs_id'].astype(np.float32) + occ[k + 'fs_np1'].astype(np.float32) + occ[k + 'fs_id1'].astype(np.float32)) / 4
        pb = (((ga[k + 'fs_np'] + ga[k + 'fs_id']) + gb[k + 'fs_np1']) + gb[k + 'fs_id1']) / 4
        ql = occ[k + 'qlast']
        for i, w in enumerate(wids):
            r = wi[wi.window_id == w].iloc[0]
            j = int(np.where(wi.window_id.to_numpy() == w)[0][0])
            org = pd.Timestamp(r.forecast_origin)
            dA = lab.decode(pa[i].ravel()).reshape(6, L); dB = lab.decode(pb[i].ravel()).reshape(6, L)
            # check vs csv
            sA = cA[cA.window_id == w]; sB = cB[cB.window_id == w]
            li = sA.link_id.map(P['li']).to_numpy(); hh = ((pd.to_datetime(sA.timestamp) - org).dt.total_seconds() // 300).astype(int).to_numpy()
            mA = np.zeros((6, L), bool); mA[hh - 1, li] = sA.queue_pred.to_numpy() == 1
            liB = sB.link_id.map(P['li']).to_numpy(); hB = ((pd.to_datetime(sB.timestamp) - org).dt.total_seconds() // 300).astype(int).to_numpy()
            mB = np.zeros((6, L), bool); mB[hB - 1, liB] = sB.queue_pred.to_numpy() == 1
            evm = np.zeros((6, L), bool); evm[hh - 1, li] = True          # scored (template) cells
            assert ((dA & evm) == mA).all() and ((dB & evm) == mB).all(), (w, ((dA & evm) != mA).sum(), ((dB & evm) != mB).sum())
            q = ql[i]
            # distance of each link to the nearest queued-at-T link
            qi = np.where(q)[0]
            dist = np.min(np.abs(np.arange(L)[:, None] - qi[None]), 1) if len(qi) else np.full(L, 99)
            ch = mA ^ mB
            # expected IoU of a decision under a probability field (independent cells)
            def eiou(p, d):
                p = p * evm; s = (p * d).sum(); return s / (d.sum() + (p * ~d).sum())
            prev = wi[(wi.condition == 'queue_ongoing')]
            rows.append(dict(panel=panel, window_id=w, day=org.day, tod=org.hour + org.minute / 60, dow=org.dayofweek,
                             hist_cov=float(r.history_coverage) if 'history_coverage' in r else np.nan,
                             L=L, ncells=int(evm.sum()), size_T=int(q.sum()), nA=int(mA.sum()), nB=int(mB.sum()),
                             nA_h1=int(mA[0].sum()), nA_h6=int(mA[5].sum()), nB_h1=int(mB[0].sum()), nB_h6=int(mB[5].sum()),
                             changed=int(ch.sum()), added=int((mB & ~mA).sum()), removed=int((mA & ~mB).sum()),
                             ch_by_h=list(ch.sum(1)), ch_in_qT=int((ch & q[None]).sum()), ch_outside_qT=int((ch & ~q[None]).sum()),
                             ch_dist_le2=int((ch & (dist[None] <= 2)).sum()), ch_dist_gt2=int((ch & (dist[None] > 2)).sum()),
                             add_in_qT=int((mB & ~mA & q[None]).sum()), add_out_qT=int((mB & ~mA & ~q[None]).sum()),
                             rem_in_qT=int((mA & ~mB & q[None]).sum()), rem_out_qT=int((mA & ~mB & ~q[None]).sum()),
                             eA_A=eiou(pa[i], mA), eA_B=eiou(pa[i], mB), eB_B=eiou(pb[i], mB), eB_A=eiou(pb[i], mA),
                             jacc=(mA & mB).sum() / max((mA | mB).sum(), 1),
                             pmeanA=float((pa[i] * evm).sum() / evm.sum()), pmeanB=float((pb[i] * evm).sum() / evm.sum()),
                             sumpA=float((pa[i] * evm).sum()), sumpB=float((pb[i] * evm).sum())))
            for h, l in zip(*np.where(ch)):
                cells.append(dict(panel=panel, window_id=w, h=h + 1, link=l, added=bool(mB[h, l]), qT=bool(q[l]), dist=int(dist[l]),
                                  pA=float(pa[i, h, l]), pB=float(pb[i, h, l])))
    lab.drop_panel(panel)
df = pd.DataFrame(rows); C = pd.DataFrame(cells)
df.to_csv(f'{OUT}/h4_windows.csv', index=False); C.to_csv(f'{OUT}/h4_cells.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
print(df[['panel', 'window_id', 'day', 'tod', 'size_T', 'nA', 'nB', 'nA_h1', 'nA_h6', 'nB_h6', 'changed', 'added', 'removed', 'ch_by_h', 'ch_in_qT', 'ch_dist_le2', 'eA_A', 'eA_B', 'eB_B', 'eB_A', 'jacc']].round(3).to_string(index=False))
print('totals: cells', df.ncells.sum(), 'changed', df.changed.sum(), 'added', df.added.sum(), 'removed', df.removed.sum(), 'posA', df.nA.sum(), 'posB', df.nB.sum())
print('windows changed', (df.changed > 0).sum())
print('changed by h', np.sum(np.stack(df.ch_by_h.to_numpy()), 0))
print('added in/out qT', df.add_in_qT.sum(), df.add_out_qT.sum(), 'removed in/out qT', df.rem_in_qT.sum(), df.rem_out_qT.sum())
print('changed dist<=2 / >2', df.ch_dist_le2.sum(), df.ch_dist_gt2.sum())
print('top-5 windows hold', df.changed.nlargest(5).sum(), 'of', df.changed.sum())
print('panel totals'); print(df.groupby('panel')[['changed', 'added', 'removed', 'nA', 'nB', 'size_T']].sum())
print('self-expected IoU mean: occ-model on occ decision', df.eA_A.mean(), 'occ-model on ogf decision', df.eA_B.mean(), '| ogf-model on ogf', df.eB_B.mean(), 'ogf-model on occ', df.eB_A.mean())
print('expected-LB if occ probs are right:', 0.15 * (df.eA_B - df.eA_A).mean(), ' if ogf probs are right:', 0.15 * (df.eB_B - df.eB_A).mean())
print('mean prob per scored cell A/B', df.pmeanA.mean(), df.pmeanB.mean(), 'sum p A/B', df.sumpA.sum(), df.sumpB.sum())
