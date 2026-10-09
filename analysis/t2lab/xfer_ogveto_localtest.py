"""Local test of kaggle_jobs/t2_ogveto.py on the downloaded kernel outputs: ast.parse, run veto_rows with the local link order
(data/t2local export, = topology order_index order), and prove the output equals the lab veto10 decisions (decode of the kernel
probabilities, analysis/t2lab/xfer_shock_eval.py) row by row.  Usage: xfer_ogveto_localtest.py <download dir>"""
import ast, os, sys, importlib.util
import numpy as np, pandas as pd
import lab
DL = sys.argv[1]
KF = f'{lab.ROOT}/kaggle_jobs/t2_ogveto.py'
ast.parse(open(KF).read()); print('ast.parse OK')
os.environ['OGVETO_LOCAL'] = '1'
spec = importlib.util.spec_from_file_location('ogveto', KF); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
order = lambda p: lab.load_panel(p, cache=False)['links']
M, info = K.veto_rows(f'{DL}/tfb-t2occ/queue_submission_fs4occ.csv', f'{DL}/tfb-t2ogfc/queue_submission_fs4ogf.csv', f'{DL}/tfb-t2occ/t2v26occ_probs.npz', order)
K.controls(M, info)
# lab decisions from the probabilities
occ = np.load(f'{DL}/tfb-t2occ/t2v26occ_probs.npz'); ga = np.load(f'{DL}/tfb-t2ogfa/t2v27a_probs.npz'); gb = np.load(f'{DL}/tfb-t2ogfb/t2v27b_probs.npz')
rows = []; csv_mis = {'A': 0, 'B': 0}
Mi = M.set_index(['window_id', 'timestamp', 'link_id'])
for panel in lab.PANELS:
    links = order(panel)
    for split in ('validation', 'private'):
        k = f'{panel}__{split}__'; wi = lab.window_index(panel, split).set_index('window_id')
        pa = (occ[k + 'fs_np'].astype(np.float32) + occ[k + 'fs_id'].astype(np.float32) + occ[k + 'fs_np1'].astype(np.float32) + occ[k + 'fs_id1'].astype(np.float32)) / 4
        pb = (((ga[k + 'fs_np'] + ga[k + 'fs_id']) + gb[k + 'fs_np1']) + gb[k + 'fs_id1']) / 4
        ql = occ[k + 'qlast']; L = len(links)
        for i, w in enumerate(occ[k + 'window_id'].astype(str)):
            A = lab.decode(pa[i].ravel()).reshape(6, L); B = lab.decode(pb[i].ravel()).reshape(6, L)
            q = np.repeat(ql[i][None], 6, 0); V = A if int((B & ~A & ~q).sum()) > 10 else B
            o = pd.Timestamp(wi.loc[w, 'forecast_origin'])
            for h in range(6):
                ts = (o + pd.Timedelta(minutes=5 * (h + 1))).strftime('%Y-%m-%dT%H:%M:%SZ')
                for l in range(L):
                    rows.append((w, ts, links[l], int(V[h, l]), int(A[h, l]), int(B[h, l])))
R = pd.DataFrame(rows, columns=['window_id', 'timestamp', 'link_id', 'lab', 'labA', 'labB']).set_index(['window_id', 'timestamp', 'link_id'])
J = R.join(Mi[['queue_pred', 'queue_pred_a', 'queue_pred_b']], how='left')
assert J.queue_pred.notna().all(), f'{int(J.queue_pred.isna().sum())} lab cells without a kernel row (timestamp format?)'
print('rows compared', len(J), '| fs4occ csv == decode(probs):', int((J.labA == J.queue_pred_a).sum()), '| fs4ogf csv == decode(probs):', int((J.labB == J.queue_pred_b).sum()))
for split in ('validation', 'private'):
    m = J.index.get_level_values(0).str.contains(f'_{split}_')
    print(f'{split}: veto10 kernel == lab decisions on {int((J.lab[m] == J.queue_pred[m]).sum())}/{int(m.sum())} rows')
assert (J.lab == J.queue_pred).all(), 'kernel output differs from the lab veto10 decisions'
print('LOCAL TEST PASSED: kernel veto10 rows == lab veto10 decisions on every ongoing cell')
