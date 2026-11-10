"""t2_ogveto: X1 'veto10' ongoing rows = ogfeat bag (fs4ogf) with a growth veto, else fs4occ.  NOT PUSHED (written 2026-10-08).
Per ongoing window (validation and private): addout = number of cells (any step h) that fs4ogf predicts queued, fs4occ does not, and
whose link is NOT in the visible queue at T (qlast of tfb-t2occ, the production q_last).  If addout > 10 the window keeps the fs4occ
rows, otherwise it takes the fs4ogf rows.  Onset rows are copied from fs4occ (0 in every ongoing kernel file).
Analysis: analysis/TRANSFER_ANALYSIS_2026-10-08.md (sections 4 and 5); lab decisions analysis/t2lab/results/xfer/shock_official_decisions.csv.
No training, no decode: decisions come from the two queue files; qlast from tfb-t2occ/t2v26occ_probs.npz.
Kernel sources: tfb-t2occ (queue_submission_fs4occ.csv, t2v26occ_probs.npz), tfb-t2ogfc (queue_submission_fs4ogf.csv)
                + the competition data (corridors/<panel>/network/lwr_mainline_topology.csv for the link order).
CONTROLS: windows vetoed per split, rows changed vs fs4occ and vs fs4ogf per split, onset rows == fs4occ (assert), same row set (assert).
Output: /kaggle/working/queue_submission_veto10.csv (window_id, timestamp, link_id, queue_pred; fs4occ row order)."""
import glob, os, time
import numpy as np
import pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-t2ogveto.py; patched lines end in [TFB]

T0 = time.time()
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
if TFB.smoke:  # [TFB] smoke fixture only (the ogfeat SMOKE panels)
    PANELS = ['D7_I10_E', 'D12_I5_S']  # [TFB]
KEY = ['window_id', 'timestamp', 'link_id']
VETO = 10


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def one(pattern):
    return TFB.kfile(*pattern.split('/**/'))  # [TFB] '<kernel>/**/<file>' -> the stage output of that kernel


def kaggle_link_order(panel):
    rel = TFB.rel  # [TFB]
    topo = pd.read_csv(f'{rel}/corridors/{panel}/network/lwr_mainline_topology.csv', dtype={'link_id': str})
    return topo.sort_values('order_index').link_id.tolist()


def veto_rows(occ_csv, ogf_csv, occ_npz, link_order, veto=VETO):
    """returns (out frame in fs4occ row order, per-window info frame)."""
    A = pd.read_csv(occ_csv, dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    B = pd.read_csv(ogf_csv, dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    assert len(A) == len(B) and not A.duplicated(KEY).any() and not B.duplicated(KEY).any(), 'row sets differ / duplicates'
    M = A.merge(B, on=KEY, how='left', suffixes=('_a', '_b'), validate='one_to_one')
    assert M.queue_pred_b.notna().all(), 'fs4ogf misses fs4occ rows'
    M['queue_pred_b'] = M.queue_pred_b.astype(int)
    Z = np.load(occ_npz, allow_pickle=False)
    qset, cond_og = {}, set()
    for panel in PANELS:
        links = np.array(link_order(panel))
        for split in ('validation', 'private'):
            k = f'{panel}__{split}__'
            wids = Z[k + 'window_id'].astype(str); ql = Z[k + 'qlast']
            assert ql.shape == (len(wids), len(links)), (panel, split, ql.shape, len(links))
            for w, q in zip(wids, ql):
                qset[w] = set(links[q].tolist()); cond_og.add(w)
    M['og'] = M.window_id.isin(cond_og)
    M['inq'] = [l in qset.get(w, ()) for w, l in zip(M.window_id, M.link_id)]
    M['addout'] = (M.og & (M.queue_pred_b == 1) & (M.queue_pred_a == 0) & ~M.inq).astype(int)
    info = M[M.og].groupby('window_id').agg(addout=('addout', 'sum'), nA=('queue_pred_a', 'sum'), nB=('queue_pred_b', 'sum')).reset_index()
    info['veto'] = info.addout > veto
    vetoed = set(info.window_id[info.veto])
    use_b = M.og & ~M.window_id.isin(vetoed)
    M['queue_pred'] = np.where(use_b, M.queue_pred_b, M.queue_pred_a).astype(int)
    M['split'] = np.where(M.window_id.str.contains('_validation_'), 'validation', np.where(M.window_id.str.contains('_private_'), 'private', 'other'))
    return M, info


def controls(M, info):
    on = ~M.og
    assert (M.queue_pred[on] == M.queue_pred_a[on]).all(), 'onset rows differ from fs4occ'
    log(f'CONTROL onset rows {int(on.sum())} == fs4occ: True | onset positives {int(M.queue_pred[on].sum())}')
    for split in ('validation', 'private'):
        m = M.split == split; o = m & M.og
        iw = info[info.window_id.str.contains(f'_{split}_')]
        log(f'CONTROL {split}: ongoing windows {len(iw)} | vetoed {int(iw.veto.sum())} {sorted(iw.window_id[iw.veto].tolist())} | '
            f'rows changed vs fs4occ {int((M.queue_pred[o] != M.queue_pred_a[o]).sum())}/{int(o.sum())} | '
            f'vs fs4ogf {int((M.queue_pred[o] != M.queue_pred_b[o]).sum())}/{int(o.sum())} | positives ongoing fs4occ {int(M.queue_pred_a[o].sum())} '
            f'fs4ogf {int(M.queue_pred_b[o].sum())} veto10 {int(M.queue_pred[o].sum())}')


if __name__ == '__main__' and not os.environ.get('OGVETO_LOCAL'):
    M, info = veto_rows(one('tfb-t2occ/**/queue_submission_fs4occ.csv'), one('tfb-t2ogfc/**/queue_submission_fs4ogf.csv'),
                        one('tfb-t2occ/**/t2v26occ_probs.npz'), kaggle_link_order)
    controls(M, info)
    M[KEY + ['queue_pred']].to_csv(f'{TFB.out}/queue_submission_veto10.csv', index=False)  # [TFB]
    info.to_csv(f'{TFB.out}/t2ogveto_windows.csv', index=False)  # [TFB]
    log('queue_submission_veto10.csv rows', len(M), '| DONE')
