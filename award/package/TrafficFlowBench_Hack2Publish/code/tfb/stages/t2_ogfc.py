"""t2_ogfeat_comb: fs4ogf = bag of the 4 ogfeat ongoing models of tfb-t2ogfa (t2_ogfeat_v27a.py: fs_np, fs_id, seed 0) and
tfb-t2ogfb (t2_ogfeat_v27b.py: fs_np1, fs_id1, seed 1).  No training; minutes on CPU.
Mean of the float32 probabilities in exactly the v26 fs4occ order, (1/4) * (((fs_np + fs_id) + fs_np1) + fs_id1), the production
expected-IoU decode (copied from v26), the v26 template mapping.  Onset rows are 0 (as in every ongoing kernel file).
Row skeleton = tfb-t2ogfa queue_submission_fs2ogf.csv (same rows and order, only queue_pred is replaced).
CONTROLS (the run fails if 1 or 2 fails):
  1. fs2ogf recomputed here from tfb-t2ogfa's probabilities == tfb-t2ogfa queue_submission_fs2ogf.csv on every row (mapping / decode parity);
     same for fs2ogf1 vs tfb-t2ogfb queue_submission_fs2ogf1.csv;
  2. both inputs carry the same ongoing windows (window_id arrays) and probability shapes;
  3. FOOTPRINT vs tfb-t2occ queue_submission_fs4occ.csv (if attached): ALARM above 3 % changed validation ongoing rows.
Kernel sources: tfb-t2ogfa, tfb-t2ogfb, tfb-t2occ (optional, footprint) + the competition data.
Output: /kaggle/working/queue_submission_fs4ogf.csv (window_id, timestamp, link_id, queue_pred)."""
import glob, time
import numpy as np
import pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-t2ogfc.py; patched lines end in [TFB]

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = TFB.out  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
if TFB.smoke:  # [TFB] smoke fixture only (the ogfeat SMOKE panels)
    PANELS = ['D7_I10_E', 'D12_I5_S']  # [TFB]
F = 6
ONGOING = 'queue_ongoing'
KEY = ['window_id', 'timestamp', 'link_id']


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def one(pattern):
    return TFB.kfile(*pattern.split('/**/'))  # [TFB] '<kernel>/**/<file>' -> the stage output of that kernel


def decode(p):
    """choose the top-m cells maximizing sum(top p) / (m + sum(rest p)); at least one cell.  (= t2_occ_v26.decode)"""
    order = np.argsort(-p); ps = p[order]; cs = np.cumsum(ps); tot = cs[-1]
    m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + (tot - cs)))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out


def mapped_rows(panel, split, P6og, wid_og):
    """decoded ongoing masks -> template rows (window_id, timestamp, link_id, queue_pred), exactly the v26 predict_split mapping."""
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    li = {l: i for i, l in enumerate(topo.link_id.tolist())}
    wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv')
    tpl = pd.read_csv(f'{REL}/task2/{panel}/{split}/sample_submission_queue.csv', dtype=str)
    W, L = len(wi), len(li)
    is_og = (wi.condition == ONGOING).to_numpy()
    assert list(wi.window_id.astype(str)[is_og]) == list(wid_og), f'{panel} {split}: ongoing window order differs from the kernel output'
    assert P6og.shape == (int(is_og.sum()), F, L), (P6og.shape, int(is_og.sum()), L)
    pr = np.zeros((W, F, L), bool)
    for k, i in enumerate(np.where(is_og)[0]):
        pr[i] = decode(P6og[k].ravel()).reshape(F, L)
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
    tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(li)
    tts = pd.to_datetime(tpl.timestamp, utc=True)
    tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
    okr = tpl.w.notna() & tpl.l.notna() & tpl.h.between(1, F)
    tpl['queue_pred'] = 0
    tpl.loc[okr, 'queue_pred'] = pr[tpl.w[okr].astype(int), tpl.h[okr].astype(int) - 1, tpl.l[okr].astype(int)].astype(int)
    tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).fillna(False).astype(bool)
    tpl['split'] = split
    return tpl[KEY + ['queue_pred', 'is_og', 'split']]


A = np.load(one('tfb-t2ogfa/**/t2v27a_probs.npz'), allow_pickle=False)
B = np.load(one('tfb-t2ogfb/**/t2v27b_probs.npz'), allow_pickle=False)
log('inputs: a keys', len(A.files), '| b keys', len(B.files))
rows = {'fs4ogf': [], 'fs2ogf': [], 'fs2ogf1': []}
for panel in PANELS:
    for split in ('validation', 'private'):
        k = f'{panel}__{split}__'
        wa, wb = A[k + 'window_id'].astype(str), B[k + 'window_id'].astype(str)
        assert np.array_equal(wa, wb), f'{panel} {split}: window ids differ between a and b'
        np_, id_, np1, id1 = A[k + 'fs_np'], A[k + 'fs_id'], B[k + 'fs_np1'], B[k + 'fs_id1']
        assert np_.dtype == id_.dtype == np1.dtype == id1.dtype == np.float32, 'probabilities must be float32'
        assert np_.shape == id_.shape == np1.shape == id1.shape
        p4 = (1.0 / 4) * sum([id_, np1, id1], np_)                      # = v26 VARIANTS order (fs_np, fs_id, fs_np1, fs_id1)
        p2 = (1.0 / 2) * sum([id_], np_)
        p21 = (1.0 / 2) * sum([id1], np1)
        for name, p in (('fs4ogf', p4), ('fs2ogf', p2), ('fs2ogf1', p21)):
            rows[name].append(mapped_rows(panel, split, p, wa))
        log(panel, split, 'ongoing windows', len(wa), '| mean prob np/id/np1/id1',
            [round(float(x.mean()), 5) for x in (np_, id_, np1, id1)], '| corr(seed0 pair, seed1 pair)',
            round(float(np.corrcoef(p2.ravel(), p21.ravel())[0, 1]), 5))
S = {n: pd.concat(v, ignore_index=True) for n, v in rows.items()}

# ---- CONTROL 1: mapping / decode parity with the kernels' own files
for name, src in (('fs2ogf', 'tfb-t2ogfa/**/queue_submission_fs2ogf.csv'), ('fs2ogf1', 'tfb-t2ogfb/**/queue_submission_fs2ogf1.csv')):
    ref = pd.read_csv(one(src), dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = ref.merge(S[name][KEY + ['queue_pred']].rename(columns={'queue_pred': 'mine'}), on=KEY, how='left', validate='one_to_one')
    assert m.mine.notna().all() and len(ref) == len(S[name]), f'CONTROL {name}: row sets differ'
    agree = float((m.queue_pred.astype(int) == m.mine.astype(int)).mean())
    log(f'CONTROL {name} recomputed vs {src}: rows {len(m)} | agreement {agree:.6f}')
    assert agree == 1.0, f'CONTROL {name} failed'

# ---- output on the tfb-t2ogfa row skeleton (same rows, same order as every ongoing kernel file)
skel = pd.read_csv(one('tfb-t2ogfa/**/queue_submission_fs2ogf.csv'), dtype={'window_id': str, 'timestamp': str, 'link_id': str})
out = skel[KEY].merge(S['fs4ogf'][KEY + ['queue_pred']], on=KEY, how='left', validate='one_to_one')
assert len(out) == len(skel) and out.queue_pred.notna().all() and not out.duplicated(KEY).any()
assert (out[KEY].to_numpy() == skel[KEY].to_numpy()).all()
out['queue_pred'] = out.queue_pred.astype(int)
assert out.queue_pred.isin([0, 1]).all()
out.to_csv(f'{OUT}/queue_submission_fs4ogf.csv', index=False)
s4 = S['fs4ogf']
log('queue_submission_fs4ogf.csv rows', len(out), '| ongoing rows', int(s4.is_og.sum()), '| positives ongoing', int(s4.queue_pred[s4.is_og].sum()),
    '| onset rows positive', int(s4.queue_pred[~s4.is_og].sum()))
for name in ('fs2ogf', 'fs2ogf1'):
    a = S[name].queue_pred.to_numpy(); b = s4.queue_pred.to_numpy(); og = s4.is_og.to_numpy()
    for sp in ('validation', 'private'):
        k = og & (s4.split == sp).to_numpy()
        log(f'fs4ogf vs {name} [{sp}]: ongoing rows changed {int((a[k] != b[k]).sum())}/{int(k.sum())} '
            f'| Jaccard {float((a[k] & b[k]).sum() / max((a[k] | b[k]).sum(), 1)):.4f}')

# ---- CONTROL 3: footprint vs the current production (tfb-t2occ fs4occ)
ref = TFB.kglob('tfb-t2occ', 'queue_submission_fs4occ.csv')  # [TFB] optional FOOTPRINT reference
if not ref:
    log('FOOTPRINT: skipped (tfb-t2occ queue_submission_fs4occ.csv not attached)')
else:
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = s4.merge(r.rename(columns={'queue_pred': 'ref'}), on=KEY, how='left')
    m = m[m.is_og & m.ref.notna()]
    for name in ('fs4ogf', 'fs2ogf'):
        mm = m if name == 'fs4ogf' else m.drop(columns='queue_pred').merge(S['fs2ogf'][KEY + ['queue_pred']], on=KEY)
        for sp, x in mm.groupby('split'):
            a = x.queue_pred.to_numpy().astype(int); b = x.ref.to_numpy().astype(int); sh = float((a != b).mean())
            chg = x.assign(c=(a != b)).groupby('window_id').c.any()
            flag = ' | ALARM: more than 3 % of the validation ongoing rows changed' if (sp == 'validation' and sh > 0.03) else ''
            log(f'FOOTPRINT {name} vs tfb-t2occ fs4occ | {sp}: windows changed {int(chg.sum())}/{len(chg)} | rows changed {int((a != b).sum())}/{len(a)}'
                f' ({sh:.4%}) | Jaccard {(a & b).sum() / max((a | b).sum(), 1):.4f} | positives {int(a.sum())} ref {int(b.sum())}{flag}')
log('COMB DONE')
