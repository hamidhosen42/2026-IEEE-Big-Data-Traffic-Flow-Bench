"""Review (4): competition aggregation and decode by hand, score_probs against the organizer's score_window, bootstrap sanity."""
import sys, itertools, numpy as np, pandas as pd, lab

sys.path.insert(0, f'{lab.ROOT}/repo/src')
fails = 0


def chk(ok, msg):
    global fails
    print(('PASS ' if ok else 'FAIL ') + msg, flush=True); fails += (not ok)


# --- aggregation by hand: 2 families, 2 panels each, different window counts
df = pd.DataFrame([('A', 'A_1', 'queue_ongoing', 0.2), ('A', 'A_1', 'queue_ongoing', 0.4), ('A', 'A_2', 'queue_ongoing', 1.0),
                   ('B', 'B_1', 'queue_ongoing', 0.5), ('B', 'B_2', 'queue_ongoing', 0.7), ('B', 'B_2', 'queue_ongoing', 0.9), ('B', 'B_2', 'queue_ongoing', 0.8)],
                  columns=['family', 'panel', 'cond', 'iou'])
# panel means: A_1 0.3, A_2 1.0 -> family A 0.65; B_1 0.5, B_2 0.8 -> B 0.65 ; overall 0.65
chk(abs(lab.aggregate(df) - 0.65) < 1e-12, f'aggregate by hand = 0.65 (got {lab.aggregate(df):.6f}); plain mean would be {df.iou.mean():.4f}')
df2 = df.copy(); df2['cond'] = 'queue_onset'; df2['iou'] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
both = pd.concat([df, df2])
# panel x cond: A_1 (0.3, 0) -> 0.15, A_2 (1.0, 0) -> 0.5 -> A 0.325 ; B_1 (0.5, 1) -> 0.75, B_2 (0.8, 1) -> 0.9 -> B 0.825 ; overall 0.575
chk(abs(lab.s_queue(both) - 0.575) < 1e-12, f's_queue by hand = 0.575 (got {lab.s_queue(both):.6f})')

# --- decode: brute force over all subsets of 10 cells maximises the production objective sum_S p / (|S| + sum_notS p)
rng = np.random.default_rng(0)
ok = True
for trial in range(50):
    p = rng.random(10) ** 3
    best, bset = -1, None
    for m in range(1, 11):
        for S in itertools.combinations(range(10), m):
            v = p[list(S)].sum() / (m + p.sum() - p[list(S)].sum())
            if v > best + 1e-12:
                best, bset = v, set(S)
    out = lab.decode(p); got = set(np.where(out)[0])
    ok &= (got == bset)
chk(ok, 'decode == brute-force maximiser of expected-IoU proxy on 50 random 10-cell cases')
chk(lab.decode(np.zeros(5)).sum() == 1, 'decode of all-zero probabilities keeps exactly one cell')

# --- score_probs against the organizer score_window on cached windows (two panels, arm A_np255 official + offlike head)
from task2.score_task2 import score_window
res = lab.load_arm('A_np255')
for panel in ('D7_I10_W', 'D12_I5_S'):
    for s in ('official', 'offlike'):
        rc = lab.rows(panel, s); P6, done = lab.load_probs('A_np255', s, panel)
        widx = np.where(done)[0][:25]
        mine = lab.score_probs(rc, P6, widx)
        cached = res[(res.set == s) & (res.panel == panel)].set_index(['d', 't']).iou
        maxd = 0.0
        for j, i in enumerate(widx):
            pred = lab.decode(P6[i].ravel()).reshape(6, -1); truth, ev = rc.truth[i], rc.ev[i]
            hh, ll = np.meshgrid(np.arange(6), np.arange(rc.L), indexing='ij')
            tgt = pd.DataFrame(dict(window_id='w', timestamp=hh.ravel(), link_id=ll.ravel(), is_score_eligible=ev.ravel().astype(int), queue_true=truth.ravel().astype(int)))
            sub = pd.DataFrame(dict(window_id='w', timestamp=hh.ravel(), link_id=ll.ravel(), queue_pred=pred.ravel().astype(int)))
            org, _, _ = score_window(tgt, sub)
            maxd = max(maxd, abs(org - mine.iou.iloc[j]), abs(org - cached.loc[(int(rc.win.d[i]), int(rc.win.t[i]))]))
            # float16-stored probabilities may re-decode slightly differently: verify the per-window IoU with production iou too
            maxd = max(maxd, abs(mine.iou.iloc[j] - lab.iou(truth, pred, ev)))
        chk(maxd < 2e-3, f'{panel}/{s}: score_probs == organizer score_window == cached results on {len(widx)} windows (max abs diff {maxd:.2e}; probs stored as float16)')
        # per-step and error-split identities
        m = mine
        chk(np.allclose(m[[f'i{h}' for h in range(1, 7)]].sum(1) / np.maximum(m[[f'u{h}' for h in range(1, 7)]].sum(1), 1), m.iou.where(m.union > 0, 1.0)), f'{panel}/{s}: sum of per-step inter/union == window IoU')
        chk(np.allclose(m.fn_new + m.fn_old + m.fp_old + m.fp_new + m.iou.where(m.union > 0, 0), 1.0, atol=1e-9) | (m.union == 0).all(), f'{panel}/{s}: error shares + IoU == 1')

# --- bootstrap sanity
a = res[(res.set == 'offlike') & res.fold.isin([0])].copy()
b = a.copy(); b['iou'] = np.clip(b.iou + 0.01, 0, 1.01)
r = lab.paired_bootstrap(a, b, n=200)
chk(abs(r['diff'] - 0.01) < 1e-9 and abs(r['lo'] - 0.01) < 1e-9 and abs(r['hi'] - 0.01) < 1e-9 and r['n_windows'] == len(a), f'bootstrap of B = A + 0.01: {r}')
r0 = lab.paired_bootstrap(a, a, n=200)
chk(r0['diff'] == 0 and r0['lo'] == 0 and r0['hi'] == 0, f'bootstrap of A vs A: {r0}')
# the observed diff equals aggregate(B) - aggregate(A) when both frames hold the same windows
bb = a.copy(); bb['iou'] = np.clip(bb.iou + rng.normal(0, 0.1, len(bb)), 0, 1)
r1 = lab.paired_bootstrap(a, bb, n=300)
chk(abs(r1['diff'] - (lab.aggregate(bb) - lab.aggregate(a))) < 1e-9 and r1['lo'] <= r1['diff'] <= r1['hi'], f'bootstrap diff == aggregate(B) - aggregate(A): {r1}')
# duplicate-key hazard: frames holding several sets (same window in official and offlike)
try:
    multi = res[res.fold == 0]
    r2 = lab.paired_bootstrap(multi, multi, n=10)
    chk(r2['n_windows'] == len(multi), f'bootstrap on a multi-set frame: n_windows {r2["n_windows"]} vs {len(multi)} rows (duplicates would inflate)')
except ValueError as e:
    chk(True, f'bootstrap refuses a multi-set / duplicate-key frame: {e}')
print('SCORING REVIEW DONE, failures:', fails)
