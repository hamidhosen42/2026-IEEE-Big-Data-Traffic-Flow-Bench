"""Score the OOF cnn probabilities (one tag, or the mean of several tags = seed ensemble) alone and blended with the tree
reference (mean of the cached OOF arms fc_F03_np255 and fc_F03_id255) on official / offlike / pseudo.
Paired bootstrap (lab.paired_bootstrap, windows resampled within panel) vs the tree reference, per-panel sign count,
per-step IoU, per-fold deltas, probability correlation (diversity).
Usage: cnn_eval.py <tag[,tag...]> [name]      -> results/cnn/cv_<name>_*.csv, cv_<name>_windows.pkl"""
import os, sys
import numpy as np
import pandas as pd
import lab
import cnn_lib as C

tags = sys.argv[1].split(',')
name = sys.argv[2] if len(sys.argv) > 2 else '+'.join(tags)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
WS = (0.1, 0.2, 0.3, 0.4, 0.5)
MAIN = ('blend20', 'blend30', 'blend50')
_done = f'{C.RES}/cv_{name}_deltas.csv'
if os.path.exists(_done) and not os.environ.get('CNN_FORCE') and all(os.path.exists(f'{C.CDIR}/{tg}/fold{k}.pkl') for tg in tags for k in range(4)) \
        and all(os.path.getmtime(_done) > os.path.getmtime(f'{C.CDIR}/{tg}/fold{k}.pkl') for tg in tags for k in range(4)):
    print(f'cnn_eval {name}: results already on disk ({_done}); set CNN_FORCE=1 to recompute')
    sys.exit(0)


def cnn_probs(set_name, panel):
    """mean over tags of the OOF cnn probabilities; done = windows present in every tag."""
    rc = lab.rows(panel, set_name)
    P6 = np.zeros((len(rc.win), lab.F, rc.L), np.float32); done = np.ones(len(rc.win), bool)
    for tg in tags:
        dn = np.zeros(len(rc.win), bool)
        for k in range(4):
            f = f'{C.CDIR}/{tg}/fold{k}.pkl'
            if not os.path.exists(f):
                continue
            pr = pd.read_pickle(f)['probs']
            if (set_name, panel) in pr:
                widx, p6 = pr[(set_name, panel)]
                P6[widx] += p6.astype(np.float32) / len(tags); dn[widx] = True
        done &= dn
    return rc, P6, done


res = []; corr = []
for s in C.EVALS:
    for panel in C.PANELS:
        rc, Pc, dc = cnn_probs(s, panel)
        Pt, dt = C.tree_ref(s, panel)
        widx = np.where(dc & dt)[0]
        if not len(widx):
            continue
        res.append(lab.score_probs(rc, Pt, widx, arm='trees', set_name=s))
        res.append(lab.score_probs(rc, Pc, widx, arm='cnn', set_name=s))
        for w in WS:
            res.append(lab.score_probs(rc, w * Pc + (1 - w) * Pt, widx, arm=f'blend{int(round(w * 100))}', set_name=s))
        a, b = lab.load_probs('fc_F03_np255', s, panel)[0][widx], lab.load_probs('fc_F03_id255', s, panel)[0][widx]
        m = (Pt[widx] > 0) | (Pc[widx] > 0)
        corr.append(dict(set=s, panel=panel, n_cells=int(m.sum()), corr_cnn_trees=float(np.corrcoef(Pc[widx][m], Pt[widx][m])[0, 1]),
                         corr_np_id=float(np.corrcoef(a[m], b[m])[0, 1]),
                         mean_abs_diff_cnn_trees=float(np.abs(Pc[widx][m] - Pt[widx][m]).mean()), mean_abs_diff_np_id=float(np.abs(a[m] - b[m]).mean())))
R = pd.concat(res, ignore_index=True)
R.to_pickle(f'{C.RES}/cv_{name}_windows.pkl')
arms = list(dict.fromkeys(R.arm))
print(f'== cnn CV ({name}): aggregated IoU (window -> panel -> family -> mean), folds present: {sorted(R.fold.unique().tolist())} ==')
rows = []
for s in C.EVALS:
    for a in arms:
        g = R[(R.set == s) & (R.arm == a)]
        if len(g):
            es = lab.error_split(g)
            rows.append(dict(set=s, arm=a, n=len(g), iou=lab.aggregate(g), **{f'h{h + 1}': v for h, v in enumerate(lab.step_profile(g))},
                             missed_new=es['missed_new'], kept_cleared=es['kept_cleared'], other=es['other']))
S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(f'{C.RES}/cv_{name}_summary.csv', index=False)

print('\n== paired deltas vs trees (paired bootstrap n=2000, windows resampled within panel) ==')
rows = []
for s in C.EVALS:
    A = R[(R.set == s) & (R.arm == 'trees')]
    for a in arms:
        if a == 'trees':
            continue
        B = R[(R.set == s) & (R.arm == a)]
        b = lab.paired_bootstrap(A, B)
        pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
        r = dict(set=s, arm=a, **b, panels_pos=f'{int((pb - pa > 0).sum())}/{len(pa)}')
        for k in sorted(A.fold.unique()):
            r[f'fold{k}'] = lab.aggregate(B[B.fold == k]) - lab.aggregate(A[A.fold == k])
        for h in range(1, lab.F + 1):
            r[f'd_h{h}'] = lab.aggregate(B, f's{h}') - lab.aggregate(A, f's{h}')
        rows.append(r)
Dl = pd.DataFrame(rows); print(Dl.round(4).to_string(index=False)); Dl.to_csv(f'{C.RES}/cv_{name}_deltas.csv', index=False)

print('\n== per panel (offlike): IoU by arm ==')
g = R[R.set == 'offlike']
pp = g.pivot_table(index='panel', columns='arm', values='iou', aggfunc='mean')[arms]
pp['n'] = g[g.arm == 'trees'].groupby('panel').size()
for a in ('cnn',) + MAIN:
    pp[f'd_{a}'] = pp[a] - pp['trees']
print(pp.round(4).to_string()); pp.to_csv(f'{C.RES}/cv_{name}_perpanel_offlike.csv')

print('\n== offlike subsets: run-start windows vs the rest (delta of blend30 vs trees) ==')
for nm, m in (('runstart', g.runstart.astype(bool)), ('not runstart', ~g.runstart.astype(bool))):
    A = g[m & (g.arm == 'trees')]; B = g[m & (g.arm == 'blend30')]; Cn = g[m & (g.arm == 'cnn')]
    print(f'  {nm}: n={len(A)} trees {lab.aggregate(A):.4f} cnn {lab.aggregate(Cn):.4f} blend30 {lab.aggregate(B):.4f} delta {lab.aggregate(B) - lab.aggregate(A):+.4f}')

Cr = pd.DataFrame(corr); Cr.to_csv(f'{C.RES}/cv_{name}_corr.csv', index=False)
print('\n== probability correlation over cells with a non-zero probability (diversity) ==')
print(Cr.groupby('set')[['corr_cnn_trees', 'corr_np_id', 'mean_abs_diff_cnn_trees', 'mean_abs_diff_np_id']].mean().round(4).to_string())
lab.log('EVAL DONE')
