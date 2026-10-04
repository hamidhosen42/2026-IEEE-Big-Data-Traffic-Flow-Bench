"""allfin: collect the measured numbers (light: reads result frames, paired bootstraps, blend decoding).
Usage: allfin_report.py            -> prints the tables and writes results/allfin/*.csv"""
import glob, os, sys
import numpy as np
import pandas as pd
import lab
import allfin_lib as af
import exp_firstcross_lib as fc

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
SETS = ('offlike', 'official', 'pseudo')
out = []


def add(r, block):
    r = dict(r, block=block); out.append(r); print('  ' + af.fmt(r), flush=True)


# ---------------------------------------------------------------- 1. parity + availability / keep coverage
st = af.load_stats()
if len(st):
    st.to_csv(f'{af.RES}/build_stats.csv', index=False)
    print(f'== 1. parity: eligible-only build of the allfin builder vs lab.rows: {len(st)} caches, exact X {int(st.par_exact.sum())}/{len(st)}, '
          f'aux equal {int(st.par_aux_equal.sum())}/{len(st)}, rows compared {int(st.par_n_ref.sum())}, max abs diff {st.par_max_abs_diff.max():.1e}')
    g = st.groupby(['cond', 'set']).agg(panels=('panel', 'size'), windows=('all_windows', 'sum'), cells=('all_cells', 'sum'),
                                        pos=('all_pos', 'sum'), pos_keep_elig=('elig_pos_in_keep', 'sum'), pos_keep_all=('all_pos_in_keep', 'sum'),
                                        rows_elig=('elig_rows_e', 'sum'), rows_all=('all_rows_e', 'sum'), rows_xlab=('all_rows_ex', 'sum'))
    for c in ('kept_share', 'has_T', 'miss_last', 'vr_last_nan', 'q_last', 'q_count'):
        for m in ('elig', 'all'):
            g[f'{c}_{m}'] = (st[f'{m}_{c}'] * st['all_cells']).groupby([st.cond, st.set]).sum() / g.cells
    g['keepcov_elig'] = g.pos_keep_elig / g.pos; g['keepcov_all'] = g.pos_keep_all / g.pos
    print('== 1b. feature availability and ongoing keep region (all-train prior, cell-weighted over panels) ==')
    print(g[['panels', 'windows', 'has_T_elig', 'has_T_all', 'miss_last_elig', 'miss_last_all', 'vr_last_nan_elig', 'vr_last_nan_all',
             'q_count_elig', 'q_count_all', 'kept_share_elig', 'kept_share_all', 'keepcov_elig', 'keepcov_all', 'rows_elig', 'rows_all', 'rows_xlab']].round(4).to_string())
    g.to_csv(f'{af.RES}/availability_keep.csv')

# ---------------------------------------------------------------- 2. ongoing
print('\n== 2. ONGOING (F03 recipe, 255 leaves / lr 0.05 / 500 trees, OOF over day blocks) ==')
ref = {c: lab.load_arm(f'fc_F03_{c}255') for c in ('np', 'id')}
b0 = af.load_arm('af_base_np255', 'elig')
if len(b0):
    for s in SETS:
        a, b = fc.paired(ref['np'][(ref['np'].set == s) & (ref['np'].fold == 0)], b0[b0.set == s])
        print(f'  sanity: my eligible-only fold-0 arm vs fc_F03_np255 [{s}]: n {len(a)}, max |IoU diff| {np.abs(a.iou.to_numpy() - b.iou.to_numpy()).max():.2e}, '
              f'agg {lab.aggregate(a):.6f} vs {lab.aggregate(b):.6f}')
    bc = af.load_arm('af_base_np255', 'all')
    for s in SETS:
        add(af.compare(b0[b0.set == s], bc[bc.set == s], f'CROSS np fold0 [{s}]: eligible-trained model fed all-finite features'), 'og_cross')
arms = {}
for c in ('np', 'id'):
    df = af.load_arm(f'af_all_{c}255', 'all')
    if not len(df):
        continue
    arms[c] = df; fd = sorted(df.fold.unique().tolist())
    print(f'  af_all_{c}255 folds on disk {fd}')
    print(lab.summary(df).round(4).to_string(index=False))
    for s in SETS:
        A = ref[c][(ref[c].set == s) & ref[c].fold.isin(fd)]
        add(af.compare(A, df[df.set == s], f'{c}255 [{s}] all-finite vs eligible-only (fc_F03_{c}255)'), f'og_{c}')
    for k in fd:
        A = ref[c][(ref[c].set == 'offlike') & (ref[c].fold == k)]; B = df[(df.set == 'offlike') & (df.fold == k)]
        add(af.compare(A, B, f'{c}255 [offlike] fold {k}'), f'og_{c}_fold')
bl = {}
if len(arms) == 2:
    for s in SETS:
        B = af.blend_frame(['af_all_np255', 'af_all_id255'], 'all', s, 'af_npid')
        A = fc.blend_frame(['fc_F03_np255', 'fc_F03_id255'], [0.5, 0.5], s, 'base_npid', folds=sorted(B.fold.unique().tolist()))
        bl[s] = (A, B)
        add(af.compare(A, B, f'np+id blend [{s}] all-finite vs eligible-only'), 'og_blend')
    A, B = fc.paired(*bl['offlike'])
    t = pd.DataFrame(dict(n=A.groupby('panel').size(), base=A.groupby('panel').iou.mean(), new=B.groupby('panel').iou.mean())); t['delta'] = t.new - t.base
    print('  per panel (offlike, np+id blend):'); print(t.round(4).to_string())
    print('  per-step IoU (offlike, np+id blend) base:', np.round(lab.step_profile(A), 4), 'new:', np.round(lab.step_profile(B), 4),
          'delta:', np.round(np.array(lab.step_profile(B)) - np.array(lab.step_profile(A)), 4))
    ea, eb = lab.error_split(A), lab.error_split(B)
    print('  error split (offlike) base:', {k: round(v, 4) for k, v in ea.items()}, '| new:', {k: round(v, 4) for k, v in eb.items()})
    for k in sorted(B.fold.unique()):
        add(af.compare(A[A.fold == k], B[B.fold == k], f'np+id blend [offlike] fold {k}'), 'og_blend_fold')
    pd.concat([A.assign(arm='base_npid'), B.assign(arm='af_npid')]).to_pickle(f'{af.RES}/ongoing_blend_offlike.pkl')
xl = af.load_arm('af_xlab_np255', 'all')
if len(xl) and 'np' in arms:
    fd = sorted(xl.fold.unique().tolist())
    for s in SETS:
        add(af.compare(arms['np'][(arms['np'].set == s) & arms['np'].fold.isin(fd)], xl[xl.set == s], f'XLABELS np255 [{s}] folds {fd}: + ineligible-valued training labels vs all-finite'), 'og_xlab')
        add(af.compare(ref['np'][(ref['np'].set == s) & ref['np'].fold.isin(fd)], xl[xl.set == s], f'XLABELS np255 [{s}] folds {fd}: vs eligible-only'), 'og_xlab')

# ---------------------------------------------------------------- 3. onset
print('\n== 3. ONSET (CatBoost 600 it / depth 8 / lr 0.08, NORAMP, fold-safe prior, trained on dense1500 onset rows outside the fold) ==')
ff = sorted(glob.glob(f'{af.AF}/onset/*_fold*.pkl'))
if ff:
    O = pd.concat([pd.read_pickle(f) for f in ff], ignore_index=True)
    fd = sorted(set(O[O.arm == 'elig_s0'].fold) & set(O[O.arm == 'all_s0'].fold)); O = O[O.fold.isin(fd)]
    print('  folds with both arms:', fd)
    rows = []
    for s in SETS:
        for a in dict.fromkeys(O.arm):
            g = O[(O.set == s) & (O.arm == a)]
            rows.append(dict(set=s, arm=a, n=len(g), iou=lab.aggregate(g)))
    print(pd.DataFrame(rows).pivot(index='arm', columns='set', values='iou').round(4).to_string())
    for s in SETS:
        for nm in ('s0', 'bag'):
            A = O[(O.set == s) & (O.arm == f'elig_{nm}')]
            add(af.compare(A, O[(O.set == s) & (O.arm == f'all_{nm}')], f'onset {nm} [{s}] all-finite vs eligible-only'), 'onset')
            add(af.compare(A, O[(O.set == s) & (O.arm == f'cross_{nm}')], f'onset {nm} [{s}] CROSS (eligible-trained model, all-finite features)'), 'onset_cross')
    for k in fd:
        A = O[(O.set == 'offlike') & (O.arm == 'elig_bag') & (O.fold == k)]
        add(af.compare(A, O[(O.set == 'offlike') & (O.arm == 'all_bag') & (O.fold == k)], f'onset bag [offlike] fold {k}'), 'onset_fold')
    A, B = fc.paired(O[(O.set == 'offlike') & (O.arm == 'elig_bag')], O[(O.set == 'offlike') & (O.arm == 'all_bag')])
    t = pd.DataFrame(dict(n=A.groupby('panel').size(), base=A.groupby('panel').iou.mean(), new=B.groupby('panel').iou.mean())); t['delta'] = t.new - t.base
    print('  per panel (offlike onset, 3-seed bag):'); print(t.round(4).to_string())
    O.to_pickle(f'{af.RES}/onset_cv.pkl')
pd.DataFrame(out).to_csv(f'{af.RES}/cv_deltas.csv', index=False)
