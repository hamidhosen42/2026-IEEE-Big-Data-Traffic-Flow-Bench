"""occ report: occupancy features vs the all-finite baseline (train OOF, pseudo-March, onset).  Writes results/occ/*.csv + report.txt.
Usage: occ_report.py [folds=0,1,2,3]"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import lab
import occ_lib as oc
import exp_firstcross_lib as fc

ARGS = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
FOLDS = [int(x) for x in ARGS.get('folds', '0,1,2,3').split(',')]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
LINES = []


def out(s=''):
    print(s, flush=True); LINES.append(str(s))


def frames(arm_np, arm_id, name):
    """per-window frames np / id / npid (blend of the two) restricted to FOLDS."""
    a = oc.load_arm(arm_np); b = oc.load_arm(arm_id)
    fr = {'np': a, 'id': b}
    bl = []
    for s in fc.EVALS:
        bl.append(oc.blend_frame([arm_np, arm_id], s, f'{name}_npid'))
    fr['npid'] = pd.concat(bl, ignore_index=True)
    return {k: v[v.fold.isin(FOLDS)] if len(v) else v for k, v in fr.items()}


rows = []
base = frames('af:af_all_np255', 'af:af_all_id255', 'af')
new = frames('occ_np255', 'occ_id255', 'occ')
out(f'== TRAIN OOF: occupancy (13 cols) vs all-finite baseline | folds requested {FOLDS} ==')
for s in ('offlike', 'official', 'pseudo'):
    for c in ('np', 'id', 'npid'):
        A = base[c][base[c].set == s]; B = new[c][new[c].set == s]
        if not len(B):
            continue
        r = oc.compare(A, B, f'{s:8s} {c:4s} occ vs af'); r.update(block='oof', set=s, cols=c, wtype='all',
                                                                  folds=','.join(str(k) for k in sorted(B.fold.unique())))
        rows.append(r); out(oc.fmt(r))
        if s == 'offlike':
            for flag, wt in ((True, 'runstart'), (False, 'midrun')):
                r = oc.compare(A[A.runstart == flag], B[B.runstart == flag], f'{s:8s} {c:4s} occ vs af [{wt}]')
                r.update(block='oof', set=s, cols=c, wtype=wt); rows.append(r); out(oc.fmt(r))
out()
out('== per fold (offlike) ==')
for c in ('np', 'id', 'npid'):
    for k in FOLDS:
        A = base[c][(base[c].set == 'offlike') & (base[c].fold == k)]; B = new[c][(new[c].set == 'offlike') & (new[c].fold == k)]
        if len(B):
            r = oc.compare(A, B, f'offlike {c:4s} fold {k}'); r.update(block='oof_fold', set='offlike', cols=c, wtype='all', folds=str(k))
            rows.append(r); out(oc.fmt(r))
out()
# per-panel table for the npid blend on offlike
A = base['npid'][base['npid'].set == 'offlike']; B = new['npid'][new['npid'].set == 'offlike']
if len(B):
    a2, b2 = fc.paired(A, B)
    t = pd.DataFrame(dict(n=a2.groupby('panel').size(), base=a2.groupby('panel').iou.mean(), new=b2.groupby('panel').iou.mean())); t['delta'] = t.new - t.base
    out('per panel (offlike, npid):'); out(t.round(4).to_string()); t.to_csv(f'{oc.RES}/perpanel_offlike_npid.csv')
out()
# second seed: replicate of the main comparison, seed noise, and the production-shaped bag of 4 (np, id x seeds 0, 1)
have = lambda arm: os.path.exists(f'{oc.arm_dir(arm)}/fold{FOLDS[0]}.pkl')
if all(have(a) for a in ('seed1_np255', 'seed1_id255', 'occs1_np255', 'occs1_id255')):
    out('== SECOND SEED (seed 1 replicas of both arms) ==')
    F1 = {}
    for s in ('offlike', 'pseudo'):
        F1[('af1', s)] = oc.blend_frame(['seed1_np255', 'seed1_id255'], s, 'af1_npid')
        F1[('occ1', s)] = oc.blend_frame(['occs1_np255', 'occs1_id255'], s, 'occ1_npid')
        F1[('afb', s)] = oc.blend_frame(['af:af_all_np255', 'af:af_all_id255', 'seed1_np255', 'seed1_id255'], s, 'af_bag4')
        F1[('occb', s)] = oc.blend_frame(['occ_np255', 'occ_id255', 'occs1_np255', 'occs1_id255'], s, 'occ_bag4')
        F1 = {k: (v[v.fold.isin(FOLDS)] if len(v) else v) for k, v in F1.items()}
        A0 = base['npid'][base['npid'].set == s]; B0 = new['npid'][new['npid'].set == s]
        for lbl, A, B, blk in ((f'{s:8s} npid seed1: occ vs af (replicate)', F1[('af1', s)], F1[('occ1', s)], 'oof_s1'),
                               (f'{s:8s} NOISE af npid seed1 vs seed0', A0, F1[('af1', s)], 'noise_af'),
                               (f'{s:8s} NOISE occ npid seed1 vs seed0', B0, F1[('occ1', s)], 'noise_occ'),
                               (f'{s:8s} BAG4 (np, id x 2 seeds): occ vs af', F1[('afb', s)], F1[('occb', s)], 'oof_bag4')):
            r = oc.compare(A, B, lbl); r.update(block=blk, set=s, cols='npid' if 'bag4' not in blk else 'bag4', wtype='all',
                                                 folds=','.join(str(k) for k in sorted(B.fold.unique())) if len(B) else '')
            rows.append(r); out(oc.fmt(r))
            if s == 'offlike' and blk == 'oof_bag4':
                for flag, wt in ((True, 'runstart'), (False, 'midrun')):
                    r = oc.compare(A[A.runstart == flag], B[B.runstart == flag], f'{s:8s} BAG4 occ vs af [{wt}]')
                    r.update(block=blk, set=s, cols='bag4', wtype=wt); rows.append(r); out(oc.fmt(r))
                a2, b2 = fc.paired(A, B)
                t = pd.DataFrame(dict(n=a2.groupby('panel').size(), base=a2.groupby('panel').iou.mean(), new=b2.groupby('panel').iou.mean())); t['delta'] = t.new - t.base
                out('per panel (offlike, bag4):'); out(t.round(4).to_string()); t.to_csv(f'{oc.RES}/perpanel_offlike_bag4.csv')
    out()
if have('occnoraw_np255'):
    ab = oc.load_arm('occnoraw_np255'); fk = sorted(ab.fold.unique())
    out(f'== MECHANISM DIAGNOSTIC (not a candidate): np + 12 occupancy ratios WITHOUT raw o_last, folds {fk} ==')
    for s in ('offlike', 'pseudo'):
        A = base['np'][(base['np'].set == s) & base['np'].fold.isin(fk)]; B = ab[ab.set == s]; C = new['np'][(new['np'].set == s) & new['np'].fold.isin(fk)]
        for lbl, x, y_, blk in ((f'{s:8s} np noraw vs af', A, B, 'abl_vs_af'), (f'{s:8s} np noraw vs occ (13 cols)', C, B, 'abl_vs_occ')):
            r = oc.compare(x, y_, lbl); r.update(block=blk, set=s, cols='np', wtype='all'); rows.append(r); out(oc.fmt(r))
            if s == 'offlike':
                for flag, wt in ((True, 'runstart'), (False, 'midrun')):
                    r = oc.compare(x[x.runstart == flag], y_[y_.runstart == flag], f'{s:8s} {lbl.split(" ", 1)[1].strip()} [{wt}]')
                    r.update(block=blk, set=s, cols='np', wtype=wt); rows.append(r); out(oc.fmt(r))
    out()
f = f'{oc.RES}/pmarch/dryrun_footprint.csv'
if os.path.exists(f):
    D = pd.read_csv(f)
    g = D.groupby('split').agg(windows=('panel', 'size'), changed=('changed', 'sum'), cells_changed=('cells_changed', 'sum'),
                               cells_af=('n_af', 'sum'), cells_occ=('n_occ', 'sum'), mean_jacc=('jacc', 'mean'))
    out('== label-free footprint on the released official ONGOING windows (decoded np+id masks, occ vs af) =='); out(g.round(4).to_string()); out()
# sanity / noise references
sb = oc.load_arm('occbase_np255')
if len(sb):
    ref = oc.load_arm('af:af_all_np255')
    for s in fc.EVALS:
        a, b = fc.paired(ref[(ref.set == s) & (ref.fold == 0)], sb[sb.set == s])
        out(f'SANITY base0 (my code path, no occupancy) vs af_all_np255 fold 0 [{s}]: n {len(a)} max |IoU diff| {np.abs(a.iou.to_numpy() - b.iou.to_numpy()).max():.2e}')
s1 = oc.load_arm('seed1_np255')
if len(s1):
    ref = oc.load_arm('af:af_all_np255')
    for s in ('offlike', 'pseudo'):
        A = ref[(ref.set == s) & ref.fold.isin(s1.fold.unique())]; B = s1[s1.set == s]
        r = oc.compare(A, B, f'NOISE REF {s} np: baseline seed 1 vs seed 0 (folds {sorted(s1.fold.unique())})')
        r.update(block='noise', set=s, cols='np', wtype='all'); rows.append(r); out(oc.fmt(r))
        Bo = new['np'][(new['np'].set == s) & new['np'].fold.isin(s1.fold.unique())]
        if len(Bo):
            r = oc.compare(B, Bo, f'          {s} np: occ seed 0 vs baseline seed 1 (same folds)')
            r.update(block='noise_occ_vs_s1', set=s, cols='np', wtype='all'); rows.append(r); out(oc.fmt(r))
out()
# feature importance (gain share of the occupancy columns)
for arm in ('occ_np255', 'occ_id255'):
    imps = []
    for k in range(4):
        ff = f'{oc.arm_dir(arm)}/fold{k}.pkl'
        if os.path.exists(ff):
            imps.append(pd.Series(pd.read_pickle(ff)['imp']))
    if imps:
        I = pd.concat(imps, axis=1).mean(1); I = I / I.sum()
        occ = I[oc.OCC_COLS].sort_values(ascending=False)
        out(f'{arm}: gain share of the 13 occupancy columns {occ.sum():.3f} | top: ' + ', '.join(f'{k} {v:.3f}' for k, v in occ.head(6).items())
            + f' | rank of best occ col {int((I > occ.iloc[0]).sum()) + 1} of {len(I)}')
        I.sort_values(ascending=False).to_csv(f'{oc.RES}/importance_{arm}.csv')
out()
# onset fold 0
ff = f'{oc.OC}/onset/occ_fold0.pkl'
if os.path.exists(ff):
    z = pd.read_pickle(ff); D = z['df']; B0 = pd.read_pickle(f'{oc.AF}/onset/all_fold0.pkl')
    out(f'== ONSET fold 0 (CatBoost NORAMP, dense1500 rows {z["n"]}, pos {z["pos"]:.4f}) ==')
    for s in ('offlike', 'official', 'pseudo'):
        a, b = fc.paired(B0[(B0.set == s) & (B0.arm == 'all_s0')], D[(D.set == s) & (D.arm == 'rebase_s0')])
        out(f'  reproduction rebase_s0 vs allfin all_s0 [{s}]: n {len(a)} max |IoU diff| {np.abs(a.iou.to_numpy() - b.iou.to_numpy()).max() if len(a) else np.nan:.2e}')
        for nm in ('s0', 'bag'):
            r = oc.compare(B0[(B0.set == s) & (B0.arm == f'all_{nm}')], D[(D.set == s) & (D.arm == f'occ_{nm}')], f'onset {s:8s} {nm:3s} occ vs af')
            r.update(block='onset_f0', set=s, cols=nm, wtype='all'); rows.append(r); out(oc.fmt(r))
    imp = pd.Series(z['imp']); imp = imp / imp.sum()
    out(f'  onset importance share of the occupancy columns {imp[oc.OCC_COLS].sum():.3f} | top: ' +
        ', '.join(f'{k} {v:.3f}' for k, v in imp[oc.OCC_COLS].sort_values(ascending=False).head(5).items()))
out()
# pseudo-March
for tag in ('og', 'og4', 'on'):
    f = f'{oc.RES}/pmarch/deltas_{tag}.csv'
    if os.path.exists(f):
        d = pd.read_csv(f)
        out(f'== PSEUDO-MARCH ({tag}) occ vs af ==')
        for _, r in d.iterrows():
            out(oc.fmt(r.to_dict()))
            rr = r.to_dict(); rr['block'] = f'pmarch_{tag}'; rows.append(rr)
        out()
pd.DataFrame(rows).to_csv(f'{oc.RES}/cv_deltas.csv', index=False)
open(f'{oc.RES}/report.txt', 'w').write('\n'.join(LINES) + '\n')
