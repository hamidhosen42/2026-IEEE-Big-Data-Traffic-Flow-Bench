"""ogpatch report: PRE-REGISTERED gate of EXP_T2_PATCH (ogpatch_lib.py) -- FM-PM validation >= +0.0015 AND private >= +0.0005, decisions
of the np + id mean of the fold-0 / fold-1 lab models (base = occ_np255 / occ_id255).  Not binding: train OOF (folds 0 + 1; np and np + id),
pseudo-March offlike_m / runstart_m paired deltas, footprint counts, gain share of the patch columns.  Info: FM-PM with np-only decisions.
Copied pattern of ogphys_report.py (never edited there).
Usage: ogpatch_report.py [arms=occP9,occP21,occP9q]"""
import sys, types
import numpy as np
import pandas as pd
import lab
import occ_lib as oc
import ogpatch_lib as gp
import exp_firstcross_lib as fc
sys.path.insert(0, f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts')
import fpmatch_lib as fpa  # noqa: E402

kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
ARMS = kw.get('arms', ','.join(gp.ARMS)).split(',')
F, ONGOING, FOLDS = lab.F, lab.ONGOING, gp.FOLDS
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def dec(p6):
    return lab.decode(np.asarray(p6, np.float32).ravel()).reshape(F, -1)


def fold_pk(arm, c, k):
    if arm == 'base':
        return pd.read_pickle(f'{oc.arm_dir(f"occ_{c}255")}/fold{k}.pkl')
    return pd.read_pickle(f'{gp.arm_dir(f"{arm}_{c}255")}/fold{k}.pkl')


def oof_frame(arm, cs):
    """per-window OOF frame (folds 0 + 1) of the mean probability over colsets cs."""
    out = []
    for k in FOLDS:
        pk = {c: fold_pk(arm, c, k)['probs'] for c in cs}
        for (s, panel), (widx, _) in pk[cs[0]].items():
            rc = oc.rows_af(panel, s, ONGOING)
            P6 = np.zeros((len(rc.win), F, rc.L), np.float32)
            for c in cs:
                w2, p = pk[c][(s, panel)]; assert np.array_equal(w2, widx)
                P6[widx] += p.astype(np.float32) / len(cs)
            out.append(lab.score_probs(rc, P6, widx, arm=arm, set_name=s))
    return pd.concat(out, ignore_index=True)


def cmp_rows(A, B, label):
    rows = []
    for s in dict.fromkeys(A.set):
        a, b = fc.paired(A[A.set == s], B[B.set == s])
        bt = lab.paired_bootstrap(a, b); pa = a.groupby('panel').iou.mean(); pb = b.groupby('panel').iou.mean()
        rows.append(dict(cmp=label, set=s, n=bt['n_windows'], base=lab.aggregate(a), new=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'],
                         hi=bt['hi'], panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}',
                         d_h6=lab.aggregate(b, 's6') - lab.aggregate(a, 's6'), d_fn_new=lab.aggregate(b, 'fn_new') - lab.aggregate(a, 'fn_new'),
                         d_fp_old=lab.aggregate(b, 'fp_old') - lab.aggregate(a, 'fp_old')))
    return rows


PM = pd.read_pickle(f'{gp.PC}/pm/pm_probs.pkl'); OFF = pd.read_pickle(f'{gp.PC}/off/off_probs.pkl')['off']


def mean_p(Pd, arm, cs):
    return sum(Pd[(arm, c, k)].astype(np.float32) for c in cs for k in FOLDS) / (len(cs) * len(FOLDS))


def pm_eval(arm, cs):
    """pseudo-March per-window frames (base, cand) + FM-PM lab rows."""
    fa, fb, lr = [], [], []
    for (panel, s), Pd in PM['P'].items():
        fr = PM['frames'][(panel, s)]
        rc = types.SimpleNamespace(truth=fr['truth'], ev=fr['ev'], qlast=fr['qlast'], win=fr['win'], name=s, panel=panel, cond=ONGOING,
                                   L=fr['qlast'].shape[1])
        pa, pb = mean_p(Pd, 'base', cs), mean_p(Pd, arm, cs)
        a = lab.score_probs(rc, pa, arm='base', set_name=s); b = lab.score_probs(rc, pb, arm=arm, set_name=s)
        fa.append(a); fb.append(b)
        for j in range(len(fr['win'])):
            lr.append(dict(set=s, panel=panel, d=int(fr['win'].d.iloc[j]), t=int(fr['win'].t.iloc[j]), tod=fr['win'].t.iloc[j] / 12,
                           dI=b.iou.iloc[j] - a.iou.iloc[j], dn=int(dec(pb[j]).sum()) - int(dec(pa[j]).sum())))
    return pd.concat(fa, ignore_index=True), pd.concat(fb, ignore_index=True), pd.DataFrame(lr)


def official_fp(arm, cs):
    rows = []
    for (panel, split), z in OFF.items():
        pa, pb = mean_p(z['P'], 'base', cs), mean_p(z['P'], arm, cs)
        for j, w in enumerate(z['window_id']):
            Da, Db = dec(pa[j]), dec(pb[j]); q = np.repeat(z['qlast'][j][None], F, 0)
            rows.append(dict(split=split, window_id=w, panel=panel, tod=z['tod'][j], nA=int(Da.sum()), nB=int(Db.sum()),
                             changed=int((Da ^ Db).sum()), add_out=int((Db & ~Da & ~q).sum()), rem=int((Da & ~Db).sum())))
    O = pd.DataFrame(rows); O['dn'] = O.nB - O.nA
    return O


out, fmrows, gain = [], [], {}
for arm in ARMS:
    print(f'\n========== {arm}')
    for cs, tag in ((('np',), 'np'), (('np', 'id'), 'npid')):
        out += cmp_rows(oof_frame('base', cs), oof_frame(arm, cs), f'{arm}[{tag}] train OOF')
        A, B, Lr = pm_eval(arm, cs)
        out += cmp_rows(A, B, f'{arm}[{tag}] pseudo-March')
        O = official_fp(arm, cs)
        O.to_csv(f'{gp.RES}/official_footprint_{arm}_{tag}.csv', index=False)
        r = dict(arm=arm, decisions=tag, binding=(tag == 'npid'))
        for split in ('validation', 'private'):
            off = O[O.split == split]; f = fpa.forecast(Lr, off)
            r[f'{split}_win'] = int((off.changed > 0).sum()); r[f'{split}_cells'] = int(off.changed.sum()); r[f'{split}_dn'] = int(off.dn.sum())
            r[f'{split}_addout'] = int(off.add_out.sum()); r[f'{split}_fm_lb'] = 0.15 * f.mean()
        r['PASS'] = bool(r['validation_fm_lb'] >= 0.0015 and r['private_fm_lb'] >= 0.0005)
        fmrows.append(r)
    for k in FOLDS:
        imp = pd.Series(fold_pk(arm, 'np', k)['imp']); imp = imp / imp.sum(); ex = gp.ARM_COLS[arm]
        gain[(arm, k)] = float(imp[ex].sum())
        if k == 0:
            print(f'gain share of the {len(ex)} patch columns (np, fold 0): total {imp[ex].sum():.4f} | top 10:',
                  imp[ex].sort_values(ascending=False).head(10).round(4).to_dict())
            base_top = imp.drop(ex).sort_values(ascending=False).head(8).round(4).to_dict()
            print('  top production / occupancy columns in the same model:', base_top)
T = pd.DataFrame(out); print(T.round(4).to_string(index=False)); T.to_csv(f'{gp.RES}/deltas.csv', index=False)
FM = pd.DataFrame(fmrows)
print('\n===== FM-PM (PRE-REGISTERED gate on the np + id rows: validation >= +0.0015 AND private >= +0.0005)')
print(FM.round(5).to_string(index=False)); FM.to_csv(f'{gp.RES}/fm.csv', index=False)
print('\ngain share of the patch columns (np models, fold 0 / 1):', {f'{a} f{k}': round(v, 4) for (a, k), v in gain.items()})

# ---- compact per-arm summary (binding rows = np + id decisions)
summ = []
for arm in ARMS:
    r = FM[(FM.arm == arm) & FM.binding].iloc[0]
    d = T[T.cmp.str.startswith(f'{arm}[npid]')]
    g = lambda lab_, s: d[(d.cmp == f'{arm}[npid] {lab_}') & (d.set == s)].iloc[0]
    row = dict(arm=arm, fm_val=r.validation_fm_lb, fm_pri=r.private_fm_lb, PASS=r.PASS, val_win=r.validation_win, pri_win=r.private_win,
               val_cells=r.validation_cells, pri_cells=r.private_cells, val_dn=r.validation_dn, pri_dn=r.private_dn)
    for s in ('offlike_m', 'runstart_m'):
        x = g('pseudo-March', s); row.update({f'{s}': x['diff'], f'{s}_lo': x.lo, f'{s}_hi': x.hi, f'{s}_pp': x.panels_pos})
    for s in ('official', 'offlike', 'pseudo'):
        x = g('train OOF', s); row.update({f'oof_{s}': x['diff'], f'oof_{s}_lo': x.lo, f'oof_{s}_hi': x.hi, f'oof_{s}_pp': x.panels_pos})
    summ.append(row)
S = pd.DataFrame(summ); S.to_csv(f'{gp.RES}/summary.csv', index=False)
print('\n===== SUMMARY (np + id decisions)'); print(S.round(5).to_string(index=False))
print('\nGATE:', {r['arm']: r['PASS'] for r in fmrows if r['binding']})
