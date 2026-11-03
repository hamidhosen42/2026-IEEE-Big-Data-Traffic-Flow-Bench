"""phase_report.py -- pre-registered gate evaluation of EXP_T2_PHASE (results/phase/PREREG.txt).  Usage: phase_report.py [ongoing] [onset]"""
import sys, types, json
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import phase_lib as ph
sys.path.insert(0, f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts')
import fpmatch_lib as fpa

ph.check_prereg()
WHAT = sys.argv[1:] or ['ongoing', 'onset']; F = lab.F; FOLDS = (0, 1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
verdict = {}


def dec(p6):
    L = p6.shape[-1]; return lab.decode(p6.ravel()).reshape(F, L)


if 'ongoing' in WHAT:
    D = f'{ph.ARMD}/og_np_phase'
    oof = pd.concat([pd.read_pickle(f'{D}/fold{k}.pkl')['df'] for k in FOLDS], ignore_index=True)
    rows = []
    for s in ('official', 'offlike', 'pseudo', 'runstart'):
        A = oof[(oof.set == s) & (oof.arm == 'og_np_base')]; B = oof[(oof.set == s) & (oof.arm == 'og_np_phase')]
        rows.append(og.delta_row(A, B, 'phase', 'train_' + s))
    pms = [pd.read_pickle(f'{D}/pm_f{k}.pkl') for k in FOLDS]
    PMZ = {panel: og.build_pm(panel) for panel in lab.PANELS}
    lab_rows = []; sc = {}
    for panel in lab.PANELS:
        for s in og.PM_SETS:
            z = PMZ[panel][s]
            if not len(z['win']):
                continue
            pb = sum(p[('base', panel, s)].astype(np.float32) for p in pms) / 2; pn = sum(p[('new', panel, s)].astype(np.float32) for p in pms) / 2
            rc = types.SimpleNamespace(truth=z['truth'], ev=z['ev'], qlast=z['qlast'], win=z['win'], name=s, panel=panel, cond=lab.ONGOING, L=z['qlast'].shape[1])
            a = lab.score_probs(rc, pb, arm='base', set_name=s); b = lab.score_probs(rc, pn, arm='phase', set_name=s)
            sc.setdefault(s, []).append((a, b))
            for j in range(len(z['win'])):
                lab_rows.append(dict(set=s, panel=panel, d=int(z['win'].d.iloc[j]), t=int(z['win'].t.iloc[j]), tod=z['win'].t.iloc[j] / 12,
                                     dI=b.iou.iloc[j] - a.iou.iloc[j], dn=int(dec(pn[j]).sum()) - int(dec(pb[j]).sum())))
    for s, lst in sc.items():
        a = pd.concat([x[0] for x in lst], ignore_index=True); b = pd.concat([x[1] for x in lst], ignore_index=True)
        bt = lab.paired_bootstrap(a, b); pa = a.groupby('panel').iou.mean(); pb_ = b.groupby('panel').iou.mean()
        r = dict(cmp='phase', set='pm_' + s, n=bt['n_windows'], base=lab.aggregate(a), new=lab.aggregate(b), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                 panels_pos=f'{int((pb_ - pa > 0).sum())}/{len(pa)}')
        for hh in (1, 3, 6):
            r[f'd_s{hh}'] = lab.aggregate(b, f's{hh}') - lab.aggregate(a, f's{hh}')
        rows.append(r)
    L_ = pd.DataFrame(lab_rows); L_['ph6'] = L_.t % 6
    print('\npseudo-March per-window dI by origin phase (info):'); print(L_.groupby(['set', 'ph6']).dI.agg(['mean', 'count']).round(4).unstack(0).to_string())
    offr = []
    o = [pd.read_pickle(f'{D}/off_f{k}.pkl') for k in FOLDS]
    for split in ('validation', 'private'):
        for panel in lab.PANELS:
            dd = [oo[(panel, split)] for oo in o]; d0 = dd[0]
            pn = sum(x['P_new'] for x in dd) / 2; pbb = sum(x['P_base'] for x in dd) / 2
            for j, w in enumerate(d0['window_id']):
                Da, Db = dec(pbb[j]), dec(pn[j])
                offr.append(dict(split=split, window_id=w, panel=panel, tod=d0['tod'][j], slot=int(d0['slot'][j]), nA=int(Da.sum()), nB=int(Db.sum()),
                                 changed=int((Da ^ Db).sum())))
    O = pd.DataFrame(offr); O['dn'] = O.nB - O.nA
    fm = {}
    for split in ('validation', 'private'):
        off = O[O.split == split].reset_index(drop=True); f = fpa.forecast(L_, off)
        fm[split] = dict(win=int((off.changed > 0).sum()), cells=int(off.changed.sum()), dn=int(off.dn.sum()), fm_lb=float(0.15 * f.mean()),
                         origin_ph1=int((off.slot % 6 == 1).sum()))
    O.to_csv(f'{ph.RES}/official_footprint.csv', index=False)
    R = pd.DataFrame(rows); print('\n===== ongoing og_np_phase vs og_np_base'); print(R.round(4).to_string(index=False)); R.to_csv(f'{ph.RES}/ongoing_deltas.csv', index=False)
    print('FM-PM:', json.dumps(fm))
    g = R.set_index('set')
    G1 = bool(g.loc['train_offlike', 'diff'] >= 0.005 and g.loc['train_runstart', 'diff'] >= 0.005)
    pmo = g.loc['pm_offlike_m']
    G2 = bool(pmo['diff'] >= 0.005 and pmo['lo'] > 0 and int(pmo['panels_pos'].split('/')[0]) >= 6)
    G3 = bool(fm['validation']['fm_lb'] >= 0.0015)
    verdict['ongoing'] = dict(G1=G1, G2=G2, G3=G3, GO=G1 and G2 and G3, fm=fm)
    for k in FOLDS:
        gs = pd.read_csv(f'{D}/gain_f{k}.csv', index_col=0).iloc[:, 0]
        print(f'gain share fold {k}:', gs[ph.PH].round(4).to_dict(), 'total', round(float(gs[ph.PH].sum()), 4))

if 'onset' in WHAT:
    PMR = pd.read_pickle(f'{ph.RES}/onset_pm.pkl')
    key = ['family', 'panel', 'd', 't']
    avg = {a: PMR[PMR.a == a].groupby(key, as_index=False).agg(iou=('iou', 'mean')) for a in ('base', 'phase')}
    for a in avg:
        avg[a]['set'] = 'march_on'; avg[a]['cond'] = lab.ONSET; avg[a]['arm'] = a
    print('\n===== onset pseudo-March (214 windows expected), seed-avg IoU')
    print('n windows', len(avg['base']))
    bt = lab.paired_bootstrap(avg['base'], avg['phase'])
    pa = avg['base'].groupby('panel').iou.mean(); pb = avg['phase'].groupby('panel').iou.mean(); npos = int((pb - pa > 0).sum())
    print(f"base {lab.aggregate(avg['base']):.4f}  phase {lab.aggregate(avg['phase']):.4f}  diff {bt['diff']:+.4f} [{bt['lo']:+.4f}, {bt['hi']:+.4f}]  panels {npos}/{len(pa)}")
    print(pd.DataFrame({'base': pa, 'phase': pb, 'delta': pb - pa, 'n': avg['base'].groupby('panel').size()}).round(4).to_string())
    for s in sorted(PMR.seed.unique()):
        a = PMR[(PMR.a == 'base') & (PMR.seed == s)]; b = PMR[(PMR.a == 'phase') & (PMR.seed == s)]
        print(f'seed {s}: base {lab.aggregate(a):.4f} phase {lab.aggregate(b):.4f} diff {lab.aggregate(b) - lab.aggregate(a):+.4f}')
    a = avg['base'].merge(avg['phase'], on=key); a['ph1'] = a.t % 6 == 1
    print('info: by origin phase (:05/:35 origin)'); print(a.groupby('ph1').apply(lambda g: pd.Series(dict(n=len(g), d=(g.iou_y - g.iou_x).mean()))).round(4))
    verdict['onset'] = dict(diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], panels=f'{npos}/{len(pa)}',
                            GO=bool(bt['diff'] >= 0.01 and bt['lo'] > 0 and npos >= 6))
    try:
        oo = pd.read_pickle(f'{ph.RES}/onset_oof.pkl')
        A = oo[oo.arm == 'base']; B = oo[oo.arm == 'phase']
        print('info: onset train OOF offlike (folds 0+1, seed 0)'); print(pd.DataFrame([og.delta_row(A, B, 'phase', 'train_offlike_onset')]).round(4).to_string(index=False))
    except FileNotFoundError:
        print('onset OOF not available')
print('\nVERDICT', json.dumps(verdict, default=str))
old = json.load(open(f'{ph.RES}/verdict.json')) if __import__('os').path.exists(f'{ph.RES}/verdict.json') else {}
old.update(verdict); json.dump(old, open(f'{ph.RES}/verdict.json', 'w'), indent=1, default=str)
