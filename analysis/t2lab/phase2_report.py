"""phase2_report.py -- pre-registered PASS/FAIL of EXP_T2_PHASE2 (results/phase2/PREREG.txt, asserted)."""
import json, os
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import phase_lib as ph

ph.check_prereg2()
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
KEY = ['family', 'panel', 'd', 't']
FRESH = list(range(5, 20)); ALL = list(range(20))
PM = pd.read_pickle(f'{ph.RES2}/pm.pkl')
out = {}


def seedavg(df, arm, seeds):
    g = df[(df.a == arm) & df.seed.isin(seeds)].groupby(KEY, as_index=False).agg(iou=('iou', 'mean'), n=('iou', 'size'))
    assert (g.n == len(seeds)).all()
    g['set'] = 'x'; g['cond'] = lab.ONSET; return g.drop(columns='n')


def e1(df, seeds, tag):
    A, B = seedavg(df, 'base', seeds), seedavg(df, 'phase', seeds)
    bt = lab.paired_bootstrap(A, B)
    pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean(); dp = pb - pa
    nt = dp[dp.abs() > 1e-9]
    per = {s: lab.aggregate(df[(df.a == 'phase') & (df.seed == s)]) - lab.aggregate(df[(df.a == 'base') & (df.seed == s)]) for s in seeds}
    m = A.merge(B, on=KEY); m['ph1'] = m.t % 6 == 1
    r = dict(tag=tag, n=bt['n_windows'], base=lab.aggregate(A), phase=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
             panels_pos=int((nt > 0).sum()), panels_nontie=len(nt), panels_min=float(nt.min()) if len(nt) else float('nan'),
             seeds_pos=int(sum(v > 0 for v in per.values())), n_seeds=len(seeds),
             d_ph1=float((m.iou_y - m.iou_x)[m.ph1].mean()), n_ph1=int(m.ph1.sum()), d_other=float((m.iou_y - m.iou_x)[~m.ph1].mean()))
    print(f"\n== E1 {tag}: base {r['base']:.4f} phase {r['phase']:.4f} diff {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
          f"panels {r['panels_pos']}/{r['panels_nontie']} non-tie (min {r['panels_min']:+.4f}), seeds>0 {r['seeds_pos']}/{len(seeds)}; "
          f":05/:35 origins (n {r['n_ph1']}) {r['d_ph1']:+.4f}, other {r['d_other']:+.4f} (raw window means)")
    print(pd.DataFrame({'base': pa, 'phase': pb, 'delta': dp, 'n': A.groupby('panel').size()}).round(4).to_string())
    print('per seed:', {s: round(v, 4) for s, v in per.items()})
    r['per_seed'] = per; r['per_panel'] = dp.to_dict()
    return r


base_pm = PM[PM.pv == 'base']
print('seed-0 base check (EXP_T2_PHASE / EXP_T2_OSP 0.6646):', round(lab.aggregate(base_pm[(base_pm.a == 'base') & (base_pm.seed == 0)]), 4))
out['E1_fresh'] = e1(base_pm, FRESH, 'seeds 5..19 (fresh, BINDING)')
out['E1_seen'] = e1(base_pm, list(range(5)), 'seeds 0..4 (seen; must reproduce EXP_T2_PHASE +0.0177)')
out['E1_all'] = e1(base_pm, ALL, 'seeds 0..19 pooled (info)')
sp = PM[PM.pv == 'sp4']
print('\nsp4 seed-0 base check (EXP_T2_OSP s4 0.7543):', round(lab.aggregate(sp[(sp.a == 'base') & (sp.seed == 0)]), 4))
out['E1_sp4_fresh'] = e1(sp, FRESH, 'sp4 prior, seeds 5..19 (info)')

# ---------------------------------------------------------------- E2
folds = [k for k in range(4) if os.path.exists(f'{ph.RES2}/oof_f{k}.pkl')]
OO = pd.concat([pd.read_pickle(f'{ph.RES2}/oof_f{k}.pkl') for k in folds], ignore_index=True)
key2 = ['family', 'panel', 'd', 't', 'cond', 'set', 'fold']


def e2(ks):
    g = OO[OO.k.isin(ks)]
    vc = ['iou', 's1', 's3', 's6', 'fn_new', 'fp_old', 'fp_new']
    A = g[g.a == 'base'].groupby(key2, as_index=False)[vc].mean(); B = g[g.a == 'phase'].groupby(key2, as_index=False)[vc].mean()
    A['arm'] = 'base'; B['arm'] = 'phase'
    r = og.delta_row(A, B, 'phase', f'E2 folds {ks}'); return r


rows = [e2([k]) for k in folds] + [e2([0, 1])] + ([e2([0, 1, 2, 3])] if folds == [0, 1, 2, 3] else [])
E2 = pd.DataFrame(rows); print('\n== E2 train OOF offlike onset (seed-avg 0..2)'); print(E2.round(4).to_string(index=False))
bind = E2.iloc[-1]; out['E2'] = E2.to_dict('records'); out['E2_binding'] = dict(set=bind['set'], diff=float(bind['diff']))

f = out['E1_fresh']
gates = dict(a=bool(f['diff'] >= 0.008 and f['lo'] > 0),
             b=bool(f['panels_nontie'] > 0 and f['panels_pos'] / f['panels_nontie'] >= 0.8 and f['panels_min'] >= -0.01),
             c=bool(bind['diff'] >= -0.002),
             d=bool(f['seeds_pos'] / f['n_seeds'] >= 0.7))
out['gates'] = gates; out['PASS'] = all(gates.values())
print('\nGATES', gates, '->', 'PASS' if out['PASS'] else 'FAIL', '| E2 binding =', bind['set'])
json.dump(out, open(f'{ph.RES2}/verdict.json', 'w'), indent=1, default=str)
