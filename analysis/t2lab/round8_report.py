"""round8_report.py -- ROUND 8 pre-registered verdicts (results/round8/PREREG.md asserted via round8_lib; reading fixed in
INTERPRETATION.txt before any number).  Order: BAG, DEC (on the BAG probabilities), FEAT (decoder chosen by the DEC verdict), triage.
Usage: round8_report.py [bag] [dec] [feat] [triage]   (default: all available).  Writes results/round8/verdict.json (merged)."""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
import lab
import round8_lib as r8

warnings.filterwarnings('ignore')
T_START = time.time()
WHAT = sys.argv[1:] or ['bag', 'dec', 'feat', 'triage']
E = r8.e1_all()
PE1 = pd.read_pickle(f'{r8.CD}/pred_e1.pkl')
PE2 = pd.read_pickle(f'{r8.CD}/pred_e2.pkl') if os.path.exists(f'{r8.CD}/pred_e2.pkl') else None
E2ROWS = {k: r8.e2_rows(k) for k in range(4)}
A, B, S10 = list(range(10)), list(range(10, 20)), list(range(10))
VJ = f'{r8.RES}/verdict.json'
V = json.load(open(VJ)) if os.path.exists(VJ) else {}
V['prereg_sha1'] = r8.check_prereg()
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def ident(p):
    return p


# ---------------------------------------------------------------- scoring helpers (window order fixed: panels, windows / folds, panels, windows)
def e1_scores(getp, decoder):
    dfs = []
    for p, e in E.items():
        P6 = r8.p6_from_rows(e['W'], e['L'], e['wi'], e['li'], getp(p))
        dfs.append(r8.score(e['truth'], e['ev'], P6, range(e['W']), decoder, dict(panel=p, win=e['win'])))
    return pd.concat(dfs, ignore_index=True)


def e2_scores(getp, decoder):
    dfs = []
    for k in range(4):
        for p, r in E2ROWS[k].items():
            rc = r['rc']
            P6 = r8.p6_from_rows(len(rc.win), rc.L, r['wi'], r['li'], getp(k, p))
            dfs.append(r8.score(rc.truth, rc.ev, P6, r['widx'], decoder, dict(panel=p, win=rc.win)))
    return pd.concat(dfs, ignore_index=True)


T1 = e1_scores(lambda p: PE1[p]['base'][0], r8.decode_prod)       # key templates
T2 = e2_scores(lambda k, p: PE2[(k, p)]['base'][0], r8.decode_prod) if PE2 is not None else None


def fr(tmpl, iou):
    return r8.frame(tmpl, iou)


def ds(tmpl, base, new):
    return r8.delta_stats(fr(tmpl, base), fr(tmpl, new))


def e1(getp, dec=r8.decode_prod):
    df = e1_scores(getp, dec); assert (df[r8.KEY].values == T1[r8.KEY].values).all(); return df.iou.to_numpy()


def e2(getp, dec=r8.decode_prod):
    df = e2_scores(getp, dec); assert (df[r8.KEY].values == T2[r8.KEY].values).all(); return df.iou.to_numpy()


def bag1(key, seeds):
    return lambda p: np.mean(PE1[p][key][seeds], 0)


def bag2(key, seeds=S10):
    return lambda k, p: np.mean(PE2[(k, p)][key][seeds], 0)


def short(d):
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}


def fmt(tag, d):
    return (f"{tag}: base {d['base']:.4f} new {d['new']:.4f} diff {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n {d['n']}, "
            f"non-tie panels {d['nontie_pos']}/{d['nontie']} pos (min {d['nontie_min']:+.4f}), non-neg {d['panels_nonneg']}/{d['panels']}")


lines = [f"ROUND 8 verdicts -- PREREG.md sha1 {V['prereg_sha1']} (asserted by every round8_*.py); reading INTERPRETATION.txt "
         f"(sha1 b4ad8ba6, saved 11:54:39 +06 before any round-8 number)."]

# ================================================================ BAG
single1 = {s: e1(lambda p, s=s: PE1[p]['base'][s]) for s in range(20)}
R0A = np.mean([single1[s] for s in A], 0); R0B = np.mean([single1[s] for s in B], 0)
bagA = e1(bag1('base', A)); bagB = e1(bag1('base', B))
single2 = {s: e2(lambda k, p, s=s: PE2[(k, p)]['base'][s]) for s in S10}
R02 = np.mean([single2[s] for s in S10], 0); bag2b = e2(bag2('base'))
if 'bag' in WHAT:
    rA = ds(T1, R0A, bagA); rB = ds(T1, R0B, bagB); rP = ds(T1, (R0A + R0B) / 2, (bagA + bagB) / 2)
    r2 = ds(T2, R02, bag2b)
    nt_ok = rP['nontie'] > 0 and rP['nontie_pos'] / rP['nontie'] >= 0.7 and rP['nontie_min'] >= -0.02
    april = dict(pooled=bool(rP['diff'] >= 0.005 and rP['lo'] > 0), replicates=bool(rA['diff'] >= 0.002 and rB['diff'] >= 0.002),
                 panels=bool(nt_ok), e2=bool(r2['diff'] >= -0.002))
    march = dict(e2_diff=bool(r2['diff'] >= 0.002), e2_lo=bool(r2['lo'] > 0), panels=bool(r2['panels_nonneg'] >= 6))
    # info
    s0A = ds(T1, single1[0], bagA); s02 = ds(T2, single2[0], bag2b)
    per_seed1 = {s: lab.aggregate(fr(T1, single1[s])) for s in range(20)}; per_seed2 = {s: lab.aggregate(fr(T2, single2[s])) for s in S10}
    sp_single = {s: e1(lambda p, s=s: PE1[p]['base_sp4'][s]) for s in range(20)}
    spA = e1(bag1('base_sp4', A)); spB = e1(bag1('base_sp4', B))
    spP = ds(T1, (np.mean([sp_single[s] for s in A], 0) + np.mean([sp_single[s] for s in B], 0)) / 2, (spA + spB) / 2)
    spA_ = ds(T1, np.mean([sp_single[s] for s in A], 0), spA); spB_ = ds(T1, np.mean([sp_single[s] for s in B], 0), spB)
    # reproduction of the EXP_T2_PHASE2 E2 seeds 0-2 (same recipe)
    rep = {}
    for s in (0, 1, 2):
        old = pd.concat([pd.read_pickle(f'{lab.ROOT}/analysis/t2lab/results/phase2/oof_f{k}.pkl') for k in range(4)])
        old = old[(old.a == 'base') & (old.seed == s)][['panel', 'd', 't', 'iou']]
        m = fr(T2, single2[s]).merge(old, on=['panel', 'd', 't'], suffixes=('', '_old'))
        rep[s] = dict(n=len(m), max_abs=float((m.iou - m.iou_old).abs().max()))
    V['BAG'] = dict(E1_repA=short(rA), E1_repB=short(rB), E1_pooled=short(rP), E2=short(r2), PASS_April=all(april.values()),
                    gates_April=april, PASS_March=all(march.values()), gates_March=march,
                    info=dict(seed0_vs_bagA_E1=short(s0A), seed0_vs_bag_E2=short(s02), E1_per_seed={k: round(v, 4) for k, v in per_seed1.items()},
                              E2_per_seed={k: round(v, 4) for k, v in per_seed2.items()}, sp4_E1_pooled=short(spP), sp4_repA=short(spA_),
                              sp4_repB=short(spB_), e2_reproduces_phase2=rep))
    lines += ['', 'ARM BAG -- bag of 10 seeds (mean probability, production decoder) vs single seeds (R0 = per-window mean of the single-seed IoUs)',
              fmt('  E1 replicate A (seeds 0-9)  ', rA), fmt('  E1 replicate B (seeds 10-19)', rB), fmt('  E1 pooled                   ', rP),
              fmt('  E2 4 folds (seeds 0-9)      ', r2),
              f"  PASS-April {V['BAG']['PASS_April']}  {april}   (pooled >= +0.005 & lo > 0; each replicate >= +0.002; >= 70 % non-tie panels "
              f"positive, none < -0.02; E2 >= -0.002)",
              f"  PASS-March {V['BAG']['PASS_March']}  {march}   (E2 >= +0.002, lo > 0, >= 6/8 panels non-negative)",
              fmt('  info seed 0 alone vs bag A, E1', s0A), fmt('  info seed 0 alone vs bag, E2  ', s02),
              f"  info E1 single seeds: mean {np.mean(list(per_seed1.values())):.4f} sd {np.std(list(per_seed1.values())):.4f} "
              f"range {min(per_seed1.values()):.4f}..{max(per_seed1.values()):.4f}; E2 single seeds sd {np.std(list(per_seed2.values())):.4f}",
              fmt('  info sp4 prior, E1 pooled     ', spP), f"  info sp4 replicates A {spA_['diff']:+.4f}, B {spB_['diff']:+.4f}",
              f"  check: E2 single seeds 0-2 reproduce EXP_T2_PHASE2 per window: {rep}"]
    print('\n'.join(lines[-14:]))

# ================================================================ DEC
iso_maps = {}


def iso_fit(key, folds):
    xs, ys = [], []
    for k in folds:
        for p, r in E2ROWS[k].items():
            pb = np.mean(PE2[(k, p)][key][S10], 0); e = r['e']
            xs.append(pb[e]); ys.append(r['y'][e].astype(float))
    m = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds='clip')
    m.fit(np.concatenate(xs), np.concatenate(ys)); return m


def iso_get(key, k):
    if (key, k) not in iso_maps:
        iso_maps[(key, k)] = iso_fit(key, range(4) if k == 'all' else [j for j in range(4) if j != k])
    return iso_maps[(key, k)].predict


def iso_e1(key):
    return iso_get(key, 'all')


def iso_e2(key, k):
    return iso_get(key, k)


def e1_var(key, seeds, variant):
    """E1 IoU of the bag of `seeds` (prediction key) under decoder variant prod | exact | iso_exact | iso_prod."""
    if variant == 'prod':
        return e1(bag1(key, seeds))
    if variant == 'exact':
        return e1(bag1(key, seeds), r8.decode_exact)
    f = iso_e1(key)
    return e1(lambda p: f(np.mean(PE1[p][key][seeds], 0)), r8.decode_exact if variant == 'iso_exact' else r8.decode_prod)


def e2_var(key, variant):
    if variant == 'prod':
        return e2(bag2(key))
    if variant == 'exact':
        return e2(bag2(key), r8.decode_exact)
    return e2(lambda k, p: iso_e2(key, k)(np.mean(PE2[(k, p)][key][S10], 0)), r8.decode_exact if variant == 'iso_exact' else r8.decode_prod)


DEC_PROPOSED = dict(april=None, march=None)
if 'dec' in WHAT or 'feat' in WHAT:
    res = {}
    prodA, prodB, prod2 = bagA, bagB, bag2b
    for name, var in (('raw', 'exact'), ('iso', 'iso_exact')):
        dA = e1_var('base', A, var); dB = e1_var('base', B, var); d2 = e2_var('base', var)
        rP = ds(T1, (prodA + prodB) / 2, (dA + dB) / 2); r2 = ds(T2, prod2, d2)
        april = dict(e1=bool(rP['diff'] >= 0.005 and rP['lo'] > 0),
                     panels=bool(rP['nontie'] > 0 and rP['nontie_pos'] / rP['nontie'] >= 0.7), e2=bool(r2['diff'] >= -0.002))
        march = dict(e2_diff=bool(r2['diff'] >= 0.003), e2_lo=bool(r2['lo'] > 0), panels=bool(r2['panels_nonneg'] >= 6))
        res[name] = dict(E1_pooled=short(rP), E1_repA=short(ds(T1, prodA, dA)), E1_repB=short(ds(T1, prodB, dB)), E2=short(r2),
                         PASS_April=all(april.values()), gates_April=april, PASS_March=all(march.values()), gates_March=march)
        if name == 'iso':                                 # info: same isotonic probabilities through the production decoder
            pA = e1_var('base', A, 'iso_prod'); pB = e1_var('base', B, 'iso_prod'); p2 = e2_var('base', 'iso_prod')
            res[name]['info_vs_prod_on_iso_E1'] = short(ds(T1, (pA + pB) / 2, (dA + dB) / 2))
            res[name]['info_vs_prod_on_iso_E2'] = short(ds(T2, p2, d2))
            res[name]['info_prod_iso_vs_prod_raw_E1'] = short(ds(T1, (prodA + prodB) / 2, (pA + pB) / 2))
            res[name]['info_prod_iso_vs_prod_raw_E2'] = short(ds(T2, prod2, p2))
    for fam, key in (('april', 'PASS_April'), ('march', 'PASS_March')):
        ok = [n for n in res if res[n][key]]
        DEC_PROPOSED[fam] = max(ok, key=lambda n: res[n]['E2']['diff']) if ok else None
    V['DEC'] = dict(variants=res, proposed=DEC_PROPOSED)
    lines += ['', 'ARM DEC -- exact expected-IoU top-m decoding on the BAG probabilities vs the production decoder on the raw BAG probabilities']
    for n, r in res.items():
        lines += [f'  variant {n}:', fmt('    E1 pooled (A+B)', r['E1_pooled']),
                  f"    E1 replicates A {r['E1_repA']['diff']:+.4f}, B {r['E1_repB']['diff']:+.4f}", fmt('    E2 4 folds     ', r['E2']),
                  f"    PASS-April {r['PASS_April']} {r['gates_April']}  (E1 >= +0.005 & lo > 0, >= 70 % non-tie panels positive, E2 >= -0.002)",
                  f"    PASS-March {r['PASS_March']} {r['gates_March']}  (E2 >= +0.003, lo > 0, >= 6/8 non-negative)"]
        if n == 'iso':
            lines += [fmt('    info iso: exact vs production decoder on the SAME isotonic probabilities, E1', r['info_vs_prod_on_iso_E1']),
                      fmt('    info iso: same, E2', r['info_vs_prod_on_iso_E2']),
                      fmt('    info production decoder: isotonic vs raw probabilities, E1', r['info_prod_iso_vs_prod_raw_E1']),
                      fmt('    info production decoder: isotonic vs raw probabilities, E2', r['info_prod_iso_vs_prod_raw_E2'])]
    lines += [f"  proposed: April {DEC_PROPOSED['april']}, March {DEC_PROPOSED['march']}"]
    print('\n'.join(lines[-20:]))

# ================================================================ FEAT
if 'feat' in WHAT:
    have1 = all(np.isfinite(PE1[p]['feat']).all() for p in E); have2 = all(np.isfinite(v['feat']).all() for v in PE2.values())
    assert have1 and have2, ('FEAT predictions incomplete', have1, have2)
    VAR = {'raw': 'exact', 'iso': 'iso_exact', None: 'prod'}
    v1 = VAR[DEC_PROPOSED['april']]; v2 = VAR[DEC_PROPOSED['march']]
    out = {}
    for tag, (dv1, dv2) in (('binding', (v1, v2)), ('info_production_decoder', ('prod', 'prod'))):
        if tag != 'binding' and (dv1, dv2) == (v1, v2):
            continue
        bA = e1_var('base', A, dv1); bB = e1_var('base', B, dv1); fA = e1_var('feat', A, dv1); fB = e1_var('feat', B, dv1)
        b2 = e2_var('base', dv2); f2 = e2_var('feat', dv2)
        rA = ds(T1, bA, fA); rB = ds(T1, bB, fB); rP = ds(T1, (bA + bB) / 2, (fA + fB) / 2); r2 = ds(T2, b2, f2)
        april = dict(pooled=bool(rP['diff'] >= 0.008 and rP['lo'] > 0), replicates=bool(rA['diff'] >= 0.003 and rB['diff'] >= 0.003),
                     panels=bool(rP['nontie'] > 0 and rP['nontie_pos'] / rP['nontie'] >= 0.7 and rP['nontie_min'] >= -0.02),
                     e2=bool(r2['diff'] >= -0.002))
        march = dict(e2_diff=bool(r2['diff'] >= 0.003), e2_lo=bool(r2['lo'] > 0), panels=bool(r2['panels_nonneg'] >= 6))
        out[tag] = dict(decoder_E1=dv1, decoder_E2=dv2, E1_repA=short(rA), E1_repB=short(rB), E1_pooled=short(rP), E2=short(r2),
                        PASS_April=all(april.values()), gates_April=april, PASS_March=all(march.values()), gates_March=march)
    # info: single-seed FEAT vs single-seed base (R0 vs R0)
    sf = {s: e1(lambda p, s=s: PE1[p]['feat'][s]) for s in range(20)}
    out['info_single_seed_R0_E1'] = short(ds(T1, (R0A + R0B) / 2, np.mean([sf[s] for s in range(20)], 0)))
    sf2 = {s: e2(lambda k, p, s=s: PE2[(k, p)]['feat'][s]) for s in S10}
    out['info_single_seed_R0_E2'] = short(ds(T2, R02, np.mean([sf2[s] for s in S10], 0)))
    out['info_per_seed_E1'] = {s: round(lab.aggregate(fr(T1, sf[s])) - lab.aggregate(fr(T1, single1[s])), 4) for s in range(20)}
    V['FEAT'] = out
    b = out['binding']
    lines += ['', f"ARM FEAT -- bag-10 FEAT (69 + 6 precursor columns) vs bag-10 base, same seeds; decoder E1 {b['decoder_E1']}, E2 {b['decoder_E2']} "
              f"(DEC proposed April {DEC_PROPOSED['april']}, March {DEC_PROPOSED['march']})",
              fmt('  E1 replicate A (seeds 0-9)  ', b['E1_repA']), fmt('  E1 replicate B (seeds 10-19)', b['E1_repB']),
              fmt('  E1 pooled                   ', b['E1_pooled']), fmt('  E2 4 folds (seeds 0-9)      ', b['E2']),
              f"  PASS-April {b['PASS_April']}  {b['gates_April']}   (pooled >= +0.008 & lo > 0; each replicate >= +0.003; >= 70 % non-tie positive, "
              f"none < -0.02; E2 >= -0.002)",
              f"  PASS-March {b['PASS_March']}  {b['gates_March']}   (E2 >= +0.003, lo > 0, >= 6/8 non-negative)"]
    if 'info_production_decoder' in out:
        o = out['info_production_decoder']
        lines += [fmt('  info production decoder, E1 pooled', o['E1_pooled']), fmt('  info production decoder, E2', o['E2'])]
    lines += [fmt('  info single-seed FEAT vs single-seed base (R0), E1', out['info_single_seed_R0_E1']),
              fmt('  info single-seed FEAT vs single-seed base (R0), E2', out['info_single_seed_R0_E2']),
              f"  info per-seed E1 deltas (FEAT - base): {out['info_per_seed_E1']}"]
    print('\n'.join(lines[-12:]))

# ================================================================ triage
if 'triage' in WHAT:
    from catboost import CatBoostClassifier, Pool
    tri = {}
    for rep, seeds in (('A', A), ('B', B)):
        df = e1_scores(bag1('base', seeds), r8.decode_prod)
        zo = df[(df.union > 0) & (df.inter == 0)]
        cats = []
        for _, w in zo.iterrows():
            e = E[w.panel]; i = int(np.where((e['win'].d == w.d) & (e['win'].t == w.t))[0][0])
            pl = np.full(e['L'], -1.0); m = e['wi'] == i; pl[e['li'][m]] = np.mean(PE1[w.panel]['base'][seeds], 0)[m]
            tl = np.where(e['truth'][i, lab.F - 1] & e['ev'][i, lab.F - 1])[0]
            if not len(tl):
                cats.append('no true T+30 link'); continue
            rk = min(1 + int((pl > pl[j]).sum()) for j in tl)
            cats.append('rank 1' if rk == 1 else 'rank 2-3' if rk <= 3 else 'rank 4-5' if rk <= 5 else 'rank > 5')
        c = pd.Series(cats).value_counts()
        tri[rep] = dict(zero_overlap_windows=int(len(zo)), of=int(len(df)), counts=c.to_dict(),
                        share={k: round(v / max(len(zo), 1), 3) for k, v in c.items()}, windows=zo[['panel', 'd', 't']].values.tolist())
    # SHAP (replicate A models) on all E1 step-6 rows and on the rows of the replicate-A zero-overlap windows
    Xall = np.concatenate([E[p]['Xm'] for p in E]); zkey = {(p, d, t) for p, d, t in tri['A']['windows']}
    zrow = np.concatenate([np.array([(p, int(E[p]['win'].d.iloc[i]), int(E[p]['win'].t.iloc[i])) in zkey for i in E[p]['wi']]) for p in E])
    sh = []
    t0 = time.time()
    for s in A:
        m = CatBoostClassifier(); m.load_model(r8.E1_MODEL.format(s=s))
        sv = m.get_feature_importance(data=Pool(Xall), type='ShapValues', thread_count=lab.THREADS)[:, :-1]
        sh.append((np.abs(sv).mean(0), np.abs(sv[zrow]).mean(0)))
    lab.log('SHAP', f'{time.time() - t0:.0f}s', Xall.shape, 'zero-overlap rows', int(zrow.sum()))
    allm = pd.Series(np.mean([a for a, _ in sh], 0), index=r8.COLS); zm = pd.Series(np.mean([b for _, b in sh], 0), index=r8.COLS)
    PT = ['pos', 'lanes', 'cap_lane', 'cap_ratio_hi', 'cap_ratio_lo', 'length', 'tod', 'dow', 'h']
    grp = {c: ('prior' if c in lab.PRIOR3 else 'position-time' if c in PT else 'state') for c in r8.COLS}
    g = pd.DataFrame({'all_rows': allm.groupby(grp).sum(), 'zero_overlap_rows': zm.groupby(grp).sum()})
    ramp = [c for c in r8.COLS if c.startswith('ramp_')]
    g.loc['(state: ramp part)'] = [allm[ramp].sum(), zm[ramp].sum()]
    g['share_all'] = g.all_rows / allm.sum(); g['share_zero'] = g.zero_overlap_rows / zm.sum()
    tri['shap_groups'] = g.round(4).to_dict(); tri['shap_top10_all'] = allm.sort_values(ascending=False).head(10).round(4).to_dict()
    tri['shap_top10_zero'] = zm.sort_values(ascending=False).head(10).round(4).to_dict()
    V['TRIAGE'] = {k: (v if k not in ('A', 'B') else {kk: vv for kk, vv in v.items() if kk != 'windows'}) for k, v in tri.items()}
    lines += ['', 'TRIAGE (no gate) -- E1 windows with zero overlap under the base bag (production decoder)']
    for rep in ('A', 'B'):
        t_ = tri[rep]
        lines += [f"  bag {rep}: {t_['zero_overlap_windows']}/{t_['of']} windows; best rank of a true T+30 link: {t_['counts']} share {t_['share']}"]
    lines += ['  mean |SHAP| per group (10 replicate-A models): ' + g.round(4).to_string().replace('\n', '\n    '),
              f"  top-10 columns all rows: {tri['shap_top10_all']}", f"  top-10 columns zero-overlap rows: {tri['shap_top10_zero']}"]
    print('\n'.join(lines[-8:]))

V['report_runtime_s'] = round(time.time() - T_START)
json.dump(V, open(VJ, 'w'), indent=1, default=str)
open(f'{r8.RES}/report_{"_".join(WHAT)}.txt', 'w').write('\n'.join(lines) + '\n')
lab.log('report done', WHAT)
