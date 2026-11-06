"""round8b_report.py -- ROUND 8b (PREREG results/round8/PREREG_8b.md asserted via round8b_lib; reading INTERPRETATION_8b.txt).
  repro   seed-0 pv (3 weights) for the available recipes (lab = old EXP_T2_OSP_MT protocol, prod, prodnoperm) vs the published
          0.8578 (cut 15, n 131) / 0.8883 (cut 20, n 88)
  main    seeds 0-19 of the production recipe: R0, seed 0, bag A (0-9), bag B (10-19), pooled, bag-of-20 (info); PASS-April-A gate"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
import lab
import round8b_lib as rb

warnings.filterwarnings('ignore')
T0 = time.time()
MODE = sys.argv[1]
PUB = {15: dict(bag=0.8578, B10=0.8580, B20=0.8523, B40=0.8515), 20: dict(bag=0.8883, B10=0.8812, B20=0.8894, B40=0.8801)}
EV = {c: rb.eval_set(c) for c in (15, 20)}


def probs(rec, cut, seed, w):
    f = rb.model_path(rec, cut, seed, w)
    if not os.path.exists(f):
        return None
    m = CatBoostClassifier(); m.load_model(f)
    return {p: m.predict_proba(s['X'], thread_count=4)[:, 1] for p, s in EV[cut].items()}


def mean_p(lst):
    return {p: np.mean([d[p] for d in lst], 0) for p in lst[0]}


def agg(df):
    return lab.aggregate(df)


if MODE == 'repro':
    out = {}
    for rec in ('lab', 'prod', 'prodnoperm'):
        for cut in (15, 20):
            ps = [probs(rec, cut, 0, w) for w in rb.WS]
            if any(p is None for p in ps):
                continue
            r = {f'B{w}': round(agg(rb.score(EV[cut], p)), 4) for w, p in zip(rb.WS, ps)}
            r['bag'] = round(agg(rb.score(EV[cut], mean_p(ps))), 4)
            r['n'] = int(len(rb.score(EV[cut], ps[0])))
            r['published'] = PUB[cut]
            out[f'{rec}_c{cut}'] = r
            print(rec, 'cut', cut, r, flush=True)
    json.dump(out, open(f'{rb.RES}/repro_8b.json', 'w'), indent=1)

if MODE == 'main':
    res = {}; lines = [f'ROUND 8b -- PREREG_8b.md sha1 {rb.check_prereg()} (asserted by every round8b_*.py); reading INTERPRETATION_8b.txt '
                       f'(sha1 1fd1a1e0, stamped 12:17:43 +06 before any round-8b model or number); recipe = production-exact pv.']
    for cut in (15, 20):
        P = {}
        for s in range(20):
            ps = [probs('prod', cut, s, w) for w in rb.WS]
            assert all(p is not None for p in ps), (cut, s)
            P[s] = ps                                                     # per seed: list over weights of panel -> probs
        seedp = {s: mean_p(P[s]) for s in range(20)}                      # single seed = mean over the 3 weights
        T = rb.score(EV[cut], seedp[0])
        iou = {s: rb.score(EV[cut], seedp[s]).iou.to_numpy() for s in range(20)}
        A, B = list(range(10)), list(range(10, 20))

        def bagp(seeds):                                                  # per weight: mean over seeds; then mean over weights
            return mean_p([mean_p([P[s][j] for s in seeds]) for j in range(len(rb.WS))])
        bA = rb.score(EV[cut], bagp(A)).iou.to_numpy(); bB = rb.score(EV[cut], bagp(B)).iou.to_numpy()
        b20 = rb.score(EV[cut], bagp(A + B)).iou.to_numpy()
        R0A = np.mean([iou[s] for s in A], 0); R0B = np.mean([iou[s] for s in B], 0)
        fr = lambda x: rb.frame(T, x)
        ds = lambda a, b: rb.delta_stats(fr(a), fr(b))
        r = dict(n=int(len(T)),
                 repA=ds(R0A, bA), repB=ds(R0B, bB), pooled=ds((R0A + R0B) / 2, (bA + bB) / 2),
                 seed0_vs_bagA=ds(iou[0], bA), seed0_vs_bagB=ds(iou[0], bB), seed0_vs_pooled_bag=ds(iou[0], (bA + bB) / 2),
                 bagA_vs_bag20=ds(bA, b20), bagB_vs_bag20=ds(bB, b20), R0_pooled_vs_bag20=ds((R0A + R0B) / 2, b20),
                 per_seed={s: round(agg(fr(iou[s])), 4) for s in range(20)},
                 levels=dict(seed0=agg(fr(iou[0])), R0A=agg(fr(R0A)), R0B=agg(fr(R0B)), bagA=agg(fr(bA)), bagB=agg(fr(bB)), bag20=agg(fr(b20))))
        res[cut] = r
    c15, c20 = res[15], res[20]; p = c15['pooled']
    gate = {'1_cut15_pooled_ge_0.008_and_lo_gt_0': bool(p['diff'] >= 0.008 and p['lo'] > 0),
            '2_each_replicate_cut15_ge_0.004': bool(c15['repA']['diff'] >= 0.004 and c15['repB']['diff'] >= 0.004),
            '3_cut15_nontie_70pct_pos_none_below_-0.02': bool(p['nontie'] > 0 and p['nontie_pos'] / p['nontie'] >= 0.7 and p['nontie_min'] >= -0.02),
            '4_cut20_pooled_ge_0': bool(c20['pooled']['diff'] >= 0)}
    V = dict(prereg_sha1=rb.SHA8B, results={str(k): v for k, v in res.items()}, gate=gate, PASS_April_A=all(gate.values()),
             runtime_report_s=round(time.time() - T0))

    def f(tag, d):
        return (f"{tag}: {d['base']:.4f} -> {d['new']:.4f}  {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  non-tie panels "
                f"{d['nontie_pos']}/{d['nontie']} positive (min {d['nontie_min']:+.4f})")
    for cut in (15, 20):
        r = res[cut]; lv = r['levels']
        lines += ['', f"cut {cut} (train March days < {cut}, evaluate days >= {cut}, n {r['n']}): seed0 {lv['seed0']:.4f}, R0 A {lv['R0A']:.4f}, "
                  f"R0 B {lv['R0B']:.4f}, bag A {lv['bagA']:.4f}, bag B {lv['bagB']:.4f}, bag-20 {lv['bag20']:.4f}",
                  f('  replicate A  R0 -> bag', r['repA']), f('  replicate B  R0 -> bag', r['repB']), f('  pooled       R0 -> bag', r['pooled']),
                  f"    pooled panel deltas {r['pooled']['panel_delta']}",
                  f('  info seed 0 -> bag A', r['seed0_vs_bagA']), f('  info seed 0 -> bag B', r['seed0_vs_bagB']),
                  f('  info seed 0 -> pooled bag', r['seed0_vs_pooled_bag']),
                  f('  info bag A -> bag-20', r['bagA_vs_bag20']), f('  info bag B -> bag-20', r['bagB_vs_bag20']),
                  f"  info single seeds: mean {np.mean(list(r['per_seed'].values())):.4f} sd {np.std(list(r['per_seed'].values())):.4f} "
                  f"range {min(r['per_seed'].values()):.4f}..{max(r['per_seed'].values()):.4f}; seed 0 rank "
                  f"{1 + sum(v > r['per_seed'][0] for v in r['per_seed'].values())}/20"]
    lines += ['', f"GATE PASS-April-A: {V['PASS_April_A']}  {gate}"]
    print('\n'.join(lines))
    json.dump(V, open(f'{rb.RES}/verdict_8b.json', 'w'), indent=1, default=str)
    open(f'{rb.RES}/report_8b.txt', 'w').write('\n'.join(lines) + '\n')
lab.log('done', MODE)
