"""pv2 report: applies the PRE-REGISTERED rule of pv2_lib.py to results/pv2/stage{A,B}.pkl.  Light (no training)."""
import numpy as np
import pandas as pd
import lab

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
KEY = ['panel', 'd', 't', 'cond']
CANDS = ['a1', 'a2', 'a3', 'b1', 'b2']


def seedmean(R, arm):
    G = R[R.arm == arm]
    g = G.groupby(KEY + ['family', 'set'], as_index=False)[['iou', 'n_pred', 'n_true']].mean()
    g['arm'] = arm
    return g


def cmp(R, a, b, seed=None):
    if seed is None:
        A, B = seedmean(R, a), seedmean(R, b)
    else:
        A = R[(R.arm == a) & (R.seed == seed)]; B = R[(R.arm == b) & (R.seed == seed)]
    bt = lab.paired_bootstrap(A, B)
    m = A[KEY + ['iou']].merge(B[KEY + ['iou']], on=KEY, suffixes=('_a', '_b'))
    dl = (m.iou_b - m.iou_a).groupby(m.panel).mean()
    return dict(arm=b, iou=lab.aggregate(B), ref=lab.aggregate(A), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], pos=int((dl > 1e-9).sum()),
                neg=int((dl < -1e-9).sum()), changed=int((abs(m.iou_b - m.iou_a) > 1e-9).sum()), n=len(m))


A = pd.read_pickle('results/pv2/stageA.pkl')
B = pd.read_pickle('results/pv2/stageB.pkl')
print('== stage A (many-day pseudo-March, 131 windows of days >= 15): levels (seed-mean) ==')
print(pd.DataFrame([dict(arm=a, iou=lab.aggregate(seedmean(A, a)), s0=lab.aggregate(A[(A.arm == a) & (A.seed == 0)]),
                         s1=lab.aggregate(A[(A.arm == a) & (A.seed == 1)])) for a in dict.fromkeys(A.arm)]).round(4).to_string(index=False))
st = pd.read_pickle('results/pvon/step2_cut15_s1_S.pkl')
for mine, theirs in (('pv', 'B_bag'), ('A_s4', 'A_s4'), ('A', 'A')):
    j = A[(A.arm == mine) & (A.seed == 0)][KEY + ['iou']].merge(st[st.arm == theirs][KEY + ['iou']], on=KEY)
    print(f'parity seed 0 {mine} vs pvon_step2 {theirs}: n {len(j)}, identical {float((j.iou_x - j.iou_y).abs().lt(1e-9).mean()):.3f}')
rows = []
for c in CANDS:
    r = cmp(A, 'pv', c); r.update(s0=cmp(A, 'pv', c, 0)['diff'], s1=cmp(A, 'pv', c, 1)['diff'])
    ob = B[B.set == 'official']; oo = B[B.set == 'offlike']
    r['off40'] = cmp(ob, 'pv', c)['diff']; r['off40_pos'] = cmp(ob, 'pv', c)['pos']; r['offlike_f0'] = cmp(oo, 'pv', c)['diff']
    r['gate1'] = (r['diff'] >= 0.01) and (r['lo'] > 0) and (r['pos'] >= 6)
    r['gate2'] = r['off40'] >= 0
    r['ADOPT'] = r['gate1'] and r['gate2']
    rows.append(r)
T = pd.DataFrame(rows)
print('\n== PRE-REGISTERED decision table (candidate - pv) ==')
print(T.round(4).to_string(index=False))
print('\n== stage B levels (seed-mean) ==')
for s in ('official', 'offlike'):
    G = B[B.set == s]
    print(s, {a: round(lab.aggregate(seedmean(G, a)), 4) for a in dict.fromkeys(G.arm)})
print('\n== reference: pv - A_s4 (many-day) ==', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cmp(A, 'A_s4', 'pv').items()})
# per-panel deltas and the zero-overlap / size decomposition
pv = seedmean(A, 'pv')
for c in CANDS:
    g = seedmean(A, c).merge(pv[KEY + ['iou']], on=KEY, suffixes=('', '_pv'))
    print(f'{c} per-panel delta:', (g.iou - g.iou_pv).groupby(g.panel).mean().round(3).to_dict())
for a in ['A_s4', 'pv'] + CANDS:
    for seed in (0,):
        G = A[(A.arm == a) & (A.seed == seed)]
        loss = 1 - G.iou
        print(f'{a:5s} seed0 zero-overlap windows {int((G.iou == 0).sum()):3d} loss share {loss[G.iou == 0].sum() / max(loss.sum(), 1e-9):.3f}'
              f' under {loss[(G.n_pred < G.n_true) & (G.iou > 0)].sum() / max(loss.sum(), 1e-9):.3f}'
              f' over {loss[(G.n_pred > G.n_true) & (G.iou > 0)].sum() / max(loss.sum(), 1e-9):.3f} mean n_pred {G.n_pred.mean():.2f} (true {G.n_true.mean():.2f})')
T.to_csv('results/pv2/decision.csv', index=False)
