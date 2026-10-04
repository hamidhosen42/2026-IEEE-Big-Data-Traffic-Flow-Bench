"""EXPERIMENT recon, step 2: scan the FD-reconciliation / TV constants of kaggle_jobs/swap_fdrecon.py on the Task 1 replica.
usage: exp_recon_scan.py [stages, e.g. ABCDEFG] [months, e.g. ABCD]
Every variant is evaluated on every (bag, pseudo-month); the per-day sufficient statistics go to
results/recon/scan_stats.pkl (resumable: finished variants are skipped) and a long table to results/recon/scan_long.csv.
recon() and tv() are the kernel's own functions (analysis/t1lab/t1lab.py compiles them from swap_fdrecon.py); GATE / A /
LOGR_MAX are shared by both steps exactly as in the kernel."""
import itertools, os, pickle, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 't1lab'))
import t1lab as L

OUT = os.path.join(HERE, 'results', 'recon'); os.makedirs(OUT, exist_ok=True)
STAGES = sys.argv[1] if len(sys.argv) > 1 else 'ABCDEFG'
MONTHS = list(sys.argv[2]) if len(sys.argv) > 2 else list('ABCD')
SEEDS = (0, 1, 2)
BAGS = ('nofd', 'fd', 'mix')
PKL = os.path.join(OUT, 'scan_stats.pkl')
TV0 = L.PROD_DEFAULT['tv']


def name_of(steps):
    p = []
    for s in steps:
        if s[0] == 'R':
            p.append(f'R[g{s[1]:g},a{s[2]:g},l{s[3]:g}]')
        elif s[0] == 'T':
            p.append(f'T[g{s[1]:g},a{s[2]:g},l{s[3]:g},tv{s[4][0]:g}/{s[4][1]:g}/{s[4][2]:g}/{s[4][3]:g}]')
        elif s[0] == 'F':
            p.append(f'F[g{s[1]:g},b{s[2]:g},{s[3]}]')
        elif s[0] == 'C':
            p.append(f'C[g{s[1]:g},rho{s[2]:g}]')
    return '>'.join(p) or 'raw'


def RT(g=0.6, a=0.75, l=0.5, tv=TV0):
    return (L.R(g, a, l), L.T(g, a, l, tv))


def variants(stage):
    V = []
    if stage == 'A':                                  # replica check: the known LB effects
        V += [(), (L.R(),), RT(), (L.T(),)]
    if stage == 'B':                                  # recon only, wide grid (cheap)
        for g, a, l in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 1.0, 2.0)):
            V.append((L.R(g, a, l),))
    if stage == 'C':                                  # recon then TV (kernel order), kernel coupling of GATE / A / LOGR_MAX
        for g, a, l in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0), (0.25, 0.5, 0.75, 1.0), (0.5, 1.0)):
            V.append(RT(g, a, l))
    if stage == 'D':                                  # TV strengths around the defaults (free / gate scaled, anchors scaled along)
        for sf, sg in itertools.product((0, 0.5, 1, 2, 4), (0, 0.5, 1, 2, 4)):
            V.append(RT(tv=(TV0[0] * sf, TV0[1] * sf, TV0[2] * sg, TV0[3] * sg)))
        for sa in (0, 0.33, 3, 10):                   # anchor strength alone
            V.append(RT(tv=(TV0[0], TV0[1] * sa, TV0[2], TV0[3] * sa)))
        for sf in (8, 16):                            # much stronger free-cell smoothing
            V.append(RT(tv=(TV0[0] * sf, TV0[1] * sf, TV0[2], TV0[3])))
    if stage == 'E':                                  # order: TV then recon; recon twice
        for g, a, l in ((0.6, 0.75, 0.5), (0.7, 0.75, 0.5), (0.8, 0.75, 0.5), (0.9, 0.75, 0.5), (0.6, 0.5, 0.5), (0.6, 1.0, 0.5), (0.6, 0.75, 1.0)):
            V.append((L.T(g, a, l), L.R(g, a, l)))
        V.append((L.R(), L.R(), L.T()))
    if stage == 'F':                                  # free-flow branch projection (new step, not in the kernel)
        for gf, b, keep in itertools.product((0.6, 0.9, 0.95, 0.97), (0.5, 1.0), ('q', 'k')):
            V.append((L.R(), ('F', gf, b, keep), L.T()))
        for gf, b, keep in ((0.95, 1.0, 'q'), (0.95, 1.0, 'k'), (0.6, 1.0, 'q')):
            V.append((L.R(), L.T(), ('F', gf, b, keep)))
            V.append((('F', gf, b, keep),))
    if stage == 'G':                                  # exact projection onto the congested branch (new step, not in the kernel)
        for g, rho in itertools.product((0.6, 0.8, 0.95), (5.0, 10.0, 20.0, 40.0, 80.0, 1e6)):
            V.append((('C', g, rho),))
            V.append((('C', g, rho), L.T(g, 0.75, 0.5)))
    return V


def main():
    Z = L.Data()
    months = [m for m in MONTHS if all(os.path.exists(L.member_path(L.MONTH_OUTER[m], s, k)) for s in SEEDS for k in ('nofd', 'fd'))]
    L.log('months with all members:', months, '| stages', STAGES)
    MO = {m: L.Month(Z, m) for m in months}
    base = {(b, m): L.load_bag(m, b, SEEDS) for b in BAGS for m in months}
    S = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
    for stage in STAGES:
        V = variants(stage); t = time.time(); n = 0
        for steps in V:
            nm = name_of(steps)
            for b in BAGS:
                for m in months:
                    if (b, m, nm) in S:
                        continue
                    v, q = L.run_pipeline(MO[m], *base[(b, m)], steps)
                    S[(b, m, nm)] = MO[m].stats(v, q).astype(np.float64); n += 1
        tmp = PKL + '.tmp'
        pickle.dump(S, open(tmp, 'wb')); os.replace(tmp, PKL)
        L.log(f'stage {stage}: {len(V)} variants, {n} new evaluations in {time.time() - t:.0f}s | stored {len(S)}')
    rows = []
    for (b, m, nm), st in S.items():
        if m not in MO:
            continue
        a = L.aggregate(st, MO[m].reg)
        rows.append(dict(bag=b, month=m, variant=nm, **{k: a[k] for k in a if not k.startswith('rmse') and not k.startswith('S_speed_') and not k.startswith('S_flow_')},
                         **{k: a[k] for k in a if k.startswith('rmse')}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'scan_long.csv'), index=False)
    L.log('rows', len(df), '->', os.path.join(OUT, 'scan_long.csv'))


if __name__ == '__main__':
    main()
