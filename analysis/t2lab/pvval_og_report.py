"""pvval ongoing tables (light: scoring + paired bootstrap only).
  pvval_og_report.py pm     pseudo-March days >= 15 from results/pvval/probs/{R,M0,M1}_15_5.pkl
  pvval_og_report.py harm   in-train harm test from results/pvval/harm_scores_H69_<variant>_5.pkl"""
import os, sys
import numpy as np
import pandas as pd
import lab
import pvval_lib as V
import pvog_lib as pv

what = sys.argv[1]
CUT = 15
pd.set_option('display.width', 250)

if what == 'pm':
    P6 = {}

    def put(tag, rename=None, only=None):
        if not os.path.exists(f'{V.RES}/probs/{tag}.pkl'):
            return
        for (s_, p_, m), v in pv.load_probs(tag).items():
            if only is not None and not m.startswith(only):
                continue
            m2 = m if rename is None else m.replace(rename[0], rename[1])
            P6.setdefault((s_, p_, m2), v)
    put('R_15_5'); put('M0_15_5'); put('M1_15_5')
    put('R_15_11', only='R_')                                       # train-only members (identical final models) if R_15_5 is absent
    put('R_15_11', rename=('RB_', 'RB11_'), only='RB_')             # March rows at WM 11 (14 days ~ the kernel's 31 days at WM 5)
    put('M1_15_11', rename=('MB_', 'MB11_'), only='MB_')
    put('R_15_2', rename=('RB_', 'RB2_'), only='RB_')
    mem = sorted({k[2] for k in P6}); print('members:', mem)
    _, EV = V.march_cached(CUT, 5.0)
    ARMS = {}

    def add(name, **w):
        if all(m in mem for m in w):
            ARMS[name] = w
    add('R:np+id', R_np=0.5, R_id=0.5)                                        # A    production-like (raw train only, no prior)
    add('R:npid+full_tr', R_np=0.25, R_id=0.25, R_full_tr=0.5)
    add('R:npid+full_s4', R_np=0.25, R_id=0.25, R_full_s4=0.5)                # A_s4 + train-only full member, blended prior
    add('RB:np+id', RB_np=0.5, RB_id=0.5)
    add('RB:npid+full', RB_np=0.25, RB_id=0.25, RB_full=0.5)                  # B    the kernel recipe (March rows x5)
    add('R:npid+RB:full', R_np=0.25, R_id=0.25, RB_full=0.5)
    add('RB11:np+id', RB11_np=0.5, RB11_id=0.5)
    add('RB11:npid+full', RB11_np=0.25, RB11_id=0.25, RB11_full=0.5)            # 14 March days at WM 11 ~ the kernel's 31 days at WM 5
    add('MB11:np+id', MB11_np=0.5, MB11_id=0.5)
    add('MB11:npid+full', MB11_np=0.25, MB11_id=0.25, MB11_full=0.5)
    add('RB2:npid+full', RB2_np=0.25, RB2_id=0.25, RB2_full=0.5)
    add('M:np+id', M_np=0.5, M_id=0.5)                                        # A_mask (masked train rows only)
    add('M:npid+full_s4', M_np=0.25, M_id=0.25, M_full_s4=0.5)
    add('MB:np+id', MB_np=0.5, MB_id=0.5)                                     # B_mask (masked train rows + March rows)
    add('MB:npid+full', MB_np=0.25, MB_id=0.25, MB_full=0.5)
    R = pd.concat([pv.score_members(EV, P6, w, nm) for nm, w in ARMS.items()], ignore_index=True)
    R.to_pickle(f'{V.RES}/pm_scores.pkl')
    PAIRS = [('R:np+id', 'R:npid+full_s4', 'A_s4 - A: train-only full member with blended prior'),
             ('R:npid+full_s4', 'RB:npid+full', 'B - A_s4: March rows (raw-train regime) = kernel vs fs4+full_s4'),
             ('R:np+id', 'RB:npid+full', 'B - A: kernel vs production-like'),
             ('R:np+id', 'RB:np+id', 'npid: March rows, raw-train regime'),
             ('R:npid+full_s4', 'R:npid+RB:full', 'only the full member March-trained'),
             ('R:npid+full_s4', 'M:npid+full_s4', 'masking adaptation of the train-only package (A_mask - A_s4)'),
             ('R:np+id', 'M:np+id', 'masking adaptation, npid only'),
             ('M:npid+full_s4', 'MB:npid+full', 'B_mask - A_mask: March rows in the matched (masked) regime'),
             ('M:np+id', 'MB:np+id', 'npid: March rows, masked regime'),
             ('RB:npid+full', 'MB:npid+full', 'B_mask - B'),
             ('RB:npid+full', 'RB11:npid+full', 'March weight x2.2 (= kernel total March weight) vs lab WM 5'),
             ('R:npid+full_s4', 'RB11:npid+full', 'WM 11 package vs A_s4'),
             ('RB:npid+full', 'RB2:npid+full', 'WM 2 vs WM 5'),
             ('R:npid+full_s4', 'RB11:npid+full', 'B11 - A_s4 (raw-train regime, kernel-equivalent March weight)'),
             ('R:np+id', 'RB11:np+id', 'npid: March rows WM 11, raw-train regime'),
             ('M:npid+full_s4', 'MB11:npid+full', 'B11_mask - A_mask: March rows WM 11 in the matched (masked) regime'),
             ('M:np+id', 'MB11:np+id', 'npid: March rows WM 11, masked regime'),
             ('RB11:npid+full', 'MB11:npid+full', 'B11_mask - B11'),
             ('R:np+id', 'MB11:npid+full', 'B11_mask - A')]
    rows = []
    for sname in ('offlike_m', 'runstart_m'):
        for a, b, note in PAIRS:
            if a in ARMS and b in ARMS:
                c = V.pair(R, a, b, sname, min_d=CUT); c['note'] = note; rows.append(c)
    T = pd.DataFrame(rows); T.to_csv(f'{V.RES}/pm_table.csv', index=False)
    print(f'\n== pseudo-March ongoing, days >= {CUT} ==')
    for sname in ('offlike_m', 'runstart_m'):
        g = R[(R.set == sname) & (R.d >= CUT)]
        print(sname, '  '.join(f'{a} {lab.aggregate(g[g.arm == a]):.4f}' for a in ARMS), f'(n={int((g.arm == "R:np+id").sum())})')
        for rg in (0.5, 0.7, 0.8):
            for a, b in (('R:npid+full_s4', 'RB11:npid+full'), ('M:npid+full_s4', 'MB11:npid+full')):
                if a in ARMS and b in ARMS:
                    print(f'  regime {rg}', V.fmt(V.pair(R, a, b, sname, min_d=CUT, sel=lambda g, rg=rg: g.regime == rg)))
    for c in rows:
        print(V.fmt(c), '|', c['note'])

elif what == 'harm':
    fr = []
    for v in ('base', 'raw', 'masked'):
        f = f'{V.RES}/harm_scores_H69_{v}_5.pkl'
        if os.path.exists(f):
            fr.append(pd.read_pickle(f))
    R = pd.concat(fr, ignore_index=True)
    arms = list(dict.fromkeys(R.arm)); print('arms:', arms)
    PAIRS = [('base:np+id', 'base:npid+full_tr', 'in-distribution: full member, right (train) prior'),
             ('base:np+id', 'base:npid+full_sc', 'in-distribution: full member, April-style blend with a masked 31-day month'),
             ('base:npid+full_tr', 'base:npid+full_sc', 'prior blend with the masked same-scenario month vs train prior'),
             ('base:np+id', 'raw:np+id', 'source month RAW x5 (npid)'),
             ('base:np+id', 'masked:np+id', 'source month MASKED x5 (npid) = kernel-style rows'),
             ('raw:np+id', 'masked:np+id', 'masked vs raw source rows (npid)'),
             ('base:npid+full_sc', 'raw:npid+full_sc', 'source month RAW x5 (package)'),
             ('base:npid+full_sc', 'masked:npid+full_sc', 'source month MASKED x5 (package) = kernel recipe in-train'),
             ('raw:npid+full_sc', 'masked:npid+full_sc', 'masked vs raw source rows (package)'),
             ('base:np+id', 'masked:npid+full_sc', 'kernel recipe vs production-like, in-train'),
             ('base:full_tr', 'base:full_sc', 'full member alone: blend vs train prior')]
    rows = []
    for sname in ('raw0', 'mask10'):
        for a, b, note in PAIRS:
            if a in arms and b in arms:
                c = V.pair(R, a, b, sname); c['note'] = note; rows.append(c)
    T = pd.DataFrame(rows); T.to_csv(f'{V.RES}/harm_table.csv', index=False)
    for sname in ('raw0', 'mask10'):
        g = R[R.set == sname]
        print(sname, '  '.join(f'{a} {lab.aggregate(g[g.arm == a]):.4f}' for a in arms), f'(n={int((g.arm == arms[0]).sum())})')
    for c in rows:
        print(V.fmt(c), '|', c['note'])
