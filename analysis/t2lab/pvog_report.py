"""pvog: tables from the saved member probabilities (light: scoring + paired bootstrap only).
  pvog_report.py pm   pseudo-March (exp 1, 2, 5): arms from results/pvog/probs/{R,M,X}_15_5.pkl (+ R_15_1 if present)
  pvog_report.py harm in-train raw-deployment test (exp 3, 4, 5): results/pvog/probs/H69_<variant>_<WM>.pkl"""
import os, sys
import numpy as np
import pandas as pd
import lab
import pvog_lib as pv

what = sys.argv[1]
CUT = 15


def have(tag):
    return os.path.exists(f'{pv.RES}/probs/{tag}.pkl')


def show(T, title):
    print(f'\n== {title} =='); print(T.round(4).to_string(index=False))


def pair(R, a, b, sname, min_d=None, extra=None):
    g = R[R.set == sname]
    if min_d is not None:
        g = g[g.d >= min_d]
    if extra is not None:
        g = g[extra(g)]
    A = g[g.arm == a]; B = g[g.arm == b]
    bt = lab.paired_bootstrap(A, B)
    pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
    return dict(set=sname, B=b, A=a, n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}')


if what == 'pm':
    tags = [t for t in ('R_15_5', 'M_15_5', 'X_15_5', 'R_15_1', 'R_15_2') if have(t)]
    P6 = {}
    for t in tags:
        z = pv.load_probs(t)
        if t in ('R_15_1', 'R_15_2'):                  # March-trained members with another weight: rename RB_* -> RB<wm>_*
            z = {(s, p, m.replace('RB_', f'RB{t[-1]}_')): v for (s, p, m), v in z.items() if m.startswith('RB_')}
        P6.update(z)
    _, EV = pv.march_data(CUT, 5.0, eval_from=0, verbose=False)
    mem = sorted({k[2] for k in P6}); print('members:', mem)
    ARMS = {}

    def add(name, **w):
        if all(m in mem for m in w):
            ARMS[name] = w
    add('R:np+id', R_np=0.5, R_id=0.5)                                             # (a) production-like, raw train only
    add('R:npid+full_tr', R_np=0.25, R_id=0.25, R_full_tr=0.5)
    add('R:npid+full_s4', R_np=0.25, R_id=0.25, R_full_s4=0.5)                     # (b) + train-only full model, blended prior
    add('R:np+id+full_s4/3', R_np=1 / 3, R_id=1 / 3, R_full_s4=1 / 3)
    add('R:full_s4', R_full_s4=1.0)
    add('RB:np+id', RB_np=0.5, RB_id=0.5)
    add('RB:npid+full', RB_np=0.25, RB_id=0.25, RB_full=0.5)                       # (c) the kernel recipe (March rows x5)
    add('RB:full', RB_full=1.0)
    add('R:npid+RB:full', R_np=0.25, R_id=0.25, RB_full=0.5)
    for pre in ('M', 'X'):
        add(f'{pre}:np+id', **{f'{pre}_np': 0.5, f'{pre}_id': 0.5})
        add(f'{pre}:npid+full_s4', **{f'{pre}_np': 0.25, f'{pre}_id': 0.25, f'{pre}_full_s4': 0.5})
        add(f'{pre}:npid+full_tr', **{f'{pre}_np': 0.25, f'{pre}_id': 0.25, f'{pre}_full_tr': 0.5})
        add(f'{pre}B:np+id', **{f'{pre}B_np': 0.5, f'{pre}B_id': 0.5})
        add(f'{pre}B:npid+full', **{f'{pre}B_np': 0.25, f'{pre}B_id': 0.25, f'{pre}B_full': 0.5})
    for wmx in ('1', '2'):
        add(f'RB{wmx}:np+id', **{f'RB{wmx}_np': 0.5, f'RB{wmx}_id': 0.5})
        add(f'RB{wmx}:npid+full', **{f'RB{wmx}_np': 0.25, f'RB{wmx}_id': 0.25, f'RB{wmx}_full': 0.5})
    R = pd.concat([pv.score_members(EV, P6, w, nm) for nm, w in ARMS.items()], ignore_index=True)
    R.to_pickle(f'{pv.RES}/report_pm.pkl')
    for sname in ('offlike_m', 'runstart_m'):
        show(pv.cmp_table(R, 'R:np+id', [sname], min_d=CUT), f'pseudo-March {sname}, days >= {CUT}: every arm vs R:np+id (production-like, raw-trained)')
    rows = []
    for sname in ('offlike_m', 'runstart_m'):
        for a, b in (('R:np+id', 'M:np+id'), ('M:np+id', 'MB:np+id'), ('R:np+id', 'RB:np+id'), ('M:npid+full_s4', 'MB:npid+full'), ('R:npid+full_s4', 'RB:npid+full'),
                     ('M:np+id', 'M:npid+full_s4'), ('R:np+id', 'R:npid+full_s4'), ('R:np+id', 'X:np+id'), ('X:np+id', 'XB:np+id'), ('X:npid+full_s4', 'XB:npid+full'),
                     ('X:np+id', 'X:npid+full_s4'), ('M:np+id', 'MB:npid+full'), ('X:np+id', 'XB:npid+full'), ('MB:npid+full', 'RB:npid+full'), ('M:npid+full_s4', 'RB:npid+full'),
                     ('RB:npid+full', 'RB1:npid+full'), ('R:np+id', 'RB1:npid+full'), ('R:np+id', 'RB1:np+id')):
            if a in ARMS and b in ARMS:
                rows.append(pair(R, a, b, sname, min_d=CUT))
    show(pd.DataFrame(rows), f'pseudo-March decomposition (B - A), days >= {CUT}')
    # dose-response by the day regime of the evaluation window (visible share of eligible cells)
    rows = []
    for sname in ('offlike_m', 'runstart_m'):
        for a, b in (('R:np+id', 'RB:np+id'), ('R:np+id', 'RB:npid+full'), ('R:np+id', 'R:npid+full_s4'), ('R:np+id', 'M:np+id'), ('M:np+id', 'MB:np+id'), ('M:npid+full_s4', 'MB:npid+full')):
            if a in ARMS and b in ARMS:
                for rg in (0.5, 0.7, 0.8):
                    r = pair(R, a, b, sname, min_d=CUT, extra=lambda g, rg=rg: g.regime == rg); r['regime'] = rg; rows.append(r)
    show(pd.DataFrame(rows), f'by day regime of the evaluation window (visible share), days >= {CUT}')
    # train-only full member on ALL pseudo-March days (it uses no March labels): robustness by day bin
    rows = []
    for sname in ('offlike_m', 'runstart_m'):
        for lo, hi in ((0, 30), (0, 4), (5, 14), (15, 30), (15, 22), (23, 30)):
            for b in ('R:npid+full_s4', 'R:np+id+full_s4/3', 'R:npid+full_tr'):
                r = pair(R, 'R:np+id', b, sname, extra=lambda g, lo=lo, hi=hi: (g.d >= lo) & (g.d <= hi)); r['days'] = f'{lo + 1}-{hi + 1}'; rows.append(r)
    show(pd.DataFrame(rows), 'exp 4a: train-only full member (blended prior) vs R:np+id by day bin (all pseudo-March days)')
    g = R[(R.set == 'offlike_m') & (R.d >= CUT)]
    t = g.pivot_table(index='panel', columns='arm', values='iou', aggfunc='mean')
    keep = [c for c in ('R:np+id', 'R:npid+full_s4', 'RB:np+id', 'RB:npid+full', 'M:np+id', 'M:npid+full_s4', 'MB:np+id', 'MB:npid+full') if c in t]
    show(t[keep].reset_index(), f'per panel, offlike_m days >= {CUT}')

if what == 'harm':
    SRC0 = int(sys.argv[2]) if len(sys.argv) > 2 else 69
    seeds = (10, 11, 12, 13)
    WM0 = sys.argv[3] if len(sys.argv) > 3 else '3'          # primary weight multiplier (kernel-matched March / train weight ratio)
    tags = [f'H{SRC0}_{v}_{wm}' for v in ('base', 'raw', 'masked', 'march') for wm in ('3', '5', '1') if have(f'H{SRC0}_{v}_{wm}')]
    P6 = {}
    for t in tags:
        z = pv.load_probs(t); wm = t.rsplit('_', 1)[1]
        if wm != WM0 and not t.startswith(f'H{SRC0}_base_'):
            z = {(s, p, m.replace('_', f'{wm}_', 1)): v for (s, p, m), v in z.items()}
        P6.update(z)
    print('probability files:', tags, '| primary WM', WM0)
    EVT = pv.target_blocks(pv.src_days(SRC0), seeds, with_alt=False)
    mem = sorted({k[2] for k in P6}); print('members:', mem)
    # test-time masking average: mean probability over the masked versions of the raw target windows, scored as set 'tta'
    EV_tta = []
    for ev in [e for e in EVT if e.rc.name == 'raw0']:
        import types
        rc = types.SimpleNamespace(**{**ev.rc.__dict__, 'name': 'tta'})
        EV_tta.append(pv.Ev(rc, None, None, None, ev.W, ev.L))
        for m in mem:
            ks = [(f'mask{s}', ev.rc.panel, m) for s in seeds]
            if all(k in P6 for k in ks):
                P6[('tta', ev.rc.panel, m)] = np.mean([P6[k] for k in ks], 0)
                P6[('tta+raw', ev.rc.panel, m)] = 0.5 * P6[('raw0', ev.rc.panel, m)] + 0.5 * P6[('tta', ev.rc.panel, m)]
    EV_mix = []
    for ev in EV_tta:
        import types
        rc = types.SimpleNamespace(**{**ev.rc.__dict__, 'name': 'tta+raw'})
        EV_mix.append(pv.Ev(rc, None, None, None, ev.W, ev.L))
    ARMS = {}

    def add(name, **w):
        if all(m in mem for m in w):
            ARMS[name] = w
    for v in ('base', 'raw', 'masked', 'march', 'masked5', 'raw5', 'march5', 'masked1', 'masked3'):
        add(f'{v}:np+id', **{f'{v}_np': 0.5, f'{v}_id': 0.5})
        add(f'{v}:np', **{f'{v}_np': 1.0}); add(f'{v}:id', **{f'{v}_id': 1.0})
        for f_ in ('tr', 'sc', 'wrong'):
            add(f'{v}:npid+full_{f_}', **{f'{v}_np': 0.25, f'{v}_id': 0.25, f'{v}_full_{f_}': 0.5})
            add(f'{v}:full_{f_}', **{f'{v}_full_{f_}': 1.0})
    R = pd.concat([pv.score_members(EVT + EV_tta + EV_mix, P6, w, nm, win_cols=()) for nm, w in ARMS.items()], ignore_index=True)
    R.to_pickle(f'{pv.RES}/report_harm_{SRC0}.pkl')
    Rm = R[R.set.isin([f'mask{s}' for s in seeds])]
    for sname in ('raw0', 'mask10', 'tta', 'tta+raw'):
        show(pv.cmp_table(R, 'base:np+id', [sname]), f'in-train target fold 0 ({sname}): every arm vs base:np+id')
    rows = []
    for sname in ('raw0', 'mask10', 'mask11'):
        for a, b in (('base:np+id', 'raw:np+id'), ('base:np+id', 'masked:np+id'), ('raw:np+id', 'masked:np+id'), ('base:np+id', 'march:np+id'),
                     ('base:np+id', 'base:npid+full_tr'), ('base:np+id', 'base:npid+full_sc'), ('base:np+id', 'base:npid+full_wrong'),
                     ('base:npid+full_sc', 'raw:npid+full_sc'), ('base:npid+full_sc', 'masked:npid+full_sc'), ('raw:npid+full_sc', 'masked:npid+full_sc'),
                     ('base:np+id', 'masked:npid+full_sc'), ('base:np+id', 'raw:npid+full_sc'),
                     ('base:np+id', 'march:npid+full_wrong'), ('base:np+id', 'march:npid+full_tr'),
                     ('base:np+id', 'masked5:np+id'), ('masked:np+id', 'masked5:np+id'), ('base:npid+full_sc', 'masked5:npid+full_sc'), ('base:np+id', 'masked5:npid+full_sc'),
                     ('base:np+id', 'masked1:np+id'), ('base:npid+full_sc', 'masked1:npid+full_sc'),
                     ('base:np', 'masked:np'), ('base:id', 'masked:id'), ('base:full_sc', 'masked:full_sc'), ('base:full_tr', 'masked:full_tr')):
            if a in ARMS and b in ARMS:
                rows.append(pair(R, a, b, sname))
    show(pd.DataFrame(rows), 'in-train decomposition (B - A)')
    # test-time masking: masked-trained model fed masked versions of the raw windows (average of 4 draws) vs fed the raw windows
    rows = []
    for v in ('base', 'raw', 'masked', 'masked5', 'march'):
        for blend in ('np+id', 'npid+full_sc', 'npid+full_tr'):
            a = f'{v}:{blend}'
            if a not in ARMS:
                continue
            for s2 in ('tta', 'tta+raw'):
                A = R[(R.set == 'raw0') & (R.arm == a)]; B = R[(R.set == s2) & (R.arm == a)].assign(set='raw0')
                bt = lab.paired_bootstrap(A, B); pa = A.groupby('panel').iou.mean(); pb = B.groupby('panel').iou.mean()
                rows.append(dict(arm=a, B=s2, A='raw0', n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                                 panels_pos=f'{int((pb - pa > 1e-9).sum())}/{len(pa)}'))
    show(pd.DataFrame(rows), 'exp 5: test-time masking (same model; masked x4 average or 50/50 with the raw prediction) vs raw features')
    g = R[R.set == 'raw0']
    t = g.pivot_table(index='panel', columns='arm', values='iou', aggfunc='mean')
    keep = [c for c in ('base:np+id', 'raw:np+id', 'masked:np+id', 'march:np+id', 'base:npid+full_tr', 'base:npid+full_sc', 'masked:npid+full_sc', 'base:npid+full_wrong', 'march:npid+full_wrong') if c in t]
    show(t[keep].reset_index(), 'per panel, raw fold-0 targets')
