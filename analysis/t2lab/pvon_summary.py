"""pvon: tables from the per-window result frames in results/pvon (no training; seconds).
Usage: pvon_summary.py step3 | shift | step2 [file ...]"""
import glob, sys
import numpy as np
import pandas as pd
import lab
import pvon_lib as L

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
what = sys.argv[1]
files = sys.argv[2:]


def table(G, arms):
    return pd.DataFrame([dict(arm=a, iou=lab.aggregate(G[G.arm == a]), plain=G[G.arm == a].iou.mean(), n=int((G.arm == a).sum())) for a in arms if (G.arm == a).any()])


def cmp(G, pairs):
    for a, b in pairs:
        if (G.arm == a).any() and (G.arm == b).any():
            c = L.compare(G, a, b)
            m = G[G.arm == a].merge(G[G.arm == b], on=['panel', 'd', 't', 'cond'], suffixes=('_a', '_b'))
            ch = int(((m.iou_b - m.iou_a).abs() > 1e-9).sum())
            print(L.fmt(c) + f' changed {ch}')


if what == 'step3':
    files = files or sorted(glob.glob(f'{L.OUT}/step3_f*_s*.pkl'))
    Rr = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    print('files', files, 'folds', sorted(Rr.fold.unique()))
    for s in ('R', 'M'):
        G = Rr[Rr.set == s]
        arms = list(dict.fromkeys(G.arm))
        print(f'\n===== step 3, target {s}: n = {int((G.arm == "i").sum())} windows, folds {sorted(G.fold.unique())}')
        print(table(G, [a for a in arms if '@' not in a]).round(4).to_string(index=False))
        cmp(G, [('i', 'ii_w1'), ('i', 'ii_bag'), ('i', 'iii_bag'), ('i', 'iv_bag'), ('ii_bag', 'iii_bag'), ('ii_bag', 'iv_bag'),
                ('i', 'ii_w10'), ('i', 'ii_w20'), ('i', 'ii_w40'), ('i', 'iv_w10'), ('i', 'iv_w20'), ('i', 'iv_w40'),
                ('ii_w10', 'iv_w10'), ('ii_w20', 'iv_w20'), ('ii_w40', 'iv_w40'),
                ('i', 'i_mS'), ('i', 'i_mix'), ('i_mS', 'iv_mS_bag'), ('i_mix', 'iv_mix_bag'), ('i', 'iv_mix_bag'), ('ii_bag', 'iv_mix_bag'),
                ('iv_bag', 'iv_mix_bag')])
        if s == 'R':
            print('-- test-time masking of the raw target history')
            cmp(G, [('i', 'i@ttm'), ('iv_bag', 'iv_bag@ttm'), ('iv_bag', 'iv_bag@ttmr'), ('ii_bag', 'iv_bag@ttm'), ('i', 'iv_bag@ttm'),
                    ('i_mS', 'i_mS@ttm'), ('i', 'i_mS@ttm'), ('i', 'iv_mS_bag@ttm'), ('i_mS@ttm', 'iv_mS_bag@ttm'), ('i_mix', 'i_mix@ttm')])
        print('-- per fold: iv_bag - i, iv_bag - ii_bag')
        for k in sorted(G.fold.unique()):
            cmp(G[G.fold == k], [('i', 'iv_bag'), ('ii_bag', 'iv_bag')])
        if s == 'M' and 'hidden_est' in G:
            print('share of M windows with an established queue in the RAW history:', round(float(G[G.arm == 'i'].hidden_est.mean()), 4))
elif what == 'shift':
    files = files or sorted(glob.glob(f'{L.OUT}/shift_f*_s*_dA-B.pkl'))
    Rr = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    print('files', files, 'folds', sorted(Rr.fold.unique()), 'sets', sorted(Rr.set.unique()))
    Rr['tgt'] = Rr.set.str[0]; Rr['pat'] = Rr.set.str[2:]
    idx = {p: i for i, p in enumerate(lab.PANELS)}
    Rr['shifted'] = [((idx[p] % 2 == 0) == (q == 'A')) if q in ('A', 'B') else (q != '0') for p, q in zip(Rr.panel, Rr.pat)]
    pairs = [('a_tr', 'a_s4'), ('a_s4', 'aseed_s4'), ('a_s4', 'amS_s4'), ('amS_s4', 'm_bag'), ('a_s4', 'm_bag'), ('c_bag', 'm_bag'), ('a_s4', 'b_bag'), ('a_s4', 'c_bag'), ('a_s4', 'd_bag'), ('b_bag', 'c_bag'), ('c_bag', 'd_bag'), ('amix_s4', 'd_bag'),
             ('a_s4', 'amix_s4'), ('a_s4', 'c_w5'), ('a_s4', 'c_w10'), ('a_s4', 'c_w20'), ('a_s4', 'c_w40'), ('a_s4', 'c_bagL'), ('c_bag', 'c_bagL'), ('c_bag', 'c_w10'), ('c_bag', 'c_w5'), ('a_s4', 'b_w10'), ('a_s4', 'b_w20'), ('a_s4', 'b_w40')]
    ttm = [('c_bag', 'c_bag@ttm'), ('c_bag', 'c_bag@ttmr'), ('a_s4', 'c_bag@ttm'), ('a_s4', 'a_s4@ttm'), ('d_bag', 'd_bag@ttm')]
    for tg in ('R', 'M'):
        G0 = Rr[Rr.tgt == tg].copy()
        for name, G in (('pooled A+B (every panel once shifted, once not)', G0.assign(d=G0.d + 1000 * (G0.pat == 'B'))),
                        ('shifted panels only', G0[G0.shifted].copy()), ('unshifted panels only', G0[~G0.shifted].copy())):
            G = G.drop(columns=['set']).assign(set=tg)
            arms = list(dict.fromkeys(G.arm))
            print(f'\n===== shift, target {tg}, {name}: n = {int((G.arm == "a_s4").sum())}')
            print(table(G, [a for a in arms if '@' not in a]).round(4).to_string(index=False))
            cmp(G, pairs + (ttm if tg == 'R' else []))
elif what == 'step2':
    files = files or sorted(glob.glob(f'{L.OUT}/step2_cut*.pkl'))
    for f in files:
        Rr = pd.read_pickle(f)
        arms = list(dict.fromkeys(Rr.arm))
        print(f'\n===== step 2 {f}: n = {int((Rr.arm == "A_s4").sum())}')
        print(table(Rr, [a for a in arms if '#' not in a]).round(4).to_string(index=False))
        fams = sorted({a[1:-4] for a in arms if a.startswith('B') and a.endswith('_bag') and '#' not in a})
        cmp(Rr, [('A', 'A_s4'), ('A_s4', 'B_w10'), ('A_s4', 'B_w20'), ('A_s4', 'B_w40'), ('A_s4', 'B_bag')])
        for f_ in fams:
            if f_:
                cmp(Rr, [('A_s4', f'A{f_}'), (f'A{f_}', f'B{f_}_bag'), ('A_s4', f'B{f_}_bag'), ('B_bag', f'B{f_}_bag')])
        Rr['reg'] = Rr.share.round(1)
        print('-- by mask regime of the evaluation day (visible share of eligible cells)')
        for reg in (0.5, 0.7, 0.8):
            G = Rr[Rr.reg == reg]
            print(f'regime {reg}: n = {int((G.arm == "A_s4").sum())}')
            cmp(G, [('A_s4', 'B_bag')] + [x for f_ in fams if f_ for x in [('A_s4', f'A{f_}'), (f'A{f_}', f'B{f_}_bag')]])
