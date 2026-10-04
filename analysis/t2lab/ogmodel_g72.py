"""TRACK ogmodel S6 (window regime): the pick-spacing flag g72 of an official(-like) ongoing window.

The official selector is one chronological greedy scan with a 360-min (72-slot) spacing to every earlier pick of BOTH
conditions.  An ongoing pick at T whose previous pick (any condition, same panel) sits exactly 72 slots earlier was blocked
until T: it is the first candidate at the end of the block, i.e. (almost always) a MID-RUN window (T-1 was a candidate too).
A pick with a gap > 72 slots was not blocked at T-1, so T-1 was not a candidate: a RUN-START.
The flag needs only the forecast origins of EARLIER windows of the same split (window_index.csv, timestamps < T).

This script (light, no training):
  1. replays the faithful greedy from 00:00 of every train day (the lab's offlike construction) and stores, per offlike
     ongoing window, the g72 flag of every replay that picks it  -> results/ogmodel/g72_offlike.pkl
  2. cross-tabulates g72 vs the lab's run-start flag (offlike), and g72 from the official window index vs run-start (official 40)
  3. counts g72 windows in the released validation / private window indexes
  4. the same replay flags for the pseudo-March offlike_m windows -> results/ogmodel/g72_pmarch.pkl"""
import numpy as np
import pandas as pd
import lab
import ogmodel_lib as og

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
SP = lab.SPACING


def replay_flags(g_on, g_og, D):
    """dict (d, t) -> list of g72 flags (one per replay that picks the ongoing window)."""
    out = {}
    for d0 in range(D):
        a, b = lab.greedy(g_on, g_og, lab.FAITHFUL_SCAN, start_day=d0)
        allp = np.sort(np.array([d * 288 + t for d, t in a + b], np.int64))
        for d, t in b:
            g = d * 288 + t; j = np.searchsorted(allp, g)
            gap = g - allp[j - 1] if j > 0 else 10 ** 9
            out.setdefault((d, t), []).append(gap == SP)
    return out


def index_gaps(panel, split):
    """g72 flag per official window of a split from the window index (previous window of the same panel, any condition)."""
    wi = lab.window_index(panel, split)
    ts = pd.to_datetime(wi.forecast_origin, utc=True)
    g = (ts.astype('int64') // 10 ** 9 // 300).to_numpy()
    o = np.argsort(g, kind='stable'); gap = np.full(len(g), 10 ** 9, np.int64); gap[o[1:]] = np.diff(g[o])
    return wi.assign(gap=gap, g72=gap == SP, slot=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), day=ts.dt.strftime('%Y-%m-%d'))


if __name__ == '__main__':
    rows = []
    for panel in lab.PANELS:
        P = lab.load_panel(panel)
        on, ogc = lab.candidates(P)
        fl = replay_flags(lab._glist(on), lab._glist(ogc), len(P['days']))
        win = lab.window_set(panel, 'offlike'); win = win[win.cond == lab.ONGOING].reset_index(drop=True)
        assert set(zip(win.d.tolist(), win.t.tolist())) == set(fl), panel
        for d, t, rs in zip(win.d, win.t, win.runstart):
            f = fl[(int(d), int(t))]
            rows.append(dict(panel=panel, d=int(d), t=int(t), runstart=bool(rs), n_replays=len(f), g72_any=bool(any(f)), g72_all=bool(all(f)), g72_share=float(np.mean(f))))
        lab.drop_panel(panel)
    G = pd.DataFrame(rows); G.to_pickle(f'{og.RES}/g72_offlike.pkl')
    print('== offlike (train replays): g72 vs run-start ==')
    print(pd.crosstab(G.runstart, [G.g72_any, G.g72_all], margins=True).to_string())
    print('windows whose flag differs between replays:', int((G.g72_any != G.g72_all).sum()), 'of', len(G))
    print(G.groupby('panel').agg(n=('g72_any', 'size'), g72=('g72_any', 'sum'), nonrunstart=('runstart', lambda x: int((~x).sum())),
                                 g72_and_nrs=('g72_any', lambda x: int((x & ~G.loc[x.index, 'runstart']).sum()))).to_string())
    # official train windows
    rows = []
    for panel in lab.PANELS:
        ix = index_gaps(panel, 'train'); ix = ix[ix.condition == lab.ONGOING]
        P = lab.load_panel(panel); pos = {d: i for i, d in enumerate(P['days'])}
        win = lab.window_set(panel, 'official'); win = win[win.cond == lab.ONGOING]
        key = {(int(d), int(t)): bool(r) for d, t, r in zip(win.d, win.t, win.runstart)}
        for r in ix.itertuples():
            rows.append(dict(panel=panel, d=pos[r.day], t=int(r.slot), g72=bool(r.g72), gap=int(r.gap), runstart=key[(pos[r.day], int(r.slot))]))
        lab.drop_panel(panel)
    O = pd.DataFrame(rows); O.to_pickle(f'{og.RES}/g72_official.pkl')
    print('\n== official train ongoing windows: g72 (window index) vs lab run-start flag =='); print(pd.crosstab(O.runstart, O.g72, margins=True).to_string())
    print(O[O.g72 == O.runstart].to_string())
    for split in ('validation', 'private'):
        t = pd.concat([index_gaps(p, split) for p in lab.PANELS]); t = t[t.condition == lab.ONGOING]
        print(f'\n{split}: ongoing windows {len(t)}, g72 {int(t.g72.sum())} | per panel', t.groupby('panel').g72.sum().to_dict())
    # pseudo-March offlike_m
    rows = []
    for panel in lab.PANELS:
        P, Pv = og.val_panel(panel)
        on, ogc, stable, pers, blank = og.select(Pv)
        fl = replay_flags(lab._glist(on), lab._glist(ogc), len(Pv['days']))
        for (d, t), f in sorted(fl.items()):
            rows.append(dict(panel=panel, d=d, t=t, runstart=bool(ogc[d, t] & ~ogc[d, t - 1]), n_replays=len(f), g72_any=bool(any(f)), g72_all=bool(all(f))))
        lab.drop_panel(panel)
    M = pd.DataFrame(rows); M.to_pickle(f'{og.RES}/g72_pmarch.pkl')
    print('\n== pseudo-March offlike_m: g72 vs run-start =='); print(pd.crosstab(M.runstart, [M.g72_any, M.g72_all], margins=True).to_string())
