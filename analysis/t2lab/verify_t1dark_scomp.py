"""verify_t1dark_scomp.py - composite S_state delta of the blackout-only override on the verifier runs.
Per panel-month and regime r: MSE = w_r * MSE_blackout + (1 - w_r) * MSE_ordinary(r), w_r = rate_r * eligible cells in the
official blackout rows / (hidden eligible cells outside them + that), computed from the released month (own code).
Arms on blackout cells only (ordinary cells keep base): dark, mix = 0.5 base + 0.5 dark (= W_DARK 0.5). Event bootstrap.
Usage: verify_t1dark_scomp.py [tag=] [B=400]"""
import glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'repo', 'src'))
from task1.baseline_task1_historical_mean import regime_of_dates  # noqa: E402

RESV = os.path.join(HERE, 'results', 'verify_t1dark')
kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
TAG = kw.get('tag', '')
B = int(kw.get('B', 400))
RATE = {'R1': .2, 'R2': .3, 'R3': .5}


def weights(panel, split):
    z = np.load(f'{ROOT}/data/t2local/t2data_{panel}.npz', allow_pickle=True)
    d = z[f'{split}_days'].astype(str); o = np.argsort(d); days = d[o]
    Vm = z[f'{split}_Vm'][o]; Fm = z[f'{split}_Fm'][o]; Em = z[f'{split}_Em'][o]
    obs = Em & np.isfinite(Vm) & np.isfinite(Fm)
    rr = np.zeros(Em.shape[:2], bool)
    di = {x: i for i, x in enumerate(days)}
    for s in z[f'{split}_wi_forecast_origin'].astype(str):
        t = int(s[11:13]) * 12 + int(s[14:16]) // 5
        rr[di[s[:10]], t + 1:t + 19] = True
    reg = np.asarray(regime_of_dates(panel, pd.Series(days))).astype(str)
    w = {}
    for r in RATE:
        m = reg == r
        if not m.any():
            continue
        inside = float((Em & rr[:, :, None])[m].sum()); outside = float((Em & ~obs & ~rr[:, :, None])[m].sum())
        w[r] = RATE[r] * inside / (outside + RATE[r] * inside)
    return w


def s_of(rv, rq):
    return 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)


rows = []
for f in sorted(glob.glob(f'{RESV}/*_f[01]{TAG}.npz')):
    base = os.path.basename(f)[:-4]
    if 'smoke' in base or (TAG == '' and base[-2:] not in ('f0', 'f1')):
        continue
    p, fold = base[:len(base) - len(TAG)].rsplit('_f', 1)
    z = np.load(f, allow_pickle=True)
    for sp in ('validation', 'private'):
        for kind in ('blk', 'ord'):
            pre = f'{sp}|std|{kind}'
            vt, qt, ln = z[f'{pre}|vt'], z[f'{pre}|qt'], z[f'{pre}|ln']
            vb, vd = z[f'{pre}|base|v'].astype(float), z[f'{pre}|dark|v'].astype(float)
            qb, qd = z[f'{pre}|base|q'].astype(float), z[f'{pre}|dark|q'].astype(float)
            rows.append(pd.DataFrame(dict(panel=p, split=sp, fold=int(fold), kind=kind, reg=z[f'{pre}|reg'],
                                          unit=(fold + ':' + pd.Series(z[f'{pre}|ev'] if kind == 'blk' else z[f'{pre}|d']).astype(str)).to_numpy(),
                                          vb=(vb - vt) ** 2, vd=(vd - vt) ** 2, vm=(0.5 * (vb + vd) - vt) ** 2,
                                          qb=((qb - qt) / ln) ** 2, qd=((qd - qt) / ln) ** 2, qm=((0.5 * (qb + qd) - qt) / ln) ** 2)))
A = pd.concat(rows, ignore_index=True)


def s_comp(blk, o, w, arm, wb=None, wo=None):
    wb = np.ones(len(blk)) if wb is None else wb
    wo = np.ones(len(o)) if wo is None else wo
    mbv = (blk[f'v{arm}'] * wb).sum() / wb.sum(); mbq = (blk[f'q{arm}'] * wb).sum() / wb.sum()
    S = []
    for r, wr in w.items():
        m = (o.reg == r).to_numpy()
        if not m.any():
            continue
        mov = (o.vb * wo * m).sum() / (wo * m).sum(); moq = (o.qb * wo * m).sum() / (wo * m).sum()
        S.append(s_of(np.sqrt(wr * mbv + (1 - wr) * mov), np.sqrt(wr * mbq + (1 - wr) * moq)))
    return float(np.mean(S))


rng = np.random.default_rng(0)
print('composite S_state delta, blackout-only override (ordinary cells keep base); cell-level bootstrap by event / day unit')
out = []
for (p, sp), g in A.groupby(['panel', 'split']):
    w = weights(p, sp)
    blk = g[g.kind == 'blk'].reset_index(drop=True); o = g[g.kind == 'ord'].reset_index(drop=True)
    ub = blk.unit.to_numpy(); uo = o.unit.to_numpy()
    lb, ib = np.unique(ub, return_inverse=True); lo_, io = np.unique(uo, return_inverse=True)
    res = {}
    for arm in ('d', 'm'):
        d0 = s_comp(blk, o, w, arm) - s_comp(blk, o, w, 'b')
        bs = []
        for _ in range(B):
            cb = np.bincount(rng.integers(0, len(lb), len(lb)), minlength=len(lb))[ib]
            co = np.bincount(rng.integers(0, len(lo_), len(lo_)), minlength=len(lo_))[io]
            bs.append(s_comp(blk, o, w, arm, cb, co) - s_comp(blk, o, w, 'b', cb, co))
        res[arm] = (d0, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    out.append(dict(panel=p, split=sp, wblk=np.mean(list(w.values())), dS_dark=res['d'][0], lo_dark=res['d'][1], hi_dark=res['d'][2],
                    dS_mix=res['m'][0], lo_mix=res['m'][1], hi_mix=res['m'][2]))
T = pd.DataFrame(out)
print(T.round(5).to_string(index=False))
print('mean over panel-months: dS_dark %+.5f | dS_mix (W_DARK 0.5) %+.5f' % (T.dS_dark.mean(), T.dS_mix.mean()))
