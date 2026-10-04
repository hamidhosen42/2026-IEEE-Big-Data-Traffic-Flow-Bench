"""early track: evaluation engine (NEW file).  Batched CatBoost prediction of many prior designs on the bundles of early_lib.

Evidence regimes:
  recent(n)      E1: the n scenario days immediately before the window's day (windows on day d < n use what exists); n = 99 -> all
  replay(starts) E2: 'month-start replay' - the official selection greedy is restarted at day d0 (the emulated first day of the
                 month); its 5 onset picks are the evaluation windows, the evidence of a pick on day d is days d0..d-1 with the
                 6-slot horizons of ALL picks of that replay (onset + ongoing) hidden, exactly like the released masked layer
                 hides the horizons of the official windows.  Saturday starts reproduce the public calendar (month starts Sat).
"""
import numpy as np
import pandas as pd
import lab, early_lib as el

F = lab.F


def items_recent(b, n, evd=None):
    evd = evd or el.Evidence(b)
    return [(i, np.arange(max(int(b['d'][i]) - n, 0), int(b['d'][i])), evd) for i in range(b['W'])], None


def items_replay(b, starts=None, max_n=None, hide=F, make=None):
    """returns items and a meta frame (d0, n = d - d0).  Windows of each replay = its onset picks.
    hide: number of slots after each pick's origin that are hidden for all links (F = 6: the forecast horizon; the released
    masked layer actually hides T+1..T+18 around every official window -> use hide=18 for the faithful replay)."""
    key = b.get('key') or {(int(a), int(c)): i for i, (a, c) in enumerate(zip(b['d'], b['t']))}
    items, meta = [], []
    for d0 in (range(b['D']) if starts is None else starts):
        if d0 not in b['picks']:
            continue
        on, og = b['picks'][d0]
        blank = np.zeros((b['D'], 288), bool)
        for (d, t) in list(on) + list(og):
            if d < b['D']:
                blank[d, t + 1:t + 1 + hide] = True
        evd = (make or el.Evidence)(b, blank)                         # make: optional evidence-layer factory (b, blank) -> Evidence
        for (d, t) in on:
            if (d, t) not in key or d >= b['D']:
                continue
            if max_n is not None and d - d0 > max_n:
                continue
            items.append((key[(d, t)], np.arange(d0, d), evd)); meta.append(dict(d0=d0, n=d - d0))
    return items, pd.DataFrame(meta)


def run(cat, bundles, designs, items_fn, tag, chunk_rows=400_000, with_base=True):
    """designs: dict name -> dict(box=(lo, hi), sp=0, sameday=False, **design_priors kwargs).
    Returns a long frame: design, panel, family, d, t, cond, iou, npred (+ d0, n for the replay)."""
    out = []
    for b in bundles:
        items, meta = items_fn(b)
        if not items:
            continue
        widx = np.array([it[0] for it in items]); L = b['L']
        Xb = b['X'][widx]                                              # [n_items, L, C]
        base3 = Xb[..., el.JP]
        cond = np.array([lab.ONSET if meta is None else f'on@{d0}' for d0 in (meta.d0 if meta is not None else widx)])
        extra = {} if meta is None else dict(d0=meta.d0.to_numpy(), n=meta.n.to_numpy())

        def emit(name, p):
            iou, npred = el.iou_items(b, widx, p)
            df = el.frame(b, widx, iou, extra=dict(design=name, npred=npred, **extra)); df['cond'] = cond
            df['set'] = tag
            out.append(df)
        if with_base:
            emit('base', cat.predict_proba(Xb.reshape(-1, Xb.shape[-1])[:, el.KEEPI], thread_count=lab.THREADS)[:, 1].reshape(len(widx), L))
        cache = {}; names = list(designs); per = max(1, chunk_rows // max(len(widx) * L, 1))
        for c0 in range(0, len(names), per):
            grp = names[c0:c0 + per]
            Xbig = np.repeat(Xb[None], len(grp), 0)
            for g, name in enumerate(grp):
                dz = dict(designs[name]); box = dz.pop('box'); sp = dz.pop('sp', 0); sd = dz.pop('sameday', False)
                eps = dz.pop('eps', 0.02)
                ck = (box, sp, sd, eps)
                if ck not in cache:
                    cache[ck] = el.item_stats(b, items, box, sp, sd, eps)
                Xbig[g][..., el.JP] = el.design_priors(base3, cache[ck], box, sp=sp, sameday=sd, **dz)
            p = cat.predict_proba(Xbig.reshape(-1, Xb.shape[-1])[:, el.KEEPI], thread_count=lab.THREADS)[:, 1].reshape(len(grp), len(widx), L)
            for g, name in enumerate(grp):
                emit(name, p[g])
    return pd.concat(out, ignore_index=True)


def table(R, by=None, ref='base', ref2=None, designs=None, boot=True):
    """one row per design: n windows, aggregated IoU, paired bootstrap vs ref (and vs ref2), per-panel sign count."""
    rows = []
    A0 = R[R.design == ref]; A2 = R[R.design == ref2] if ref2 else None
    for name in designs or list(dict.fromkeys(R.design)):
        B0 = R[R.design == name]
        r = dict(design=name, n=len(B0), iou=lab.aggregate(B0))
        if name != ref:
            if boot:
                bt = el.boot(A0, B0)
                r.update(d_base=bt['diff'], lo=bt['lo'], hi=bt['hi'], pan=f"+{bt['pos']}/-{bt['neg']}")
            else:
                r.update(d_base=lab.aggregate(B0) - lab.aggregate(A0))
        if ref2 and name not in (ref, ref2) and boot:
            bt = el.boot(A2, B0)
            r.update(d_ref2=bt['diff'], lo2=bt['lo'], hi2=bt['hi'], pan2=f"+{bt['pos']}/-{bt['neg']}")
        rows.append(r)
    return pd.DataFrame(rows)


def panel_table(R, designs):
    return pd.DataFrame({name: R[R.design == name].groupby('panel').iou.mean() for name in designs}).assign(
        n=R[R.design == designs[0]].groupby('panel').size())
