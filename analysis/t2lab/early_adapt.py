"""early track (light): ADAPTIVE box.  The 90-minute blocks hidden after official windows often blank the +-6 box of the
previous days exactly at the recurrent onset time.  Variant: per evidence day and link use the +-6 box if it holds at least `emin`
visible cells, otherwise +-12, otherwise +-24 (each day is still compared with the train prior over its own visible cells).
Evaluated on the faithful replay (18 hidden slots) and E1."""
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
BOXES = ((-6, 6), (-12, 12), (-24, 24))


def stats_adaptive(b, items, emin, eps=0.02):
    n = len(items); L = b['L']
    S = {sk: {g: {q: np.zeros((n, L), np.float32) for q in ('nd', 'F', 'LL0', 'R')} for g in ('same', 'other', 'sd')} for sk in ('s6', 't')}
    for k, (i, days, evd) in enumerate(items):
        d, t = int(b['d'][i]), int(b['t'][i]); days = np.asarray(days, int)
        if not len(days):
            continue
        for sk, s in (('s6', min(t + 6, 287)), ('t', t)):
            O = e = X = None
            for box in BOXES:
                o2, e2, x2 = evd.box(days, s, *box)
                if O is None:
                    O, e, X = o2.copy(), e2.copy(), x2.copy()
                else:
                    m = e < emin                                   # not enough visible cells in the narrower box
                    O[m], e[m], X[m] = o2[m], e2[m], x2[m]
            ok = e >= emin
            f = np.where(ok, O / np.maximum(e, 1e-9), 0.0); r = np.where(ok, X / np.maximum(e, 1e-9), 0.0); rc = np.clip(r, eps, 1 - eps)
            T = S[sk]['same']                                       # classes pooled (cw = 1): everything goes to one group
            T['nd'][k] = ok.sum(0); T['F'][k] = f.sum(0); T['R'][k] = r.sum(0)
            T['LL0'][k] = np.where(ok, f * np.log(rc) + (1 - f) * np.log(1 - rc), 0.0).sum(0)
    return S


def run(items_fn, emin, pis=(0.9, 0.95)):
    out = []
    for b in B:
        items, meta = items_fn(b)
        widx = np.array([it[0] for it in items]); base3 = b['X'][widx][..., el.JP]
        cond = np.array([lab.ONSET if meta is None else f'on@{d0}' for d0 in (meta.d0 if meta is not None else widx)])
        variants = {'base': None}
        S6 = el.item_stats(b, items, (-6, 6))
        for pi in pis:
            variants[f'mix{pi}'] = el.design_priors(base3, S6, (-6, 6), form='bma', pi=pi, mu=0.05, nu=1.0, cw=1.0)
            if emin:
                variants[f'adapt{pi}'] = el.design_priors(base3, stats_adaptive(b, items, emin), (-6, 6), form='bma', pi=pi, mu=0.05, nu=1.0, cw=1.0)
        for name, p3 in variants.items():
            iou, _ = el.iou_items(b, widx, el.predict_items(cat, b, widx, p3))
            df = el.frame(b, widx, iou, extra=dict(design=name)); df['cond'] = cond
            if meta is not None:
                df['n'] = meta.n.to_numpy(); df['d0'] = meta.d0.to_numpy()
            out.append(df)
    return pd.concat(out, ignore_index=True)


for emin in (1, 3):
    print(f'\n################ adaptive box, a day needs >= {emin} visible cells in a box')
    R = run(lambda b: en.items_replay(b, hide=18), emin)
    ds = [x for x in dict.fromkeys(R.design) if x != 'base']
    sat = R[R.d0.isin([7, 14, 21, 28])]
    print('-- faithful Saturday replay'); print(en.table(sat, ref='base', ref2='mix0.9', designs=['base'] + ds).round(4).to_string(index=False))
    for n in (1, 2, 3, 4):
        print(f'-- faithful replay all starts n = {n}'); print(en.table(R[R.n == n], ref='base', ref2='mix0.9', designs=['base'] + ds).round(4).to_string(index=False))
    for n in (1, 2):
        R1 = run(lambda b: en.items_recent(b, n), emin)
        print(f'-- E1 n = {n}'); print(en.table(R1, ref='base', ref2='mix0.9', designs=['base'] + ds).round(4).to_string(index=False))
