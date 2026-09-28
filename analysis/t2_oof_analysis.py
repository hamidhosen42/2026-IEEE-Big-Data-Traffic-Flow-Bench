"""Task 2 post-hoc experiments on the OOF probability cache (kernel hosen42/tfb-t2oof).

All choices that are tuned (per-panel / per-TOD blend weights, thresholds) are selected with nested
CV: chosen on the other 3 folds, applied to the held fold. Reported with the competition aggregation
(window -> panel x condition -> panel -> family -> mean) on 'pseudo' (builder-faithful) and 'official' windows.
Baseline = the 0.85309 recipe: onset 0.7*main + 0.3*prior-free, ongoing prior-free, expected-IoU decode.
"""
import glob, sys, itertools
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else 'kaggle_jobs/out/t2oof'
W = []
for f in sorted(glob.glob(f'{D}/oof_*.npz')):
    name = f.split('/')[-1][4:-4]; panel = '_'.join(name.split('_')[:3]); tag = name.split('_')[3]; fold = int(name[-1])
    z = np.load(f, allow_pickle=True)
    for i in range(len(z['cond'])):
        W.append(dict(panel=panel, set=tag, fold=fold, cond=str(z['cond'][i]), dow=int(z['dow'][i]), tod=int(z['t'][i]),
                      P=z['P'][i].astype(np.float32), Pn=z['Pn'][i].astype(np.float32), Pc=z['Pc'][i].astype(np.float32),
                      truth=z['truth'][i], ev=z['ev'][i], qlast=z['qlast'][i],
                      rec=float(z['prior_now'][i][z['qlast'][i]].mean()) if z['qlast'][i].any() else np.nan))
print(len(W), 'windows loaded;', pd.Series([w['set'] for w in W]).value_counts().to_dict())


def decode(p):
    p = p.ravel(); o = np.argsort(-p); cs = np.cumsum(p[o]); m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + cs[-1] - cs)); out = np.zeros(len(p), bool); out[o[:j + 1]] = True
    return out


def iou(w, pred):
    t, e = w['truth'].ravel(), w['ev'].ravel(); pred = pred.ravel()
    u = (t & e | pred & e).sum()
    return 1.0 if u == 0 else ((t & pred) & e).sum() / u


def agg(rows):
    df = pd.DataFrame(rows)
    g = df.groupby(['panel', 'cond']).iou.mean().groupby('panel').mean().reset_index()
    g['fam'] = g.panel.str.rsplit('_', n=1).str[0]
    return g.groupby('fam').iou.mean().mean(), df.groupby('cond').iou.mean().to_dict()


def evaluate(fn, sets=('pseudo', 'official')):
    out = {}
    for s in sets:
        rows = [dict(panel=w['panel'], cond=w['cond'], iou=iou(w, fn(w))) for w in W if w['set'] == s]
        out[s] = agg(rows)
    return out


RES = []


def report(name, fn):
    r = evaluate(fn)
    RES.append(dict(exp=name, pseudo=r['pseudo'][0], pseudo_onset=r['pseudo'][1].get('queue_onset'),
                    pseudo_ongoing=r['pseudo'][1].get('queue_ongoing'), official=r['official'][0],
                    off_onset=r['official'][1].get('queue_onset'), off_ongoing=r['official'][1].get('queue_ongoing')))
    x = RES[-1]
    print(f"{name:42s} pseudo {x['pseudo']:.4f} (on {x['pseudo_onset']:.4f} / og {x['pseudo_ongoing']:.4f}) | "
          f"official {x['official']:.4f} (on {x['off_onset']:.4f} / og {x['off_ongoing']:.4f})", flush=True)


on = lambda w: w['cond'] == 'queue_onset'
base = lambda w: decode(0.7 * w['P'] + 0.3 * w['Pn'] if on(w) else w['Pn'])
report('EXP_000 baseline (on 70/30, og prior-free)', base)

print('\n--- A1 onset blend sweep (ongoing fixed = prior-free)')
for a in (1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3):
    report(f'A1 onset {a:.2f}/{1 - a:.2f}', lambda w, a=a: decode(a * w['P'] + (1 - a) * w['Pn'] if on(w) else w['Pn']))

# nested selection helper: choose a key-specific weight on the other folds, apply on the held fold
GRID = (1.0, 0.85, 0.7, 0.55, 0.4)


def nested(keyfn, cond='queue_onset', grid=GRID, mix=lambda w, a: a * w['P'] + (1 - a) * w['Pn']):
    choice = {}
    for s in ('pseudo',):
        for k in range(4):
            tr = [w for w in W if w['set'] == s and w['fold'] != k and w['cond'] == cond]
            for key in set(keyfn(w) for w in tr):
                ws = [w for w in tr if keyfn(w) == key]
                sc = [np.mean([iou(w, decode(mix(w, a))) for w in ws]) for a in grid]
                choice[(k, key)] = grid[int(np.argmax(sc))]
    def fn(w):
        if w['cond'] != cond:
            return decode(w['Pn']) if cond == 'queue_onset' else decode(0.7 * w['P'] + 0.3 * w['Pn'])
        a = choice.get((w['fold'], keyfn(w)), 0.7)
        return decode(mix(w, a))
    return fn, choice


print('\n--- A2 per-panel onset blend (nested CV)')
fn, ch = nested(lambda w: w['panel']); report('A2 per-panel onset blend', fn)
print('   chosen weights (fold 0):', {k[1]: v for k, v in ch.items() if k[0] == 0})
print('\n--- A3 per-time-of-day onset blend (nested CV)')
tb = lambda w: 'night' if w['tod'] < 72 else 'am_peak' if w['tod'] < 120 else 'midday' if w['tod'] < 180 else 'pm_peak' if w['tod'] < 234 else 'evening'
fn, ch = nested(tb); report('A3 per-TOD onset blend', fn)
print('   chosen weights (fold 0):', {k[1]: v for k, v in ch.items() if k[0] == 0})
fn, ch = nested(lambda w: 'weekend' if w['dow'] >= 5 else 'weekday'); report('A3b weekday/weekend onset blend', fn)

print('\n--- B4 CatBoost diversity')
report('B4 onset = CatBoost(all feats)', lambda w: decode(w['Pc'] if on(w) else w['Pn']))
report('B4 onset = 0.5 LGB-blend70 + 0.5 CatBoost', lambda w: decode(0.5 * (0.7 * w['P'] + 0.3 * w['Pn']) + 0.5 * w['Pc'] if on(w) else w['Pn']))
report('B4 ongoing = CatBoost(prior-free)', lambda w: decode(0.7 * w['P'] + 0.3 * w['Pn'] if on(w) else w['Pc']))
report('B4 ongoing = 0.5 LGB-nop + 0.5 CatBoost', lambda w: decode(0.7 * w['P'] + 0.3 * w['Pn'] if on(w) else 0.5 * w['Pn'] + 0.5 * w['Pc']))
report('B4 both blended with CatBoost 50/50', lambda w: decode(0.5 * (0.7 * w['P'] + 0.3 * w['Pn']) + 0.5 * w['Pc'] if on(w) else 0.5 * w['Pn'] + 0.5 * w['Pc']))
corr = [np.corrcoef(w['Pn'].ravel(), w['Pc'].ravel())[0, 1] for w in W if w['cond'] == 'queue_ongoing' and w['Pn'].std() > 0 and w['Pc'].std() > 0]
print(f'   ongoing prob correlation LGB-nop vs CatBoost: {np.nanmean(corr):.3f}')

print('\n--- Phase 6 decoder audit')
bp = lambda w: 0.7 * w['P'] + 0.3 * w['Pn'] if on(w) else w['Pn']
for th in (0.3, 0.4, 0.5):
    report(f'D fixed threshold {th}', lambda w, th=th: (bp(w) >= th) | (bp(w) == bp(w).max()) if True else None)


def close_gaps(pred):
    out = pred.copy()
    for s in range(pred.shape[0]):
        r = pred[s]; idx = np.where(r)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a == 2:
                out[s, a + 1] = True
    return out


def drop_isolated(pred):
    out = pred.copy()
    for s in range(pred.shape[0]):
        r = pred[s]
        for l in np.where(r)[0]:
            if not (l > 0 and r[l - 1]) and not (l + 1 < len(r) and r[l + 1]):
                out[s, l] = False
    return out


report('D expected-IoU + close 1-link gaps (ongoing)', lambda w: close_gaps(decode(bp(w)).reshape(w['P'].shape)) if not on(w) else decode(bp(w)))
report('D expected-IoU + drop isolated (ongoing)', lambda w: drop_isolated(decode(bp(w)).reshape(w['P'].shape)) if not on(w) else decode(bp(w)))
report('D expected-IoU per horizon step (ongoing)', lambda w: np.stack([decode(bp(w)[h]) for h in range(6)]) if not on(w) else decode(bp(w)))

print('\n--- robustness slices (baseline vs best candidates)')
pd.DataFrame(RES).to_csv('analysis/t2_oof_results.csv', index=False)
print('saved analysis/t2_oof_results.csv')
