"""t1dark: aggregate the target-month blackout pseudo-holdout (results/t1dark/<PANEL>_f<FOLD><tag>.npz).

Every simulated blackout is evaluated exactly once over the two folds (fold 0 = half B, fold 1 = half A); the ordinary
cells are the same in both folds (two model pairs) -> their squared errors are averaged over the folds.
Arms: base / dark (seed 9), base2 / dark2 (seed 10; from the same file or from tag _s10), derived:
  mix  = 0.5 base + 0.5 dark;  bagB = mean(base, base2);  bagD = mean(dark, dark2);  mixB = 0.5 bagB + 0.5 dark.
Comparisons (B - A): dark-base, dark2-base2, bagD-bagB, mix-base, mixB-bagB, and base2-base (seed-noise reference).
Bootstrap: events (blackouts) resampled within panel-month; ordinary cells by day within panel-month; B = 2000.
Composite S_state per panel-month: per regime r, MSE = w_r * MSE_blackout + (1 - w_r) * MSE_ordinary(r) with
w_r = scored real blackout targets / all scored targets of that regime (rate x eligible cells in the official blackout
rows; production log check: D7_I10_W gap share 0.017), S_r = 0.54 max(0, 1 - rmse_v/25) + 0.46 max(0, 1 - rmse_q/600),
S_state = mean over regimes. 'blk-only' = the arm is used on blackout rows only (ordinary cells keep the reference).
Usage: t1dark_report.py [tag=] [panels=P1,P2,...] [msfx=] [B=2000]"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', 't1dark')
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', 't1dark')
kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
TAG = kw.get('tag', '')
SPL = ('validation', 'private')
B = int(kw.get('B', 2000))
PAIRS = [('base', 'dark'), ('base2', 'dark2'), ('bagB', 'bagD'), ('base', 'mix'), ('bagB', 'mixB'), ('base', 'base2')]


def arm_preds(zs_by_fold, key):
    """{arm: (v, q)} for one evaluation key of one fold (zs: npz objects of that fold; the second one = tag _s10)."""
    P = {}
    for j, z in enumerate(zs_by_fold):
        for a in ('base', 'dark', 'base2', 'dark2'):
            if f'{a}|{key}|v' in z.files:
                name = a if j == 0 else a + '2'
                P[name] = (z[f'{a}|{key}|v'].astype(np.float64), z[f'{a}|{key}|q'].astype(np.float64))
    mean = lambda a, b, wa=0.5: (wa * P[a][0] + (1 - wa) * P[b][0], wa * P[a][1] + (1 - wa) * P[b][1])
    P['mix'] = mean('base', 'dark')
    if 'base2' in P and 'dark2' in P:
        P['bagB'] = mean('base', 'base2'); P['bagD'] = mean('dark', 'dark2')
        P['mixB'] = mean('bagB', 'dark')
    return P


def load(panel):
    out = {}
    fs = [f'{RES}/{panel}_f{f}{TAG}.npz' for f in (0, 1)]
    if not all(os.path.exists(f) for f in fs):
        return None
    zs = [[np.load(f, allow_pickle=True)] for f in fs]
    for f in (0, 1):
        f2 = f'{RES}/{panel}_f{f}_s10.npz'
        if TAG == '' and os.path.exists(f2):
            zs[f].append(np.load(f2, allow_pickle=True))
    for sp in SPL:
        tabs = {}
        for kind in ('blk', 'ord'):
            per = []
            for f in (0, 1):
                z = zs[f][0]; k = f'{sp}_{kind}'
                vt, qt, ln, reg = z[f'{k}|vt'], z[f'{k}|qt'], z[f'{k}|ln'], z[f'{k}|reg']
                unit = z[f'{k}|ev'] if kind == 'blk' else z[f'{k}|d']
                P = arm_preds(zs[f], k)
                df = pd.DataFrame({'u': unit, 'reg': reg}); df['n'] = 1.0
                for a, (pv, pq) in P.items():
                    df[f'sv_{a}'] = (pv - vt) ** 2; df[f'sq_{a}'] = ((pq - qt) / ln) ** 2
                g = df.groupby('u').agg(dict(reg='first', n='sum', **{c: 'sum' for c in df.columns if c[:3] in ('sv_', 'sq_')}))
                per.append(g)
            if kind == 'blk':
                t = pd.concat(per).reset_index()
            else:
                t = per[0].copy()
                for c in t.columns:
                    if c[:3] in ('sv_', 'sq_'):
                        t[c] = 0.5 * (per[0][c] + per[1][c])
                t = t.reset_index()
            tabs[kind] = t
        out[sp] = dict(blk=tabs['blk'], ord=tabs['ord'], info=[json.loads(str(z[0]['info'])) for z in zs])
    meta = np.load(f'{CACHE}/{panel}_meta{kw.get("msfx", "")}.npz', allow_pickle=True)
    for sp in SPL:
        out[sp]['w'] = {r: (float(meta[f'{sp}_nblk_{r}']), float(meta[f'{sp}_ntgt_{r}'])) for r in ('R1', 'R2', 'R3')}
        sim = meta[f'{sp}_sim']
        out[sp]['blk']['cond'] = np.where(sim[out[sp]['blk'].u.to_numpy(), 2] == 0, 'onset', 'ongoing')
    return out


def arms_of(data):
    common = None
    for Dp in data.values():
        for sp in SPL:
            a = {c[3:] for c in Dp[sp]['blk'].columns if c.startswith('sv_')}
            common = a if common is None else common & a
    return common


def rmse(df, a, w=None):
    w = np.ones(len(df)) if w is None else w
    n = (df.n.to_numpy() * w).sum()
    return np.sqrt((df[f'sv_{a}'].to_numpy() * w).sum() / n), np.sqrt((df[f'sq_{a}'].to_numpy() * w).sum() / n)


def s_comp(D, a_blk, a_ord, wb=None, wo=None):
    """composite S_state of one panel-month: arm a_blk on blackout cells, arm a_ord on ordinary cells."""
    blk, o, w = D['blk'], D['ord'], D['w']
    wb = np.ones(len(blk)) if wb is None else wb; wo = np.ones(len(o)) if wo is None else wo
    nb = (blk.n.to_numpy() * wb).sum()
    mbv = (blk[f'sv_{a_blk}'].to_numpy() * wb).sum() / nb; mbq = (blk[f'sq_{a_blk}'].to_numpy() * wb).sum() / nb
    S = []
    for r in ('R1', 'R2', 'R3'):
        nbl, nt = w[r]
        m = (o.reg == r).to_numpy()
        if nt == 0 or not m.any():
            continue
        no = (o.n.to_numpy() * wo * m).sum()
        if no <= 0:
            continue
        mov = (o[f'sv_{a_ord}'].to_numpy() * wo * m).sum() / no; moq = (o[f'sq_{a_ord}'].to_numpy() * wo * m).sum() / no
        wr = nbl / nt
        rv = np.sqrt(wr * mbv + (1 - wr) * mov); rq = np.sqrt(wr * mbq + (1 - wr) * moq)
        S.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
    return float(np.mean(S))


def main():
    panels = kw['panels'].split(',') if 'panels' in kw else sorted({os.path.basename(f).rsplit('_f0', 1)[0]
                                                                   for f in glob.glob(f'{RES}/*_f0{TAG}.npz')})
    data = {p: load(p) for p in panels}
    data = {p: d for p, d in data.items() if d is not None}
    arms = arms_of(data)
    pairs = [(a, b) for a, b in PAIRS if a in arms and b in arms]
    keys = [(p, sp) for p in data for sp in SPL]
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    rows = []
    for p, sp in keys:
        D = data[p][sp]; blk, o = D['blk'], D['ord']
        r = dict(panel=p, month=sp[:3], n_ev=len(blk), n_blk=int(blk.n.sum()), n_ord=int(o.n.sum()),
                 wblk=sum(v[0] for v in D['w'].values()) / max(1, sum(v[1] for v in D['w'].values())),
                 dark_rows=sum(i[f'{sp}_dark_rows'] for i in D['info']))
        for a in sorted(arms):
            r[f'bv_{a}'], r[f'bq_{a}'] = rmse(blk, a)
            r[f'ov_{a}'], r[f'oq_{a}'] = rmse(o, a)
        rows.append(r)
    T = pd.DataFrame(rows)
    show_arms = [a for a in ('base', 'dark', 'base2', 'dark2', 'bagB', 'bagD', 'mix', 'mixB') if a in arms]
    print('per panel-month BLACKOUT RMSE (speed km/h | flow per lane veh/h); n_ev events, n_blk cells, wblk scored blackout share')
    print(T[['panel', 'month', 'n_ev', 'n_blk', 'wblk', 'dark_rows'] + [f'bv_{a}' for a in show_arms] + [f'bq_{a}' for a in show_arms]]
          .round(3).to_string(index=False))
    print('\nper panel-month ORDINARY RMSE')
    print(T[['panel', 'month', 'n_ord'] + [f'ov_{a}' for a in show_arms] + [f'oq_{a}' for a in show_arms]].round(3).to_string(index=False))
    print('\nblackout RMSE by condition of the simulated origin (pooled over panels and months)')
    for cond in ('onset', 'ongoing'):
        acc = {a: [0.0, 0.0] for a in show_arms}; n = 0.0; ne = 0
        for p, sp in keys:
            df = data[p][sp]['blk']; df = df[df.cond == cond]; n += df.n.sum(); ne += len(df)
            for a in show_arms:
                acc[a][0] += df[f'sv_{a}'].sum(); acc[a][1] += df[f'sq_{a}'].sum()
        if n:
            print(f'  {cond:8s} events {ne:4d} cells {int(n):7d} | speed ' + ' '.join(f'{a} {np.sqrt(acc[a][0] / n):.3f}' for a in show_arms)
                  + ' | flow/lane ' + ' '.join(f'{a} {np.sqrt(acc[a][1] / n):.2f}' for a in show_arms))
    rng = np.random.default_rng(0)

    def pooled(ks, kind, wts=None):
        acc = {}; n = 0.0
        for i, (p, sp) in enumerate(ks):
            df = data[p][sp][kind]; w = np.ones(len(df)) if wts is None else wts[i]
            n += (df.n.to_numpy() * w).sum()
            for a in arms:
                v = acc.setdefault(a, [0.0, 0.0])
                v[0] += (df[f'sv_{a}'].to_numpy() * w).sum(); v[1] += (df[f'sq_{a}'].to_numpy() * w).sum()
        return {a: (np.sqrt(v[0] / n), np.sqrt(v[1] / n)) for a, v in acc.items()}

    print('\npooled RMSE (cell-weighted over panels); delta = B - A with 95% bootstrap CI; win = panel-months where B has the lower RMSE')
    for month in ('validation', 'private', 'both'):
        ks = [k for k in keys if month == 'both' or k[1] == month]
        for kind in ('blk', 'ord'):
            pt = pooled(ks, kind)
            bs = []
            for b in range(B):
                wts = [np.bincount(rng.integers(0, len(data[p][sp][kind]), len(data[p][sp][kind])), minlength=len(data[p][sp][kind]))
                       for p, sp in ks]
                bs.append(pooled(ks, kind, wts))
            for a, bb in pairs:
                dv = np.array([x[bb][0] - x[a][0] for x in bs]); dq = np.array([x[bb][1] - x[a][1] for x in bs])
                win_v = sum(rmse(data[p][sp][kind], bb)[0] < rmse(data[p][sp][kind], a)[0] for p, sp in ks)
                win_q = sum(rmse(data[p][sp][kind], bb)[1] < rmse(data[p][sp][kind], a)[1] for p, sp in ks)
                print(f'  {month:10s} {kind} {bb:5s}-{a:5s} speed {pt[a][0]:.3f} -> {pt[bb][0]:.3f} d {pt[bb][0] - pt[a][0]:+.3f} '
                      f'[{np.percentile(dv, 2.5):+.3f}, {np.percentile(dv, 97.5):+.3f}] win {win_v}/{len(ks)} | flow/lane {pt[a][1]:.2f} -> '
                      f'{pt[bb][1]:.2f} d {pt[bb][1] - pt[a][1]:+.2f} [{np.percentile(dq, 2.5):+.2f}, {np.percentile(dq, 97.5):+.2f}] win {win_q}/{len(ks)}')
    print('\ncomposite S_state delta (B - A) per panel-month, mean over panel-months, 95% bootstrap CI; x0.35 = S_total share of a')
    print('blackout panel; "all" = arm on every target, "blk-only" = arm on blackout rows only (ordinary cells keep A)')
    for a, bb in pairs:
        for scope in ('all', 'blk-only'):
            for month in ('validation', 'private', 'both'):
                ks = [k for k in keys if month == 'both' or k[1] == month]
                ao = bb if scope == 'all' else a
                d0 = np.array([s_comp(data[p][sp], bb, ao) - s_comp(data[p][sp], a, a) for p, sp in ks])
                bd = []
                for b in range(B // 4):
                    vals = []
                    for p, sp in ks:
                        D = data[p][sp]
                        wb = np.bincount(rng.integers(0, len(D['blk']), len(D['blk'])), minlength=len(D['blk']))
                        wo = np.bincount(rng.integers(0, len(D['ord']), len(D['ord'])), minlength=len(D['ord']))
                        vals.append(s_comp(D, bb, ao, wb, wo) - s_comp(D, a, a, wb, wo))
                    bd.append(np.mean(vals))
                print(f'  {bb:5s}-{a:5s} {scope:8s} {month:10s} dS_state {d0.mean():+.5f} [{np.percentile(bd, 2.5):+.5f}, '
                      f'{np.percentile(bd, 97.5):+.5f}] | 0.35*d {0.35 * d0.mean():+.6f} | up {int((d0 > 0).sum())}/{len(d0)}')
    T.to_csv(f'{RES}/summary{TAG}.csv', index=False)


if __name__ == '__main__':
    main()
