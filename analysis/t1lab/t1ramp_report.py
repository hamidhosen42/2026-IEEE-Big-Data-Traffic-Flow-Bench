"""t1ramp: aggregate the target-month blackout pseudo-holdout with ramp features (results/t1ramp/<PANEL>_f<FOLD>.npz).

Derived from t1dark_report.py (same bootstrap units: blackout events within panel-month / ordinary cells by day).
Arms: base, ramp, dark, darkramp (seed 9), base2, ramp2 (seed 10); bagB = mean(base, base2), bagR = mean(ramp, ramp2).
Pairs (B - A): bagR-bagB (primary: production bag recipe +- ramps), ramp-base, ramp2-base2, darkramp-dark (t1dark
blackout-member recipe +- ramps), base2-base (seed-noise reference), dark-base (t1dark reproduction).
J proxy per panel-month = 0.35 dS_state (t1dark composite, regime mean) + 0.10 dLWR + 0.05 dS_FD where (per regime r)
dLWR_r = -(nblk_r * d e_blk + nord_r * d e_ord_r) / den_r, e = mean per-cell |dN_pred - dN_true| of the consecutive
eligible pairs touching the cell (t1lab LWR proxy), den_r = sum |dN_true| over the month's eligible pairs (scaled from the
released pairs); dS_FD_r likewise with the per-lane |q - q_FD(k)| (t1lab / score_task3 form). Levels are not observable
here (only deltas). 'blk-only' = arm B on blackout cells only (ordinary cells keep A).
Usage: t1ramp_report.py [panels=P1,...] [B=2000]"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', 't1ramp')
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', 't1ramp')
kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
SPL = ('validation', 'private')
REGS = ('R1', 'R2', 'R3')
B = int(kw.get('B', 2000))
PAIRS = [('bagB', 'bagR'), ('base', 'ramp'), ('base2', 'ramp2'), ('dark', 'darkramp'), ('dark2', 'darkramp2'), ('bagD', 'bagDR'),
         ('base', 'base2'), ('dark', 'dark2'), ('base', 'dark'), ('mixB', 'mixBR'), ('mixB', 'mixBR75'), ('mixB', 'darkramp'),
         ('mixB', 'mixBRb'), ('bagB', 'mixB')]
# deployment (TRACKER EXP_T1_DARK): the t1dark member is used on blackout rows only at W_DARK 0.5 ->
#   mixB = 0.5 bagB + 0.5 dark (current), mixBR = 0.5 bagB + 0.5 darkramp (ramp member replaces it), mixBR75 = 0.25/0.75,
#   mixBRb = 0.5 bagB + 0.5 bagDR (two ramp-dark seeds). lw / fd of mixtures = mixture of the per-cell errors (approx.).
STATS = ('sv', 'sq', 'lw', 'fd')


def arm_preds(z, key, z10=None):
    P = {}
    for a in ('base', 'ramp', 'dark', 'darkramp', 'base2', 'ramp2'):
        if f'{a}|{key}|v' in z.files:
            P[a] = {c: z[f'{a}|{key}|{c}'].astype(np.float64) for c in ('v', 'q', 'lw', 'fd')}
    if z10 is not None:                                   # tag _s10: dark / darkramp with seed 10
        for a in ('dark', 'darkramp'):
            if f'{a}|{key}|v' in z10.files:
                P[a + '2'] = {c: z10[f'{a}|{key}|{c}'].astype(np.float64) for c in ('v', 'q', 'lw', 'fd')}
    def mix(nm, parts):
        if all(a in P for a, _ in parts):
            P[nm] = {c: sum(w * P[a][c] for a, w in parts) for c in ('v', 'q', 'lw', 'fd')}   # lw / fd approx (nonlinear)
    mix('bagB', [('base', .5), ('base2', .5)]); mix('bagR', [('ramp', .5), ('ramp2', .5)])
    mix('bagD', [('dark', .5), ('dark2', .5)]); mix('bagDR', [('darkramp', .5), ('darkramp2', .5)])
    mix('mixB', [('bagB', .5), ('dark', .5)]); mix('mixBR', [('bagB', .5), ('darkramp', .5)])
    mix('mixBR75', [('bagB', .25), ('darkramp', .75)]); mix('mixBRb', [('bagB', .5), ('bagDR', .5)])
    return P


def load(panel):
    fs = [f'{RES}/{panel}_f{f}.npz' for f in (0, 1)]
    if not all(os.path.exists(f) for f in fs):
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]
    z10 = [np.load(f'{RES}/{panel}_f{f}_s10.npz', allow_pickle=True) if os.path.exists(f'{RES}/{panel}_f{f}_s10.npz') else None
           for f in (0, 1)]
    infos = [json.loads(str(z['info'])) for z in zs]
    meta = np.load(f'{CACHE}/{panel}_meta.npz', allow_pickle=True)
    out = {}
    for sp in SPL:
        tabs = {}
        for kind in ('blk', 'ord'):
            per = []
            for f in (0, 1):
                z = zs[f]; k = f'{sp}_{kind}'
                vt, qt, ln, reg = z[f'{k}|vt'], z[f'{k}|qt'], z[f'{k}|ln'], z[f'{k}|reg']
                unit = z[f'{k}|ev'] if kind == 'blk' else z[f'{k}|d']
                P = arm_preds(z, k, z10[f])
                df = pd.DataFrame({'u': unit, 'reg': reg}); df['n'] = 1.0
                for a, D in P.items():
                    df[f'sv_{a}'] = (D['v'] - vt) ** 2; df[f'sq_{a}'] = ((D['q'] - qt) / ln) ** 2
                    df[f'lw_{a}'] = D['lw']; df[f'fd_{a}'] = D['fd']
                aggc = {c: 'sum' for c in df.columns if c[:3] in ('sv_', 'sq_', 'lw_', 'fd_')}
                g = df.groupby(['u', 'reg']).agg(dict(n='sum', **aggc)) if kind == 'ord' else df.groupby('u').agg(dict(reg='first', n='sum', **aggc))
                per.append(g)
            if kind == 'blk':
                t = pd.concat(per).reset_index()
            else:
                t = per[0].copy()
                for c in t.columns:
                    if c[:3] in ('sv_', 'sq_', 'lw_', 'fd_'):
                        t[c] = 0.5 * (per[0][c] + per[1][c])
                t = t.reset_index()
            tabs[kind] = t
        w = {r: (float(meta[f'{sp}_nblk_{r}']), float(meta[f'{sp}_ntgt_{r}'])) for r in REGS}
        den = {r: (infos[0].get(f'{sp}_lwrden_{r}', np.nan), infos[0].get(f'{sp}_fdden_{r}', np.nan)) for r in REGS}
        sim = meta[f'{sp}_sim']
        tabs['blk']['cond'] = np.where(sim[tabs['blk'].u.to_numpy(), 2] == 0, 'onset', 'ongoing')
        out[sp] = dict(blk=tabs['blk'], ord=tabs['ord'], w=w, den=den)
    return out


def rmse(df, a, w=None):
    w = np.ones(len(df)) if w is None else w
    n = (df.n.to_numpy() * w).sum()
    return np.sqrt((df[f'sv_{a}'].to_numpy() * w).sum() / n), np.sqrt((df[f'sq_{a}'].to_numpy() * w).sum() / n)


def comp(D, a_blk, a_ord, wb=None, wo=None):
    """(S_state, -LWRnum/den, -FDnum/den) of one panel-month (regime means); only differences between arms are meaningful
    for the last two (the untouched truth part is dropped)."""
    blk, o, w, den = D['blk'], D['ord'], D['w'], D['den']
    wb = np.ones(len(blk)) if wb is None else wb; wo = np.ones(len(o)) if wo is None else wo
    nb = (blk.n.to_numpy() * wb).sum()
    mb = {c: (blk[f'{c}_{a_blk}'].to_numpy() * wb).sum() / nb for c in STATS}
    S, L, Fq = [], [], []
    for r in REGS:
        nbl, nt = w[r]
        m = (o.reg == r).to_numpy()
        if nt == 0 or not m.any():
            continue
        no = (o.n.to_numpy() * wo * m).sum()
        if no <= 0:
            continue
        mo = {c: (o[f'{c}_{a_ord}'].to_numpy() * wo * m).sum() / no for c in STATS}
        wr = nbl / nt
        rv = np.sqrt(wr * mb['sv'] + (1 - wr) * mo['sv']); rq = np.sqrt(wr * mb['sq'] + (1 - wr) * mo['sq'])
        S.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
        L.append(-(nbl * mb['lw'] + (nt - nbl) * mo['lw']) / den[r][0])
        Fq.append(-(nbl * mb['fd'] + (nt - nbl) * mo['fd']) / den[r][1])
    return np.array([np.mean(S), np.mean(L), np.mean(Fq)])


def main():
    panels = kw['panels'].split(',') if 'panels' in kw else sorted({os.path.basename(f).rsplit('_f0', 1)[0]
                                                                   for f in glob.glob(f'{RES}/*_f0.npz')})
    data = {p: load(p) for p in panels}
    data = {p: d for p, d in data.items() if d is not None}
    arms = None
    for Dp in data.values():
        for sp in SPL:
            a = {c[3:] for c in Dp[sp]['blk'].columns if c.startswith('sv_')}
            arms = a if arms is None else arms & a
    pairs = [(a, b) for a, b in PAIRS if a in arms and b in arms]
    keys = [(p, sp) for p in data for sp in SPL]
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    show = [a for a in ('base', 'bagB', 'ramp', 'bagR', 'dark', 'darkramp', 'bagD', 'bagDR', 'mixB', 'mixBR') if a in arms]
    rows = []
    for p, sp in keys:
        D = data[p][sp]; blk, o = D['blk'], D['ord']
        r = dict(panel=p, month=sp[:3], n_ev=len(blk), n_blk=int(blk.n.sum()), n_ord=int(o.n.sum()))
        for a in show:
            r[f'bv_{a}'], r[f'bq_{a}'] = rmse(blk, a); r[f'ov_{a}'], r[f'oq_{a}'] = rmse(o, a)
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f'panels: {list(data)} | arms: {sorted(arms)}')
    print('per panel-month BLACKOUT RMSE (speed km/h | flow per lane veh/h)')
    print(T[['panel', 'month', 'n_ev', 'n_blk'] + [f'bv_{a}' for a in show] + [f'bq_{a}' for a in show]].round(3).to_string(index=False))
    print('\nper panel-month ORDINARY RMSE')
    print(T[['panel', 'month', 'n_ord'] + [f'ov_{a}' for a in show] + [f'oq_{a}' for a in show]].round(3).to_string(index=False))
    print('\nblackout RMSE by condition of the simulated origin (pooled)')
    for cond in ('onset', 'ongoing'):
        acc = {a: [0.0, 0.0] for a in show}; n = 0.0; ne = 0
        for p, sp in keys:
            df = data[p][sp]['blk']; df = df[df.cond == cond]; n += df.n.sum(); ne += len(df)
            for a in show:
                acc[a][0] += df[f'sv_{a}'].sum(); acc[a][1] += df[f'sq_{a}'].sum()
        print(f'  {cond:8s} events {ne:4d} cells {int(n):7d} | speed ' + ' '.join(f'{a} {np.sqrt(acc[a][0] / n):.3f}' for a in show)
              + ' | flow/lane ' + ' '.join(f'{a} {np.sqrt(acc[a][1] / n):.2f}' for a in show))
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

    print('\npooled RMSE; delta = B - A with 95% bootstrap CI; win = panel-months where B has the lower RMSE')
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
                print(f'  {month:10s} {kind} {bb:8s}-{a:5s} speed {pt[a][0]:.3f} -> {pt[bb][0]:.3f} d {pt[bb][0] - pt[a][0]:+.3f} '
                      f'({100 * (pt[bb][0] / pt[a][0] - 1):+.1f}%) [{np.percentile(dv, 2.5):+.3f}, {np.percentile(dv, 97.5):+.3f}] win {win_v}/{len(ks)} | '
                      f'flow/lane {pt[a][1]:.2f} -> {pt[bb][1]:.2f} d {pt[bb][1] - pt[a][1]:+.2f} ({100 * (pt[bb][1] / pt[a][1] - 1):+.1f}%) '
                      f'[{np.percentile(dq, 2.5):+.2f}, {np.percentile(dq, 97.5):+.2f}] win {win_q}/{len(ks)}')
    print('\nJ proxy delta (B - A) per panel-month, mean over panel-months, 95% bootstrap CI (events / ordinary days resampled)')
    print('  J = 0.35 dS_state + 0.10 dLWR + 0.05 dS_FD; up = panel-months with dJ > 0')
    for a, bb in pairs:
        for scope in ('all', 'blk-only'):
            for month in ('validation', 'private', 'both'):
                ks = [k for k in keys if month == 'both' or k[1] == month]
                ao = bb if scope == 'all' else a
                d0 = np.array([comp(data[p][sp], bb, ao) - comp(data[p][sp], a, a) for p, sp in ks])
                J0 = d0 @ np.array([0.35, 0.10, 0.05])
                bd = []
                for b in range(B // 4):
                    vals = []
                    for p, sp in ks:
                        D = data[p][sp]
                        wb = np.bincount(rng.integers(0, len(D['blk']), len(D['blk'])), minlength=len(D['blk']))
                        ud = D['ord'].u.to_numpy(); uu = np.unique(ud)
                        cnt = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))
                        wo = cnt[np.searchsorted(uu, ud)]
                        vals.append(comp(D, bb, ao, wb, wo) - comp(D, a, a, wb, wo))
                    bd.append(np.mean(vals, 0))
                bd = np.array(bd); bJ = bd @ np.array([0.35, 0.10, 0.05])
                ci = lambda v: f'[{np.percentile(v, 2.5):+.5f}, {np.percentile(v, 97.5):+.5f}]'
                print(f'  {bb:8s}-{a:5s} {scope:8s} {month:10s} dS_state {d0[:, 0].mean():+.5f} {ci(bd[:, 0])} dLWR {d0[:, 1].mean():+.5f} '
                      f'{ci(bd[:, 1])} dS_FD {d0[:, 2].mean():+.5f} | dJ {J0.mean():+.6f} {ci(bJ)} up {int((J0 > 0).sum())}/{len(J0)}')
    T.to_csv(f'{RES}/summary.csv', index=False)


if __name__ == '__main__':
    main()
