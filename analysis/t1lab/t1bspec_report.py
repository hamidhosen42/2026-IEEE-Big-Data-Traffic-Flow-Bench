"""t1bspec: aggregate the selection-faithful blackout pseudo-holdout (results/t1bspec/<PANEL>_f<FOLD>.npz).
Same bootstrap units / J proxy as t1ramp_report.py (blackout events within panel-month; ordinary cells by day;
J = 0.35 dS_state + 0.10 dLWR + 0.05 dS_FD per panel-month, regime means). Deployment mixes on blackout rows
(bagB = mean(base, base2) stands for the T1 FD bag):
  PROD    = 0.25 bagB + 0.75 dR                    current production (ramp member, seed 13)
  PROD2   = 0.25 bagB + 0.75 mean(dR, dR2)         after the second ramp member (seed 17) lands
  SPEC    = 0.25 bagB + 0.75 bs                    specialist alone
  SPECAVG = 0.25 bagB + 0.75 mean(dR, bs)          specialist averaged with the ramp member
  SPEC2   = 0.25 bagB + 0.75 mean(bs, bs2)         two specialist seeds
  PLACE   = 0.25 bagB + 0.75 dF                    faithful placement only (no timing columns)
  FEATR   = 0.25 bagB + 0.75 bsR                   timing columns only (random placement)
'blk-only' = B on blackout cells only (ordinary cells keep A), which is the deployment.
Usage: t1bspec_report.py [panels=P1,...] [B=2000]"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', 't1bspec')
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', 't1bspec')
kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
SPL = ('validation', 'private')
REGS = ('R1', 'R2', 'R3')
B = int(kw.get('B', 2000))
STATS = ('sv', 'sq', 'lw', 'fd')
RAW = ('base', 'base2', 'dR', 'dR2', 'dF', 'bsR', 'bs', 'bs2')
PAIRS = [('PROD', 'SPEC'), ('PROD', 'SPECAVG'), ('PROD2', 'SPECAVG'), ('PROD2', 'SPEC2'), ('PROD', 'SPEC2'), ('PROD', 'PROD2'),
         ('PROD', 'PLACE'), ('PROD', 'FEATR'), ('dR', 'bs'), ('dR', 'dR2'), ('bs', 'bs2')]
SHOW = ('bagB', 'dR', 'dF', 'bsR', 'bs', 'PROD', 'PROD2', 'SPEC', 'SPECAVG')


def arm_preds(z, key):
    P = {a: {c: z[f'{a}|{key}|{c}'].astype(np.float64) for c in ('v', 'q', 'lw', 'fd')} for a in RAW if f'{a}|{key}|v' in z.files}

    def mix(nm, parts):
        if all(a in P for a, _ in parts):
            P[nm] = {c: sum(w * P[a][c] for a, w in parts) for c in ('v', 'q', 'lw', 'fd')}   # lw / fd approx (nonlinear)
    mix('bagB', [('base', .5), ('base2', .5)])
    mix('PROD', [('bagB', .25), ('dR', .75)]); mix('PROD2', [('bagB', .25), ('dR', .375), ('dR2', .375)])
    mix('SPEC', [('bagB', .25), ('bs', .75)]); mix('SPECAVG', [('bagB', .25), ('dR', .375), ('bs', .375)])
    mix('SPEC2', [('bagB', .25), ('bs', .375), ('bs2', .375)])
    mix('PLACE', [('bagB', .25), ('dF', .75)]); mix('FEATR', [('bagB', .25), ('bsR', .75)])
    return P


def load(panel):
    fs = [f'{RES}/{panel}_f{f}.npz' for f in (0, 1)]
    if not all(os.path.exists(f) for f in fs):
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]
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
                P = arm_preds(z, k)
                cols = {'u': unit, 'reg': reg, 'n': np.ones(len(vt))}
                for a, D in P.items():
                    cols[f'sv_{a}'] = (D['v'] - vt) ** 2; cols[f'sq_{a}'] = ((D['q'] - qt) / ln) ** 2
                    cols[f'lw_{a}'] = D['lw']; cols[f'fd_{a}'] = D['fd']
                df = pd.DataFrame(cols)
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
        sim = meta[f'{sp}_simF']
        tabs['blk']['cond'] = np.where(sim[tabs['blk'].u.to_numpy(), 2] == 0, 'onset', 'ongoing')
        out[sp] = dict(blk=tabs['blk'], ord=tabs['ord'], w=w, den=den)
    return out


def rmse(df, a):
    n = df.n.sum()
    return np.sqrt(df[f'sv_{a}'].sum() / n), np.sqrt(df[f'sq_{a}'].sum() / n)


def comp(D, a_blk, a_ord, wb=None, wo=None):
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
    keys = [(p, sp) for p in data for sp in SPL]
    arms = set.intersection(*[{c[3:] for c in data[p][sp]['blk'].columns if c.startswith('sv_')} for p, sp in keys])
    pairs = [(a, b) for a, b in PAIRS if a in arms and b in arms]
    show = [a for a in SHOW if a in arms]
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    rows = []
    for p, sp in keys:
        D = data[p][sp]; blk, o = D['blk'], D['ord']
        r = dict(panel=p, month=sp[:3], n_ev=len(blk), n_on=int((blk.cond == 'onset').sum()), n_blk=int(blk.n.sum()))
        for a in show:
            r[f'bv_{a}'], r[f'bq_{a}'] = rmse(blk, a); r[f'ov_{a}'], r[f'oq_{a}'] = rmse(o, a)
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f'panels: {list(data)} | arms: {sorted(arms)}')
    print('per panel-month BLACKOUT RMSE (faithful evaluation blackouts; speed km/h | flow per lane veh/h)')
    print(T[['panel', 'month', 'n_ev', 'n_on', 'n_blk'] + [f'bv_{a}' for a in show] + [f'bq_{a}' for a in show]].round(3).to_string(index=False))
    print('\nper panel-month ORDINARY RMSE (raw members; deployment never uses the specialist here)')
    print(T[['panel', 'month'] + [f'ov_{a}' for a in show[:5]] + [f'oq_{a}' for a in show[:5]]].round(3).to_string(index=False))
    print('\nblackout RMSE by condition (pooled)')
    for cond in ('onset', 'ongoing'):
        acc = {a: [0.0, 0.0] for a in show}; n = 0.0; ne = 0
        for p, sp in keys:
            df = data[p][sp]['blk']; df = df[df.cond == cond]; n += df.n.sum(); ne += len(df)
            for a in show:
                acc[a][0] += df[f'sv_{a}'].sum(); acc[a][1] += df[f'sq_{a}'].sum()
        print(f'  {cond:8s} events {ne:4d} cells {int(n):7d} | speed ' + ' '.join(f'{a} {np.sqrt(acc[a][0] / n):.3f}' for a in show)
              + ' | flow/lane ' + ' '.join(f'{a} {np.sqrt(acc[a][1] / n):.2f}' for a in show))
    rng = np.random.default_rng(0)

    def pooled(ks, kind, wts=None, cond=None):
        acc = {}; n = 0.0
        for i, (p, sp) in enumerate(ks):
            df = data[p][sp][kind]; w = np.ones(len(df)) if wts is None else wts[i]
            if cond is not None:
                w = w * (df.cond == cond).to_numpy()
            n += (df.n.to_numpy() * w).sum()
            for a in arms:
                v = acc.setdefault(a, [0.0, 0.0])
                v[0] += (df[f'sv_{a}'].to_numpy() * w).sum(); v[1] += (df[f'sq_{a}'].to_numpy() * w).sum()
        return {a: (np.sqrt(v[0] / n), np.sqrt(v[1] / n)) for a, v in acc.items()}

    print('\npooled RMSE, both months; delta = B - A with 95% bootstrap CI (events / days resampled); win = panel-months B lower')
    for kind, cond in (('blk', None), ('blk', 'onset'), ('blk', 'ongoing'), ('ord', None)):
        pt = pooled(keys, kind, cond=cond)
        bs = []
        for b in range(B // 2):
            wts = [np.bincount(rng.integers(0, len(data[p][sp][kind]), len(data[p][sp][kind])), minlength=len(data[p][sp][kind])) for p, sp in keys]
            bs.append(pooled(keys, kind, wts, cond))
        for a, bb in pairs:
            dv = np.array([x[bb][0] - x[a][0] for x in bs]); dq = np.array([x[bb][1] - x[a][1] for x in bs])
            win_v = sum(rmse(data[p][sp][kind], bb)[0] < rmse(data[p][sp][kind], a)[0] for p, sp in keys)
            print(f'  {kind}{"/" + cond if cond else "":9s} {bb:8s}-{a:6s} speed {pt[a][0]:.3f} -> {pt[bb][0]:.3f} ({100 * (pt[bb][0] / pt[a][0] - 1):+.1f}%) '
                  f'[{np.percentile(dv, 2.5):+.3f}, {np.percentile(dv, 97.5):+.3f}] win {win_v}/{len(keys)} | flow/lane {pt[a][1]:.2f} -> {pt[bb][1]:.2f} '
                  f'({100 * (pt[bb][1] / pt[a][1] - 1):+.1f}%) [{np.percentile(dq, 2.5):+.2f}, {np.percentile(dq, 97.5):+.2f}]')
    print('\nJ proxy delta (B - A), BLACKOUT ROWS ONLY (deployment), mean over panel-months, 95% bootstrap CI; up = panel-months dJ > 0')
    JT = []
    for a, bb in pairs:
        for month in ('validation', 'private', 'both'):
            ks = [k for k in keys if month == 'both' or k[1] == month]
            d0 = np.array([comp(data[p][sp], bb, a) - comp(data[p][sp], a, a) for p, sp in ks])
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
                    vals.append(comp(D, bb, a, wb, wo) - comp(D, a, a, wb, wo))
                bd.append(np.mean(vals, 0))
            bJ = np.array(bd) @ np.array([0.35, 0.10, 0.05])
            lo, hi = np.percentile(bJ, 2.5), np.percentile(bJ, 97.5)
            print(f'  {bb:8s}-{a:6s} {month:10s} dS_state {d0[:, 0].mean():+.5f} dLWR {d0[:, 1].mean():+.5f} dS_FD {d0[:, 2].mean():+.5f} '
                  f'| dJ {J0.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] up {int((J0 > 0).sum())}/{len(J0)}')
            JT.append(dict(A=a, B=bb, month=month, dJ=J0.mean(), lo=lo, hi=hi, up=int((J0 > 0).sum()), n=len(J0),
                           per=' '.join(f'{v:+.5f}' for v in J0)))
    pd.DataFrame(JT).to_csv(f'{RES}/jproxy_blkonly.csv', index=False)
    T.to_csv(f'{RES}/summary.csv', index=False)


if __name__ == '__main__':
    main()
