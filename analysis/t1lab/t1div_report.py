"""t1div: paired evaluation of the T1 diversity members (EXP_T1_CATB / EXP_T1_UNET; control base3 = third LightGBM seed).

Arms per panel-month (cells and units as t1bspec_report.py; mix = mean(base, base2) stands in for the T1 bag):
  A          ordinary cells mix,                          blackout cells 0.25 mix + 0.75 bs (blackout specialist)
  <cand>@w   ordinary cells (1 - w) mix + w cand,         blackout cells 0.25 ((1 - w) mix + w cand) + 0.75 bs
Post-processing (gate): production recon + TV (functions of kaggle_jobs/swap_fdrecon.py, compiled via t1lab) applied to the
evaluation cells of each fold-month (runs = consecutive evaluation slots of a station; anchors = released cells of the
evaluation view); 'raw' = no post-processing (info). Per-cell S_FD / LWR-proxy errors are recomputed after the blend and
post-processing (exact, not the linear approximation of t1bspec_report). J = 0.35 dS_state + 0.10 dLWR + 0.05 dS_FD per
panel-month (regime means, composite target weights), 95% bootstrap CI (blackout events / ordinary days resampled).
PRE-REGISTERED GATE: see t1div_lib.py.
Usage: t1div_report.py [cands=catb,unet,base3] [B=2000] [tag=]
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1lab  # noqa: E402
from t1ramp_exp import load_panel, month_cube, SPLITS  # noqa: E402
import t1div_lib as L  # noqa: E402

kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
CANDS = kw.get('cands', 'catb,unet,base3').split(',')
WS = tuple(float(w) for w in kw.get('ws', '0.15,0.25,0.35').split(','))
B = int(kw.get('B', 2000))
REGS = ('R1', 'R2', 'R3')
sw = t1lab._swap_module()
GATE, A_, LOGR = sw.GATE, sw.A, sw.LOGR_MAX


# ------------------------------------------------------------------ recon + TV on evaluation cells (swap_fdrecon formulas)
def fd_station(st):
    ln = np.clip(st.lanes.to_numpy(float), 1.0, None); vf = st.vf.to_numpy(float); cap = st.cap.to_numpy(float)
    kc = cap / np.clip(vf, 1, None); kj = st.k_jam.to_numpy(float); kj = np.where(np.isfinite(kj), kj, 100.0 * ln)
    kj = np.maximum(kj, 1.05 * kc)
    capl, kcl, kjl = cap / ln, kc / ln, kj / ln
    w = capl / np.maximum(kjl - kcl, 1e-6)
    ok = (w >= sw.W_RANGE[0]) & (w <= sw.W_RANGE[1]) & np.isfinite(w) & (kjl >= sw.KJL_RANGE[0]) & (kjl <= sw.KJL_RANGE[1]) & np.isfinite(kjl)
    return dict(ln=ln, vf=vf, w=w, kjl=kjl, ok=ok)


def recon(v, q, s, FD):
    ln, vf, w, kjl = FD['ln'][s], FD['vf'][s], FD['w'][s], FD['kjl'][s]
    g = FD['ok'][s] & (v < GATE * vf)
    k = q / ln / np.clip(v, 1, None)
    r = np.where(g, sw.kfd_of(v, w, kjl) / np.maximum(k, 1e-9), 1.0)
    r = np.exp(np.clip(np.log(np.maximum(r, 1e-9)), -LOGR, LOGR))
    return np.where(g, np.clip(v * r ** (-A_), 3, 1.05 * vf), v), np.where(g, np.maximum(q * r ** (1 - A_), 20.0), q)


class Runs:
    """TV runs of evaluation cells (one fold-month): consecutive slots of the same station and day, vf_ok stations only."""

    def __init__(self, d, t, s, FD, kobs, daynum, T=288):
        """daynum[d] = calendar day number of month day d (runs and anchors continue across midnight, as the kernel's
        timestamp-based runs do)."""
        self.n = len(d)
        ok = FD['ok'][s]
        idx = np.flatnonzero(ok)
        g = daynum[d].astype(np.int64) * T + t
        key = s[idx].astype(np.int64) * 10 ** 8 + g[idx]
        o = idx[np.argsort(key, kind='mergesort')]
        so, go = s[o], g[o]
        new = np.ones(len(o), bool); new[1:] = (so[1:] != so[:-1]) | (go[1:] - go[:-1] != 1)
        starts = np.flatnonzero(new); lens = np.diff(np.append(starts, len(o)))
        span = int(daynum.max() + 1) * T
        KF = np.full((span + 2, kobs.shape[2]), np.nan)
        for dd in range(kobs.shape[0]):
            KF[1 + daynum[dd] * T:1 + (daynum[dd] + 1) * T] = kobs[dd]
        e = starts + lens - 1
        self.aL = KF[1 + go[starts] - 1, so[starts]]; self.aR = KF[1 + go[e] + 1, so[e]]
        self.o, self.starts, self.lens = o, starts, lens

    def tv(self, v, q, s, FD):
        ln, vf = FD['ln'][s], FD['vf'][s]
        vn, qn = v.copy(), q.copy()
        K = q / ln / np.clip(v, 1, None); G = v < GATE * vf
        for n in np.unique(self.lens):
            sel = self.lens == n
            idx = self.o[self.starts[sel][:, None] + np.arange(n)[None, :]]
            kk = K[idx]; gg = G[idx]
            mk = np.maximum(kk.mean(1), 1e-6)
            xf = sw.tv_solve(kk, self.aL[sel], self.aR[sel], sw.TV_FREE * mk, sw.TV_FREE_A * mk, sw.TV_ITERS)
            xg = sw.tv_solve(kk, self.aL[sel], self.aR[sel], sw.TV_GATE * mk, sw.TV_GATE_A * mk, sw.TV_ITERS)
            kn = np.where(gg, xg, xf)
            rr = np.exp(np.clip(np.log(np.maximum(kn, 1e-9) / np.maximum(kk, 1e-9)), -LOGR, LOGR))
            a = np.where(gg, A_, 0.0); chg = rr != 1.0
            vv, qq = v[idx], q[idx]
            vn[idx] = np.where(chg & gg, np.clip(vv * rr ** (-a), 3, 1.05 * vf[idx]), vv)
            qn[idx] = np.where(chg, np.maximum(qq * rr ** (1 - a), 20.0), qq)
        return vn, qn


# ------------------------------------------------------------------ per-cell statistics of every arm
def arms_of(P):
    """P: raw member predictions {name: (v, q)} on the concatenated evaluation cells; is_blk flags blackout cells."""
    out = {}
    mix = (0.5 * (P['base'][0] + P['base2'][0]), 0.5 * (P['base'][1] + P['base2'][1]))
    bs = P['bs']; blk = P['is_blk']

    def dep(m):
        return (np.where(blk, 0.25 * m[0] + 0.75 * bs[0], m[0]), np.where(blk, 0.25 * m[1] + 0.75 * bs[1], m[1]))
    out['A'] = dep(mix)
    for c in CANDS:
        if c in P:
            for w in WS:
                out[f'{c}@{w}'] = dep(((1 - w) * mix[0] + w * P[c][0], (1 - w) * mix[1] + w * P[c][1]))
    return out


def panel_tables(panel):
    z, x, st = load_panel(panel)
    meta = L.load_meta(panel)
    FD = fd_station(st)
    vf = st.vf.to_numpy(); ln_s = st.lanes.to_numpy(); Ls = st.length_km.to_numpy()
    kc_l = st.critical_density.to_numpy() / ln_s; kj_l = st.k_jam.to_numpy() / ln_s; cap_l = st.cap.to_numpy() / ln_s
    res = {}
    raw_rmse = []
    for split in SPLITS:
        Mc = month_cube(panel, z, x, split); obs = Mc['obs']
        per = {'blk': [], 'ord': []}
        for fold in (0, 1):
            s0 = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
            info = json.loads(str(s0['info']))
            EM = L.eval_masks(Mc, meta, split, fold)
            gone = EM['HO'] | EM['EB']
            rel = obs & ~gone
            kobs = np.where(rel, Mc['mq'] / ln_s[None, None, :] / np.clip(Mc['mv'], 1, None), np.nan)
            keys = [f'{split}_blk', f'{split}_ord']
            cat = lambda c: np.concatenate([s0[f'{k}|{c}'] for k in keys])
            d, t, s = cat('d'), cat('t'), cat('s'); vt, qt = cat('vt').astype(float), cat('qt').astype(float)
            reg = cat('reg'); unit = np.concatenate([s0[f'{keys[0]}|ev'], s0[f'{keys[1]}|d']])
            nb = len(s0[f'{keys[0]}|d'])
            is_blk = np.arange(len(d)) < nb
            P = {a: (np.concatenate([s0[f'{a}|{k}|v'] for k in keys]).astype(float),
                     np.concatenate([s0[f'{a}|{k}|q'] for k in keys]).astype(float)) for a in ('base', 'base2', 'bs')}
            for c in CANDS:
                f = f'{L.RES}/{panel}_f{fold}_{c}{kw.get("tag", "")}.npz'
                if os.path.exists(f):
                    zc = np.load(f, allow_pickle=True)
                    P[c] = (np.concatenate([zc[f'{k}|v'] for k in keys]).astype(float), np.concatenate([zc[f'{k}|q'] for k in keys]).astype(float))
            P['is_blk'] = is_blk
            # raw member RMSE + error correlation with the mix (diversity)
            mixv = 0.5 * (P['base'][0] + P['base2'][0])
            for kind, m in (('blk', is_blk), ('ord', ~is_blk)):
                r = dict(panel=panel, month=split[:3], fold=fold, kind=kind)
                for a in ['base', 'base2'] + [c for c in CANDS if c in P]:
                    r[f'v_{a}'] = np.sqrt(np.mean((P[a][0][m] - vt[m]) ** 2)); r[f'q_{a}'] = np.sqrt(np.mean(((P[a][1][m] - qt[m]) / ln_s[s[m]]) ** 2))
                    r[f'corr_{a}'] = np.corrcoef(P[a][0][m] - vt[m], mixv[m] - vt[m])[0, 1]
                r['v_mix'] = np.sqrt(np.mean((mixv[m] - vt[m]) ** 2))
                raw_rmse.append(r)
            ARMS = arms_of(P)
            dn = pd.to_datetime(pd.Series(Mc['days'])); daynum = ((dn - dn.iloc[0]).dt.days).to_numpy()
            runs = Runs(d, t, s, FD, kobs, daynum)
            Nt = np.where(obs, Mc['mq'] / np.clip(Mc['mv'], 1, None) * Ls[None, None, :], 0.0)
            okp = obs[:, :-1] & obs[:, 1:]; dNt = Nt[:, 1:] - Nt[:, :-1]
            E3 = np.zeros(obs.shape, bool); E3[d, t, s] = True
            cols = {'u': unit, 'reg': reg, 'n': np.ones(len(d)), 'blk': is_blk}
            for pipe in ('raw', 'post'):
                for a, (v, q) in ARMS.items():
                    if pipe == 'post':
                        v, q = recon(v, q, s, FD); v, q = runs.tv(v, q, s, FD)
                    nm = f'{a}|{pipe}'
                    cols[f'sv_{nm}'] = (v - vt) ** 2; cols[f'sq_{nm}'] = ((q - qt) / ln_s[s]) ** 2
                    kl = q / np.maximum(v, 1) / ln_s[s]; ql = q / ln_s[s]
                    qfd = np.where(kl <= kc_l[s], vf[s] * kl, cap_l[s] / np.maximum(kj_l[s] - kc_l[s], 1e-9) * np.maximum(kj_l[s] - kl, 0))
                    cols[f'fd_{nm}'] = np.abs(ql - qfd)
                    NP = Nt.copy(); NP[d, t, s] = q / np.clip(v, 1, None) * Ls[s]
                    err = np.where(okp, np.abs((NP[:, 1:] - NP[:, :-1]) - dNt), 0.0)
                    lw = np.zeros(NP.shape)
                    lw[:, :-1] += np.where(E3[:, :-1], err, 0); lw[:, 1:] += np.where(~E3[:, :-1] & E3[:, 1:], err, 0)
                    cols[f'lw_{nm}'] = lw[d, t, s]
            df = pd.DataFrame(cols)
            aggc = {c: 'sum' for c in df.columns if c[:3] in ('sv_', 'sq_', 'lw_', 'fd_')}
            per['blk'].append(df[df.blk].groupby('u').agg(dict(reg='first', n='sum', **aggc)))
            per['ord'].append(df[~df.blk].groupby(['u', 'reg']).agg(dict(n='sum', **aggc)))
            if fold == 0:
                den = {r: (info.get(f'{split}_lwrden_{r}', np.nan), info.get(f'{split}_fdden_{r}', np.nan)) for r in REGS}
        tb = pd.concat(per['blk']).reset_index()
        to = per['ord'][0].copy()
        for c in to.columns:
            if c[:3] in ('sv_', 'sq_', 'lw_', 'fd_'):
                to[c] = 0.5 * (per['ord'][0][c] + per['ord'][1][c])
        to = to.reset_index()
        w = {r: (float(meta[f'{split}_nblk_{r}']), float(meta[f'{split}_ntgt_{r}'])) for r in REGS}
        res[split] = dict(blk=tb, ord=to, w=w, den=den)
    return res, raw_rmse


STATS = ('sv', 'sq', 'lw', 'fd')


def prep_arrays(D, arms):
    """numpy views for fast composite computation."""
    blk, o = D['blk'], D['ord']
    return dict(nb=blk.n.to_numpy(float), no=o.n.to_numpy(float), oreg=o.reg.to_numpy(), w=D['w'], den=D['den'],
                ob=o.u.to_numpy(),
                B={a: np.stack([blk[f'{c}_{a}'].to_numpy(float) for c in STATS]) for a in arms},
                O={a: np.stack([o[f'{c}_{a}'].to_numpy(float) for c in STATS]) for a in arms})


def comp(X, a_blk, a_ord, wb=None, wo=None):
    """= t1bspec_report.comp on numpy arrays: [S_state, -LWR err, -FD err] composite of a panel-month."""
    wb = np.ones(len(X['nb'])) if wb is None else wb; wo = np.ones(len(X['no'])) if wo is None else wo
    mb = (X['B'][a_blk] * wb).sum(1) / (X['nb'] * wb).sum()
    S, Lw, Fq = [], [], []
    for r in REGS:
        nbl, nt = X['w'][r]
        m = X['oreg'] == r
        if nt == 0 or not m.any():
            continue
        ww = wo * m
        no = (X['no'] * ww).sum()
        if no <= 0:
            continue
        mo = (X['O'][a_ord] * ww).sum(1) / no
        wr = nbl / nt
        rv = np.sqrt(wr * mb[0] + (1 - wr) * mo[0]); rq = np.sqrt(wr * mb[1] + (1 - wr) * mo[1])
        S.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
        Lw.append(-(nbl * mb[2] + (nt - nbl) * mo[2]) / X['den'][r][0])
        Fq.append(-(nbl * mb[3] + (nt - nbl) * mo[3]) / X['den'][r][1])
    return np.array([np.mean(S), np.mean(Lw), np.mean(Fq)])


def main():
    data, raws = {}, []
    for p in L.PANELS:
        if not all(os.path.exists(f'{L.RES}/{p}_f{f}_{c}{kw.get("tag", "")}.npz') for f in (0, 1) for c in CANDS):
            print('missing candidate files for', p, '- skipped'); continue
        data[p], rr = panel_tables(p); raws += rr
        print('tables', p, flush=True)
    keys = [(p, sp) for p in data for sp in SPLITS]
    arms = ['A'] + [f'{c}@{w}' for c in CANDS for w in WS]
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    RW = pd.DataFrame(raws)
    print('\nRAW member RMSE pooled over panel-month-folds (speed km/h | flow per lane) and corr(error, mix error):')
    for kind in ('ord', 'blk'):
        g = RW[RW.kind == kind]
        print(f'  {kind}: ' + ' '.join(f'{a} v{g[f"v_{a}"].mean():.3f}/q{g[f"q_{a}"].mean():.2f}/corr{g[f"corr_{a}"].mean():.3f}'
                                       for a in ['base', 'base2'] + [c for c in CANDS if f'v_{c}' in g]) + f' | mix v{g.v_mix.mean():.3f}')
    RW.to_csv(f'{L.RES}/raw_rmse.csv', index=False)
    rng = np.random.default_rng(0)
    out = []
    W3 = np.array([0.35, 0.10, 0.05])
    for pipe in ('post', 'raw'):
        an = [f'{a}|{pipe}' for a in arms]
        X = {k: prep_arrays(data[k[0]][k[1]], an) for k in keys}
        # shared bootstrap weights
        BW = []
        for b in range(B // 4):
            row = {}
            for k in keys:
                nb = len(X[k]['nb']); wb = np.bincount(rng.integers(0, nb, nb), minlength=nb)
                ud = X[k]['ob']; uu = np.unique(ud); cnt = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))
                row[k] = (wb, cnt[np.searchsorted(uu, ud)])
            BW.append(row)
        A = f'A|{pipe}'
        for a in an[1:]:
            for scope in ('deploy', 'ord', 'blk'):
                ab, ao = {'deploy': (a, a), 'ord': (A, a), 'blk': (a, A)}[scope]
                d0 = np.array([comp(X[k], ab, ao) - comp(X[k], A, A) for k in keys])
                J0 = d0 @ W3
                bJ = np.array([np.mean([(comp(X[k], ab, ao, *BW[b][k]) - comp(X[k], A, A, *BW[b][k])) @ W3 for k in keys]) for b in range(len(BW))])
                r = dict(pipe=pipe, arm=a.split('|')[0], scope=scope, dS_state=d0[:, 0].mean(), dLWR=d0[:, 1].mean(), dS_FD=d0[:, 2].mean(),
                         dJ=J0.mean(), lo=np.percentile(bJ, 2.5), hi=np.percentile(bJ, 97.5), up=int((J0 > 0).sum()), n=len(J0),
                         dJ_val=J0[[i for i, k in enumerate(keys) if k[1] == 'validation']].mean(),
                         dJ_pri=J0[[i for i, k in enumerate(keys) if k[1] == 'private']].mean(), per=' '.join(f'{v:+.5f}' for v in J0))
                r['gate'] = bool(scope == 'deploy' and pipe == 'post' and r['dJ'] >= 3e-4 and r['lo'] > 0 and r['up'] >= 7)
                out.append(r)
                print(f'{pipe:4s} {r["arm"]:11s} {scope:6s} dS_state {r["dS_state"]:+.5f} dLWR {r["dLWR"]:+.5f} dS_FD {r["dS_FD"]:+.5f} | '
                      f'dJ {r["dJ"]:+.6f} [{r["lo"]:+.6f}, {r["hi"]:+.6f}] up {r["up"]}/{r["n"]} (val {r["dJ_val"]:+.6f} pri {r["dJ_pri"]:+.6f})'
                      + ('  GATE PASS' if r['gate'] else ''), flush=True)
    T = pd.DataFrame(out); T.to_csv(f'{L.RES}/jproxy{kw.get("tag", "")}{kw.get("out", "")}.csv', index=False)
    print('\nGATE (post, deploy):', {c: bool(T[(T.arm.str.startswith(c + '@')) & T.gate].shape[0] > 0) for c in CANDS})


if __name__ == '__main__':
    main()
