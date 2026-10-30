"""t1div round 4 POST-PROCESSING (EXP_T1_PP4) on top of stand-in S3 + recon + TV (design + PRE-REGISTERED gate: t1div_lib.py ROUND 4).

Arms (evaluation cells of the t1bspec pseudo-holdout, 4 panels x 2 months x 2 folds; A = S3 deploy -> recon -> TV):
  noTV                 recon only (info: what TV is worth in this lab)
  Aon:w:alpha:r        A, then honghanhhh interp-N (released anchors, run <= r)            (27 configs)
  Ainst:w:alpha:r      recon, then interp-N, no TV                                          (27 configs)
  B1 nb94 / B2 nb94cap / B3 robina   robinamirbahar neighbour smoothing (+ FD cap / literal 3-stage operator), on top of A
Per-cell statistics, J (0.35 dS_state + 0.10 dLWR + 0.05 dS_FD per panel-month), bootstrap: identical to t1div_report.py
(functions imported from it). Gates: post-processing gate (dJ >= +0.0002, CI lo > 0, >= 7/8); A families cross-fitted over
months (config chosen on the other month); B secondary arms also positive in both months.
Usage: t1div_pp.py [B=2000] [out=_r4pp] [member=lrtcQ@0.35]   (member: INFO rerun with a member blended into S3 first)
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ARGV = sys.argv[1:]
_member = dict(a.split('=') for a in _ARGV if '=' in a).get('member', '')
sys.argv = [sys.argv[0], 'stand=xgb', 'cands=base3' + (',' + _member.split('@')[0] if _member else ''), 'wx=', 'wc=']
import t1div_report as R  # noqa: E402  (stand-in S3 = production steps 1-4; recon / TV / comp / prep_arrays)
sys.argv = [sys.argv[0]] + _ARGV
from t1ramp_exp import load_panel, month_cube, SPLITS  # noqa: E402
import t1div_lib as L  # noqa: E402

kw = dict(a.split('=') for a in _ARGV if '=' in a)
B = int(kw.get('B', 2000)); OUT = kw.get('out', '_r4pp'); MEMBER = kw.get('member', '')
GRID = [(w, a, r) for w in (0.25, 0.5, 1.0) for a in (0.0, 0.5, 1.0) for r in (1, 2, 4)]
ARMS = ['noTV'] + [f'Aon:{w}:{a}:{r}' for w, a, r in GRID] + [f'Ainst:{w}:{a}:{r}' for w, a, r in GRID] + ['B1', 'B2', 'B3']
REGS = ('R1', 'R2', 'R3')
W3 = np.array([0.35, 0.10, 0.05])


def clips(v, q, vf):
    return np.clip(v, 3, 1.05 * vf), np.maximum(q, 20.0)


class Ctx:
    """released-anchor geometry of one fold-month (continuous axis g = 1 + daynum * 288 + t, one padding row each side)."""

    def __init__(self, d, t, s, kobs, daynum, cell):
        span = int(daynum.max() + 1) * 288
        Kg = np.full((span + 2, kobs.shape[2]), np.nan)
        for dd in range(kobs.shape[0]):
            Kg[1 + daynum[dd] * 288:1 + (daynum[dd] + 1) * 288] = kobs[dd]
        self.Kg, self.s, self.cell = Kg, s, cell
        self.g = 1 + daynum[d].astype(np.int64) * 288 + t
        n = Kg.shape[0]; fin = np.isfinite(Kg); ix = np.arange(n)[:, None]
        Pv = np.maximum.accumulate(np.where(fin, ix, -1), axis=0)
        Nx = np.minimum.accumulate(np.where(fin, ix, n)[::-1], axis=0)[::-1]
        self.p = Pv[self.g - 1, s]; self.n = Nx[self.g + 1, s]
        self.has = (self.p >= 0) & (self.n < n)
        pp, nn = np.maximum(self.p, 0), np.minimum(self.n, n - 1)
        self.run = np.where(self.has, nn - pp - 1, 10 ** 6)
        self.ki = np.where(self.has, Kg[pp, s] + (Kg[nn, s] - Kg[pp, s]) * (self.g - pp) / np.maximum(nn - pp, 1), np.nan)
        day = (self.g - 1) // 288
        self.same_day = self.has & ((pp - 1) // 288 == day) & ((nn - 1) // 288 == day)
        self.kp, self.kn = Kg[self.g - 1, s], Kg[self.g + 1, s]           # exact neighbours (released or NaN)


def interp_a(v, q, C, w, alpha, r):
    c = C.cell; k = q / c['ln'] / np.maximum(v, 1)
    ok = C.has & (C.run <= r) & np.isfinite(C.ki) & (C.ki > 0) & (k > 0)
    dl = np.where(ok, w * (np.log(np.maximum(C.ki, 1e-9)) - np.log(np.maximum(k, 1e-9))), 0.0)
    return clips(v * np.exp(-(1 - alpha) * dl), q * np.exp(alpha * dl), c['vf'])


def nb94(v, q, C):
    c = C.cell; k = q / c['ln'] / np.maximum(v, 1)
    both = np.isfinite(C.kp) & np.isfinite(C.kn) & (C.kp > 0) & (C.kn > 0) & (k > 0)
    lk = np.log(np.maximum(k, 1e-9))
    lnew = 0.94 * lk + 0.03 * (np.log(np.where(both, C.kp, 1.0)) + np.log(np.where(both, C.kn, 1.0)))
    dl = np.clip(np.where(both, lnew - lk, 0.0), -0.35, 0.35)
    return clips(v * np.exp(-0.22 * dl), q * np.exp(0.78 * dl), c['vf'])


def fdcap(v, q, C):
    c = C.cell; kl = q / c['ln'] / np.maximum(v, 1)
    cong = c['ok'] & (kl > c['kcl'])
    qstar = c['ln'] * v * c['w'] * c['kjl'] / (v + c['w'])
    return clips(v, np.where(cong, np.minimum(q, qstar), q), c['vf'])


def robina(v, q, C):
    c = C.cell; ln, vf, w, kjl, kcl, capl, ok = c['ln'], c['vf'], c['w'], c['kjl'], c['kcl'], c['capl'], c['ok']
    # stage 1: congested-branch FD + ray-FD intersection (beta 0.70, fq 0.95)
    kl0 = q / ln / np.maximum(v, 1)
    q_tgt = 0.3 * w * np.maximum(kjl - kl0, 0) * ln + 0.7 * np.minimum(capl, v * w * kjl / np.maximum(v + w, 1e-9)) * ln
    q1 = np.where(ok & (kl0 > kcl), 0.05 * q + 0.95 * q_tgt, q)
    # stage 2: interp-N anchor (same day, run <= 4) + neighbour smoothing on the released + evaluation log-density grid
    lkb = np.log(np.maximum(q1 / ln / np.maximum(v, 1), 1e-6))
    okn = C.same_day & (C.run <= 4) & np.isfinite(C.ki) & (C.ki > 0)
    wn = np.where(C.run == 1, 0.14, np.where(C.run == 2, 0.08, 0.04))
    lka = np.where(okn, (1 - wn) * lkb + wn * np.log(np.maximum(np.where(okn, C.ki, 1.0), 1e-6)), lkb)
    LG = np.log(np.maximum(C.Kg, 1e-6)); LG[~np.isfinite(C.Kg)] = np.nan
    LG[C.g, C.s] = lka                                                      # evaluation cells carry their anchored value
    lp, lnx = LG[C.g - 1, C.s], LG[C.g + 1, C.s]
    hp, hn = np.isfinite(lp), np.isfinite(lnx)
    op, on = np.isfinite(C.kp), np.isfinite(C.kn)                            # released neighbours
    both = op & on
    wp = np.where(~hp, 0.0, np.where(both, 0.06, np.where(op, 0.04, 0.05)))
    wq = np.where(~hn, 0.0, np.where(both, 0.06, np.where(on, 0.04, 0.05)))
    lf = lka + wp * (np.where(hp, lp, lka) - lka) + wq * (np.where(hn, lnx, lka) - lka)
    dt = np.clip(lf - lkb, -0.35, 0.35)
    v2 = np.clip(v * np.exp(-0.22 * dt), 5.0, 132.0); q2 = np.clip(q1 * np.exp(0.78 * dt), 0.0, 150000.0)
    # stage 3: gamma isodensity FD scaling (gf 0.035, gc 0.120)
    kl = q2 / ln / np.maximum(v2, 1); ql = np.maximum(q2 / ln, 1e-6)
    qfd = np.where(kl <= kcl, vf * kl, w * np.maximum(kjl - kl, 0))
    gm = (1 - np.where(kl <= kcl, 0.035, 0.12)) + np.where(kl <= kcl, 0.035, 0.12) * np.clip(qfd / ql, 0.8, 1.2)
    vc = v2 * gm; iso = ok & (vc >= 5) & (vc <= 132)
    return clips(np.where(iso, vc, v2), np.where(iso, np.clip(q2 * gm, 0, 150000), q2), vf)


def panel_tables(panel):
    z, x, st = load_panel(panel)
    meta = L.load_meta(panel)
    FD = R.fd_station(st)
    vf = st.vf.to_numpy(); ln_s = st.lanes.to_numpy(); Ls = st.length_km.to_numpy()
    kc_l = st.critical_density.to_numpy() / ln_s; kj_l = st.k_jam.to_numpy() / ln_s; cap_l = st.cap.to_numpy() / ln_s
    kcl_r = st.cap.to_numpy() / np.clip(vf, 1, None) / FD['ln']                # recon convention (kc = cap / vf)
    res, diag = {}, []
    for split in SPLITS:
        Mc = month_cube(panel, z, x, split); obs = Mc['obs']
        per = {'blk': [], 'ord': []}
        for fold in (0, 1):
            s0 = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
            info = json.loads(str(s0['info']))
            EM = L.eval_masks(Mc, meta, split, fold)
            rel = obs & ~(EM['HO'] | EM['EB'])
            kobs = np.where(rel, Mc['mq'] / ln_s[None, None, :] / np.clip(Mc['mv'], 1, None), np.nan)
            keys = [f'{split}_blk', f'{split}_ord']
            cat = lambda c: np.concatenate([s0[f'{k}|{c}'] for k in keys])
            d, t, s = cat('d'), cat('t'), cat('s'); vt, qt = cat('vt').astype(float), cat('qt').astype(float)
            reg = cat('reg'); unit = np.concatenate([s0[f'{keys[0]}|ev'], s0[f'{keys[1]}|d']])
            is_blk = np.arange(len(d)) < len(s0[f'{keys[0]}|d'])
            P = {a: (np.concatenate([s0[f'{a}|{k}|v'] for k in keys]).astype(float),
                     np.concatenate([s0[f'{a}|{k}|q'] for k in keys]).astype(float)) for a in ('base', 'base2', 'bs')}
            for c in R.LOAD:
                zc = np.load(f'{L.RES}/{panel}_f{fold}_{R.fname(c)}.npz', allow_pickle=True)
                P[c] = (np.concatenate([zc[f'{k}|v'] for k in keys]).astype(float), np.concatenate([zc[f'{k}|q'] for k in keys]).astype(float))
            mix = R.stand_mix(P)
            if MEMBER:                                                      # INFO: S3 + member (flow-only if name ends in Q)
                c, wm = MEMBER.split('@'); wm = float(wm)
                mv_ = mix[0] if c.endswith('Q') else P[c][0]
                mix = ((1 - wm) * mix[0] + wm * mv_, (1 - wm) * mix[1] + wm * P[c][1])
            bs = P['bs']
            v0 = np.where(is_blk, 0.25 * mix[0] + 0.75 * bs[0], mix[0]); q0 = np.where(is_blk, 0.25 * mix[1] + 0.75 * bs[1], mix[1])
            dn = pd.to_datetime(pd.Series(Mc['days'])); daynum = ((dn - dn.iloc[0]).dt.days).to_numpy()
            runs = R.Runs(d, t, s, FD, kobs, daynum)
            v1, q1 = R.recon(v0, q0, s, FD)
            v2, q2 = runs.tv(v1, q1, s, FD)
            cell = dict(ln=ln_s[s], vf=vf[s], w=FD['w'][s], kjl=FD['kjl'][s], kcl=kcl_r[s], capl=cap_l[s], ok=FD['ok'][s])
            C = Ctx(d, t, s, kobs, daynum, cell)
            ARM = {'A': (v2, q2), 'noTV': (v1, q1)}
            for w, a, r in GRID:
                ARM[f'Aon:{w}:{a}:{r}'] = interp_a(v2, q2, C, w, a, r)
                ARM[f'Ainst:{w}:{a}:{r}'] = interp_a(v1, q1, C, w, a, r)
            vb, qb = nb94(v2, q2, C)
            ARM['B1'] = (vb, qb); ARM['B2'] = fdcap(vb, qb, C); ARM['B3'] = robina(v2, q2, C)
            ordc = ~is_blk
            diag.append(dict(panel=panel, split=split, fold=fold, n_ord=int(ordc.sum()), fd_bad_links=int((~FD['ok']).sum()),
                             anch2=float(C.has[ordc].mean()), run1=float((C.run[ordc] == 1).mean()), run_le2=float((C.run[ordc] <= 2).mean()),
                             run_le4=float((C.run[ordc] <= 4).mean()), both_nb=float((np.isfinite(C.kp) & np.isfinite(C.kn))[ordc].mean()),
                             cong=float((cell['ok'] & (q2 / cell['ln'] / np.maximum(v2, 1) > cell['kcl']))[ordc].mean()),
                             k_err_model=float(np.mean(np.abs(np.log(q2 / np.maximum(v2, 1)) - np.log(qt / np.maximum(vt, 1)))[ordc & np.isfinite(C.ki) & (C.run <= 1)])),
                             k_err_interp=float(np.mean(np.abs(np.log(C.ki * ln_s[s]) - np.log(qt / np.maximum(vt, 1)))[ordc & np.isfinite(C.ki) & (C.run <= 1)]))))
            Nt = np.where(obs, Mc['mq'] / np.clip(Mc['mv'], 1, None) * Ls[None, None, :], 0.0)
            okp = obs[:, :-1] & obs[:, 1:]; dNt = Nt[:, 1:] - Nt[:, :-1]
            E3 = np.zeros(obs.shape, bool); E3[d, t, s] = True
            cols = {'u': unit, 'reg': reg, 'n': np.ones(len(d)), 'blk': is_blk}
            for a, (v, q) in ARM.items():
                nm = f'{a}|post'
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
        wgt = {r: (float(meta[f'{split}_nblk_{r}']), float(meta[f'{split}_ntgt_{r}'])) for r in REGS}
        res[split] = dict(blk=tb, ord=to, w=wgt, den=den)
    return res, diag


def main():
    data, diags = {}, []
    for p in L.PANELS:
        data[p], dg = panel_tables(p); diags += dg
        print('tables', p, flush=True)
    DG = pd.DataFrame(diags)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print('\nanchor geometry of the ORDINARY evaluation cells (per fold-month) and |log k| error, run-1 cells:')
    print(DG.round(4).to_string(index=False))
    keys = [(p, sp) for p in data for sp in SPLITS]
    an = [f'{a}|post' for a in ['A'] + ARMS]
    X = {k: R.prep_arrays(data[k[0]][k[1]], an) for k in keys}
    rng = np.random.default_rng(0)
    BW = []
    for b in range(B // 4):
        row = {}
        for k in keys:
            nb = len(X[k]['nb']); wb = np.bincount(rng.integers(0, nb, nb), minlength=nb)
            ud = X[k]['ob']; uu = np.unique(ud); cnt = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))
            row[k] = (wb, cnt[np.searchsorted(uu, ud)])
        BW.append(row)
    A = 'A|post'
    base0 = np.array([R.comp(X[k], A, A) for k in keys])
    baseB = np.array([[R.comp(X[k], A, A, *BW[b][k]) for k in keys] for b in range(len(BW))])      # (nB, nK, 3)
    iv = [i for i, k in enumerate(keys) if k[1] == 'validation']; ip = [i for i, k in enumerate(keys) if k[1] == 'private']
    out, PJ = [], {}
    for a in ARMS:
        nm = f'{a}|post'
        scopes = ('deploy', 'ord', 'blk') if (a in ('noTV', 'B1', 'B2', 'B3')) else ('deploy',)
        for scope in scopes:
            ab, ao = {'deploy': (nm, nm), 'ord': (A, nm), 'blk': (nm, A)}[scope]
            d0 = np.array([R.comp(X[k], ab, ao) for k in keys]) - base0
            J0 = d0 @ W3
            DJ = (np.array([[R.comp(X[k], ab, ao, *BW[b][k]) for k in keys] for b in range(len(BW))]) - baseB) @ W3   # (nB, nK)
            bJ = DJ.mean(1)
            r = dict(arm=a, scope=scope, dS_state=d0[:, 0].mean(), dLWR=d0[:, 1].mean(), dS_FD=d0[:, 2].mean(), dJ=J0.mean(),
                     lo=np.percentile(bJ, 2.5), hi=np.percentile(bJ, 97.5), up=int((J0 > 0).sum()), n=len(J0),
                     dJ_val=J0[iv].mean(), dJ_pri=J0[ip].mean(), per=' '.join(f'{v:+.5f}' for v in J0))
            r['gate_plain'] = bool(scope == 'deploy' and r['dJ'] >= 2e-4 and r['lo'] > 0 and r['up'] >= 7)
            if scope == 'deploy':
                PJ[a] = (J0, DJ)
            out.append(r)
            if not a.startswith(('Aon', 'Ainst')) or scope != 'deploy' or True:
                print(f'{a:16s} {scope:6s} dS_state {r["dS_state"]:+.5f} dLWR {r["dLWR"]:+.5f} dS_FD {r["dS_FD"]:+.5f} | dJ {r["dJ"]:+.6f} '
                      f'[{r["lo"]:+.6f}, {r["hi"]:+.6f}] up {r["up"]}/{r["n"]} (val {r["dJ_val"]:+.6f} pri {r["dJ_pri"]:+.6f})'
                      + ('  plain-gate' if r['gate_plain'] else ''), flush=True)
    T = pd.DataFrame(out)
    # ---- gates
    verdict = {}
    for fam in ('Aon', 'Ainst'):
        arms = [a for a in ARMS if a.startswith(fam + ':')]
        best = max(arms, key=lambda a: PJ[a][0].mean())
        sel_v = max(arms, key=lambda a: PJ[a][0][ip].mean())         # chosen on the private month -> scored on validation
        sel_p = max(arms, key=lambda a: PJ[a][0][iv].mean())         # chosen on the validation month -> scored on private
        J0 = np.zeros(len(keys)); DJ = np.zeros_like(PJ[best][1])
        J0[iv] = PJ[sel_v][0][iv]; DJ[:, iv] = PJ[sel_v][1][:, iv]
        J0[ip] = PJ[sel_p][0][ip]; DJ[:, ip] = PJ[sel_p][1][:, ip]
        bx = DJ.mean(1); xd, xlo, xhi, xup = J0.mean(), np.percentile(bx, 2.5), np.percentile(bx, 97.5), int((J0 > 0).sum())
        rb = T[(T.arm == best) & (T.scope == 'deploy')].iloc[0]
        ok = bool(xd >= 2e-4 and xlo > 0 and xup >= 7 and rb.gate_plain)
        verdict[fam] = ok
        print(f'\n{fam}: full-data best {best} dJ {rb.dJ:+.6f} [{rb.lo:+.6f}, {rb.hi:+.6f}] up {rb.up}/8 | cross-fit: config on pri -> val '
              f'{sel_v}, on val -> pri {sel_p}: dJ {xd:+.6f} [{xlo:+.6f}, {xhi:+.6f}] up {xup}/8 -> {"PASS" if ok else "FAIL"}')
        top = T[(T.arm.str.startswith(fam + ':')) & (T.scope == 'deploy')].sort_values('dJ', ascending=False).head(6)
        print(top[['arm', 'dS_state', 'dLWR', 'dS_FD', 'dJ', 'lo', 'up', 'dJ_val', 'dJ_pri']].round(6).to_string(index=False))
        T.loc[len(T)] = dict(arm=f'{fam}:XFIT', scope='deploy', dJ=xd, lo=xlo, hi=xhi, up=xup, n=8, per=' '.join(f'{v:+.5f}' for v in J0),
                             gate_plain=ok)
    for a in ('B1', 'B2', 'B3'):
        r = T[(T.arm == a) & (T.scope == 'deploy')].iloc[0]
        ok = bool(r.gate_plain and (a == 'B2' or (r.dJ_val > 0 and r.dJ_pri > 0)))
        verdict[a] = ok
    print('\nGATE (post-processing, pre-registered ROUND 4' + (f'; INFO base = S3 + {MEMBER}' if MEMBER else '') + '):', verdict)
    T.to_csv(f'{L.RES}/jproxy{OUT}.csv', index=False); DG.to_csv(f'{L.RES}/pp_geometry{OUT}.csv', index=False)


if __name__ == '__main__':
    main()
