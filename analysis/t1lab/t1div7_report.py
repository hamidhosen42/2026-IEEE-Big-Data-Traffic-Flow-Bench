"""t1div round 7 report (EXP_T1_R7) on stand-in S5 = mirror of tfb-k9 (design + PRE-REGISTERED gate: t1div_lib.py ROUND 7).

S5 = t1div5_report.arms_of posthoc 'r2only+blkVQ@0.15' (rebuilt explicitly here and asserted identical): mixX = 0.65 (0.65 mean(base,
base2) + 0.35 catb) + 0.35 xgb; ordinary cells (mixX_v, max(0.65 mixX_q + 0.35 lrtc2r_q, 20)); blackout cells b = 0.25 mixX + 0.75 bs,
(clip(0.85 b_v + 0.15 lrtc_v, 3, 1.05 vf), max(0.85 b_q + 0.15 lrtc_q, 20)); then recon + TV (t1div_report.py = production formulas).
Arms (ROUND 7): knnQ@w / knnVQ@w (ordinary cells, on top of S5), controls base3Q@w / base3VQ@w; lrtc3m (step-M member -> mean of
the flow LRTC ranks (8, 40, 10), (24, 40, 10), (16, 40, 20)), control ctl3m (mean(lrtc2r, base3, base3)); INFO: S4 (tfb-k7 stand-in,
parity), knn0Q@w / knn0VQ@w (plain spec), knnQ_blk@0.15 / knnVQ_blk@0.15 (kNN on blackout cells after step B).
Per-cell statistics, J (0.35 dS_state + 0.10 dLWR + 0.05 dS_FD per panel-month), bootstrap: identical to t1div5_report.py (functions
imported). Scopes: deploy, ord (= S5 on blackout cells), blk (= S5 on ordinary cells).
Usage: t1div7_report.py [B=2000] [out=_r7]
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], 'posthoc=1']
import t1div5_report as R5  # noqa: E402  (posthoc arms: 'A' = S4, 'r2only+blkVQ@0.15' = S5; R5.R = t1div_report with stand=xgb)
sys.argv = [sys.argv[0]] + _ARGV
from t1ramp_exp import load_panel, month_cube, SPLITS  # noqa: E402
import t1div_lib as L  # noqa: E402

R = R5.R
kw = dict(a.split('=') for a in _ARGV if '=' in a)
B = int(kw.get('B', 2000)); OUT = kw.get('out', '_r7')
WS = (0.15, 0.25, 0.35)
S5_ARM = 'r2only+blkVQ@0.15'
GATE_ARMS = [f'{c}@{w}' for c in ('knnQ', 'knnVQ') for w in WS] + ['lrtc3m']
CTRL_ARMS = [f'{c}@{w}' for c in ('base3Q', 'base3VQ') for w in WS] + ['ctl3m']
INFO_ARMS = ['S4'] + [f'{c}@{w}' for c in ('knn0Q', 'knn0VQ') for w in WS] + ['knnQ_blk@0.15', 'knnVQ_blk@0.15']
CONTROL = {**{f'{c}Q@{w}': f'base3Q@{w}' for c in ('knn', 'knn0') for w in WS}, **{f'{c}VQ@{w}': f'base3VQ@{w}' for c in ('knn', 'knn0') for w in WS},
           'lrtc3m': 'ctl3m', 'knnQ_blk@0.15': None, 'knnVQ_blk@0.15': None}
ARMS = CTRL_ARMS + GATE_ARMS + INFO_ARMS
REGS = ('R1', 'R2', 'R3')
W3 = np.array([0.35, 0.10, 0.05])
_T5 = pd.read_csv(f'{L.RES}/jproxy_r5posthoc.csv')
POSTHOC_S5 = float(_T5[(_T5.arm == 'r2only+blkVQ@0.15') & (_T5.scope == 'deploy')].dJ.iloc[0])   # dJ(S5 - S4) of round 5 (+0.000374)


def arms_of7(P, is_blk, vf):
    """deploy-scope (v, q) of every round-7 arm on the concatenated evaluation cells (blackout cells first)."""
    po = R5.arms_of(P, is_blk, vf)
    mv, mq = R.stand_mix(P); bv, bq = P['bs']; lv, lq = P['lrtc']
    w5 = 0.15
    bkv = 0.25 * mv + 0.75 * bv; bkq = 0.25 * mq + 0.75 * bq
    v5 = np.where(is_blk, np.clip((1 - w5) * bkv + w5 * lv, 3, 1.05 * vf), mv)
    q5 = np.where(is_blk, np.maximum((1 - w5) * bkq + w5 * lq, 20.0), np.maximum((1 - R5.W_LRTC) * mq + R5.W_LRTC * P['lrtc2r'], 20.0))
    assert np.array_equal(v5, po[S5_ARM][0]) and np.array_equal(q5, po[S5_ARM][1]), 'S5 rebuild != t1div5_report arms_of r2only+blkVQ@0.15'
    o = ~is_blk

    def onq(qc, w, cells):
        return (v5, np.where(cells, np.maximum((1 - w) * q5 + w * qc, 20.0), q5))

    def onvq(vc, qc, w, cells):
        return (np.where(cells, np.clip((1 - w) * v5 + w * vc, 3, 1.05 * vf), v5), np.where(cells, np.maximum((1 - w) * q5 + w * qc, 20.0), q5))

    def stepm(qm):
        return (v5, np.where(o, np.maximum((1 - R5.W_LRTC) * mq + R5.W_LRTC * qm, 20.0), q5))
    out = {'A': (v5, q5), 'S4': po['A']}
    for w in WS:
        out[f'base3Q@{w}'] = onq(P['base3'][1], w, o); out[f'base3VQ@{w}'] = onvq(P['base3'][0], P['base3'][1], w, o)
        for c in ('knn', 'knn0'):
            out[f'{c}Q@{w}'] = onq(P[c][1], w, o); out[f'{c}VQ@{w}'] = onvq(P[c][0], P[c][1], w, o)
    out['lrtc3m'] = stepm((lq + P['lrtc2r'] + P['lrtc3']) / 3.0)
    out['ctl3m'] = stepm((P['lrtc2r'] + P['base3'][1] + P['base3'][1]) / 3.0)
    out['knnQ_blk@0.15'] = onq(P['knn'][1], 0.15, is_blk); out['knnVQ_blk@0.15'] = onvq(P['knn'][0], P['knn'][1], 0.15, is_blk)
    assert set(out) == {'A', *ARMS} and len(out) == len(ARMS) + 1
    return out


def load_P(panel, fold, split):
    s0 = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
    keys = [f'{split}_blk', f'{split}_ord']
    P = {a: (np.concatenate([s0[f'{a}|{k}|v'] for k in keys]).astype(float),
             np.concatenate([s0[f'{a}|{k}|q'] for k in keys]).astype(float)) for a in ('base', 'base2', 'bs')}
    for c in ('catb', 'xgb', 'base3', 'lrtc', 'knn', 'knn0'):
        zc = np.load(f'{L.RES}/{panel}_f{fold}_{c}.npz', allow_pickle=True)
        P[c] = (np.concatenate([zc[f'{k}|v'] for k in keys]).astype(float), np.concatenate([zc[f'{k}|q'] for k in keys]).astype(float))
    for c in R5.MEMBER_FILES + ('lrtc3',):
        zc = np.load(f'{L.RES}/{panel}_f{fold}_{c}.npz', allow_pickle=True)
        P[c] = np.concatenate([zc[f'{k}|{"k" if c == "lrtcK" else "q"}'] for k in keys]).astype(float)
    return s0, keys, P


def panel_tables(panel):
    z, x, st = load_panel(panel)
    meta = L.load_meta(panel)
    FD = R.fd_station(st)
    vf = st.vf.to_numpy(); ln_s = st.lanes.to_numpy(); Ls = st.length_km.to_numpy()
    kc_l = st.critical_density.to_numpy() / ln_s; kj_l = st.k_jam.to_numpy() / ln_s; cap_l = st.cap.to_numpy() / ln_s
    res, raws = {}, []
    for split in SPLITS:
        Mc = month_cube(panel, z, x, split); obs = Mc['obs']
        per = {'blk': [], 'ord': []}
        for fold in (0, 1):
            s0, keys, P = load_P(panel, fold, split)
            info = json.loads(str(s0['info']))
            EM = L.eval_masks(Mc, meta, split, fold)
            rel = obs & ~(EM['HO'] | EM['EB'])
            kobs = np.where(rel, Mc['mq'] / ln_s[None, None, :] / np.clip(Mc['mv'], 1, None), np.nan)
            cat = lambda c: np.concatenate([s0[f'{k}|{c}'] for k in keys])
            d, t, s = cat('d'), cat('t'), cat('s'); vt, qt = cat('vt').astype(float), cat('qt').astype(float)
            reg = cat('reg'); unit = np.concatenate([s0[f'{keys[0]}|ev'], s0[f'{keys[1]}|d']])
            is_blk = np.arange(len(d)) < len(s0[f'{keys[0]}|d'])
            ARM = arms_of7(P, is_blk, vf[s])
            v5, q5 = ARM['A']; mv, mq = R.stand_mix(P)
            ens3 = (P['lrtc'][1] + P['lrtc2r'] + P['lrtc3']) / 3.0
            for kind, m in (('ord', ~is_blk), ('blk', is_blk)):
                r = dict(panel=panel, month=split[:3], fold=fold, kind=kind)
                cq = {'S5': q5, 'mixX': mq, 'base3': P['base3'][1], 'lrtc': P['lrtc'][1], 'lrtc2r': P['lrtc2r'], 'lrtc3': P['lrtc3'], 'ens3': ens3,
                      'knn': P['knn'][1], 'knn0': P['knn0'][1]}
                for a, q in cq.items():
                    r[f'q_{a}'] = np.sqrt(np.mean(((q[m] - qt[m]) / ln_s[s[m]]) ** 2))
                    r[f'corrq_{a}'] = np.corrcoef(q[m] - qt[m], q5[m] - qt[m])[0, 1]
                for a, v in {'S5': v5, 'base3': P['base3'][0], 'lrtc': P['lrtc'][0], 'knn': P['knn'][0], 'knn0': P['knn0'][0]}.items():
                    r[f'v_{a}'] = np.sqrt(np.mean((v[m] - vt[m]) ** 2))
                    r[f'corrv_{a}'] = np.corrcoef(v[m] - vt[m], v5[m] - vt[m])[0, 1]
                raws.append(r)
            dn = pd.to_datetime(pd.Series(Mc['days'])); daynum = ((dn - dn.iloc[0]).dt.days).to_numpy()
            runs = R.Runs(d, t, s, FD, kobs, daynum)
            Nt = np.where(obs, Mc['mq'] / np.clip(Mc['mv'], 1, None) * Ls[None, None, :], 0.0)
            okp = obs[:, :-1] & obs[:, 1:]; dNt = Nt[:, 1:] - Nt[:, :-1]
            E3 = np.zeros(obs.shape, bool); E3[d, t, s] = True
            cols = {'u': unit, 'reg': reg, 'n': np.ones(len(d)), 'blk': is_blk}
            for a, (v, q) in ARM.items():
                v, q = R.recon(v, q, s, FD); v, q = runs.tv(v, q, s, FD)
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
                den = {r_: (info.get(f'{split}_lwrden_{r_}', np.nan), info.get(f'{split}_fdden_{r_}', np.nan)) for r_ in REGS}
        tb = pd.concat(per['blk']).reset_index()
        to = per['ord'][0].copy()
        for c in to.columns:
            if c[:3] in ('sv_', 'sq_', 'lw_', 'fd_'):
                to[c] = 0.5 * (per['ord'][0][c] + per['ord'][1][c])
        to = to.reset_index()
        wgt = {r_: (float(meta[f'{split}_nblk_{r_}']), float(meta[f'{split}_ntgt_{r_}'])) for r_ in REGS}
        res[split] = dict(blk=tb, ord=to, w=wgt, den=den)
    return res, raws


def main():
    data, raws = {}, []
    for p in L.PANELS:
        data[p], rr = panel_tables(p); raws += rr
        print('tables', p, flush=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    RW = pd.DataFrame(raws); RW.to_csv(f'{L.RES}/raw_rmse{OUT}.csv', index=False)
    for kind in ('ord', 'blk'):
        g = RW[RW.kind == kind]
        print(f'RAW [{kind}] flow RMSE per lane / corr(err, S5 err): ' + ' '.join(f'{c[2:]} {g[c].mean():.2f}/{g["corrq_" + c[2:]].mean():.3f}'
                                                                         for c in g.columns if c.startswith('q_')))
        print(f'RAW [{kind}] speed RMSE km/h / corr(err, S5 err): ' + ' '.join(f'{c[2:]} {g[c].mean():.3f}/{g["corrv_" + c[2:]].mean():.3f}'
                                                                       for c in g.columns if c.startswith('v_')))
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
    baseB = np.array([[R.comp(X[k], A, A, *BW[b][k]) for k in keys] for b in range(len(BW))])
    iv = [i for i, k in enumerate(keys) if k[1] == 'validation']; ip = [i for i, k in enumerate(keys) if k[1] == 'private']
    out, JJ = [], {}
    for a in ARMS:                                         # controls first (CTRL_ARMS lead ARMS)
        nm = f'{a}|post'
        for scope in ('deploy', 'ord', 'blk'):
            ab, ao = {'deploy': (nm, nm), 'ord': (A, nm), 'blk': (nm, A)}[scope]
            d0 = np.array([R.comp(X[k], ab, ao) for k in keys]) - base0
            J0 = d0 @ W3
            bJ = ((np.array([[R.comp(X[k], ab, ao, *BW[b][k]) for k in keys] for b in range(len(BW))]) - baseB) @ W3).mean(1)
            JJ[(a, scope)] = (J0, bJ)
            r = dict(arm=a, scope=scope, role='gate' if a in GATE_ARMS else 'control' if a in CTRL_ARMS else 'info',
                     dS_state=d0[:, 0].mean(), dLWR=d0[:, 1].mean(), dS_FD=d0[:, 2].mean(), dJ=J0.mean(),
                     lo=np.percentile(bJ, 2.5), hi=np.percentile(bJ, 97.5), up=int((J0 > 0).sum()), n=len(J0),
                     dJ_val=J0[iv].mean(), dJ_pri=J0[ip].mean(), per=' '.join(f'{v:+.5f}' for v in J0))
            gate = None
            if CONTROL.get(a) and (CONTROL[a], scope) in JJ:
                Jc, bJc = JJ[(CONTROL[a], scope)]
                r.update(ctrl=CONTROL[a], dJ_ctrl=Jc.mean(), dJ_div=J0.mean() - Jc.mean(), div_lo=np.percentile(bJ - bJc, 2.5),
                         div_hi=np.percentile(bJ - bJc, 97.5), div_up=int((J0 > Jc).sum()))
            if scope == 'deploy' and a in GATE_ARMS:
                gate = bool(r['dJ'] >= 3e-4 and r['lo'] > 0 and r['up'] >= 7 and r['dJ_div'] >= 1.5e-4)
            r['gate'] = gate
            out.append(r)
            print(f'{r["role"]:7s} {a:15s} {scope:6s} dS_state {r["dS_state"]:+.5f} dLWR {r["dLWR"]:+.5f} dS_FD {r["dS_FD"]:+.5f} | dJ {r["dJ"]:+.6f} '
                  f'[{r["lo"]:+.6f}, {r["hi"]:+.6f}] up {r["up"]}/{r["n"]} (val {r["dJ_val"]:+.6f} pri {r["dJ_pri"]:+.6f})'
                  + (f' | vs {r["ctrl"]} {r["dJ_div"]:+.6f} [{r["div_lo"]:+.6f}, {r["div_hi"]:+.6f}] {r["div_up"]}/{r["n"]}' if 'ctrl' in r else '')
                  + ('  GATE PASS' if gate else ''), flush=True)
    T = pd.DataFrame(out); T.to_csv(f'{L.RES}/jproxy{OUT}.csv', index=False)
    fam = lambda a: a.split('@')[0]
    verdict = {}
    for a in GATE_ARMS:
        verdict.setdefault(fam(a), False)
        verdict[fam(a)] |= bool(T[(T.arm == a) & (T.scope == 'deploy')].gate.iloc[0])
    print('\nGATE (pre-registered ROUND 7, post recon + TV, deploy scope vs S5):', verdict)
    ps = float(T[(T.arm == 'S4') & (T.scope == 'deploy')].dJ.iloc[0])
    print(f'parity: dJ(S4 - S5) = {ps:+.7f}; round-5 post-hoc dJ(S5 - S4) = {POSTHOC_S5:+.7f} -> sum {ps + POSTHOC_S5:+.1e}')
    assert abs(ps + POSTHOC_S5) < 1e-12, 'S5 stand-in does not reproduce the round-5 post-hoc number'
    passing = T[(T.scope == 'deploy') & (T.gate == True)]  # noqa: E712
    if len(passing):
        b = passing.sort_values('dJ', ascending=False).iloc[0]
        print(f'DEPLOYMENT (pre-registered rule): {b.arm} dJ {b.dJ:+.6f}')
    else:
        print('DEPLOYMENT: none (no member passes)')


if __name__ == '__main__':
    main()
