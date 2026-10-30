"""t1div round 5 report (EXP_T1_DIV5 members / EXP_T1_W5 weights) on stand-in S4 (design + PRE-REGISTERED gates: t1div_lib.py ROUND 5).

S4 = mirror of tfb-k7: mixX = 0.65 (0.65 mean(base, base2) + 0.35 catb) + 0.35 xgb; ordinary cells (mixX_v, 0.65 mixX_q + 0.35 lrtc_q),
blackout cells 0.25 mixX + 0.75 bs; then recon + TV (functions of t1div_report.py = production swap_fdrecon formulas).
Arms (see t1div_lib.py ROUND 5): S3 (info: S4 without step M), lrtc2m_r / lrtc2m_i / ctl2m, lrtcK@w / lrtcL@w / stkQ@w / base3Q@w,
wq@0.5, blkQ@w / blkVQ@w. Per-cell statistics, J (0.35 dS_state + 0.10 dLWR + 0.05 dS_FD per panel-month), bootstrap: identical to
t1div_report.py / t1div_pp.py (functions imported). Scopes: deploy, ord (= S4 on blackout cells), blk (= S4 on ordinary cells).
Usage: t1div5_report.py [B=2000] [out=_r5]
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], 'stand=xgb', 'cands=base3', 'wx=', 'wc=']
import t1div_report as R  # noqa: E402  (stand_mix = production steps 1-3; recon / TV / comp / prep_arrays)
sys.argv = [sys.argv[0]] + _ARGV
from t1ramp_exp import load_panel, month_cube, SPLITS  # noqa: E402
import t1div_lib as L  # noqa: E402

kw = dict(a.split('=') for a in _ARGV if '=' in a)
B = int(kw.get('B', 2000)); OUT = kw.get('out', '_r5')
WS = (0.15, 0.25, 0.35)
W_LRTC = 0.35
MEMBER_FILES = ('lrtc2r', 'lrtc2i', 'lrtcK', 'lrtcL', 'stkQ')
ARMS = ['S3', 'lrtc2m_r', 'lrtc2m_i', 'ctl2m'] + [f'{c}@{w}' for c in ('base3Q', 'lrtcK', 'lrtcL', 'stkQ') for w in WS] + \
       ['wq@0.5'] + [f'{c}@{w}' for c in ('blkQ', 'blkVQ') for w in (0.15, 0.35)]
CONTROL = {'lrtc2m_r': 'ctl2m', 'lrtc2m_i': 'ctl2m', **{f'{c}@{w}': f'base3Q@{w}' for c in ('lrtcK', 'lrtcL', 'stkQ') for w in WS}}
MEMBER_ARMS = set(CONTROL)
WEIGHT_ARMS = {'wq@0.5', 'blkQ@0.15', 'blkQ@0.35', 'blkVQ@0.15', 'blkVQ@0.35'}
POSTHOC = kw.get('posthoc', '') == '1'           # POST-HOC info arms (written after the gate result; never gate-eligible)
if POSTHOC:
    ARMS = ['r2only', 'stkonly', 'r2only+blkVQ@0.15', 'lrtc2m_r+blkVQ@0.15', 'stkQ@0.35+blkVQ@0.15']
    CONTROL, MEMBER_ARMS, WEIGHT_ARMS = {}, set(), set()
REGS = ('R1', 'R2', 'R3')
W3 = np.array([0.35, 0.10, 0.05])


def arms_of(P, is_blk, vf):
    """deploy-scope (v, q) of every arm on the concatenated evaluation cells (blackout cells first)."""
    mv, mq = R.stand_mix(P)
    bv, bq = P['bs']; lv, lq = P['lrtc']
    blk_v = 0.25 * mv + 0.75 * bv; blk_q = 0.25 * mq + 0.75 * bq
    q4 = (1 - W_LRTC) * mq + W_LRTC * lq                                     # step M on ordinary cells
    S4 = (np.where(is_blk, blk_v, mv), np.where(is_blk, blk_q, q4))

    def ordq(qo):
        return (S4[0], np.where(is_blk, S4[1], np.maximum(qo, 20.0)))
    out = {'A': S4, 'S3': ordq(mq)}
    out['lrtc2m_r'] = ordq((1 - W_LRTC) * mq + W_LRTC * 0.5 * (lq + P['lrtc2r']))
    out['lrtc2m_i'] = ordq((1 - W_LRTC) * mq + W_LRTC * 0.5 * (lq + P['lrtc2i']))
    out['ctl2m'] = ordq((1 - W_LRTC) * mq + W_LRTC * 0.5 * (lq + P['base3'][1]))
    qK = np.maximum(P['lrtcK'] * mv, 20.0)                                   # density member x stand-in speed
    for w in WS:
        for c, qc in (('base3Q', P['base3'][1]), ('lrtcK', qK), ('lrtcL', P['lrtcL']), ('stkQ', P['stkQ'])):
            out[f'{c}@{w}'] = ordq((1 - w) * q4 + w * qc)
    out['wq@0.5'] = ordq(0.5 * mq + 0.5 * lq)
    for w in (0.15, 0.35):
        out[f'blkQ@{w}'] = (S4[0], np.where(is_blk, np.maximum((1 - w) * blk_q + w * lq, 20.0), S4[1]))
        out[f'blkVQ@{w}'] = (np.where(is_blk, np.clip((1 - w) * blk_v + w * lv, 3, 1.05 * vf), S4[0]),
                             np.where(is_blk, np.maximum((1 - w) * blk_q + w * lq, 20.0), S4[1]))
    if POSTHOC:
        def blkvq(a, w=0.15):
            return (np.where(is_blk, np.clip((1 - w) * blk_v + w * lv, 3, 1.05 * vf), a[0]), np.where(is_blk, np.maximum((1 - w) * blk_q + w * lq, 20.0), a[1]))
        po = {'A': S4, 'r2only': ordq((1 - W_LRTC) * mq + W_LRTC * P['lrtc2r']), 'stkonly': ordq((1 - W_LRTC) * mq + W_LRTC * P['stkQ'])}
        po['r2only+blkVQ@0.15'] = blkvq(po['r2only']); po['lrtc2m_r+blkVQ@0.15'] = blkvq(out['lrtc2m_r'])
        po['stkQ@0.35+blkVQ@0.15'] = blkvq(out['stkQ@0.35'])
        return po
    assert set(out) == {'A', *ARMS} and len(out) == len(ARMS) + 1
    return out


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
            for c in ('catb', 'xgb', 'base3', 'lrtc'):
                zc = np.load(f'{L.RES}/{panel}_f{fold}_{c}.npz', allow_pickle=True)
                P[c] = (np.concatenate([zc[f'{k}|v'] for k in keys]).astype(float), np.concatenate([zc[f'{k}|q'] for k in keys]).astype(float))
            for c in MEMBER_FILES:
                zc = np.load(f'{L.RES}/{panel}_f{fold}_{c}.npz', allow_pickle=True)
                P[c] = np.concatenate([zc[f'{k}|{"k" if c == "lrtcK" else "q"}'] for k in keys]).astype(float)
            ARM = arms_of(P, is_blk, vf[s])
            # raw ordinary / blackout flow RMSE per lane of the members, error corr with S4's flow (diversity info)
            q4 = ARM['A'][1]; mv = R.stand_mix(P)[0]
            for kind, m in (('ord', ~is_blk), ('blk', is_blk)):
                r = dict(panel=panel, month=split[:3], fold=fold, kind=kind)
                cand = {'S4': q4, 'mixX': R.stand_mix(P)[1], 'lrtc': P['lrtc'][1], 'base3': P['base3'][1], 'lrtc2r': P['lrtc2r'], 'lrtc2i': P['lrtc2i'],
                        'lrtcL': P['lrtcL'], 'stkQ': P['stkQ'], 'lrtcK_xv': np.maximum(P['lrtcK'] * mv, 20.0)}
                for a, q in cand.items():
                    r[f'q_{a}'] = np.sqrt(np.mean(((q[m] - qt[m]) / ln_s[s[m]]) ** 2))
                    r[f'corr_{a}'] = np.corrcoef(q[m] - qt[m], q4[m] - qt[m])[0, 1]
                r['v_lrtc'] = np.sqrt(np.mean((P['lrtc'][0][m] - vt[m]) ** 2)); r['v_S4'] = np.sqrt(np.mean((ARM['A'][0][m] - vt[m]) ** 2))
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
                den = {r: (info.get(f'{split}_lwrden_{r}', np.nan), info.get(f'{split}_fdden_{r}', np.nan)) for r in REGS}
        tb = pd.concat(per['blk']).reset_index()
        to = per['ord'][0].copy()
        for c in to.columns:
            if c[:3] in ('sv_', 'sq_', 'lw_', 'fd_'):
                to[c] = 0.5 * (per['ord'][0][c] + per['ord'][1][c])
        to = to.reset_index()
        wgt = {r: (float(meta[f'{split}_nblk_{r}']), float(meta[f'{split}_ntgt_{r}'])) for r in REGS}
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
        print(f'RAW flow-per-lane RMSE / corr(err, S4 err) [{kind}]: ' + ' '.join(
            f'{c[2:]} {g[c].mean():.2f}/{g["corr_" + c[2:]].mean():.3f}' for c in g.columns if c.startswith('q_')) +
            f' | speed RMSE lrtc {g.v_lrtc.mean():.3f} S4 {g.v_S4.mean():.3f}')
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
    order = sorted(ARMS, key=lambda a: (a not in CONTROL.values(), ARMS.index(a)))     # controls first
    for a in order:
        nm = f'{a}|post'
        for scope in ('deploy', 'ord', 'blk'):
            ab, ao = {'deploy': (nm, nm), 'ord': (A, nm), 'blk': (nm, A)}[scope]
            d0 = np.array([R.comp(X[k], ab, ao) for k in keys]) - base0
            J0 = d0 @ W3
            bJ = ((np.array([[R.comp(X[k], ab, ao, *BW[b][k]) for k in keys] for b in range(len(BW))]) - baseB) @ W3).mean(1)
            JJ[(a, scope)] = (J0, bJ)
            r = dict(arm=a, scope=scope, dS_state=d0[:, 0].mean(), dLWR=d0[:, 1].mean(), dS_FD=d0[:, 2].mean(), dJ=J0.mean(),
                     lo=np.percentile(bJ, 2.5), hi=np.percentile(bJ, 97.5), up=int((J0 > 0).sum()), n=len(J0),
                     dJ_val=J0[iv].mean(), dJ_pri=J0[ip].mean(), per=' '.join(f'{v:+.5f}' for v in J0))
            gate = None
            if scope == 'deploy' and a in MEMBER_ARMS:
                Jc, bJc = JJ[(CONTROL[a], 'deploy')]
                r.update(ctrl=CONTROL[a], dJ_ctrl=Jc.mean(), dJ_div=J0.mean() - Jc.mean(), div_lo=np.percentile(bJ - bJc, 2.5),
                         div_hi=np.percentile(bJ - bJc, 97.5), div_up=int((J0 > Jc).sum()))
                gate = bool(r['dJ'] >= 3e-4 and r['lo'] > 0 and r['up'] >= 7 and r['dJ_div'] >= 1.5e-4)
            elif scope == 'deploy' and a in WEIGHT_ARMS:
                gate = bool(r['dJ'] >= 2e-4 and r['lo'] > 0 and r['up'] >= 7)
                if a.startswith('blkVQ'):
                    gate = bool(gate and r['dJ_val'] > 0 and r['dJ_pri'] > 0)
            r['gate'] = gate
            out.append(r)
            print(f'{a:11s} {scope:6s} dS_state {r["dS_state"]:+.5f} dLWR {r["dLWR"]:+.5f} dS_FD {r["dS_FD"]:+.5f} | dJ {r["dJ"]:+.6f} '
                  f'[{r["lo"]:+.6f}, {r["hi"]:+.6f}] up {r["up"]}/{r["n"]} (val {r["dJ_val"]:+.6f} pri {r["dJ_pri"]:+.6f})'
                  + (f' | vs {r["ctrl"]} {r["dJ_div"]:+.6f} [{r["div_lo"]:+.6f}, {r["div_hi"]:+.6f}] {r["div_up"]}/{r["n"]}' if 'ctrl' in r else '')
                  + ('  GATE PASS' if gate else ''), flush=True)
    T = pd.DataFrame(out); T.to_csv(f'{L.RES}/jproxy{OUT}.csv', index=False)
    fam = lambda a: a.split('@')[0] if '@' in a else a
    verdict = {}
    for a in [x for x in ARMS if x in MEMBER_ARMS | WEIGHT_ARMS]:
        verdict.setdefault(fam(a), False)
        verdict[fam(a)] |= bool(T[(T.arm == a) & (T.scope == 'deploy')].gate.iloc[0])
    print('\nGATE (pre-registered ROUND 5, post recon + TV, deploy scope vs S4):', verdict)
    if not POSTHOC:
        print('parity: dJ(S3 - S4) should be -(round-4 lrtcQ@0.35 ord scope +0.000537):',
              round(float(T[(T.arm == 'S3') & (T.scope == 'deploy')].dJ.iloc[0]), 6))


if __name__ == '__main__':
    main()
