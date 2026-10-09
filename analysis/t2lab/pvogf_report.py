"""pvogf REPORT (light: decode + scoring only, no training).  Design and the PRE-REGISTERED rule: pvogf_lib.py.
Inputs: results/pvogf/probs/{MD_s0, MD_s1, T0_s0, FULL_s0}.pkl, the pvogf_build.py caches, results/pvogf/oof_f0.pkl (optional) and,
for the fidelity check, the kernel outputs of tfb-t2occ / tfb-t2ogfa / tfb-t2ogfb (t2v26occ_probs.npz, t2v27{a,b}_probs.npz) plus the
lab pairs of the original all2 FM-PM (ogf_combo_pm.pkl) in DL / OUT (optional arguments).
Sections
  1 many-day pseudo-March (days >= 15): candidates (a) pvogf, (b) pvogfv, (c) pvotv vs pvo; seed-average (binding), seed 0, seed 1
  2 FM-PM private: lab-emulated April decisions (FULL_s0, T0_s0) on the 40 official private ongoing windows, priced with the seed-0
    many-day lab pairs (fpmatch_lib.forecast: panel-exact, +-2 h, all-cell counts)
  3 fidelity of the lab emulation: train-only all2 footprint, lab (T0) vs kernel (fs4occ -> fs4ogf) on the same private windows
  4 train OOF (fold 0, pvo recipe), if available
Usage: pvogf_report.py [DL=<dir with tfb-t2occ, tfb-t2ogfa, tfb-t2ogfb>] [OUT=<dir with ogf_combo_pm.pkl>]"""
import os, sys
from importlib import util
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G

kw = dict(a.split('=', 1) for a in sys.argv[1:])
DL, OUTX = kw.get('DL'), kw.get('OUT')
F, ONGOING = G.F, G.ONGOING
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
_s = util.spec_from_file_location('fpa', f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts/fpmatch_lib.py')
fpa = util.module_from_spec(_s); _s.loader.exec_module(fpa)
doc = G.__doc__; print(doc[doc.index('PRE-REGISTERED'):doc.index('Data and code')])


def load(tag):
    f = f'{G.RES}/probs/{tag}.pkl'
    return {k: v.astype(np.float32) for k, v in pd.read_pickle(f).items()} if os.path.exists(f) else None


def complete(P, members):
    return P if P is not None and members <= {k[2] for k in P} else None


M6 = {f'{a}_{m}' for a in ('base', 'ogf') for m in ('np', 'id', 'full')}
MD = {s: complete(load(f'MD_s{s}'), M6) for s in (0, 1)}; MD = {s: v for s, v in MD.items() if v is not None}
T0 = complete(load('T0_s0'), {'base_np', 'base_id', 'ogf_np', 'ogf_id'}); FULL = complete(load('FULL_s0'), M6)
print('complete inputs: MD seeds', sorted(MD), '| T0', T0 is not None, '| FULL', FULL is not None)
pvo_p = lambda P, s, p, arm: 0.25 * P[(s, p, f'{arm}_np')] + 0.25 * P[(s, p, f'{arm}_id')] + 0.5 * P[(s, p, f'{arm}_full')]
npid = lambda P, s, p, arm: 0.5 * (P[(s, p, f'{arm}_np')] + P[(s, p, f'{arm}_id')])
CANDS = ('pvogf', 'pvogfv', 'pvotv')

# ================================================================ 1. many-day pseudo-March
cfgs = {f's{s}': [s] for s in MD}
if len(MD) == 2:
    cfgs['avg'] = [0, 1]
rows = []
for p in lab.PANELS:
    fam = p.rsplit('_', 1)[0]
    for s in G.PM_SETS:
        z = np.load(f'{G.CACHE}/ev_{p}__{s}.npz'); truth, ev, ql = z['truth'], z['ev'], z['qlast']
        Gtr = G.decode6(npid(T0, s, p, 'ogf')) if T0 else None; Btr = G.decode6(npid(T0, s, p, 'base')) if T0 else None
        for cfg, seeds in cfgs.items():
            pA = sum(pvo_p(MD[x], s, p, 'base') for x in seeds) / len(seeds); pB = sum(pvo_p(MD[x], s, p, 'ogf') for x in seeds) / len(seeds)
            A = G.decode6(pA); B = G.decode6(pB); Bv, addB = G.veto(A, B, ql)
            arms = dict(pvo=A, pvogf=B, pvogfv=Bv)
            addC = np.full(len(A), -1)
            if T0:
                Cv, addC = G.veto(A, Gtr, ql); arms.update(pvotv=Cv, tr_base=Btr, tr_ogf=Gtr)
            for arm, dec in arms.items():
                rows.append(pd.DataFrame(dict(cfg=cfg, arm=arm, set=s, panel=p, family=fam, cond=ONGOING, d=z['d'], t=z['t'], addB=addB, addC=addC,
                                              chg=(dec ^ A).sum((1, 2)), **G.score_dec(dec, truth, ev, ql))))
D = pd.concat(rows, ignore_index=True); D.to_pickle(f'{G.RES}/pm_frames.pkl')


def cmp(cfg, s, a, b, sub=None):
    g = D[(D.cfg == cfg) & (D.set == s)]
    if sub is not None:
        g = g[sub(g)]
    A = g[g.arm == a]; B = g[g.arm == b]
    bt = lab.paired_bootstrap(A, B); dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
    return dict(cfg=cfg, set=s, B=b, A=a, n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                p_pos=bt['p_pos'], panels=f'+{int((dl > 1e-9).sum())}/-{int((dl < -1e-9).sum())}', win_changed=int((B.chg.to_numpy() > 0).sum()) if a == 'pvo' else -1,
                dn_all=int(B.n_all.sum() - A.n_all.sum()), per_panel={k: round(v, 4) for k, v in dl.items()})


T1 = []
print('\n== 1. MANY-DAY pseudo-March (March days >= 15; training March rows days < 15; lab scale 255 leaves / 500 trees) ==')
for cfg in [c for c in ('avg', 's0', 's1') if c in cfgs]:
    for s in G.PM_SETS:
        for a, b in [('pvo', c) for c in CANDS] + [('tr_base', 'pvo'), ('tr_base', 'tr_ogf')]:
            if b in set(D.arm) and a in set(D.arm):
                r = cmp(cfg, s, a, b); T1.append(r)
                print(f"{cfg:3s} {s:10s} {b:7s} - {a:7s} n={r['n']:4d} {r['iou_B']:.4f} vs {r['iou_A']:.4f} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                      f" p_pos {r['p_pos']:.3f} panels {r['panels']} windows changed {r['win_changed']} dcells {r['dn_all']:+d}")
T1 = pd.DataFrame(T1); T1.to_csv(f'{G.RES}/pm_table.csv', index=False)
print('\nper-panel deltas (seed average if available):')
cfg0 = 'avg' if 'avg' in cfgs else 's0'
for s in G.PM_SETS:
    for c in CANDS:
        r = T1[(T1.cfg == cfg0) & (T1.set == s) & (T1.B == c) & (T1.A == 'pvo')]
        if len(r):
            print(f'  {s:10s} {c:7s}', r.iloc[0]['per_panel'])
# growth / shrink split of the pvogf change (label-free direction, as in the transfer analysis)
g = D[(D.cfg == cfg0)]
A_ = g[g.arm == 'pvo'].set_index(['set', 'panel', 'd', 't']); B_ = g[g.arm == 'pvogf'].set_index(['set', 'panel', 'd', 't'])
dn = B_.n_all - A_.n_all; dI = B_.iou - A_.iou
print('\npvogf vs pvo by size change (all-cell dn), seed average:')
for lo, hi, lb in ((-999, -10, 'dn <= -10'), (-9, -1, '-9..-1'), (0, 0, '0'), (1, 9, '1..9'), (10, 999, 'dn >= 10')):
    m = (dn >= lo) & (dn <= hi)
    print(f'  {lb:10s} windows {int(m.sum()):4d} mean dIoU {float(dI[m].mean()) if m.any() else float("nan"):+.4f}')
print(f'  windows where pvogf adds > 10 cells outside qlast (vetoed in (b)): {int((B_.addB > 10).sum())} of {len(B_)}; mean dIoU there {float(dI[B_.addB > 10].mean()) if (B_.addB > 10).any() else float("nan"):+.4f}')

# ================================================================ 2. FM-PM private (lab-emulated April decisions)
FM = []
if FULL is not None and T0 is not None:
    off = {c: [] for c in CANDS}; dec_priv = {}
    for p in lab.PANELS:
        z = np.load(f'{G.CACHE}/priv_{p}.npz'); og = z['cond'] == ONGOING; ql = z['qlast'][og]
        A = G.decode6(pvo_p(FULL, 'private', p, 'base'))[og]; B = G.decode6(pvo_p(FULL, 'private', p, 'ogf'))[og]
        Bv, addB = G.veto(A, B, ql); Gt = G.decode6(npid(T0, 'private', p, 'ogf'))[og]; Cv, addC = G.veto(A, Gt, ql)
        Bt = G.decode6(npid(T0, 'private', p, 'base'))[og]
        dec_priv[p] = dict(pvo=A, pvogf=B, pvogfv=Bv, pvotv=Cv, tr_base=Bt, tr_ogf=Gt, qlast=ql, wid=z['window_id'][og], tod=z['tod'][og])
        for c, dec in (('pvogf', B), ('pvogfv', Bv), ('pvotv', Cv)):
            for i in range(len(A)):
                off[c].append(dict(window_id=z['window_id'][og][i], panel=p, tod=float(z['tod'][og][i]), nA=int(A[i].sum()), nB=int(dec[i].sum()),
                                   changed=int((A[i] ^ dec[i]).sum()), addout=int(((dec[i] & ~A[i]) & ~np.repeat(ql[i][None], F, 0)).sum())))
    pd.to_pickle(dec_priv, f'{G.RES}/private_decisions_emulated.pkl')
    print('\n== 2. FM-PM PRIVATE (lab-emulated April decisions: FULL_s0 = F03 + all 31 March days; T0_s0 train-only) ==')
    for c in CANDS:
        o = pd.DataFrame(off[c]); o['dn'] = o.nB - o.nA
        for lab_cfg, pooled in (('s0', False), ('s1', False), ('avg', False), ('s0', True)):
            if lab_cfg not in cfgs:
                continue
            g = D[(D.cfg == lab_cfg)]
            L = g[g.arm == 'pvo'][['set', 'panel', 'd', 't', 'iou', 'n_all']].merge(g[g.arm == c][['set', 'panel', 'd', 't', 'iou', 'n_all']],
                                                                                    on=['set', 'panel', 'd', 't'], suffixes=('_a', '_b'))
            L['dI'] = L.iou_b - L.iou_a; L['dn'] = L.n_all_b - L.n_all_a; L['tod'] = L.t / 12
            f = fpa.forecast(L, o, panel_exact=not pooled)
            r = dict(cand=c, lab_pairs=lab_cfg + (' pooled' if pooled else ''), windows_changed=int((o.changed > 0).sum()), cells_changed=int(o.changed.sum()),
                     pos_pvo=int(o.nA.sum()), pos_cand=int(o.nB.sum()), added_outside_qlast=int(o.addout.sum()), fm_dIoU=float(f.mean()), fm_lb=0.15 * float(f.mean()))
            FM.append(r)
            print(f"{c:7s} lab pairs {r['lab_pairs']:10s} | windows changed {r['windows_changed']:2d}/40 cells {r['cells_changed']:4d} | positives pvo {r['pos_pvo']} -> {r['pos_cand']}"
                  f" (added outside qlast {r['added_outside_qlast']}) | FM-PM dIoU {r['fm_dIoU']:+.4f} -> LB-equiv {r['fm_lb']:+.5f}")
            if lab_cfg == 's0' and not pooled:
                o.assign(fm=f).to_csv(f'{G.RES}/fm_private_{c}.csv', index=False)
    pd.DataFrame(FM).to_csv(f'{G.RES}/fm_private.csv', index=False)

# ================================================================ 3. fidelity of the emulation (train-only all2 footprint, lab vs kernel)
if DL and T0 is not None and os.path.exists(f'{DL}/tfb-t2occ/t2v26occ_probs.npz'):
    occ = np.load(f'{DL}/tfb-t2occ/t2v26occ_probs.npz'); ga = np.load(f'{DL}/tfb-t2ogfa/t2v27a_probs.npz'); gb = np.load(f'{DL}/tfb-t2ogfb/t2v27b_probs.npz')
    dec_priv = pd.read_pickle(f'{G.RES}/private_decisions_emulated.pkl')
    rows = []; ql_ok = 0; nw = 0
    for p in lab.PANELS:
        k = f'{p}__private__'
        pa = (occ[k + 'fs_np'].astype(np.float32) + occ[k + 'fs_id'].astype(np.float32) + occ[k + 'fs_np1'].astype(np.float32) + occ[k + 'fs_id1'].astype(np.float32)) / 4
        pb = (((ga[k + 'fs_np'] + ga[k + 'fs_id']) + gb[k + 'fs_np1']) + gb[k + 'fs_id1']) / 4
        wid = [str(w) for w in occ[k + 'window_id']]; e = dec_priv[p]; pos = {w: i for i, w in enumerate(e['wid'])}
        for i, w in enumerate(wid):
            j = pos[w]; L_ = pa.shape[2]
            Ak = G.K.decode(pa[i].ravel()).reshape(F, L_); Bk = G.K.decode(pb[i].ravel()).reshape(F, L_)
            Al, Bl = e['tr_base'][j], e['tr_ogf'][j]; q = np.repeat(e['qlast'][j][None], F, 0)
            nw += 1; ql_ok += int(np.array_equal(occ[k + 'qlast'][i], e['qlast'][j]))
            rows.append(dict(panel=p, window_id=w, tod=float(e['tod'][j]), dn_kernel=int(Bk.sum() - Ak.sum()), dn_lab=int(Bl.sum() - Al.sum()),
                             chg_kernel=int((Ak ^ Bk).sum()), chg_lab=int((Al ^ Bl).sum()), addout_kernel=int((Bk & ~Ak & ~q).sum()), addout_lab=int((Bl & ~Al & ~q).sum()),
                             jacc_base=float((Ak & Al).sum() / max((Ak | Al).sum(), 1)), jacc_ogf=float((Bk & Bl).sum() / max((Bk | Bl).sum(), 1)),
                             nA_kernel=int(Ak.sum()), nB_kernel=int(Bk.sum()), nA_lab=int(Al.sum()), nB_lab=int(Bl.sum())))
    Fd = pd.DataFrame(rows); Fd.to_csv(f'{G.RES}/fidelity_private.csv', index=False)
    print('\n== 3. FIDELITY of the lab emulation on the 40 private windows: train-only all2 footprint, lab (T0 npid) vs kernel (fs4occ -> fs4ogf) ==')
    print(f'qlast identical lab vs kernel: {ql_ok}/{nw} windows')
    print(f"kernel: windows changed {int((Fd.chg_kernel > 0).sum())}, cells changed {Fd.chg_kernel.sum()}, dn {Fd.dn_kernel.sum():+d}, added outside qlast {Fd.addout_kernel.sum()}")
    print(f"lab   : windows changed {int((Fd.chg_lab > 0).sum())}, cells changed {Fd.chg_lab.sum()}, dn {Fd.dn_lab.sum():+d}, added outside qlast {Fd.addout_lab.sum()}")
    print(f"per-window dn correlation {np.corrcoef(Fd.dn_kernel, Fd.dn_lab)[0, 1]:+.3f} | sign agreement on windows changed in both "
          f"{int(((np.sign(Fd.dn_kernel) == np.sign(Fd.dn_lab)) & (Fd.chg_kernel > 0) & (Fd.chg_lab > 0)).sum())}/{int(((Fd.chg_kernel > 0) & (Fd.chg_lab > 0)).sum())}"
          f" | decision Jaccard lab vs kernel: base {Fd.jacc_base.mean():.3f}, ogf {Fd.jacc_ogf.mean():.3f}")
    if OUTX and os.path.exists(f'{OUTX}/ogf_combo_pm.pkl'):
        C0 = pd.read_pickle(f'{OUTX}/ogf_combo_pm.pkl')
        L_ = C0[['set', 'panel', 'd', 't', 'iou_base', 'iou_all2', 'n_base', 'n_all2']].rename(columns={'iou_all2': 'iou_b', 'iou_base': 'iou_a', 'n_base': 'n_pred_a', 'n_all2': 'n_pred_b'})
        L_['dI'] = L_.iou_b - L_.iou_a; L_['dn'] = L_.n_pred_b - L_.n_pred_a; L_['tod'] = L_.t / 12
        for nm, (na, nb, ch) in (('kernel footprint', ('nA_kernel', 'nB_kernel', 'chg_kernel')), ('lab-emulated footprint', ('nA_lab', 'nB_lab', 'chg_lab'))):
            o = Fd.rename(columns={na: 'nA', nb: 'nB', ch: 'changed'})[['panel', 'tod', 'nA', 'nB', 'changed']].copy(); o['dn'] = o.nB - o.nA
            f = fpa.forecast(L_, o)
            print(f'FM-PM private of train-only all2 with the ORIGINAL lab pairs (og_np_base/all2, all March days): {nm}: dIoU {f.mean():+.4f} -> LB {0.15 * f.mean():+.5f}')

# ================================================================ 4. train OOF (pvo recipe, train scenario)
f = f'{G.RES}/oof_f0.pkl'
if os.path.exists(f) and 'df' in pd.read_pickle(f):
    Do = pd.read_pickle(f)['df']
    print('\n== 4. TRAIN OOF fold 0 (pvo recipe: F03 folds != 0 + all March days x5; train-scenario prior) ==')
    for s in ('official', 'offlike', 'runstart'):
        A = Do[(Do.set == s) & (Do.arm == 'pvo_base')]; B = Do[(Do.set == s) & (Do.arm == 'pvo_ogf')]
        if len(A):
            bt = lab.paired_bootstrap(A, B); dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
            print(f'{s:9s} n={len(A):4d} pvo {lab.aggregate(A):.4f} pvo-ogf {lab.aggregate(B):.4f} diff {bt["diff"]:+.4f} [{bt["lo"]:+.4f}, {bt["hi"]:+.4f}] panels +{int((dl > 1e-9).sum())}/{len(dl)}')

# ================================================================ verdict under the pre-registered rule
print('\n== VERDICT (pre-registered rule) ==')
for c in CANDS:
    r = T1[(T1.cfg == cfg0) & (T1.set == 'offlike_m') & (T1.B == c) & (T1.A == 'pvo')]
    if not len(r):
        continue
    r = r.iloc[0]; ok1 = (r['diff'] >= 0.005) and (r['lo'] > 0)
    fm = [x for x in FM if x['cand'] == c and x['lab_pairs'] == 's0']
    ok2 = (fm[0]['fm_lb'] >= 0.0005) if fm else None
    print(f"{c:7s}: (1) offlike_m many-day {cfg0} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] -> {'PASS' if ok1 else 'FAIL'} | "
          f"(2) FM-PM private {fm[0]['fm_lb'] if fm else float('nan'):+.5f} -> {('PASS' if ok2 else 'FAIL') if ok2 is not None else 'n/a'} | "
          f"=> {'ADOPT' if ok1 and ok2 is not False else 'DO NOT ADOPT'}")
