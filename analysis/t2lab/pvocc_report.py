"""pvocc REPORT (light: decode + scoring only, no training).  Design and the PRE-REGISTERED rule: pvocc_lib.py (asserted unchanged).
Inputs: the cached pvogf probabilities results/pvogf/probs/{MD_s0, MD_s1, FULL_s0}.pkl (arms base = pvo, ogf = pvogf; read only), the
pvocc probabilities results/pvocc/probs/{MD_s0, MD_s1, FULL_s0}.pkl (arms baseO = pvoO, ogfO = pvogfO), the pvogf caches (truth / ev /
qlast of the pseudo-March windows, private window ids / origins), results/pvogf/oof_f0.pkl + results/pvocc/oof_f0.pkl (train OOF, info).
Sections
  0 INTEGRITY (i): the cached pvogf numbers recomputed from the cached probabilities (pm_table.csv, fm_private.csv); (iv) from parity_iv.json
  1 many-day pseudo-March (days >= 15): O = pvogfO, vO = pvogfvO vs pvogf (binding: seed-average offlike_m), vs pvogfv, seeds alone
  2 FM-PM private: lab-emulated April decisions (FULL_s0) on the 40 official private ongoing windows, priced with the seed-0 lab pairs
  3 train OOF fold 0 (info)
  4 VERDICT: binding rule + the pre-registered March-only secondary verdict
Usage: pvocc_report.py [b0=cached|new]"""
import json, os, sys
from importlib import util
import numpy as np
import pandas as pd
import lab
import pvogf_lib as G
import pvocc_lib as V

kw = dict(a.split('=', 1) for a in sys.argv[1:])
B0 = kw.get('b0', 'cached')
F, ONGOING = G.F, G.ONGOING
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
_s = util.spec_from_file_location('fpa', f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts/fpmatch_lib.py')
fpa = util.module_from_spec(_s); _s.loader.exec_module(fpa)
print(V.check_prereg()); print(f'(pre-registered text sha1 {V.PREREG_SHA1}: unchanged)\n')


def load(d, tag):
    f = f'{d}/probs/{tag}.pkl'
    return {k: v.astype(np.float32) for k, v in pd.read_pickle(f).items()} if os.path.exists(f) else {}


M_OLD = {f'{a}_{m}' for a in ('base', 'ogf') for m in ('np', 'id', 'full')}
M_NEW = {f'{a}_{m}' for a in ('baseO', 'ogfO') for m in ('np', 'id', 'full')}


SELFTEST = kw.get('selftest', '0') == '1'      # code check only: the O arms are aliased to the cached arms (every delta must be 0); writes to results/pvocc/selftest


def merged(tag):
    old = load(G.RES, tag); new = load(V.RES, tag)
    if SELFTEST:
        new = {(s_, p_, m_.replace('base', 'baseO').replace('ogf', 'ogfO')): v for (s_, p_, m_), v in old.items() if m_ in M_OLD}
    if B0 == 'cached':
        new = {k: v for k, v in new.items() if k[2] in M_NEW}            # the integrity (iv) refits never replace the cached baseline
    out = dict(old); out.update(new)
    have = {k[2] for k in out}
    return out if (M_OLD | M_NEW) <= have else None, have


MD = {}
for s in (0, 1):
    P, have = merged(f'MD_s{s}')
    print(f'MD_s{s}: members', sorted(have), '| complete', P is not None)
    if P is not None:
        MD[s] = P
FULL, haveF = merged('FULL_s0'); print('FULL_s0: members', sorted(haveF), '| complete', FULL is not None)
cfgs = {f's{s}': [s] for s in MD}
if len(MD) == 2:
    cfgs['avg'] = [0, 1]
CANDS = ('pvogfO', 'pvogfvO')
OUTR = f'{V.RES}/selftest' if SELFTEST else V.RES
os.makedirs(OUTR, exist_ok=True)


def mixP(P, s, p, arm, seeds):
    """seed-averaged 0.25 np + 0.25 id + 0.5 full (same float operation order as pvogf_report.py)."""
    return sum(V.mix(P[x], s, p, arm) for x in seeds) / len(seeds)


# ================================================================ per-window frames (pseudo-March, days >= 15)
rows = []
for p in lab.PANELS:
    fam = p.rsplit('_', 1)[0]
    for s in G.PM_SETS:
        z = np.load(f'{G.CACHE}/ev_{p}__{s}.npz'); truth, ev, ql = z['truth'], z['ev'], z['qlast']
        for cfg, seeds in cfgs.items():
            dec = dict(pvo=G.decode6(mixP(MD, s, p, 'base', seeds)), pvogf=G.decode6(mixP(MD, s, p, 'ogf', seeds)),
                       pvoO=G.decode6(mixP(MD, s, p, 'baseO', seeds)), pvogfO=G.decode6(mixP(MD, s, p, 'ogfO', seeds)))
            dec['pvogfv'], addB = G.veto(dec['pvo'], dec['pvogf'], ql)
            dec['pvogfvO'], addO = G.veto(dec['pvoO'], dec['pvogfO'], ql)
            for arm, d_ in dec.items():
                rows.append(pd.DataFrame(dict(cfg=cfg, arm=arm, set=s, panel=p, family=fam, cond=ONGOING, d=z['d'], t=z['t'], addB=addB, addO=addO,
                                              chg=(d_ ^ dec['pvogf']).sum((1, 2)), **G.score_dec(d_, truth, ev, ql))))
D = pd.concat(rows, ignore_index=True); D.to_pickle(f'{OUTR}/pm_frames.pkl')


def cmp(cfg, s, a, b):
    g = D[(D.cfg == cfg) & (D.set == s)]
    A = g[g.arm == a]; B = g[g.arm == b]
    bt = lab.paired_bootstrap(A, B); dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
    return dict(cfg=cfg, set=s, B=b, A=a, n=bt['n_windows'], iou_A=lab.aggregate(A), iou_B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'],
                p_pos=bt['p_pos'], panels_pos=int((dl > 0).sum()), panels_neg=int((dl < 0).sum()),
                win_changed=int((B.chg.to_numpy() > 0).sum()) if a == 'pvogf' else -1,
                dn_all=int(B.n_all.sum() - A.n_all.sum()), per_panel={k: round(v, 4) for k, v in dl.items()})


def fmt(r):
    return (f"{r['cfg']:3s} {r['set']:10s} {r['B']:8s} - {r['A']:7s} n={r['n']:4d} {r['iou_B']:.4f} vs {r['iou_A']:.4f} {r['diff']:+.4f} "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}] p_pos {r['p_pos']:.3f} panels +{r['panels_pos']}/-{r['panels_neg']} | windows changed {r['win_changed']} dcells {r['dn_all']:+d}")


# ================================================================ 0. INTEGRITY (i): cached pvogf numbers reproduced
print('== 0. INTEGRITY ==')
ref = pd.read_csv(f'{G.RES}/pm_table.csv'); ok_i = True
for cfg in cfgs:
    for s in G.PM_SETS:
        for b in ('pvogf', 'pvogfv'):
            r = cmp(cfg, s, 'pvo', b); q = ref[(ref.cfg == cfg) & (ref.set == s) & (ref.B == b) & (ref.A == 'pvo')].iloc[0]
            ok = all(abs(r[k] - q[k]) < 1e-12 for k in ('diff', 'lo', 'hi', 'iou_A', 'iou_B')) and r['n'] == q['n']
            ok_i &= ok
            if cfg == 'avg' and s == 'offlike_m':
                print(f"(i) {fmt(r)} | cached pm_table {q['diff']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] -> {'IDENTICAL' if ok else 'DIFFERENT'}")
print(f'(i) all {len(cfgs) * len(G.PM_SETS) * 2} cached pseudo-March comparisons reproduced exactly: {ok_i}')
pf = f'{V.RES}/parity_iv.json'
if os.path.exists(pf):
    for r in json.load(open(pf)):
        print('(iv)', r)

# ================================================================ 1. many-day pseudo-March
T1 = []
print('\n== 1. MANY-DAY pseudo-March (March days >= 15; training March rows days < 15; lab 255 leaves / 500 trees; members seed-averaged in avg) ==')
for cfg in [c for c in ('avg', 's0', 's1') if c in cfgs]:
    for s in G.PM_SETS:
        for a, b in [('pvogf', c) for c in CANDS] + [('pvogfv', c) for c in CANDS] + [('pvo', 'pvoO'), ('pvo', 'pvogf'), ('pvogf', 'pvogfv')]:
            r = cmp(cfg, s, a, b); T1.append(r); print(fmt(r))
T1 = pd.DataFrame(T1); T1.to_csv(f'{OUTR}/pm_table.csv', index=False)
cfg0 = 'avg' if 'avg' in cfgs else 's0'
print('\nper-panel deltas vs pvogf (seed average):')
for s in G.PM_SETS:
    for c in CANDS:
        r = T1[(T1.cfg == cfg0) & (T1.set == s) & (T1.B == c) & (T1.A == 'pvogf')].iloc[0]
        print(f'  {s:10s} {c:8s}', r['per_panel'])
g = D[D.cfg == cfg0]
A_ = g[g.arm == 'pvogf'].set_index(['set', 'panel', 'd', 't']); B_ = g[g.arm == 'pvogfO'].set_index(['set', 'panel', 'd', 't'])
dn = B_.n_all - A_.n_all; dI = B_.iou - A_.iou
print('\npvogfO vs pvogf by size change (all-cell dn), seed average, both sets:')
for lo, hi, lb in ((-999, -10, 'dn <= -10'), (-9, -1, '-9..-1'), (0, 0, '0'), (1, 9, '1..9'), (10, 999, 'dn >= 10')):
    m = (dn >= lo) & (dn <= hi)
    print(f'  {lb:10s} windows {int(m.sum()):4d} mean dIoU {float(dI[m].mean()) if m.any() else float("nan"):+.4f}')
for s in G.PM_SETS:
    gs = g[(g.arm == 'pvogfO') & (g.set == s)]
    print(f'  {s}: windows where pvogfO adds > 10 cells outside qlast vs pvoO (vetoed in vO): {int((gs.addO > 10).sum())} of {len(gs)}'
          f' | pvogfv veto fired {int((gs.addB > 10).sum())}')

# ================================================================ 2. FM-PM private
FM = []; dec_priv = {}
if FULL is not None:
    off = {c: [] for c in CANDS + ('pvogf_vs_pvo',)}
    for p in lab.PANELS:
        z = np.load(f'{G.CACHE}/priv_{p}.npz'); og = z['cond'] == ONGOING; ql = z['qlast'][og]
        A = G.decode6(mixP({0: FULL}, 'private', p, 'ogf', [0]))[og]; A0 = G.decode6(mixP({0: FULL}, 'private', p, 'base', [0]))[og]
        AO = G.decode6(mixP({0: FULL}, 'private', p, 'baseO', [0]))[og]; B = G.decode6(mixP({0: FULL}, 'private', p, 'ogfO', [0]))[og]
        Bv, addO = G.veto(AO, B, ql); Av, addB = G.veto(A0, A, ql)
        dec_priv[p] = dict(pvogf=A, pvo=A0, pvogfv=Av, pvoO=AO, pvogfO=B, pvogfvO=Bv, qlast=ql, wid=z['window_id'][og], tod=z['tod'][og], addO=addO, addB=addB)
        for c, base_, dec in (('pvogfO', A, B), ('pvogfvO', A, Bv), ('pvogf_vs_pvo', A0, A)):
            for i in range(len(A)):
                off[c].append(dict(window_id=z['window_id'][og][i], panel=p, tod=float(z['tod'][og][i]), nA=int(base_[i].sum()), nB=int(dec[i].sum()),
                                   changed=int((base_[i] ^ dec[i]).sum()), addout=int(((dec[i] & ~base_[i]) & ~np.repeat(ql[i][None], F, 0)).sum())))
    pd.to_pickle(dec_priv, f'{OUTR}/private_decisions_emulated.pkl')
    vetoA = sum(int((d_['addB'] > 10).sum()) for d_ in dec_priv.values()); vetoO = sum(int((d_['addO'] > 10).sum()) for d_ in dec_priv.values())
    print(f'\n== 2. FM-PM PRIVATE (lab-emulated April decisions: FULL_s0 = F03 + all 31 March days; 40 official private ongoing windows) ==')
    print(f'veto fired on April: pvogfv (pvogf vs pvo) {vetoA}/40 | pvogfvO (pvogfO vs pvoO) {vetoO}/40')
    for c in CANDS + ('pvogf_vs_pvo',):
        o = pd.DataFrame(off[c]); o['dn'] = o.nB - o.nA
        a_arm, b_arm = ('pvo', 'pvogf') if c == 'pvogf_vs_pvo' else ('pvogf', c)
        for lab_cfg, pooled in (('s0', False), ('s1', False), ('avg', False), ('s0', True)):
            if lab_cfg not in cfgs:
                continue
            g = D[D.cfg == lab_cfg]
            L = g[g.arm == a_arm][['set', 'panel', 'd', 't', 'iou', 'n_all']].merge(g[g.arm == b_arm][['set', 'panel', 'd', 't', 'iou', 'n_all']],
                                                                                  on=['set', 'panel', 'd', 't'], suffixes=('_a', '_b'))
            L['dI'] = L.iou_b - L.iou_a; L['dn'] = L.n_all_b - L.n_all_a; L['tod'] = L.t / 12
            f = fpa.forecast(L, o, panel_exact=not pooled)
            r = dict(cand=c, lab_pairs=lab_cfg + (' pooled' if pooled else ''), windows_changed=int((o.changed > 0).sum()), cells_changed=int(o.changed.sum()),
                     pos_base=int(o.nA.sum()), pos_cand=int(o.nB.sum()), added_outside_qlast=int(o.addout.sum()), fm_dIoU=float(f.mean()), fm_lb=0.15 * float(f.mean()))
            FM.append(r)
            print(f"{c:12s} lab pairs {r['lab_pairs']:10s} | windows changed {r['windows_changed']:2d}/40 cells {r['cells_changed']:4d} | positives {a_arm} {r['pos_base']} -> {r['pos_cand']}"
                  f" (added outside qlast {r['added_outside_qlast']}) | FM-PM dIoU {r['fm_dIoU']:+.4f} -> LB-equiv {r['fm_lb']:+.5f}")
            if lab_cfg == 's0' and not pooled:
                o.assign(fm=f).to_csv(f'{OUTR}/fm_private_{c}.csv', index=False)
    FMd = pd.DataFrame(FM); FMd.to_csv(f'{OUTR}/fm_private.csv', index=False)
    q = pd.read_csv(f'{G.RES}/fm_private.csv'); q = q[(q.cand == 'pvogf') & (q.lab_pairs == 's0')].iloc[0]
    r = FMd[(FMd.cand == 'pvogf_vs_pvo') & (FMd.lab_pairs == 's0')].iloc[0]
    ok_fm = abs(r.fm_lb - q.fm_lb) < 1e-12 and int(r.windows_changed) == int(q.windows_changed)
    print(f'(i) FM-PM private pvogf vs pvo (s0) recomputed {r.fm_lb:+.5f} vs cached {q.fm_lb:+.5f} -> {"IDENTICAL" if ok_fm else "DIFFERENT"}')
    ok_i &= ok_fm

# ================================================================ 3. train OOF fold 0 (info)
fo, fn = f'{G.RES}/oof_f0.pkl', f'{V.RES}/oof_f0.pkl'
OOF = []
if os.path.exists(fo) and os.path.exists(fn):
    Po = {k: v.astype(np.float32) for k, v in pd.read_pickle(fo)['P'].items()}
    Pn = {k: v.astype(np.float32) for k, v in pd.read_pickle(fn).items() if k[2] in M_NEW}
    Pa = dict(Po); Pa.update(Pn)
    if (M_OLD | M_NEW) <= {k[2] for k in Pa}:
        rr = []
        for p in lab.PANELS:
            fam = p.rsplit('_', 1)[0]
            for s in ('official', 'offlike', 'runstart'):
                f = f'{G.CACHE}/oof0/ev_{p}__{s}.npz'
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                dec = {arm: G.decode6(V.mix(Pa, s, p, arm)) for arm in ('base', 'ogf', 'baseO', 'ogfO')}
                dec['ogfv'], _ = G.veto(dec['base'], dec['ogf'], z['qlast']); dec['ogfvO'], _ = G.veto(dec['baseO'], dec['ogfO'], z['qlast'])
                for arm, d_ in dec.items():
                    rr.append(pd.DataFrame(dict(arm=arm, set=s, panel=p, family=fam, cond=ONGOING, d=z['d'], t=z['t'], **G.score_dec(d_, z['truth'], z['ev'], z['qlast']))))
        Do = pd.concat(rr, ignore_index=True)
        print('\n== 3. TRAIN OOF fold 0 (pvo recipe: F03 folds != 0 + all March days x5; train-scenario prior; info only) ==')
        for s in ('official', 'offlike', 'runstart'):
            for a, b, nm in (('ogf', 'ogfO', 'pvogfO - pvogf'), ('ogf', 'ogfvO', 'pvogfvO - pvogf'), ('ogfv', 'ogfvO', 'pvogfvO - pvogfv'), ('base', 'baseO', 'pvoO - pvo')):
                A = Do[(Do.set == s) & (Do.arm == a)]; B = Do[(Do.set == s) & (Do.arm == b)]
                bt = lab.paired_bootstrap(A, B); dl = B.groupby('panel').iou.mean() - A.groupby('panel').iou.mean()
                OOF.append(dict(set=s, cmp=nm, n=len(A), A=lab.aggregate(A), B=lab.aggregate(B), diff=bt['diff'], lo=bt['lo'], hi=bt['hi'], panels_pos=int((dl > 0).sum())))
                print(f"{s:9s} {nm:17s} n={len(A):4d} {lab.aggregate(B):.4f} vs {lab.aggregate(A):.4f} diff {bt['diff']:+.4f} [{bt['lo']:+.4f}, {bt['hi']:+.4f}] panels +{int((dl > 0).sum())}/{len(dl)}")
        pd.DataFrame(OOF).to_csv(f'{OUTR}/oof_table.csv', index=False)

# ================================================================ 4. VERDICT
print('\n== 4. VERDICT (pre-registered rule; integrity (i) ' + ('PASSED' if ok_i else 'FAILED') + ') ==')
res = {}
for c in CANDS:
    r = T1[(T1.cfg == cfg0) & (T1.set == 'offlike_m') & (T1.B == c) & (T1.A == 'pvogf')].iloc[0]
    ok1 = bool(cfg0 == 'avg' and r['diff'] >= 0.003 and r['lo'] > 0 and r['panels_pos'] >= 6)
    fm = [x for x in FM if x['cand'] == c and x['lab_pairs'] == 's0']
    ok2 = bool(fm and fm[0]['fm_lb'] >= 0.0005)
    res[c] = dict(d1=float(r['diff']), lo1=float(r['lo']), hi1=float(r['hi']), panels=int(r['panels_pos']), ok1=ok1,
                  fm=float(fm[0]['fm_lb']) if fm else float('nan'), ok2=ok2, adopt=ok1 and ok2)
    print(f"{c:8s}: (1) offlike_m many-day {cfg0} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] panels +{r['panels_pos']}/8 -> {'PASS' if ok1 else 'FAIL'}"
          f" | (2) FM-PM private {res[c]['fm']:+.5f} -> {'PASS' if ok2 else 'FAIL'} | => {'ADOPT-ELIGIBLE' if ok1 and ok2 else 'NOT ADOPTED'}")
el = [c for c in CANDS if res[c]['adopt']]
if len(el) == 2:
    pick = 'pvogfO' if abs(res['pvogfO']['fm'] - res['pvogfvO']['fm']) < 1e-6 else max(el, key=lambda c: res[c]['fm'])
elif el:
    pick = el[0]
else:
    pick = None
el2 = [c for c in CANDS if res[c]['ok1']]
pick2 = max(el2, key=lambda c: res[c]['d1']) if el2 else None
print(f'BINDING VERDICT: {"ADOPT " + pick if pick else "DO NOT ADOPT (pick A keeps pvogfv)"}')
print(f'SECONDARY March-only verdict (not binding): {"ADOPT " + pick2 if pick2 else "DO NOT ADOPT"}'
      + ('' if pick == pick2 else '  <-- DIFFERS from the binding verdict: the user decides'))
json.dump(dict(integrity_i=bool(ok_i), cands=res, binding=pick, secondary=pick2, b0=B0), open(f'{OUTR}/verdict.json', 'w'), indent=1)
