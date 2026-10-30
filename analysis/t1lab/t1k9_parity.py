"""Parity of kaggle_jobs/swap_lrtc_k9.py (tfb-k9, EXP_T1_K9_PREP) with the round-5 lab arm 'r2only+blkVQ@0.15' (t1div5_lrtc.py /
t1div5_report.py posthoc=1). The kernel's step-M / step-B functions and constants are compiled from the kernel FILE via ast (t1lab._compile,
no kernel main code runs) and applied to the LAB's inputs:
  1. members (default all 4 lab panels x 2 folds x 2 months = 16 views): evaluation view of t1div_lrtc.views (released cells minus HO | EB) ->
       kernel lrtc_member + member_values: speed BLK_V_RANKS, flow BLK_Q_RANKS, flow LRTC_RANKS (asserted = the lab pilot ranks) vs
       (a) the lab functions on the same view (t1div_lrtc.norm_view + lrtc, t1div5_lrtc.est_lrtc2r): whole cubes, bit-identical required;
       (b) the saved lab member files <panel>_f<fold>_lrtc.npz (|v, |q) / _lrtc2r.npz (|q) at the evaluation cells (blk + ord; float32).
  2. blend (all 4 panels x 2 folds x 2 months): tfb-k5d stand-in = S4 without step M (mixX on ordinary cells, 0.25 mixX + 0.75 bs on
       blackout cells) -> kernel blend_m (ordinary cells) then blend_b (blackout cells) with
       (a) the saved lab members -> must equal t1div5_report.arms_of(...)['r2only+blkVQ@0.15'] exactly (both channels, both cell kinds);
       (b) (step-1 panels / folds) the kernel's own float64 members of step 1 -> differences = float32 storage of the lab members only.
Usage: t1k9_parity.py [panels=D12_I5_S,D7_I10_E,D7_I210_W,D7_I405_S] [folds=0,1]   (log: analysis/t1lab/logs/t1k9_parity.log)
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], 'posthoc=1']
import t1div5_report as R5  # noqa: E402  (POSTHOC arms incl. r2only+blkVQ@0.15; imports t1div_report with stand=xgb)
sys.argv = [sys.argv[0]] + _ARGV
import t1lab  # noqa: E402
import t1div_lib as L  # noqa: E402
import t1div_lrtc as R4  # noqa: E402
import t1div5_lrtc as D5  # noqa: E402
from t1ramp_exp import load_panel, SPLITS  # noqa: E402

kw = dict(a.split('=') for a in _ARGV if '=' in a)
PANELS1 = kw.get('panels', ','.join(L.PANELS)).split(','); FOLDS1 = [int(f) for f in kw.get('folds', '0,1').split(',')]
KERNEL = os.path.join(ROOT, 'kaggle_jobs', 'swap_lrtc_k9.py')
K = t1lab._compile(KERNEL, ('lr_unfold', 'lr_refold', 'lr_proj', 'lr_init', 'lrtc', 'lrtc_member', 'member_values', 'blend_m', 'blend_b'),
                   ('W_LRTC', 'LRTC_RANKS', 'LRTC_ITERS', 'W_BLK', 'BLK_V_RANKS', 'BLK_Q_RANKS'), dict(np=np))
RK4 = R4.load_ranks(); RK2 = tuple(json.load(open(D5.PILOT5))['chosen']['lrtc2r'])
assert K.BLK_V_RANKS == RK4['v'] == (8, 20, 20) and K.BLK_Q_RANKS == RK4['q'] == (8, 40, 10) and K.LRTC_RANKS == RK2 == (24, 40, 10)
assert K.LRTC_ITERS == R4.ITERS == D5.ITERS == 60 and K.W_LRTC == R5.W_LRTC == 0.35 and K.W_BLK == 0.15   # arm name '...blkVQ@0.15'
ARM = 'r2only+blkVQ@0.15'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


log('kernel', os.path.relpath(KERNEL, ROOT), '| ranks M', K.LRTC_RANKS, 'B v', K.BLK_V_RANKS, 'q', K.BLK_Q_RANKS, '| W', K.W_LRTC, K.W_BLK)


def mx(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if len(a) else 0.0


# ------------------------------------------------------------------ 1. members on the lab's evaluation views
KMEM = {}                                         # (panel, fold, split) -> kernel members at the evaluation cells (blk then ord)
rows1 = []
for panel in PANELS1:
    z, x, st = load_panel(panel); meta = L.load_meta(panel); vf = st.vf.to_numpy()
    for fold in FOLDS1:
        saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
        f4 = np.load(f'{L.RES}/{panel}_f{fold}_lrtc.npz', allow_pickle=True); f5 = np.load(f'{L.RES}/{panel}_f{fold}_lrtc2r.npz', allow_pickle=True)
        for split in SPLITS:
            M, obs, EM, mv, mq = R4.views(panel, fold, split, z, x, meta)
            t = time.time()
            kv, kq, kq2 = K.lrtc_member(mv, K.BLK_V_RANKS), K.lrtc_member(mq, K.BLK_Q_RANKS), K.lrtc_member(mq, K.LRTC_RANKS)
            tk = time.time() - t
            u, f, mu_v, mu_q = R4.norm_view(mv, mq)
            lv, lq, lq2 = R4.lrtc(u, RK4['v'])[0] * mu_v, R4.lrtc(f, RK4['q'])[0] * mu_q, D5.est_lrtc2r(f, mu_q, RK2)
            r = dict(panel=panel, fold=fold, split=split, cube=mv.shape, kernel_s=round(tk, 1),
                     cube_v=mx(kv.ravel(), lv.ravel()), cube_q=mx(kq.ravel(), lq.ravel()), cube_q2=mx(kq2.ravel(), lq2.ravel()))
            parts = []
            for kind, m in L.eval_cells(EM, obs):
                k = f'{split}_{kind}'
                d_, t_, s_ = np.nonzero(m)
                assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(t_, saved[f'{k}|t']) and np.array_equal(s_, saved[f'{k}|s']), k
                ev, eq, eq2 = K.member_values(kv, kq, kq2, (d_, t_, s_), vf[s_])
                sv, sq, sq2 = f4[f'{k}|v'], f4[f'{k}|q'], f5[f'{k}|q']
                r.update({f'{kind}_n': len(d_), f'{kind}_v': mx(ev, sv), f'{kind}_q': mx(eq, sq), f'{kind}_q2': mx(eq2, sq2),
                          f'{kind}_f32_mismatch': int((ev.astype(np.float32) != sv).sum() + (eq.astype(np.float32) != sq).sum()
                                                      + (eq2.astype(np.float32) != sq2).sum())})
                parts.append((ev, eq, eq2))
            KMEM[(panel, fold, split)] = tuple(np.concatenate([p_[i] for p_ in parts]) for i in range(3))
            assert r['cube_v'] == 0.0 and r['cube_q'] == 0.0 and r['cube_q2'] == 0.0, f'kernel LRTC differs from the lab functions: {r}'
            rows1.append(r)
            log(f'1 {panel} f{fold} {split:10s} cube {mv.shape} kernel 3 LRTC {tk:.1f}s | max |kernel - lab fn| cube v {r["cube_v"]:.1e} '
                f'q {r["cube_q"]:.1e} q2 {r["cube_q2"]:.1e} | vs saved lab files (float32): blk n {r["blk_n"]} v {r["blk_v"]:.2e} q {r["blk_q"]:.2e} '
                f'q2 {r["blk_q2"]:.2e} | ord n {r["ord_n"]} v {r["ord_v"]:.2e} q {r["ord_q"]:.2e} q2 {r["ord_q2"]:.2e} | float32 mismatches '
                f'{r["blk_f32_mismatch"] + r["ord_f32_mismatch"]}')

# ------------------------------------------------------------------ 2. blend on the lab's stand-in arrays (= t1div5_report.panel_tables)
rows2 = []
for panel in L.PANELS:
    z, x, st = load_panel(panel); vf = st.vf.to_numpy(); ln_s = st.lanes.to_numpy()
    for fold in (0, 1):
        s0 = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
        for split in SPLITS:
            keys = [f'{split}_blk', f'{split}_ord']
            cat = lambda c: np.concatenate([s0[f'{k}|{c}'] for k in keys])
            s = cat('s'); is_blk = np.arange(len(s)) < len(s0[f'{keys[0]}|d'])
            P = {a: (np.concatenate([s0[f'{a}|{k}|v'] for k in keys]).astype(float),
                     np.concatenate([s0[f'{a}|{k}|q'] for k in keys]).astype(float)) for a in ('base', 'base2', 'bs')}
            for c in ('catb', 'xgb', 'base3', 'lrtc'):
                zc = np.load(f'{L.RES}/{panel}_f{fold}_{c}.npz', allow_pickle=True)
                P[c] = (np.concatenate([zc[f'{k}|v'] for k in keys]).astype(float), np.concatenate([zc[f'{k}|q'] for k in keys]).astype(float))
            for c in R5.MEMBER_FILES:
                zc = np.load(f'{L.RES}/{panel}_f{fold}_{c}.npz', allow_pickle=True)
                P[c] = np.concatenate([zc[f'{k}|{"k" if c == "lrtcK" else "q"}'] for k in keys]).astype(float)
            lab_v, lab_q = R5.arms_of(P, is_blk, vf[s])[ARM]
            mvx, mqx = R5.R.stand_mix(P); bv, bq = P['bs']
            base_v = np.where(is_blk, 0.25 * mvx + 0.75 * bv, mvx); base_q = np.where(is_blk, 0.25 * mqx + 0.75 * bq, mqx)   # tfb-k5d stand-in
            q1 = K.blend_m(base_q, P['lrtc2r'], ~is_blk)
            v2, q2 = K.blend_b(base_v, q1, P['lrtc'][0], P['lrtc'][1], is_blk, vf[s])
            r = dict(panel=panel, fold=fold, split=split, n_blk=int(is_blk.sum()), n_ord=int((~is_blk).sum()),
                     a_v=mx(v2, lab_v), a_q=mx(q2, lab_q), chg_ord_q=int((q2 != base_q)[~is_blk].sum()), chg_ord_v=int((v2 != base_v)[~is_blk].sum()),
                     chg_blk_v=int((v2 != base_v)[is_blk].sum()), chg_blk_q=int((q2 != base_q)[is_blk].sum()))
            assert r['a_v'] == 0.0 and r['a_q'] == 0.0, f'kernel blend differs from the lab arm: {r}'
            assert r['chg_ord_v'] == 0, 'kernel changed ordinary speed'
            if (panel, fold, split) in KMEM:
                ev, eq, eq2 = KMEM[(panel, fold, split)]
                q1k = K.blend_m(base_q, eq2, ~is_blk)
                v2k, q2k = K.blend_b(base_v, q1k, ev, eq, is_blk, vf[s])
                r.update(b_v_blk=mx(v2k[is_blk], lab_v[is_blk]), b_q_blk=mx(q2k[is_blk], lab_q[is_blk]), b_v_ord=mx(v2k[~is_blk], lab_v[~is_blk]),
                         b_q_ord=mx(q2k[~is_blk], lab_q[~is_blk]),
                         b_q_ord_lane=float(np.max(np.abs(q2k - lab_q)[~is_blk] / ln_s[s][~is_blk])))
            rows2.append(r)
            log(f'2 {panel:10s} f{fold} {split:10s} cells blk {r["n_blk"]} ord {r["n_ord"]} | (a) kernel blend on lab members vs lab arm: max |dv| '
                f'{r["a_v"]:.1e} |dq| {r["a_q"]:.1e} | changed vs k5d stand-in: ord q {r["chg_ord_q"]} ord v {r["chg_ord_v"]} blk v {r["chg_blk_v"]} '
                f'blk q {r["chg_blk_q"]}' + (f' | (b) kernel members: blk |dv| {r["b_v_blk"]:.2e} |dq| {r["b_q_blk"]:.2e}, ord |dv| {r["b_v_ord"]:.1e} '
                                            f'|dq| {r["b_q_ord"]:.2e}' if 'b_v_blk' in r else ''))

log('SUMMARY 1 (members; kernel functions vs lab functions on identical inputs = whole cubes):',
    'max |dv|', max(r['cube_v'] for r in rows1), '| max |dq| (8,40,10)', max(r['cube_q'] for r in rows1), '| max |dq| (24,40,10)', max(r['cube_q2'] for r in rows1))
log('SUMMARY 1 (members vs saved lab files, float32 storage): max |dv|', f'{max(max(r["blk_v"], r["ord_v"]) for r in rows1):.2e}',
    '| max |dq| (8,40,10)', f'{max(max(r["blk_q"], r["ord_q"]) for r in rows1):.2e}', '| max |dq| (24,40,10)',
    f'{max(max(r["blk_q2"], r["ord_q2"]) for r in rows1):.2e}', '| float32 mismatches', sum(r['blk_f32_mismatch'] + r['ord_f32_mismatch'] for r in rows1))
log('SUMMARY 2a (kernel blend on lab members vs lab arm, 16 panel-fold-months):', 'max |dv|', max(r['a_v'] for r in rows2), '| max |dq|',
    max(r['a_q'] for r in rows2), '| cells', sum(r['n_blk'] + r['n_ord'] for r in rows2))
rb = [r for r in rows2 if 'b_v_blk' in r]
if rb:
    log('SUMMARY 2b (kernel members + kernel blend vs lab arm): blackout max |dv|', f'{max(r["b_v_blk"] for r in rb):.2e}', 'km/h |dq|',
        f'{max(r["b_q_blk"] for r in rb):.2e}', 'vph | ordinary max |dv|', f'{max(r["b_v_ord"] for r in rb):.1e}', '|dq|', f'{max(r["b_q_ord"] for r in rb):.2e}',
        f'vph ({max(r["b_q_ord_lane"] for r in rb):.2e} per lane)')
log('K9 PARITY PASSED')
