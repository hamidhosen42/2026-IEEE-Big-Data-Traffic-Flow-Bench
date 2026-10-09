"""Analyse results/xfer/ogf_diag_f<k>_L<leaves>.pkl:
 (A) lab cell footprint of all2 vs base (to compare with the March footprint of tfb-h4 vs tfb-h1) + precision of the changed cells
 (B) 'who is right when they disagree': realised dIoU vs each model's self-assessed expected-IoU change
 (C) ongoing loss decomposition of the production-like base model (base_afocc): error types, steps, growth / shrink / stable,
     size vs location (window size oracle, per-step size oracle), panels."""
import sys
sys.path.insert(0, '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/analysis/t2lab')
import numpy as np, pandas as pd
import lab
F = 6
f = sys.argv[1]
Z = pd.read_pickle(f); D = Z['df']; PS = Z['P']; AUX = Z['aux']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)


def eiou(p, d, ev):
    p = p * ev; s = (p * d).sum(); den = d[ev].sum() + (p * ~d).sum()
    return s / den if den > 0 else 1.0


rows = []; cellrows = []
for (panel, s), (widx, truth, ev, ql, win) in AUX.items():
    wa, Pa = PS[('base_afocc', panel, s)]; wb, Pb = PS[('all2_afocc', panel, s)]
    L = ql.shape[1]
    for j in range(len(widx)):
        pa = Pa[j].astype(np.float32); pb = Pb[j].astype(np.float32)
        A = lab.decode(pa.ravel()).reshape(F, L); B = lab.decode(pb.ravel()).reshape(F, L)
        T = truth[j] & ev[j]; E = ev[j]; q = np.repeat(ql[j][None], F, 0)
        iouA = ((T & A & E).sum() / max(((T | A) & E).sum(), 1)) if ((T | A) & E).any() else 1.0
        iouB = ((T & B & E).sum() / max(((T | B) & E).sum(), 1)) if ((T | B) & E).any() else 1.0
        add = B & ~A; rem = A & ~B
        # truth-class of the window: size at T (visible queue) vs true size at h6
        nT = ql[j].sum(); n6 = T[5].sum(); ev6 = E[5].mean()
        # oracles on the base model: window size oracle (top-n_true over all eligible cells), per-step size oracle
        pe = np.where(E, pa, -1.0).ravel(); ntrue = T.sum()
        if ntrue:
            o = np.argsort(-pe)[:ntrue]; Po = np.zeros(F * L, bool); Po[o] = True; Po = Po.reshape(F, L)
            ior = (T & Po).sum() / max((T | Po).sum(), 1)
            Ps = np.zeros((F, L), bool)
            for h in range(F):
                nh = T[h].sum()
                if nh:
                    oo = np.argsort(-np.where(E[h], pa[h], -1.0))[:nh]; Ps[h, oo] = True
            iors = (T & Ps).sum() / max((T | Ps).sum(), 1)
        else:
            ior = iors = 1.0 if not (A & E).any() else 0.0
        rows.append(dict(set=s, panel=panel, family=panel.rsplit('_', 1)[0], d=int(win.d[j]), t=int(win.t[j]), runstart=bool(win.runstart[j]) if 'runstart' in win else np.nan,
                         iouA=iouA, iouB=iouB, dI=iouB - iouA, nA=int((A & E).sum()), nB=int((B & E).sum()), ntrue=int(ntrue), nT=int(nT), n6=int(n6),
                         nA_all=int(A.sum()), nB_all=int(B.sum()), ncell=F * L, changed=int((A ^ B).sum()),
                         add_in=int((add & q).sum()), add_out=int((add & ~q).sum()), rem_in=int((rem & q).sum()), rem_out=int((rem & ~q).sum()),
                         add_ev=int((add & E).sum()), add_true=int((add & T).sum()), rem_ev=int((rem & E).sum()), rem_true=int((rem & T).sum()),
                         pB_add=float(pb[add].mean()) if add.any() else np.nan, pA_add=float(pa[add].mean()) if add.any() else np.nan,
                         pB_rem=float(pb[rem].mean()) if rem.any() else np.nan, pA_rem=float(pa[rem].mean()) if rem.any() else np.nan,
                         eA_A=eiou(pa, A, E), eA_B=eiou(pa, B, E), eB_B=eiou(pb, B, E), eB_A=eiou(pb, A, E),
                         ior_size=ior, ior_step=iors,
                         **{f'err{h + 1}': int(((T[h] ^ A[h]) & E[h]).sum()) for h in range(F)}, union=int(((T | A) & E).sum()),
                         fn_new=int((T & ~A & ~q).sum()), fn_old=int((T & ~A & q).sum()), fp_old=int((A & ~T & q & E).sum()), fp_new=int((A & ~T & ~q & E).sum())))
W = pd.DataFrame(rows)
W.to_pickle(f.replace('.pkl', '_windows.pkl'))


def wagg(df, col):
    pm = df.groupby(['family', 'panel'])[col].mean()
    return float(pm.groupby(level=0).mean().mean())


print('=== (A) footprint all2 vs base (cells over all 6 x L cells), March tfb-h4 vs h1: 0.70 % changed, added 197 / removed 107, '
      'added out-of-qT 159 / in 38, removed in-qT 85 / out 22, positives +1.7 %, windows changed 28/40')
for s, g in W.groupby('set'):
    print(f'{s:9s} n {len(g):5d} | changed {g.changed.sum() / g.ncell.sum() * 100:.2f}% | windows changed {np.mean(g.changed > 0):.2f} | added {g.add_in.sum() + g.add_out.sum()} '
          f'(out {g.add_out.sum()}, in {g.add_in.sum()}) removed {g.rem_in.sum() + g.rem_out.sum()} (in {g.rem_in.sum()}, out {g.rem_out.sum()}) | positives {(g.nB_all.sum() / g.nA_all.sum() - 1) * 100:+.2f}% | '
          f'precision added {g.add_true.sum() / max(g.add_ev.sum(), 1):.3f} (mean pB {np.nanmean(g.pB_add):.3f}, pA {np.nanmean(g.pA_add):.3f}) | removed-and-truly-queued {g.rem_true.sum() / max(g.rem_ev.sum(), 1):.3f} '
          f'(mean pA {np.nanmean(g.pA_rem):.3f}, pB {np.nanmean(g.pB_rem):.3f}) | dIoU agg {wagg(g, "dI"):+.4f}')
    g = g.assign(dn=g.nB_all - g.nA_all)
    for lo, hi in ((10, 999), (3, 9), (-9, -3), (-999, -10)):
        gg = g[(g.dn >= lo) & (g.dn <= hi)]
        if len(gg):
            print(f'     dn in [{lo},{hi}]: n {len(gg)} mean dI {gg.dI.mean():+.4f} P(dI<0) {(gg.dI < 0).mean():.2f} precision added {gg.add_true.sum() / max(gg.add_ev.sum(), 1):.3f}')
print('\n=== (B) who is right when they disagree (changed windows): realised dI vs expected changes under each model')
for s, g in W[W.changed > 0].groupby('set'):
    xa = g.eA_B - g.eA_A; xb = g.eB_B - g.eB_A
    # realised dI = lam * xb + (1 - lam) * xa  -> least squares lam
    lam = float(((g.dI - xa) * (xb - xa)).sum() / ((xb - xa) ** 2).sum())
    print(f'{s:9s} n {len(g)} | mean dI {g.dI.mean():+.4f} | E_base[dI] {xa.mean():+.4f} | E_all2[dI] {xb.mean():+.4f} | lambda (weight on all2 view) {lam:.2f}')
print('March tfb-h4: E_base[dI] -0.0330, E_all2[dI] +0.0366 (40 windows, unchanged = 0), realised (LB) -0.0081 -> lambda_March = '
      f'{(-0.0081 + 0.0330) / (0.0366 + 0.0330):.2f}')
print('\n=== (C) loss decomposition of the production-like base model (base_afocc = F03 + all-finite + occupancy, np, 255 leaves, fold 0)')
for s, g in W.groupby('set'):
    g = g.copy(); un = g.union.clip(lower=1)
    g['loss'] = 1 - g.iouA
    for c in ('fn_new', 'fn_old', 'fp_old', 'fp_new'):
        g['l_' + c] = g[c] / un
    for h in range(1, F + 1):
        g[f'l_h{h}'] = g[f'err{h}'] / un
    r = np.where(np.maximum(g.ntrue, g.nA) == 0, 1.0, np.minimum(g.ntrue, g.nA) / np.maximum(np.maximum(g.ntrue, g.nA), 1))
    g['l_size'] = 1 - r; g['l_loc'] = r - g.iouA
    g['cls'] = np.where(g.n6 > 1.25 * g.nT + 1, 'grow', np.where(g.n6 < 0.75 * g.nT - 1, 'shrink', 'stable'))
    tot = wagg(g, 'loss')
    print(f'--- {s}: n {len(g)} | IoU {1 - tot:.4f} | loss {tot:.4f} | size-oracle IoU {wagg(g, "ior_size"):.4f} (+{wagg(g, "ior_size") - (1 - tot):.4f}) | '
          f'per-step size oracle {wagg(g, "ior_step"):.4f} (+{wagg(g, "ior_step") - (1 - tot):.4f})')
    print('   error types (share of loss): ' + ', '.join(f'{c} {wagg(g, "l_" + c) / tot:.2f}' for c in ('fn_new', 'fn_old', 'fp_old', 'fp_new')))
    print('   by step h (share of loss):   ' + ', '.join(f'h{h} {wagg(g, f"l_h{h}") / tot:.2f}' for h in range(1, F + 1)))
    print(f'   size vs location: size {wagg(g, "l_size") / tot:.2f}, location {wagg(g, "l_loc") / tot:.2f}')
    for c, gg in g.groupby('cls'):
        wsh = len(gg) / len(g)
        print(f'   class {c:6s}: share of windows {wsh:.2f} | mean IoU {gg.iouA.mean():.4f} | share of loss (plain) {(gg.loss.sum() / g.loss.sum()):.2f} | '
              f'size-oracle gain {(gg.ior_size - gg.iouA).mean():+.4f} | dI(all2) {gg.dI.mean():+.4f}')
    pp = g.groupby('panel').agg(n=('loss', 'size'), iou=('iouA', 'mean'), loss=('loss', 'mean'), size_or=('ior_size', 'mean'))
    pp['loss_share'] = pp.loss / pp.loss.sum()
    print(pp.round(3).to_string())
