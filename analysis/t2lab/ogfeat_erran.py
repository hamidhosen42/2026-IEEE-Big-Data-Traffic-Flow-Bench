"""ogfeat G4: error analysis of the baseline F03 np255 OOF predictions on the official-like windows (step 6 focus)."""
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
pd.set_option('display.width', 250)
rows = []; cells = []
for panel in lab.PANELS:
    rc = lab.rows(panel, 'offlike'); P6, done = lab.load_probs('fc_F03_np255', 'offlike', panel)
    L = rc.L
    for i in np.where(done)[0]:
        truth, ev, ql = rc.truth[i], rc.ev[i], rc.qlast[i]
        pred = lab.decode(P6[i].ravel()).reshape(6, -1)
        T6 = truth[5] & ev[5]; Pp = pred[5] & ev[5]; q = ql & ev[5]
        qc, dql, dqh, dfl, dfh = og.og_edges(ql[None]); qc, dql, dqh, dfl, dfh = qc[0], dql[0], dqh[0], dfl[0], dfh[0]
        dq = np.minimum(dql, dqh); size = np.where(qc, dfl + dfh - 1, 0); edge = np.minimum(dfl, dfh)
        new = T6 & ~ql; clr = ql & ev[5] & ~truth[5]
        u = (T6 | Pp).sum()
        w = rc.win.iloc[i]
        r = dict(panel=panel, d=int(w.d), t=int(w.t), hour=int(w.t) // 12, nT=int(q.sum()), n6=int(T6.sum()), new=int(new.sum()), clr=int(clr.sum()),
                 s6=1.0 if u == 0 else (T6 & Pp).sum() / u, u6=int(u), fn_new=int((new & ~Pp).sum()), fp_old=int((clr & Pp).sum()),
                 fp_new=int((Pp & ~T6 & ~ql).sum()), fn_old=int((T6 & ql & ~Pp).sum()))
        # geometry of the new / cleared cells at step 6
        r['new_adj'] = int((new & (dq <= 2)).sum()); r['new_mid'] = int((new & (dq > 2) & (dq <= 8)).sum()); r['new_far'] = int((new & (dq > 8)).sum())
        r['fn_new_adj'] = int((new & ~Pp & (dq <= 2)).sum()); r['fn_new_mid'] = int((new & ~Pp & (dq > 2) & (dq <= 8)).sum()); r['fn_new_far'] = int((new & ~Pp & (dq > 8)).sum())
        # cleared cells: whole cluster cleared (no cell of the cluster queued at step 6 among observed) vs partial
        cs = np.zeros(L, bool)
        if clr.any():
            start = np.where(qc & ~np.r_[False, qc[:-1]])[0]; end = np.where(qc & ~np.r_[qc[1:], False])[0]
            for a, b in zip(start, end):
                seg = slice(a, b + 1)
                if not (truth[5][seg] & ev[5][seg]).any():
                    cs[seg] = True
        r['clr_whole'] = int((clr & cs).sum()); r['clr_part'] = int((clr & ~cs).sum())
        r['fp_old_whole'] = int((clr & Pp & cs).sum()); r['fp_old_part'] = int((clr & Pp & ~cs).sum())
        r['clr_edge12'] = int((clr & ~cs & (edge <= 2)).sum()); r['clr_inner'] = int((clr & ~cs & (edge > 2)).sum())
        rows.append(r)
D = pd.DataFrame(rows); D.to_pickle(f'{og.RES}/erran_offlike.pkl')
D['type'] = np.where(D.n6 > D.nT * 1.15, 'grow', np.where(D.n6 < D.nT * 0.85, 'shrink', 'stable'))
print('windows', len(D))
print('\n== step-6 by window type (truth size at T+30 vs queue at T) ==')
g = D.groupby('type'); t = pd.DataFrame(dict(n=g.size(), s6=g.s6.mean(), nT=g.nT.mean(), n6=g.n6.mean(), new=g.new.mean(), clr=g.clr.mean(),
     fn_new=g.fn_new.sum() / g.u6.sum(), fp_old=g.fp_old.sum() / g.u6.sum(), fp_new=g.fp_new.sum() / g.u6.sum(), fn_old=g.fn_old.sum() / g.u6.sum()))
print(t.round(3).to_string())
tot = D[['new', 'new_adj', 'new_mid', 'new_far', 'fn_new', 'fn_new_adj', 'fn_new_mid', 'fn_new_far', 'clr', 'clr_whole', 'clr_part', 'clr_edge12', 'clr_inner', 'fp_old', 'fp_old_whole', 'fp_old_part', 'fp_new', 'fn_old', 'u6']].sum()
print('\n== step-6 cell totals over all windows =='); print(tot.to_string())
print(f"\nnew cells: adjacent (<=2 links from the closed queue at T) {tot.new_adj / tot.new:.3f}, 3-8 links {tot.new_mid / tot.new:.3f}, >8 links {tot.new_far / tot.new:.3f}")
print(f"missed share of new cells: adjacent {tot.fn_new_adj / max(tot.new_adj, 1):.3f}, 3-8 {tot.fn_new_mid / max(tot.new_mid, 1):.3f}, >8 {tot.fn_new_far / max(tot.new_far, 1):.3f}")
print(f"cleared cells: whole cluster cleared {tot.clr_whole / tot.clr:.3f}, partial {tot.clr_part / tot.clr:.3f} (edge<=2 {tot.clr_edge12 / tot.clr:.3f}, inner {tot.clr_inner / tot.clr:.3f})")
print(f"kept share of cleared cells: whole-cluster {tot.fp_old_whole / max(tot.clr_whole, 1):.3f}, partial {tot.fp_old_part / max(tot.clr_part, 1):.3f}")
print('\n== step-6 IoU by hour of day (window mean) =='); print(D.groupby('hour').agg(n=('s6', 'size'), s6=('s6', 'mean'), grow=('type', lambda x: (x == 'grow').mean()), shrink=('type', lambda x: (x == 'shrink').mean())).round(3).T.to_string())
