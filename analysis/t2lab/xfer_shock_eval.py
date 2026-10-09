"""X2 evaluation (pre-registered, analysis/TRANSFER_ANALYSIS_2026-10-08.md section 5.0): shock gate on fs4occ-like (base) and all2.
Sets: train offlike / runstart folds 0 and 2 (results/xfer/ogf_diag_f{0,2}_L255.pkl, production inputs), pseudo-March offlike_m /
runstart_m (og_np_base / og_np_all2 fold-0/1 mean), official validation / private (kernel probabilities fs4occ / fs4ogf) -> FM-PM.
Usage: xfer_shock_eval.py <dir with downloaded kernel outputs>"""
import sys, os
import numpy as np, pandas as pd
import lab, ogfeat_lib as og
import xfer_shock_lib as xs
sys.path.insert(0, f'{lab.ROOT}/analysis/t2lab/results/xfer/scripts')
import fpmatch_lib as fpa

DL = sys.argv[1]
RES = f'{lab.ROOT}/analysis/t2lab/results/xfer'
F = 6
ARMS = ('base', 'all2', 'shockA', 'shockB', 'veto10')


def decisions(pa, pb, ql, allowed):
    L = ql.shape[0]
    A = lab.decode(pa.ravel()).reshape(F, L); B = lab.decode(pb.ravel()).reshape(F, L)
    q = np.repeat(ql[None], F, 0); addout = int((B & ~A & ~q).sum())
    return {'base': A, 'all2': B, 'shockA': xs.gate(A, ql, allowed), 'shockB': xs.gate(B, ql, allowed), 'veto10': A if addout > 10 else B}


def score(Dd, truth, ev):
    T = truth & ev; out = {}
    for a, P in Dd.items():
        u = ((T | P) & ev).sum(); out['iou_' + a] = 1.0 if u == 0 else (T & P & ev).sum() / u; out['n_' + a] = int(P.sum())
    return out


rows = []
# ---------------- train folds 0 and 2
for k in (0, 2):
    Z = pd.read_pickle(f'{RES}/ogf_diag_f{k}_L255.pkl')
    for (panel, s), (widx, truth, ev, ql, win) in Z['aux'].items():
        if s not in ('offlike', 'runstart'):
            continue
        P = lab.load_panel(panel)
        d = win.d.to_numpy(); t = win.t.to_numpy(); idx = t[:, None] + np.arange(-lab.H, 0)[None]
        allowed, _ = xs.allowed_region(P['V'][d[:, None], idx], P['Fq'][d[:, None], idx], ql, P['lanes'], P['length'], panel)
        Pa = Z['P'][('base_afocc', panel, s)][1]; Pb = Z['P'][('all2_afocc', panel, s)][1]
        for j in range(len(widx)):
            r = dict(src=f'train_f{k}', set=s, panel=panel, family=panel.rsplit('_', 1)[0], d=int(d[j]), t=int(t[j]))
            r.update(score(decisions(Pa[j].astype(np.float32), Pb[j].astype(np.float32), ql[j], allowed[j]), truth[j], ev[j]))
            rows.append(r)
    lab.drop_panel()
    lab.log(f'train fold {k} done')


# ---------------- pseudo-March
def pm(name, pmk=(0, 1)):
    acc = None
    for kk in pmk:
        z = pd.read_pickle(f'{og.ODIR}/arms/{name}/pm_f{kk}.pkl')
        acc = {key: v.astype(np.float32) for key, v in z.items()} if acc is None else {key: acc[key] + z[key].astype(np.float32) for key in acc}
    return {kk: v / len(pmk) for kk, v in acc.items()}


PA, PB = pm('og_np_base'), pm('og_np_all2')
for panel in lab.PANELS:
    P, Pv = og.val_panel(panel); zz = og.build_pm(panel)
    for s in og.PM_SETS:
        z = zz[s]; win = z['win']
        if not len(win):
            continue
        d = win.d.to_numpy(); t = win.t.to_numpy(); idx = t[:, None] + np.arange(-lab.H, 0)[None]
        allowed, _ = xs.allowed_region(Pv['V'][d[:, None], idx], Pv['Fq'][d[:, None], idx], z['qlast'], P['lanes'], P['length'], panel)
        pa, pb = PA[(panel, s)], PB[(panel, s)]
        for j in range(len(win)):
            r = dict(src='pmarch', set=s, panel=panel, family=panel.rsplit('_', 1)[0], d=int(d[j]), t=int(t[j]))
            r.update(score(decisions(pa[j], pb[j], z['qlast'][j], allowed[j]), z['truth'][j], z['ev'][j]))
            rows.append(r)
    lab.drop_panel()
lab.log('pseudo-March done')
W = pd.DataFrame(rows); W.to_pickle(f'{RES}/shock_lab_windows.pkl')

# ---------------- official footprints (label-free)
occ = np.load(f'{DL}/tfb-t2occ/t2v26occ_probs.npz'); ga = np.load(f'{DL}/tfb-t2ogfa/t2v27a_probs.npz'); gb = np.load(f'{DL}/tfb-t2ogfb/t2v27b_probs.npz')
orow = []
for panel in lab.PANELS:
    P = lab.load_panel(panel); z = np.load(P['npz'], allow_pickle=False)
    for split in ('validation', 'private'):
        k = f'{panel}__{split}__'
        wi = lab.window_index(panel, split); pos = {w: i for i, w in enumerate(wi.window_id)}
        wids = list(occ[k + 'window_id']); ii = np.array([pos[w] for w in wids])
        Vh = z[f'{split}_Vh'].astype(np.float32)[ii]; Fh = z[f'{split}_Fh'].astype(np.float32)[ii]
        ql = occ[k + 'qlast']
        allowed, ws = xs.allowed_region(Vh, Fh, ql, P['lanes'], P['length'], panel)
        pa = (occ[k + 'fs_np'].astype(np.float32) + occ[k + 'fs_id'].astype(np.float32) + occ[k + 'fs_np1'].astype(np.float32) + occ[k + 'fs_id1'].astype(np.float32)) / 4
        pb = (((ga[k + 'fs_np'] + ga[k + 'fs_id']) + gb[k + 'fs_np1']) + gb[k + 'fs_id1']) / 4
        for j, w in enumerate(wids):
            Dd = decisions(pa[j], pb[j], ql[j], allowed[j]); o = pd.Timestamp(wi.forecast_origin.iloc[ii[j]])
            for a, Pd in Dd.items():
                orow.append(dict(split=split, arm=a, window_id=w, panel=panel, tod=o.hour + o.minute / 60, nA=int(Dd['base'].sum()), nB=int(Pd.sum()),
                                 changed=int((Dd['base'] ^ Pd).sum()), w_shock=';'.join(f'{x:.1f}' for x in ws[j])))
    lab.drop_panel()
O = pd.DataFrame(orow); O.to_csv(f'{RES}/shock_official_decisions.csv', index=False)

# ---------------- report
pd.set_option('display.width', 250)
print('== lab gates (aggregated dIoU vs base) ==')
for (src, s), g in W.groupby(['src', 'set']):
    agg = lambda c: g.groupby(['family', 'panel'])[c].mean().groupby(level=0).mean().mean()
    b = agg('iou_base')
    pp = lambda a: int(((g.groupby('panel')['iou_' + a].mean() - g.groupby('panel').iou_base.mean()) > 0).sum())
    print(f'{src:9s} {s:10s} n {len(g):5d} | ' + ' | '.join(f'{a} {agg("iou_" + a) - b:+.4f} ({pp(a)}/8)' for a in ARMS[1:]) +
          f' | positives shockA {g.n_shockA.sum() / g.n_base.sum() - 1:+.3%}')
print('== FM-PM (pre-registered: panel-exact, +-2 h, all-cell counts, pseudo-March offlike_m + runstart_m, baseline = base/fs4occ) ==')
Lpm = W[W.src == 'pmarch']
res = []
for a in ARMS[1:]:
    L_ = Lpm[['set', 'panel', 'd', 't']].copy(); L_['dI'] = Lpm['iou_' + a] - Lpm.iou_base; L_['dn'] = Lpm['n_' + a] - Lpm.n_base; L_['tod'] = Lpm.t / 12
    r = dict(arm=a)
    for split in ('validation', 'private'):
        off = O[(O.split == split) & (O.arm == a)].copy(); off['dn'] = off.nB - off.nA
        f = fpa.forecast(L_, off); r[f'{split}_win'] = int((off.changed > 0).sum()); r[f'{split}_cells'] = int(off.changed.sum())
        r[f'{split}_fm_iou'] = f.mean(); r[f'{split}_fm_lb'] = 0.15 * f.mean()
    r['PASS'] = bool(r['validation_fm_lb'] >= 0.0005 and r['private_fm_lb'] >= 0.0005)
    res.append(r)
R = pd.DataFrame(res); print(R.round(5).to_string(index=False)); R.to_csv(f'{RES}/shock_fm.csv', index=False)
