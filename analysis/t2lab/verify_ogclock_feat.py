"""VERIFY ogclock: (1) plain-loop re-implementation of the 16 ck features from the documented semantics, built from inputs this script
assembles itself (thinned train cube only for slots < T-12, raw history T-12..T-1, production slot-T draw at T; never anything later),
compared with the track's cached values (train sets and pseudo-March sets); (2) perturbation test of the LAB and of the KERNEL functions
(data at slots >= T-12 of the day cube / after T must not move any feature); (3) kernel block identical to the lab block.
Light script (single thread).  Usage: verify_ogclock_feat.py [PANELS] [N_PER_SET]"""
import ast, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
import lab

PANELS_CHK = sys.argv[1].split(',') if len(sys.argv) > 1 else ['D7_I10_W', 'D7_I405_N', 'D12_I5_S']
NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 40
CKN = ['ck_run', 'ck_first', 'ck_clear', 'ck_frac', 'ck_minr', 'ck_nep', 'cc_first', 'cc_peak', 'cc_peak_sh', 'cc_since_peak', 'cc_rel',
       'cc_d6', 'cc_d12', 'cc_d24', 'cc_age', 'cc_area']
FED = f'{lab.CACHE}/ogclock'
H = 12
KFILE = f'{lab.ROOT}/kaggle_jobs/t2_ogclock_v25.py'
LFILE = f'{lab.ROOT}/analysis/t2lab/ogclock_lib.py'


# ---------------------------------------------------------------- independent pieces
def vis_days_indep():
    out = []
    for panel in lab.PANELS:
        z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False)
        for split in ('validation', 'private'):
            Vm = z[f'{split}_Vm']; Em = z[f'{split}_Em']; Vis = np.isfinite(Vm)
            blank = (Em.sum(2) >= 5) & (Vis.sum(2) == 0)
            for d in range(len(Vm)):
                nb = (~blank[d])[:, None] & Em[d]
                out.append(float(Vis[d][nb].mean()))
    return np.array(out)


def thin_indep(pi, D, L, vd):
    rates = np.random.default_rng(7000 + pi).choice(vd, D, replace=True)
    return np.stack([np.random.default_rng(10 ** 6 * (pi + 1) + d).random((288, L)) < rates[d] for d in range(D)])


def feats_loop(x, T):
    """x [T+1, L] speed ratio of slots 0..T (NaN = not visible). Plain loops, documented semantics."""
    S1, L = x.shape
    assert S1 == T + 1
    out = {c: np.full(L, np.nan) for c in CKN}
    st = np.full((S1, L), -1, int)          # closed state of every link at every slot (latest observation), -1 = none yet
    for l in range(L):
        vi = [s for s in range(S1) if np.isfinite(x[s, l])]
        q = {s: bool(x[s, l] <= 1.0) for s in vi}
        qc = dict(q)
        for a in range(1, len(vi) - 1):
            if (not q[vi[a]]) and q[vi[a - 1]] and q[vi[a + 1]]:
                qc[vi[a]] = True
        if vi:
            last = vi[-1]
            if qc[last]:
                frees = [s for s in vi if not qc[s]]
                start = frees[-1] if frees else vi[0] - 1
                out['ck_run'][l] = T - start
            else:
                out['ck_run'][l] = 0.0
            qs = [s for s in vi if q[s]]
            out['ck_first'][l] = T - qs[0] if qs else np.nan
            tr = [vi[a] for a in range(1, len(vi)) if (not qc[vi[a]]) and qc[vi[a - 1]]]
            out['ck_clear'][l] = T - tr[-1] if tr else np.nan
            out['ck_frac'][l] = len(qs) / len(vi)
            out['ck_minr'][l] = min(x[s, l] for s in vi)
        else:
            out['ck_first'][l] = np.nan; out['ck_clear'][l] = np.nan; out['ck_frac'][l] = np.nan; out['ck_minr'][l] = np.nan
        out['ck_nep'][l] = sum(1 for a in range(len(vi)) if qc[vi[a]] and (a == 0 or not qc[vi[a - 1]]))
        cur = -1
        for s in range(S1):
            if s in qc:
                cur = int(qc[s])
            st[s, l] = cur
    nq = [int((st[s] == 1).sum()) for s in range(S1)]
    peak = max(nq)
    lastpk = max(s for s in range(S1) if nq[s] == peak)
    f2 = [s for s in range(S1) if nq[s] >= 2]
    z = lambda s: True if s < 0 else nq[s] == 0
    lz = [s for s in range(S1) if z(s) and z(s - 1) and z(s - 2)]
    f1 = [s for s in range(S1) if nq[s] >= 1]
    cc = dict(cc_first=T - f2[0] if f2 else np.nan, cc_peak=peak, cc_peak_sh=peak / L, cc_since_peak=T - lastpk,
              cc_rel=nq[T] / max(peak, 1), cc_d6=nq[T] - nq[max(T - 6, 0)], cc_d12=nq[T] - nq[max(T - 12, 0)],
              cc_d24=nq[T] - nq[max(T - 24, 0)],
              cc_age=(T - lz[-1] if lz else T - (f1[0] if f1 else 0) + 1) if nq[T] > 0 else 0.0, cc_area=sum(nq) / L)
    for c, v in cc.items():
        out[c][:] = v
    return np.stack([out[c] for c in CKN], -1).astype(np.float32)


def cmp(a, b):
    """max abs diff with identical NaN pattern"""
    na, nb_ = np.isnan(a), np.isnan(b)
    if (na != nb_).any():
        return np.inf, int((na != nb_).sum())
    m = ~na
    return (float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0), 0


# ---------------------------------------------------------------- (3) kernel block identical to the lab block
def fsrc(path, names):
    src = open(path).read(); tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}


blk = ['ck_ffill', 'ck_bfill', 'ck_features', 'ck_compose']
ls_, ks_ = fsrc(LFILE, blk), fsrc(KFILE, blk)
lab.log('kernel block identical to lab block:', {n: ls_[n] == ks_[n] for n in blk})

# compile kernel functions (top-level defs + needed constants only; main() never runs)
ksrc = open(KFILE).read(); ktree = ast.parse(ksrc)
keep = [n for n in ktree.body if isinstance(n, ast.FunctionDef) and n.name in
        ('ck_ffill', 'ck_bfill', 'ck_features', 'ck_compose', 'ck_thin', 'ck_train', 'ck_day_cube', 'ck_split')]
KNS = dict(np=np, pd=pd, H=12, F=6, PANELS=list(lab.PANELS), CK_REGIMES=(0.507, 0.705, 0.803), CK_REGIME_P=(0.297, 0.359, 0.344))
exec(compile(ast.Module(body=keep, type_ignores=[]), KFILE, 'exec'), KNS)
KNS['CK_ALL'] = CKN
import ogclock_lib as ck
assert ck.CK_ALL == CKN

# ---------------------------------------------------------------- (1) independent recomputation vs cache, train sets
vd = vis_days_indep()
vj = np.array(json.load(open(f'{ck.RES}/vis_days.json')))
lab.log(f'vis_days: n {len(vd)} (cache {len(vj)}), max abs diff {np.abs(vd - vj).max() if len(vd) == len(vj) else "len differs"}; '
        f'mean {vd.mean():.3f}')
rng = np.random.default_rng(12345)
summary = []
for panel in PANELS_CHK:
    P = lab.load_panel(panel); V = P['V']; cut = P['cut'].astype(np.float32); D, _, L = V.shape
    pi = lab.PANELS.index(panel)
    TH = thin_indep(pi, D, L, vd)
    for s in ('official', 'offlike', 'runstart', 'faith3', 'pseudo'):
        rc = lab.rows(panel, s, lab.ONGOING); W = len(rc.win)
        if not W:
            continue
        FE = np.load(f'{FED}/fe_{panel}__{s}.npy', mmap_mode='r')
        d_all = rc.win.d.to_numpy(); t_all = rc.win.t.to_numpy()
        vis_T = np.random.default_rng(W).random((W, L)) < 0.62
        pick = np.sort(rng.choice(W, min(NPER, W), replace=False))
        worst = 0.0; nanmis = 0
        for i in pick:
            d, T = int(d_all[i]), int(t_all[i])
            x = np.full((T + 1, L), np.nan, np.float32)
            for s_ in range(T - H):
                x[s_] = np.where(TH[d, s_], V[d, s_], np.nan)
            x[T - H:T] = V[d, T - H:T]
            x[T] = np.where(vis_T[i], V[d, T], np.nan)
            mine = feats_loop(x / cut[None, :], T)
            md, nm = cmp(mine.astype(np.float64), np.asarray(FE[i], np.float64))
            worst = max(worst, md); nanmis += nm
        summary.append(dict(panel=panel, set=s, n=len(pick), max_abs_diff=worst, nan_mismatch=nanmis))
        lab.log(f'{panel} {s}: {len(pick)} windows, max abs diff vs cache {worst:.2e}, NaN-pattern mismatches {nanmis}')

    # ------------------------------------------------------------ (2) perturbation: LAB and KERNEL functions
    rc = lab.rows(panel, 'offlike', lab.ONGOING)
    pick = np.sort(rng.choice(len(rc.win), min(60, len(rc.win)), replace=False))
    d = rc.win.d.to_numpy()[pick]; t = rc.win.t.to_numpy()[pick]
    Vh = V[d[:, None], t[:, None] + np.arange(-H, 0)[None]]
    VT = np.where(rng.random((len(d), L)) < 0.62, V[d, t], np.nan).astype(np.float32)
    day = V[d].copy()
    junk = day.copy(); s3 = np.arange(288)[None, :, None]
    junk = np.where(s3 >= t[:, None, None] - H, rng.uniform(5, 120, junk.shape).astype(np.float32), junk)   # rewrite every slot >= T-12
    out = {}
    for nm, fn_c, fn_f in (('lab', ck.ck_compose, ck.ck_features), ('kernel', KNS['ck_compose'], KNS['ck_features'])):
        a = fn_f(fn_c(day, t, Vh, VT, cut, TH[d]), t); b = fn_f(fn_c(junk, t, Vh, VT, cut, TH[d]), t)
        ch_future = sum(int((~np.isclose(a[c], b[c], equal_nan=True)).any(1).sum()) for c in CKN)
        Vh2 = Vh.copy(); Vh2[:, -1] = np.where(np.isfinite(Vh2[:, -1]), Vh2[:, -1] * 0.3, 20.0)
        c2 = fn_f(fn_c(day, t, Vh2, VT, cut, TH[d]), t)
        ch_hist = int(sum((~np.isclose(a[c], c2[c], equal_nan=True)).any(1).sum() for c in CKN) > 0)
        early = day.copy(); early[:, :max(int(t.min()) - H - 1, 1)] *= 0.2
        c3 = fn_f(fn_c(early, t, Vh, VT, cut, TH[d]), t)
        ch_early = int(sum((~np.isclose(a[c], c3[c], equal_nan=True)).any(1).sum() for c in CKN) > 0)
        out[nm] = a
        lab.log(f'{panel} perturb [{nm}]: features changed by data at slots >= T-12 of the day cube: {ch_future} (must be 0); '
                f'history T-1 reaches features: {bool(ch_hist)}; early visible data reaches features: {bool(ch_early)}')
    lab.log(f'{panel} lab == kernel functions on identical inputs:', all(np.array_equal(out['lab'][c], out['kernel'][c], equal_nan=True) for c in CKN))
    # kernel ck_train (all-finite variant): eligible cells thinned, data >= T-12 must not matter
    Pk = dict(V=V, Vf=V, E=np.isfinite(V), THIN=TH, cut=cut)
    a = KNS['ck_train'](Pk, d, t, Vh, VT)
    Vj = V.copy()
    for i_ in range(len(d)):
        Vj[d[i_], t[i_] - H:] = rng.uniform(5, 120, Vj[d[i_], t[i_] - H:].shape)
    # windows on the same day share a day row: only compare windows whose day appears once in the sample
    uniq = np.array([np.sum(d == x) == 1 for x in d])
    b = KNS['ck_train'](dict(Pk, V=Vj, Vf=Vj), d, t, Vh, VT)
    ch = sum(int((~np.isclose(a[c][uniq], b[c][uniq], equal_nan=True)).any(1).sum()) for c in CKN)
    lab.log(f'{panel} kernel ck_train: features changed by train data at slots >= T-12: {ch} (of {int(uniq.sum())} windows; must be 0)')
    # kernel ck_split: masked day cube with the full day (future included) -> future must not matter
    wi = pd.DataFrame(dict(date=[f'd{x}' for x in d]))
    DAY = {f'd{x}': V[x] for x in np.unique(d)}
    a = KNS['ck_split'](dict(cut=cut), wi, DAY, t, Vh, VT)
    DAY2 = {}
    for x in np.unique(d):
        tt = t[d == x].min(); c_ = V[x].copy(); c_[tt - H:] = rng.uniform(5, 120, c_[tt - H:].shape); DAY2[f'd{x}'] = c_
    b = KNS['ck_split'](dict(cut=cut), wi, DAY2, t, Vh, VT)
    ch = sum(int((~np.isclose(a[c][uniq], b[c][uniq], equal_nan=True)).any(1).sum()) for c in CKN)
    lab.log(f'{panel} kernel ck_split: features changed by masked-day data at slots >= T-12: {ch} (of {int(uniq.sum())} windows; must be 0)')
    lab.drop_panel(panel)

# ---------------------------------------------------------------- (1b) pseudo-March cache: masked layer of the whole day <= T
for panel in PANELS_CHK:
    P = lab.load_panel(panel); cut = P['cut'].astype(np.float32)
    z = np.load(P['npz'], allow_pickle=False)
    days = np.array([str(x) for x in z['validation_days']]); o = np.argsort(days)
    Vm = z['validation_Vm'][o].astype(np.float32)
    for s in ('offlike_m', 'runstart_m'):
        f = f'{FED}/pm_{panel}__{s}'
        if not os.path.exists(f + '.npz'):
            continue
        FE = np.load(f + '.npz')['FE']; win = pd.read_pickle(f + '.pkl')
        pick = np.sort(rng.choice(len(win), min(NPER, len(win)), replace=False)); worst = 0.0; nanmis = 0
        for i in pick:
            d, T = int(win.d.iloc[i]), int(win.t.iloc[i])
            mine = feats_loop(Vm[d, :T + 1] / cut[None, :], T)
            md, nm = cmp(mine.astype(np.float64), FE[i].astype(np.float64)); worst = max(worst, md); nanmis += nm
        summary.append(dict(panel=panel, set=s, n=len(pick), max_abs_diff=worst, nan_mismatch=nanmis))
        lab.log(f'{panel} {s}: {len(pick)} windows, max abs diff vs cache {worst:.2e}, NaN-pattern mismatches {nanmis}')
    lab.drop_panel(panel)
print(pd.DataFrame(summary).to_string(index=False))
lab.log('DONE')
