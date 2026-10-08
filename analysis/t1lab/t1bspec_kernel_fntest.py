"""Function-level local run of kaggle_jobs/t1_bspec_member_a.py on one Task 2 panel (released-data npz stands in for the
parquet release): official origins + condition from window_index, faithful sim_blackouts, features() incl. the timing
columns on (i) the released month (official blackouts), (ii) a simulated-blackout view, (iii) the darkened train split;
parity of the kernel BSPEC functions with the lab (t1bspec_exp) on the same views."""
import os, sys, types, glob as _glob
import numpy as np, pandas as pd

ROOT = sys.argv[1]; PANEL = sys.argv[2] if len(sys.argv) > 2 else 'D7_I210_W'
TMP = sys.argv[3]
sys.path.insert(0, os.path.join(ROOT, 'analysis', 't1lab'))
import t1bspec_exp as L  # noqa: E402

src = open(os.path.join(ROOT, 'kaggle_jobs', 't1_bspec_member_a.py')).read()
cut = src.index('results = []\nsubs = []')
src = src[:cut].replace("REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]", f"REL = {TMP!r}")
ns = {'__name__': 'kernel'}
exec(compile(src, 'kernel_a', 'exec'), ns)
K = types.SimpleNamespace(**ns)
print('kernel BCOLS', K.N_BS, '| PANELS', K.PANELS)
assert K.BCOLS == L.BCOLS

z, x, st = L.load_panel(PANEL)
meta = np.load(f'{L.R0.CACHE}/{PANEL}_meta.npz', allow_pickle=True); hv, hq = meta['hv'], meta['hq']
vf = st.vf.to_numpy(); dirn = L.direction(PANEL)
assert dirn == K.ramp_direction(PANEL)
for split in ('validation', 'private'):
    os.makedirs(f'{TMP}/task2/{PANEL}/{split}', exist_ok=True)
    pd.DataFrame(dict(forecast_origin=z[f'{split}_wi_forecast_origin'].astype(str), condition=z[f'{split}_wi_condition'].astype(str))
                 ).to_csv(f'{TMP}/task2/{PANEL}/{split}/window_index.csv', index=False)
    M = L.month_cube(PANEL, z, x, split)
    RM = L.ramp_cubes(PANEL, z, split, order=np.argsort(z[f'{split}_days'].astype(str)))
    cs = K.Cube.__new__(K.Cube)
    cs.days = list(M['days']); cs.mv = M['mv']; cs.mq = M['mq']; cs.mo = np.full(M['mv'].shape, np.nan, np.float32)
    cs.melig = M['Em']; cs.dow = M['dow']; cs.reg = M['reg']; cs.S = len(st)
    cs.ramp = dict(ON=RM['ON'], OFF=RM['OFF'], has_on=RM['has_on'], has_off=RM['has_off'], dirn=dirn)
    cs.bs_origins = K.official_origins(PANEL, split, cs)
    lab_off = [(d, t, 0 if c == 'queue_onset' else 1) for (d, t), c in zip(M['real'], z[f'{split}_wi_condition'].astype(str))]
    assert sorted(cs.bs_origins) == sorted(lab_off), (cs.bs_origins, lab_off)
    rr = K.real_rows(PANEL, split, cs)
    tb, n_c, n_s, sims = K.sim_blackouts(cs, st, rr, K.DARK_SEED + (split == 'private'))
    print(split, 'official origins', cs.bs_origins, '| faithful cands', n_c, 'selected', n_s, 'onset', sum(s[2] == 0 for s in sims))
    assert not (tb & rr).any()
    # (i) released month with the official origins
    F, tv, tq = K.features(cs, st, hv[cs.dow], hq[cs.dow])
    cols = [k for k in F if k not in {'hv', 'hq', 'tod', 'dow'}]
    assert len(F) == K.V12_NFEAT + K.N_FD + K.N_RAMP + K.N_BS and list(F)[-K.N_BS:] == K.BCOLS
    bc = F['bs_cond']; inblk = np.zeros(bc.shape[:2], bool)
    for d, t, _ in cs.bs_origins:
        inblk[d, t + 1:t + 19] = True
    assert np.array_equal(np.isfinite(bc).any(2), inblk) and np.isfinite(bc[inblk]).all()
    tgt = M['Em'] & ~M['obs']                                          # template-like cells (hidden eligible)
    tin = tgt & inblk[:, :, None]
    print(split, 'hidden eligible cells inside official blackouts', int(tin.sum()), 'onset', int((bc[tin] == 0).sum()),
          'ongoing', int((bc[tin] == 1).sum()), '| NaN timing outside', bool(not np.isfinite(bc[~inblk]).any()))
    for k in K.BCOLS:
        fin = np.isfinite(F[k][tin]).mean()
        if fin < 0.95:
            print('   finite share on blackout targets', k, round(float(fin), 3))
    # lab parity on the same view
    Fd = L.feats(L.View(cs.mv, cs.mq, cs.dow, cs.reg), st, hv, hq, RM)
    L.add_bs(Fd, cs.mv, cs.mq, cs.bs_origins, st, dirn)
    for k in K.BCOLS + K.RCOLS + ['tv', 'tq', 'gp', 'gn']:
        assert np.array_equal(np.asarray(F[k]), np.asarray(Fd[k]), equal_nan=True), k
    # (ii) simulated-blackout view
    cv = K.Cube.__new__(K.Cube); cv.__dict__.update(cs.__dict__); cv.bs_origins = sims
    TB = np.broadcast_to(tb[:, :, None], cs.mv.shape)
    cv.mv = np.where(TB, np.nan, cs.mv); cv.mq = np.where(TB, np.nan, cs.mq)
    Fv, _, _ = K.features(cv, st, hv[cv.dow], hq[cv.dow])
    hid = M['obs'] & TB
    print(split, 'simulated-blackout rows', int(hid.sum()), 'timing finite', bool(np.isfinite(Fv['bs_cond'][hid]).all()),
          'onset share', round(float((Fv['bs_cond'][hid] == 0).mean()), 3), '| pwd_v mean |.|', round(float(np.nanmean(np.abs(Fv['bs_pwd_v'][hid]))), 4))
# (iii) train split: pseudo gap origins with cond (glob patched to local files)
pseudo = os.path.join(ROOT, 'kaggle_jobs', 'out', 't2lab_full', f'pseudo_{PANEL}.npz')
tdays = z['train_days'].astype(str)
fake = [f'/x/date={d}/part_{d.replace("-", "_")}.parquet' for d in tdays]
ns['glob'] = types.SimpleNamespace(glob=lambda pat, recursive=False: [pseudo] if 'pseudo_' in pat else fake)
T = L.train_cube(PANEL, z, x)
c = K.Cube.__new__(K.Cube)
idx = np.arange(40)
c.days = list(tdays[idx]); c.mv = np.where(T['target'][idx], np.nan, T['V'][idx]); c.mq = np.where(T['target'][idx], np.nan, T['Q'][idx])
c.mo = np.full(c.mv.shape, np.nan, np.float32); c.dow = T['dow'][idx]; c.reg = T['reg'][idx]; c.S = len(st)
RT = L.ramp_cubes(PANEL, z, 'train')
c.ramp = dict(ON=RT['ON'][idx], OFF=RT['OFF'][idx], has_on=RT['has_on'], has_off=RT['has_off'], dirn=dirn)
origins = K.gap_origins(PANEL, c.days)
pz = np.load(pseudo, allow_pickle=True)
print('train: pseudo origins in the first 40 days', len(origins), 'onset', sum(o[2] == 0 for o in origins),
      '| file onset share', round(float((pz['cond'] == 'queue_onset').mean()), 3))
c.mv, c.mq, c.mo, gapm = K.darken(c, origins)
c.bs_origins = origins
Ft, _, _ = K.features(c, st, hv[c.dow], hq[c.dow])
tmask = T['target'][idx] & T['elig'][idx]
X = np.column_stack([np.asarray(Ft[k])[tmask] for k in Ft]); colsT = list(Ft)
gp = gapm[tmask]; bcol = X[:, colsT.index('bs_cond')]
assert np.array_equal(np.isfinite(bcol), gp)
print('train: target rows', int(tmask.sum()), 'in gaps', int(gp.sum()), 'onset share', round(float((bcol[gp] == 0).mean()), 3),
      'boundary finite', round(float(np.isfinite(X[gp, colsT.index('bs_pre_v0')]).mean()), 3))
print('KERNEL FUNCTION TEST OK')
