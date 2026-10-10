"""parity test: t1div_report recon / Runs.tv == kaggle_jobs/swap_fdrecon.py recon / tv (driven through its own globals)."""
import sys, os, numpy as np, pandas as pd
sys.argv = [sys.argv[0], 'cands=none']
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t1div_report as R, t1div_lib as L
from t1ramp_exp import load_panel, month_cube
panel, fold, split = 'D7_I210_W', 0, 'validation'
z, x, st = load_panel(panel); meta = L.load_meta(panel); FD = R.fd_station(st)
Mc = month_cube(panel, z, x, split); obs = Mc['obs']; EM = L.eval_masks(Mc, meta, split, fold)
rel = obs & ~(EM['HO'] | EM['EB']); ln_s = st.lanes.to_numpy()
s0 = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
keys = [f'{split}_blk', f'{split}_ord']; cat = lambda c: np.concatenate([s0[f'{k}|{c}'] for k in keys])
d, t, s = cat('d'), cat('t'), cat('s'); v = np.concatenate([s0[f'base|{k}|v'] for k in keys]).astype(float)
q = np.concatenate([s0[f'base|{k}|q'] for k in keys]).astype(float)
kobs = np.where(rel, Mc['mq'] / ln_s[None, None, :] / np.clip(Mc['mv'], 1, None), np.nan)
v1, q1 = R.recon(v, q, s, FD); dn = pd.to_datetime(pd.Series(Mc['days'])); daynum = ((dn - dn.iloc[0]).dt.days).to_numpy()
v1, q1 = R.Runs(d, t, s, FD, kobs, daynum).tv(v1, q1, s, FD)
# production functions with their globals
sw = R.sw; days = Mc['days']
day_ns = pd.to_datetime(pd.Series(days)).to_numpy('datetime64[ns]').astype('int64')
links = st.index.to_numpy()
ks = pd.DataFrame({'panel': panel, 'split': split, 'station_id': links[s], 'link_id': links[s]})
tsns = day_ns[d] + t.astype('int64') * 300 * 10 ** 9
dr, tr, sr = np.nonzero(rel)
relf = pd.DataFrame({'station_id': links[sr], 'ts': day_ns[dr] + tr.astype('int64') * 300 * 10 ** 9,
                     'k': Mc['mq'][rel] / ln_s[sr] / np.clip(Mc['mv'][rel], 1, None)})
sw.__dict__.update(ks=ks, tsns=tsns, hasfd=FD['ok'][s], ln=FD['ln'][s], vf=FD['vf'][s], kjl=FD['kjl'][s], w=FD['w'][s], vf_ok=FD['ok'][s],
                   released=lambda p_, s_: relf, print=lambda *a, **k: None)
v2, q2 = sw.recon(v, q); v2, q2 = sw.tv(v2, q2, 'test')
print('cells', len(v), 'changed by post', int((np.abs(v1 - v) > 1e-9).sum() + (np.abs(q1 - q) > 1e-9).sum()),
      '| max |dv| lab-vs-kernel', np.abs(v1 - v2).max(), 'max |dq|', np.abs(q1 - q2).max())
bad = np.abs(q1 - q2) > 1e-6; print("n bad", bad.sum(), "t of bad", np.unique(t[bad])[:20], "d", np.unique(d[bad])[:10]); assert np.allclose(v1, v2, atol=1e-6) and np.allclose(q1, q2, atol=1e-6)
print('post parity OK')
