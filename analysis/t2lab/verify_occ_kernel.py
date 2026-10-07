"""verify_occ_kernel.py : the kernel's own occ_calib / occ_features / ffill_time / shift_links (compiled from
kaggle_jobs/t2_occ_v26.py with ast, main never runs) vs my independent builder, on 2 panels (offlike + runstart windows)."""
import ast, warnings
import numpy as np
import lab
import verify_occ_feat as vf

SRC = f'{lab.ROOT}/kaggle_jobs/t2_occ_v26.py'
tree = ast.parse(open(SRC).read())
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('occ_calib', 'occ_features', 'ffill_time', 'shift_links')]
g = dict(np=np, warnings=warnings, H=12, NB=(-3, -2, -1, 1, 2, 3))
g['OCC_COLS'] = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in g['NB']] + ['dens_dev']
exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, 'exec'), g)
assert all(k in g for k in ('occ_calib', 'occ_features', 'ffill_time', 'shift_links'))
for panel in ('D7_I10_E', 'D12_I5_S'):
    C = vf.cubes(panel); P = C['P']
    oc, cc, n_est = g['occ_calib'](C['Vall'], C['Fall'], C['O'], P['cut'], P['vf'], P['lanes'])
    myoc, mycc = vf.calib_all(panel)
    msg = [f'calib kernel vs mine (k=4): max|d ocrit| {np.max(np.abs(oc - myoc[4])):.2e} max|d c| {np.max(np.abs(cc - mycc[4])):.2e} est {n_est}']
    KP = dict(lanes=P['lanes'], ocrit=oc, occ_c=cc)
    for name in ('offlike', 'runstart'):
        rc = vf.rows_of(panel, name)
        d = rc.win.d.to_numpy(); t = rc.win.t.to_numpy()
        vis = vf.slotT_visible(C, name, lab.ONGOING, d, t)
        idx = t[:, None] + np.arange(-12, 0)[None]
        Oh = C['O'][d[:, None], idx]; Vh = C['Vall'][d[:, None], idx]; Fh = C['Fall'][d[:, None], idx]
        OT = np.where(vis, C['O'][d, t], np.nan).astype(np.float32)
        kf = g['occ_features'](Oh, OT, Vh, Fh, KP)
        R = vf.raw_from(Oh, OT, Vh, Fh, P['lanes'])
        W, L = OT.shape
        wi, li = np.meshgrid(np.arange(W), np.arange(L), indexing='ij'); wi = wi.ravel(); li = li.ravel()
        mine = vf.feats_rows(R, myoc[4], mycc[4], wi, li)
        kern = np.stack([kf[c][wi, li] for c in g['OCC_COLS']], 1)
        same = np.isclose(mine, kern, rtol=1e-5, atol=1e-6, equal_nan=True)
        msg.append(f'{name}: W {W} kernel==mine {same.all()} (frac diff {1 - same.mean():.1e})')
    lab.log(panel, ' | '.join(msg))
lab.log('KERNEL CHECK DONE')
