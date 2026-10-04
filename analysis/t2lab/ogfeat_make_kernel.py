"""ogfeat: generate kaggle_jobs/t2_ogfeat_v23.py from kaggle_jobs/t2_model_v22.py (read-only source) by asserted text edits.
The feature block is copied verbatim from ogfeat_lib.py (between the 'kernel-portable block' markers), so the kernel computes exactly the
lab features.  Usage: ogfeat_make_kernel.py <static names, comma separated | @file> <step names | -> <added production cols | ->"""
import ast, json, sys

ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
SRC = f'{ROOT}/kaggle_jobs/t2_model_v22.py'
DST = f'{ROOT}/kaggle_jobs/t2_ogfeat_v23.py'
LIB = f'{ROOT}/analysis/t2lab/ogfeat_lib.py'


def lst(a):
    if a in ('-', ''):
        return []
    if a.startswith('@'):
        return json.load(open(a[1:]))
    return a.split(',')


static, step, addprod = lst(sys.argv[1]), lst(sys.argv[2]), lst(sys.argv[3])
s = open(SRC).read()
lib = open(LIB).read()
a = lib.index('# ============================================================================================ kernel-portable block')
b = lib.index('# ============================================================================================ end of portable block')
block = lib[a:b] + '# ============================================================================================ end of portable block\n'
block = block.replace("    with warnings.catch_warnings(), np.errstate(all='ignore'):\n        warnings.simplefilter('ignore')\n",
                      "    with np.errstate(all='ignore'):\n")
assert 'warnings' not in block


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- 1. docstring
rep('"""v22 = v21 restricted to the selection-faithful F03 recipe, bagged: F03 np511 + id511 with seeds 0 and 1 (4 models).',
    '"""v23 (track ogfeat) = v22 (F03 bag: np511 + id511 x seeds 0, 1) with EXTRA ONGOING FEATURES appended to both column sets.\n'
    'New columns (all from the 60-min history, the masked slot-T value, static vectors and the ramp columns; nothing after T):\n'
    '  OG_STATIC (per window x link), OG_STEP (step-dependent extrapolations), OG_ADD_PROD (production columns re-admitted to np / id).\n'
    'Diff to t2_model_v22.py: (a) feature block og_* (copied from analysis/t2lab/ogfeat_lib.py), (b) make_rows / load_split append the new\n'
    'columns, (c) the v20-style CUR rows are no longer built (they were only logged in v22), (d) column sets / variant names:\n'
    'fs4new = mean of the 4 new-feature models (main), fs2new = np + id seed 0 (fallback if the time guard skips a model), fs_npnew.\n'
    'CONTROL / footprint: fs4new vs tfb-t2v22 queue_submission_fs4.csv when that output is attached (agreement, Jaccard; not expected to be 1).\n'
    'Outputs: queue_submission_<variant>.csv (onset rows 0), t2v23_probs.npz, t2v23_footprint.csv, t2v23_trainsets.csv.\n\n'
    'v22 notes: v22 = v21 restricted to the selection-faithful F03 recipe, bagged: F03 np511 + id511 with seeds 0 and 1 (4 models).')
# ---- 2. config + feature block
rep("ONGOING = 'queue_ongoing'\n",
    "ONGOING = 'queue_ongoing'\n"
    f"OG_STATIC = {static!r}\n"
    f"OG_STEP = {step!r}\n"
    f"OG_ADD_PROD = {addprod!r}          # production columns re-admitted to the np / id sets\n"
    "CAPD = 30.0\nBIG = 10 ** 6\n")
rep("def established(Qh):\n", block.replace('OG_STEP = [', 'OG_STEP_ALL = [') + "\n\ndef established(Qh):\n")
# ---- 3. make_rows (train rows)
rep("    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n    W, L = qlast.shape\n    rows = []\n",
    "    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n"
    "    OGX = og_extra(P, Vh, Fh, VT, Fe); Fe.update({k: OGX[k] for k in OG_STATIC})          # v23: extra ongoing features\n"
    "    W, L = qlast.shape\n    rows = []\n")
rep("        feats = {k: v for k, v in Fe.items()}\n        feats['h'] = np.full((W, L), h, np.float32)\n",
    "        feats = {k: v for k, v in Fe.items()}\n        feats['h'] = np.full((W, L), h, np.float32)\n"
    "        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})\n")
# ---- 4. load_split (released windows)
rep("    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))\n"
    "    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])\n"
    "    for h in range(1, F + 1):\n"
    "        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)\n",
    "    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))\n"
    "    OGX = og_extra(P, Vh, Fh, VT, Fe); Fe.update({k: OGX[k] for k in OG_STATIC})          # v23: extra ongoing features\n"
    "    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])\n"
    "    for h in range(1, F + 1):\n"
    "        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)\n"
    "        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})\n")
# ---- 5. panel_rows: F03 rows only
rep('''    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    R = make_rows(P, P['trainw'], P['prior_all'])
    X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    cur = dict(X=X[e], y=y[e], rs=P['cur_rs'][meta[:, 0]][e])
    del R, X; gc.collect()
    R = make_rows(P, P['exw'], P['prior_all'])
    if R is None:
        ex = dict(X=np.zeros((0, len(cols)), np.float32), y=np.zeros(0, bool), rs=np.zeros(0, bool), incur=np.zeros(0, bool))
    else:
        X, y, e, meta, _ = to_matrix(R, ONGOING, cols, ongoing_keep(R))
        ex = dict(X=X[e], y=y[e], rs=P['ex_rs'][meta[:, 0]][e], incur=P['ex_incur'][meta[:, 0]][e])
        del R, X; gc.collect()
    return cols, cur, ex
''', '''    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    R = make_rows(P, P['exw'], P['prior_all'])                              # v23: the v20-style CUR rows are not built
    X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    ex = dict(X=X[e], y=y[e], rs=P['ex_rs'][meta[:, 0]][e], incur=P['ex_incur'][meta[:, 0]][e])
    del R, X; gc.collect()
    return cols, ex
''')
# ---- 6. variants / footprint reference / control
rep('''VARIANTS = {'fs2': ('fs_np', 'fs_id'),                                 # = v21 fs2 (control)
            'fs4': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),             # F03 bag of 4
            'fs_np': ('fs_np',)}
''', '''VARIANTS = {'fs2new': ('fs_np', 'fs_id'),                              # np + id seed 0 with the new features (fallback)
            'fs4new': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),          # F03 bag of 4 with the new features (main)
            'fs_npnew': ('fs_np',)}
''')
rep("            a, b = pr[i], preds['fs2'][i]\n", "            a, b = pr[i], preds['fs2new'][i]\n")
rep('''        log(f'FOOTPRINT {split}: ongoing windows {g[g.variant == "fs2"].shape[0]} | mean queued cells at T (q_last) '
            f'{g[g.variant == "fs2"].size_T.mean():.2f}')''',
    '''        log(f'FOOTPRINT {split}: ongoing windows {g[g.variant == "fs2new"].shape[0]} | mean queued cells at T (q_last) '
            f'{g[g.variant == "fs2new"].size_T.mean():.2f}')''')
rep("mean Jaccard vs fs2 {x.jacc_cur2.mean():.4f}", "mean Jaccard vs fs2new {x.jacc_cur2.mean():.4f}")
a = s.index('def control_check(S, names):'); b = s.index('# ------------------------------------------------------------------ main')
s = s[:a] + '''def control_check(S, names):
    """footprint of fs4new (fallback fs2new) against tfb-t2v22 queue_submission_fs4.csv on ongoing rows, if that output is attached.
    The models differ by design (new features), so this is a sanity range, not an identity: Jaccard well below 0.85 = suspicious."""
    ref = sorted(glob.glob('/kaggle/input/**/tfb-t2v22/**/queue_submission_fs4.csv', recursive=True))
    v = 'fs4new' if 'fs4new' in names else ('fs2new' if 'fs2new' in names else None)
    if not ref or v is None:
        log('CONTROL: skipped (tfb-t2v22 queue_submission_fs4.csv not attached or no variant)', ref)
        return None
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = S.merge(r.rename(columns={'queue_pred': 'ref'}), on=['window_id', 'timestamp', 'link_id'], how='left')
    og = m.is_og.to_numpy() & m.ref.notna().to_numpy()
    a = m[f'queue_pred_{v}'].to_numpy()[og].astype(int); b = m.ref.to_numpy()[og].astype(int)
    agree = float((a == b).mean()) if og.any() else float('nan')
    jac = float((a & b).sum() / max((a | b).sum(), 1))
    isval = m.window_id.astype(str).str.contains('val', case=False).to_numpy()[og]
    for nm, k in (('validation', isval), ('private', ~isval)):
        if k.any():
            log(f'CONTROL {v} vs v22 fs4 [{nm}]: ongoing rows {int(k.sum())} | changed {int((a[k] != b[k]).sum())} | Jaccard '
                f'{float((a[k] & b[k]).sum() / max((a[k] | b[k]).sum(), 1)):.5f} | positives new {int(a[k].sum())} ref {int(b[k].sum())}')
    log(f'CONTROL {v} vs {ref[0]}: ongoing rows {int(og.sum())} | agreement {agree:.5f} | Jaccard of queued cells {jac:.5f}'
        f' | positives {v} {int(a.sum())} ref {int(b.sum())}' + (' | OK' if jac >= 0.85 else ' | WARNING'))
    return agree


''' + s[b:]
# ---- 7. main: rows, column sets, models, outputs
rep('''    cols = None; CUR = []; EX = []
    for panel, P in panels.items():
        cols, cur, ex = panel_rows(P, cols)
        log(panel, f'rows CUR {len(cur["y"])} (pos {cur["y"].mean():.3f}) | EX {len(ex["y"])} (pos {ex["y"].mean() if len(ex["y"]) else float("nan"):.3f};'
            f' run-start rows {int(ex["rs"].sum())}, pos {ex["y"][ex["rs"]].mean() if ex["rs"].any() else float("nan"):.3f})')
        CUR.append(cur); EX.append(ex)
''', '''    cols = None; EX = []
    for panel, P in panels.items():
        cols, ex = panel_rows(P, cols)
        log(panel, f'rows EX {len(ex["y"])} (pos {ex["y"].mean() if len(ex["y"]) else float("nan"):.3f};'
            f' run-start rows {int(ex["rs"].sum())}, pos {ex["y"][ex["rs"]].mean() if ex["rs"].any() else float("nan"):.3f})')
        EX.append(ex)
''')
rep('''    Xc, yc, rc = cat(CUR, 'X'), cat(CUR, 'y').astype(np.float32), cat(CUR, 'rs')
    Xe, ye, re_, ie = cat(EX, 'X'), cat(EX, 'y').astype(np.float32), cat(EX, 'rs'), cat(EX, 'incur')
    del CUR, EX; gc.collect()
    cols_np = [x for x in cols if x not in PRIOR_COLS]; ix_np = [cols.index(x) for x in cols_np]
    cols_id = [x for x in cols if x not in ID_DROP]; ix_id = [cols.index(x) for x in cols_id]
    w_f = np.where(re_, W_RS, W_DENSE).astype(np.float32)
    hm = re_ & ~ie                                                         # run-start rows whose window is not already in CUR
    y_h = np.concatenate([yc, ye[hm]]); w_h = np.concatenate([np.where(rc, W_RS, W_DENSE), np.full(int(hm.sum()), W_RS)]).astype(np.float32)
    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)}')
    log(f'POSITIVE RATES | CUR rows {len(yc)} pos {yc.mean():.4f} | F03 rows {len(ye)} pos {ye.mean():.4f} weighted {np.average(ye, weights=w_f) if len(ye) else float("nan"):.4f}'
        f' (run-start rows {int(re_.sum())} pos {ye[re_].mean() if re_.any() else float("nan"):.4f}; rest pos {ye[~re_].mean() if (~re_).any() else float("nan"):.4f})'
        f' | H rows {len(y_h)} pos {y_h.mean():.4f} weighted {np.average(y_h, weights=w_h):.4f} (CUR run-start rows {int(rc.sum())}, added run-start rows {int(hm.sum())})')
''', '''    Xe, ye, re_ = cat(EX, 'X'), cat(EX, 'y').astype(np.float32), cat(EX, 'rs')
    del EX; gc.collect()
    cols_np = [x for x in cols if x not in PRIOR_COLS or x in OG_ADD_PROD]; ix_np = [cols.index(x) for x in cols_np]
    cols_id = [x for x in cols if x not in ID_DROP or x in OG_ADD_PROD]; ix_id = [cols.index(x) for x in cols_id]
    w_f = np.where(re_, W_RS, W_DENSE).astype(np.float32)
    new_cols = [x for x in cols_np if x in OG_STATIC or x in OG_STEP or x in OG_ADD_PROD]
    assert len(new_cols) == len(OG_STATIC) + len(OG_STEP) + len(OG_ADD_PROD), 'a new column is missing from the matrix'
    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | new {len(new_cols)}: {new_cols}')
    log('new feature finite share / mean:', {c: (round(float(np.isfinite(Xe[:200000, cols.index(c)]).mean()), 3),
                                               round(float(np.nanmean(Xe[:200000, cols.index(c)])), 3)) for c in new_cols})
    log(f'POSITIVE RATES | F03 rows {len(ye)} pos {ye.mean():.4f} weighted {np.average(ye, weights=w_f) if len(ye) else float("nan"):.4f}'
        f' (run-start rows {int(re_.sum())} pos {ye[re_].mean() if re_.any() else float("nan"):.4f}; rest pos {ye[~re_].mean() if (~re_).any() else float("nan"):.4f})')
''')
rep("    del Xc, Xe; gc.collect()\n", "    del Xe; gc.collect()\n")
rep("t2v22_trainsets.csv", "t2v23_trainsets.csv", 2)
rep("t2v22_probs.npz", "t2v23_probs.npz", 2)
rep("t2v22_footprint.csv", "t2v23_footprint.csv", 2)
open(DST, 'w').write(s)
ast.parse(s)
print('written', DST, len(s.splitlines()), 'lines;', len(static), 'static +', len(step), 'step +', len(addprod), 'production columns')
