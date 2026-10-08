"""ogfeat (2026-10-07, E1 of analysis/LEVERS_2026-10-07.md): generate the v27 ongoing kernels from kaggle_jobs/t2_occ_v26.py
(kernel tfb-t2occ = current March ongoing production: F03 bag np511 + id511, all-finite inputs + 13 occupancy features) by
asserted text edits.  The ogfeat feature block is copied verbatim from ogfeat_lib.py (between the 'kernel-portable block'
markers; only the warnings context is replaced by np.errstate), so the kernel computes exactly the lab features.

Feature set = lab arm og_*_all2 (groups g1b + g2 + g3 + g1c + g2b): names.json + names2.json static columns + the 5 kx step
columns; NO production columns re-admitted (no tod / dow).  The new features are computed from exactly the arrays that v26
feeds window_features (all-finite history Vf / Ff, all-finite slot-T value) and the ramp columns of add_ramp.

Outputs (written next to the source):
  kaggle_jobs/t2_ogfeat_v27a.py        PART='a': fs_np + fs_id (seed 0)      -> queue_submission_fs2ogf.csv, fs_npogf
  kaggle_jobs/t2_ogfeat_v27b.py        PART='b': fs_np1 + fs_id1 (seed 1)    -> queue_submission_fs2ogf1.csv
  kaggle_jobs/t2_ogfeat_v27a_smoke.py  SMOKE copy of a (2 panels, 90 days, 40 trees)
Usage: .venv/bin/python analysis/t2lab/ogfeat_make_kernel_v27.py"""
import ast, json

ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
SRC = f'{ROOT}/kaggle_jobs/t2_occ_v26.py'
LIB = f'{ROOT}/analysis/t2lab/ogfeat_lib.py'
ODIR = f'{ROOT}/data/t2local/cache/ogfeat'
STATIC = json.load(open(f'{ODIR}/names.json')) + json.load(open(f'{ODIR}/names2.json'))
STEP = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']
ADDPROD = []
assert len(STATIC) == len(set(STATIC)) == 98, len(STATIC)

lib = open(LIB).read()
a = lib.index('# ============================================================================================ kernel-portable block')
b = lib.index('# ============================================================================================ end of portable block')
BLOCK = lib[a:b] + '# ============================================================================================ end of portable block\n'
n_w = BLOCK.count("    with warnings.catch_warnings(), np.errstate(all='ignore'):\n        warnings.simplefilter('ignore')\n")
assert n_w == 2, n_w                                                    # og_extra, og_extra2
BLOCK = BLOCK.replace("    with warnings.catch_warnings(), np.errstate(all='ignore'):\n        warnings.simplefilter('ignore')\n",
                      "    with np.errstate(all='ignore'):\n")
assert 'warnings' not in BLOCK
BLOCK = BLOCK.replace('OG_STEP = [', 'OG_STEP_ALL = [')
assert BLOCK.count('OG_STEP_ALL = [') == 1


def build(part, smoke):
    s = open(SRC).read()

    def rep(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (s.count(old), old[:100])
        s = s.replace(old, new)

    tag = f'v27{part}'
    # ---- 1. docstring
    rep('"""v26 = v22af (F03 bag: np511 + id511, seeds 0 and 1, ALL-FINITE INPUTS) + 13 OCCUPANCY features.',
        f'"""{tag} (track ogfeat, E1 of analysis/LEVERS_2026-10-07.md) = v26 (tfb-t2occ: F03 np511 + id511, all-finite inputs, 13 occupancy\n'
        'features) + 98 + 5 OGFEAT columns (lab arm all2 = g1b corridor context, g2 per-link kinematics + kx step extrapolations,\n'
        'g3 demand / supply at the queue ends, g1c corridor flow / speed level and trend, g2b per-link volatility / smoothed state).\n'
        'The new columns use exactly the inputs of the v26 history features: the all-finite 60-min history [T-60, T), the all-finite\n'
        'masked slot-T value and the ramp columns of add_ramp (slots < T).  Nothing after T; no tod / dow added.\n'
        'Feature block copied verbatim from analysis/t2lab/ogfeat_lib.py by analysis/t2lab/ogfeat_make_kernel_v27.py.\n'
        "Seed split (Kaggle time limit): PART 'a' trains fs_np + fs_id (seed 0) -> fs2ogf, fs_npogf; PART 'b' trains fs_np1 + fs_id1\n"
        "(seed 1) -> fs2ogf1.  Training rows are identical in both parts; kaggle_jobs/t2_ogfeat_comb.py averages the 4 float32\n"
        'probability sets in the v26 fs4occ order and decodes -> fs4ogf.  The v20-style CUR rows of v26 (only logged) are not built.\n'
        f'Outputs: queue_submission_<variant>.csv (onset rows 0), t2{tag}_probs.npz (float32 ongoing probabilities), t2{tag}_footprint.csv,\n'
        f't2{tag}_trainsets.csv.  CONTROL / footprint: vs tfb-t2occ queue_submission_fs4occ.csv (ALARM above 3 % changed validation\n'
        'ongoing rows).\n\n'
        'v26 notes: v26 = v22af (F03 bag: np511 + id511, seeds 0 and 1, ALL-FINITE INPUTS) + 13 OCCUPANCY features.')
    if smoke:
        rep('SMOKE = False\n', 'SMOKE = True\n')
    # two models of ~2x the v26 width per kernel (~2.4 h each): let the second model start unless it would pass 10.5 h (limit 12 h)
    rep("TIME_LIMIT_S = 8.3 * 3600                   # a model is skipped if its predicted end exceeds this (Kaggle budget 9 h)\n",
        "TIME_LIMIT_S = 10.5 * 3600                  # v27: a model is skipped if its predicted end exceeds this (Kaggle CPU limit 12 h)\n")
    # ---- 2. config
    rep("OCC_COLS = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in NB] + ['dens_dev']\n",
        "OCC_COLS = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in NB] + ['dens_dev']\n"
        f"PART = {part!r}                                   # 'a' = seed-0 pair, 'b' = seed-1 pair, 'all' = bag of 4 in one kernel\n"
        f"TAG = {tag!r}\n"
        f"OG_STATIC = {STATIC!r}\n"
        f"OG_STEP = {STEP!r}\n"
        f"OG_ADD_PROD = {ADDPROD!r}          # production columns re-admitted to the np / id sets (none)\n"
        "CAPD = 30.0\nBIG = 10 ** 6\n")
    rep("def established(Qh):\n", BLOCK + "\n\ndef established(Qh):\n")
    # ---- 3. make_rows (train rows): same Vh / Fh / VT as window_features, Fe after add_ramp
    rep("    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n    occ = None\n",
        "    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n"
        "    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # v27: ogfeat columns\n"
        "    Fe.update({k: OGX[k] for k in OG_STATIC})\n"
        "    occ = None\n")
    rep("        if occ is not None:                                                # appended after the production columns (lab column order)\n"
        "            feats.update(occ)\n",
        "        if occ is not None:                                                # appended after the production columns (lab column order)\n"
        "            feats.update(occ)\n"
        "        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # v27: step-dependent columns\n")
    # ---- 4. load_split (released windows)
    rep("    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))\n",
        "    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))\n"
        "    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # v27: ogfeat columns\n"
        "    Fe.update({k: OGX[k] for k in OG_STATIC})\n")
    rep("        if occ is not None:\n            feats.update(occ)\n        R['rows'].append((h, feats, None, None))\n",
        "        if occ is not None:\n            feats.update(occ)\n"
        "        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # v27: step-dependent columns\n"
        "        R['rows'].append((h, feats, None, None))\n")
    # ---- 5. panel_rows: F03 rows only (CUR rows were only logged in v26; skipping them saves memory with ~190 columns)
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
    R = make_rows(P, P['exw'], P['prior_all'])                              # v27: the v20-style CUR rows are not built
    X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    ex = dict(X=X[e], y=y[e], rs=P['ex_rs'][meta[:, 0]][e], incur=P['ex_incur'][meta[:, 0]][e])
    del R, X; gc.collect()
    return cols, ex
''')
    # ---- 6. variants, float32 probabilities, output names
    rep('''VARIANTS = {'fs2occ': ('fs_np', 'fs_id'),                              # seed-0 pair (all-finite + occupancy inputs)
            'fs4occ': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),          # F03 bag of 4 with all-finite + occupancy inputs (candidate)
            'fs_np': ('fs_np',)}
''', '''VARIANTS = {'fs2ogf': ('fs_np', 'fs_id'),                              # seed-0 pair + ogfeat (PART a; Q4 candidate if b is late)
            'fs4ogf': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),          # bag of 4 (only when PART == 'all'; else t2_ogfeat_comb.py)
            'fs2ogf1': ('fs_np1', 'fs_id1'),                           # seed-1 pair (PART b)
            'fs_npogf': ('fs_np',)}
''')
    rep("        probs[f'{panel}__{split}__{name}'] = p[is_og].astype(np.float16)\n",
        "        probs[f'{panel}__{split}__{name}'] = p[is_og].astype(np.float32)          # v27: float32 (exact combine)\n")
    rep("t2v26occ_trainsets.csv", "t2{TAG}_trainsets.csv")
    rep("    np.savez_compressed(f'{OUT}/t2v26occ_probs.npz', **probs)\n", "    np.savez_compressed(f'{OUT}/t2{TAG}_probs.npz', **probs)\n")
    rep("        fp.to_csv(f'{OUT}/t2v26occ_footprint.csv', index=False)\n", "        fp.to_csv(f'{OUT}/t2{TAG}_footprint.csv', index=False)\n")
    assert 't2v26occ' not in s
    # ---- 7. control: footprint vs tfb-t2occ fs4occ, alarm above 3 % changed validation ongoing rows
    i = s.index('def control_check(S, names):'); j = s.index('VAL_IDS = set()')
    s = s[:i] + '''def control_check(S, names):
    """FOOTPRINT of every variant vs tfb-t2occ/queue_submission_fs4occ.csv (current March ongoing production), by split.
    The models differ by design (new features); ALARM if more than 3 % of the validation ongoing rows change."""
    ref = sorted(glob.glob('/kaggle/input/**/tfb-t2occ/**/queue_submission_fs4occ.csv', recursive=True))
    if not ref:
        log('FOOTPRINT: skipped (tfb-t2occ queue_submission_fs4occ.csv not attached)'); return None
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = S.merge(r.rename(columns={'queue_pred': 'ref'}), on=['window_id', 'timestamp', 'link_id'], how='left')
    log('FOOTPRINT: reference rows matched', int(m.ref.notna().sum()), 'of', len(m), '| unmatched ongoing rows', int((m.is_og & m.ref.isna()).sum()))
    m = m[m.is_og & m.ref.notna()]
    m['split'] = np.where(m.window_id.isin(VAL_IDS), 'validation', 'private')
    out = {}
    for name in names:
        for split, x in m.groupby('split'):
            a = x[f'queue_pred_{name}'].to_numpy().astype(int); b = x.ref.to_numpy().astype(int)
            chg = x.assign(c=(a != b)).groupby('window_id').c.any(); sh = float((a != b).mean())
            flag = ' | ALARM: more than 3 % of the validation ongoing rows changed' if (split == 'validation' and sh > 0.03) else ''
            log(f'FOOTPRINT {name} vs tfb-t2occ fs4occ | {split}: windows changed {int(chg.sum())}/{len(chg)} | rows changed {int((a != b).sum())}/{len(a)}'
                f' ({sh:.4%}) | Jaccard of queued cells {(a & b).sum() / max((a | b).sum(), 1):.4f} | positives {int(a.sum())} ref {int(b.sum())}{flag}')
            out[(name, split)] = sh
    return out


''' + s[j:]
    # ---- 8. main: rows, column sets, models of this part
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
            f' run-start rows {int(ex["rs"].sum())}, pos {ex["y"][ex["rs"]].mean() if ex["rs"].any() else float("nan"):.3f}) | cols {len(cols)}')
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
    new_cols = [x for x in cols if x in OG_STATIC or x in OG_STEP]
    assert len(new_cols) == len(OG_STATIC) + len(OG_STEP), 'a new column is missing from the matrix'
    assert all(c in cols_np and c in cols_id for c in new_cols) and all(c in cols_np and c in cols_id for c in OCC_COLS)
    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | new {len(new_cols)} | Xe {Xe.shape} {Xe.nbytes / 1e9:.2f} GB')
    log('new feature finite share / mean (first 200k rows):', {c: (round(float(np.isfinite(Xe[:200000, cols.index(c)]).mean()), 3),
                                                              round(float(np.nanmean(Xe[:200000, cols.index(c)])), 3)) for c in new_cols})
    log(f'POSITIVE RATES | F03 rows {len(ye)} pos {ye.mean():.4f} weighted {np.average(ye, weights=w_f) if len(ye) else float("nan"):.4f}'
        f' (run-start rows {int(re_.sum())} pos {ye[re_].mean() if re_.any() else float("nan"):.4f}; rest pos {ye[~re_].mean() if (~re_).any() else float("nan"):.4f})')
''')
    rep('''    fit('fs_np', lambda: Xe[:, ix_np], ye, w_f, cols_np)
    fit('fs_id', lambda: Xe[:, ix_id], ye, w_f, cols_id)
    fit('fs_np1', lambda: Xe[:, ix_np], ye, w_f, cols_np, S1)
    fit('fs_id1', lambda: Xe[:, ix_id], ye, w_f, cols_id, S1)
    del Xc, Xe; gc.collect()
''', '''    if PART in ('a', 'all'):
        fit('fs_np', lambda: Xe[:, ix_np], ye, w_f, cols_np)
        fit('fs_id', lambda: Xe[:, ix_id], ye, w_f, cols_id)
    if PART in ('b', 'all'):
        fit('fs_np1', lambda: Xe[:, ix_np], ye, w_f, cols_np, S1)
        fit('fs_id1', lambda: Xe[:, ix_id], ye, w_f, cols_id, S1)
    log('PART', PART, 'models trained:', list(models))
    del Xe; gc.collect()
''')
    assert 'Xc' not in s and 'CUR = []' not in s
    ast.parse(s)
    return s


if __name__ == '__main__':
    for part, smoke, dst in (('a', False, 't2_ogfeat_v27a.py'), ('b', False, 't2_ogfeat_v27b.py'), ('a', True, 't2_ogfeat_v27a_smoke.py')):
        src = build(part, smoke)
        open(f'{ROOT}/kaggle_jobs/{dst}', 'w').write(src)
        print('written', dst, len(src.splitlines()), 'lines | static', len(STATIC), 'step', len(STEP))
