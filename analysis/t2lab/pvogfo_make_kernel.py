"""pvogfo (EXP_T2_PVOCC, 2026-10-30; lab analysis/t2lab/pvocc_*.py, PRE-REGISTERED rule in pvocc_lib.py): generate pick A's April ongoing
kernel WITH the 13 fs4occ occupancy columns from the shipped kaggle_jobs/t2_ongoing_pvogf.py (kernel tfb-t2ogpvogf) by ASSERTED text edits.
  * occ_calib / occ_features: copied verbatim from kaggle_jobs/t2_occ_v26.py (the fs4occ kernel tfb-t2occ; identical in t2_ogfeat_v27a.py);
  * load_panel / make_rows / load_split: the v26 OCC lines verbatim (train cube + calibration from the 273 train days; history cells as
    Vh, slot T where the slot-T speed is visible), except that the occupancy columns are appended AFTER the ogfeat step columns, so the
    kernel column order is the lab's (172 pvogf columns + OCC_COLS);
  * masked_month(occ=True) reads the occupancy of every masked-month cell that carries a speed value (= export cube_o) and masks it to
    the all-finite masked speed; march_rows uses it with the make_rows convention (history cells as Vh, slot T where VT is finite);
  * VETO_O: also train pvoO (pvo columns + occupancy: npO / idO / fullO) and write vO = pvogfvo (pvogfO unless it adds > VETO_G cells
    outside the queue at T vs pvoO, else pvoO), computed inside the kernel;
  * LightGBM deterministic=True, force_row_wise=True (repeatable rerun at award review); the float training matrix is freed after the
    Dataset is constructed (memory), which leaves the model unchanged (checked by pvogfo_kernel_test.py);
  * the veto vs the attached tfb-t2ogpvaf pvo file is dropped; CONTROL lines compare every output with pick A's current April ongoing rows
    (tfb-t2ogpvogf queue_submission_pvogfv.csv).
Outputs (written next to the source):
  kaggle_jobs/t2_ongoing_pvogfo.py        full run (kernel tfb-t2ogpvogfo; sources tfb-t2v22af, tfb-t2ogpvogf)
  kaggle_jobs/t2_ongoing_pvogfo_smoke.py  SMOKE copy (2 panels, 90 train days, 60 trees)
Usage: .venv/bin/python analysis/t2lab/pvogfo_make_kernel.py veto=0|1 [lab=<one-line lab summary>] [out=<dir>]"""
import ast, sys

ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
SRC = f'{ROOT}/kaggle_jobs/t2_ongoing_pvogf.py'
V26 = f'{ROOT}/kaggle_jobs/t2_occ_v26.py'
kw = dict(a.split('=') for a in sys.argv[1:])
VETO_O = bool(int(kw.get('veto', 0)))
LAB = kw.get('lab', '')          # one-line lab summary for the docstring (filled from results/pvocc/verdict.json by the caller)


def fn_src(path, name):
    src = open(path).read()
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise KeyError(name)


OCC_FNS = fn_src(V26, 'occ_calib') + '\n\n\n' + fn_src(V26, 'occ_features') + '\n'


def build(smoke):
    s = open(SRC).read()

    def rep(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (s.count(old), old[:160])
        s = s.replace(old, new)

    # ---- 1. docstring
    rep('"""pvogf (track pvogf, E2 2026-10-09; lab analysis/t2lab/pvogf_*.py, generator analysis/t2lab/pvogf_make_kernel.py) = pvaf (tfb-t2ogpvaf,\n',
        '"""pvogfo (EXP_T2_PVOCC 2026-10-30; lab analysis/t2lab/pvocc_*.py, PRE-REGISTERED rule in pvocc_lib.py, generator analysis/t2lab/\n'
        "pvogfo_make_kernel.py) = pvogf (tfb-t2ogpvogf, pick A's April ongoing) + the 13 OCCUPANCY columns of fs4occ (tfb-t2occ, t2_occ_v26.py:\n"
        'o_last, or_last, or_T, or_mean3, or_slope, or_max12, or_n-3..or_n+3, dens_dev; or = o / ocrit, per-link calibration from the 273\n'
        'train days only) in npB / idB / fullB, appended after the 172 pvogf columns.  Occupancy uses exactly the cells of the speed features:\n'
        'train F03 rows = the train cube (history cells as Vh, slot T where the simulated slot-T speed is visible), released-March rows = the\n'
        'masked March month (every cell with a speed value, slot T where the all-finite masked speed is finite), April windows =\n'
        'window_history + the masked layer at T (load_split).  Nothing after T: an April window uses train + all of March + its own history\n'
        '<= T and the slot-T row, and the April days < d in the fullB prior, so its prediction is unchanged when every row stamped after its T\n'
        'is removed.  LightGBM deterministic=True, force_row_wise=True (repeatable rerun).  VETO_O = True also trains pvoO (pvo + occupancy)\n'
        'and writes vO (pvogfvo: pvogfO unless it adds > VETO_G cells outside the queue at T vs pvoO).\n'
        + (f'Lab: {LAB}\n' if LAB else '') +
        'Outputs: queue_submission_pvogfo.csv (0.25 npB + 0.25 idB + 0.5 fullB), queue_submission_pvogfo_npid.csv (+ pvogfvo.csv and pvoo.csv\n'
        'if VETO_O), t2pvogfo_probs.npz (private member probabilities, window ids, qlast) (+ t2pvogfo_veto_windows.csv).  Validation ongoing\n'
        'rows are copied from tfb-t2v22af fs4af (as pvogf; the swap kernel takes only the private ongoing rows); onset rows 0.  CONTROL lines vs\n'
        "pick A's current April ongoing rows (tfb-t2ogpvogf queue_submission_pvogfv.csv).\n\n"
        'pvogf notes: pvogf (track pvogf, E2 2026-10-09; lab analysis/t2lab/pvogf_*.py, generator analysis/t2lab/pvogf_make_kernel.py) = pvaf (tfb-t2ogpvaf,\n')
    rep('import glob, json, os, time, gc\n', 'import glob, json, os, time, gc, warnings                  # pvogfo: warnings for occ_features (verbatim v26)\n')
    if smoke:
        rep('SMOKE = False\n', 'SMOKE = True\n')
    # ---- 2. constants
    rep("ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)\n",
        "ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)\n"
        "OCC = True                                   # pvogfo: occupancy features (fs4occ, t2_occ_v26.py; same history / slot-T inputs as the speed features)\n"
        "NB = (-3, -2, -1, 1, 2, 3)\n"
        "OCC_COLS = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in NB] + ['dens_dev']\n"
        f"VETO_O = {VETO_O!r}                               # pvogfo: also train pvoO (pvo + occupancy) and write vO = pvogfvo\n")
    # ---- 3. load_panel (v26 OCC lines)
    rep("    Vx = V.copy() if ALLFIN else None; Fx = V.copy() if ALLFIN else None       # ineligible cells that carry a value\n",
        "    Vx = V.copy() if ALLFIN else None; Fx = V.copy() if ALLFIN else None       # ineligible cells that carry a value\n"
        "    O = V.copy() if OCC else None                                          # occupancy of every cell with a speed value\n")
    rep("        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])\n        x = x[x.speed_kmh.notna()]\n",
        "        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'] + (['occupancy'] if OCC else []))\n        x = x[x.speed_kmh.notna()]\n")
    rep("dropna(subset=['l'])\n        el = (x.is_score_eligible == 1).to_numpy()\n",
        "dropna(subset=['l'])\n        if OCC:\n"
        "            g = x[x.occupancy.notna()].groupby(['t', 'l']).occupancy.mean().reset_index()\n"
        "            O[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.occupancy.to_numpy()\n"
        "        el = (x.is_score_eligible == 1).to_numpy()\n")
    rep("        log(panel, 'all-finite inputs: eligible share', round(float(E.mean()), 4), '| + ineligible valued', round(float(np.isfinite(Vx).mean()), 4))\n    return P\n",
        "        log(panel, 'all-finite inputs: eligible share', round(float(E.mean()), 4), '| + ineligible valued', round(float(np.isfinite(Vx).mean()), 4))\n"
        "    if OCC:\n"
        "        Va = P.get('Vf', V)\n"
        "        O = np.where(np.isfinite(Va), O, np.nan).astype(np.float32)          # same visibility as the speed feature cube\n"
        "        P['O'] = O\n"
        "        P['ocrit'], P['occ_c'], n_est = occ_calib(Va, P.get('Ff', Fq), O, cut, vf, lanes)\n"
        "        log(panel, 'occupancy: finite share', round(float(np.isfinite(O).mean()), 4), '| speed-cube share', round(float(np.isfinite(Va).mean()), 4),\n"
        "            '| ocrit estimated links', n_est, 'of', L, 'q10/50/90', np.quantile(P['ocrit'], [0.1, 0.5, 0.9]).round(3).tolist())\n"
        "    return P\n\n\n" + OCC_FNS)
    # ---- 4. make_rows (v26 OCC block; occupancy appended after the ogfeat step columns)
    rep("    Fe.update({k: OGX[k] for k in OG_STATIC})\n    W, L = qlast.shape\n    rows = []\n",
        "    Fe.update({k: OGX[k] for k in OG_STATIC})\n"
        "    occ = None\n"
        "    if 'O' in P:                                                           # occupancy: history cells as Vh, slot T where VT is visible\n"
        "        Oh = P['O'][d[:, None], idx]; OT = np.where(np.isfinite(VT), P['O'][d, t], np.nan).astype(np.float32)\n"
        "        occ = occ_features(Oh, OT, Vh, Fh, P)\n"
        "    W, L = qlast.shape\n    rows = []\n")
    STEP = "        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # pvogf: step-dependent columns\n"
    OCC_UPD = ("        if occ is not None:                                                # pvogfo: occupancy appended after the 172 pvogf columns (lab order)\n"
               "            feats.update(occ)\n")
    rep(STEP + "        lab = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None\n",
        STEP + OCC_UPD + "        lab = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None\n")
    # ---- 5. load_split (v26 OCC lines)
    rep("    Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy()\n    wpos = ",
        "    Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy(); Oh = Vh.copy()\n    wpos = ")
    rep("    Vh[g.w, g.s, g.l] = g.speed_kmh; Fh[g.w, g.s, g.l] = g.flow_vph\n",
        "    Vh[g.w, g.s, g.l] = g.speed_kmh; Fh[g.w, g.s, g.l] = g.flow_vph\n"
        "    if OCC:                                                                # occupancy of the same history cells\n"
        "        go = hh[hh.occupancy.notna()].groupby(['w', 's', 'l']).occupancy.mean().reset_index()\n"
        "        Oh[go.w, go.s, go.l] = go.occupancy\n"
        "        Oh = np.where(np.isfinite(Vh), Oh, np.nan).astype(np.float32)\n")
    rep("    VT = np.full((W, L), np.nan, np.float32)\n    for date, grp in wi.groupby('date'):\n",
        "    VT = np.full((W, L), np.nan, np.float32); OT = VT.copy()\n    for date, grp in wi.groupby('date'):\n")
    rep("        m = pd.read_parquet(fs[0], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])\n",
        "        m = pd.read_parquet(fs[0], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'] + (['occupancy'] if OCC else []))\n")
    rep("            x = x[x.link_id.isin(P['li'])].groupby('link_id').speed_kmh.mean()\n"
        "            VT[wpos[r.window_id], x.index.map(P['li']).to_numpy()] = x.to_numpy()\n",
        "            x = x[x.link_id.isin(P['li'])]\n"
        "            xs = x.groupby('link_id').speed_kmh.mean()\n"
        "            VT[wpos[r.window_id], xs.index.map(P['li']).to_numpy()] = xs.to_numpy()\n"
        "            if OCC:\n"
        "                xo = x[x.occupancy.notna()].groupby('link_id').occupancy.mean()\n"
        "                OT[wpos[r.window_id], xo.index.map(P['li']).to_numpy()] = xo.to_numpy()\n")
    rep("    Fe.update({k: OGX[k] for k in OG_STATIC})\n    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])\n",
        "    Fe.update({k: OGX[k] for k in OG_STATIC})\n"
        "    occ = None\n"
        "    if OCC:\n"
        "        OT = np.where(np.isfinite(VT), OT, np.nan).astype(np.float32)\n"
        "        log(panel, split, 'occupancy: history finite share', round(float(np.isfinite(Oh).mean()), 3), '(speed', round(float(np.isfinite(Vh).mean()), 3),\n"
        "            ') | slot-T finite share', round(float(np.isfinite(OT).mean()), 3), '(speed', round(float(np.isfinite(VT).mean()), 3), ')')\n"
        "        occ = occ_features(Oh, OT, Vh, Fh, P)\n"
        "    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])\n")
    rep(STEP + "        R['rows'].append((h, feats, None, None))\n", STEP + OCC_UPD + "        R['rows'].append((h, feats, None, None))\n")
    # ---- 6. masked month occupancy (export cube_o semantics) + scenario + march_rows
    rep("def masked_month(panel, split, li):\n", "def masked_month(panel, split, li, occ=False):\n")
    rep("    Vx = V.copy(); Fx = V.copy()                                           # ineligible cells that carry a value (never masked)\n",
        "    Vx = V.copy(); Fx = V.copy()                                           # ineligible cells that carry a value (never masked)\n"
        "    O = V.copy() if occ else None                                          # pvogfo: occupancy of every cell with a speed value\n")
    rep("        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])\n        ts = pd.to_datetime(x.timestamp, utc=True)\n",
        "        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'] + (['occupancy'] if occ else []))\n"
        "        ts = pd.to_datetime(x.timestamp, utc=True)\n")
    rep("        Fx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()\n    if ALLFIN:\n",
        "        Fx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()\n"
        "        if occ:                                                            # pvogfo: rows with a speed and an occupancy value\n"
        "            g = x[x.speed_kmh.notna() & x.occupancy.notna()].groupby(['t', 'l']).occupancy.mean().reset_index()\n"
        "            O[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.occupancy.to_numpy()\n"
        "    if ALLFIN:\n")
    rep("        Vx, Fx = V, Fq\n    return days, V, Fq, Em, Vx, Fx\n",
        "        Vx, Fx = V, Fq\n"
        "    if occ:                                                                # pvogfo: same visibility as the all-finite feature cube\n"
        "        O = np.where(np.isfinite(Vx), O, np.nan).astype(np.float32)\n"
        "    return days, V, Fq, Em, Vx, Fx, O\n")
    rep("    dv, Vv, Fv, Ev, Vfv, Ffv = masked_month(P['panel'], 'validation', P['li']); dp, Vp, _, _, _, _ = masked_month(P['panel'], 'private', P['li'])\n",
        "    dv, Vv, Fv, Ev, Vfv, Ffv, Ov = masked_month(P['panel'], 'validation', P['li'], occ=OCC)\n"
        "    dp, Vp, _, _, _, _, _ = masked_month(P['panel'], 'private', P['li'])\n")
    rep("    M = dict(days=dv, V=Vv, Fq=Fv, Em=Ev, Vf=Vfv, Ff=Ffv, ON=ONv, OFF=OFFv, dow=pd.to_datetime(pd.Series(dv)).dt.dayofweek.to_numpy())\n",
        "    M = dict(days=dv, V=Vv, Fq=Fv, Em=Ev, Vf=Vfv, Ff=Ffv, ON=ONv, OFF=OFFv, dow=pd.to_datetime(pd.Series(dv)).dt.dayofweek.to_numpy(), O=Ov)\n")
    rep("    Fe.update({k: OGX[k] for k in OG_STATIC})\n    W_, L_ = qlast.shape\n",
        "    Fe.update({k: OGX[k] for k in OG_STATIC})\n"
        "    occ = None\n"
        "    if M.get('O') is not None:                                             # pvogfo: occupancy as make_rows (history cells as Vh, slot T where VT is finite)\n"
        "        Oh = M['O'][d[:, None], idx]; OT = np.where(np.isfinite(VT), M['O'][d, t], np.nan).astype(np.float32)\n"
        "        occ = occ_features(Oh, OT, Vh, Fh, P)\n"
        "    W_, L_ = qlast.shape\n")
    rep(STEP + "        R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))\n",
        STEP + OCC_UPD + "        R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))\n")
    # ---- 7. config: LightGBM reproducibility, CONTROL reference = pick A's current rows, in-kernel veto helper
    rep("P_PV = dict(PARAMS, num_leaves=511, min_data_in_leaf=40, learning_rate=0.05)\n",
        "P_PV = dict(PARAMS, num_leaves=511, min_data_in_leaf=40, learning_rate=0.05, deterministic=True, force_row_wise=True)   # pvogfo: repeatable\n")
    rep("REF_PVO = ('tfb-t2ogpvaf', 'queue_submission_pvo.csv')   # pvogf: pick A's current April ongoing rows (veto reference + CONTROL)\n",
        "REF_PVO = ('tfb-t2ogpvogf', 'queue_submission_pvogfv.csv')   # pvogfo: pick A's CURRENT April ongoing rows (CONTROL only, optional)\n")
    rep("            log('REF pvo:', f[0], len(r), 'rows')\n", "            log('REF pick A (pvogfv):', f[0], len(r), 'rows')\n")
    rep("            log('REF pvo: not attached (no veto variant, no CONTROL vs pvo)')\n", "            log('REF pick A (pvogfv): not attached (no CONTROL lines)')\n")
    rep("def main():\n",
        "def veto_arrays(A, B, qlast, is_og, g=None):\n"
        "    \"\"\"pvogfo: per ongoing window B unless it adds more than VETO_G cells outside the queue at T (qlast, all 6 steps) vs A, then A\n"
        "    (= the lab's pvogf_lib.veto).  Returns (decisions [W, 6, L], added cells per window).\"\"\"\n"
        "    g = VETO_G if g is None else g\n"
        "    q = np.repeat(qlast[:, None, :], A.shape[1], 1)\n"
        "    add = (B & ~A & ~q).sum((1, 2))\n"
        "    vet = is_og & (add > g)\n"
        "    return np.where(vet[:, None, None], A, B), add\n\n\n"
        "def main():\n")
    # ---- 8. main(): drop O after the F03 rows, column checks, members, private predictions, outputs
    rep("        for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'trainw', 'Vf', 'Ff'):\n",
        "        for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'trainw', 'Vf', 'Ff', 'O'):\n")
    rep("    assert 'tod' not in cols_np and 'dow' not in cols_id\n",
        "    assert 'tod' not in cols_np and 'dow' not in cols_id\n"
        "    assert cols[-len(OCC_COLS):] == OCC_COLS and len(cols) == 172 + len(OCC_COLS), 'occupancy columns missing / misplaced'   # pvogfo\n"
        "    assert all(c in cols_np and c in cols_id for c in OCC_COLS)\n"
        "    cols_b = [x for x in cols if x not in OG_STATIC and x not in OG_STEP]; ix_b = [cols.index(x) for x in cols_b]      # pvogfo: pvoO columns\n"
        "    cols_bnp = [x for x in cols_b if x not in PRIOR_COLS]; ix_bnp = [cols.index(x) for x in cols_bnp]\n"
        "    cols_bid = [x for x in cols_b if x not in ID_DROP]; ix_bid = [cols.index(x) for x in cols_bid]\n"
        "    log(f'pvogfo: occupancy columns {len(OCC_COLS)} (finite share or_last, first 200k F03 rows {float(np.isfinite(Xe[:200000, cols.index(\"or_last\")]).mean()):.3f})'\n"
        "        f' | pvoO members {VETO_O}: np {len(cols_bnp)}, id {len(cols_bid)}, full {len(cols_b)}')\n")
    rep("""    for name, getX in (('npB', lambda: np.concatenate([Xe[:, ix_np], Xm[:, ix_np]])),
                       ('idB', lambda: np.concatenate([Xe[:, ix_id], Xm[:, ix_id]])),
                       ('fullB', lambda: np.concatenate([Xe, Xmb]))):
        t1 = time.time(); X = getX()
        models[name] = lgb.train(P_PV, lgb.Dataset(X, y_all, weight=w_all), N_PV)
        del X; gc.collect()
""", """    MEMB = [('npB', lambda: np.concatenate([Xe[:, ix_np], Xm[:, ix_np]])),
            ('idB', lambda: np.concatenate([Xe[:, ix_id], Xm[:, ix_id]])),
            ('fullB', lambda: np.concatenate([Xe, Xmb]))]
    if VETO_O:                                                                         # pvogfo: pvoO members (veto reference of vO)
        MEMB += [('npO', lambda: np.concatenate([Xe[:, ix_bnp], Xm[:, ix_bnp]])),
                 ('idO', lambda: np.concatenate([Xe[:, ix_bid], Xm[:, ix_bid]])),
                 ('fullO', lambda: np.concatenate([Xe[:, ix_b], Xmb[:, ix_b]]))]
    for name, getX in MEMB:
        t1 = time.time(); X = getX()
        ds = lgb.Dataset(X, y_all, weight=w_all, params=P_PV); ds.construct(); del X; gc.collect()   # pvogfo: float matrix freed before boosting
        models[name] = lgb.train(P_PV, ds, N_PV)
        del ds; gc.collect()
""")
    rep("    subs = []; stats = []; PROBS = {}; VETO = dict(ok=pvo_rows() is not None, info=[])\n",
        "    subs = []; stats = []; PROBS = {}; VETO = dict(ok=VETO_O, info=[]); pvo_rows()\n")
    rep("            preds = {n: np.zeros((W, F, L), bool) for n in ('pvogf', 'pvogf_npid')}\n",
        "            preds = {n: np.zeros((W, F, L), bool) for n in ('pvogfo', 'pvogfo_npid') + (('pvoo',) if VETO_O else ())}\n")
    rep("                for nm_, p_ in (('npB', pn), ('idB', pi), ('fullB', pf)):                      # pvogf: member probabilities\n",
        "                MP = [('npB', pn), ('idB', pi), ('fullB', pf)]\n"
        "                if VETO_O:                                                                     # pvogfo: pvoO members\n"
        "                    qn = models['npO'].predict(X[:, ix_bnp]); qi = models['idO'].predict(X[:, ix_bid]); qf = models['fullO'].predict(Xb[:, ix_b])\n"
        "                    MP += [('npO', qn), ('idO', qi), ('fullO', qf)]\n"
        "                for nm_, p_ in MP:                                                             # pvogf: member probabilities\n")
    rep("                for name, p in (('pvogf', 0.25 * pn + 0.25 * pi + 0.5 * pf), ('pvogf_npid', 0.5 * (pn + pi))):\n",
        "                for name, p in (('pvogfo', 0.25 * pn + 0.25 * pi + 0.5 * pf), ('pvogfo_npid', 0.5 * (pn + pi))) + \\\n"
        "                        ((('pvoo', 0.25 * qn + 0.25 * qi + 0.5 * qf),) if VETO_O else ()):\n")
    rep("                                  size_pvogf=float(preds['pvogf'][is_og].sum((1, 2)).mean() / F),\n"
        "                                  size_npid=float(preds['pvogf_npid'][is_og].sum((1, 2)).mean() / F)))\n",
        "                                  size_pvogfo=float(preds['pvogfo'][is_og].sum((1, 2)).mean() / F),\n"
        "                                  size_npid=float(preds['pvogfo_npid'][is_og].sum((1, 2)).mean() / F)))\n")
    rep("""            tpl['q_pvogfv'] = tpl['q_pvogf']                                       # pvogf: growth veto vs pvo (private ongoing windows)
            if split == 'private' and VETO['ok']:
                try:
                    tpl['q_pvogfv'], vinfo = veto_vs_pvo(tpl, preds['pvogf'], R['qlast'], is_og, wi)
                    VETO['info'].append(vinfo.assign(panel=panel))
                except Exception as ex:
                    VETO['ok'] = False; log('VETO vs pvo failed for', panel, type(ex).__name__, str(ex)[:200])
            tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).astype(bool); tpl['split'] = split
            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split', 'q_pvogf', 'q_pvogf_npid', 'q_pvogfv']])
""", """            tpl['q_pvogfvo'] = tpl['q_pvogfo']                                     # pvogfo: vO = growth veto of pvogfO vs pvoO (private ongoing windows)
            if split == 'private' and VETO_O:
                vdec, vadd = veto_arrays(preds['pvoo'], preds['pvogfo'], R['qlast'], is_og)
                tpl['q_pvogfvo'] = vdec[tpl.w.astype(int), tpl.h.astype(int) - 1, tpl.l.astype(int)].astype(int)
                VETO['info'].append(pd.DataFrame(dict(window_id=wi.window_id.to_numpy(), addout=vadd, veto=is_og & (vadd > VETO_G),
                                                      n_pvoo=preds['pvoo'].sum((1, 2)), n_pvogfo=preds['pvogfo'].sum((1, 2)),
                                                      cells_changed=(preds['pvoo'] != preds['pvogfo']).sum((1, 2))))[is_og].assign(panel=panel))
            tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).astype(bool); tpl['split'] = split
            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split'] + [f'q_{n}' for n in preds] + ['q_pvogfvo']])
""")
    rep("    OUTS = ('pvogf', 'pvogf_npid') + (('pvogfv',) if VETO['ok'] else ())\n",
        "    OUTS = ('pvogfo', 'pvogfo_npid') + (('pvogfvo', 'pvoo') if VETO_O else ())\n")
    rep("    # pvogf CONTROL: every output vs tfb-t2ogpvaf queue_submission_pvo.csv, per split (validation rows: both are fs4af copies)\n",
        "    # pvogfo CONTROL: every output vs pick A's current rows tfb-t2ogpvogf queue_submission_pvogfv.csv, per split (validation: both fs4af copies)\n")
    rep("            log(f'CONTROL pvo reference rows matched {int(mm.q_ref.notna().sum())} of {len(mm)}')\n",
        "            log(f'CONTROL pick-A reference rows matched {int(mm.q_ref.notna().sum())} of {len(mm)}')\n")
    rep("                    log(f'CONTROL {name} vs pvo | {sp_}: rows changed {int((a != b).sum())}/{len(a)} ({float((a != b).mean()):.4%}) | windows changed'\n"
        "                        f' {int(wch.sum())}/{len(wch)} | positives {int(a.sum())} vs pvo {int(b.sum())}'\n",
        "                    log(f'CONTROL {name} vs pick A pvogfv | {sp_}: rows changed {int((a != b).sum())}/{len(a)} ({float((a != b).mean()):.4%}) | windows changed'\n"
        "                        f' {int(wch.sum())}/{len(wch)} | positives {int(a.sum())} vs pick A {int(b.sum())}'\n")
    rep("            vi = pd.concat(VETO['info'], ignore_index=True); vi.to_csv(f'{OUT}/t2pvogf_veto_windows.csv', index=False)\n"
        "            log(f'VETO (> {VETO_G} cells outside the queue at T vs pvo): private ongoing windows vetoed {int(vi.veto.sum())}/{len(vi)}'\n"
        "                f' {sorted(vi.window_id[vi.veto].tolist())} | pvogf windows changed vs pvo {int((vi.cells_changed > 0).sum())}')\n",
        "            vi = pd.concat(VETO['info'], ignore_index=True); vi.to_csv(f'{OUT}/t2pvogfo_veto_windows.csv', index=False)\n"
        "            log(f'VETO vO (> {VETO_G} cells outside the queue at T vs pvoO): private ongoing windows vetoed {int(vi.veto.sum())}/{len(vi)}'\n"
        "                f' {sorted(vi.window_id[vi.veto].tolist())} | pvogfO windows changed vs pvoO {int((vi.cells_changed > 0).sum())}')\n")
    rep("    except Exception as ex:\n        log('CONTROL vs pvo skipped:', type(ex).__name__, str(ex)[:200])\n    np.savez_compressed(f'{OUT}/t2pvogf_probs.npz', **PROBS)\n",
        "    except Exception as ex:\n        log('CONTROL vs pick A skipped:', type(ex).__name__, str(ex)[:200])\n    np.savez_compressed(f'{OUT}/t2pvogfo_probs.npz', **PROBS)\n")
    rep("            a, b = mm.out_pvogf.to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1\n"
        "            log(f'CONTROL chain-base ongoing vs pvogf ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')\n",
        "            a, b = mm.out_pvogfo.to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1\n"
        "            log(f'CONTROL chain-base ongoing vs pvogfo ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')\n")
    rep("    log('PVOGF DONE')\n", "    log('PVOGFO DONE')\n")
    for bad in ("'pvogf'", "'pvogf_npid'", 'out_pvogf.', "q_pvogf'", "'q_pvogfv'", '= veto_vs_pvo('):
        assert bad not in s, bad
    ast.parse(s)
    return s


if __name__ == '__main__':
    ODIR = kw.get('out', f'{ROOT}/kaggle_jobs')      # out=<dir> writes a test copy elsewhere (scratchpad)
    for smoke, dst in ((False, 't2_ongoing_pvogfo.py'), (True, 't2_ongoing_pvogfo_smoke.py')):
        src = build(smoke)
        open(f'{ODIR}/{dst}', 'w').write(src)
        print('written', dst, len(src.splitlines()), 'lines | VETO_O', VETO_O)
