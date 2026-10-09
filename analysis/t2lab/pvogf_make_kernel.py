"""pvogf (2026-10-09, E2 of analysis/LEVERS_2026-10-07.md; lab: analysis/t2lab/pvogf_*.py): generate the pick-A April ongoing kernel
with the ogfeat 'all2' columns from kaggle_jobs/t2_ongoing_pvaf.py (kernel tfb-t2ogpvaf = pvo: F03 np / id / full bag + released-March
faithful windows x WM 5, all-finite inputs, full member fed the March + April days < d scenario prior) by ASSERTED text edits.
The ogfeat feature block is copied verbatim from analysis/t2lab/ogfeat_lib.py (between the 'kernel-portable block' markers; only the
warnings context is replaced by np.errstate), exactly as analysis/t2lab/ogfeat_make_kernel_v27.py did for tfb-t2ogfa/b.
Feature set = lab arm all2: names.json + names2.json static columns (98) + the 5 kx step columns; no tod / dow added.  The new columns
are computed from exactly the arrays pvaf feeds window_features in its three row builders (train F03 rows: make_rows; released-March
rows: march_rows, a function now; released windows: load_split) and appended to npB, idB and fullB.
Outputs (written next to the source):
  kaggle_jobs/t2_ongoing_pvogf.py        full run (kernel tfb-t2ogpvogf)
  kaggle_jobs/t2_ongoing_pvogf_smoke.py  SMOKE copy (2 panels, 90 train days, 60 trees)
Usage: .venv/bin/python analysis/t2lab/pvogf_make_kernel.py"""
import ast, json

ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
SRC = f'{ROOT}/kaggle_jobs/t2_ongoing_pvaf.py'
LIB = f'{ROOT}/analysis/t2lab/ogfeat_lib.py'
ODIR = f'{ROOT}/data/t2local/cache/ogfeat'
STATIC = json.load(open(f'{ODIR}/names.json')) + json.load(open(f'{ODIR}/names2.json'))
STEP = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']
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

OGX_LINE = "OGX = og_extra(P, {v}); OGX.update(og_extra2(P, {v}))          # pvogf: ogfeat columns\n"
STEP_LINE = "feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # pvogf: step-dependent columns\n"

MARCH_FN = '''def march_rows(P, M, Qm, Emv, d, t, cols):
    """pvogf: the released-March training rows of pvaf main() as a function, + the ogfeat columns.  Windows (d, t) of the masked March
    month: all-finite masked history + all-finite masked slot T (Vf / Ff), visible labels Qm / Emv, all-train prior in the prior columns."""
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh, Fh, VT = M['Vf'][d[:, None], idx], M['Ff'][d[:, None], idx], M['Vf'][d, t]
    Fe, qlast = window_features(P, Vh, Fh, t, M['dow'][d], P['prior_all'], VT)
    add_ramp(Fe, M['ON'], M['OFF'], d, t, P['cap'])
    ''' + OGX_LINE.format(v='Vh, Fh, VT, Fe') + '''    Fe.update({k: OGX[k] for k in OG_STATIC})
    W_, L_ = qlast.shape
    R = dict(d=d, t=t, cond=np.array([ONGOING] * W_), qlast=qlast, W=W_, L=L_, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W_, L_), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], M['dow'][d], t, h)
        ''' + STEP_LINE + '''        R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))
    X, y, e, meta, _ = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    return R, X, y, e, meta


PVO_CACHE = {}


def pvo_rows():
    """pvogf: tfb-t2ogpvaf queue_submission_pvo.csv (pick A's current April ongoing rows), parsed once; None if not attached."""
    if 'df' not in PVO_CACHE:
        f = sorted(glob.glob(f'/kaggle/input/**/{REF_PVO[0]}/**/{REF_PVO[1]}', recursive=True))
        PVO_CACHE['df'] = None
        if f:
            r = pd.read_csv(f[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
            r['ts'] = pd.to_datetime(r.timestamp, utc=True)
            PVO_CACHE['df'] = r[['window_id', 'ts', 'link_id', 'queue_pred']].rename(columns={'queue_pred': 'q_ref'})
            log('REF pvo:', f[0], len(r), 'rows')
        else:
            log('REF pvo: not attached (no veto variant, no CONTROL vs pvo)')
    return PVO_CACHE['df']


def veto_vs_pvo(tpl, B, qlast, is_og, wi, g=None):
    """pvogf: per ongoing window the pvogf decision B unless it adds more than VETO_G cells outside the queue at T (qlast, all 6 steps)
    compared with pvo; then the pvo decision.  Returns (template column, per-window info frame)."""
    g = VETO_G if g is None else g
    ref = pvo_rows()
    assert ref is not None, 'pvo reference not attached'
    mm = tpl.assign(ts=pd.to_datetime(tpl.timestamp, utc=True))[['window_id', 'ts', 'link_id']].merge(
        ref, on=['window_id', 'ts', 'link_id'], how='left', validate='one_to_one')
    assert len(mm) == len(tpl) and mm.q_ref.notna().all(), 'pvo rows missing'
    W, F_, L = B.shape
    w_, h_, l_ = tpl.w.to_numpy().astype(int), tpl.h.to_numpy().astype(int) - 1, tpl.l.to_numpy().astype(int)
    A = np.zeros((W, F_, L), bool); A[w_, h_, l_] = mm.q_ref.to_numpy().astype(int) == 1
    q = np.repeat(qlast[:, None, :], F_, 1)
    add = (B & ~A & ~q).sum((1, 2))
    vet = is_og & (add > g)
    out = np.where(vet[:, None, None], A, B)
    info = pd.DataFrame(dict(window_id=wi.window_id.to_numpy(), addout=add, veto=vet, n_pvo=A.sum((1, 2)), n_pvogf=B.sum((1, 2)),
                             cells_changed=(A != B).sum((1, 2))))[is_og]
    return out[w_, h_, l_].astype(int), info


'''


def build(smoke):
    s = open(SRC).read()

    def rep(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (s.count(old), old[:120])
        s = s.replace(old, new)

    # ---- 1. docstring
    rep('"""Task 2 ONGOING for the PRIVATE month with same-scenario data AND all-finite inputs (ogpvaf) = t2_ongoing_pv.py with every finite\n',
        '"""pvogf (track pvogf, E2 2026-10-09; lab analysis/t2lab/pvogf_*.py, generator analysis/t2lab/pvogf_make_kernel.py) = pvaf (tfb-t2ogpvaf,\n'
        "pick A's April ongoing pvo) + the 98 + 5 OGFEAT columns of lab arm all2 (g1b corridor context, g2 per-link kinematics + kx step\n"
        'extrapolations, g3 demand / supply at the queue ends, g1c corridor flow / speed level and trend, g2b per-link volatility / smoothed\n'
        'state; no tod / dow) in all three members npB / idB / fullB.  The new columns use exactly the inputs pvaf feeds window_features in its\n'
        'three row builders: the all-finite 60-min history [T-60, T), the all-finite (masked) slot-T value and the ramp columns of add_ramp\n'
        '(slots < T).  Nothing after T.  April windows use only March + the April days before d (scenario prior), as pvaf.\n'
        'Feature block copied verbatim from analysis/t2lab/ogfeat_lib.py.  The released-March rows are built by march_rows() (same code as pvaf).\n'
        'Outputs: queue_submission_pvogf.csv (0.25 npB + 0.25 idB + 0.5 fullB), queue_submission_pvogf_npid.csv, queue_submission_pvogfv.csv\n'
        '(growth veto: a private ongoing window keeps the pvo rows of tfb-t2ogpvaf if pvogf adds more than VETO_G cells outside the queue at T;\n'
        'written only if tfb-t2ogpvaf is attached), t2pvogf_probs.npz (private member probabilities, window ids, qlast),\n'
        't2pvogf_veto_windows.csv.  Validation ongoing rows are copied from tfb-t2v22af fs4af (as pvaf); onset rows 0.  CONTROL lines vs\n'
        'tfb-t2ogpvaf queue_submission_pvo.csv (rows / windows changed per split).\n\n'
        'pvaf notes: Task 2 ONGOING for the PRIVATE month with same-scenario data AND all-finite inputs (ogpvaf) = t2_ongoing_pv.py with every finite\n')
    if smoke:
        rep('SMOKE = False\n', 'SMOKE = True\n')
    # ---- 2. constants + the portable feature block
    rep("ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)\n",
        "ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)\n"
        f"OG_STATIC = {STATIC!r}\n"
        f"OG_STEP = {STEP!r}\n"
        "CAPD = 30.0\nBIG = 10 ** 6\n")
    rep("def established(Qh):\n", BLOCK + "\n\ndef established(Qh):\n")
    # ---- 3. make_rows (train F03 rows)
    rep("    Fe, qlast = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)\n    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n",
        "    Fe, qlast = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)\n    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n"
        "    " + OGX_LINE.format(v='Vh, Fh, VT, Fe') + "    Fe.update({k: OGX[k] for k in OG_STATIC})\n")
    rep("        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)\n"
        "        lab = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None\n",
        "        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)\n"
        "        " + STEP_LINE +
        "        lab = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None\n")
    # ---- 4. load_split (released windows)
    rep("    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))\n",
        "    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))\n"
        "    " + OGX_LINE.format(v='Vh, Fh, VT, Fe') + "    Fe.update({k: OGX[k] for k in OG_STATIC})\n")
    rep("        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)\n"
        "        R['rows'].append((h, feats, None, None))\n",
        "        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)\n"
        "        " + STEP_LINE +
        "        R['rows'].append((h, feats, None, None))\n")
    # ---- 5. config: pvo reference + veto threshold; new functions before main()
    rep("REF_BASE = ('tfb-f5', 'submission_F5.csv')             # current chain base (CONTROL)\n",
        "REF_BASE = ('tfb-f5', 'submission_F5.csv')             # current chain base (CONTROL)\n"
        "REF_PVO = ('tfb-t2ogpvaf', 'queue_submission_pvo.csv')   # pvogf: pick A's current April ongoing rows (veto reference + CONTROL)\n"
        "VETO_G = 10                                 # pvogf: growth veto threshold (cells outside the queue at T, all 6 steps)\n")
    rep("def main():\n", MARCH_FN + "def main():\n")
    # ---- 6. main(): released-March rows through march_rows (ogfeat inside)
    rep('''        idx = t[:, None] + np.arange(-H, 0)[None]
        Fe, qlast = window_features(P, M['Vf'][d[:, None], idx], M['Ff'][d[:, None], idx], t, M['dow'][d], P['prior_all'], M['Vf'][d, t])
        add_ramp(Fe, M['ON'], M['OFF'], d, t, P['cap'])
        W_, L_ = qlast.shape
        R = dict(d=d, t=t, cond=np.array([ONGOING] * W_), qlast=qlast, W=W_, L=L_, rows=[])
        for h in range(1, F + 1):
            feats = dict(Fe); feats['h'] = np.full((W_, L_), h, np.float32)
            feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], M['dow'][d], t, h)
            R['rows'].append((h, feats, Qm[d, np.minimum(t + h, 287)], Emv[d, np.minimum(t + h, 287)]))
        X, y, e, meta, _ = to_matrix(R, ONGOING, cols, ongoing_keep(R))
''', '''        R, X, y, e, meta = march_rows(P, M, Qm, Emv, d, t, cols)                       # pvogf: pvaf code + ogfeat columns
''')
    rep("        del R, X, Xb, Fe, M; gc.collect()\n", "        del R, X, Xb, M; gc.collect()\n")
    # ---- 7. column sets: the 103 columns in every member
    rep("    cols_id = [x for x in cols if x not in ID_DROP]; ix_id = [cols.index(x) for x in cols_id]\n",
        "    cols_id = [x for x in cols if x not in ID_DROP]; ix_id = [cols.index(x) for x in cols_id]\n"
        "    new_cols = [x for x in cols if x in OG_STATIC or x in OG_STEP]                       # pvogf: ogfeat columns in npB / idB / fullB\n"
        "    assert len(new_cols) == len(OG_STATIC) + len(OG_STEP) and all(c in cols_np and c in cols_id for c in new_cols), 'ogfeat column missing'\n"
        "    assert 'tod' not in cols_np and 'dow' not in cols_id\n"
        "    log(f'pvogf features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | new {len(new_cols)} | Xe {Xe.shape} {Xe.nbytes / 1e9:.2f} GB'\n"
        "        f' | Xm {Xm.shape} | new-column finite share (first 200k F03 rows) min '\n"
        "        f'{min(float(np.isfinite(Xe[:200000, cols.index(c)]).mean()) for c in new_cols):.3f}')\n")
    # ---- 8. released windows: names, member probabilities, veto vs pvo
    rep("    subs = []; stats = []\n", "    subs = []; stats = []; PROBS = {}; VETO = dict(ok=pvo_rows() is not None, info=[])\n")
    rep("            preds = {n: np.zeros((W, F, L), bool) for n in ('pvo', 'pvo_npid')}\n",
        "            preds = {n: np.zeros((W, F, L), bool) for n in ('pvogf', 'pvogf_npid')}\n")
    rep("                pn = models['npB'].predict(X[:, ix_np]); pi = models['idB'].predict(X[:, ix_id]); pf = models['fullB'].predict(Xb)\n",
        "                pn = models['npB'].predict(X[:, ix_np]); pi = models['idB'].predict(X[:, ix_id]); pf = models['fullB'].predict(Xb)\n"
        "                for nm_, p_ in (('npB', pn), ('idB', pi), ('fullB', pf)):                      # pvogf: member probabilities\n"
        "                    P6m = np.zeros((W, F, L), np.float32); P6m[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = p_\n"
        "                    PROBS[f'{panel}__private__{nm_}'] = P6m[is_og]\n"
        "                PROBS[f'{panel}__private__window_id'] = wi.window_id.to_numpy()[is_og].astype(str)\n"
        "                PROBS[f'{panel}__private__qlast'] = R['qlast'][is_og]\n")
    rep("                for name, p in (('pvo', 0.25 * pn + 0.25 * pi + 0.5 * pf), ('pvo_npid', 0.5 * (pn + pi))):\n",
        "                for name, p in (('pvogf', 0.25 * pn + 0.25 * pi + 0.5 * pf), ('pvogf_npid', 0.5 * (pn + pi))):\n")
    rep("                                  size_pvo=float(preds['pvo'][is_og].sum((1, 2)).mean() / F),\n"
        "                                  size_npid=float(preds['pvo_npid'][is_og].sum((1, 2)).mean() / F)))\n",
        "                                  size_pvogf=float(preds['pvogf'][is_og].sum((1, 2)).mean() / F),\n"
        "                                  size_npid=float(preds['pvogf_npid'][is_og].sum((1, 2)).mean() / F)))\n")
    rep('''            for name, pr in preds.items():
                tpl[f'q_{name}'] = pr[tpl.w.astype(int), tpl.h.astype(int) - 1, tpl.l.astype(int)].astype(int)
            tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).astype(bool); tpl['split'] = split
            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split', 'q_pvo', 'q_pvo_npid']])
''', '''            for name, pr in preds.items():
                tpl[f'q_{name}'] = pr[tpl.w.astype(int), tpl.h.astype(int) - 1, tpl.l.astype(int)].astype(int)
            tpl['q_pvogfv'] = tpl['q_pvogf']                                       # pvogf: growth veto vs pvo (private ongoing windows)
            if split == 'private' and VETO['ok']:
                try:
                    tpl['q_pvogfv'], vinfo = veto_vs_pvo(tpl, preds['pvogf'], R['qlast'], is_og, wi)
                    VETO['info'].append(vinfo.assign(panel=panel))
                except Exception as ex:
                    VETO['ok'] = False; log('VETO vs pvo failed for', panel, type(ex).__name__, str(ex)[:200])
            tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).astype(bool); tpl['split'] = split
            subs.append(tpl[['window_id', 'timestamp', 'link_id', 'is_og', 'split', 'q_pvogf', 'q_pvogf_npid', 'q_pvogfv']])
''')
    rep("    for name in ('pvo', 'pvo_npid'):\n", "    OUTS = ('pvogf', 'pvogf_npid') + (('pvogfv',) if VETO['ok'] else ())\n    for name in OUTS:\n")
    rep("    log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()))\n",
        "    log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()))\n"
        "    # pvogf CONTROL: every output vs tfb-t2ogpvaf queue_submission_pvo.csv, per split (validation rows: both are fs4af copies)\n"
        "    try:\n"
        "        ref = pvo_rows()\n"
        "        if ref is not None:\n"
        "            mm = S[['window_id', 'ts', 'link_id', 'split', 'is_og']].merge(ref, on=['window_id', 'ts', 'link_id'], how='left', validate='one_to_one')\n"
        "            log(f'CONTROL pvo reference rows matched {int(mm.q_ref.notna().sum())} of {len(mm)}')\n"
        "            for name in OUTS:\n"
        "                for sp_ in ('validation', 'private'):\n"
        "                    m = (mm.split == sp_).to_numpy() & mm.q_ref.notna().to_numpy()\n"
        "                    a = S[f'out_{name}'].to_numpy()[m].astype(int); b = mm.q_ref.to_numpy()[m].astype(int)\n"
        "                    wch = pd.Series(a != b).groupby(mm.window_id.to_numpy()[m]).any()\n"
        "                    log(f'CONTROL {name} vs pvo | {sp_}: rows changed {int((a != b).sum())}/{len(a)} ({float((a != b).mean()):.4%}) | windows changed'\n"
        "                        f' {int(wch.sum())}/{len(wch)} | positives {int(a.sum())} vs pvo {int(b.sum())}'\n"
        "                        f' | Jaccard {float((a & b).sum() / max((a | b).sum(), 1)):.4f}')\n"
        "        if VETO['info']:\n"
        "            vi = pd.concat(VETO['info'], ignore_index=True); vi.to_csv(f'{OUT}/t2pvogf_veto_windows.csv', index=False)\n"
        "            log(f'VETO (> {VETO_G} cells outside the queue at T vs pvo): private ongoing windows vetoed {int(vi.veto.sum())}/{len(vi)}'\n"
        "                f' {sorted(vi.window_id[vi.veto].tolist())} | pvogf windows changed vs pvo {int((vi.cells_changed > 0).sum())}')\n"
        "    except Exception as ex:\n"
        "        log('CONTROL vs pvo skipped:', type(ex).__name__, str(ex)[:200])\n"
        "    np.savez_compressed(f'{OUT}/t2pvogf_probs.npz', **PROBS)\n")
    rep("            a, b = mm.out_pvo.to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1\n"
        "            log(f'CONTROL chain-base ongoing vs pvo ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')\n",
        "            a, b = mm.out_pvogf.to_numpy().astype(bool), mm.queue_pred.fillna(-1).to_numpy() == 1\n"
        "            log(f'CONTROL chain-base ongoing vs pvogf ({sp_}): agreement {float((a == b).mean()):.6f} changed share {float((a != b).mean()):.5f} rows {len(mm)}')\n")
    rep("    log('OGPVAF DONE')\n", "    log('PVOGF DONE')\n")
    assert "'pvo'" not in s and "'pvo_npid'" not in s and 'out_pvo.' not in s and 'q_pvo\'' not in s
    ast.parse(s)
    return s


if __name__ == '__main__':
    for smoke, dst in ((False, 't2_ongoing_pvogf.py'), (True, 't2_ongoing_pvogf_smoke.py')):
        src = build(smoke)
        open(f'{ROOT}/kaggle_jobs/{dst}', 'w').write(src)
        print('written', dst, len(src.splitlines()), 'lines | static', len(STATIC), 'step', len(STEP))
