"""draft: derive t2_ogclock_v25.py from kaggle_jobs/t2_model_v22af.py (string patches, each anchor must match exactly once)."""
import sys, ast, re
ROOT = '/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench'
src = open(f'{ROOT}/kaggle_jobs/t2_model_v22af.py').read()
lab_src = open(f'{ROOT}/analysis/t2lab/ogclock_lib.py').read()
out_path = sys.argv[1]

# kernel-portable block copied verbatim from the lab module (the exact code that produced the lab numbers)
a = lab_src.index('# ============================================================================================ kernel-portable block')
b = lab_src.index('# ============================================================================================ end of kernel-portable block')
block = lab_src[a:b]
ck_lists = re.search(r"CK_LINK = \[.*?\]\nCK_CORR = \[.*?\]\nCK_ALL = CK_LINK \+ CK_CORR\n", lab_src, re.S).group(0)


def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:80])
    src = src.replace(old, new)


DOC_OLD = src[:src.index('"""', 3) + 3]
rep(DOC_OLD, '''"""v25ck = v22af (F03 bag: np511 + id511, seeds 0 and 1, ALL-FINITE inputs) + SAME-DAY CONTEXT ("episode clock") features
for the ongoing model (track ogclock).  16 columns appended to both np and id: per link ck_run / ck_first / ck_clear / ck_frac /
ck_minr / ck_nep (current queued run, first queued observation today, last queued->free transition, share of today's observed
slots queued, minimum speed ratio today, episodes today) and per corridor cc_first / cc_peak / cc_peak_sh / cc_since_peak /
cc_rel / cc_d6 / cc_d12 / cc_d24 / cc_age / cc_area (first slot with >= 2 queued links, peak queued-link count so far, time since
the peak, count at T / peak, 30 / 60 / 120-min count change, corridor episode age, queued link-slots today).
Inputs (all timestamped <= T): slots < T-12 = the same-day Task-1-MASKED layer (released windows: mainline_states_masked of the
window's day; train rows: the train cube with eligible cells thinned i.i.d. at a per-day visible share drawn from the measured
released regimes 0.507 / 0.705 / 0.803 (shares 0.297 / 0.359 / 0.344); ineligible-valued cells are never masked); slots T-12..T-1
= window_history (train: raw history, as production); slot T = the masked slot-T value (train: the production 62 % draw).
Lab (track ogclock; F03 rows, 255 leaves / lr 0.05 / 500 trees, eligible-only inputs): train OOF official-like windows (n 2,992)
np +0.0065 (CI +0.0045..+0.0085, 7/8 panels), id +0.0054 (+0.0035..+0.0074, 5/8); pseudo-March offlike_m (n 298) np +0.0030
(-0.0050..+0.0109), id -0.0017 (-0.0102..+0.0065), np+id blend +0.0012 (-0.0068..+0.0087); the new-scenario gain sits on mid-run
windows (+0.017..+0.022, CI > 0) and comes from the corridor group.  Not measured: ALL-FINITE inputs together with ck columns.
Variant fs4ck -> queue_submission_fs4ck.csv.  CONTROL: footprint vs the attached tfb-t2v22 queue_submission_fs4.csv.
"""''')
rep("ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)\n",
    "ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)\n"
    "CKF = True                                   # same-day context features (track ogclock)\n"
    "CK_REGIMES, CK_REGIME_P = (0.507, 0.705, 0.803), (0.297, 0.359, 0.344)   # measured per-day visible share of the masked layer\n")

KBLOCK = block + '''

''' + ck_lists + '''

def ck_thin(panel, D, L):
    """[D, 288, L] bool: emulated masked-layer visibility of the train days (per-day regime, i.i.d. cells)."""
    pi = PANELS.index(panel) if panel in PANELS else 0
    rates = np.random.default_rng(7000 + pi).choice(np.array(CK_REGIMES), D, p=np.array(CK_REGIME_P))
    out = np.zeros((D, 288, L), bool)
    for d in range(D):
        out[d] = np.random.default_rng(10 ** 6 * (pi + 1) + d).random((288, L)) < rates[d]
    return out


def ck_train(P, d, t, Vh, VT, chunk=192):
    """ck features of train windows: eligible cells before T-12 thinned by P['THIN'], ineligible-valued cells always visible."""
    V = P.get('Vf', P['V']); L = V.shape[2]
    out = {k: np.empty((len(d), L), np.float32) for k in CK_ALL}
    for a in range(0, len(d), chunk):
        sl = slice(a, a + chunk); dd = d[sl]
        pre = np.where(P['E'][dd] & ~P['THIN'][dd], np.nan, V[dd]).astype(np.float32)
        X = ck_features(ck_compose(pre, t[sl], Vh[sl], VT[sl], P['cut']), t[sl])
        for k in CK_ALL:
            out[k][sl] = X[k]
    return out


def ck_day_cube(m, li, L):
    """[288, L] speed of one masked-layer day file (already filtered to the released cells used as features)."""
    ts = pd.to_datetime(m.timestamp, utc=True)
    x = m.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=m.link_id.astype(str).map(li)).dropna(subset=['l'])
    g = x.groupby(['t', 'l']).speed_kmh.mean().reset_index()
    C = np.full((288, L), np.nan, np.float32)
    C[g.t.to_numpy().astype(int), g.l.to_numpy().astype(int)] = g.speed_kmh.to_numpy()
    return C


def ck_split(P, wi, DAY, tod, Vh, VT, chunk=192):
    """ck features of the released windows: the masked layer of the window's day (slots < T-12), window_history, slot-T value."""
    W, L = VT.shape
    out = {k: np.empty((W, L), np.float32) for k in CK_ALL}
    dates = wi.date.astype(str).to_numpy(); empty = np.full((288, L), np.nan, np.float32)
    for a in range(0, W, chunk):
        sl = slice(a, a + chunk)
        pre = np.stack([DAY.get(dt_, empty) for dt_ in dates[sl]])
        X = ck_features(ck_compose(pre, tod[sl], Vh[sl], VT[sl], P['cut']), tod[sl])
        for k in CK_ALL:
            out[k][sl] = X[k]
    return out


'''
rep("# ------------------------------------------------------------------ features\ndef ffill_time(a):",
    "# ------------------------------------------------------------------ same-day context features (track ogclock)\n" + KBLOCK +
    "# ------------------------------------------------------------------ features\ndef ffill_time(a):")
rep("    return P\n\n\ndef ramp_cube", "    if CKF:\n        P['THIN'] = ck_thin(panel, D, L)\n        log(panel, 'ck: emulated masked-layer share of eligible cells', round(float(P['THIN'][E].mean()), 4))\n    return P\n\n\ndef ramp_cube")
rep("    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n", "    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])\n    CK = ck_train(P, d, t, Vh, VT) if CKF else {}\n")
rep("        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)\n",
    "        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)\n        feats.update(CK)\n")
rep("    VT = np.full((W, L), np.nan, np.float32)\n", "    VT = np.full((W, L), np.nan, np.float32)\n    DAY = {}\n")
rep("        m = m[((m.is_score_eligible == 1) | ALLFIN) & m.speed_kmh.notna()]\n",
    "        m = m[((m.is_score_eligible == 1) | ALLFIN) & m.speed_kmh.notna()]\n        if CKF:\n            DAY[date] = ck_day_cube(m, P['li'], L)\n")
rep("    Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)\n",
    "    Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)\n    CK = ck_split(P, wi, DAY, tod, Vh, VT) if CKF else {}\n"
    "    if CKF:\n        log(panel, split, 'ck: masked-layer days', len(DAY), '| ck_run > 0 share', round(float((CK['ck_run'] > 0).mean()), 3),\n"
    "            '| cc_peak mean', round(float(CK['cc_peak'][:, 0].mean()), 2))\n")
rep("        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)\n",
    "        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)\n        feats.update(CK)\n")
rep("""VARIANTS = {'fs2': ('fs_np', 'fs_id'),                                 # seed-0 pair (all-finite inputs)
            'fs4af': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),           # F03 bag of 4 with all-finite inputs (candidate)
            'fs_np': ('fs_np',)}""",
    """VARIANTS = {'fs2ck': ('fs_np', 'fs_id'),                               # seed-0 pair (all-finite inputs + ck columns)
            'fs4ck': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),           # F03 bag of 4, all-finite inputs + ck columns (candidate)
            'fs_npck': ('fs_np',)}""")
rep("for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'Vf', 'Ff'):", "for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'Vf', 'Ff', 'THIN'):")
rep("t2v22af_trainsets.csv", "t2v25ck_trainsets.csv"); rep("t2v22af_probs.npz", "t2v25ck_probs.npz"); rep("t2v22af_footprint.csv", "t2v25ck_footprint.csv")
rep("    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)}')\n",
    "    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | ck columns in np: {[c for c in cols_np if c in CK_ALL]}')\n")
ast.parse(src)
open(out_path, 'w').write(src)
print('written', out_path, len(src.splitlines()), 'lines')
