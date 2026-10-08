"""Generate kaggle_jobs/t1_bspec_member_a/b.py from kaggle_jobs/t1_ramp_member_a/b.py (asserted text edits) + the BSPEC BLOCK
of analysis/t1lab/t1bspec_exp.py copied verbatim."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LAB = open(os.path.join(HERE, 't1bspec_exp.py')).read()
i0 = LAB.index('# ================================================================== BSPEC BLOCK')
i1 = LAB.index('# ================================================================== END BSPEC BLOCK')
BLOCK = LAB[i0:i1] + '# ================================================================== END BSPEC BLOCK\n\n\n'

HEADER = '''"""Task 1 t1bspec member = BLACKOUT SPECIALIST (t1ramp member, seed 13, with selection-faithful simulated blackouts and
30 timing columns). BLACKOUT-ONLY role (swap kernel: Task 2 blackout rows only). Same panel halves, data loading, outputs,
gates as the source kaggle_jobs/t1_ramp_member_a/b.py except
  (1) simulated blackouts in the validation / private months at SELECTION-FAITHFUL origins (bs_faithful_cands): onset =
      first crossing (no observed queued cell in T+1..T+5, one at T+6, as the official onset windows), ongoing = run-starts
      of the faithful ongoing map (time-filled persistence <= 0.9), balanced half onset (official 5 + 5), DARK_N per month;
  (2) features() = t1ramp features + 30 timing columns BCOLS (bspec_feats; NaN outside blackout rows): condition flag,
      slot offset, onset breakdown flag (k >= 6), oriented boundary values before T / after T+18 (own, +-2 stations),
      corridor queued share at both boundaries, onset-aware piecewise interpolation minus linear, ramp flow over the gap.
      Origins per view: train split = pseudo gap origins with their cond (tfb-t2lab-full pseudo_<panel>.npz); simulated
      blackout view = the simulated origins with their type; released month (template prediction) = the OFFICIAL Task 2
      origins with the condition of task2/<panel>/<split>/window_index.csv; transductive rounds = none;
  (3) feature-count gate = V12 + FD + 9 + 30.
  Local pseudo-holdout: analysis/t1lab/t1bspec_exp.py / t1bspec_report.py (results analysis/t2lab/results/t1bspec).
Previous notes:
'''

EDITS = [
    ('"""Task 1 t1ramp member', HEADER + 'Task 1 t1ramp member'),
    ('def features(c, st, hv, hq):\n    """t1ramp:', BLOCK + 'def features(c, st, hv, hq):\n    """t1ramp:'),
    ("""    F.update(ramp_feats(c.mq, c.ramp, st, F['tq']))
    assert len(F) == V12_NFEAT + N_FD + N_RAMP and list(F)[-N_RAMP:] == RCOLS
""", """    F.update(ramp_feats(c.mq, c.ramp, st, F['tq']))
    F.update(bspec_feats(c.mv, c.mq, getattr(c, 'bs_origins', []), st.vf.to_numpy(), st.lanes.to_numpy(), c.ramp['dirn'],
                         F['tv'], F['tq'], F['rp_on'], F['rp_off'], F['rp_net']))   # t1bspec: timing columns (blackout rows only)
    assert len(F) == V12_NFEAT + N_FD + N_RAMP + N_BS and list(F)[-N_BS - N_RAMP:-N_BS] == RCOLS and list(F)[-N_BS:] == BCOLS
"""),
    ("""    return [(pos[all_days[d]], int(t)) for d, t in zip(z['d'], z['t']) if all_days[d] in pos]""",
     """    return [(pos[all_days[d]], int(t), 0 if str(cc) == 'queue_onset' else 1) for d, t, cc in zip(z['d'], z['t'], z['cond'])
            if all_days[d] in pos]                                   # t1bspec: + condition (0 onset / 1 ongoing)"""),
    ("""    for d, t in origins:
        mv[d, t + 1:t + 1 + GAP] = np.nan""", """    for d, t, _ in origins:
        mv[d, t + 1:t + 1 + GAP] = np.nan"""),
    ("""    c.mv, c.mq, c.mo, gapm = darken(c, origins)                     # train on the darkened view
""", """    c.mv, c.mq, c.mo, gapm = darken(c, origins)                     # train on the darkened view
    c.bs_origins = origins                                          # t1bspec: timing columns on the train gap rows
"""),
    ("""    assert len(cols) == V12_NFEAT + N_FD + N_RAMP and cols[V12_NFEAT:V12_NFEAT + N_FD] == FD_COLS and cols[-N_RAMP:] == RCOLS, (len(cols), cols[-N_RAMP:])
""", """    assert len(cols) == V12_NFEAT + N_FD + N_RAMP + N_BS and cols[V12_NFEAT:V12_NFEAT + N_FD] == FD_COLS \\
        and cols[-N_BS - N_RAMP:-N_BS] == RCOLS and cols[-N_BS:] == BCOLS, (len(cols), cols[-N_BS:])
    _gp = gapm[tmask]; _bc = X[:, cols.index('bs_cond')]
    assert np.array_equal(np.isfinite(_bc), _gp), 't1bspec: timing columns must be finite exactly on the train gap rows'
    log(panel, 't1bspec: train gap rows', int(_gp.sum()), 'onset share', round(float((_bc[_gp] == 0).mean()), 3) if _gp.any() else '-',
        'boundary finite share', round(float(np.isfinite(X[_gp, cols.index('bs_pre_v0')]).mean()), 3) if _gp.any() else '-')
"""),
    ("""        cs.ramp = dict(ramp_cube(panel, split, st, cs.days), **RHAS)   # t1ramp: released ramp flows of the target month
""", """        cs.ramp = dict(ramp_cube(panel, split, st, cs.days), **RHAS)   # t1ramp: released ramp flows of the target month
        cs.bs_origins = official_origins(panel, split, cs)               # t1bspec: official windows + window_index condition
"""),
    ("""            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__)
            gone = hid | hold
""", """            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__); cv.bs_origins = []
            gone = hid | hold
"""),
    ("""            tb, n_c, n_s = sim_blackouts(cs, st, rr, DARK_SEED + (split == 'private'))""",
     """            tb, n_c, n_s, sims = sim_blackouts(cs, st, rr, DARK_SEED + (split == 'private'))"""),
    ("""            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__)
            gone = TB | hold
""", """            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__); cv.bs_origins = sims   # t1bspec: simulated types
            gone = TB | hold
"""),
    ("""            log(panel, split, 't1dark: real blackout rows', int(rr.sum()), '| simulated blackouts', n_s, 'of', n_c,""",
     """            log(panel, split, 't1bspec: official origins', len(cs.bs_origins), 'onset', sum(o[2] == 0 for o in cs.bs_origins),
                '| simulated onset', sum(s[2] == 0 for s in sims), 'ongoing', sum(s[2] == 1 for s in sims))
            log(panel, split, 't1dark: real blackout rows', int(rr.sum()), '| simulated blackouts', n_s, 'of', n_c,"""),
    # selection-faithful candidates replace the t1dark selector-like candidates
    ('''def sim_blackouts(cs, st, rr, seed):
    """[day, 288] rows of up to DARK_N simulated blackouts at queue-like origins (onset: no established queue in T-12..T-1;
    ongoing: established queue, persistence <= 0.9), >= GAP slots from the real blackouts and from each other."""
    vf = st.vf.to_numpy()
    obs = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq)
    Qm = obs & (cs.mv <= 0.6 * vf[None, None, :])
    cands = []
    for d in range(cs.mv.shape[0]):
        for t in range(12, 288 - GAP):
            if rr[d, max(t + 1 - GAP, 0):min(t + 2 * GAP + 1, 288)].any():
                continue
            if cs.melig[d, t - 12:t].mean() < 0.7 or not Qm[d, t + 1:t + 7].any():
                continue
            if (Qm[d, t - 12:t].sum(0) >= 2).any():
                S = Qm[d, t]; Tf = Qm[d, t + 1:t + 7]; Mf = obs[d, t + 1:t + 7]
                u = ((Tf | S[None]) & Mf).sum(); i = (Tf & S[None] & Mf).sum()
                if (1.0 if u == 0 else i / u) > 0.9:
                    continue
                cands.append((d, t, 1))
            else:
                cands.append((d, t, 0))
''', '''def official_origins(panel, split, cs):
    """t1bspec: (day, T, c) of the official Task 2 windows of this month, c = 0 queue_onset / 1 queue_ongoing (window_index)."""
    f = f'{REL}/task2/{panel}/{split}/window_index.csv'
    if not os.path.exists(f):
        return []
    wi = pd.read_csv(f); pos = {str(d): i for i, d in enumerate(cs.days)}; out = []
    for o, cond in zip(pd.to_datetime(wi.forecast_origin, utc=True), wi.condition.astype(str)):
        d = o.strftime('%Y-%m-%d')
        if d in pos:
            out.append((pos[d], o.hour * 12 + o.minute // 5, 0 if cond == 'queue_onset' else 1))
    assert set(wi.condition.astype(str)) <= {'queue_onset', 'queue_ongoing'}, set(wi.condition)
    return out


def sim_blackouts(cs, st, rr, seed):
    """t1bspec: [day, 288] rows of up to DARK_N simulated blackouts at SELECTION-FAITHFUL origins (bs_faithful_cands: onset
    first crossing at T+6, ongoing run-starts), balanced half onset, >= 2 GAP apart; also returns the (d, T, c) list."""
    vf = st.vf.to_numpy()
    obs = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq)
    cands = bs_faithful_cands(cs.mv, obs, cs.melig, rr, vf, GAP, 12, 6)
'''),
    ("""    tb = np.zeros(cs.mv.shape[:2], bool)
    for d, t, _ in acc:
        tb[d, t + 1:t + 1 + GAP] = True
    return tb, len(cands), len(acc)""", """    tb = np.zeros(cs.mv.shape[:2], bool)
    for d, t, _ in acc:
        tb[d, t + 1:t + 1 + GAP] = True
    return tb, len(cands), len(acc), sorted(acc)"""),
    ("""        Xs = np.column_stack([np.asarray(Fs[k])[di[ok], ti[ok], si[ok]] for k in cols]).astype(np.float32)
""", """        Xs = np.column_stack([np.asarray(Fs[k])[di[ok], ti[ok], si[ok]] for k in cols]).astype(np.float32)
        _bc = Xs[:, cols.index('bs_cond')]
        log(panel, split, 't1bspec: template rows inside official blackouts', int(np.isfinite(_bc).sum()),
            'onset', int((_bc == 0).sum()), 'ongoing', int((_bc == 1).sum()))
"""),
]


def make(half):
    src = open(os.path.join(ROOT, 'kaggle_jobs', f't1_ramp_member_{half}.py')).read()
    for old, new in EDITS:
        assert src.count(old) == 1, (half, old[:80], src.count(old))
        src = src.replace(old, new)
    dst = os.path.join(ROOT, 'kaggle_jobs', f't1_bspec_member_{half}.py')
    open(dst, 'w').write(src)
    print('wrote', dst, len(src.splitlines()), 'lines')


if __name__ == '__main__':
    for h in ('a', 'b'):
        make(h)
