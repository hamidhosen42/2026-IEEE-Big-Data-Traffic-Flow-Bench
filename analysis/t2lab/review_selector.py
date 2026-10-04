"""Review (3): independent (plain-loop) replay of the official selector on the train split, per panel and rule variant,
compared with the official train window origins; also an independent check of lab.fill_time and of the vectorised stats.
Usage: review_selector.py [PANEL ...]  (default 3 panels)."""
import sys, numpy as np, pandas as pd, lab

panels = sys.argv[1:] or ['D7_I10_W', 'D12_I5_S', 'D7_I210_E']
H, F, SP = 12, 6, 72

# --- fill_time independent check on a synthetic series
a = np.full((1, 288, 2), np.nan, np.float32); a[0, 10, 0] = 10; a[0, 14, 0] = 18; a[0, 20, 0] = 0; a[0, 100, 0] = 5
b = lab.fill_time(a, 3)
assert np.allclose(b[0, 11:14, 0], [12, 14, 16]) and np.isnan(b[0, 15:20, 0]).all() and np.isnan(b[0, 21:100, 0]).all() and np.isnan(b[0, :10, 0]).all()
b = lab.fill_time(a, 5); assert np.isfinite(b[0, 15:20, 0]).all() and np.isnan(b[0, 21:100, 0]).all() and np.isnan(b[0, 101:, 0]).all()
b = lab.fill_time(a, 80); assert np.isfinite(b[0, 10:101, 0]).all() and np.isnan(b[0, :, 1]).all()
print('fill_time: interior gaps <= maxgap interpolated, larger gaps and edges untouched: PASS', flush=True)


def iou(t, p, m):
    tt, pp = t[m], p[m]; u = (tt | pp).sum()
    return 1.0 if u == 0 else (tt & pp).sum() / u


def stats_loop(P, QF, MF, d, t):
    """all selector quantities at (d, t) with plain slicing."""
    E, Q = P['E'], P['Q']
    cov = E[d, t - H:t].mean()
    est = (Q[d, t - H:t].sum(0) >= 2).any()
    hq_obs = Q[d, t + 1:t + 1 + F].any(); hq_fill = QF[d, t + 1:t + 1 + F].any()
    rep = lambda s: np.repeat(s[None], F, 0)
    pers = {('obs', 'E'): iou(Q[d, t + 1:t + 1 + F], rep(Q[d, t]), E[d, t + 1:t + 1 + F]),
            ('fill', 'E'): iou(Q[d, t + 1:t + 1 + F], rep(QF[d, t]), E[d, t + 1:t + 1 + F]),
            ('fill', 'all'): iou(QF[d, t + 1:t + 1 + F], rep(QF[d, t]), MF[d, t + 1:t + 1 + F])}
    return cov, est, hq_obs, hq_fill, pers


def greedy_loop(cands, scan, quota=5):
    """cands: sorted list of (g, cond). interleaved: one scan, spacing to every earlier pick. two_pass: onset first."""
    picks = []
    if scan == 'interleaved':
        for g, c in cands:
            if sum(1 for _, cc in picks if cc == c) >= quota:
                continue
            if all(abs(g - x) >= SP for x, _ in picks):
                picks.append((g, c))
            if all(sum(1 for _, cc in picks if cc == q) >= quota for q in ('on', 'og')):
                break
    else:
        for want in ('on', 'og'):
            for g, c in cands:
                if c != want or sum(1 for _, cc in picks if cc == want) >= quota:
                    continue
                if all(abs(g - x) >= SP for x, _ in picks):
                    picks.append((g, c))
    return picks


RULES = {'prod obs/E': ('hb', 'obs', 'E'), 'hb fill/E': ('hb', 'fill', 'E'), 'hb fill/all': ('hb', 'fill', 'all'),
         't36 fill/E': ('t36', 'fill', 'E'), 't36 fill/all': ('t36', 'fill', 'all')}
rows = []
for panel in panels:
    P = lab.load_panel(panel); D = len(P['days'])
    off = lab.official_windows(P)
    o_on = [d * 288 + t for d, t, c in off if c == lab.ONSET]; o_og = [d * 288 + t for d, t, c in off if c == lab.ONGOING]
    fills = {k: lab.filled_state(P, k) for k in ('hb', 't36')}
    # independent stats for every candidate slot (plain loops), per fill kind
    S = {}
    for fk, FS in fills.items():
        QF, MF = FS['QF'], FS['MF']
        rec = {}
        for d in range(D):
            for t in range(H, 288 - F):
                rec[(d, t)] = stats_loop(P, QF, MF, d, t)
        S[fk] = rec
        # vectorised stats of lab agree?
        V = lab.selector_stats(P, fk)
        dd = np.array([k[0] for k in rec]); tt = np.array([k[1] for k in rec])
        cov = np.array([v[0] for v in rec.values()]); est = np.array([v[1] for v in rec.values()])
        hqo = np.array([v[2] for v in rec.values()]); hqf = np.array([v[3] for v in rec.values()])
        ok = (np.allclose(cov, V['cov'][dd, tt]) and np.array_equal(est, V['est'][dd, tt]) and np.array_equal(hqo, V['hq_obs'][dd, tt])
              and np.array_equal(hqf, V['hq_fill'][dd, tt]))
        for key in (('obs', 'E'), ('fill', 'E'), ('fill', 'all')):
            pv = np.array([v[4][key] for v in rec.values()])
            ok &= bool(np.allclose(pv, V['pers'][key][dd, tt], atol=1e-6))
        print(panel, fk, 'vectorised selector_stats == plain-loop stats:', 'PASS' if ok else 'FAIL', flush=True)
    for rname, (fk, st, cells) in RULES.items():
        rec = S[fk]
        cands = []
        for (d, t), (cov, est, hqo, hqf, pers) in rec.items():
            if cov < 0.7:
                continue
            g = d * 288 + t
            if not est:
                if hqo:
                    cands.append((g, 'on'))
            else:
                hq = hqo if cells == 'E' else hqf
                if hq and pers[(st, cells)] <= 0.9:
                    cands.append((g, 'og'))
        cands.sort()
        for scan in ('interleaved', 'two_pass'):
            pk = greedy_loop(cands, scan)
            s_on = [g for g, c in pk if c == 'on']; s_og = [g for g, c in pk if c == 'og']
            r = dict(panel=panel, rule=rname, scan=scan,
                     on_exact=sum(g in s_on for g in o_on), on_pm1=sum(any(abs(g - x) <= 1 for x in s_on) for g in o_on),
                     og_exact=sum(g in s_og for g in o_og), og_pm1=sum(any(abs(g - x) <= 1 for x in s_og) for g in o_og),
                     og_picks=[(g // 288, g % 288) for g in s_og], og_official=[(g // 288, g % 288) for g in o_og])
            # compare with lab.greedy on lab.candidates for the same rule
            on_m, og_m = lab.candidates(P, dict(pers_state=st, cells=cells, fill=fk))
            a, b = lab.greedy(on_m, og_m, scan)
            r['lab_same'] = ([g // 288 for g in s_on] == [x[0] for x in a] and [g % 288 for g in s_on] == [x[1] for x in a]
                             and [(g // 288, g % 288) for g in s_og] == b)
            rows.append(r)
            print({k: v for k, v in r.items() if k not in ('og_official',)}, flush=True)
    print(panel, 'official ongoing origins:', [(g // 288, g % 288) for g in o_og], 'onset:', [(g // 288, g % 288) for g in o_on], flush=True)
    lab.drop_panel(panel)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('\nTOTALS over', panels)
print(T.groupby(['rule', 'scan'])[['on_exact', 'on_pm1', 'og_exact', 'og_pm1']].sum().to_string())
print('lab.greedy/candidates identical to the plain-loop replay in every case:', bool(T.lab_same.all()))
ref = pd.read_csv(f'{lab.CACHE}/replay_table.csv')
ref = ref[ref.panel.isin(panels)]
print('\nreference replay_table.csv totals for the same panels:')
print(ref.groupby(['fill', 'pers_state', 'cells', 'scan'])[['on_exact', 'on_pm1', 'og_exact', 'og_pm1']].sum().to_string())
print('SELECTOR REVIEW DONE')
