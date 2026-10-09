"""Task 2 lab (runs on Kaggle): replicate the window builder on train, build pseudo-windows, score strategies.

Truth proxy: a link is queued when the mean eligible observed speed on it is <= 0.6 * free speed.
The official truth uses the noise-free state on every link; the proxy only sees sensored, eligible
links, so every IoU here is reported on sensored-eligible cells ("IoU_s").
"""
import glob, json, os, time
import numpy as np
import pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-t2lab-full.py; patched lines end in [TFB]

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = TFB.out  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
H, F = 12, 6            # history steps (T-55..T) and horizon steps (T+5..T+30)
SPACING = 72            # 360 minutes
SMOKE = TFB.smoke  # [TFB] False in every real run (True: 2 panels, 40 train days)
if SMOKE:
    PANELS = PANELS[:2]


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def load_panel(panel):
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str})
    topo = topo.sort_values('order_index').reset_index(drop=True)
    links = topo.link_id.tolist()
    li = {l: i for i, l in enumerate(links)}
    cut = 0.6 * pd.to_numeric(topo.free_speed_kmh, errors='coerce').fillna(105).to_numpy()
    fd = pd.read_csv(f'{net}/fd_parameters.csv', dtype={'link_id': str}).set_index('link_id')
    if 'free_speed_kmh' in fd:
        cut = 0.6 * fd.free_speed_kmh.reindex(links).fillna(pd.Series(cut / 0.6, index=links)).to_numpy()
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    if SMOKE:
        files = files[:40]
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in files]
    D, L = len(files), len(links)
    V = np.full((D, 288, L), np.nan, np.float32)      # mean eligible speed per link
    Fq = np.full((D, 288, L), np.nan, np.float32)     # mean eligible flow per link
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        x = x[(x.is_score_eligible == 1) & x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li))
        x = x.dropna(subset=['l'])
        g = x.groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    E = np.isfinite(V)
    Q = E & (V <= cut[None, None, :])
    return dict(topo=topo, links=links, li=li, cut=cut, days=days, V=V, Fq=Fq, E=E, Q=Q,
                sensored=E.any((0, 1)))


def established(Qh):
    """Visible history holds >=2 queued observations with >=2 on one link."""
    return Qh.sum() >= 2 and (Qh.sum(0) >= 2).any()


def iou(truth, pred, mask):
    t, p = truth[mask], pred[mask]
    u = (t | p).sum()
    return 1.0 if u == 0 else (t & p).sum() / u


def build_windows(P, cap=None, min_cov=0.7, max_pers_iou=0.9, global_spacing=True):
    """Chronological scan, one condition at a time, >=360 min spacing (hypothesised builder)."""
    D = len(P['days']); L = len(P['links'])
    out = []; taken = []
    for cond in ('queue_onset', 'queue_ongoing'):
        n = 0
        if not global_spacing:
            taken = []
        for d in range(D):
            for t in range(H - 1, 288 - F):
                g = d * 288 + t
                if any(abs(g - x) < SPACING for x in taken[-400:]):
                    continue
                Qh = P['Q'][d, t - H + 1:t + 1]; Eh = P['E'][d, t - H + 1:t + 1]
                Qf = P['Q'][d, t + 1:t + 1 + F]; Ef = P['E'][d, t + 1:t + 1 + F]
                if not Qf.any():
                    continue
                cov = Eh[:, P['sensored']].mean()
                if cov < min_cov:
                    continue
                est = established(Qh)
                if (cond == 'queue_ongoing') != est:
                    continue
                if cond == 'queue_ongoing':
                    pers = np.repeat(P['Q'][d, t][None], F, 0)
                    if iou(Qf, pers, Ef) > max_pers_iou:
                        continue
                out.append(dict(d=d, t=t, date=P['days'][d], cond=cond, cov=float(cov)))
                taken.append(g); taken.sort(); n += 1
                if cap and n >= cap:
                    break
            if cap and n >= cap:
                break
    return out


def origin_of(ts):
    t = pd.Timestamp(ts)
    return t.strftime('%Y-%m-%d'), t.hour * 12 + t.minute // 5


# ------------------------------------------------------------------ strategies
SOUK_CLUSTERS = {  # copied from the base notebook (get_onset_cluster)
    'D12_I5_N': lambda h: ["L5N-147", "L5N-235", "L5N-237"] if h >= 13 else ["L5N-059", "L5N-061", "L5N-104"],
    'D12_I5_S': lambda h: ["L5S-152", "L5S-249", "L5S-065"] if h >= 11 else ["L5S-051", "L5S-101", "L5S-132", "L5S-278"],
    'D7_I10_E': lambda h: ["L10E-043", "L10E-052", "L10E-053", "L10E-124", "L10E-125", "L10E-180", "L10E-243"],
    'D7_I10_W': lambda h: ["L10W-212", "L10W-013"],
    'D7_I210_E': lambda h: ["L210E-001", "L210E-129", "L210E-131", "L210E-132", "L210E-261"],
    'D7_I210_W': lambda h: ["L210W-190", "L210W-246"] if h in (8, 9) else ["L210W-004", "L210W-028", "L210W-029", "L210W-172"],
    'D7_I405_N': lambda h: ["L405N-041", "L405N-145"],
    'D7_I405_S': lambda h: ["L405S-127", "L405S-317", "L405S-319", "L405S-373"] if h in (9, 10, 11, 12) else ["L405S-001", "L405S-017", "L405S-056", "L405S-065", "L405S-260", "L405S-264"],
}


def fill_unsensored(pred, sensored):
    """shuangranse: mark unsensored links lying between two predicted-queued sensored links."""
    out = pred.copy()
    for s in range(pred.shape[0]):
        hot = np.where(pred[s] & sensored)[0]
        for a, b in zip(hot[:-1], hot[1:]):
            if b - a > 1:
                out[s, a + 1:b] |= ~sensored[a + 1:b]
    return out


def template_table(P, wins, exclude_days):
    """P(link queued at T+30 | onset window, hour) and P(link queued at step | ongoing), from training windows."""
    L = len(P['links'])
    on = {}; og = np.zeros((F, L)); n_og = 0; ogh = {}
    for w in wins:
        if w['d'] in exclude_days:
            continue
        d, t = w['d'], w['t']
        Qf = P['Q'][d, t + 1:t + 1 + F]
        if w['cond'] == 'queue_onset':
            h = t // 12
            a = on.setdefault(h, [np.zeros(L), 0]); a[0] += Qf[-1]; a[1] += 1
        else:
            og += Qf; n_og += 1
            a = ogh.setdefault(t // 12, [np.zeros((F, L)), 0]); a[0] += Qf; a[1] += 1
    return on, og / max(n_og, 1), ogh


def predict(strategy, P, panel, w, tables=None, thr=0.3):
    d, t, cond = w['d'], w['t'], w['cond']
    L = len(P['links'])
    pred = np.zeros((F, L), bool)
    qT = P['Q'][d, t]
    if strategy == 'zeros':
        return pred
    if strategy.startswith('persist'):
        pred[:] = qT
        if 'fill' in strategy:
            pred = fill_unsensored(pred, P['sensored'])
        return pred
    if strategy == 'base_heuristic':          # base notebook without its (unavailable) booster
        if cond == 'queue_ongoing':
            pred[:] = qT
        else:
            for l in SOUK_CLUSTERS[panel]((t + 1) // 12):
                if l in P['li']:
                    pred[-1, P['li'][l]] = True
        return pred
    if strategy.startswith('template'):
        on, og, ogh = tables
        if cond == 'queue_onset':
            if 'baseon' in strategy:
                for l in SOUK_CLUSTERS[panel]((t + 1) // 12):
                    if l in P['li']:
                        pred[-1, P['li'][l]] = True
            else:
                h = t // 12
                prob = on[h][0] / on[h][1] if h in on and on[h][1] >= 3 else np.zeros(L)
                pred[-1] = prob >= thr
        else:
            pred[:] = qT
            if 'unionh' in strategy:
                h = t // 12
                ph = ogh[h][0] / ogh[h][1] if h in ogh and ogh[h][1] >= 5 else og
                pred |= ph >= thr
            elif 'union' in strategy:
                pred |= og >= thr
        if 'fill' in strategy:
            pred = fill_unsensored(pred, P['sensored'])
        return pred
    raise ValueError(strategy)


def evaluate(P, panel, wins, strategies, folds=4, train_wins=None):
    rows = []
    days = np.arange(len(P['days']))
    for k in range(folds):
        test_days = set(days[k::folds])
        tables = template_table(P, train_wins or wins, test_days)
        for w in wins:
            if w['d'] not in test_days:
                continue
            d, t = w['d'], w['t']
            Qf = P['Q'][d, t + 1:t + 1 + F]; Ef = P['E'][d, t + 1:t + 1 + F]
            for s, kw in strategies:
                pr = predict(s, P, panel, w, tables, **kw)
                rows.append(dict(panel=panel, cond=w['cond'], d=d, t=t, strategy=s + str(kw.get('thr', '')),
                                 iou=iou(Qf, pr, Ef), first_step=int(np.argmax(Qf.any(1)))))
    return rows


STRATS = [('zeros', {}), ('persist', {}), ('base_heuristic', {})] + \
         [('template', dict(thr=t)) for t in (0.3, 0.4)] + \
         [('template_union', dict(thr=t)) for t in (0.3, 0.4, 0.5)] + \
         [('template_baseon_union', dict(thr=t)) for t in (0.3, 0.4, 0.5)] + \
         [('template_baseon_unionh', dict(thr=t)) for t in (0.3, 0.4, 0.5)]

report = {}; all_rows = []; official_rows = []
for panel in PANELS:
    P = load_panel(panel)
    log(panel, 'links', len(P['links']), 'sensored', int(P['sensored'].sum()), 'days', len(P['days']),
        'queued share', float(P['Q'].sum() / max(P['E'].sum(), 1)))
    wi = pd.read_csv(f'{REL}/task2/{panel}/train/window_index.csv')
    official = []
    for r in wi.itertuples():
        date, t = origin_of(r.forecast_origin)
        official.append(dict(d=P['days'].index(date), t=t, date=date, cond=r.condition, cov=r.history_coverage))
    mine = build_windows(P, cap=5)
    match = {(w['date'], w['t'], w['cond']) for w in mine} & {(w['date'], w['t'], w['cond']) for w in official}
    # coverage definition check
    cov_chk = [(w['cov'], float(P['E'][w['d'], w['t'] - H + 1:w['t'] + 1][:, P['sensored']].mean()), float(P['E'][w['d'], w['t'] - H + 1:w['t'] + 1].mean())) for w in official]
    first = [int(np.argmax(P['Q'][w['d'], w['t'] + 1:w['t'] + 1 + F].any(1))) if P['Q'][w['d'], w['t'] + 1:w['t'] + 1 + F].any() else -1
             for w in official]
    report[panel] = dict(official=[(w['date'], w['t'], w['cond']) for w in official],
                         mine=[(w['date'], w['t'], w['cond']) for w in mine], n_match=len(match),
                         cov_official_vs_sensored=cov_chk[:4], first_queued_step_official=first)
    log(panel, 'builder match', len(match), '/ 10;  first queued horizon step (official):', first)
    pseudo = build_windows(P, cap=None)
    trainw = build_windows(P, cap=None, global_spacing=False)
    log(panel, 'pseudo windows', len(pseudo), {c: sum(w['cond'] == c for w in pseudo) for c in ('queue_onset', 'queue_ongoing')},
        'train windows', {c: sum(w['cond'] == c for w in trainw) for c in ('queue_onset', 'queue_ongoing')})
    all_rows += evaluate(P, panel, pseudo, STRATS, train_wins=trainw)
    # official train windows, templates learned from windows on other days
    tables = template_table(P, trainw, {w['d'] for w in official})
    for w in official:
        d, t = w['d'], w['t']
        Qf = P['Q'][d, t + 1:t + 1 + F]; Ef = P['E'][d, t + 1:t + 1 + F]
        for s, kw in STRATS:
            official_rows.append(dict(panel=panel, cond=w['cond'], strategy=s + str(kw.get('thr', '')),
                                      iou=iou(Qf, predict(s, P, panel, w, tables, **kw), Ef)))
    np.savez_compressed(f'{OUT}/pseudo_{panel}.npz', d=[w['d'] for w in pseudo], t=[w['t'] for w in pseudo],
                        cond=[w['cond'] for w in pseudo])
    del P

json.dump(report, open(f'{OUT}/builder_report.json', 'w'), indent=1, default=str)
R = pd.DataFrame(all_rows); O = pd.DataFrame(official_rows)
R.to_csv(f'{OUT}/pseudo_scores.csv', index=False); O.to_csv(f'{OUT}/official_train_scores.csv', index=False)


def summarize(df, name):
    # competition aggregation: window -> panel x condition -> panel -> family -> mean
    g = df.groupby(['strategy', 'panel', 'cond']).iou.mean().groupby(['strategy', 'panel']).mean()
    fam = g.reset_index().assign(fam=lambda x: x.panel.str.rsplit('_', n=1).str[0]).groupby(['strategy', 'fam']).iou.mean()
    tot = fam.groupby('strategy').mean()
    byc = df.groupby(['strategy', 'cond']).iou.mean().unstack()
    print(f'\n===== {name}: S_queue proxy (sensored cells) =====')
    print(pd.concat([tot.rename('S_queue'), byc], axis=1).sort_values('S_queue', ascending=False).round(4).to_string())


summarize(R, 'PSEUDO windows, 4-fold by day')
summarize(O, 'OFFICIAL 80 train windows')
print('\nfirst queued horizon step distribution (pseudo):')
print(R[R.strategy == 'zeros'].groupby('cond').first_step.value_counts().unstack().fillna(0).astype(int).to_string())
log('done')

# [TFB] self-check: the pseudo windows are a hidden input of every Task 1 member; they must equal the original
# [TFB] tfb-t2lab-full outputs (window counts per condition and SHA-1 of json([[d, t, cond], ...]) in tfb/config.py PSEUDO_EXPECTED)
import hashlib  # [TFB]
if not SMOKE:  # [TFB]
    for _p, (_n, _non, _nog, _sha) in TFB.cfg['expected'].items():  # [TFB]
        _z = np.load(f'{OUT}/pseudo_{_p}.npz', allow_pickle=True)  # [TFB]
        _trip = [[int(a), int(b), str(c)] for a, b, c in zip(_z['d'], _z['t'], _z['cond'])]  # [TFB]
        _got = (len(_trip), int((_z['cond'] == 'queue_onset').sum()), int((_z['cond'] == 'queue_ongoing').sum()), hashlib.sha1(json.dumps(_trip).encode()).hexdigest())  # [TFB]
        log('PSEUDO CHECK', _p, 'windows', _got[0], 'onset', _got[1], 'ongoing', _got[2], 'sha1', _got[3], 'OK' if _got == (_n, _non, _nog, _sha) else 'MISMATCH')  # [TFB]
        assert _got == (_n, _non, _nog, _sha), f'pseudo windows of {_p} differ from tfb-t2lab-full: {_got} vs {(_n, _non, _nog, _sha)}'  # [TFB]
