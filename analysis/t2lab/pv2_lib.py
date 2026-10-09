"""TRACK pv2 (E3, 2026-10-09): improve pick A's APRIL ONSET rows (today tfb-t2opv queue_submission_pv.csv = CatBoost bag of 3,
train rows + released-March onset windows at weights 10 / 20 / 40, sp4 scenario-prior blend).  New file; nothing existing is edited.

PRE-REGISTERED DECISION RULE (written 2026-10-09 16:05 +06, before any pv2 candidate number was computed; not changed afterwards)
  Reference = pv (lab analog = pvon_step2 arm B_bag: raw train onset rows of lab dense1500 + March onset windows of days < 15,
  production onset rule, 72-slot spacing, visible labels, blended prior; weights 10 / 20 / 40, bag = mean probability; no ramp
  columns; CatBoost 600 it / depth 8 / lr 0.08).
  Candidates (each against pv, everything else equal):
    a1  pv + 3 same-scenario onset-LOCATION columns (first-appearance onset events of the visible state, see pv2_events):
          on_k6  = P(link queued | an onset event at slot T+6 +- 6, weekday class)  [train: all train days, own day left out;
                   March / April rows: blended (1-a) train + a scenario(days < d), a = n / (n + 4), n = visible-at-event count]
          on_k12 = the same, all days, T+6 +- 12 slots;   on_all = the same, all days, whole day (location frequency)
    a2  a1 + 4 window-relative columns: r_prior = prior / max over the window's links, r_on12 = on_k12 / max, d_hot = |l - argmax
          on_k12| (99 if max 0), d_hotp = |l - argmax prior| (99 if max 0)   (= "distance to the links that queued most so far")
    a3  a2 + 3 same-scenario speed columns: typ_T / typ_6 = typical visible v / v_cut at (link, T +- 2) / (link, T+6 +- 2) of the
          days < d (blended as above, n = count / 5), anom = vr_last - typ_T
    b1  pv with 4x heavier March weight (bag 40 / 80 / 160)
    b2  0.5 pv + 0.5 March-ONLY CatBoost (same params, trained on the March rows only)
  ADOPT a candidate only if BOTH
    (1) MANY-DAY pseudo-March: delta = aggregate(candidate) - aggregate(pv) >= +0.01 IoU AND 95 % CI lower bound > 0 AND
        >= 6 of 8 panels with a positive per-panel mean delta.  Windows: the 131 pseudo-March onset windows of March days >= 15
        (official-like multi-start greedy on the masked layer, visible labels, masked history + masked slot T, pvon_step2 set);
        training March rows = March days < 15.  Every CatBoost model is trained with seeds 0 and 1 (same seeds in both arms);
        per-window IoU averaged over the 2 seeds; aggregation window -> panel -> family -> mean; CI = lab.paired_bootstrap
        (2000 resamples within panel) on the seed-averaged per-window IoU.
    (2) OFFICIAL-40 train-window sign not negative: delta >= 0 on the 40 official TRAIN onset windows (all in fold 0; raw history,
        production slot-T simulation of the lab cache), pv recipe in the TRAIN scenario (pvon_step3 design): base rows = lab
        dense1500 onset rows of the days outside fold 0 and outside the source month (31 train days right after fold 0), old-scenario
        prior / tables from those days; "March" = the source month on the emulated masked layer (mask seed 1), windows re-selected
        with the production onset rule, masked features, visible labels; scenario for the blend / new columns = masked source month
        + masked fold-0 days < d (kernel timeline: March then April).  Seeds 0 and 1 averaged as in (1).
  Several passing candidates: the one with the larger many-day delta.  Combinations were not pre-registered and are not adopted.
  Reported, not binding: each seed alone, per-panel deltas, official-like fold-0 onset windows (offlike, n ~ 520) in the train analog,
  zero-overlap / size decomposition, FM-PM-style private pricing (lab-emulated April decisions; each changed private onset window
  priced with the mean many-day per-window dIoU of the changed lab windows of the same panel, pooled if n < 3; 0.15 x mean over 40).

Kernel-portable block below: numpy only, copied verbatim into the kernel by pv2_make_kernel.py.
"""
import numpy as np

# ============================================================================================ kernel-portable block (pv2)
PV2_K = 4.0                                                   # blend constant (the kernel's K_BLEND)
PV2_A = ['on_k6', 'on_k12', 'on_all']
PV2_B = ['r_prior', 'r_on12', 'd_hot', 'd_hotp']
PV2_C = ['typ_T', 'typ_6', 'anom']
PV2_TABS = ('A6', 'B6', 'N6', 'A12', 'B12', 'N12', 'Aal', 'Bal', 'Nal', 'S2', 'C2')


def pv2_box(a, k):
    """sum over the slots s-k..s+k of the same day (zero padded at the day edges); a [D, 288, ...]."""
    pad = [(0, 0), (k + 1, k)] + [(0, 0)] * (a.ndim - 2)
    c = np.cumsum(np.pad(a, pad), 1)
    return c[:, 2 * k + 1:] - c[:, :-2 * k - 1]


def pv2_events(Q, E):
    """first-appearance onset events of a state (Q visible queued, E visible; [D, 288, L] bool) -> new [D, 288] bool:
    a visible queued cell at slot s, none in s-5..s-1, no established queue (some link with >= 2 queued slots) in s-18..s-7 (= the
    12-slot history of an onset window with origin T = s-6), and no fully invisible slot in s-18..s."""
    D, S, L = Q.shape
    cs = np.zeros((D, S + 1, L), np.int16); cs[:, 1:] = np.cumsum(Q, 1, dtype=np.int16)
    s = np.arange(18, S)
    est = (cs[:, s - 6] - cs[:, s - 18]).max(2) >= 2
    qa = Q.any(2); ea = E.any(2)
    cq = np.zeros((D, S + 1), np.int32); cq[:, 1:] = np.cumsum(qa, 1)
    cb = np.zeros((D, S + 1), np.int32); cb[:, 1:] = np.cumsum(~ea, 1)
    recent = (cq[:, s] - cq[:, s - 5]) > 0
    blank = (cb[:, s + 1] - cb[:, s - 18]) > 0
    new = np.zeros((D, S), bool)
    new[:, s] = qa[:, s] & ~recent & ~est & ~blank
    return new


def pv2_day_tables(V, Q, E, cut, new):
    """per-day sums (float32): A*/B* = queued / visible cells at onset events (box +-6, +-12 slots, whole day), N* = events,
    S2 / C2 = visible v / v_cut sum / count (box +-2)."""
    nf = new.astype(np.float32)
    A = (Q & new[:, :, None]).astype(np.float32); B = (E & new[:, :, None]).astype(np.float32)
    out = {}
    for k in (6, 12):
        out[f'A{k}'] = pv2_box(A, k); out[f'B{k}'] = pv2_box(B, k); out[f'N{k}'] = pv2_box(nf, k)
    out['Aal'] = A.sum(1); out['Bal'] = B.sum(1); out['Nal'] = nf.sum(1)
    del A, B
    with np.errstate(all='ignore'):
        vr = np.where(E, V / cut[None, None, :], 0).astype(np.float32)
    out['S2'] = pv2_box(vr, 2); out['C2'] = pv2_box(E.astype(np.float32), 2)
    return out


def pv2_class_tot(T, dow, mask):
    """[3, ...] sums of the per-day tables over the days in `mask`, by class (0 weekday, 1 weekend, 2 all)."""
    wk = np.asarray(dow) >= 5
    return {n: np.stack([a[m].sum(0) for m in (mask & ~wk, mask & wk, mask)]).astype(np.float32) for n, a in T.items()}


def pv2_class_cum(T, dow):
    """[3, D+1, ...] sums of the per-day tables over the days < j (timeline order), by class."""
    wk = np.asarray(dow) >= 5; D = len(wk); out = {}
    for n, a in T.items():
        c = np.zeros((3, D + 1) + a.shape[1:], np.float32)
        for ci, m in enumerate((~wk, wk, np.ones(D, bool))):
            c[ci, 1:] = np.cumsum(a * m.reshape((-1,) + (1,) * (a.ndim - 1)), 0)
        out[n] = c
    return out


def _pv2_get(tab, n, cc, s, li, jj=None):
    a = tab[n]
    pre = (cc,) if jj is None else (cc, jj)
    if n in ('Aal', 'Bal'):
        return a[pre + (li,)]
    if n == 'Nal':
        return a[pre]
    if n[0] == 'N':
        return a[pre + (s,)]
    return a[pre + (s, li)]


def pv2_row_feats(c, s6, tT, li, tot, day=None, dd=None, cum=None, jj=None, k=PV2_K):
    """the 5 table columns on_k6 / on_k12 / on_all / typ_T / typ_6 for rows (class c, slot T+6 s6, origin slot tT, link li) [n].
    train part = class totals `tot` minus the row's own day (per-day tables `day`, own day index `dd`) when `day` is given;
    scenario part (when `cum` is given) = cumulative tables at timeline position `jj` (days < the window's day), blended
    (1 - a) train + a scenario with a = n / (n + k)."""
    c = np.asarray(c); s6 = np.asarray(s6); tT = np.asarray(tT); li = np.asarray(li)
    c2 = np.full_like(c, 2)

    def tr(n, cc, s):
        v = _pv2_get(tot, n, cc, s, li).astype(np.float64)
        if day is not None:
            v = v - _pv2_get(day, n, dd, s, li)
        return v

    def sc(n, cc, s):
        return _pv2_get(cum, n, cc, s, li, jj).astype(np.float64)

    out = {}
    with np.errstate(all='ignore'):
        for name, (an, bn, cc, s) in (('on_k6', ('A6', 'B6', c, s6)), ('on_k12', ('A12', 'B12', c2, s6)),
                                      ('on_all', ('Aal', 'Bal', c2, s6))):
            a_, b_ = tr(an, cc, s), tr(bn, cc, s)
            v = np.where(b_ > 0, a_ / np.maximum(b_, 1e-9), 0.0)
            if cum is not None:
                a2, b2 = sc(an, cc, s), sc(bn, cc, s)
                w = b2 / (b2 + k)
                v = (1 - w) * v + w * np.where(b2 > 0, a2 / np.maximum(b2, 1e-9), 0.0)
            out[name] = v.astype(np.float32)
        for name, s in (('typ_T', tT), ('typ_6', s6)):
            a_, b_ = tr('S2', c, s), tr('C2', c, s)
            v = np.where(b_ > 0, a_ / np.maximum(b_, 1e-9), np.nan)
            if cum is not None:
                a2, b2 = sc('S2', c, s), sc('C2', c, s)
                w = (b2 / 5.0) / (b2 / 5.0 + k)
                v2 = np.where(b2 > 0, a2 / np.maximum(b2, 1e-9), np.nan)
                v = np.where(np.isnan(v), v2, np.where(np.isnan(v2), v, (1 - w) * v + w * v2))
            out[name] = v.astype(np.float32)
    return out


def pv2_window_rel(wi, li, prior, on12):
    """window-relative columns for flattened onset rows (window index wi, link li; ALL links of each window present)."""
    wi = np.asarray(wi); li = np.asarray(li)
    W = int(wi.max()) + 1 if len(wi) else 0
    out = {}
    for name, dname, v in (('r_prior', 'd_hotp', np.asarray(prior, np.float64)), ('r_on12', 'd_hot', np.asarray(on12, np.float64))):
        v = np.nan_to_num(v, nan=0.0)
        mx = np.zeros(W); np.maximum.at(mx, wi, v)
        out[name] = np.where(mx[wi] > 0, v / np.maximum(mx[wi], 1e-12), 0.0).astype(np.float32)
        o = np.lexsort((-v, wi)); first = np.r_[True, wi[o][1:] != wi[o][:-1]] if len(o) else np.zeros(0, bool)
        best = np.zeros(W, np.int64); best[wi[o][first]] = li[o][first]
        out[dname] = np.where(mx[wi] > 0, np.abs(li - best[wi]), 99).astype(np.float32)
    return out


def pv2_columns(X, cols, wi, li, c, s6, tT, tot, day=None, dd=None, cum=None, jj=None):
    """X [n, len(cols)] onset rows (all links of every window, prior columns already final) -> X with the 10 pv2 columns appended
    in the order PV2_A + PV2_B + PV2_C (the kernel slices the first 64 + 3 / 7 / 10 columns per candidate)."""
    f = pv2_row_feats(c, s6, tT, li, tot, day, dd, cum, jj)
    f.update(pv2_window_rel(wi, li, X[:, cols.index('prior')], f['on_k12']))
    f['anom'] = (X[:, cols.index('vr_last')] - f['typ_T']).astype(np.float32)
    return np.concatenate([X, np.stack([f[n] for n in PV2_A + PV2_B + PV2_C], 1)], 1)
# ============================================================================================ end of portable block (pv2)
