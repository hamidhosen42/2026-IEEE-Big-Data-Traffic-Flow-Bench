"""Row-level comparison of a rerun against the original Kaggle kernel outputs. Prints one 'PARITY ...' line per block.

Status per block:
  EXACT    every value identical (required for queue rows and path_flow)
  NUMERIC  state / path_flow differ by less than 1e-6 (absolute for state, relative for path_flow) - floating-point drift
  DIFF     state rows differ more (reported with max |d| and RMS; expected for the lean recipe)
  FAIL     queue rows differ, ids / row sets differ, or path_flow differs beyond NUMERIC
  SKIP     the original file is not attached
summary() prints the overall verdict: PASS (all EXACT), CLOSE (only NUMERIC / DIFF state blocks), FAIL otherwise.
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

RESULTS = []


def _line(check, block, status, **kw):
    s = f'PARITY {check} {block} status={status}' + ''.join(f' {k}={v}' for k, v in kw.items())
    print(s, flush=True)
    RESULTS.append(dict(check=check, block=block, status=status, **{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in kw.items()}))


_CACHE = {}


def _key_info(rel):
    """per key row (index submission_id): split ('validation' / 'private', from the key's split column or the month) and,
    for state rows, the blackout flag of tfb-i4d (timestamp in (origin, origin + 90 min] of an official window of the panel)."""
    if rel is None:
        return None
    if 'info' not in _CACHE:
        k = pd.read_csv(f'{rel}/submission_key.csv', dtype=str,
                        usecols=lambda c: c in ('submission_id', 'task', 'split', 'panel', 'timestamp', 'departure_time'))
        if 'split' in k.columns and k.split.notna().all():
            sp = k.split.str.strip().str.lower().replace({'val': 'validation', 'public': 'validation', 'test': 'private'}).to_numpy()
        else:
            t = k.timestamp.fillna(k['departure_time'] if 'departure_time' in k.columns else k.timestamp).fillna('')
            sp = np.where(t.str[:7] == '2031-03', 'validation', np.where(t.str[:7] == '2031-04', 'private', 'unknown'))
        inb = np.zeros(len(k), bool)
        st = np.flatnonzero((k.task == 'state').to_numpy())
        ts = pd.to_datetime(k.timestamp.to_numpy()[st], utc=True)
        for panel, idx in pd.Series(k.panel.to_numpy()[st]).groupby(k.panel.to_numpy()[st]).indices.items():
            tsp = ts[idx]
            hit = np.zeros(len(idx), bool)
            for split in ('validation', 'private'):
                wi_p = f'{rel}/task2/{panel}/{split}/window_index.csv'
                if os.path.exists(wi_p):
                    for o in pd.to_datetime(pd.read_csv(wi_p).forecast_origin, utc=True):
                        hit |= np.asarray((tsp > o) & (tsp <= o + pd.Timedelta(minutes=90)))
            inb[st[idx[hit]]] = True
        _CACHE['info'] = pd.DataFrame(dict(split=sp, blackout=inb), index=k.submission_id.astype('int64').to_numpy())
    return _CACHE['info']


def _windows(rel):
    """window_id -> (split, condition) from the released window indexes."""
    if rel is None:
        return None
    if 'win' not in _CACHE:
        w = []
        for f in sorted(glob.glob(f'{rel}/task2/*/*/window_index.csv')):
            sp = f.split('/')[-2]
            if sp != 'train':
                w.append(pd.read_csv(f, dtype=str)[['window_id', 'condition']].assign(split=sp))
        _CACHE['win'] = pd.concat(w).drop_duplicates('window_id').set_index('window_id') if w else None
    return _CACHE['win']


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def _stat(d):
    nd = int((d != 0).sum())
    mx = float(np.nanmax(np.abs(d))) if len(d) else 0.0
    rms = float(np.sqrt(np.nanmean(d.astype(np.float64) ** 2))) if len(d) else 0.0
    return nd, mx, rms


def compare_submission(ours, orig, check, tasks=('state', 'queue', 'odme'), rel=None):
    """two submission files (submission_id, task, speed_kmh, flow_vph, queue_pred, path_flow)."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    a = pd.read_csv(ours)
    b = pd.read_csv(orig)
    same = len(a) == len(b) and np.array_equal(a.submission_id.to_numpy(), b.submission_id.to_numpy())
    if same:
        _line(check, 'ids', 'EXACT', rows=len(a), bytes_identical=_sha256(ours) == _sha256(orig))
    else:
        _line(check, 'ids', 'FAIL', rows_ours=len(a), rows_orig=len(b), note='aligned on submission_id for the value checks')
        b = b.merge(a[['submission_id']], on='submission_id', how='inner')
        a = a.merge(b[['submission_id']], on='submission_id', how='inner')
    if not (a.task.to_numpy() == b.task.to_numpy()).all():
        _line(check, 'task', 'FAIL', n_diff=int((a.task.to_numpy() != b.task.to_numpy()).sum()))
    info = _key_info(rel) if rel is not None else None
    if info is not None:
        info = info.reindex(b.submission_id.to_numpy())
        sp, bo = info.split.to_numpy(), info.blackout.fillna(False).to_numpy(bool)
    else:
        sp = bo = None
    for task in tasks:
        m = (b.task == task).to_numpy()
        if task == 'state':
            for col in ('speed_kmh', 'flow_vph'):
                d = a[col].to_numpy()[m] - b[col].to_numpy()[m]
                nd, mx, rms = _stat(d)
                st = 'EXACT' if nd == 0 else ('NUMERIC' if mx < 1e-6 else 'DIFF')
                _line(check, f'state.{col}', st, rows=int(m.sum()), n_diff=nd, max_abs=f'{mx:.4g}', rms=f'{rms:.4g}')
                if nd and sp is not None:                 # DESIGN 4.7: per split, blackout and ordinary rows separately
                    for s in ('validation', 'private'):
                        for rows_name, rsel in (('all', np.ones(int(m.sum()), bool)), ('blackout', bo[m]), ('ordinary', ~bo[m])):
                            ms = (sp[m] == s) & rsel
                            nd2, mx2, rms2 = _stat(d[ms])
                            _line(check, f'state.{col}.{s}.{rows_name}', 'EXACT' if nd2 == 0 else ('NUMERIC' if mx2 < 1e-6 else 'DIFF'),
                                  rows=int(ms.sum()), n_diff=nd2, max_abs=f'{mx2:.4g}', rms=f'{rms2:.4g}')
        elif task == 'queue':
            qa = a.queue_pred.to_numpy()[m].astype(int)
            qb = b.queue_pred.to_numpy()[m].astype(int)
            nd = int((qa != qb).sum())
            kw = dict(rows=int(m.sum()), n_diff=nd, pos_ours=int(qa.sum()), pos_orig=int(qb.sum()))
            if sp is not None:
                for s in ('validation', 'private'):
                    kw[f'n_diff_{s}'] = int(((qa != qb) & (sp[m] == s)).sum())
            _line(check, 'queue', 'EXACT' if nd == 0 else 'FAIL', **kw)
        elif task == 'odme':
            pa = a.path_flow.to_numpy()[m].astype(np.float64)
            pb = b.path_flow.to_numpy()[m].astype(np.float64)
            d = pa - pb
            nd, mx, _ = _stat(d)
            relmax = float(np.max(np.abs(d) / np.maximum(np.abs(pb), 1e-12))) if len(d) else 0.0
            st = 'EXACT' if nd == 0 else ('NUMERIC' if relmax < 1e-6 else 'FAIL')
            _line(check, 'odme.path_flow', st, rows=int(m.sum()), n_diff=nd, max_abs=f'{mx:.4g}', max_rel=f'{relmax:.4g}')


def compare_queue_file(ours, orig, check, rel=None):
    """two queue files (window_id, timestamp, link_id, queue_pred): exact equality, per (split, condition) block."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    a = pd.read_csv(ours, dtype={'window_id': str, 'link_id': str})
    b = pd.read_csv(orig, dtype={'window_id': str, 'link_id': str})
    for x in (a, b):
        x['ts'] = pd.to_datetime(x.timestamp, utc=True)
    m = b[['window_id', 'ts', 'link_id', 'queue_pred']].merge(a[['window_id', 'ts', 'link_id', 'queue_pred']], on=['window_id', 'ts', 'link_id'],
                                                              how='outer', suffixes=('_orig', '_ours'), indicator=True)
    unmatched = int((m._merge != 'both').sum())
    m = m[m._merge == 'both']
    qa, qb = m.queue_pred_ours.astype(int).to_numpy(), m.queue_pred_orig.astype(int).to_numpy()
    nd = int((qa != qb).sum())
    _line(check, 'queue', 'EXACT' if nd == 0 and unmatched == 0 else 'FAIL', rows=len(m), unmatched=unmatched, n_diff=nd,
          pos_ours=int(qa.sum()), pos_orig=int(qb.sum()))
    win = _windows(rel)
    if win is not None:
        m = m.assign(split=m.window_id.map(win.split).fillna('?').to_numpy(), cond=m.window_id.map(win.condition).fillna('?').to_numpy(),
                     d=(qa != qb), a=qa, b=qb)
        for (s, c), g in m.groupby(['split', 'cond']):
            jw = []                                       # Jaccard of every window that differs (1 - J bounds |dIoU|)
            for wid in g.loc[g.d, 'window_id'].unique():
                w_all = g[g.window_id == wid]
                uni = int(((w_all.a == 1) | (w_all.b == 1)).sum())
                inter = int(((w_all.a == 1) & (w_all.b == 1)).sum())
                jw.append(1.0 if uni == 0 else inter / uni)
            # score bound (DESIGN 4.7): Jaccard distance is a metric, so per window |dIoU| <= 1 - J; one official window is worth
            # 0.15 / 40 = 0.00375 of the total score per IoU unit -> |dS| <= 0.00375 * sum_w (1 - J_w) for this block
            _line(check, f'queue.{s}.{c}', 'EXACT' if not g.d.any() else 'FAIL', rows=len(g), n_diff=int(g.d.sum()),
                  windows_diff=len(jw), min_jaccard=f'{min(jw):.4f}' if jw else '1', sum_1mJ=f'{sum(1 - j for j in jw):.4f}',
                  score_bound=f'{0.00375 * sum(1 - j for j in jw):.5f}', pos_ours=int(g.a.sum()), pos_orig=int(g.b.sum()))


STATE_KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']


def compare_state_file(ours, orig, check):
    """Task 1 member output (panel, timestamp, station_id, link_id, mask_regime, speed_kmh, flow_vph): row set, then speed /
    flow max |d| and RMS overall, per split (timestamp month) and per panel."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    dt = {'station_id': str, 'link_id': str, 'mask_regime': str, 'panel': str, 'timestamp': str}
    a, b = pd.read_csv(ours, dtype=dt), pd.read_csv(orig, dtype=dt)
    same_order = len(a) == len(b) and (a[STATE_KEYS].to_numpy() == b[STATE_KEYS].to_numpy()).all()
    if not same_order:
        m = b.merge(a, on=STATE_KEYS, how='outer', suffixes=('_o', '_r'), indicator=True)
        unmatched = int((m._merge != 'both').sum())
        m = m[m._merge == 'both']
        a = m[STATE_KEYS].assign(speed_kmh=m.speed_kmh_r, flow_vph=m.flow_vph_r)
        b = m[STATE_KEYS].assign(speed_kmh=m.speed_kmh_o, flow_vph=m.flow_vph_o)
    else:
        unmatched = 0
    _line(check, 'rows', 'EXACT' if same_order else ('FAIL' if unmatched else 'NUMERIC'), rows_ours=len(a), unmatched=unmatched,
          same_order=same_order, bytes_identical=_sha256(ours) == _sha256(orig))
    split = np.where(b.timestamp.str[:7].to_numpy() == '2031-03', 'validation', 'private')
    for col in ('speed_kmh', 'flow_vph'):
        d = a[col].to_numpy(np.float64) - b[col].to_numpy(np.float64)
        nd, mx, rms = _stat(d)
        _line(check, f'state.{col}', 'EXACT' if nd == 0 else ('NUMERIC' if mx < 1e-6 else 'DIFF'), rows=len(d), n_diff=nd,
              max_abs=f'{mx:.4g}', rms=f'{rms:.4g}')
        if nd:
            for s in ('validation', 'private'):
                ms = split == s
                nd2, mx2, rms2 = _stat(d[ms])
                _line(check, f'state.{col}.{s}', 'EXACT' if nd2 == 0 else ('NUMERIC' if mx2 < 1e-6 else 'DIFF'), rows=int(ms.sum()),
                      n_diff=nd2, max_abs=f'{mx2:.4g}', rms=f'{rms2:.4g}')
            for p, idx in pd.Series(np.arange(len(d))).groupby(b.panel.to_numpy()).indices.items():
                nd2, mx2, rms2 = _stat(d[idx])
                _line(check, f'state.{col}.{p}', 'EXACT' if nd2 == 0 else ('NUMERIC' if mx2 < 1e-6 else 'DIFF'), rows=len(idx),
                      n_diff=nd2, max_abs=f'{mx2:.4g}', rms=f'{rms2:.4g}')


def compare_npz(ours, orig, check):
    """every array of two .npz files: integer / bool / string arrays must be EXACT; float arrays report max |d|
    (probabilities: the decoded queue files are the decisive check)."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    A, B = np.load(ours, allow_pickle=True), np.load(orig, allow_pickle=True)
    ka, kb = set(A.files), set(B.files)
    _line(check, 'keys', 'EXACT' if ka == kb else 'FAIL', n_ours=len(ka), n_orig=len(kb), only_ours=sorted(ka - kb)[:5], only_orig=sorted(kb - ka)[:5])
    worst, n_exact, n_float_diff, bad = 0.0, 0, 0, []
    for k in sorted(ka & kb):
        x, y = A[k], B[k]
        if x.shape != y.shape:
            bad.append(f'{k}:shape'); continue
        if x.dtype.kind == 'f' and y.dtype.kind == 'f':
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            m = float(np.nanmax(d)) if d.size else 0.0
            if m == 0 and np.array_equal(np.isnan(x), np.isnan(y)):
                n_exact += 1
            else:
                n_float_diff += 1; worst = max(worst, m)
        elif np.array_equal(x, y):
            n_exact += 1
        else:
            bad.append(k)
    st = 'FAIL' if bad else ('EXACT' if n_float_diff == 0 else ('NUMERIC' if worst < 1e-6 else 'DIFF'))
    _line(check, 'arrays', st, arrays=len(ka & kb), exact=n_exact, float_diff=n_float_diff, max_abs=f'{worst:.4g}', mismatched=bad[:6])


def summary(save=None):
    st = [r['status'] for r in RESULTS]
    counts = {s: st.count(s) for s in ('EXACT', 'NUMERIC', 'DIFF', 'FAIL', 'SKIP')}
    verdict = 'FAIL' if counts['FAIL'] else ('PASS' if counts['NUMERIC'] + counts['DIFF'] == 0 else 'CLOSE')
    print(f'PARITY SUMMARY verdict={verdict} ' + ' '.join(f'{k}={v}' for k, v in counts.items()), flush=True)
    if save:
        with open(save, 'w') as f:
            json.dump(dict(verdict=verdict, counts=counts, results=RESULTS), f, indent=1)
    return verdict
