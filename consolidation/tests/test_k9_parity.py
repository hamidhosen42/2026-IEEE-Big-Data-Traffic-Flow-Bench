#!/usr/bin/env python3
"""Level-1 local parity of the A3 / B3 light stages (no Kaggle, no submission_key.csv needed):

  python tests/test_k9_parity.py TMPDIR          (heavy-ish: 3 LRTCs on one real panel-month, ~1-3 min)

A synthetic release is built around the REAL local D12_I405_S private month (data/kaggle_public: network + masked states,
symlinked), as analysis/t1lab/t1lrtc_k9_fntest.py does: state rows = every released cell with a missing flow, 5 official-like
private windows (blackout rows), 2 March windows, 30 queue rows per window, 4 odme rows. Synthetic member outputs for every
member part of the A3 recipe (8 old FTA + 6 FD members a/b, t1bspec a/b, t1catb / t1catb2 / t1xgb a-d) = per-station time
interpolation of the released cells + member noise. Nothing here is a competition answer.

Each ORIGINAL kernel (consolidation/lineage_src/<kernel>.py, /kaggle paths redirected to TMPDIR) and the VENDORED stage
(tfb/stages/*, run through tfb.runner.Runner exactly as run_all.py runs it) process the same inputs:
  1. t1_mix_k5d  vs tfb-k5d  (base tfb-k3/submission_FDRTV.csv)        -> submission_K5D.csv           byte-identical
  2. t1_lrtc     vs tfb-k9   (base = the K5D of step 1)                -> submission_LRTC/FDRTV.csv    byte-identical
  3. assemble_k10 (pick B3, original mode: base tfb-k9 FDRTV, the 2 private parts) vs tfb-k10 -> submission_K10.csv  byte-identical
  4. production wiring (pick A3 / B3 with the real tfb/config.py constants): base = tfb-k9 FDRTV with every queue_pred
     set to 0 (the T4 skeleton has no queue rows), the 4 condition x split parts carry the queue rows (March + April
     ongoing / onset as tfb-j4 had them) -> B3 file == tfb-k10 submission_K10.csv byte-identical; A3 file == tfb-k9
     submission_FDRTV.csv on every parsed value (tfb-k9 wrote its file straight from memory, the entry script's A3 file
     passes one more pandas write / read cycle in the assembly, so some floats are re-formatted in the last digit).
Exit code 1 on any failure.
"""
import glob
import hashlib
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)
ROOT = os.path.dirname(CONS)
sys.path.insert(0, CONS)
from tfb import config as C                  # noqa: E402
from tfb.runner import Runner                # noqa: E402

PANEL, SPLIT = 'D12_I405_S', 'private'
SRC_REL = os.path.join(ROOT, 'data', 'kaggle_public')
LSRC = os.path.join(CONS, 'lineage_src')
FAILS = []


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def same_file(a, b, what):
    ok = os.path.exists(a) and os.path.exists(b) and sha(a) == sha(b)
    extra = ''
    if not ok and os.path.exists(a) and os.path.exists(b):
        x, y = pd.read_csv(a), pd.read_csv(b)
        extra = f' | rows {len(x)} vs {len(y)}'
        if x.shape == y.shape:
            for c in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'):
                d = np.abs(x[c].to_numpy(float) - y[c].to_numpy(float))
                extra += f' | {c}: {int((d > 0).sum())} differ, max {d.max():.3g}'
    check(ok, f'{what}: vendored == original (sha256 {sha(a)[:12] if os.path.exists(a) else "missing"}){extra}')


def run_original(kernel, inp, outd):
    """the original kernel source with /kaggle/input -> inp, /kaggle/working -> outd (as on Kaggle: CWD = working dir)."""
    os.makedirs(outd, exist_ok=True)
    src = open(f'{LSRC}/{kernel}.py').read().replace('/kaggle/input', inp).replace('/kaggle/working', outd)
    cwd = os.getcwd()
    os.chdir(outd)
    t0 = time.time()
    try:
        exec(compile(src, f'{kernel}.py', 'exec'), {'__name__': '__main__'})
    finally:
        os.chdir(cwd)
    print(f'[orig] {kernel} done in {time.time() - t0:.1f}s', flush=True)


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def write(df, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)


def tinterp(Cb):
    out = Cb.copy()
    for di in range(Cb.shape[0]):
        for s in range(Cb.shape[2]):
            y = Cb[di, :, s]
            ok = np.isfinite(y)
            if ok.any():
                out[di, :, s] = np.interp(np.arange(288), np.flatnonzero(ok), y[ok])
    return out


def build(T):
    """synthetic release + inputs; returns the queue key frame."""
    rng = np.random.default_rng(0)
    REL = f'{T}/data/kaggle_public'
    link(f'{SRC_REL}/corridors/{PANEL}/network', f'{REL}/corridors/{PANEL}/network')
    link(f'{SRC_REL}/corridors/{PANEL}/{SPLIT}/mainline_states_masked', f'{REL}/corridors/{PANEL}/{SPLIT}/mainline_states_masked')
    files = sorted(glob.glob(f'{SRC_REL}/corridors/{PANEL}/{SPLIT}/mainline_states_masked/*/*.parquet'))
    d = pd.concat([pd.read_parquet(f, columns=['date', 'timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'mask_regime'])
                   for f in files], ignore_index=True)
    d['station_id'] = d.station_id.astype(str); d['link_id'] = d.link_id.astype(str)
    d['ts'] = pd.to_datetime(d.timestamp, utc=True)
    days = sorted(d.date.astype(str).unique()); stations = sorted(d.station_id.unique())
    d['di'] = d.date.astype(str).map({x: i for i, x in enumerate(days)}); d['slot'] = d.ts.dt.hour * 12 + d.ts.dt.minute // 5
    d['si'] = d.station_id.map({x: i for i, x in enumerate(stations)})
    Qc = np.full((len(days), 288, len(stations)), np.nan); Vc = Qc.copy()
    fq = d.flow_vph.notna().to_numpy(); fv = d.speed_kmh.notna().to_numpy()
    Qc[d.di[fq], d.slot[fq], d.si[fq]] = d.flow_vph[fq]; Vc[d.di[fv], d.slot[fv], d.si[fv]] = d.speed_kmh[fv]
    tgt = d[d.flow_vph.isna()].sort_values(['ts', 'station_id']).reset_index(drop=True)
    print(f'released month {PANEL} {SPLIT}: days {len(days)} stations {len(stations)} | synthetic state rows {len(tgt)}', flush=True)
    st = tgt[['timestamp', 'station_id', 'link_id', 'mask_regime']].assign(task='state', panel=PANEL, window_id='')
    st['mask_regime'] = st.mask_regime.astype(str)
    # official-like windows: 5 April (blackout rows of the state key), 2 March (queue rows only)
    wins = pd.DataFrame({'window_id': [f'P{i}' for i in range(5)] + ['V0', 'V1'],
                         'forecast_origin': ['2031-04-03T07:00:00Z', '2031-04-08T16:30:00Z', '2031-04-15T17:00:00Z', '2031-04-22T07:30:00Z',
                                             '2031-04-28T16:00:00Z', '2031-03-04T07:00:00Z', '2031-03-11T16:00:00Z'],
                         'condition': ['queue_onset', 'queue_ongoing'] * 3 + ['queue_onset'],
                         'split': ['private'] * 5 + ['validation'] * 2})
    for sp, g in wins.groupby('split'):
        write(g[['window_id', 'forecast_origin', 'condition']], f'{REL}/task2/{PANEL}/{sp}/window_index.csv')
    links = sorted(tgt.link_id.unique())[:5]
    q = []
    for w in wins.itertuples():
        o = pd.Timestamp(w.forecast_origin)
        for h in range(1, 7):
            for lk in links:
                q.append(dict(task='queue', panel=PANEL, timestamp=(o + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                              station_id='', link_id=lk, mask_regime='', window_id=w.window_id))
    q = pd.DataFrame(q)
    od = pd.DataFrame({'task': 'odme', 'panel': PANEL, 'timestamp': '', 'station_id': '', 'link_id': '', 'mask_regime': '', 'window_id': ''}, index=range(4))
    key = pd.concat([st, q, od], ignore_index=True)[['task', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'window_id']]
    key.insert(0, 'submission_id', np.arange(1, len(key) + 1))
    key.to_csv(f'{REL}/submission_key.csv', index=False)
    key[['submission_id']].to_csv(f'{REL}/sample_submission.csv', index=False)
    # member outputs: time interpolation of the released cells + member noise (one shared "truth")
    Vi, Qi = tinterp(Vc), tinterp(Qc)
    td, tt, ts_ = tgt.di.to_numpy(), tgt.slot.to_numpy(), tgt.si.to_numpy()
    v_t = np.clip(np.nan_to_num(Vi[td, tt, ts_], nan=100.0), 5, 130)
    q_t = np.maximum(np.nan_to_num(Qi[td, tt, ts_], nan=3000.0), 60)
    cols = st[['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].reset_index(drop=True)
    kernels = C.t1_member_kernels('full', 'A3')
    for i, k in enumerate(kernels):
        r = np.random.default_rng(1000 + i)
        sp = np.clip(v_t * r.normal(1, 0.01, len(cols)), 3, None); fl = np.maximum(q_t * r.normal(1, 0.02, len(cols)), 20)
        if k.startswith('tfb-t1bspec'):
            sp = np.clip(v_t * r.normal(0.9, 0.05, len(cols)), 3, None)
        # every row in the first part of the member, the other parts empty (the stage concatenates the parts)
        first = k[-1] == 'a'
        df = cols.assign(speed_kmh=sp, flow_vph=fl) if first else cols.iloc[:0].assign(speed_kmh=[], flow_vph=[])
        write(df, f'{T}/in/{k}/state_submission_FTA.csv')
    # base of tfb-k5d (= the tfb-k3 file in the chain): queue / odme rows matter, state rows are recomputed
    n = len(key); iss = (key.task == 'state').to_numpy(); isq = (key.task == 'queue').to_numpy()
    base = pd.DataFrame({'submission_id': key.submission_id, 'task': key.task, 'speed_kmh': 0.0, 'flow_vph': 0.0,
                         'queue_pred': np.where(isq, (rng.random(n) < 0.3).astype(int), 0),
                         'path_flow': np.where(key.task == 'odme', rng.uniform(5, 500, n).round(6), 0.0)})
    base.loc[iss, 'speed_kmh'] = v_t * 1.01; base.loc[iss, 'flow_vph'] = q_t * 0.99
    write(base, f'{T}/in/tfb-k3/submission_FDRTV.csv')
    # private queue parts of tfb-k10 (every queue key row; only the private rows of their condition are used)
    kq = key[isq][['window_id', 'timestamp', 'link_id']]
    for kern, f, seed in (('tfb-t2osp', 'queue_submission_base.csv', 1), ('tfb-t2ogveto', 'queue_submission_veto10.csv', 2)):
        write(kq.assign(queue_pred=(np.random.default_rng(seed).random(len(kq)) < 0.4).astype(int)), f'{T}/parts/{kern}/{f}')
    return key, wins


def queue_parts_from(fpath, key, wins, T, parts):
    """part files (production QUEUE_PARTS) whose rows = the queue_pred of an existing submission file (the tfb-j4 rows)."""
    sub = pd.read_csv(fpath)
    kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].merge(sub[['submission_id', 'queue_pred']], on='submission_id')
    for kern, f, cond, split in parts:
        fn = f.split('|')[0]
        p = f'{T}/{kern}/{fn}'
        if not os.path.exists(p):
            write(kq[['window_id', 'timestamp', 'link_id', 'queue_pred']], p)


def zero_queue_text(src, dst):
    """copy a submission CSV with queue_pred := 0 on queue rows, every other byte unchanged (no float re-formatting)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as f, open(dst, 'w') as g:
        hdr = f.readline(); g.write(hdr)
        cols = hdr.strip().split(','); it, iq = cols.index('task'), cols.index('queue_pred')
        for line in f:
            x = line.rstrip('\n').split(',')
            if x[it] == 'queue':
                x[iq] = '0'
            g.write(','.join(x) + '\n')


def main():
    T = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '/tmp/tfb_k9_parity')
    shutil.rmtree(T, ignore_errors=True)
    t0 = time.time()
    key, wins = build(T)
    REL = f'{T}/data/kaggle_public'
    O = f'{T}/orig'
    # ---------------- originals (chain of Kaggle kernels, each reading its upstream as /kaggle/input/**/<kernel>/**/<file>)
    link(f'{T}/data', f'{T}/in/data')
    run_original('tfb-k5d', f'{T}/in', f'{O}/tfb-k5d')
    link(f'{T}/data', f'{T}/in_k9/data'); link(f'{O}/tfb-k5d', f'{T}/in_k9/tfb-k5d')
    run_original('tfb-k9', f'{T}/in_k9', f'{O}/tfb-k9')
    link(f'{T}/data', f'{T}/in_k10/data'); link(f'{O}/tfb-k9', f'{T}/in_k10/tfb-k9')
    for k in ('tfb-t2osp', 'tfb-t2ogveto'):
        link(f'{T}/parts/{k}', f'{T}/in_k10/{k}')
    run_original('tfb-k10', f'{T}/in_k10', f'{O}/tfb-k10')
    # ---------------- vendored stages through the runner (original mode: same bases / parts as the originals)
    roots = [dict(root=f'{T}/in', mode='flat'), dict(root=f'{T}/parts', mode='flat')]
    ov = {'t1_mix_k5d': dict(base=['tfb-k3', 'submission_FDRTV.csv']),
          'assemble': dict(parts=[list(p) for p in C.QUEUE_PARTS_ORIGINAL['B3']], march_from_parts=False, require_full_cover=False)}
    W = f'{T}/work'
    r = Runner(REL, W, 'B3', 'full', input_roots=roots, overrides=ov, log=print)
    for st in ('t1_mix_k5d', 't1_lrtc', 'assemble'):
        r.run_stage(st)
    same_file(f'{W}/kernels/tfb-k5d/submission_K5D.csv', f'{O}/tfb-k5d/submission_K5D.csv', '1 t1_mix_k5d vs tfb-k5d: submission_K5D.csv')
    for f in ('submission_LRTC.csv', 'submission_FDRTV.csv'):
        same_file(f'{W}/kernels/tfb-k9/{f}', f'{O}/tfb-k9/{f}', f'2 t1_lrtc vs tfb-k9: {f}')
    same_file(f'{W}/kernels/tfb-k10/submission_K10.csv', f'{O}/tfb-k10/submission_K10.csv', '3 assemble_k10 (B3, original mode) vs tfb-k10: submission_K10.csv')
    # rerun of a cached stage must be a no-op, a second run of the stage must be byte-identical
    check(r.run_stage('t1_lrtc') == 'cached', 'runner cache: t1_lrtc is skipped on a rerun with unchanged inputs')
    # ---------------- production wiring (real config, no overrides)
    W2 = f'{T}/work2'
    zero_queue_text(f'{O}/tfb-k9/submission_FDRTV.csv', f'{W2}/kernels/tfb-k9/submission_FDRTV.csv')
    P2 = f'{T}/parts2'
    priv_b = {(k, f) for k, f, _, _ in C.QUEUE_PARTS_ORIGINAL['B3']}
    for k, f in priv_b:                                    # B3 private parts = the original tfb-k10 inputs
        os.makedirs(f'{P2}/{k}', exist_ok=True); shutil.copy(f'{T}/parts/{k}/{f}', f'{P2}/{k}/{f}')
    queue_parts_from(f'{O}/tfb-k9/submission_FDRTV.csv', key, wins, P2, [p for p in C.QUEUE_PARTS['A3']])   # tfb-j4 rows
    for pick, ref in (('A3', f'{O}/tfb-k9/submission_FDRTV.csv'), ('B3', f'{O}/tfb-k10/submission_K10.csv')):
        r2 = Runner(REL, W2, pick, 'full', input_roots=[dict(root=P2, mode='flat')], log=print)
        r2.run_stage('assemble')
        dst = r2.finalize()
        x, y = pd.read_csv(dst), pd.read_csv(ref)
        exact = x.shape == y.shape and (x.submission_id.to_numpy() == y.submission_id.to_numpy()).all() and (x.task == y.task).all() and all(
            np.array_equal(x[c].to_numpy(), y[c].to_numpy()) for c in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'))
        nq = int((x.queue_pred != y.queue_pred).sum()) if x.shape == y.shape else -1
        with open(dst) as f1, open(ref) as f2:
            ntext = sum(l1 != l2 for l1, l2 in zip(f1, f2))
        check(exact, f'4 production assembly {pick} ({C.ASSEMBLE_MODULE[pick]}, 4 parts on the zero-queue base): final {os.path.basename(dst)} == '
                     f'{C.PICK_REFERENCE[pick][0]}/{C.PICK_REFERENCE[pick][1]} on every parsed value (EXACT; queue rows differing {nq}) | '
                     f'text lines differing {ntext} (CSV float re-formatting, see README "Determinism")')
        if pick == 'B3':
            check(ntext == 0, '4 B3 final file byte-identical to tfb-k10/submission_K10.csv')
    lr = pd.read_csv(f'{O}/tfb-k9/submission_LRTC.csv'); k5 = pd.read_csv(f'{O}/tfb-k5d/submission_K5D.csv')
    iss = (k5.task == 'state').to_numpy()
    check(int((lr.flow_vph.to_numpy() != k5.flow_vph.to_numpy())[iss].sum()) > 0.9 * iss.sum() and int((lr.speed_kmh != k5.speed_kmh).sum()) > 0,
          f'non-trivial test: steps M / B changed {int((lr.flow_vph != k5.flow_vph).sum())} flows and {int((lr.speed_kmh != k5.speed_kmh).sum())} speeds')
    print(f'K9 PARITY {"FAILED" if FAILS else "OK"} ({time.time() - t0:.0f}s)', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
