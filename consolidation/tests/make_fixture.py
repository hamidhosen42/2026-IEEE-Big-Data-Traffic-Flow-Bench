#!/usr/bin/env python3
"""Local smoke fixture: a miniature kaggle_public tree plus stand-in kernel outputs, built only from local files.

  python tests/make_fixture.py OUT [--ndays 90]

OUT/kaggle_public   - D7_I10_E, D12_I5_S: real-shaped Task 2 inputs (train / validation / private layers, the official window
                      index / history / templates) rebuilt from the lab exports by analysis/t2lab/pvogf_fakerel.py (first
                      NDAYS train days); masked layers re-filed as mask_regime=R*/synthetic_mainline_<date>.parquet with a
                      station_id (= 'S_' + link_id) and a synthetic fd_parameters.csv
                    - D12_I405_S: the real local network files and private masked layer (real FD parameters, so the recon /
                      TV branch of t1_recon is exercised); D12_I405_N: real network files
                    - the 6 other panels: tiny synthetic networks (T4 loops over all ten panels)
                    - task4/<panel>/<split>: synthetic counts + weak prior; submission_key.csv / sample_submission.csv with
                      state rows (masked cells + cells inside the 90-min post-window gaps, about 2 % of the state rows),
                      every queue template row and every odme path
OUT/kernels         - stand-in outputs of the milestone-2 stages: state_submission_FTA.csv for every Task 1 member part of
                      the full recipe (shared synthetic truth + member noise) and random 0/1 queue files for tfb-t2occ
                      (fs4occ), tfb-t2ogpvogf (pvogfv), tfb-t2ogveto (veto10)
Nothing here is a competition answer; the fixture only tests code paths, I/O and gates.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)
ROOT = os.path.dirname(CONS)
sys.path.insert(0, CONS)
from tfb import config as C        # noqa: E402

REAL = f'{ROOT}/data/kaggle_public'
T2P = ['D7_I10_E', 'D12_I5_S']
DEP = {'validation': '2031-03-15T07:00:00Z', 'private': '2031-04-15T07:00:00Z'}


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def refile_masked(rel, panel):
    """R<k>/part_<date>.parquet -> mask_regime=R<k+1>/synthetic_mainline_<date>.parquet (+ station_id, mask_regime)."""
    for split in ('validation', 'private'):
        base = f'{rel}/corridors/{panel}/{split}/mainline_states_masked'
        for f in sorted(glob.glob(f'{base}/R[0-9]/*.parquet')):
            k = int(os.path.basename(os.path.dirname(f))[1:]) + 1
            d = pd.read_parquet(f)
            d['station_id'] = 'S_' + d.link_id.astype(str)
            d['mask_regime'] = f'R{k}'
            out = f'{base}/mask_regime=R{k}/synthetic_mainline_{os.path.basename(f)[-18:]}'
            os.makedirs(os.path.dirname(out), exist_ok=True)
            d.to_parquet(out, index=False)
            os.remove(f)
        for dd in glob.glob(f'{base}/R[0-9]'):
            os.rmdir(dd)


def fd_from_links(rel, panel):
    lk = pd.read_csv(f'{rel}/corridors/{panel}/network/links.csv', dtype={'link_id': str})
    lanes = lk.lanes.fillna(4).clip(lower=1)
    capl = (lk.capacity_vph.fillna(7200) / lanes).clip(1500, 2200)
    fd = pd.DataFrame(dict(link_id=lk.link_id, length_km=lk.length_km, lanes=lanes, free_speed_kmh=lk.free_speed_kmh.fillna(105),
                           capacity_vph=capl * lanes))
    fd['critical_density'] = fd.capacity_vph / fd.free_speed_kmh
    fd['k_jam'] = 100.0 * fd.lanes
    write_csv(fd, f'{rel}/corridors/{panel}/network/fd_parameters.csv')


def synth_network(rel, panel, rng, links=None):
    net = f'{rel}/corridors/{panel}/network'
    if links is None:
        links = [f'L{panel[-5:].replace("_", "")}-{i:03d}' for i in range(1, 11)]
        write_csv(pd.DataFrame(dict(link_id=links, order_index=range(len(links)))), f'{net}/lwr_mainline_topology.csv')
        write_csv(pd.DataFrame(dict(link_id=links, length_km=0.5, lanes=4, free_speed_kmh=105.0, capacity_vph=7600.0)), f'{net}/links.csv')
        fd = pd.read_csv(f'{net}/links.csv')
        fd['critical_density'] = fd.capacity_vph / fd.free_speed_kmh
        fd['k_jam'] = 400.0
        write_csv(fd, f'{net}/fd_parameters.csv')
    seg = links[:12]
    ps, inc = [], []
    for i in range(len(seg) - 2):
        for j in range(i + 2, min(i + 6, len(seg)) + 1):
            pid = f'P{i:03d}_{j:03d}'
            ps.append(dict(path_id=pid, origin_zone=f'Z{i}', destination_zone=f'Z{j}', link_seq=';'.join(seg[i:j])))
            inc += [dict(link_id=l, path_id=pid) for l in seg[i:j]]
    write_csv(pd.DataFrame(ps), f'{net}/path_set.csv')
    write_csv(pd.DataFrame(inc), f'{net}/path_link_incidence.csv')


def task4(rel, panel, rng):
    net = f'{rel}/corridors/{panel}/network'
    ps = pd.read_csv(f'{net}/path_set.csv', dtype=str)
    inc = pd.read_csv(f'{net}/path_link_incidence.csv', dtype=str)
    pids = ps.path_id.tolist(); lids = inc.link_id.drop_duplicates().tolist()
    A = np.zeros((len(lids), len(pids)))
    A[inc.link_id.map({l: i for i, l in enumerate(lids)}), inc.path_id.map({p: i for i, p in enumerate(pids)})] = 1
    for split in ('validation', 'private'):
        f = rng.gamma(2.0, 60.0, len(pids))
        meas = rng.random(len(lids)) < 0.6
        cnt = A[meas] @ f * rng.normal(1, 0.05, int(meas.sum()))
        d = f'{rel}/task4/{panel}/{split}'
        write_csv(pd.DataFrame(dict(link_id=np.array(lids)[meas], count=np.round(cnt, 3))), f'{d}/synthetic_link_counts.csv')
        write_csv(pd.DataFrame(dict(panel=panel, departure_time=DEP[split], path_id=pids, path_flow=np.round(f * rng.normal(1, 0.3, len(f)).clip(0.2), 3))),
                  f'{d}/synthetic_weak_prior.csv')
        write_csv(pd.DataFrame(dict(panel=panel, departure_time=DEP[split], path_id=pids, path_flow=0.0)), f'{d}/sample_submission_path_flow.csv')


def state_rows(rel, panel, split, rng, gap_links=10, gap_share=0.025, cap=60000):
    fs = sorted(glob.glob(f'{rel}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'is_score_eligible', 'mask_regime']) for f in fs])
    d['station_id'] = d.station_id.astype(str); d['link_id'] = d.link_id.astype(str)
    reg = d.assign(date=d.timestamp.str[:10]).groupby('date').mask_regime.first()
    tgt = d[(d.is_score_eligible == 1) & d.speed_kmh.isna()][['timestamp', 'station_id', 'link_id', 'mask_regime']]
    wi_p = f'{rel}/task2/{panel}/{split}/window_index.csv'
    if os.path.exists(wi_p):
        st = d[['station_id', 'link_id']].drop_duplicates('station_id')
        st = st.iloc[rng.choice(len(st), min(gap_links, len(st)), replace=False)]
        gaps = []
        for o in pd.to_datetime(pd.read_csv(wi_p).forecast_origin, utc=True):
            for h in range(1, 19):
                ts = (o + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ')
                gaps.append(st.assign(timestamp=ts))
        gaps = pd.concat(gaps)
        gaps = gaps[gaps.timestamp.str[:10].isin(reg.index)]
        gaps['mask_regime'] = gaps.timestamp.str[:10].map(reg)
        n_other = int(len(gaps) / gap_share * (1 - gap_share))
        tgt = tgt[~tgt.timestamp.isin(set(gaps.timestamp))]                 # other rows: outside every post-window gap
        tgt = tgt.iloc[np.sort(rng.choice(len(tgt), min(n_other, len(tgt)), replace=False))]
        out = pd.concat([tgt, gaps[['timestamp', 'station_id', 'link_id', 'mask_regime']]])
    else:
        out = tgt.iloc[np.sort(rng.choice(len(tgt), min(cap, len(tgt)), replace=False))]
    return out.drop_duplicates(['timestamp', 'station_id']).assign(panel=panel, split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--ndays', type=int, default=90)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    rel = f'{out}/kaggle_public'
    rng = np.random.default_rng(2026)
    # 1. real-shaped Task 2 panels from the lab exports
    tmp = f'{out}/_fakerel'
    subprocess.check_call([sys.executable, 'pvogf_fakerel.py', tmp, f'panels={",".join(T2P)}', f'ndays={a.ndays}'], cwd=f'{ROOT}/analysis/t2lab')
    shutil.move(f'{tmp}/kaggle_public', rel)
    shutil.rmtree(tmp)
    for p in T2P:
        refile_masked(rel, p)
        fd_from_links(rel, p)
    # 2. real D12_I405_S / D12_I405_N files
    for p in ('D12_I405_S', 'D12_I405_N'):
        shutil.copytree(f'{REAL}/corridors/{p}/network', f'{rel}/corridors/{p}/network')
    shutil.copytree(f'{REAL}/corridors/D12_I405_S/private/mainline_states_masked', f'{rel}/corridors/D12_I405_S/private/mainline_states_masked')
    for f in ('README.md',):
        shutil.copy(f'{REAL}/{f}', f'{rel}/{f}')
    shutil.copytree(f'{REAL}/config', f'{rel}/config')
    # 3. networks + Task 4 for all ten panels
    for p in C.PANELS_ALL:
        if not os.path.exists(f'{rel}/corridors/{p}/network/path_set.csv'):
            links = None
            if os.path.exists(f'{rel}/corridors/{p}/network/lwr_mainline_topology.csv'):
                links = pd.read_csv(f'{rel}/corridors/{p}/network/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').link_id.tolist()
            synth_network(rel, p, rng, links)
        task4(rel, p, rng)
    # 4. submission key + sample submission (+ task1 templates)
    st = pd.concat([x for p in ('D7_I10_E', 'D12_I5_S', 'D12_I405_S') for s in ('validation', 'private')
                    for x in [state_rows(rel, p, s, rng)] if x is not None], ignore_index=True)
    st = st.iloc[rng.permutation(len(st))].reset_index(drop=True)            # no ordering assumption in the stages
    for (p, s), g in st.groupby(['panel', 'split']):
        write_csv(g[['timestamp', 'station_id', 'link_id', 'mask_regime']].assign(speed_kmh=0, flow_vph=0), f'{rel}/task1/{p}/{s}/sample_submission_state.csv')
    st['task'] = 'state'
    q = []
    for f in sorted(glob.glob(f'{rel}/task2/*/*/sample_submission_queue.csv')):
        sp = f.split('/')[-2]
        if sp == 'train':
            continue
        t = pd.read_csv(f, dtype=str)[['window_id', 'timestamp', 'link_id']]
        q.append(t.assign(task='queue', split=sp, panel=f.split('/')[-3]))
    q = pd.concat(q, ignore_index=True)
    od = []
    for p in C.PANELS_ALL:
        ps = pd.read_csv(f'{rel}/corridors/{p}/network/path_set.csv', dtype=str)
        for s in ('validation', 'private'):
            od.append(ps[['path_id', 'origin_zone', 'destination_zone']].assign(task='odme', split=s, panel=p, departure_time=DEP[s]))
    od = pd.concat(od, ignore_index=True)
    cols = ['submission_id', 'task', 'split', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'window_id', 'departure_time',
            'path_id', 'origin_zone', 'destination_zone']
    key = pd.concat([st, q, od], ignore_index=True)
    key['submission_id'] = np.arange(1, len(key) + 1)
    key = key.reindex(columns=cols)
    key.to_csv(f'{rel}/submission_key.csv', index=False)
    pd.DataFrame(dict(submission_id=key.submission_id, task=key.task, speed_kmh=0, flow_vph=0, queue_pred=0, path_flow=0)).to_csv(
        f'{rel}/sample_submission.csv', index=False)
    print('key rows', key.task.value_counts().to_dict(), flush=True)
    # 5. stand-in member outputs (one shared synthetic truth, member-specific noise)
    kern = f'{out}/kernels'
    fdp = pd.concat([pd.read_csv(f'{rel}/corridors/{p}/network/fd_parameters.csv', dtype={'link_id': str}).assign(panel=p)
                     .groupby(['panel', 'link_id'], as_index=False)[['free_speed_kmh', 'lanes']].mean() for p in ('D7_I10_E', 'D12_I5_S', 'D12_I405_S')])
    s = st.merge(fdp, on=['panel', 'link_id'], how='left')
    vf = s.free_speed_kmh.fillna(105).to_numpy(); ln = s.lanes.fillna(4).to_numpy()
    cong = rng.random(len(s)) < 0.15
    v0 = np.where(cong, vf * rng.uniform(0.2, 0.55, len(s)), vf * rng.uniform(0.85, 1.0, len(s)))
    q0 = np.where(cong, ln * v0 * rng.uniform(25, 45, len(s)), ln * rng.uniform(300, 1600, len(s)))
    half_a, half_b = set(C.PANELS_ALL[:5]), set(C.PANELS_ALL[5:])
    for i, k in enumerate(C.t1_member_kernels('full')):
        member, part = k[:-1], k[-1]
        if member == 'tfb-t1bspec':
            panels = {'a': set(C.PANELS_ALL[:4]), 'b': set(C.PANELS_ALL[5:]), 'c': {'D7_I405_N'}}[part]
        else:
            panels = half_a if part == 'a' else half_b
        r = np.random.default_rng(100 + i // 2 + (0 if member != 'tfb-t1bspec' else 50))
        sp = np.clip(v0 + r.normal(0, 2.0, len(s)), 3, None); fl = np.clip(q0 + r.normal(0, 40.0, len(s)), 20, None)
        m = s.panel.isin(panels).to_numpy()
        df = s.loc[m, ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].assign(speed_kmh=sp[m], flow_vph=fl[m])
        write_csv(df, f'{kern}/{k}/state_submission_FTA.csv')
    # 6. stand-in milestone-2 queue files (every template row)
    qt = q[['window_id', 'timestamp', 'link_id']]
    for k, f in (('tfb-t2occ', 'queue_submission_fs4occ.csv'), ('tfb-t2ogpvogf', 'queue_submission_pvogfv.csv'), ('tfb-t2ogveto', 'queue_submission_veto10.csv')):
        write_csv(qt.assign(queue_pred=(rng.random(len(qt)) < 0.05).astype(int)), f'{kern}/{k}/{f}')
    print('FIXTURE DONE', out, flush=True)


if __name__ == '__main__':
    main()
