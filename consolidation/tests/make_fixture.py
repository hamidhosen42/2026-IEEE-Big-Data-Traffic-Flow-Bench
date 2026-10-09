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


def add_station_cols(rel, panel):
    """fake-release layers of a Task 2 panel: + date, milepost, occupancy (station_id = 'S_' + link_id) so the Task 1 /
    occupancy kernels can read them; topology + free_speed_kmh (the pseudo-window kernel reads it there)."""
    net = f'{rel}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str})
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).set_index('link_id')
    topo['free_speed_kmh'] = topo.link_id.map(lk.free_speed_kmh).fillna(105.0).to_numpy()
    topo.to_csv(f'{net}/lwr_mainline_topology.csv', index=False)
    mp = dict(zip(topo.link_id, topo.order_index.astype(float) * 0.5))
    lanes = lk.lanes.fillna(4).to_dict()
    for f in sorted(glob.glob(f'{rel}/corridors/{panel}/*/mainline_states*/*/*.parquet')):
        d = pd.read_parquet(f)
        d['link_id'] = d.link_id.astype(str)
        d['station_id'] = 'S_' + d.link_id
        d['date'] = d.timestamp.str[:10]
        d['milepost'] = d.link_id.map(mp).astype(float)
        ln = d.link_id.map(lanes).fillna(4).to_numpy()
        d['occupancy'] = np.clip(d.flow_vph.to_numpy() / np.maximum(d.speed_kmh.to_numpy(), 1.0) / ln / 150.0, 0, 1)
        d.to_parquet(f, index=False)
    for f in sorted(glob.glob(f'{rel}/task2/{panel}/*/window_history.parquet')):   # the occupancy kernels read it here too
        d = pd.read_parquet(f)
        ln = d.link_id.astype(str).map(lanes).fillna(4).to_numpy()
        d['occupancy'] = np.clip(d.flow_vph.to_numpy() / np.maximum(d.speed_kmh.to_numpy(), 1.0) / ln / 150.0, 0, 1)
        d.to_parquet(f, index=False)


def shift_month(src_dir, dst_dir, frm='2031-04', to='2031-03'):
    """synthesise a missing released month from another one (dates and file names shifted; values unchanged)."""
    for f in sorted(glob.glob(f'{src_dir}/*/*.parquet')):
        d = pd.read_parquet(f)
        for c in ('date', 'timestamp'):
            if c in d.columns:
                d[c] = d[c].astype(str).str.replace(frm, to, regex=False)
        out = f'{dst_dir}/{os.path.basename(os.path.dirname(f)).replace(frm, to)}/{os.path.basename(f).replace(frm.replace("-", "_"), to.replace("-", "_"))}'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        d.to_parquet(out, index=False)


def station_panel(rel, src_panel, dst_panel):
    """a station-level Task 1 panel from the real local D12_I405_S files: train full (days of the masked layer only, so the
    SMOKE day subset is the same in both layers), train masked, private masked + ramps, validation = private shifted to March."""
    S, D = f'{REAL}/corridors/{src_panel}', f'{rel}/corridors/{dst_panel}'
    if not os.path.exists(f'{D}/network'):
        shutil.copytree(f'{S}/network', f'{D}/network')
    mdays = {os.path.basename(f)[-18:-8] for f in glob.glob(f'{S}/train/mainline_states_masked/*/*.parquet')}
    for f in sorted(glob.glob(f'{S}/train/mainline_states/*/*.parquet')):
        if os.path.basename(f)[-18:-8] in mdays:
            out = f'{D}/train/mainline_states/{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}'
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy(f, out)
    for sub in ('train/mainline_states_masked', 'private/mainline_states_masked', 'private/ramp_states'):
        if os.path.exists(f'{S}/{sub}') and not os.path.exists(f'{D}/{sub}'):
            shutil.copytree(f'{S}/{sub}', f'{D}/{sub}')
    os.makedirs(f'{D}/train/ramp_states', exist_ok=True)
    shift_month(f'{D}/private/mainline_states_masked', f'{D}/validation/mainline_states_masked')
    shift_month(f'{D}/private/ramp_states', f'{D}/validation/ramp_states')
    return sorted(os.path.basename(f)[-18:-8] for f in glob.glob(f'{D}/train/mainline_states/*/*.parquet'))   # train days (files)


def t1_templates(rel, panel, splits, rng, cap=150000):
    """Task 1 templates = masked eligible cells (speed blanked) of the masked layer (subsampled)."""
    for split in splits:
        fs = sorted(glob.glob(f'{rel}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet'))
        d = pd.concat([pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'is_score_eligible', 'mask_regime']) for f in fs])
        t = d[(d.is_score_eligible == 1) & d.speed_kmh.isna()][['timestamp', 'station_id', 'link_id', 'mask_regime']]
        t = t.iloc[np.sort(rng.choice(len(t), min(cap, len(t)), replace=False))]
        write_csv(t.assign(speed_kmh=0, flow_vph=0), f'{rel}/task1/{panel}/{split}/sample_submission_state.csv')


def train_window_index(rel, panel, n_days=40):
    """the real official train window index of the panel (lab export), restricted to the first n_days train days of the
    fixture (the pseudo-window kernel's SMOKE switch loads 40 days and indexes every official window's day); a panel with
    none left gets 4 synthetic windows (the Task 1 members only check that the file exists)."""
    days = sorted(os.path.basename(f)[-18:-8].replace('_', '-') for f in glob.glob(f'{rel}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    out = f'{rel}/task2/{panel}/train/window_index.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.check_call([sys.executable, '-c', f"import lab; lab.window_index('{panel}', 'train').to_csv('{out}', index=False)"],
                          cwd=f'{ROOT}/analysis/t2lab', stdout=subprocess.DEVNULL)
    wi = pd.read_csv(out, dtype=str)
    wi = wi[wi.forecast_origin.str[:10].isin(set(days[:n_days]))]
    if len(wi) == 0:
        wi = pd.DataFrame([dict(window_id=f'Q2_{panel}_train_{i:03d}', panel=panel, split='train', date=days[5 * i + 2],
                                forecast_origin=f'{days[5 * i + 2]}T08:00:00Z', condition=('queue_onset', 'queue_ongoing')[i % 2],
                                history_coverage='0.8') for i in range(4)])
    wi.to_csv(out, index=False)


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
        add_station_cols(rel, p)
    for p in T2P:
        train_window_index(rel, p)
    # 2. station-level panels from the real local D12_I405_S files: D12_I405_S itself and a clone named D7_I10_W (the Task 1
    #    member SMOKE panels) with the real D7_I10_W window index; real D12_I405_N network files
    shutil.copytree(f'{REAL}/corridors/D12_I405_N/network', f'{rel}/corridors/D12_I405_N/network')
    station_panel(rel, 'D12_I405_S', 'D12_I405_S')
    wdays = station_panel(rel, 'D12_I405_S', 'D7_I10_W')
    tmp = f'{out}/_fakerel_w'
    subprocess.check_call([sys.executable, 'pvogf_fakerel.py', tmp, 'panels=D7_I10_W', 'ndays=3'], cwd=f'{ROOT}/analysis/t2lab',
                          stdout=subprocess.DEVNULL)
    for sp in ('validation', 'private'):
        os.makedirs(f'{rel}/task2/D7_I10_W/{sp}', exist_ok=True)
        shutil.copy(f'{tmp}/kaggle_public/task2/D7_I10_W/{sp}/window_index.csv', f'{rel}/task2/D7_I10_W/{sp}/window_index.csv')
    shutil.rmtree(tmp)
    train_window_index(rel, 'D7_I10_W')
    t1_templates(rel, 'D12_I405_S', ('train', 'validation'), rng)
    t1_templates(rel, 'D7_I10_W', ('train', 'validation', 'private'), rng)
    z = np.load(f'{ROOT}/kaggle_jobs/out/t2lab_full/pseudo_D7_I10_W.npz', allow_pickle=True)    # real pseudo windows (seed for
    keep = z['d'] < len(wdays)                                                                  # the member smoke runs)
    os.makedirs(f'{out}/kernels_t1/tfb-t2lab-full', exist_ok=True)
    np.savez_compressed(f'{out}/kernels_t1/tfb-t2lab-full/pseudo_D7_I10_W.npz', d=z['d'][keep], t=z['t'][keep], cond=z['cond'][keep])
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
