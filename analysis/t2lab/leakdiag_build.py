"""DIAGNOSTIC - NOT FOR SUBMISSION (analysis/LEAK_DIAG_2026-10-29.md).  Reads post-T data on purpose (see leakdiag_lib).

Builds, per panel, for the official-like train sets (offlike = selection-faithful greedy restarted daily; official = 40 + 40):
  * production-like base probabilities, out of fold:
      ongoing  fc_F03_np255 (F03 selection-faithful recipe, cached OOF probabilities, 4 folds)
      onset    CatBoost NORAMP 600 it / depth 8 / lr 0.08 (production v13-like), trained here on dense1500 onset rows outside
               the fold, fold-safe priors (as allfin_job.onset_cv), one seed
  * truth / ev / qlast from the lab row caches (observed queue state on observed cells)
  * leak features: on-ramp flows at T+5..T+30 (ramp-link tables learned on the training folds only) and post-buffer
    mainline cells at T+95..T+105 thinned to the val/private visibility
  * cell-level statistics of the ramp switch-off on all train cells.
Run:  .venv/bin/python analysis/t2lab/run_heavy.py -- .venv/bin/python analysis/t2lab/leakdiag_build.py
"""
import gc, json, os, time
import numpy as np
import pandas as pd

import lab
import leakdiag_lib as ld

F, ONSET, ONGOING = lab.F, lab.ONSET, lab.ONGOING
OUT = f'{lab.CACHE}/leakdiag'
RES = f'{lab.ROOT}/analysis/t2lab/results/leakdiag'
for _d in (OUT, RES):
    os.makedirs(_d, exist_ok=True)
SETS = ('offlike', 'official')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)


def onset_oof():
    """CatBoost NORAMP onset OOF probabilities for the onset windows of SETS: {(set, panel): P6 [W, 6, L]} (h = 6 only)."""
    f = f'{OUT}/onset_oof.npz'
    if os.path.exists(f):
        z = np.load(f)
        return {tuple(k.split('|')): z[k] for k in z.files}
    from catboost import CatBoostClassifier
    cols = lab.all_cols(); KEEPC = [i for i, c in enumerate(cols) if not c.startswith('ramp_')]
    out = {}
    for k in range(4):
        t1 = time.time(); Xt, yt = [], []
        for panel in lab.PANELS:
            rc = lab.rows(panel, 'dense1500', ONSET)
            m = rc.e & (rc.fold != k)
            Xt.append(rc.matrix(m, cols, k)[:, KEEPC]); yt.append(rc.y[m].astype(int))
        Xt = np.concatenate(Xt); yt = np.concatenate(yt)
        cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=0,
                                 allow_writing_files=False)
        cat.fit(Xt, yt)
        for s in SETS:
            for panel in lab.PANELS:
                ev = lab.rows(panel, s, ONSET)
                widx = np.where(ev.win.fold.to_numpy() == k)[0]
                if not len(widx):
                    continue
                me = np.isin(ev.wi, widx)
                p = cat.predict_proba(ev.matrix(me, cols, k)[:, KEEPC])[:, 1]
                P6 = out.setdefault((s, panel), np.zeros((len(ev.win), F, ev.L), np.float32))
                P6[ev.wi[me], F - 1, ev.li[me]] = p
        lab.log(f'onset CatBoost fold {k}: {len(Xt)} rows, pos {yt.mean():.4f}, {time.time() - t1:.0f}s')
        del Xt, yt, cat; gc.collect()
    np.savez_compressed(f, **{f'{s}|{p}': v for (s, p), v in out.items()})
    return out


def cell_stats(panel, P, R, ok, flag, A):
    """switch-off statistics on all train cells (daytime 06-21 h) of one panel."""
    Q, E = P['Q'], P['E']; vr = P['V'] / P['vf'][None, None, :]
    day = np.zeros(288, bool); day[72:252] = True
    okd = ok & day[None, :, None]; fl = flag & day[None, :, None]
    Qn = Q.copy(); Qn[:, :, 1:] |= Q[:, :, :-1]; Qn[:, :, :-1] |= Q[:, :, 1:]
    r = dict(panel=panel, L=Q.shape[2], ramp_links=len(A), ramp_link_share=len(A) / Q.shape[2],
             ramp_cell_usable=float(okd[:, :, A].mean()),
             prec_self=float(Q[fl & E].mean()), prec_pm1=float(Qn[fl & E].mean()),
             rec_self=float(fl[Q & okd].mean()), rec_deep=float(fl[E & (vr < 0.3) & okd].mean()),
             rec_shallow=float(fl[Q & (vr >= 0.3) & okd].mean()), false_flag_rate=float(fl[E & ~Q & okd].mean()),
             queued_share_deep=float((vr[Q] < 0.3).mean()))
    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 9]
    m = okd & E
    b = pd.cut(vr[m], bins)
    tab = pd.DataFrame({'ratio_mean': pd.Series(R[m]).groupby(b, observed=False).mean(),
                        'switched_off': pd.Series(fl[m]).groupby(b, observed=False).mean(),
                        'n': pd.Series(R[m]).groupby(b, observed=False).size()})
    tab['panel'] = panel
    return r, tab.reset_index().rename(columns={'index': 'speed_bin'})


def main():
    t0 = time.time()
    on_p = onset_oof()
    stats, tabs, mapinfo = [], [], []
    for panel in lab.PANELS:
        t1 = time.time()
        P = lab.load_panel(panel); L = len(P['links'])
        R, ok, flag, A, prof = ld.ramp_cubes(P['ON'], P['dow'])
        maps = {k: ld.mapping(flag, ok, P['Q'], P['E'], P['block'] != k, A) for k in range(4)}
        maps[4] = ld.mapping(flag, ok, P['Q'], P['E'], np.ones(len(P['days']), bool), A)
        np.savez_compressed(f'{OUT}/maps_{panel}.npz', A=A, prof=prof, **{f'{nm}{k}': maps[k][i] for k in maps for i, nm in enumerate(('Mpos', 'Rc', 'Mneg'))})
        Mp, Rc_, _ = maps[4]
        best = np.argmax(Mp + Rc_, 1)
        mapinfo.append(dict(panel=panel, ramp_links=len(A), best_is_self=float(np.mean(best == A)), best_within1=float(np.mean(np.abs(best - A) <= 1)),
                            best_within3=float(np.mean(np.abs(best - A) <= 3)), med_abs_offset=float(np.median(np.abs(best - A)))))
        r, tab = cell_stats(panel, P, R, ok, flag, A); stats.append(r); tabs.append(tab)
        rng = np.random.default_rng(12345)
        vis = P['E'] & (rng.random(P['E'].shape) < ld.VIS_PB)
        for s in SETS:
            for cond in (ONSET, ONGOING):
                rc = lab.rows(panel, s, cond)
                win = rc.win.reset_index(drop=True); W = len(win)
                if not W:
                    continue
                d = win.d.to_numpy(); t = win.t.to_numpy(); fold = win.fold.to_numpy()
                if cond == ONSET:
                    P6 = on_p[(s, panel)]
                else:
                    z = np.load(f'{lab.CACHE}/oof/fc_F03_np255/probs__{s}__{panel}.npz')
                    assert np.array_equal(z['d'], d) and np.array_equal(z['t'], t) and z['done'].all(), (panel, s)
                    P6 = z['P6'].astype(np.float32)
                FR = np.zeros((W, F, L, len(ld.RAMP_FEATS)), np.float32)
                for k in np.unique(fold):
                    ix = np.where(fold == k)[0]
                    FR[ix] = ld.ramp_feats(d[ix], t[ix], flag, ok, R, A, *maps[int(k)], L)
                FPB = ld.pb_feats(d, t, P['V'], vis, P['cut'], P['vf'], L)
                np.savez(f'{OUT}/cells_{panel}__{s}__{cond}.npz', d=d, t=t, fold=fold, truth=rc.truth, ev=rc.ev, qlast=rc.qlast,
                         P6=P6.astype(np.float32), FR=FR.astype(np.float16), FPB=FPB.astype(np.float16), A=A)
                win.to_pickle(f'{OUT}/cells_{panel}__{s}__{cond}.pkl')
        lab.log(panel, f'L {L}, ramp links {len(A)}, {time.time() - t1:.0f}s', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        del P, R, ok, flag, vis, maps; lab.drop_panel(panel); gc.collect()
    S = pd.DataFrame(stats); T = pd.concat(tabs, ignore_index=True); M = pd.DataFrame(mapinfo)
    S.to_csv(f'{RES}/cell_stats.csv', index=False); T.to_csv(f'{RES}/speed_bins.csv', index=False); M.to_csv(f'{RES}/mapping_info.csv', index=False)
    print('\n== switch-off statistics, daytime train cells (prec = P(link queued | ramp switched off); rec = P(switched off | link queued)) ==')
    print(S.round(3).to_string(index=False))
    pooled = T.groupby('speed_bin', observed=False).apply(lambda g: pd.Series(dict(ratio_mean=np.average(g.ratio_mean.fillna(0), weights=g.n),
                                                                                    switched_off=np.average(g.switched_off.fillna(0), weights=g.n), n=g.n.sum())))
    print('\n== pooled over panels: ramp flow / same-slot profile by mainline speed ratio of the attached link ==')
    print(pooled.round(3).to_string())
    print('\n== learned ramp -> link tables (all train days): best-matching link of each ramp link ==')
    print(M.round(3).to_string(index=False))
    lab.log(f'build done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
