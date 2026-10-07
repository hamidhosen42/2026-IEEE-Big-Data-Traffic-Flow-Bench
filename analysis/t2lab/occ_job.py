"""occ worker (run through run_heavy).  Usage: occ_job.py [only=<task,task,...>] [budget=<minutes>]

Generic worker (copied pattern of allfin_job.py): walks the task list in priority order, claims the first runnable task nobody
else holds (flock on cache/occ/locks/<task>.lock; done marker on success), stops taking tasks when the next one would not fit
into its budget (default 22 min).  Every task is idempotent (per-fold persistence).

tasks:
  build                 occupancy side-cars for every panel / set + calibration + parity of my slot-T draw / history vs the
                        cached all-finite rows (has_T, vr_T, vr_last must be identical)
  og_np_f<k>/og_id_f<k> F03 np255 / id255 on all-finite rows + 13 occupancy columns, held-out fold k   (arms occ_np255 / occ_id255)
  base0                 sanity: the same code path WITHOUT occupancy columns, np, fold 0 -> must reproduce af_all_np255 fold 0
  final_np / final_id   final models on all train days (k = 4)
  pm_og / pm_on         occ_pmarch.py og / on (pseudo-March on the new scenario seed)
  on_f0                 onset CatBoost (production-like, NORAMP, seeds 0-2) with the occupancy columns, held-out fold 0
                        (+ the baseline seed 0 refit for a reproduction check against cache/allfin/onset/all_fold0.pkl)
  seed1_np_f<k>         noise reference: the all-finite baseline (no occupancy) with seed 1, folds 0 / 1
  seed1_<c>_f<k> / occs1_<c>_f<k>   seed-1 replicas of the baseline / occupancy arms (k = 0..3 OOF, k = 4 final) -> bag of 4 as fs4
  pm_og4                occ_pmarch.py og4: pseudo-March with the seed-0 + seed-1 bags (np, id x 2 seeds) of both arms
  abl_np_f<k>           mechanism diagnostic (not a candidate): np + the 12 occupancy ratios without the raw o_last (o_last / or_last
                        would reveal the per-link constant ocrit, i.e. link identity, which the np set excludes), folds 0 / 1
"""
import fcntl, gc, os, subprocess, sys, time
import numpy as np
import pandas as pd
import lab
import occ_lib as oc
import exp_firstcross_lib as fc

T_START = time.time()
ARGS = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
BUDGET = float(ARGS.get('budget', 22)) * 60
ONLY = ARGS['only'].split(',') if 'only' in ARGS else None
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
F, ONSET, ONGOING = lab.F, lab.ONSET, lab.ONGOING

TASKS = [('build', 4, ()), ('og_np_f0', 4, ('build',)), ('og_id_f0', 4, ('build',)), ('og_np_f1', 4, ('build',)), ('og_id_f1', 4, ('build',)),
         ('base0', 3.5, ('build',)), ('final_np', 4, ('build',)), ('final_id', 4, ('build',)),
         ('og_np_f2', 4, ('build',)), ('og_id_f2', 4, ('build',)), ('og_np_f3', 4, ('build',)), ('og_id_f3', 4, ('build',)),
         ('pm_og', 9, ('build', 'final_np', 'final_id')), ('on_f0', 3, ('build',)), ('pm_on', 6, ('build',)),
         ('seed1_np_f0', 3.5, ('build',)), ('seed1_np_f1', 3.5, ('build',)),
         ('abl_np_f0', 3.5, ('build',)), ('abl_np_f1', 3.5, ('build',))]
# second seed of BOTH arms (production bag shape fs4 = np, id x seeds 0, 1): baseline seed 1 (seed1_*) and occupancy seed 1 (occs1_*)
TASKS += [(f'{a}_{c}_f{k}', 4.5 if k == 4 else 3.5, ('build',)) for k in (4, 0, 1, 2, 3) for c in ('np', 'id') for a in ('seed1', 'occs1')
          if (a, c, k) not in (('seed1', 'np', 0), ('seed1', 'np', 1))]
TASKS += [('pm_og4', 6, ('build', 'final_np', 'final_id', 'seed1_np_f4', 'seed1_id_f4', 'occs1_np_f4', 'occs1_id_f4'))]


def done(task):
    return os.path.exists(f'{oc.OC}/done/{task}.done')


def failed(task):
    return os.path.exists(f'{oc.OC}/done/{task}.failed')


def onset_fold(k, seeds=(0, 1, 2), evals=('offlike', 'official', 'pseudo')):
    from catboost import CatBoostClassifier
    ff = f'{oc.OC}/onset/occ_fold{k}.pkl'
    if os.path.exists(ff):
        return
    cols = lab.all_cols(); KEEPC = [i for i, c in enumerate(cols) if not c.startswith('ramp_')]       # NORAMP = production onset
    t1 = time.time(); Xt, Xb, yt = [], [], []
    for panel in lab.PANELS:
        rc = oc.rows_af(panel, 'dense1500', ONSET); S = oc.Side(panel, 'dense1500', ONSET)
        m = rc.e & (rc.fold != k); r = np.where(m)[0]
        xb = rc.matrix(m, cols, k)[:, KEEPC]
        Xb.append(xb); Xt.append(np.hstack([xb, S.rows(rc.wi[r], rc.li[r], k)])); yt.append(rc.y[m].astype(int))
    Xt = np.concatenate(Xt); Xb = np.concatenate(Xb); yt = np.concatenate(yt)

    def cat(sd):
        return CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=sd,
                                  allow_writing_files=False)
    models = []
    for sd in seeds:
        c_ = cat(sd); c_.fit(Xt, yt); models.append(c_)
    base = cat(0); base.fit(Xb, yt)                               # reproduction check of the allfin 'all_s0' arm
    res = []
    for s in evals:
        for panel in lab.PANELS:
            ev = oc.rows_af(panel, s, ONSET); S = oc.Side(panel, s, ONSET)
            widx = np.where(ev.win.fold.to_numpy() == k)[0]
            if not len(widx):
                continue
            me = np.isin(ev.wi, widx); r = np.where(me)[0]
            xb = ev.matrix(me, cols, k)[:, KEEPC]; xe = np.hstack([xb, S.rows(ev.wi[r], ev.li[r], k)])
            ps = [m_.predict_proba(xe)[:, 1] for m_ in models]
            for nm, p in (('occ_s0', ps[0]), ('occ_bag', np.mean(ps, 0)), ('rebase_s0', base.predict_proba(xb)[:, 1])):
                P6 = np.zeros((len(ev.win), F, ev.L), np.float32); P6[ev.wi[me], F - 1, ev.li[me]] = p
                res.append(lab.score_probs(ev, P6, widx, arm=nm, set_name=s))
    df = pd.concat(res, ignore_index=True)
    imp = dict(zip([cols[i] for i in KEEPC] + oc.OCC_COLS, models[0].get_feature_importance().tolist()))
    pd.to_pickle(dict(df=df, imp=imp, n=len(Xt), pos=float(yt.mean())), ff)
    g = df[df.set == 'offlike']
    lab.log(f'onset occ fold {k}: {len(Xt)} train rows, pos {yt.mean():.4f}, {time.time() - t1:.0f}s | offlike ' +
            ', '.join(f'{a} {lab.aggregate(g[g.arm == a]):.4f}' for a in dict.fromkeys(g.arm)))


def run(task):
    if task == 'build':
        st = oc.ensure_side()
        st.to_csv(f'{oc.RES}/side_stats.csv', index=False)
        cols = [c for c in ['panel', 'set', 'cond', 'windows', 'rows', 'T_vis', 'hist_vis', 'par_has_T', 'par_vr_T', 'par_vr_last', 'par_ok'] if c in st]
        print(st[cols].to_string(index=False))
        ok = bool(st.par_ok.fillna(True).astype(bool).all())
        print('ALL PARITY OK:', ok, '| side-cars', len(st), flush=True)
        assert ok and len(st) == len(lab.PANELS) * len(oc.SETS), 'parity failed or side-cars missing'
        rows = []
        for panel in lab.PANELS:
            z = np.load(f'{oc.OC}/calib/{panel}.npz')
            for k in range(5):
                rows.append(dict(panel=panel, k=k, links=z['ocrit5'].shape[1], estimated=int(z['est'][k].sum()),
                                 ocrit_q10=float(np.quantile(z['ocrit5'][k], 0.1)), ocrit_q50=float(np.median(z['ocrit5'][k])),
                                 ocrit_q90=float(np.quantile(z['ocrit5'][k], 0.9)), err_med=float(np.nanmedian(z['err5'][k])),
                                 err_max=float(np.nanmax(z['err5'][k])), c_q50=float(np.median(z['c5'][k]))))
        pd.DataFrame(rows).to_csv(f'{oc.RES}/calib_stats.csv', index=False)
    elif task.startswith('og_'):
        _, c, fk = task.split('_')
        oc.run_arm(f'occ_{c}255', c, True, folds=(int(fk[1:]),))
    elif task == 'base0':
        oc.run_arm('occbase_np255', 'np', False, folds=(0,))
        ref = oc.load_arm('af:af_all_np255'); ref = ref[ref.fold == 0]; df = oc.load_arm('occbase_np255')
        for s in fc.EVALS:
            a, b = fc.paired(ref[ref.set == s], df[df.set == s])
            lab.log(f'SANITY base0 vs af_all_np255 [{s}]: n {len(a)} | max abs IoU diff {np.abs(a.iou.to_numpy() - b.iou.to_numpy()).max():.2e}'
                    f' | agg {lab.aggregate(a):.6f} vs {lab.aggregate(b):.6f}')
    elif task in ('final_np', 'final_id'):
        c = task.split('_')[1]
        oc.run_arm(f'occ_{c}255', c, True, folds=(4,))
        assert os.path.exists(f'{oc.arm_dir(f"occ_{c}255")}/model_f4.txt')
    elif task.startswith('seed1_') or task.startswith('occs1_'):
        a, c, fk = task.split('_'); k = int(fk[1:])
        oc.run_arm(f'{a}_{c}255', c, a == 'occs1', folds=(k,), seed=1)
        if k == 4:
            assert os.path.exists(f'{oc.arm_dir(f"{a}_{c}255")}/model_f4.txt')
    elif task.startswith('abl_'):                                        # mechanism diagnostic: occupancy without the raw o_last
        _, c, fk = task.split('_')
        oc.run_arm(f'occnoraw_{c}255', c, True, folds=(int(fk[1:]),), occ_set='noraw')
    elif task == 'on_f0':
        onset_fold(0)
    elif task in ('pm_og', 'pm_on', 'pm_og4'):
        part = task.split('_')[1]
        here = os.path.dirname(os.path.abspath(__file__))
        with open(f'{lab.LOGS}/occ_{task}.log', 'w') as fh:
            rc = subprocess.call([sys.executable, os.path.join(here, 'occ_pmarch.py'), part], stdout=fh, stderr=subprocess.STDOUT, cwd=here)
        assert rc == 0, f'occ_pmarch.py {part} exit {rc} (see logs/occ_{task}.log)'
    else:
        raise KeyError(task)


lab.log(f'worker start: budget {BUDGET / 60:.0f} min, only={ONLY}')
ran = []
while True:
    picked = None
    for task, est, deps in TASKS:
        if ONLY is not None and task not in ONLY:
            continue
        if done(task) or failed(task) or not all(done(d) for d in deps):
            continue
        if time.time() - T_START + est * 60 > BUDGET and ran:
            continue
        lk = open(f'{oc.OC}/locks/{task}.lock', 'w')
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lk.close(); continue
        if done(task):
            lk.close(); continue
        picked = (task, lk); break
    if picked is None:
        if not done('build') and not failed('build') and time.time() - T_START < 8 * 60:
            time.sleep(10); continue
        break
    task, lk = picked
    lab.log(f'=== task {task} (elapsed {(time.time() - T_START) / 60:.1f} min)')
    t1 = time.time()
    try:
        run(task)
        open(f'{oc.OC}/done/{task}.done', 'w').write(f'{time.time() - t1:.0f}s\n')
        lab.log(f'=== task {task} done in {time.time() - t1:.0f}s')
    except Exception:
        import traceback
        traceback.print_exc()
        open(f'{oc.OC}/done/{task}.failed', 'w').write(traceback.format_exc())
        lab.log(f'=== task {task} FAILED after {time.time() - t1:.0f}s (marker cache/occ/done/{task}.failed; delete it to retry)')
    lk.close(); ran.append(task); gc.collect()
lab.log('WORKER DONE', ran, f'| elapsed {(time.time() - T_START) / 60:.1f} min')
