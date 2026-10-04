"""allfin worker (run through run_heavy).  Usage: allfin_job.py [anything] [only=<task,task,...>] [budget=<minutes>]

The semaphore queue is long, so every queued instance is a generic WORKER: it walks the task list below in priority order,
claims the first runnable task that nobody else holds (flock on cache/allfin/locks/<task>.lock; a done marker is written on
success), and stops taking tasks when the next one would not fit into its time budget (default 22 min -> a job stays < 25 min).
Several workers can run at the same time; every task is idempotent (per-fold persistence).

tasks:
  build            all-finite row caches for every panel / set (+ parity of the eligible-only build vs lab.rows, keep coverage)
  on_f<k>          fold-safe CatBoost onset CV, fold k (production-like, NORAMP): eligible-only vs all-finite features, seeds 0-2,
                   + the eligible-trained model fed all-finite rows (no retrain)
  og_np_f<k> / og_id_f<k>   F03 np255 / id255 on all-finite rows, held-out fold k
  base0            sanity: F03 np255 on eligible-only rows through THIS code path, fold 0 (must reproduce fc_F03_np255) + cross evaluation
  final_np / final_id       final models on all train days (k = 4) with all-finite rows
  pm_ogon / pm_dry          allfin_pmarch.py og,on / dry (pseudo-March on the new scenario seed; label-free footprint)
  xlab_np_f<k>     og_np with additional training labels from ineligible-valued cells
"""
import fcntl, gc, os, subprocess, sys, time
import numpy as np
import pandas as pd
import lab
import allfin_lib as af
import exp_firstcross_lib as fc

T_START = time.time()
ARGS = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
BUDGET = float(ARGS.get('budget', 22)) * 60
ONLY = ARGS['only'].split(',') if 'only' in ARGS else None
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
F, ONSET, ONGOING = lab.F, lab.ONSET, lab.ONGOING
for _d in ('locks', 'done'):
    os.makedirs(f'{af.AF}/{_d}', exist_ok=True)

# (task, estimated minutes, dependencies) in priority order
TASKS = [('build', 3, ()), ('on_f0', 2, ('build',)), ('og_np_f0', 3, ('build',)), ('base0', 3.5, ('build',)), ('on_f1', 2, ('build',)),
         ('og_np_f1', 3, ('build',)), ('og_id_f0', 3, ('build',)), ('og_id_f1', 3, ('build',)),
         ('final_np', 3, ('build',)), ('final_id', 3, ('build',)), ('pm_dry', 2.5, ('build', 'final_np', 'final_id')),
         ('pm_ogon', 9, ('build', 'final_np', 'final_id')),
         ('og_np_f2', 3, ('build',)), ('og_id_f2', 3, ('build',)), ('og_np_f3', 3, ('build',)), ('og_id_f3', 3, ('build',)),
         ('on_f2', 2, ('build',)), ('on_f3', 2, ('build',)), ('xlab_np_f0', 3.5, ('build',)), ('xlab_np_f1', 3.5, ('build',))]


def done(task):
    return os.path.exists(f'{af.AF}/done/{task}.done')


def onset_cv(folds, seeds=(0, 1, 2), evals=('offlike', 'official', 'pseudo')):
    from catboost import CatBoostClassifier
    cols = lab.all_cols(); KEEPC = [i for i, c in enumerate(cols) if not c.startswith('ramp_')]       # NORAMP = production onset
    for mode in ('elig', 'all'):
        rows = af.rows_any(mode)
        for k in folds:
            ff = f'{af.AF}/onset/{mode}_fold{k}.pkl'
            if os.path.exists(ff):
                continue
            t1 = time.time(); Xt, yt = [], []
            for panel in lab.PANELS:
                rc = rows(panel, 'dense1500', ONSET)
                m = rc.e & (rc.fold != k)
                Xt.append(rc.matrix(m, cols, k)[:, KEEPC]); yt.append(rc.y[m].astype(int))
            Xt = np.concatenate(Xt); yt = np.concatenate(yt)
            models = []
            for sd in seeds:
                cat = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=lab.THREADS, verbose=0, random_seed=sd,
                                         allow_writing_files=False)
                cat.fit(Xt, yt); models.append(cat)
            res = []
            for em in ((mode, 'all') if mode == 'elig' else (mode,)):                 # elig model also on all-finite rows (no retrain)
                erows = af.rows_any(em)
                for s in evals:
                    for panel in lab.PANELS:
                        ev = erows(panel, s, ONSET)
                        widx = np.where(ev.win.fold.to_numpy() == k)[0]
                        if not len(widx):
                            continue
                        me = np.isin(ev.wi, widx)
                        Xe = ev.matrix(me, cols, k)[:, KEEPC]
                        ps = [m_.predict_proba(Xe)[:, 1] for m_ in models]
                        tag = mode if em == mode else 'cross'
                        for nm, p in (('s0', ps[0]), ('bag', np.mean(ps, 0))):
                            P6 = np.zeros((len(ev.win), F, ev.L), np.float32); P6[ev.wi[me], F - 1, ev.li[me]] = p
                            res.append(lab.score_probs(ev, P6, widx, arm=f'{tag}_{nm}', set_name=s))
            df = pd.concat(res, ignore_index=True); df.to_pickle(ff)
            g = df[df.set == 'offlike']
            lab.log(f'onset {mode} fold {k}: {len(Xt)} train rows, pos {yt.mean():.4f}, {time.time() - t1:.0f}s | offlike ' +
                    ', '.join(f'{a} {lab.aggregate(g[g.arm == a]):.4f}' for a in dict.fromkeys(g.arm)) + f' (n={int((g.arm == g.arm.iloc[0]).sum())})')
            del Xt, yt, models; gc.collect()


def run(task):
    if task == 'build':
        af.ensure_built()
        st = af.load_stats(); st.to_csv(f'{af.RES}/build_stats.csv', index=False)
        par = st[['panel', 'set', 'cond', 'par_n_ref', 'par_n_mine', 'par_exact', 'par_max_abs_diff', 'par_aux_equal']]
        print('== parity of the eligible-only build (my builder) vs lab.rows =='); print(par.to_string(index=False))
        ok = bool(par.par_exact.all() and par.par_aux_equal.all())
        print('ALL EXACT:', ok, '| caches', len(par), flush=True)
        assert len(par) == len(lab.PANELS) * len(af.SETS), 'missing caches'
    elif task.startswith('on_f'):
        onset_cv((int(task[-1]),))
    elif task == 'base0':
        df = af.run_arm('af_base_np255', 'elig', 'np', folds=(0,), extra_eval=('all',))
        ref = lab.load_arm('fc_F03_np255'); ref = ref[ref.fold == 0]
        for s in fc.EVALS:
            a, b = fc.paired(ref[ref.set == s], df[df.set == s])
            lab.log(f'SANITY base0 vs fc_F03_np255 [{s}]: n {len(a)} | max abs IoU diff {np.abs(a.iou.to_numpy() - b.iou.to_numpy()).max():.2e}'
                    f' | agg {lab.aggregate(a):.6f} vs {lab.aggregate(b):.6f}')
    elif task.startswith('og_') or task.startswith('xlab_'):
        kind, c, fk = task.split('_')
        arm = f'af_all_{c}255' if kind == 'og' else f'af_xlab_{c}255'
        af.run_arm(arm, 'all', c, folds=(int(fk[1:]),), xlabels=kind == 'xlab')
    elif task in ('final_np', 'final_id'):
        c = task.split('_')[1]
        af.run_arm(f'af_all_{c}255', 'all', c, folds=(4,))
        assert os.path.exists(f'{af.arm_dir(f"af_all_{c}255")}/model_f4.txt')
    elif task in ('pm_ogon', 'pm_dry'):
        part = 'og,on' if task == 'pm_ogon' else 'dry'
        here = os.path.dirname(os.path.abspath(__file__))
        with open(f'{lab.LOGS}/allfin_{task}.log', 'w') as fh:
            rc = subprocess.call([sys.executable, os.path.join(here, 'allfin_pmarch.py'), part], stdout=fh, stderr=subprocess.STDOUT, cwd=here)
        assert rc == 0, f'allfin_pmarch.py {part} exit {rc} (see logs/allfin_{task}.log)'
    else:
        raise KeyError(task)


def failed(task):
    return os.path.exists(f'{af.AF}/done/{task}.failed')


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
            continue                                                   # would not fit any more (a worker always runs at least one task)
        lk = open(f'{af.AF}/locks/{task}.lock', 'w')
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lk.close(); continue                                       # another worker is on it
        if done(task):
            lk.close(); continue
        picked = (task, lk); break
    if picked is None:
        # nothing runnable: if only the build is in the way (another worker is building), wait for it briefly, else stop
        if not done('build') and not failed('build') and time.time() - T_START < 6 * 60:
            time.sleep(10); continue
        break
    task, lk = picked
    lab.log(f'=== task {task} (elapsed {(time.time() - T_START) / 60:.1f} min)')
    t1 = time.time()
    try:
        run(task)
        open(f'{af.AF}/done/{task}.done', 'w').write(f'{time.time() - t1:.0f}s\n')
        lab.log(f'=== task {task} done in {time.time() - t1:.0f}s')
    except Exception:                                                  # keep the worker alive for the independent tasks
        import traceback
        traceback.print_exc()
        open(f'{af.AF}/done/{task}.failed', 'w').write(traceback.format_exc())
        lab.log(f'=== task {task} FAILED after {time.time() - t1:.0f}s (marker cache/allfin/done/{task}.failed; delete it to retry)')
    lk.close(); ran.append(task); gc.collect()
lab.log('WORKER DONE', ran, f'| elapsed {(time.time() - T_START) / 60:.1f} min')
