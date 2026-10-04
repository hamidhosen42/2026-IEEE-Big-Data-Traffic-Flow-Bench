"""CV folds of the cnn ongoing member: for every fold k in <folds> train on the F03 windows outside fold k, predict the
held-out windows of official / offlike / pseudo, score CNN alone and the blends with the cached tree reference
(mean of fc_F03_np255 and fc_F03_id255).  Fold 4 = final model on ALL train windows (saved only; used by cnn_pmarch.py).
Usage (through run_heavy): cnn_cv.py <tag> <folds, e.g. 0,1,2,3,4> [epochs=12] [bs=32] [lr=0.002] [c=64] [seed=0] [cap=1500]
                                      [hist_drop=0] [drop=0.05] [dils=1,2,4,8,1,2,4,8] [wd=0.01]
Writes data/t2local/cache/cnn/<tag>/fold<k>.pkl (probabilities + per-window frames) and model_f<k>.pt; existing folds are skipped."""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import lab
import cnn_lib as C

tag = sys.argv[1]; folds = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
kw = dict(epochs=12, bs=32, lr=2e-3, c=64, seed=0, cap=1500, hist_drop=0.0, drop=0.05, dils='1,2,4,8,1,2,4,8', wd=1e-2)
for a in sys.argv[3:]:
    key, v = a.split('='); kw[key] = v if isinstance(kw[key], str) else type(kw[key])(v)

# ---- claims: the semaphore is a lottery, so the same job may be queued more than once; an atomic mkdir makes sure that each
# tag is trained by exactly ONE job (the others leave their slot within a second).  `auto` = take the first unclaimed tag.
AUTO = (('v1', dict(seed=0)), ('v1b', dict(seed=1)))


def claim(tg):
    try:
        os.mkdir(f'{C.CDIR}/claim_{tg}')
    except FileExistsError:
        return False
    open(f'{C.CDIR}/claim_{tg}/owner', 'w').write(f'pid {os.getpid()} at {time.strftime("%H:%M:%S")} argv {sys.argv[1:]}')
    return True


if tag == 'auto':
    for tg, cfg in AUTO:
        if claim(tg):
            tag = tg; kw.update(cfg); break
    else:
        print('cnn_cv auto: every tag is already claimed, nothing to do'); sys.exit(0)
elif tag in dict(AUTO) and not os.environ.get('CNN_PANELS') and not claim(tag):
    print(f'cnn_cv {tag}: claimed by another job ({open(f"{C.CDIR}/claim_{tag}/owner").read()}), leaving the slot'); sys.exit(3)
dils = tuple(int(x) for x in kw['dils'].split(','))
odir = f'{C.CDIR}/{tag}'; os.makedirs(odir, exist_ok=True)
json.dump(kw, open(f'{odir}/config.json', 'w'))
BLENDS = (0.2, 0.3, 0.5)
ARMS = ('trees', 'cnn', 'blend20', 'blend30', 'blend50')
lab.log(f'{tag} folds {folds} config {json.dumps(kw)} device {C.DEVICE}')

for k in folds:
    if os.path.exists(f'{odir}/model_f{k}.pt') and (k == 4 or os.path.exists(f'{odir}/fold{k}.pkl')):
        lab.log(f'{tag} fold {k}: found on disk, skipped'); continue
    t1 = time.time()
    pds = C.train_panels(k, cap=kw['cap'])
    lab.log(f'{tag} fold {k}: training windows per panel ' + json.dumps({p.panel: [p.n, round(float(p.w.sum()), 1)] for p in pds}))
    model = C.train(pds, epochs=kw['epochs'], bs=kw['bs'], lr=kw['lr'], wd=kw['wd'], seed=kw['seed'] * 10 + k, c=kw['c'], dils=dils,
                    drop=kw['drop'], hist_drop=kw['hist_drop'], tag=f'{tag} f{k}')
    torch.save(model.state_dict(), f'{odir}/model_f{k}.pt')
    train_secs = time.time() - t1
    del pds; gc.collect()
    if k == 4:
        lab.log(f'{tag} final model (all train windows) saved, training {train_secs / 60:.1f} min'); continue
    res = []; probs = {}
    for s in C.EVALS:
        for panel in C.PANELS:
            p, widx = C.eval_panel(panel, s, k)
            if p is None:
                continue
            rc = lab.rows(panel, s)
            Pc = np.zeros((len(rc.win), lab.F, rc.L), np.float32); Pc[widx] = C.predict(model, p)
            Pt, done = C.tree_ref(s, panel)
            assert done[widx].all(), (s, panel, 'tree reference missing for held-out windows')
            probs[(s, panel)] = (widx, Pc[widx].astype(np.float16))
            res.append(lab.score_probs(rc, Pc, widx, arm='cnn', set_name=s))
            res.append(lab.score_probs(rc, Pt, widx, arm='trees', set_name=s))
            for w in BLENDS:
                res.append(lab.score_probs(rc, w * Pc + (1 - w) * Pt, widx, arm=f'blend{int(w * 100)}', set_name=s))
    df = pd.concat(res, ignore_index=True)
    pd.to_pickle(dict(df=df, probs=probs, cfg=kw, train_secs=train_secs), f'{odir}/fold{k}.pkl')
    for s in C.EVALS:
        g = df[df.set == s]
        lab.log(f'{tag} fold {k} {s} (n={int((g.arm == "cnn").sum())}): ' + ', '.join(f'{a} {lab.aggregate(g[g.arm == a]):.4f}' for a in ARMS))
    lab.log(f'{tag} fold {k} DONE, training {train_secs / 60:.1f} min')

dfs = [pd.read_pickle(f'{odir}/fold{k}.pkl')['df'] for k in range(4) if os.path.exists(f'{odir}/fold{k}.pkl')]
if dfs:
    df = pd.concat(dfs, ignore_index=True)
    for s in C.EVALS:
        g = df[df.set == s]
        lab.log(f'{tag} folds on disk {sorted(g.fold.unique().tolist())} {s} (n={int((g.arm == "cnn").sum())}): ' + ', '.join(f'{a} {lab.aggregate(g[g.arm == a]):.4f}' for a in ARMS))
lab.log('CV DONE')


# ---- follow-up inside the same semaphore slot: evaluation + pseudo-March of this tag, and of the 2-seed ensemble v1 + v1b
# when both tags are complete (whichever job finishes last runs it; cnn_eval.py / cnn_pmarch.py skip work that is already on disk)
def complete(tg):
    return all(os.path.exists(f'{C.CDIR}/{tg}/fold{j}.pkl') for j in range(4)) and os.path.exists(f'{C.CDIR}/{tg}/model_f4.pt')


if complete(tag) and not os.environ.get('CNN_PANELS'):
    import subprocess
    sys.stdout.flush()
    for args in ([tag], ):
        subprocess.call([sys.executable, 'cnn_eval.py'] + args); subprocess.call([sys.executable, 'cnn_pmarch.py'] + args)
    if tag in ('v1', 'v1b') and complete('v1') and complete('v1b'):
        subprocess.call([sys.executable, 'cnn_eval.py', 'v1,v1b', 'ens2']); subprocess.call([sys.executable, 'cnn_pmarch.py', 'v1,v1b', 'ens2'])
