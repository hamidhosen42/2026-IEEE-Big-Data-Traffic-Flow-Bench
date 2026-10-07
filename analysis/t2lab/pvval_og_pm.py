"""pvval ONGOING on pseudo-March (days >= CUT; March days < CUT = same-scenario training rows, F03 recipe x WM = the kernel's
March rows): the kernel recipe and the MASKING-ADAPTATION control.  LightGBM at lab scale (255 leaves, lr 0.05, 500 trees).
MODE
  R    raw-train regime: final k=4 F03 models (np / id / full, results/firstcross/final) as the train-only members and the
       March-augmented members RB_np / RB_id / RB_full (F03 rows of all train days + March rows x WM; full = blended prior)
       -> probs R_<CUT>_<WM>  (same arms as pvog_marchtrain.py, which was killed before it finished)
  Mb   build the masked F03 row cache (train F03 windows; history + slot T from the emulated masked layer: day regime
       0.5 / 0.7 / 0.8 drawn from the March shares, cells hidden i.i.d.; labels = every observed cell)
  M0   masked-train regime, train-only: M_np / M_id / M_full on the masked F03 rows        -> probs M0_<CUT>_<WM>
  M1   masked-train regime + March rows: MB_np / MB_id / MB_full                           -> probs M1_<CUT>_<WM>
Scoring: pvval_og_report.py pm.  Real same-scenario effect = MB - M (both trained in the evaluation's feature regime) vs RB - R."""
import gc, os, sys, time
import numpy as np
import lightgbm as lgb
import lab
import exp_firstcross_lib as fc
import pvval_lib as V
import pvog_lib as pv

MODE = sys.argv[1]
CUT = int(sys.argv[2]) if len(sys.argv) > 2 else 15
WM = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
SMOKE = bool(int(os.environ.get('SMOKE', 0)))
NT = 20 if SMOKE else 500
PANELS = lab.PANELS[:2] if SMOKE else lab.PANELS
TAG = f'{MODE}_{CUT}_{WM:g}' + ('_smoke' if SMOKE else '')
sp = fc.ARMS['F03']
t00 = time.time()

if MODE == 'Mb':
    for p in PANELS:
        t0 = time.time(); z = pv.masked_f03_rows(p)
        lab.log(p, 'masked F03 rows', len(z['y']), 'pos', round(float(z['y'].mean()), 4), 'windows', int(z['n_windows']),
                'day regimes', np.unique(z['p_day'], return_counts=True)[1].tolist(), f'{time.time() - t0:.0f}s')
        lab.drop_panel(p); gc.collect()
    lab.log('DONE'); sys.exit(0)

if SMOKE:
    TR, EV = pv.march_data(CUT, WM, eval_from=0, panels=PANELS, verbose=True)
else:
    TR, EV = V.march_cached(CUT, 5.0)                 # one cache for every WM: March row weights are linear in WM
    if WM != 5.0:
        TR = dict(TR, w=TR['w'] * (WM / 5.0))
lab.log('March training rows', len(TR['y']), 'pos', round(float(TR['y'].mean()), 4), 'weight sum', round(float(TR['w'].sum()), 1),
        '| eval blocks', len(EV), f'{time.time() - t00:.0f}s')


def march_X(cs):
    return TR['Xf'] if cs == 'full' else np.ascontiguousarray(TR['X'][:, pv.IX[cs]])


members = {}
if MODE == 'R':
    for cs in ('np', 'id', 'full'):
        m0 = lgb.Booster(model_file=f'{pv.MD_FINAL}/model_F03_{cs}255_500.txt')
        if cs == 'full':
            members['R_full_s4'] = (m0, 'full', True); members['R_full_tr'] = (m0, 'full', False)
        else:
            members[f'R_{cs}'] = (m0, cs, False)
        t0 = time.time()
        X, y, w, info = fc.train_matrix(sp['train'], 4, pv.COLS[cs], sp['wfn'], panels=PANELS)
        n_tr, ws = len(y), float(w.sum())
        m1 = pv.fit_plus(X, y, w, march_X(cs), TR['y'], TR['w'], pv.COLS[cs], NT)
        if not SMOKE:
            m1.save_model(f'{V.RES}/models/model_{TAG}_{cs}B.txt')
        members[f'RB_{cs}' if cs != 'full' else 'RB_full'] = (m1, cs, cs == 'full')
        lab.log(f'RB_{cs}: {n_tr} train rows (weight {ws:.0f}) + {len(TR["y"])} March rows (weight {TR["w"].sum():.0f}), {time.time() - t0:.0f}s')
        del X, y, w; gc.collect()
else:
    parts = [pv.masked_f03_rows(p) for p in PANELS]
    lab.log('masked F03 rows', sum(len(z['y']) for z in parts), 'regime shares', np.round(pv.march_regime_share(), 3).tolist())
    for cs in ('np', 'id', 'full'):
        t0 = time.time(); ix = pv.IX[cs]
        X = np.concatenate([np.ascontiguousarray(z['X'][:, ix]) for z in parts])
        y = np.concatenate([z['y'] for z in parts]).astype(np.float32); w = np.concatenate([z['w'] for z in parts]).astype(np.float32)
        if MODE == 'M0':
            m = pv.fit_plus(X, y, w, None, None, None, pv.COLS[cs], NT)
            if cs == 'full':
                members['M_full_s4'] = (m, 'full', True); members['M_full_tr'] = (m, 'full', False)
            else:
                members[f'M_{cs}'] = (m, cs, False)
        else:
            m = pv.fit_plus(X, y, w, march_X(cs), TR['y'], TR['w'], pv.COLS[cs], NT)
            members[f'MB_{cs}'] = (m, cs, cs == 'full')
        if not SMOKE:
            m.save_model(f'{V.RES}/models/model_{TAG}_{cs}.txt')
        lab.log(f'{MODE} {cs}: {len(y)} masked train rows (weight {w.sum():.0f})' + ('' if MODE == 'M0' else f' + {len(TR["y"])} March rows')
                + f', {time.time() - t0:.0f}s')
        del X, y, w; gc.collect()

P6 = pv.predict_members(EV, members)
pv.save_probs(TAG, P6)
lab.log('saved', f'{V.RES}/probs/{TAG}.pkl', sorted({k[2] for k in P6}), f'{time.time() - t00:.0f}s')
lab.log('DONE')
