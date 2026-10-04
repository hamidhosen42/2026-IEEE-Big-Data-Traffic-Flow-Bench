"""pvon: lab emulation of what the April kernels do on the REAL official onset windows (no labels exist: descriptive only).
Models as kaggle_jobs/t2_onset_pv.py / t2_pvon_onset.py: A = production onset CatBoost (train rows), B_w = train rows + ALL released
March onset windows (masked layer, blended prior of the March days before their own day) x w for w in 5 / 10 / 20 / 40.
Official windows (lab.split_rows: RAW history, masked slot T), prior columns blended with the scenario prior of the scenario days
before the window's day (March windows: March days < d; April windows: all March + April days < d), k = 4.
Reports per split: predicted onset cells and windows that differ between base, sp4, pv (10/20/40) and pvon (5/10/20).
No April data at or after any window's T is used: features are the released history + slot T, the scenario prior uses days < d."""
import gc, sys, time
import numpy as np
import pandas as pd
import lab
import pvon_lib as L

H, F = lab.H, lab.F
KC = L.KEEPC
WS = (5.0, 10.0, 20.0, 40.0)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

Xt, yt, Xm, ym, EV = [], [], [], [], []
for panel in L.PANELS:
    P, Pv = L.val_panel(panel)
    _, Pp = L.val_panel(panel, 'private')
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    # March training rows: all released March onset windows (kernel rule), blended prior of the March days < own day
    on, _ = L.select(Pv, need_og=False)
    picks = L.spaced_picks(on)
    cQm, cEm = L.cum_tables(Pv)
    R = L.onset_rows(P, Pv, picks, prior_tr); X, y, e, meta = L.matrix(R)
    Xk = L.blend_rows(X, R, meta, Pv['dow'], prior_tr, cQm, cEm)
    Xm.append(Xk[e][:, KC]); ym.append(y[e].astype(int))
    del cQm, cEm
    rt = lab.rows(panel, 'dense1500', lab.ONSET); r = np.where(rt.e)[0]
    Xt.append(np.asarray(rt.X[r])[:, KC]); yt.append(rt.y[r].astype(int))
    # scenario timeline March + April
    days = Pv['days'] + Pp['days']; assert days == sorted(days)
    S = dict(Q=np.concatenate([Pv['Q'], Pp['Q']]), E=np.concatenate([Pv['E'], Pp['E']]), dow=np.concatenate([Pv['dow'], Pp['dow']]))
    cQ, cE = L.cum_tables(S); dpos = {d: i for i, d in enumerate(days)}
    for split in ('validation', 'private'):
        Rs = lab.split_rows(P, split, prior_tr)
        Xc, _, _, meta, call = lab.to_matrix(Rs, lab.ONSET, None, None); assert list(call) == L.COLS
        wi = Rs['wi']; W = Rs['W']
        jd = np.array([dpos[d] for d in wi.date]); dow = Rs['dow']
        fake = dict(d=np.arange(W), t=Rs['t'])
        Xk = L.blend_rows(Xc, fake, meta, dow, prior_tr, cQ, cE, day=jd)
        EV.append(dict(panel=panel, split=split, X=Xc[:, KC], Xk=Xk[:, KC], meta=meta, W=W, L=Rs['L'], cond=Rs['cond'], jd=jd,
                       date=wi.date.to_numpy(), tod=Rs['t']))
    lab.log(panel, 'March windows', len(picks), 'rows', len(Xm[-1]))
    del cQ, cE; lab.drop_panel(panel); gc.collect()
Xt = np.concatenate(Xt); yt = np.concatenate(yt); Xm = np.concatenate(Xm); ym = np.concatenate(ym)
lab.log('train rows', len(Xt), 'March rows', len(Xm), 'pos', int(ym.sum()))
A = L.fit_cat(Xt, yt); lab.log('A trained')
B = {}
for w in WS:
    B[w] = L.fit_cat(np.concatenate([Xt, Xm]), np.concatenate([yt, ym]), np.r_[np.ones(len(Xt)), np.full(len(Xm), w)]); lab.log(f'B w{w:g} trained')

rows = []
for ev in EV:
    meta, W, Lk = ev['meta'], ev['W'], ev['L']
    pr = {'base': A.predict_proba(ev['X'])[:, 1], 'sp4': A.predict_proba(ev['Xk'])[:, 1]}
    pb = {w: B[w].predict_proba(ev['Xk'])[:, 1] for w in WS}
    pr['pv'] = np.mean([pb[w] for w in (10.0, 20.0, 40.0)], 0); pr['pvon'] = np.mean([pb[w] for w in (5.0, 10.0, 20.0)], 0)
    for w in WS:
        pr[f'w{w:g}'] = pb[w]
    dec = {}
    for n, p in pr.items():
        P6 = np.zeros((W, F, Lk), np.float32); P6[meta[:, 0], F - 1, meta[:, 1]] = p
        dec[n] = np.stack([lab.decode(P6[i].ravel()).reshape(F, Lk) for i in range(W)])
    for i in np.where(ev['cond'] == lab.ONSET)[0]:
        r = dict(panel=ev['panel'], split=ev['split'], date=ev['date'][i], tod=int(ev['tod'][i]), scen_days=int(ev['jd'][i]))
        for n in dec:
            r[f'n_{n}'] = int(dec[n][i].sum())
        for a, b in (('base', 'sp4'), ('sp4', 'pv'), ('sp4', 'pvon'), ('pv', 'pvon'), ('sp4', 'w5'), ('sp4', 'w10'), ('sp4', 'w40')):
            r[f'diff_{a}_{b}'] = int((dec[a][i] != dec[b][i]).sum())
            u = (dec[a][i] | dec[b][i]).sum(); r[f'jac_{a}_{b}'] = 1.0 if u == 0 else float((dec[a][i] & dec[b][i]).sum() / u)
        rows.append(r)
df = pd.DataFrame(rows); df.to_pickle(f'{L.OUT}/april_emulation{L.STAG}.pkl')
for split in ('validation', 'private'):
    g = df[df.split == split]
    print(f'\n== official {split} onset windows: n = {len(g)}, scenario days before the window: mean {g.scen_days.mean():.1f} ==')
    print('predicted cells:', {c[2:]: int(g[c].sum()) for c in g if c.startswith('n_')})
    for c in [c for c in g if c.startswith('diff_')]:
        j = 'jac_' + c[5:]
        print(f'{c[5:]:12s} cells differing {int(g[c].sum()):4d} | windows differing {int((g[c] > 0).sum()):3d} of {len(g)} | mean window Jaccard {g[j].mean():.3f}')
    print(g.groupby('panel')[['n_base', 'n_sp4', 'n_pv', 'n_pvon', 'diff_sp4_pv', 'diff_sp4_pvon', 'diff_pv_pvon']].sum().to_string())
lab.log('DONE')
