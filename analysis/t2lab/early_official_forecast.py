"""early track (light): window-level forecast of the public effect.  For each of the 40 official validation onset windows the
step-6 link set predicted by a variant is scored against the visible T+30 truth of SIMILAR pseudo-March onset windows (same panel,
same weekday class, origin within +-6 slots, any later day of the month), averaged.  Rough transfer estimate (uses March labels
after the official windows: probe motivation only; visible-only labels; local replica of the production CatBoost)."""
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
pd.set_option('display.width', 250)
F = lab.F
cat = el.load_cat()
M = dict(box=(-6, 6), form='bma', mu=0.05, nu=1.0, cw=1.0)
designs = {'sp4': dict(box=(-2, 2), ref='base', k=4, cw=0.0, ncount='obs'), 'mix9': dict(M, pi=0.9), 'mix95': dict(M, pi=0.95), 'mix8': dict(M, pi=0.8)}
names = ['base'] + list(designs); rows = []
for panel in lab.PANELS:
    P, Pv = el.val_panel(panel)
    prior_tr = lab.prior_table(P, np.ones(len(P['days']), bool))
    R = lab.split_rows(P, 'validation', prior_tr)
    X, _, _, meta, cols = lab.to_matrix(R, lab.ONSET, None, None)
    wi = R['wi']; on = np.where(R['cond'] == lab.ONSET)[0]; L = R['L']
    dpos = {d: i for i, d in enumerate(Pv['days'])}
    d = np.array([dpos[x] for x in wi.date.to_numpy()[on]]); t = R['t'][on]
    b = dict(panel=panel, L=L, D=len(Pv['days']), W=len(on), wk=Pv['dow'] >= 5, Qv=Pv['Q'], Ev=Pv['E'], prior_tr=prior_tr, X=X.reshape(len(on), L, -1), d=d, t=t)
    items, _ = en.items_recent(b, 99); base3 = b['X'][..., el.JP]; pred = {}
    for name in names:
        p3 = None
        if name != 'base':
            dz = dict(designs[name]); box = dz.pop('box'); p3 = el.design_priors(base3, el.item_stats(b, items, box), box, **dz)
        p = el.predict_items(cat, b, np.arange(len(on)), p3)
        pr = np.zeros((len(on), L), bool)
        for i in range(len(on)):
            P6 = np.zeros((F, L), np.float32); P6[F - 1] = p[i]; pr[i] = lab.decode(P6.ravel()).reshape(F, L)[F - 1]
        pred[name] = pr
    mb = el.march_bundle(panel)
    for i in range(len(on)):
        sim = [j for j in range(mb['W']) if abs(int(mb['t'][j]) - int(t[i])) <= 6 and bool(mb['wk'][mb['d'][j]]) == bool(b['wk'][d[i]])]
        r = dict(panel=panel, day=int(d[i]) + 1, t=int(t[i]), n_sim=len(sim), changed=int(any((pred[n][i] != pred['sp4'][i]).any() for n in names)))
        for n in names:
            v = []
            for j in sim:
                T = mb['truth'][j][5] & mb['ev'][j][5]; Pp = pred[n][i] & mb['ev'][j][5]
                u = (T | Pp).sum(); v.append(1.0 if u == 0 else (T & Pp).sum() / u)
            r[n] = float(np.mean(v)) if v else np.nan
        rows.append(r)
    lab.drop_panel(panel)
T = pd.DataFrame(rows); T.to_csv(f'{el.OUT}/official_forecast.csv', index=False)
ok = T[T.n_sim > 0]
print('official validation onset windows with >= 1 similar pseudo-March window:', len(ok), 'of', len(T))
print('mean transfer IoU over those windows:', ok[names].mean().round(4).to_dict())
ch = ok[ok.changed == 1]
print('\nwindows whose prediction differs between variants (transfer IoU per variant):'); print(ch.round(3).to_string(index=False))
for n in ('mix9', 'mix95', 'mix8'):
    dl = (ok[n] - ok['sp4'])
    print(f'{n} vs sp4: sum of window IoU differences {dl.sum():+.3f} over {int((dl.abs() > 1e-9).sum())} windows -> LB estimate {dl.sum() * 0.00375:+.5f}'
          f' (one window = 0.00375); vs base {(ok[n] - ok["base"]).sum() * 0.00375:+.5f}; sp4 vs base {(ok["sp4"] - ok["base"]).sum() * 0.00375:+.5f}')
