"""ogfeat report: paired deltas of feature arms against a baseline arm.
Usage: ogfeat_report.py <base arm dir name> <arm[,arm...]> [folds=01] [pm=01|4] [tag=...]
OOF sets: official / offlike / pseudo on the folds present in both arms.  Pseudo-March: mean of the probabilities of the
models listed in pm (fold models 0..3 and / or 4 = final all-train model) for both arms, production decode."""
import os, sys, types
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc
import ogfeat_lib as og

base = sys.argv[1]; arms = sys.argv[2].split(',')
kw = dict(a.split('=', 1) for a in sys.argv[3:])
folds = [int(c) for c in kw.get('folds', '01')]; pmk = [int(c) for c in kw.get('pm', '01')]; tag = kw.get('tag', '')
if os.environ.get('OGF_SMOKE'):
    lab.PANELS[:] = ['D7_I10_W', 'D7_I210_E']
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)


def oof(name):
    d = f'{og.ODIR}/arms/{name}'; out = []
    for k in folds:
        f = f'{d}/fold{k}.pkl'
        if os.path.exists(f):
            out.append(pd.read_pickle(f)['df'])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pm(name):
    d = f'{og.ODIR}/arms/{name}'; acc = None; n = 0
    for k in pmk:
        f = f'{d}/pm_f{k}.pkl'
        if not os.path.exists(f):
            return None
        z = pd.read_pickle(f)
        acc = {key: v.astype(np.float32) for key, v in z.items()} if acc is None else {key: acc[key] + z[key].astype(np.float32) for key in acc}
        n += 1
    res = []
    for (panel, s), P6 in acc.items():
        zz = og.build_pm(panel)[s]
        rc = types.SimpleNamespace(truth=zz['truth'], ev=zz['ev'], qlast=zz['qlast'], win=zz['win'], name=s, panel=panel, cond=lab.ONGOING, L=zz['qlast'].shape[1])
        df = lab.score_probs(rc, P6 / n, arm=name, set_name=s); df['vis'] = zz['win'].vis.to_numpy(); df['day'] = zz['win'].d.to_numpy()
        res.append(df)
    return pd.concat(res, ignore_index=True)


A = oof(base); Apm = pm(base)
if kw.get('oofbase'):                                                  # baseline OOF frame from a lab arm (e.g. fc_F03_np255)
    A = lab.load_arm(kw['oofbase']); A = A[A.fold.isin(folds)].reset_index(drop=True)
    mine = oof(base)
    if len(mine):                                                      # parity of my pipeline with the lab arm on the folds I ran
        m = A.merge(mine, on=['set', 'panel', 'd', 't', 'cond'], suffixes=('_lab', '_og'))
        print(f"PARITY {base} vs {kw['oofbase']}: {len(m)} windows on folds {sorted(mine.fold.unique().tolist())}, max |iou diff| {np.abs(m.iou_lab - m.iou_og).max():.6f}, "
              f"identical {int((m.iou_lab == m.iou_og).sum())}")
rows = []; summ = []
for s in ('official', 'offlike', 'pseudo'):
    g = A[A.set == s] if len(A) else A
    if len(g):
        summ.append(dict(arm=base, **lab.summary(g).iloc[0].to_dict()))
if Apm is not None:
    for s in og.PM_SETS:
        summ.append(dict(arm=base, **lab.summary(Apm[Apm.set == s]).iloc[0].to_dict()))
for a in arms:
    B = oof(a)
    if len(B) and len(A):
        for s in ('official', 'offlike', 'pseudo'):
            rows.append(og.delta_row(A[A.set == s], B[B.set == s], a, s))
            summ.append(dict(arm=a, **lab.summary(B[B.set == s]).iloc[0].to_dict()))
    Bpm = pm(a)
    if Bpm is not None and Apm is not None:
        for s in og.PM_SETS:
            rows.append(og.delta_row(Apm[Apm.set == s], Bpm[Bpm.set == s], a, s))
            summ.append(dict(arm=a, **lab.summary(Bpm[Bpm.set == s]).iloc[0].to_dict()))
        m = Apm.day < 12                                              # public-like part of the month (official ongoing: days 1-12)
        for s in og.PM_SETS:
            r = og.delta_row(Apm[(Apm.set == s) & m], Bpm[Bpm.set == s], a, s + '_d1-12'); rows.append(r)
            r = og.delta_row(Apm[(Apm.set == s) & ~m], Bpm[Bpm.set == s], a, s + '_d13+'); rows.append(r)
S = pd.DataFrame(summ); D = pd.DataFrame(rows)
print(f'== summary (OOF folds {folds}; pseudo-March models {pmk}) =='); print(S.round(4).to_string(index=False))
print(f'\n== paired deltas vs {base} (B - A; CI = paired bootstrap within panel, 2000 draws) =='); print(D.round(4).to_string(index=False))
if tag:
    S.to_csv(f'{og.RES}/summary_{tag}.csv', index=False); D.to_csv(f'{og.RES}/deltas_{tag}.csv', index=False)
