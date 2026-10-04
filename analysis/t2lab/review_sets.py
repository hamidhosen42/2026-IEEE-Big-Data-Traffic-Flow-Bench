"""Review: window-set facts (official inside offlike, duplicates, fold spread, filled-persistence claims) and an onset /
ongoing run_arm plumbing smoke test on 2 panels (tiny model). Run through run_heavy."""
import numpy as np, pandas as pd, lab

pd.set_option('display.width', 250)
tot = dict(off_og=0, off_in_offlike=0, off_in_pseudo=0, offlike=0, offlike_pf_gt09=0, official_pf_gt09=0, off_runstart=0)
for panel in lab.PANELS:
    sets = {s: lab.window_set(panel, s) for s in ('official', 'offlike', 'pseudo', 'dense1500', 'faith3', 'runstart')}
    for s, w in sets.items():
        assert not w.duplicated(['d', 't', 'cond']).any(), (panel, s, 'duplicate windows')
        assert w.t.between(12, 281).all() and (w.fold == lab.load_panel(panel)['block'][w.d]).all()
    og = {s: set(zip(w.d[w.cond == lab.ONGOING], w.t[w.cond == lab.ONGOING])) for s, w in sets.items()}
    o = sets['official']; oo = o[o.cond == lab.ONGOING]
    tot['off_og'] += len(oo); tot['off_in_offlike'] += len(og['official'] & og['offlike']); tot['off_in_pseudo'] += len(og['official'] & og['pseudo'])
    tot['offlike'] += len(og['offlike']); tot['offlike_pf_gt09'] += int((sets['offlike'].pers_fill > 0.9)[sets['offlike'].cond == lab.ONGOING].sum())
    tot['official_pf_gt09'] += int((oo.pers_fill > 0.9).sum()); tot['off_runstart'] += int(oo.runstart.sum())
    assert og['runstart'] <= set(zip(*np.where(lab.candidates(lab.load_panel(panel))[1])))
    print(panel, 'official ongoing in offlike', len(og['official'] & og['offlike']), '/', len(oo), '| official fold ids', sorted(o.fold.unique()),
          '| offlike ongoing per fold', np.bincount(sets['offlike'][sets['offlike'].cond == lab.ONGOING].fold, minlength=4).tolist(),
          '| dense1500 faithful share', round(float(sets['dense1500'][sets['dense1500'].cond == lab.ONGOING].faithful.mean()), 3),
          '| official pers_fill', np.round(oo.pers_fill.to_numpy(), 3).tolist(), flush=True)
    lab.drop_panel(panel)
print('TOTALS', tot)

# plumbing smoke: onset + ongoing arms, 2 panels, fold 0, tiny model (numbers are NOT baselines)
for cond in (lab.ONSET, lab.ONGOING):
    df = lab.run_arm(f'review_smoke_{cond}', train='dense1500', evals=('official', 'offlike'), cols='np', params=lab.lgb_params(num_leaves=31),
                     n_trees=30, folds=(0,), cond=cond, panels=['D7_I10_W', 'D12_I5_S'], save=False)
    print(cond, 'smoke:', lab.summary(df).round(3).to_string(index=False))
    assert (df.cond == cond).all() and set(df.panel) == {'D7_I10_W', 'D12_I5_S'} and (df.fold == 0).all()
print('SETS REVIEW DONE')
