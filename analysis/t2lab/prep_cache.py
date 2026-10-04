"""Build window sets and row caches for the given panels (default: all).  Usage: prep_cache.py [PANEL ...]"""
import sys, numpy as np, pandas as pd, lab
panels = sys.argv[1:] or lab.PANELS
SETS = ('official', 'pseudo', 'offlike', 'runstart', 'faith3', 'dense1500')
for panel in panels:
    P = lab.load_panel(panel)
    for s in SETS + ('faith',):
        w = lab.window_set(panel, s)
        og = w[w.cond == lab.ONGOING]
        lab.log(panel, s, 'windows', len(w), 'ongoing', len(og), 'faithful share', round(float(og.faithful.mean()), 3) if len(og) else None,
                'pers_fill>0.9', round(float((og.pers_fill > 0.9).mean()), 3) if len(og) else None,
                'runstart share', round(float(og.runstart.mean()), 3) if len(og) else None, 'per fold', np.bincount(og.fold, minlength=4).tolist())
    # scan-order sensitivity of the official-like set
    on, og = lab.candidates(P); g_on, g_og = lab._glist(on), lab._glist(og); s2 = set()
    for d0 in range(len(P['days'])):
        _, b = lab.greedy(g_on, g_og, 'two_pass', start_day=d0); s2.update(b)
    w = lab.window_set(panel, 'offlike'); s1 = set(zip(w.d[w.cond == lab.ONGOING], w.t[w.cond == lab.ONGOING]))
    lab.log(panel, 'offlike ongoing: interleaved', len(s1), 'two_pass', len(s2), 'common', len(s1 & s2))
    for s in SETS:
        for cond in ((lab.ONGOING,) if s in ('runstart', 'faith3') else (lab.ONGOING, lab.ONSET)):
            rc = lab.rows(panel, s, cond)
            lab.log(panel, s, cond, 'rows', rc.n, 'windows', len(rc.win), 'keep share (fold prior 0)', round(float(rc.keep5[:, 0].mean()), 3) if rc.n else None)
    lab.drop_panel(panel)
lab.log('PREP DONE', panels)
