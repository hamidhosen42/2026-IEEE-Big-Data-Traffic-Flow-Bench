"""ogfeat: build the static extra-feature caches (train window sets + pseudo-March sets).
Usage (through run_heavy): ogfeat_build.py [train|pm|all]"""
import sys, time
import numpy as np
import lab, ogfeat_lib as og

what = sys.argv[1] if len(sys.argv) > 1 else 'all'
for panel in lab.PANELS:
    t1 = time.time()
    if what in ('train', 'all'):
        for s in ('official', 'offlike', 'pseudo', 'runstart', 'faith3'):
            FE = og.build_fe(panel, s)
            lab.log(panel, s, FE.shape, 'finite share', round(float(np.isfinite(np.asarray(FE[:200])).mean()), 3))
    if what in ('pm', 'all'):
        pm = og.build_pm(panel)
        lab.log(panel, 'pseudo-March', {s: (len(v['win']), v['X'].shape) for s, v in pm.items()})
    lab.log(panel, f'done in {time.time() - t1:.0f}s')
    lab.drop_panel(panel)
lab.log('names', len(og.static_names()), og.static_names())
lab.log('DONE')
