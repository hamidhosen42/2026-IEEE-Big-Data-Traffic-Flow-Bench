"""TRACK ogclock: build the ck feature caches (train-side masked-layer emulation) for every panel / set and the pseudo-March rows."""
import sys, time
import numpy as np, pandas as pd
import lab
import ogclock_lib as ck
sets = sys.argv[1].split(',') if len(sys.argv) > 1 else ['official', 'offlike', 'pseudo', 'runstart', 'faith3']
panels = sys.argv[2].split(',') if len(sys.argv) > 2 else lab.PANELS
for panel in panels:
    t1 = time.time()
    for s in sets:
        if s == 'pm':
            ck.build_pm(panel)
        else:
            FE = ck.build_fe(panel, s)
            lab.log(panel, s, FE.shape)
    lab.drop_panel(panel)
    lab.log(panel, f'done in {time.time() - t1:.0f}s')
lab.log('DONE')
