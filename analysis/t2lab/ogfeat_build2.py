"""ogfeat: build the round-2 static caches (seconds per panel, single thread)."""
import numpy as np
import lab, ogfeat_lib as og
for panel in lab.PANELS:
    for s in ('official', 'offlike', 'pseudo', 'runstart', 'faith3'):
        FE = og.build_fe2(panel, s)
    pm = og.build_pm2(panel)
    lab.log(panel, FE.shape, {s: v.shape for s, v in pm.items()}); lab.drop_panel(panel)
lab.log('names2', og.static_names2()); lab.log('DONE')
