"""pvog: build the masked-history version of the F03 training rows (all 8 panels; cached under data/t2local/cache/pvog/)."""
import sys, time
import numpy as np
import lab
import pvog_lib as pv

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print('March regime shares (0.5 / 0.7 / 0.8 visible):', np.round(pv.march_regime_share(), 4).tolist(), flush=True)
tot = 0
for panel in lab.PANELS:
    t0 = time.time()
    z = pv.masked_f03_rows(panel, seed)
    rc3 = lab.rows(panel, 'faith3'); rcr = lab.rows(panel, 'runstart')
    tot += len(z['y'])
    lab.log(panel, 'masked F03 rows', len(z['y']), 'pos', round(float(z['y'].mean()), 4), 'windows', int(z['n_windows']), 'masked half rows', int(z['half'].sum()),
            'day regimes', {float(r): int((z['p_day'] == r).sum()) for r in pv.REGIMES}, f'{time.time() - t0:.0f}s')
    lab.drop_panel(panel)
lab.log('total masked F03 rows', tot)
