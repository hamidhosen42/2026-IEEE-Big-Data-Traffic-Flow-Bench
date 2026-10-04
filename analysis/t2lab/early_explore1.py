"""early track, light exploration (seconds): the CURRENT sp4 design family (blend, box +-2, class-separate, n = obs / 5)
with only the n most recent scenario days (E1), n = 1, 2, 3, 4, all; k in {0.5, 1, 2, 4, 8}."""
import sys
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
cat = el.load_cat()
B = [el.march_bundle(p) for p in lab.PANELS]
designs = {f'sp{k}': dict(box=(-2, 2), ref='base', k=k, cw=0.0, ncount='obs', cw_all=1.0) for k in (0.5, 1, 2, 4, 8)}
for n in (1, 2, 3, 4, 99):
    R = en.run(cat, B, designs, lambda b: en.items_recent(b, n), f'recent{n}')
    print(f'\n== E1 recent n={n} (214 pseudo-March onset windows; evidence = the n days before the window day) ==')
    print(en.table(R).round(4).to_string(index=False))
