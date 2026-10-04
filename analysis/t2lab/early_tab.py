"""early track helper (light): bootstrap table + per-panel table for chosen designs from a saved scan frame."""
import sys
import numpy as np, pandas as pd
import lab, early_lib as el, early_engine as en
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
f = sys.argv[1]; names = sys.argv[2].split(','); ref2 = sys.argv[3] if len(sys.argv) > 3 else 'sp4'
R = pd.read_pickle(f)
for n in sorted(R.nrec.unique()):
    x = R[R.nrec == n]
    print(f'\n== nrec={n} ==')
    print(en.table(x, ref='base', ref2=ref2, designs=['base'] + names).round(4).to_string(index=False))
    print(en.panel_table(x, ['base'] + names).round(3).to_string())
