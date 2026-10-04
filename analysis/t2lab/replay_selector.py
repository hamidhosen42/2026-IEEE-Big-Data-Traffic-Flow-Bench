"""Selector reproduction table: per rule variant, exact and +-5 min matches of the official train onset / ongoing origins
(replay of the greedy from the first train day with the 5 + 5 quota).  Usage: replay_selector.py [fill,fill,...]"""
import sys, numpy as np, pandas as pd
import lab
fills = tuple(sys.argv[1].split(',')) if len(sys.argv) > 1 else ('hb', 't3', 't12', 't36', 't288', 't36s300')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
T = lab.replay_table(fills=fills, verbose=False)
T.to_csv(f'{lab.CACHE}/replay_table.csv', index=False)
key = ['fill', 'pers_state', 'cells', 'scan']
print('TOTAL over 8 panels (40 onset + 40 ongoing official train windows)')
print(T.groupby(key)[['on_exact', 'on_pm1', 'og_exact', 'og_pm1']].sum().to_string())
for v in ('og_exact', 'og_pm1', 'on_exact'):
    print(f'\n{v} per panel'); print(T.pivot_table(index=key, columns='panel', values=v).astype(int).to_string())
