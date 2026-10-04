import numpy as np, pandas as pd, lab
lab.FILL_KINDS.update({'t288': (288, 0, 0), 't36s2': (36, 2, 0), 't36s4': (36, 4, 0), 't36s300': (36, 300, 0), 't288s300': (288, 300, 0), 't24': (24, 0, 0), 't72': (72, 0, 0)})
pd.set_option('display.width', 250)
fills = ('t36', 't24', 't72', 't288', 't36s2', 't36s4', 't36s300', 't288s300')
T = lab.replay_table(fills=fills, verbose=False)
T = T[(T.scan == 'interleaved') & (T.pers_state == 'fill')]
print(T.groupby(['fill', 'pers_state', 'cells'])[['on_exact', 'on_pm1', 'og_exact', 'og_pm1']].sum().to_string())
for panel in lab.PANELS:
    P = lab.load_panel(panel)
    print(panel, 'finite share', {f: round(float(lab.filled_state(P, f)['MF'].mean()), 4) for f in ('hb', 't36', 't288', 't36s300')})
    for fill, cells in (('t36', 'all'), ('t288', 'all'), ('t36s300', 'all')):
        on, og = lab.candidates(P, dict(pers_state='fill', cells=cells, fill=fill))
        a, b = lab.greedy(on, og, 'interleaved'); off = lab.official_windows(P)
        og_ = sorted((d, t) for d, t, c in off if c == lab.ONGOING)
        S = lab.selector_stats(P, fill); pe = S['pers'][('fill', cells)]
        if sorted(b) != og_:
            print('  ', fill, cells, 'off', og_, 'mine', b, 'pers at off', [round(float(pe[d, t]), 3) for d, t in og_], 'pers at mine', [round(float(pe[d, t]), 3) for d, t in b])
        # is the official origin a first crossing: pers at t-1, t-2 and est at t-1
        print('  ', fill, cells, 'pers(t-1) at off', [round(float(pe[d, t - 1]), 3) for d, t in og_], 'cand(t-1)', [bool(og[d, t - 1]) for d, t in og_])
    # onset with horizon queue on the filled state
    S = lab.selector_stats(P, 't36'); on2 = S['cov_ok'] & ~S['est'] & S['hq_fill']
    _, og = lab.candidates(P, dict(pers_state='fill', cells='all', fill='t36'))
    a, b = lab.greedy(on2, og, 'interleaved'); off = lab.official_windows(P)
    oo = sorted((d, t) for d, t, c in off if c == lab.ONSET); og_ = sorted((d, t) for d, t, c in off if c == lab.ONGOING)
    print('   onset hq_fill(t36): onset exact', len(set(a) & set(oo)), 'ongoing exact', len(set(b) & set(og_)), '' if sorted(a) == oo else (oo, a))
