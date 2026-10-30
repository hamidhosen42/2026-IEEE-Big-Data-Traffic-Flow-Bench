"""DIAGNOSTIC - NOT FOR SUBMISSION (analysis/LEAK_DIAG_2026-10-29.md).  Uses post-T data on purpose (see leakdiag_lib).
Oracle ceilings of leakdiag_bounds.py on pseudo-March (new scenario, production-like base at its LB-like level)."""
import numpy as np
import pandas as pd

import lab
import leakdiag_eval as le
import leakdiag_bounds as lbd
import leakdiag_pmarch as lpm

F, ONSET, ONGOING = lab.F, lab.ONSET, lab.ONGOING
RI = le.RI


def main():
    on_items, _, og_items, _ = lpm.build_march()
    out = []
    for cond, items in ((ONSET, on_items), (ONGOING, og_items)):
        res = []
        for it in items:
            ok_ = it['FR'][..., RI['self_ok']] > 0; L = ok_.shape[2]
            near = np.zeros_like(ok_)
            for r in range(-2, 3):
                if r >= 0:
                    near[..., :L - r] |= ok_[..., r:]
                else:
                    near[..., -r:] |= ok_[..., :L + r]
            res.append(le.score(it, it['P6'], cond, 'pmarch', 'base'))
            res.append(le.score(it, lbd.oracle(it, cond, ok_), cond, 'pmarch', 'O_ramp'))
            res.append(le.score(it, lbd.oracle(it, cond, near), cond, 'pmarch', 'O_ramp_pm2'))
        df = pd.concat(res, ignore_index=True); b = lab.aggregate(df[df.arm == 'base'])
        for arm in ('base', 'O_ramp', 'O_ramp_pm2'):
            out.append(dict(cond=cond, arm=arm, iou=lab.aggregate(df[df.arm == arm]), d_base=lab.aggregate(df[df.arm == arm]) - b))
        out.append(dict(cond=cond, arm='O_all', iou=1.0, d_base=1.0 - b))
    O = pd.DataFrame(out); O.to_csv(f'{le.RES}/pmarch_oracle_bounds.csv', index=False)
    print('== pseudo-March oracle ceilings ==')
    print(O.round(4).to_string(index=False))
    for arm in ('O_ramp', 'O_ramp_pm2', 'O_all'):
        a = O[O.arm == arm].set_index('cond')
        print(f'{arm}: S_total ceiling = {0.15 * a.loc[ONSET, "d_base"] + 0.15 * a.loc[ONGOING, "d_base"]:.4f}')


if __name__ == '__main__':
    main()
