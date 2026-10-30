"""DIAGNOSTIC - NOT FOR SUBMISSION (analysis/LEAK_DIAG_2026-10-29.md).  Uses post-T data on purpose (see leakdiag_lib).

Upper bounds for any leak that only informs the cells it can see (train offlike / official cell tables of leakdiag_build.py):
  O_ramp      truth substituted at every cell whose own on-ramp is usable at T+h (the most a perfect ramp reader could know
              about the attached links), production-like model elsewhere
  O_ramp_pm2  truth substituted at every link within +-2 links of a usable on-ramp
  O_pb        truth substituted at every link visible after the buffer (T+95..T+105) - a perfect post-buffer reader
  O_all       perfect forecast (IoU 1 by construction on scored cells) - the Task 2 ceiling
plus the share of the base model's error cells (missed + false under the production decode) that sit at those cells.
"""
import numpy as np
import pandas as pd

import lab
import leakdiag_eval as le
import leakdiag_lib as ld

F, ONSET, ONGOING = lab.F, lab.ONSET, lab.ONGOING
RI = le.RI
pd.set_option('display.width', 250)


def oracle(it, cond, mask):
    st = le.steps_of(cond)
    P6 = it['P6'].copy(); m = np.zeros_like(mask); m[:, st] = mask[:, st]
    tr = (it['truth'] & it['ev']).astype(np.float32)
    P6[m] = np.where(tr[m] > 0, 0.999, 0.0)
    return P6


def main():
    rows, err = [], []
    for cond in (ONSET, ONGOING):
        for s in ('offlike', 'official'):
            for it in le.load(s, cond):
                FR, FPB = it['FR'], it['FPB']
                ok_ = FR[..., RI['self_ok']] > 0
                L = ok_.shape[2]
                near = np.zeros_like(ok_)
                for r in range(-2, 3):
                    if r >= 0:
                        near[..., :L - r] |= ok_[..., r:]
                    else:
                        near[..., -r:] |= ok_[..., :L + r]
                pbv = FPB[..., ld.PB_FEATS.index('pb_vis')] > 0
                masks = {'O_ramp': ok_, 'O_ramp_pm2': near, 'O_pb': pbv}
                res = [le.score(it, it['P6'], cond, s, 'base')]
                for nm, m in masks.items():
                    res.append(le.score(it, oracle(it, cond, m), cond, s, nm))
                rows.append(pd.concat(res, ignore_index=True))
                # where are the base model's errors?
                st = le.steps_of(cond)
                for w in range(len(it['win'])):
                    pred = lab.decode(it['P6'][w].ravel()).reshape(F, -1)
                    T = it['truth'][w] & it['ev'][w]; Pp = pred & it['ev'][w]
                    e_ = (T ^ Pp)
                    err.append(dict(cond=cond, set=s, panel=it['panel'], n_err=int(e_.sum()), n_union=int((T | Pp).sum()),
                                    err_ramp=int((e_ & ok_[w]).sum()), err_pm2=int((e_ & near[w]).sum()), err_pb=int((e_ & pbv[w]).sum())))
    df = pd.concat(rows, ignore_index=True)
    out = []
    for (cond, s), g in df.groupby(['cond', 'set'], sort=False):
        b = lab.aggregate(g[g.arm == 'base'])
        for arm in dict.fromkeys(g.arm):
            out.append(dict(cond=cond, set=s, arm=arm, iou=lab.aggregate(g[g.arm == arm]), d_base=lab.aggregate(g[g.arm == arm]) - b))
        out.append(dict(cond=cond, set=s, arm='O_all', iou=1.0, d_base=1.0 - b))
    O = pd.DataFrame(out); O.to_csv(f'{le.RES}/oracle_bounds.csv', index=False)
    print('== oracle bounds (truth substituted on the cells a leak could see; production decode) ==')
    print(O.round(4).to_string(index=False))
    E = pd.DataFrame(err)
    g = E.groupby(['cond', 'set'])[['n_err', 'err_ramp', 'err_pm2', 'err_pb']].sum()
    for c in ('err_ramp', 'err_pm2', 'err_pb'):
        g[c + '_share'] = g[c] / g.n_err
    print('\n== share of the base model error cells (missed + false, scored cells) at cells a leak could see ==')
    print(g.round(3).to_string())
    g.to_csv(f'{le.RES}/error_location.csv')
    G = []
    for arm in ('O_ramp', 'O_ramp_pm2', 'O_pb', 'O_all'):
        a = O[(O.set == 'offlike') & (O.arm == arm)].set_index('cond')
        G.append(dict(arm=arm, d_onset=a.loc[ONSET, 'd_base'], d_ongoing=a.loc[ONGOING, 'd_base'],
                      S_total_gain=0.15 * a.loc[ONSET, 'd_base'] + 0.15 * a.loc[ONGOING, 'd_base']))
    print('\n== implied S_total ceilings (train offlike) ==')
    print(pd.DataFrame(G).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
