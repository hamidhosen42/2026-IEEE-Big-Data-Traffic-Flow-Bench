"""phase_lib.py -- EXP_T2_PHASE: half-hour clock phase columns (pre-registered in results/phase/PREREG.txt).
t = slot of the origin T (0..287, slot 0 = 00:00), h = step 1..6; a ':05/:35 slot' s has s mod 6 == 1.  Clock of T only (<= T)."""
import hashlib, os
import numpy as np
import lab

PH = ['ph6', 'ph_on', 'ph_next', 'ph_x']
RES = f'{lab.ROOT}/analysis/t2lab/results/phase'
ARMD = f'{lab.CACHE}/phase'


def phase_cols(t, h):
    t = np.asarray(t, np.int64); h = np.asarray(h, np.int64)
    ph6 = t % 6
    ph_on = ((t + h) % 6 == 1)
    ph_next = 5 * (((-t) % 6) + 1)
    ph_x = (h * 5 >= ph_next)
    return np.stack([ph6, ph_on, ph_next, ph_x], 1).astype(np.float32)


def check_prereg():
    txt = open(f'{RES}/rule.txt', 'rb').read()
    s = hashlib.sha1(txt).hexdigest()
    assert s == '0640cceb83cd8ca36d6c2803b91098bc600fc9da', s
    assert open(f'{RES}/PREREG.txt').read().split('\n', 2)[2] == txt.decode()
    return s


if __name__ == '__main__':
    print(check_prereg())
    t = np.arange(12); print(phase_cols(np.repeat(t, 6), np.tile(np.arange(1, 7), 12)).reshape(12, 6, 4)[:, :, [0, 2]][:, 0])
    print(phase_cols(np.repeat(t, 6), np.tile(np.arange(1, 7), 12)).reshape(12, 6, 4)[:, :, 1])
    print(phase_cols(np.repeat(t, 6), np.tile(np.arange(1, 7), 12)).reshape(12, 6, 4)[:, :, 3])


# ---------------------------------------------------------------- EXP_T2_PHASE2 (second pre-registration, results/phase2/PREREG.txt)
RES2 = f'{lab.ROOT}/analysis/t2lab/results/phase2'


def check_prereg2():
    txt = open(f'{RES2}/rule.txt', 'rb').read()
    s = hashlib.sha1(txt).hexdigest()
    assert s == '0ff2a4359d24be8706a1483bdd6f98e7c73b3ea9', s
    assert open(f'{RES2}/PREREG.txt').read().split('\n', 2)[2] == txt.decode()
    return s


def onset_rows(P, Pv, wins, prior):
    """pseudo-March onset rows (verbatim copy of phase_onset.py / exp_onset_splitprior.py onset_rows)."""
    import warnings
    H, F = lab.H, lab.F
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins])
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = Pv['V'][d[:, None], idx]; Fh = Pv['Fq'][d[:, None], idx]; VT = Pv['V'][d, t]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        Fe, qlast = lab.window_features(P, Vh, Fh, t, Pv['dow'][d], prior, VT)
        lab.add_ramp(Fe, Pv['ON'], Pv['OFF'], d, t, P['cap'])
    W, L = qlast.shape
    R = dict(d=d, t=t, cond=np.array([lab.ONSET] * W), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = lab.prior_feats(prior, Pv['dow'][d], t, h)
        R['rows'].append((h, feats, Pv['Q'][d, np.minimum(t + h, 287)], Pv['E'][d, np.minimum(t + h, 287)]))
    return R


def cum_tables(Pv, smooth):
    """cumQ, cumE [3, D+1, 288, L] over the March days < d per class (copy of exp_onset_splitprior.cum_tables, tau=None)."""
    Q = Pv['Q'].astype(np.float32); E = Pv['E'].astype(np.float32)
    if smooth:
        def box(a):
            c = np.cumsum(np.pad(a, ((0, 0), (3, 2), (0, 0))), 1)
            return c[:, 5:] - c[:, :-5]
        Q, E = box(Q), box(E)
    D = Q.shape[0]; wk = Pv['dow'] >= 5
    cQ = np.zeros((3, D + 1) + Q.shape[1:], np.float32); cE = cQ.copy()
    for c, m in enumerate((~wk, wk, np.ones(D, bool))):
        cQ[c, 1:] = np.cumsum(Q * m[:, None, None], 0); cE[c, 1:] = np.cumsum(E * m[:, None, None], 0)
    return cQ, cE
