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
