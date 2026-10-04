"""Build the cnn tensors for every panel / window set and check parity with the production features of the row cache:
the slot-T visibility draw (has_T), the last-valid queue state (qlast / q_last), vr_last, fr_last, ramp_on3 inside the keep region.
Usage (through run_heavy): cnn_prep.py"""
import sys, time
import numpy as np
import torch
import lab
import cnn_lib as C

SETS = ('official', 'offlike', 'pseudo', 'runstart', 'faith3')
for panel in lab.PANELS:
    t1 = time.time(); info = []
    for s in SETS:
        z = C.prep(panel, s)
        info.append(f'{s} {len(z["d"])}')
        if s in ('official', 'offlike'):
            rc = lab.rows(panel, s)
            p, widx = C.eval_panel(panel, s, 4, widx=np.arange(len(z['d'])))
            x, ql = C.assemble(p.vr, p.fr, p.vT, p.ramp, p.static, p.vis)
            assert x.shape[1] == C.N_CH, (x.shape, C.N_CH)
            assert np.array_equal(ql.numpy().astype(bool), rc.qlast), (panel, s, 'qlast differs')
            assert np.array_equal(z['Y'], rc.truth) and np.array_equal(z['M'], rc.ev)
            m = rc.h == 1
            col = lambda c: np.asarray(rc.X[np.where(m)[0]][:, rc.cols.index(c)])
            wi, li = rc.wi[m], rc.li[m]
            hasT = x[:, 62].numpy()[wi, li]
            assert np.array_equal(hasT, col('has_T')), (panel, s, 'has_T differs')
            last0 = x[:, 63].numpy()[wi, li]; ref = col('vr_last')
            ok = np.isfinite(ref)
            assert np.allclose(last0[ok], np.clip(ref[ok], 0, 2.5) - 1.0, atol=2e-3), (panel, s, 'vr_last differs', np.abs(last0[ok] - (np.clip(ref[ok], 0, 2.5) - 1.0)).max())
            fl = x[:, 60].numpy()[wi, li]; ref = col('fr_last'); ok = np.isfinite(ref)
            assert np.allclose(fl[ok], np.clip(ref[ok], 0, 1.5), atol=2e-3), (panel, s, 'fr_last differs')
            r3 = x[:, 67].numpy()[wi, li]; ref = col('ramp_on3'); ok = np.isfinite(ref)
            assert np.allclose(r3[ok], np.clip(ref[ok] * 10, 0, 3), atol=2e-2), (panel, s, 'ramp_on3 differs', np.abs(r3[ok] - np.clip(ref[ok] * 10, 0, 3)).max())
            assert np.array_equal(x[:, 70].numpy()[wi, li] > 0.5, ok), (panel, s, 'ramp mask differs')
            kk = np.zeros_like(z['keep'][:, 4]); m4 = m & rc.keep5[:, 4]; kk[rc.wi[m4], rc.li[m4]] = True
            assert np.array_equal(kk, z['keep'][:, 4])
            info[-1] += ' (parity ok)'
    lab.log(panel, ', '.join(info), f'{time.time() - t1:.0f}s')
    lab.drop_panel(panel)
lab.log('PREP DONE')
