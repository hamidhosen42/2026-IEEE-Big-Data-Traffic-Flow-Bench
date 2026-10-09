#!/usr/bin/env python3
"""Regression test for the tfb-par-chaint1 failure ('written path_flow differs from base' in t1_recon, Kaggle 2026-10-09).

Cause: pandas' default CSV float parser (precise_xstrtod) is not correctly rounded, so one to_csv + read_csv cycle g() is not
idempotent - some values (mostly the tiny 1e-14-level path flows NNLS leaves on near-zero paths) need several cycles to
reach a fixed point. The Kaggle chain ran 26+ cycles from tfb-final to tfb-j4 and ended on fixed points; the entry script
has only 2 cycles before tfb-i4's re-read assert. Fix (t4 stage): write the fixed point of g, i.e. the chain's values.
This test (1) reproduces the failure with raw values, (2) shows the fixed point passes the tfb-i4 assert pattern and
equals the value after 26 chain cycles, (3) checks that the t4 stage contains the fix.
"""
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)


def g(v):
    return pd.read_csv(io.StringIO(pd.DataFrame({'path_flow': v}).to_csv(index=False)))['path_flow'].to_numpy()


def fixed_point(v, n=100):
    for _ in range(n):
        nx = g(v)
        if np.array_equal(nx, v):
            return v
        v = nx
    raise AssertionError('no fixed point')


def test_csv_fixed_point():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.random(200_000) * 10.0 ** rng.integers(-18, -12, 200_000), rng.gamma(2.0, 60.0, 50_000)])
    hist = [x]
    for _ in range(30):
        hist.append(g(hist[-1]))
    conv = np.array_equal(hist[-1], hist[-2]) or None
    keep = hist[26] == hist[27]                       # values the chain could carry (tfb-i4's assert excludes 2-cycles)
    x = x[keep]
    chain26 = g(x)
    for _ in range(25):
        chain26 = g(chain26)
    # old entry script: t4 writes raw -> t1_mix cycle -> t1_recon reads (base) -> writes + re-reads (chk)
    base = g(g(x))
    chk = g(base)
    n_old = int((chk != base).sum())
    # fixed: t4 writes the fixed point
    c = fixed_point(x)
    base2 = g(g(c))
    chk2 = g(base2)
    print(f'values {len(x)} | old: assert pattern fails on {n_old} values | fixed: fails on {int((chk2 != base2).sum())}, '
          f'== chain after 26 cycles: {bool(np.array_equal(base2, chain26))} | moved vs first parse {int((c != g(x)).sum())}')
    assert n_old > 0, 'expected the raw values to reproduce the Kaggle failure'
    assert np.array_equal(chk2, base2) and np.array_equal(base2, chain26)
    src = open(f'{CONS}/tfb/stages/t4_nnls.py').read()
    assert '_csv_cycle' in src and "odme['path_flow'] = _pf" in src, 't4 stage lacks the CSV fixed-point fix'
    return conv


if __name__ == '__main__':
    test_csv_fixed_point()
    print('CSV ROUNDTRIP OK')
