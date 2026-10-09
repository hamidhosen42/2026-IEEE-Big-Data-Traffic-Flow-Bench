#!/usr/bin/env python3
"""Local dry run of the generated parity kernels (consolidation/kaggle_parity/*.py) on the smoke fixture.

  python tests/dryrun_parity.py FIXTURE [--only tfb-par-t4,...]

Builds FIXTURE/kaggle_input (a fake /kaggle/input: competition data + every 'original' kernel output) where the
"originals" are the outputs of a previous local smoke run (FIXTURE/work, A2 + B2, full recipe) and the stand-in member /
ongoing files (FIXTURE/kernels). Every kernel then runs exactly as on Kaggle (TFB_KAGGLE_INPUT / TFB_KAGGLE_WORKING point
to the fake tree, TFB_PARITY_SMOKE=1 for the 2-panel onset subset). Expected: verdict PASS for every kernel (the rerun is
deterministic on the same machine) except tfb-par-lean1 (CLOSE: the lean recipe differs from the full one by design).
This tests the bundling, lookups and comparisons - not numerical parity with Kaggle.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    fx = os.path.abspath(a.fixture)
    kin = f'{fx}/kaggle_input'
    os.makedirs(f'{kin}/competition', exist_ok=True)
    links = {f'{kin}/competition/kaggle_public': f'{fx}/kaggle_public'}
    for k in os.listdir(f'{fx}/kernels'):
        links[f'{kin}/{k}'] = f'{fx}/kernels/{k}'
    for k in ('tfb-final', 'tfb-t2osp', 'tfb-t2opv', 'tfb-i4d', 'tfb-i4', 'tfb-j4', 'tfb-j3'):
        assert os.path.isdir(f'{fx}/work/kernels/{k}'), f'run the A2 + B2 smoke first ({k} missing)'
        links[f'{kin}/{k}'] = f'{fx}/work/kernels/{k}'
    for dst, src in links.items():
        if not os.path.islink(dst):
            os.symlink(src, dst)
    only = set(x for x in a.only.split(',') if x)
    verdicts = {}
    for f in sorted(glob.glob(f'{CONS}/kaggle_parity/tfb-par-*.py')):
        slug = os.path.basename(f)[:-3]
        if only and slug not in only:
            continue
        kwd = f'{fx}/kaggle_working/{slug}'
        os.makedirs(kwd, exist_ok=True)
        env = dict(os.environ, TFB_KAGGLE_INPUT=kin, TFB_KAGGLE_WORKING=kwd, TFB_PARITY_SMOKE='1')
        p = subprocess.run([sys.executable, f], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        with open(f'{kwd}/log.txt', 'w') as lf:
            lf.write(p.stdout)
        lines = [x for x in p.stdout.splitlines() if x.startswith('PARITY ')]
        summ = [x for x in lines if x.startswith('PARITY SUMMARY')]
        verdicts[slug] = summ[-1] if summ else f'NO SUMMARY (exit {p.returncode}; see {kwd}/log.txt)'
        print(f'== {slug}: exit {p.returncode}')
        for x in lines:
            print('   ' + x)
    print('\nDRY-RUN VERDICTS')
    for k, v in verdicts.items():
        print(f'  {k:20s} {v}')


if __name__ == '__main__':
    main()
