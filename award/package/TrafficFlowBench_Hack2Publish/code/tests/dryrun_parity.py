#!/usr/bin/env python3
"""Local dry run of the generated parity kernels (consolidation/kaggle_parity/*.py) on the smoke fixture.

  python tests/dryrun_parity.py FIXTURE [--only tfb-par-t4,...]

For every kernel a fake /kaggle/input is built (FIXTURE/kaggle_input/<slug>/): the competition data plus exactly the
kernel sources its SPEC lists, each linked to a stand-in 'original' = the output of an earlier local smoke run:
  FIXTURE/work      light stages (A2 + B2, full)      FIXTURE/work_t2   Task 2 ongoing stages (A2 + B2)
  FIXTURE/work_t1   member parts (one per template)   FIXTURE/work_ps   pseudo windows
  FIXTURE/kernels   stand-in member / ongoing files   FIXTURE/kernels_t1/tfb-t2lab-full   pseudo seed of the member runs
plus two synthetic references for tfb-par-pvogf (tfb-t2v22af fs4af = the fs4occ smoke file, tfb-t2ogpvaf pvo = the pvogf
smoke file). Kernels whose stand-ins are missing are skipped. Each kernel then runs exactly as on Kaggle
(TFB_KAGGLE_INPUT / TFB_KAGGLE_WORKING, TFB_PARITY_SMOKE=1 = the kernels' SMOKE switches). Expected: PASS everywhere
(a rerun on the same machine is deterministic) except tfb-par-lean1 (CLOSE by design) and tfb-par-t4 (CLOSE: the
fixture's 'original' tfb-final is our own fixed-point skeleton, so the comparison vs the stand-in tfb-j4 is EXACT and the
NUMERIC line comes only from real chains). Tests bundling, lookups and comparisons - not numerical parity with Kaggle.
"""
import argparse
import ast
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T2_SLUGS = ('tfb-par-occ', 'tfb-par-pvogf', 'tfb-par-ogfa', 'tfb-par-ogfb', 'tfb-par-ogfc', 'tfb-par-ogveto')
CONS = os.path.dirname(HERE)


def spec_of(path):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'SPEC':
            return json.loads(node.value.args[0].value)
    raise ValueError(path)


def stand_in(fx, slug, kernel, member_kernel=False):
    if kernel == 'tfb-t2lab-full':
        return f'{fx}/work_ps/kernels/{kernel}' if slug == 'tfb-par-pseudo' else f'{fx}/kernels_t1/{kernel}'
    if kernel.startswith('tfb-t1'):              # member outputs: the smoke run for a member parity kernel, else the stand-ins
        p = f'{fx}/work_t1/kernels/{kernel}' if member_kernel else f'{fx}/kernels/{kernel}'
        return p if os.path.isdir(p) else None
    if kernel in ('tfb-t2occ', 'tfb-t2ogpvogf', 'tfb-t2ogveto') and slug not in T2_SLUGS and os.path.isdir(f'{fx}/kernels/{kernel}'):
        return f'{fx}/kernels/{kernel}'          # milestone-1 kernels: the stand-ins the light smoke assembled tfb-j4 / j3 from
    for w in ('work', 'work_t2', 'work_t1', 'work_ps'):
        p = f'{fx}/{w}/kernels/{kernel}'
        if os.path.isdir(p) and not os.path.exists(f'{p}/.seeded') and os.path.exists(f'{p}/.done.json'):
            return p
    p = f'{fx}/kernels/{kernel}'
    return p if os.path.isdir(p) else None


def synthetic_refs(fx):
    """tfb-par-pvogf references (fs4af / pvo): stand-ins built from the smoke outputs."""
    out = f'{fx}/kernels_refs'
    for k, f, src in (('tfb-t2v22af', 'queue_submission_fs4af.csv', f'{fx}/work_t2/kernels/tfb-t2occ/queue_submission_fs4occ.csv'),
                      ('tfb-t2ogpvaf', 'queue_submission_pvo.csv', f'{fx}/work_t2/kernels/tfb-t2ogpvogf/queue_submission_pvogf.csv')):
        if os.path.exists(src):
            os.makedirs(f'{out}/{k}', exist_ok=True)
            shutil.copy(src, f'{out}/{k}/{f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    fx = os.path.abspath(a.fixture)
    refs = synthetic_refs(fx)
    only = set(x for x in a.only.split(',') if x)
    verdicts = {}
    for f in sorted(glob.glob(f'{CONS}/kaggle_parity/tfb-par-*.py')):
        slug = os.path.basename(f)[:-3]
        if only and slug not in only:
            continue
        spec = spec_of(f)
        kin = f'{fx}/kaggle_input/{slug}'
        if os.path.exists(kin):
            shutil.rmtree(kin)
        os.makedirs(f'{kin}/competition')
        os.symlink(f'{fx}/kaggle_public', f'{kin}/competition/kaggle_public')
        missing = []
        for k in spec['sources']:
            src = stand_in(fx, slug, k, spec['stages'][0].startswith('t1m:')) or (f'{refs}/{k}' if os.path.isdir(f'{refs}/{k}') else None)
            if src is None:
                missing.append(k)
            else:
                os.symlink(src, f'{kin}/{k}')
        if missing:
            verdicts[slug] = f'SKIPPED (no local stand-in for {missing[:3]})'
            continue
        kwd = f'{fx}/kaggle_working/{slug}'
        if os.path.exists(kwd):
            shutil.rmtree(kwd)
        os.makedirs(kwd)
        env = dict(os.environ, TFB_KAGGLE_INPUT=kin, TFB_KAGGLE_WORKING=kwd, TFB_PARITY_SMOKE='1')
        p = subprocess.run([sys.executable, f], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        with open(f'{kwd}/log.txt', 'w') as lf:
            lf.write(p.stdout)
        lines = [x for x in p.stdout.splitlines() if x.startswith('PARITY ')]
        summ = [x for x in lines if x.startswith('PARITY SUMMARY')]
        verdicts[slug] = summ[-1] if summ else f'NO SUMMARY (exit {p.returncode}; see {kwd}/log.txt)'
        print(f'== {slug}: exit {p.returncode}')
        for x in lines:
            print('   ' + x[:200])
    print('\nDRY-RUN VERDICTS')
    for k, v in verdicts.items():
        print(f'  {k:20s} {v}')


if __name__ == '__main__':
    main()
