#!/usr/bin/env python3
"""Single entry point: rebuild a final-pick submission of team hosen42 (2026 IEEE Big Data Cup - TrafficFlowBench).

  python run_all.py --data-root /data/kaggle_public --work-dir ./work --pick A2 --recipe lean1

--data-root   the competition's kaggle_public folder (holds submission_key.csv, sample_submission.csv, corridors/, task1/2/4/)
--work-dir    cache of every stage (<work>/kernels/<kernel>/...); a rerun skips finished stages
--pick        A2 = tfb-j4 (candidate A) | B2 = tfb-j3 (hedge B)
--recipe      full = all 15 Task 1 members | lean1 = 9 members (DESIGN.md 3.3; expected private delta -0.0002..-0.0003)
--stages      all (default) | light (t4, onset_sp, onset_pv, t1_mix, t1_recon, assemble) | comma list of stage names
--threads     LightGBM threads of the Task 1 members (Kaggle: 4); every Task 2 / Task 4 stage keeps the Kaggle value (4)
Result: <work>/submission_<pick>_<recipe>.csv (+ .sha256), manifest.json, and per stage .done.json (runtime, peak RSS).

Every stage of both picks is ported (milestone 2). Development only: outputs of a stage can be provided instead of
recomputed with --seed-kernel-outputs DIR (DIR/<kernel>/<file>, e.g. downloaded Kaggle outputs) or --input-root DIR[:glob].
Single member parts: --stages t1m:tfb-t1v12s4a (all parts of the recipe: t1_members).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tfb import config as C            # noqa: E402
from tfb.runner import Runner, StageError, parse_stages   # noqa: E402


def parse_set(items):
    """--set stage.key=JSON  ->  {stage: {key: value}}"""
    out = {}
    for it in items or ():
        k, v = it.split('=', 1)
        stage, key = k.split('.', 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        out.setdefault(stage, {})[key] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--pick', choices=C.PICKS, default='A2')
    ap.add_argument('--recipe', choices=C.RECIPES, default='lean1')
    ap.add_argument('--stages', default='all')
    ap.add_argument('--threads', type=int, default=4, help='Task 1 member LightGBM threads (Kaggle: 4)')
    ap.add_argument('--env-threads', type=int, default=4, help='OMP / BLAS threads of every stage process (Kaggle: 4 vCPU)')
    ap.add_argument('--smoke', action='store_true', help='kernel SMOKE switches (subsets); never for a real submission')
    ap.add_argument('--force', action='store_true', help='rerun the selected stages even if cached')
    ap.add_argument('--seed-kernel-outputs', metavar='DIR', help='development: copy DIR/<kernel>/... into the work dir first')
    ap.add_argument('--input-root', action='append', default=[], metavar='DIR[:glob]',
                    help='extra place to look for kernel outputs (flat DIR/<kernel>/<file>, or :glob = DIR/**/<kernel>/**/<file>)')
    ap.add_argument('--set', action='append', default=[], metavar='STAGE.KEY=JSON', help='override a stage constant (parity runs)')
    ap.add_argument('--list', action='store_true', help='print the stage plan and exit')
    a = ap.parse_args(argv)

    stages = parse_stages(a.stages, a.pick, a.recipe)
    roots = []
    for r in a.input_root:
        root, _, mode = r.partition(':')
        roots.append(dict(root=os.path.abspath(root), mode=mode or 'flat'))
    if a.list:
        for s in stages:
            print(f'{s:20s} module {str(C.stage_module(s)):12s} kernel {C.stage_kernel(s, a.pick):16s} outputs {C.stage_outputs(s, a.pick)}')
        return 0
    r = Runner(a.data_root, a.work_dir, a.pick, a.recipe, threads=a.threads, env_threads=a.env_threads, smoke=a.smoke,
               input_roots=roots, overrides=parse_set(a.set), force=a.force)
    if a.seed_kernel_outputs:
        r.seed(a.seed_kernel_outputs)
    try:
        res = r.run(stages)
    except StageError as e:
        print(f'[tfb] ERROR: {e}', flush=True)
        return 1
    print('[tfb] stages:', json.dumps(res), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
