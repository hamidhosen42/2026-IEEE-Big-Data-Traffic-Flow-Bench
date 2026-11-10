"""run_batch.py JOBS FULL_ROOT REF_WORK CASES_DIR RESULTS_CSV PREFIX [key=value ... passed to cut_job.py]
Each line of JOBS: CUT MODE. Three cases at a time, each under analysis/t2lab/run_heavy.py (the shared 3-slot semaphore)."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
JOBS, FULL, REF, CASES, RES, PFX = sys.argv[1:7]
EXTRA = sys.argv[7:]
PY = f'{ROOT}/.venv/bin/python'
os.makedirs(CASES, exist_ok=True)


def run(line):
    cut, mode = line.split()
    name = f'{PFX}_{mode}_{cut[:16]}'
    cmd = [PY, f'{ROOT}/analysis/t2lab/run_heavy.py', '--', PY, f'{ROOT}/award/t2_postT/cut_job.py', FULL, REF, CASES, cut, name, RES,
           f'mode={mode}'] + EXTRA
    with open(f'{CASES}/{name}.log', 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    tail = open(f'{CASES}/{name}.log').read().strip().split('\n')[-2:]
    print(name, 'rc', rc, '|', ' / '.join(tail), flush=True)


lines = [l for l in open(JOBS).read().split('\n') if l.strip()]
with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, lines))
print('BATCH DONE', len(lines), flush=True)
