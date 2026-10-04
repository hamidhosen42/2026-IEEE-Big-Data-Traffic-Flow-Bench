"""pvon work queue: run the next pending pvon job (called as the heavy command of run_heavy.py, so every job holds one slot).
A job is claimed with an atomic mkdir (results/pvon/.job_<name>): several waiters may call this script, each job runs once.
Usage: run_heavy.py -- python pvon_next.py"""
import os, subprocess, sys
JOBS = [
    ('step2_cut15_s1_S', ['pvon_step2.py', '15', '1', 'S']),
    ('shift_f0-1_s1_AB', ['pvon_shift.py', '0,1', '1', 'A,B', '0', 'bcm']),
    ('shift_f2-3_s1_AB', ['pvon_shift.py', '2,3', '1', 'A,B', '0', 'bcm']),
    ('step2_cut15_s2-3_S', ['pvon_step2.py', '15', '2,3', 'S']),
    ('step2_cut20_s1_S', ['pvon_step2.py', '20', '1', 'S']),
]
for name, args in JOBS:
    try:
        os.mkdir(f'results/pvon/.job_{name}')
    except FileExistsError:
        continue
    with open(f'logs/pvon_{name}.log', 'w') as f:
        rc = subprocess.call([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT)
    print('job', name, 'exit', rc, flush=True)
    sys.exit(rc)
print('no pending job', flush=True)
