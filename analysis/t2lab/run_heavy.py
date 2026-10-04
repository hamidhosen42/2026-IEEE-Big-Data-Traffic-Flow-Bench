"""Run a heavy local command under a 3-slot semaphore (shared by all agents on this machine: 12 cores / 24 GB).
Usage:  .venv/bin/python analysis/t2lab/run_heavy.py -- <command ...>
Waits for a free slot, then runs the command with OMP/LightGBM threads capped via T2LAB_THREADS (default 4)."""
import fcntl, os, subprocess, sys, time
LOCKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.locks')
os.makedirs(LOCKDIR, exist_ok=True)
args = sys.argv[1:]
if args and args[0] == '--':
    args = args[1:]
assert args, 'no command given'
held = None
t0 = time.time()
while held is None:
    for i in range(3):
        f = open(os.path.join(LOCKDIR, f'slot{i}.lock'), 'w')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = f; f.write(f'{os.getpid()} {" ".join(args)[:200]}\n'); f.flush()
            break
        except BlockingIOError:
            f.close()
    if held is None:
        time.sleep(5)
print(f'[run_heavy] slot acquired after {time.time() - t0:.0f}s', flush=True)
env = dict(os.environ)
thr = env.get('T2LAB_THREADS', '4')
env.update(OMP_NUM_THREADS=thr, T2LAB_THREADS=thr, OPENBLAS_NUM_THREADS=thr, VECLIB_MAXIMUM_THREADS=thr)
rc = subprocess.call(args, env=env)
print(f'[run_heavy] exit {rc} after {time.time() - t0:.0f}s', flush=True)
sys.exit(rc)
