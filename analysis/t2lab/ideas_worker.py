"""ideas slot worker (track `ideas`).  Run as `run_heavy.py -- ../../.venv/bin/python ideas_worker.py` so that it holds ONE
semaphore slot.  Takes pending jobs from results/ideas/queue (atomic claim by rename) and runs them one after another while the
slot budget allows (no job is started that would push the slot past ~24 minutes); exits at once when nothing is pending.
At most 2 ideas workers are active at a time (own non-blocking locks), so this track never holds all 3 slots.
Job file <NN_name>.job: line 1 = "<estimated minutes> <log path>", line 2 = shell command (cwd = analysis/t2lab)."""
import fcntl, glob, os, subprocess, sys, time

Q = 'results/ideas/queue'; BUDGET = 24 * 60
os.makedirs(Q, exist_ok=True)
if not glob.glob(f'{Q}/*.job'):
    print('[ideas worker] nothing pending, leaving the slot', flush=True); sys.exit(0)
held = None
for i in range(2):
    f = open(f'{Q}/active{i}.lock', 'a')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); held = f; break
    except BlockingIOError:
        f.close()
if held is None:
    print('[ideas worker] two ideas workers are already active, leaving the slot', flush=True); sys.exit(0)
t0 = time.time(); n = 0
while True:
    took = None
    for j in sorted(glob.glob(f'{Q}/*.job')):
        try:
            est = float(open(j).readline().split()[0]) * 60
        except Exception:
            continue
        if n > 0 and time.time() - t0 + est > BUDGET:
            continue
        tgt = j[:-4] + f'.run{os.getpid()}'
        try:
            os.rename(j, tgt)
        except OSError:
            continue
        took = tgt; break
    if took is None:
        break
    lines = open(took).read().strip().split('\n')
    log = lines[0].split()[1]; cmd = lines[1]; ts = time.time(); n += 1
    print(f'[ideas worker {os.getpid()}] {time.strftime("%H:%M:%S")} start {os.path.basename(took)}: {cmd}', flush=True)
    with open(log, 'w') as f:
        f.write(f'[ideas worker {os.getpid()}] start {time.strftime("%H:%M:%S")} cmd: {cmd}\n'); f.flush()
        rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        f.write(f'exit {rc} after {time.time() - ts:.0f}s\n')
    os.rename(took, took.rsplit('.', 1)[0] + ('.done' if rc == 0 else '.failed'))
    print(f'[ideas worker {os.getpid()}] {time.strftime("%H:%M:%S")} finished {os.path.basename(took)} rc {rc} in {time.time() - ts:.0f}s '
          f'(slot held {time.time() - t0:.0f}s)', flush=True)
print(f'[ideas worker {os.getpid()}] exit: {n} job(s), slot held {time.time() - t0:.0f}s', flush=True)
