"""ogfeat worker: runs inside ONE semaphore slot and works through a list of runner jobs that no other ogfeat worker has claimed.
Usage (through run_heavy): ogfeat_worker.py <minutes budget> "<args of ogfeat_run.py>" ["<args>" ...]
A job is claimed by atomically creating data/t2local/cache/ogfeat/claims/<name>.claim, so several queued workers (several lottery
tickets for the shared 3-slot semaphore) never run the same job twice; a worker that finds nothing to do exits at once.
At most 2 ogfeat workers are active at a time (own non-blocking locks), so this track never holds all 3 slots.
No new job is started after the budget (default 22 min) has elapsed since the slot was acquired."""
import fcntl, os, re, subprocess, sys, time
import ogfeat_lib as og

budget = float(sys.argv[1]) * 60; jobs = sys.argv[2:]
cdir = f'{og.ODIR}/claims'; os.makedirs(cdir, exist_ok=True)
todo = [j for j in jobs if not os.path.exists(f"{cdir}/{re.sub(r'[^A-Za-z0-9_.=+-]+', '_', j)}.claim")]
if not todo:
    print('[worker] nothing to do', flush=True); sys.exit(0)
held = None
for i in range(2):
    f = open(f'{cdir}/active{i}.lock', 'a')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); held = f; break
    except BlockingIOError:
        f.close()
if held is None:
    print('[worker] two ogfeat workers are already active, leaving the slot', flush=True); time.sleep(20); sys.exit(0)
t0 = time.time(); py = sys.executable; here = os.path.dirname(os.path.abspath(__file__))
for job in jobs:
    if time.time() - t0 > budget:
        print(f'[worker] budget used, leaving: {job}', flush=True); continue
    name = re.sub(r'[^A-Za-z0-9_.=+-]+', '_', job)
    try:
        fd = os.open(f'{cdir}/{name}.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        continue
    os.write(fd, f'{os.getpid()} {time.ctime()}\n'.encode()); os.close(fd)
    print(f'[worker] {time.ctime()} start: {job}', flush=True)
    with open(f'{here}/logs/ogfeat_job_{name}.log', 'w') as lf:
        rc = subprocess.call([py, f'{here}/ogfeat_run.py'] + job.split(), stdout=lf, stderr=subprocess.STDOUT, cwd=here)
    print(f'[worker] {time.ctime()} exit {rc} after {time.time() - t0:.0f}s: {job}', flush=True)
print('[worker] done', flush=True)
