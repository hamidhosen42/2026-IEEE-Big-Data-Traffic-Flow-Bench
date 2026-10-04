"""pvog slot worker.  Run as `run_heavy.py -- ../../.venv/bin/python pvog_worker.py` (so it holds ONE semaphore slot):
takes pending jobs from results/pvog/queue (atomic claim by rename) and runs them one after another while the slot budget
(about 25 minutes per slot, the lab rule) allows; exits when nothing pending fits.
Job file <NN_name>.job: line 1 = "<estimated minutes> <log path>", line 2 = shell command (cwd = analysis/t2lab)."""
import glob, os, subprocess, sys, time

Q = 'results/pvog/queue'; BUDGET = 26 * 60
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
    print(f'[worker {os.getpid()}] {time.strftime("%H:%M:%S")} start {os.path.basename(took)}: {cmd}', flush=True)
    with open(log, 'w') as f:
        f.write(f'[worker {os.getpid()}] start {time.strftime("%H:%M:%S")} cmd: {cmd}\n'); f.flush()
        rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        f.write(f'exit {rc} after {time.time() - ts:.0f}s\n')
    os.rename(took, took.rsplit('.', 1)[0] + ('.done' if rc == 0 else '.failed'))
    print(f'[worker {os.getpid()}] {time.strftime("%H:%M:%S")} finished {os.path.basename(took)} rc {rc} in {time.time() - ts:.0f}s (slot held {time.time() - t0:.0f}s)', flush=True)
print(f'[worker {os.getpid()}] exit: {n} job(s), slot held {time.time() - t0:.0f}s', flush=True)
