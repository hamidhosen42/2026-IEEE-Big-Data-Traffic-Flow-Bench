"""TRACK ogmodel: run my pending tasks in priority order under ONE semaphore slot (start through run_heavy).
A worker takes the first pending task of results/ogmodel/tasks.json (file-locked), runs it, and continues with the next
one only while the time since it got the slot plus the task's estimate stays within BUDGET minutes (default 24).
Usage: run_heavy.py -- python ogmodel_worker.py [budget_min]        |   ogmodel_worker.py add <name> <est_min> <log> [VAR=val ...] -- cmd ...
       ogmodel_worker.py list"""
import fcntl, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = f'{HERE}/results/ogmodel/tasks.json'; LOCK = f'{HERE}/results/ogmodel/tasks.lock'


def locked(fn):
    with open(LOCK, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        tasks = json.load(open(TASKS)) if os.path.exists(TASKS) else []
        out = fn(tasks)
        json.dump(tasks, open(TASKS, 'w'), indent=1)
        return out


if len(sys.argv) > 1 and sys.argv[1] == 'add':
    name, est, log = sys.argv[2], float(sys.argv[3]), sys.argv[4]
    rest = sys.argv[5:]; i = rest.index('--'); env = dict(a.split('=', 1) for a in rest[:i]); cmd = rest[i + 1:]
    pos = int(env.pop('POS', -1))

    def add(tasks):
        t = dict(name=name, est_min=est, log=log, env=env, cmd=cmd, state='pending')
        tasks.insert(pos, t) if pos >= 0 else tasks.append(t)
    locked(add); print('added', name); sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == 'list':
    for t in locked(lambda tasks: list(tasks)):
        print(t['state'], t['name'], t['est_min'], ' '.join(t['cmd']), t.get('secs', ''))
    sys.exit(0)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
t0 = time.time(); n = 0
while True:
    def take(tasks):
        for t in tasks:
            if t['state'] == 'pending' and (n == 0 or (time.time() - t0) / 60 + t['est_min'] <= BUDGET):
                t['state'] = 'running'; t['pid'] = os.getpid(); t['started'] = time.strftime('%H:%M:%S'); return dict(t)
        return None
    t = locked(take)
    if t is None:
        break
    t1 = time.time()
    with open(f'{HERE}/{t["log"]}', 'w') as lf:
        rc = subprocess.call(t['cmd'], stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, **t['env']), cwd=HERE)

    def done(tasks, name=t['name'], rc=rc, secs=time.time() - t1):
        for x in tasks:
            if x['name'] == name:
                x['state'] = 'done' if rc == 0 else f'failed({rc})'; x['secs'] = round(secs)
    locked(done); n += 1
    print(f'[worker] {t["name"]} exit {rc} after {time.time() - t1:.0f}s', flush=True)
print(f'[worker] {n} task(s), {time.time() - t0:.0f}s', flush=True)
