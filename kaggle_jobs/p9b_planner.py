"""2026-10-09 planner B (log p9b_planner.log, no secrets).
Q4: when tfb-t2ogpvogf (pick A April ongoing = pvo + 103 ogfeat columns; E2 lab: many-day pseudo-March +0.020 IoU 8/8,
    FM-PM private +0.0015) completed cleanly -> queue2 on the chain base (tfb-i4): private ongoing rows <- pvogfv (pre-registered
    tie-break; fallback pvogf if the veto file is missing).  max_val_changed 0 (public must not move), thr 0.0 -> the new file
    becomes the chain base = new final pick A.  Not clean by 17:15 UTC -> skip.
Q5: skip."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
LOG = open(os.path.join(HERE, 'p9b_planner.log'), 'a', buffering=1)
UNTIL = dt.datetime(2026, 10, 9, 17, 15)


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    LOG.write(re.sub(r'KGAT_[A-Za-z0-9]+', 'KGAT_[REDACTED]', msg) + '\n')


def call(fn, *a, **k):
    for _ in range(60):
        try:
            return fn(*a, **k)
        except Exception as e:
            log('network retry', type(e).__name__); time.sleep(30)
    raise RuntimeError('network failed')


def wait_kernel(slug, until):
    while dt.datetime.utcnow() < until:
        s = str(call(kjob.status, slug).get('status', '')).lower()
        if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
            return s
        time.sleep(120)
    return 'timeout'


def out(slug):
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    return ''.join(x['data'] for x in json.loads(r.get('log') or '[]')), {f['fileName'] for f in r.get('files', [])}


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct9_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


log('p9b planner start')
for slug in ('tfb-t2ogpvogfsm', 'tfb-t2ogpvogf'):
    s = wait_kernel(slug, UNTIL)
    t, files = out(slug) if s.endswith('complete') else ('', set())
    log(slug, 'status', s, '| done', 'PVOGF DONE' in t, '| traceback', 'Traceback' in t, '| files', sorted(files))
    for line in t.splitlines():
        if 'CONTROL' in line or 'veto' in line.lower() or 'peak' in line.lower() or 'skipped' in line.lower():
            log('  ', slug, line.strip()[:260])
ok = s.endswith('complete') and 'PVOGF DONE' in t and 'Traceback' not in t
fname = 'queue_submission_pvogfv.csv' if 'queue_submission_pvogfv.csv' in files else 'queue_submission_pvogf.csv'
if ok and fname in files:
    write('Q4', {"type": "queue2", "parts": [["tfb-t2ogpvogf", fname, "queue_ongoing", "private"]], "thr": 0.0, "max_val_changed": 0,
                 "desc": "FINAL CANDIDATE A2 (same-scenario bet) on tfb-i4: April ongoing = March-augmented F03 np/id/full bag + scenario prior "
                         f"+ 103 ogfeat columns ({fname.split('_')[-1][:-4]}; many-day pseudo-March +0.020 IoU 8/8, FM-PM private +0.0015); "
                         "March rows unchanged"})
else:
    write('Q4', {"type": "skip", "desc": "tfb-t2ogpvogf not clean in time"})
write('Q5', {"type": "skip", "desc": "free"})
log('p9b planner done')
