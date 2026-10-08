"""Write the 2026-10-07 chain's Q4 override: March ongoing rows from the ogfeat (all2) F03 bag on top of all-finite + occupancy.
- waits for tfb-t2ogfa / tfb-t2ogfb (pushed 10-07 ~04:00 UTC, ~6 h each);
- both clean -> pushes the combine kernel tfb-t2ogfc (t2_ogfeat_comb.py) and uses fs4ogf; b failed or comb failed -> fs2ogf of a;
- a kernel is clean when complete, no Traceback, its marker logged and the csv present; the validation footprint vs fs4occ is logged
  and must be <= MAX_SHARE (the 3 % ALARM in the kernels is informational: the new features are expected to move rows).
Only validation (March) ongoing rows change, so the private rows (= final picks A tfb-h1 / B tfb-h3) are untouched.
Log: p7c_planner.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
LOG = open(os.path.join(HERE, 'p7c_planner.log'), 'a', buffering=1)
UNTIL = dt.datetime(2026, 10, 7, 17, 15)
MAX_SHARE = 0.10


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


def kernel_out(slug):
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    return ''.join(x['data'] for x in json.loads(r.get('log') or '[]')), {f['fileName'] for f in r.get('files', [])}


def clean(slug, marker, fname, until):
    s = wait_kernel(slug, until)
    if not s.endswith('complete'):
        log(slug, 'status', s); return False, ''
    t, files = kernel_out(slug)
    ok = marker in t and 'Traceback' not in t and fname in files
    log(slug, 'status', s, '| marker', marker in t, '| traceback', 'Traceback' in t, '|', fname, fname in files)
    for line in t.splitlines():
        if 'FOOTPRINT' in line or 'CONTROL' in line or 'skipped' in line.lower() or 'peak' in line.lower():
            log('  ', slug, line.strip()[:260])
    return ok, t


def val_share(t, name):
    m = re.findall(rf'FOOTPRINT {name} vs tfb-t2occ fs4occ \| validation: .*?\(([0-9.]+)%\)', t)
    return float(m[-1]) / 100 if m else None


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct7_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


DESC = ('March ongoing rows from the F03 bag (all-finite + occupancy) with 103 extra ONGOING features from the 60-min history and slot T '
        '(corridor context, per-link kinematics, queue-end demand/supply, volatility-normalised speed; nothing after T); April unchanged')
log('p7c planner start')
sm = wait_kernel('tfb-t2ogfsm', UNTIL)
t_sm, _ = kernel_out('tfb-t2ogfsm')
log('smoke tfb-t2ogfsm', sm, '| traceback', 'Traceback' in t_sm, '| done', 'done' in t_sm)
okA, tA = clean('tfb-t2ogfa', 'done', 'queue_submission_fs2ogf.csv', UNTIL)
okB, tB = clean('tfb-t2ogfb', 'done', 'queue_submission_fs2ogf1.csv', UNTIL) if okA else (False, '')
choice = None
if okA and okB:
    r = call(kjob.push, 'tfb-t2ogfc', os.path.join(HERE, 't2_ogfeat_comb.py'),
             kernel_sources=['hosen42/tfb-t2ogfa', 'hosen42/tfb-t2ogfb', 'hosen42/tfb-t2occ'])
    log('pushed tfb-t2ogfc', {k: v for k, v in r.items() if k in ('versionNumber', 'error')})
    time.sleep(90)
    okC, tC = clean('tfb-t2ogfc', 'COMB DONE', 'queue_submission_fs4ogf.csv', UNTIL + dt.timedelta(minutes=30))
    sh = val_share(tC, 'fs4ogf') if okC else None
    log('fs4ogf validation share changed', sh)
    if okC and sh is not None and sh <= MAX_SHARE:
        choice = ('tfb-t2ogfc', 'queue_submission_fs4ogf.csv', 'bag of 4')
if choice is None and okA:
    sh = val_share(tA, 'fs2ogf')
    log('fs2ogf validation share changed', sh)
    if sh is not None and sh <= MAX_SHARE:
        choice = ('tfb-t2ogfa', 'queue_submission_fs2ogf.csv', 'np + id seed 0')
if choice:
    write('Q4', {"type": "queue2", "parts": [[choice[0], choice[1], "queue_ongoing", "validation"]], "thr": 0.0,
                 "max_val_changed": 100000, "desc": f"{DESC} [{choice[2]}]"})
else:
    write('Q4', {"type": "skip", "desc": "ogfeat kernels not clean in time"})
write('Q5', {"type": "skip", "desc": "free"})
log('p7c planner done')
