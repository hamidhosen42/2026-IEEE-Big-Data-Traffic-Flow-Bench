"""2026-10-06 chain helper: Q3 = March ongoing rows from tfb-t2occ fs4occ (all-finite + 13 occupancy features, F03 bag of 4).
Written only if the run completed cleanly: 'done' in the log, no Traceback, no 'SKIPPED model' (a time-guard skip counts as a
failure), queue_submission_fs4occ.csv present.  Smoke checks are logged for review.  Log: q3_planner.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
LOG = open(os.path.join(HERE, 'q3_planner.log'), 'a', buffering=1)


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    LOG.write(re.sub(r'KGAT_[A-Za-z0-9]+', 'KGAT_[REDACTED]', msg) + '\n')


def call(fn, *a):
    for _ in range(60):
        try:
            return fn(*a)
        except Exception as e:
            log('network retry', type(e).__name__); time.sleep(30)
    raise RuntimeError('network failed')


def wait(slug, until):
    while dt.datetime.utcnow() < until:
        s = str(call(kjob.status, slug).get('status', '')).lower()
        if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
            return s
        time.sleep(120)
    return 'timeout'


def out(slug):
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    return ''.join(x['data'] for x in json.loads(r.get('log') or '[]')), {f['fileName'] for f in r.get('files', [])}


log('Q3 planner started')
s = wait('tfb-t2occ-smoke', dt.datetime(2026, 10, 5, 12, 0)); log('smoke', s)
t, files = out('tfb-t2occ-smoke')
for line in t.splitlines():
    if any(k in line for k in ('occupancy', 'ocrit', 'Traceback', 'FOOTPRINT', ' done', 'trained')):
        log('  smoke:', line.strip()[:230])
s = wait('tfb-t2occ', dt.datetime(2026, 10, 6, 13, 30)); log('full', s)
t, files = out('tfb-t2occ')
for line in t.splitlines():
    if any(k in line for k in ('ocrit', 'Traceback', 'FOOTPRINT', ' done', 'trained', 'SKIPPED')):
        log('  full:', line.strip()[:230])
ok = s.endswith('complete') and 'Traceback' not in t and 'SKIPPED model' not in t and ' done' in t and 'queue_submission_fs4occ.csv' in files
p = os.path.join(HERE, 'auto_chain_oct6_override_Q3.json')
if ok and not os.path.exists(p):
    ov = {'type': 'queue2', 'parts': [['tfb-t2occ', 'queue_submission_fs4occ.csv', 'queue_ongoing', 'validation']], 'thr': 0.0,
          'max_val_changed': 100000,
          'desc': 'March ongoing rows from the F03 bag with all-finite inputs + 13 OCCUPANCY features (occupancy / per-link critical occupancy, '
                  'neighbours, slope, density agreement); April unchanged'}
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log('Q3 override written', json.dumps(ov))
else:
    log('Q3 not written: usable', ok, '| override exists', os.path.exists(p))
