"""Write the 2026-10-05 chain's P1 override: the PRIVATE package (April rows only; March rows must not change).
  part 1: April onset rows from tfb-t2opv/queue_submission_pv.csv (same-scenario CatBoost bag) - verified 2026-10-04
  part 2: April ongoing rows from tfb-t2ogpv/queue_submission_pvo.csv (March-augmented F03 models + scenario prior), only if that
          kernel completed cleanly ('OGPV DONE', no Traceback, file present, validation rows identical to the chain base)
Waits for tfb-t2ogpv until 2026-10-05 05:30 UTC at the latest.  Log: p1_planner.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 'p1_planner.log'), 'a', buffering=1)
OV = os.path.join(HERE, 'auto_chain_oct5_override_P1.json')
LAST = dt.datetime(2026, 10, 5, 5, 30)


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


def kernel_log(slug):
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    return ''.join(x['data'] for x in json.loads(r.get('log') or '[]')), {f['fileName'] for f in r.get('files', [])}


log('P1 planner started')
parts = [['tfb-t2opv', 'queue_submission_pv.csv', 'queue_onset', 'private']]
t, files = kernel_log('tfb-t2opv')
assert 'PV DONE' in t and 'Traceback' not in t and 'queue_submission_pv.csv' in files, 'tfb-t2opv output not usable'
s = ''
while dt.datetime.utcnow() < LAST:
    s = str(call(kjob.status, 'tfb-t2ogpv').get('status', '')).lower()
    if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
        break
    time.sleep(120)
log('tfb-t2ogpv status', s)
og_ok = False
if s.endswith('complete'):
    t, files = kernel_log('tfb-t2ogpv')
    for line in t.splitlines():
        if any(k in line for k in ('trained', 'CONTROL', 'changed share', 'Traceback', 'OGPV DONE', 'March ongoing')):
            log('  ogpv:', line.strip()[:220])
    m = re.search(r'CONTROL chain-base ongoing vs pvo \(validation\): agreement ([0-9.]+)', t)
    og_ok = ('OGPV DONE' in t and 'Traceback' not in t and 'queue_submission_pvo.csv' in files and m is not None and float(m.group(1)) == 1.0)
log('ongoing part usable:', og_ok)
if og_ok:
    parts.append(['tfb-t2ogpv', 'queue_submission_pvo.csv', 'queue_ongoing', 'private'])
if os.path.exists(OV):
    log('override already exists -> not overwritten'); sys.exit(0)
ov = {'type': 'queue2', 'parts': parts, 'thr': 0.0, 'max_val_changed': 0,
      'desc': 'P1: PRIVATE package (April rows only, March unchanged): onset = same-scenario CatBoost bag (train + released March windows, '
              'scenario prior)' + ('; ongoing = March-augmented F03 np/id/full + scenario prior' if og_ok else '')}
tmp = OV + '.tmp'
json.dump(ov, open(tmp, 'w'), indent=1); os.replace(tmp, OV)
log('P1 override written', json.dumps(ov))
