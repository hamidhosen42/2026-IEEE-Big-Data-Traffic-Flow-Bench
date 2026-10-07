"""Write the 2026-10-05 chain's P3 override once P2 (LODO onset bag, sp4 prior) has been scored:
  P2 kept  -> P3 = March onset rows from tfb-t2lodo lodo_mix9 (LODO bag + early-month mixture prior)
  P2 not   -> P3 = March onset rows from tfb-t2lodo mix9 (production model + early-month mixture prior)
Validation (March) rows only; April rows unchanged.  Log: p3_planner.log."""
import json, os, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_F = os.path.join(HERE, 'auto_chain_oct5_state.json')
LOG = open(os.path.join(HERE, 'p3_planner.log'), 'a', buffering=1)


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a) + '\n')


def done():
    return {d['slot']: d for d in json.load(open(STATE_F)).get('done', [])} if os.path.exists(STATE_F) else {}


log('P3 planner started')
while 'P2' not in done() and dt.datetime.utcnow() < dt.datetime(2026, 10, 5, 13, 50):
    time.sleep(30)
r = done().get('P2'); log('P2 result', r)
p = os.path.join(HERE, 'auto_chain_oct5_override_P3.json')
if os.path.exists(p):
    log('P3 override exists -> kept'); raise SystemExit
if r and r.get('keep'):
    f, why = 'queue_submission_lodo_mix9.csv', 'LODO bag (kept in P2) + early-month mixture prior mix9'
else:
    f, why = 'queue_submission_mix9.csv', 'production CatBoost + early-month mixture prior mix9 (P2 LODO not kept)'
ov = {'type': 'queue2', 'parts': [['tfb-t2lodo', f, 'queue_onset', 'validation']], 'thr': 0.0, 'max_val_changed': 100000,
      'desc': f'March onset rows: {why} (per-cell two-hypothesis prior from the 1-4 March days before each window, W=6, pi=0.9); April rows unchanged'}
json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p)
log('P3 override written', json.dumps(ov))
