"""Hand the same-scenario onset prior (tfb-t2osp, variant sp4) to the 2026-10-04 chain as the S5 probe.
Steps: wait for tfb-t2osp to complete; require a clean log (no Traceback, 'OSP DONE', all variant files) and the CONTROL:
the base variant must reproduce the chain base's onset rows (agreement >= 0.999); then write the override for S5 if S5 is still
free (not done, no override file yet, before the 19:30 UTC fallback of s5_planner.py). Log: osp_to_override.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 'osp_to_override.log'), 'a', buffering=1)
SLUG, VARIANT, SLOT = 'tfb-t2osp', 'sp4', 'S5'
LAST = dt.datetime(2026, 10, 4, 19, 25)
STATE_F = os.path.join(HERE, 'auto_chain_oct4b_state.json')
OV = os.path.join(HERE, f'auto_chain_oct4b_override_{SLOT}.json')


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


log('osp helper started')
while True:
    s = str(call(kjob.status, SLUG).get('status', '')).lower()
    if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
        break
    time.sleep(60)
log('osp status', s)
if not s.endswith('complete'):
    log('osp did not complete -> no override'); sys.exit(0)
r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={SLUG}')
t = ''.join(x['data'] for x in json.loads(r.get('log') or '[]'))
files = {f['fileName'] for f in r.get('files', [])}
for line in t.splitlines():
    if any(k in line for k in ('trained', 'CONTROL', 'Traceback', 'Error', 'dupes', 'OSP DONE')):
        log('  osp:', line.strip()[:240])
need = {f'queue_submission_{n}.csv' for n in ('base', 'sp2', 'sp4', 'sp8')}
m = re.search(r'CONTROL chain-base onset vs base: agreement ([0-9.]+)', t)
agree = float(m.group(1)) if m else 0.0
if 'Traceback' in t or 'OSP DONE' not in t or not need <= files:
    log('osp output not usable (traceback / unfinished / missing files) -> no override'); sys.exit(0)
if agree < 0.999:
    log(f'CONTROL failed: chain-base onset agreement {agree} < 0.999 -> no override'); sys.exit(0)
done = {d['slot'] for d in json.load(open(STATE_F)).get('done', [])} if os.path.exists(STATE_F) else set()
if SLOT in done or os.path.exists(OV) or dt.datetime.utcnow() >= LAST:
    log(f'{SLOT} not free (done / override exists / past {LAST:%H:%M} UTC) -> no override'); sys.exit(0)
ov = {'type': 'queue', 'src_kernel': SLUG, 'src_file': f'queue_submission_{VARIANT}.csv', 'cond': 'queue_onset', 'thr': 0.0,
      'max_share': 0.10,
      'desc': f'EXP_COMBO_OSP: chain base + onset rows with the same-scenario prior ({VARIANT}: train prior blended per cell with '
              f'the March/April masked-layer prior of the days before T, smoothed +-2 slots, a=n/(n+4))'}
tmp = OV + '.tmp'
json.dump(ov, open(tmp, 'w'), indent=1); os.replace(tmp, OV)
log('override written for', SLOT, json.dumps(ov))
