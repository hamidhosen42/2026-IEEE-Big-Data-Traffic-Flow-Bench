"""Hand the v22 F03 bag (fs4) to the 2026-10-04 chain, but only if the v21 F03 probe was KEPT on the LB.
Steps: wait for tfb-t2v22 to complete (no Traceback, fs4 file present, none of its 4 models skipped); find the chain slot the
v21 helper used (v21_to_override.log) and wait until the chain has scored it; if it was kept, write the override for the first
free slot (not done, no override file yet, before its deadline). Log: v22_to_override.log (no secrets are written)."""
import json, os, re, sys, time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 'v22_to_override.log'), 'a', buffering=1)
SLUG = 'tfb-t2v22'
DEADLINES = {'S3': dt.datetime(2026, 10, 4, 9, 0), 'S4': dt.datetime(2026, 10, 4, 15, 0), 'S5': dt.datetime(2026, 10, 4, 21, 0)}
STATE_F = os.path.join(HERE, 'auto_chain_oct4b_state.json')


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


def done_slots():
    return {d['slot']: d for d in json.load(open(STATE_F)).get('done', [])} if os.path.exists(STATE_F) else {}


while True:
    s = str(call(kjob.status, SLUG).get('status', '')).lower()
    if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
        break
    time.sleep(60)
log('v22 status', s)
if not s.endswith('complete'):
    log('v22 did not complete -> no override'); sys.exit(0)
r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={SLUG}')
t = ''.join(x['data'] for x in json.loads(r.get('log') or '[]'))
files = {f['fileName'] for f in r.get('files', [])}
for line in t.splitlines():
    if any(k in line for k in ('trained', 'CONTROL', 'SKIPPED', 'FOOTPRINT', 'mean predicted size', 'Traceback')):
        log('  v22:', line.strip()[:240])
skipped = set(re.findall(r'SKIPPED model (\w+)', t))
if 'Traceback' in t or 'queue_submission_fs4.csv' not in files or skipped & {'fs_np', 'fs_id', 'fs_np1', 'fs_id1'}:
    log('v22 output not usable (traceback / missing fs4 / skipped model) -> no override'); sys.exit(0)

# which slot did the v21 probe use, and was it kept?
v21_slot = None
while v21_slot is None:
    p = os.path.join(HERE, 'v21_to_override.log')
    m = re.findall(r'override written for (S\d)', open(p).read()) if os.path.exists(p) else []
    if m:
        v21_slot = m[-1]; break
    if os.path.exists(p) and re.search(r'no override|CONTROL failed|did not complete', open(p).read()):
        log('v21 probe was never submitted -> no v22 override (F03 not validated on the LB)'); sys.exit(0)
    time.sleep(60)
log('v21 probe used slot', v21_slot)
while v21_slot not in done_slots():
    time.sleep(60)
res = done_slots()[v21_slot]
log('v21 probe result:', res)
if not res.get('keep'):
    log('v21 F03 probe was not kept -> no v22 override'); sys.exit(0)
done = set(done_slots())
now = dt.datetime.utcnow()
slot = next((k for k in ('S3', 'S4', 'S5') if k not in done and now < DEADLINES[k] - dt.timedelta(minutes=2)
             and not os.path.exists(os.path.join(HERE, f'auto_chain_oct4b_override_{k}.json'))), None)
if slot is None:
    log('no free chain slot left today -> no override'); sys.exit(0)
ov = {'type': 'queue', 'src_kernel': SLUG, 'src_file': 'queue_submission_fs4.csv', 'cond': 'queue_ongoing', 'thr': 0.0, 'max_share': 0.10,
      'desc': 'EXP_COMBO_v22: chain base + ongoing rows from v22 fs4 (selection-faithful F03 bag: np/id x 2 seeds)'}
tmp = os.path.join(HERE, f'auto_chain_oct4b_override_{slot}.json.tmp')
json.dump(ov, open(tmp, 'w'), indent=1); os.replace(tmp, tmp[:-4])
log('override written for', slot, json.dumps(ov))
