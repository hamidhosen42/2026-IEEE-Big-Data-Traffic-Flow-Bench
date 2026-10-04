"""Wait for the tfb-t2v21 full run, verify it, then hand its ongoing file to the 2026-10-04 chain as an override slot.
Checks before writing anything: kernel complete, no Traceback, CONTROL line with agreement >= 0.999 AND Jaccard >= 0.98,
the chosen file exists, and no model it depends on was SKIPPED by the time guard.
Variant: auto_chain_oct4b_v21_choice.json {"src_file": "...", "thr": 0.0, "max_share": 0.10} if present, else fs2.
Target slot: the first of S3 / S4 / S5 that the chain has not done, whose override file does not exist and whose deadline
has not passed (S3 09:00, S4 15:00, S5 21:00 UTC). Log: v21_to_override.log."""
import json, os, re, sys, time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
_t0 = time.time()
while not os.path.exists(TOKEN_F):          # the Kaggle token file was removed on 2026-10-04 ~00:09 UTC; wait until it is back
    if time.time() - _t0 < 5:
        print('waiting for ~/.kaggle/access_token ...', flush=True)
    time.sleep(60)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 'v21_to_override.log'), 'a', buffering=1)
SLUG = 'tfb-t2v21'
DEADLINES = {'S3': dt.datetime(2026, 10, 4, 9, 0), 'S4': dt.datetime(2026, 10, 4, 15, 0), 'S5': dt.datetime(2026, 10, 4, 21, 0)}
NEEDS = {'queue_submission_fs2.csv': ('fs_np', 'fs_id'), 'queue_submission_fs_np.csv': ('fs_np',), 'queue_submission_fs_id.csv': ('fs_id',),
         'queue_submission_fsblend.csv': ('cur_np', 'cur_id', 'fs_np', 'fs_id'), 'queue_submission_h_np.csv': ('h_np',)}


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + '\n')


def call(fn, *a):
    for _ in range(40):
        try:
            return fn(*a)
        except Exception as e:
            log('network retry', type(e).__name__, str(e)[:80]); time.sleep(30)
    raise RuntimeError('network failed')


# wait until the full kernel exists and has finished
while True:
    s = str(call(kjob.status, SLUG).get('status', '')).lower()
    if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
        break
    time.sleep(60)
log('v21 status', s)
if not s.endswith('complete'):
    log('v21 did not complete -> no override written'); sys.exit(0)
r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={SLUG}')
L = json.loads(r.get('log') or '[]'); t = ''.join(x['data'] for x in L)
files = {f['fileName'] for f in r.get('files', [])}
for line in t.splitlines():
    if any(k in line for k in ('trained', 'CONTROL', 'SKIPPED', 'FOOTPRINT', 'mean predicted size', 'Traceback')):
        log('  v21:', line.strip()[:240])
if 'Traceback' in t:
    log('Traceback in v21 log -> no override'); sys.exit(0)
m = re.search(r'CONTROL cur2 vs .*?agreement ([0-9.]+) \| Jaccard of queued cells ([0-9.]+)', t)
if not m:
    log('no CONTROL line -> no override'); sys.exit(0)
agree, jac = float(m.group(1)), float(m.group(2))
if agree < 0.999 or jac < 0.98:
    log(f'CONTROL failed (agreement {agree}, Jaccard {jac}) -> no override'); sys.exit(0)
choice_f = os.path.join(HERE, 'auto_chain_oct4b_v21_choice.json')
choice = json.load(open(choice_f)) if os.path.exists(choice_f) else {}
src = choice.get('src_file', 'queue_submission_fs2.csv')
skipped = set(re.findall(r'SKIPPED model (\w+)', t))
if src not in files or skipped & set(NEEDS.get(src, ())):
    log(f'chosen file {src} missing or a needed model was skipped ({skipped}) -> no override'); sys.exit(0)
state_f = os.path.join(HERE, 'auto_chain_oct4b_state.json')
done = {d['slot'] for d in json.load(open(state_f)).get('done', [])} if os.path.exists(state_f) else set()
now = dt.datetime.utcnow()
slot = next((k for k in ('S3', 'S4', 'S5') if k not in done and now < DEADLINES[k] - dt.timedelta(minutes=2)
             and not os.path.exists(os.path.join(HERE, f'auto_chain_oct4b_override_{k}.json'))), None)
if slot is None:
    log('no free chain slot left today -> no override'); sys.exit(0)
ov = {'type': 'queue', 'src_kernel': SLUG, 'src_file': src, 'cond': 'queue_ongoing',
      'thr': choice.get('thr', 0.0), 'max_share': choice.get('max_share', 0.10),
      'desc': choice.get('desc', f'EXP_COMBO_v21: chain base + ongoing rows from v21 {src.replace("queue_submission_", "").replace(".csv", "")} '
                                 '(selection-faithful ongoing training: faithful windows 0.3 + run-starts 1.0)')}
tmp = os.path.join(HERE, f'auto_chain_oct4b_override_{slot}.json.tmp')
json.dump(ov, open(tmp, 'w'), indent=1); os.replace(tmp, tmp[:-4])
log('override written for', slot, json.dumps(ov))
