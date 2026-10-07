"""Write the 2026-10-05 chain's P4 and P5 overrides (validation ongoing rows only; April rows unchanged).
P4: as soon as tfb-t2v24 completed cleanly (no Traceback, 'done', sw03 file, CONTROL sw03 on non-flagged windows: 0 rows changed):
    flagged (g72) March ongoing windows from v24 sw03 = F10 pair (all faithful windows 1.0).  Lab: train OOF +0.0041 (8/8 panels),
    pseudo-March +0.0030 (7/8).
P5: after P4 is scored and tfb-t2v22af (+ tfb-t2v24af if P4 was kept) completed cleanly:
    P4 kept -> non-flagged windows from v22af fs4af (all-finite F03 bag) + flagged windows from v24af f10 (all-finite F10 pair)
    P4 not  -> all March ongoing windows from v22af fs4af.  Lab: all-finite inputs np +0.0031 (8/8), id +0.0019 OOF; pseudo-March
    +0.003..+0.0065.
Log: p45b_planner.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
STATE_F = os.path.join(HERE, 'auto_chain_oct5_state.json')
LOG = open(os.path.join(HERE, 'p45b_planner.log'), 'a', buffering=1)


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


def done():
    return {d['slot']: d for d in json.load(open(STATE_F)).get('done', [])} if os.path.exists(STATE_F) else {}


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


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct5_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


def clean(slug, need, extra_ok=None):
    t, files = kernel_out(slug)
    for line in t.splitlines():
        if any(k in line for k in ('trained', 'CONTROL', 'Traceback', 'FOOTPRINT', ' done')):
            log(f'  {slug}:', line.strip()[:230])
    ok = 'Traceback' not in t and need <= files and (extra_ok(t) if extra_ok else True)
    log(slug, 'usable:', ok)
    return ok


log('P4/P5 planner started')
# ---------------- P4
s = wait_kernel('tfb-t2v24', dt.datetime(2026, 10, 5, 17, 40)); log('tfb-t2v24', s)
ctl = lambda t: (len(re.findall(r'CONTROL sw03 vs tfb-t2v22 fs4 \| \w+ g72=0: windows changed 0/\d+ \| rows changed 0/', t)) == 2)
if s.endswith('complete') and clean('tfb-t2v24', {'queue_submission_sw03.csv'}, ctl):
    write('P4', {'type': 'queue2', 'parts': [['tfb-t2v24', 'queue_submission_sw03.csv', 'queue_ongoing', 'validation', 'g72']], 'thr': 0.0,
                 'max_val_changed': 100000,
                 'desc': 'March ongoing rows of the spacing-blocked (g72) windows from an F10 pair (np+id 511, all faithful windows 1.0); '
                         'other windows and April unchanged'})
else:
    write('P4', {'type': 'skip'})
# ---------------- P5
while 'P4' not in done() and dt.datetime.utcnow() < dt.datetime(2026, 10, 5, 21, 30):
    time.sleep(60)
p4 = done().get('P4'); kept = bool(p4 and p4.get('keep')); log('P4 result', p4)
s22 = wait_kernel('tfb-t2v22af', dt.datetime(2026, 10, 5, 21, 30)); log('tfb-t2v22af', s22)
ok22 = s22.endswith('complete') and clean('tfb-t2v22af', {'queue_submission_fs4af.csv'})
if kept:
    s24 = wait_kernel('tfb-t2v24af', dt.datetime(2026, 10, 5, 21, 30)); log('tfb-t2v24af', s24)
    ok24 = s24.endswith('complete') and clean('tfb-t2v24af', {'queue_submission_f10.csv'})
    if ok22 and ok24:
        write('P5', {'type': 'queue2', 'parts': [['tfb-t2v22af', 'queue_submission_fs4af.csv', 'queue_ongoing', 'validation', 'nog72'],
                                                   ['tfb-t2v24af', 'queue_submission_f10.csv', 'queue_ongoing', 'validation', 'g72']],
                     'thr': 0.0, 'max_val_changed': 100000,
                     'desc': 'March ongoing rows with ALL-FINITE inputs (ineligible-but-valued cells as features): F03 bag on other windows, '
                             'F10 pair on g72 windows; April unchanged'})
    elif ok22:
        write('P5', {'type': 'queue2', 'parts': [['tfb-t2v22af', 'queue_submission_fs4af.csv', 'queue_ongoing', 'validation', 'nog72']],
                     'thr': 0.0, 'max_val_changed': 100000,
                     'desc': 'March ongoing rows of the non-g72 windows from the ALL-FINITE F03 bag (ineligible-but-valued cells as features); April unchanged'})
    else:
        write('P5', {'type': 'skip'})
else:
    if ok22:
        write('P5', {'type': 'queue2', 'parts': [['tfb-t2v22af', 'queue_submission_fs4af.csv', 'queue_ongoing', 'validation']],
                     'thr': 0.0, 'max_val_changed': 100000,
                     'desc': 'March ongoing rows from the ALL-FINITE F03 bag (ineligible-but-valued cells as features); April unchanged'})
    else:
        write('P5', {'type': 'skip'})
log('P4/P5 planner done')
