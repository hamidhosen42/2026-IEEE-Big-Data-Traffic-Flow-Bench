"""p8c = p8b for the oct8b run (specialist parts a v2 + b + c)."""
"""2026-10-08 chain planner B (log p8c_planner.log, no secrets).
Q4: when tfb-t1bspeca/b (blackout specialist, official timing; lab J +0.00044, 8/8) completed cleanly and Q3 is done ->
    t1dark swap: blackout rows 0.25 mix + 0.75 specialist (replaces the ramp member(s) there); FD bag = the chain base's fd_used.
Q5: after Q4: base changed -> hedge B on the new base; else skip."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
STATE_F = os.path.join(HERE, 'auto_chain_oct8b_state.json')
LOG = open(os.path.join(HERE, 'p8c_planner.log'), 'a', buffering=1)


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


def clean_member(slug, until):
    s = wait_kernel(slug, until)
    if not s.endswith('complete'):
        log(slug, 'status', s); return False
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    t = ''.join(x['data'] for x in json.loads(r.get('log') or '[]')); files = {f['fileName'] for f in r.get('files', [])}
    m = re.search(r'state rows (\d+) NaN (\d+)', t)
    ok = 'Traceback' not in t and 'state_submission_FTA.csv' in files and m is not None and m.group(2) == '0'
    log(slug, 'status', s, '| traceback', 'Traceback' in t, '| state rows/NaN', m.groups() if m else None, '| ok', ok)
    return ok


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct8b_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))



log('p8c planner start')
ok = all(clean_member(k, dt.datetime(2026, 10, 8, 22, 0)) for k in ('tfb-t1bspeca', 'tfb-t1bspecb', 'tfb-t1bspecc'))


def done_slots():
    st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}
    return st, {d['slot'] for d in st.get('done', [])}


st, _ = done_slots()
st = st or json.load(open(os.path.join(HERE, 'auto_chain_oct8_state.json')))
if ok:
    write('Q4', {"type": "t1dark", "swap_script": "t1_t1dark_swap.py", "dark": "tfb-t1bspec", "w_dark": 0.75, "thr": 0.0001,
                 "desc": "T1 blackout rows = 0.25 mix + 0.75 BLACKOUT SPECIALIST (trained on official-like blackout placement: onset queues "
                         "start at T+30, ongoing inside queues; 30 timing columns: onset/ongoing flag, offset, boundary values before/after "
                         "the gap, ramp flow over the gap); FD bag unchanged; recon + TV"})
else:
    write('Q4', {"type": "skip", "desc": "t1bspec members not clean in time"})
base0 = st['base_kernel']
while 'Q4' not in done_slots()[1]:
    time.sleep(60)
st, _ = done_slots()
log('Q4 done; base', st['base_kernel'], st['base_score'])
if st['base_kernel'] != base0:
    write('Q5', {"type": "queue2", "parts": [["tfb-t2osp", "queue_submission_base.csv", "queue_onset", "private"],
                                             ["tfb-t2v22af", "queue_submission_fs4af.csv", "queue_ongoing", "private"]],
                 "thr": 1.0, "max_val_changed": 0,
                 "desc": f"FINAL CANDIDATE B (hedge) on {st['base_kernel']} (blackout specialist): April onset = production CatBoost with the "
                         "train prior; April ongoing = all-finite F03 bag (train-only models); March rows unchanged"})
else:
    write('Q5', {"type": "skip", "desc": "base unchanged after Q4"})
log('p8c planner done')
