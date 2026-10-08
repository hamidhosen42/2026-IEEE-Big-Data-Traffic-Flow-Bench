"""2026-10-08 chain planner (log p8_planner.log, no secrets).
Q2: when tfb-t1ramp2a/b (ramp member, seed 17) completed cleanly -> t1dark swap: blackout rows 0.25 mix + 0.75 mean(t1ramp, t1ramp2),
    FD bag of 6 (v13, s10, s11, t1dark, t1ramp, t1ramp2).  Not clean by 11:30 UTC -> skip.
Q3: after Q2: base changed -> hedge B on the new base; else skip (tfb-i1 is hedge B of tfb-h5)."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
STATE_F = os.path.join(HERE, 'auto_chain_oct8_state.json')
LOG = open(os.path.join(HERE, 'p8_planner.log'), 'a', buffering=1)


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
    p = os.path.join(HERE, f'auto_chain_oct8_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


log('p8 planner start')
until = dt.datetime(2026, 10, 8, 11, 30)
ok = clean_member('tfb-t1ramp2a', until) and clean_member('tfb-t1ramp2b', until)
if ok:
    write('Q2', {"type": "t1dark", "swap_script": "t1_t1dark_swap.py", "dark": "tfb-t1ramp+tfb-t1ramp2", "w_dark": 0.75, "thr": 0.0001,
                 "fd_members": ["tfb-t1v13", "tfb-t1v13s10", "tfb-t1v13s11", "tfb-t1dark", "tfb-t1ramp", "tfb-t1ramp2"],
                 "desc": "T1: second ramp member (seed 17): blackout rows 0.25 mix + 0.75 mean of the two ramp members; FD bag of 6 on all rows "
                         "(v13, s10, s11, t1dark, t1ramp, t1ramp2; W_FD 0.5); recon + TV"})
else:
    write('Q2', {"type": "skip", "desc": "t1ramp2 members not clean in time"})
while True:
    st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}
    if 'Q2' in {d['slot'] for d in st.get('done', [])}:
        break
    time.sleep(60)
log('Q2 done; base', st['base_kernel'], st['base_score'])
if st['base_kernel'] != 'tfb-h5':
    write('Q3', {"type": "queue2", "parts": [["tfb-t2osp", "queue_submission_base.csv", "queue_onset", "private"],
                                             ["tfb-t2v22af", "queue_submission_fs4af.csv", "queue_ongoing", "private"]],
                 "thr": 1.0, "max_val_changed": 0,
                 "desc": f"FINAL CANDIDATE B (hedge) on {st['base_kernel']} (two ramp members): April onset = production CatBoost with the train "
                         "prior; April ongoing = all-finite F03 bag (train-only models); March rows unchanged"})
else:
    write('Q3', {"type": "skip", "desc": "base unchanged -> hedge B is tfb-i1"})
log('p8 planner done')
