"""(p9, 2026-10-09: Q2 = hedge B if Q1 changed the base from tfb-i4.) Write the 2026-10-07 chain's Q3 override: hedge B (= Q5 of 2026-10-06) on the NEW chain base, only if Q1/Q2 changed the base
(base != tfb-g4); otherwise Q3 = skip (tfb-g5 already is hedge B of the unchanged base). Log: p9_planner.log (no secrets)."""
import json, os, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_F = os.path.join(HERE, 'auto_chain_oct9_state.json')
LOG = open(os.path.join(HERE, 'p9_planner.log'), 'a', buffering=1)
HEDGE = {"type": "queue2",
         "parts": [["tfb-t2osp", "queue_submission_base.csv", "queue_onset", "private"],
                   ["tfb-t2v22af", "queue_submission_fs4af.csv", "queue_ongoing", "private"]],
         "thr": 1.0, "max_val_changed": 0,
         "desc": "FINAL CANDIDATE B (hedge) on the specialist-in-FD-bag base: April onset = production CatBoost with the train prior; "
                 "April ongoing = all-finite F03 bag (train-only models); March rows unchanged"}


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a) + '\n')


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct9_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


log('p9 planner start')
while True:
    st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}
    done = {d['slot'] for d in st.get('done', [])}
    if 'Q1' in done:
        break
    if dt.datetime.utcnow() > dt.datetime(2026, 10, 9, 9, 50):
        log('Q1 not finished before the Q3 deadline -> nothing written'); raise SystemExit
    time.sleep(60)
log('Q1 done; base', st['base_kernel'], st['base_score'])
if st['base_kernel'] != 'tfb-i4':
    write('Q2', HEDGE)
else:
    write('Q2', {"type": "skip", "desc": "base unchanged (tfb-i4) -> hedge B is tfb-i5"})
log('p9 planner done')
