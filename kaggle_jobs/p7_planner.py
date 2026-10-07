"""Write the 2026-10-07 chain's Q3 override: hedge B (= Q5 of 2026-10-06) on the NEW chain base, only if Q1/Q2 changed the base
(base != tfb-g4); otherwise Q3 = skip (tfb-g5 already is hedge B of the unchanged base). Log: p7_planner.log (no secrets)."""
import json, os, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_F = os.path.join(HERE, 'auto_chain_oct7_state.json')
LOG = open(os.path.join(HERE, 'p7_planner.log'), 'a', buffering=1)
HEDGE = {"type": "queue2",
         "parts": [["tfb-t2osp", "queue_submission_base.csv", "queue_onset", "private"],
                   ["tfb-t2v22af", "queue_submission_fs4af.csv", "queue_ongoing", "private"]],
         "thr": 1.0, "max_val_changed": 0,
         "desc": "FINAL CANDIDATE B (hedge) on the ramp-member base: April onset = production CatBoost with the train prior; "
                 "April ongoing = all-finite F03 bag (train-only models); March rows unchanged"}


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a) + '\n')


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct7_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


log('p7 planner start')
while True:
    st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}
    done = {d['slot'] for d in st.get('done', [])}
    if {'Q1', 'Q2'} <= done:
        break
    if dt.datetime.utcnow() > dt.datetime(2026, 10, 7, 13, 50):
        log('Q1/Q2 not finished before the Q3 deadline -> nothing written'); raise SystemExit
    time.sleep(60)
log('Q1/Q2 done; base', st['base_kernel'], st['base_score'])
if st['base_kernel'] != 'tfb-g4':
    write('Q3', HEDGE)
else:
    write('Q3', {"type": "skip", "desc": "base unchanged (tfb-g4) -> hedge B is tfb-g5"})
log('p7 planner done')
