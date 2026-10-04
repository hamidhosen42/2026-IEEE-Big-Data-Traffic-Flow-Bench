"""Write the 2026-10-05 chain's P4 / P5 overrides once P3 has been scored: if the per-day March onset probe (P3) was NOT kept,
P4 = March onset rows from the k=8 scenario-prior blend (tfb-t2osp sp8) and P5 = k=2 (sp2), validation rows only; if P3 was kept,
both are skipped (they would overwrite the P3 rows).  Log: p45_planner.log."""
import json, os, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_F = os.path.join(HERE, 'auto_chain_oct5_state.json')
LOG = open(os.path.join(HERE, 'p45_planner.log'), 'a', buffering=1)


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a) + '\n')


def done():
    return {d['slot']: d for d in json.load(open(STATE_F)).get('done', [])} if os.path.exists(STATE_F) else {}


def write(slot, ov):
    p = os.path.join(HERE, f'auto_chain_oct5_override_{slot}.json')
    if os.path.exists(p):
        log(slot, 'override exists -> kept'); return
    json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log(slot, 'override written', json.dumps(ov))


log('P4/P5 planner started')
while 'P3' not in done() and dt.datetime.utcnow() < dt.datetime(2026, 10, 5, 17, 50):
    time.sleep(60)
r = done().get('P3')
log('P3 result', r)
if r and r.get('keep'):
    write('P4', {'type': 'skip'}); write('P5', {'type': 'skip'})
else:
    for slot, k in (('P4', 8), ('P5', 2)):
        write(slot, {'type': 'queue2', 'parts': [['tfb-t2osp', f'queue_submission_sp{k}.csv', 'queue_onset', 'validation']], 'thr': 0.0,
                     'max_val_changed': 100000,
                     'desc': f'{slot}: March onset rows with the scenario-prior blend k={k} (tfb-t2osp sp{k}) instead of k=4; April rows unchanged'})
