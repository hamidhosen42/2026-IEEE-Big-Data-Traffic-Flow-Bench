"""2026-10-06 chain helper.
Q2: when tfb-t2onaf (all-finite onset inputs) completed cleanly ('ONSET AF DONE', sp4 file), write Q2 = March onset rows from its
    sp4 variant (validation only).  Same input change as the +0.0009 ongoing probe; onset lab evidence weak (CIs span 0).
Q4: when tfb-t2ogpvaf (April ongoing package with all-finite inputs) completed cleanly ('OGPVAF DONE', validation agreement 1.0
    with the chain base), replace the April ongoing part of the Q4 override (candidate A) by its pvo file.
Log: q24_planner.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402
LOG = open(os.path.join(HERE, 'q24_planner.log'), 'a', buffering=1)


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


def status(slug):
    return str(call(kjob.status, slug).get('status', '')).lower()


def out(slug):
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    return ''.join(x['data'] for x in json.loads(r.get('log') or '[]')), {f['fileName'] for f in r.get('files', [])}


log('Q2/Q4 planner started')
q2_done = q4_done = False
until = dt.datetime(2026, 10, 6, 17, 30)
while not (q2_done and q4_done) and dt.datetime.utcnow() < until:
    if not q2_done:
        s = status('tfb-t2onaf')
        if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
            q2_done = True
            t, files = out('tfb-t2onaf')
            for line in t.splitlines():
                if 'CONTROL' in line or 'Traceback' in line or 'DONE' in line or 'trained' in line:
                    log('  onaf:', line.strip()[:220])
            p = os.path.join(HERE, 'auto_chain_oct6_override_Q2.json')
            if s.endswith('complete') and 'ONSET AF DONE' in t and 'Traceback' not in t and 'queue_submission_sp4.csv' in files and not os.path.exists(p):
                ov = {'type': 'queue2', 'parts': [['tfb-t2onaf', 'queue_submission_sp4.csv', 'queue_onset', 'validation']], 'thr': 0.0,
                      'max_val_changed': 100000,
                      'desc': 'March onset rows from CatBoost onset with ALL-FINITE inputs (ineligible-but-valued cells as features), same sp4 scenario prior; April unchanged'}
                json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log('Q2 override written', json.dumps(ov))
            else:
                log('Q2: tfb-t2onaf not usable or override exists ->', s)
    if not q4_done:
        s = status('tfb-t2ogpvaf')
        if s.endswith('complete') or s.endswith('error') or s.endswith('cancelacknowledged'):
            q4_done = True
            t, files = out('tfb-t2ogpvaf')
            for line in t.splitlines():
                if 'CONTROL' in line or 'Traceback' in line or 'DONE' in line or 'trained' in line or 'changed share' in line:
                    log('  ogpvaf:', line.strip()[:220])
            m = re.search(r'CONTROL chain-base ongoing vs pvo \(validation\): agreement ([0-9.]+)', t)
            ok = s.endswith('complete') and 'OGPVAF DONE' in t and 'Traceback' not in t and 'queue_submission_pvo.csv' in files and m and float(m.group(1)) == 1.0
            p = os.path.join(HERE, 'auto_chain_oct6_override_Q4.json')
            if ok and os.path.exists(p):
                ov = json.load(open(p))
                ov['parts'] = [x if x[2] != 'queue_ongoing' else ['tfb-t2ogpvaf', 'queue_submission_pvo.csv', 'queue_ongoing', 'private'] for x in ov['parts']]
                ov['desc'] = ov['desc'].replace('April ongoing = March-augmented F03 np/id/full + scenario prior',
                                                'April ongoing = March-augmented F03 np/id/full + scenario prior, all-finite inputs')
                json.dump(ov, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p); log('Q4 override updated', json.dumps(ov))
            else:
                log('Q4: tfb-t2ogpvaf not usable (keeping pvo) ->', s, 'control', m.group(1) if m else None)
    time.sleep(120)
log('Q2/Q4 planner done')
