"""K7 chain (EXP_T1_DIV4 LRTC flow member, prepared 2026-10-30, NOT run): RUN ONLY AFTER THE USER / TEAM MEMBER APPROVES THE PUSH.
push tfb-k7 (swap_lrtc_k7.py: tfb-k5d + flow-only low-rank tensor completion member on ordinary rows, W 0.35, then recon + TV)
-> wait + check -> push tfb-k8 (swap_k8.py: hedge B = K6's April queue rows on tfb-k7) -> wait + check -> push tfb-t4val
(t4_validate_k7.py on tfb-k7 / tfb-k8) -> wait + check. Does NOT submit. Log: k7_followup.log (no secrets).
Candidate files afterwards: tfb-k7/submission_FDRTV.csv (pick A') and tfb-k8/submission_K8.csv (pick B')."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kjob  # noqa: E402
LOG = open(os.path.join(HERE, 'k7_followup.log'), 'a', buffering=1)


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + re.sub(r'KGAT_[A-Za-z0-9]+', 'KGAT_[REDACTED]', ' '.join(map(str, a))) + '\n')


def call(fn, *a, **k):
    for _ in range(60):
        try:
            return fn(*a, **k)
        except Exception as e:
            log('net retry', type(e).__name__); time.sleep(30)
    raise RuntimeError('network')


def wait(slug):
    while True:
        s = str(call(kjob.status, slug).get('status', '')).lower()
        if s.endswith(('complete', 'error', 'cancelacknowledged')):
            return s
        time.sleep(120)


def out(slug):
    r = call(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    return ''.join(x['data'] for x in json.loads(r.get('log') or '[]')), {f['fileName'] for f in r.get('files', [])}


def push(slug, f, src):
    while True:
        r = call(kjob.push, slug, os.path.join(HERE, f), kernel_sources=['hosen42/' + s for s in src])
        if not r.get('error') and not r.get('http_error') and r.get('versionNumber'):
            log('pushed', slug, 'v', r['versionNumber']); time.sleep(90); return
        log('push retry', slug, str(r)[:200]); time.sleep(180)


log('start')
push('tfb-k7', 'swap_lrtc_k7.py', ['tfb-k5d'])
s = wait('tfb-k7'); t, files = out('tfb-k7')
for l in t.splitlines():
    if l.startswith(('M ', 'M:', 'M OK', 'R:', 'FDRTV:', 'EXCLUDED panels', 'DONE')):
        log('   ', l.strip()[:230])
m = re.search(r'ordinary rows blended (\d+) share ([0-9.]+) .*rms \(q_lrtc - q\)/lane ([0-9.]+) corr ([0-9.]+)', t)
good = (s.endswith('complete') and 'Traceback' not in t and 'M OK' in t and 'EXCLUDED panels: []' in t and 'DONE' in t
        and 'submission_FDRTV.csv' in files and m is not None and float(m.group(2)) > 0.97)
log('tfb-k7', s, '| M share / rms / corr', m.groups()[1:] if m else None, '| files', sorted(files), '| ok', good)
if not good:
    log('k7 not clean -> stop'); raise SystemExit
push('tfb-k8', 'swap_k8.py', ['tfb-k7', 'tfb-t2osp', 'tfb-t2ogveto'])
s = wait('tfb-k8'); t, files = out('tfb-k8')
for l in t.splitlines():
    if l.startswith(('PART', 'TOTAL', 'K8')) or 'changed rows' in l:
        log('   ', l.strip()[:230])
tot = re.search(r'TOTAL changed rows: validation (\d+) \| private (\d+)', t)
good = (s.endswith('complete') and 'Traceback' not in t and 'K8 OK' in t and 'submission_K8.csv' in files
        and tot is not None and tot.group(1) == '0')
log('tfb-k8', s, '| changed validation / private', tot.groups() if tot else None, '| ok', good)
if not good:
    log('k8 not clean -> stop'); raise SystemExit
push('tfb-t4val', 't4_validate_k7.py', ['tfb-k7', 'tfb-k8'])
s = wait('tfb-t4val'); t, files = out('tfb-t4val')
log('tfb-t4val', s, '| PASSED' if 'T4 VALIDATION PASSED' in t else '| NOT PASSED', '| traceback', 'Traceback' in t)
log('READY (no submission made): A\' = tfb-k7/submission_FDRTV.csv, B\' = tfb-k8/submission_K8.csv'
    if (s.endswith('complete') and 'T4 VALIDATION PASSED' in t) else 'NOT READY')
