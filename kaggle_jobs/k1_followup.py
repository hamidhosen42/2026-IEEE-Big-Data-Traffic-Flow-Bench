"""EXP_T1_CATB follow-up (user approved 'push koro' 2026-10-10): wait for tfb-t1catba..d -> push tfb-k1d -> push tfb-k1.
Does NOT submit. Log: k1_followup.log (no secrets)."""
import json, os, re, sys, time
import datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kjob  # noqa: E402
LOG = open(os.path.join(HERE, 'k1_followup.log'), 'a', buffering=1)
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
ok = True
for p in 'abcd':
    k = f'tfb-t1catb{p}'; s = wait(k); t, files = out(k)
    m = re.search(r'state rows (\d+) NaN (\d+)', t); pk = re.findall(r'peak RSS ([0-9.]+) GB', t)
    good = s.endswith('complete') and 'Traceback' not in t and m is not None and m.group(2) == '0' and 'state_submission_FTA.csv' in files
    log(k, s, '| state rows/NaN', m.groups() if m else None, '| peak', pk[-1] if pk else None, '| ok', good)
    for l in t.splitlines():
        if 'val_FTA' in l or 'panel time' in l:
            log('   ', l.strip()[:200])
    ok &= good
if not ok:
    log('members not clean -> stop'); raise SystemExit
O = ['tfb-t1v11', 'tfb-t1v12'] + [f'tfb-t1v12s{i}' for i in range(2, 8)]
F = ['tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11', 'tfb-t1dark', 'tfb-t1ramp', 'tfb-t1ramp2']
S = ['tfb-j4'] + [k + p for k in O + F for p in 'ab'] + ['tfb-t1bspeca', 'tfb-t1bspecb', 'tfb-t1bspecc'] + [f'tfb-t1catb{p}' for p in 'abcd']
push('tfb-k1d', 't1_catb_swap.py', S)
s = wait('tfb-k1d'); t, files = out('tfb-k1d')
log('tfb-k1d', s, '| OK marker', 'K1D OK' in t, '| traceback', 'Traceback' in t)
for l in t.splitlines():
    if 't1catb' in l or 'blackout' in l or 'rms' in l:
        log('   ', l.strip()[:220])
if not (s.endswith('complete') and 'K1D OK' in t and 'Traceback' not in t):
    log('k1d failed -> stop'); raise SystemExit
push('tfb-k1', 'swap_fdrecon_k1.py', ['tfb-k1d'])
s = wait('tfb-k1'); t, files = out('tfb-k1')
log('tfb-k1', s, '| files', sorted(files), '| traceback', 'Traceback' in t)
for l in t.splitlines():
    if 'EXCLUDED panels' in l or 'DONE' in l or l.startswith('FDR:') or 'gate share' in l:
        log('   ', l.strip()[:220])
log('READY' if (s.endswith('complete') and 'submission_FDRTV.csv' in files and 'EXCLUDED panels: []' in t and 'Traceback' not in t) else 'NOT READY')
