"""Automated chain for 2026-10-04. Base = tfb-c20 / submission_TV.csv (LB 0.87271).
  S1 T1 MIX (0.5 FD-feature bag + 0.5 old bag) -> FD recon + TV (rebuilt from raw members)   keep if dLB >= +0.0001
  S2 ongoing og8 bag (v20)                                                                   keep if dLB >= 0 ; skip if changed share > 2%
  S3 FD recon with GATE 0.7 on the chain base (recon only)                                   keep if dLB >= +0.0002
  S4 T1 FD-only bag (W_FD = 1.0) -> recon (kept gate) + TV                                     keep if dLB >= +0.0001
  S5 GATE 0.8 if S3 kept, else LOGR_MAX 1.0, recon only                                        keep if dLB >= +0.0002
"""
import json, os, re, subprocess, sys, time, urllib.request
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 'auto_chain_oct4.log'), 'a', buffering=1)
STATE_F = os.path.join(HERE, 'auto_chain_oct4_state.json')
REUSE = set()


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + '\n')


state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {
    'base_kernel': 'tfb-c20', 'base_file': 'submission_TV.csv', 'base_score': 0.87271, 'best_score': 0.87271, 'done': [], 'gate': 0.6, 'raw_kernel': None}


def save():
    json.dump(state, open(STATE_F, 'w'), indent=1)


def _retry(fn, *a, **k):
    for i in range(20):
        try:
            return fn(*a, **k)
        except Exception as e:
            log('network retry', fn.__name__, type(e).__name__, str(e)[:80]); time.sleep(30)
    raise RuntimeError('network failed 20 times')


def status(slug):
    return str(_retry(kjob.status, slug).get('status', ''))


def push(slug, script, sources):
    if slug in REUSE and status(slug).lower().endswith('complete'):
        log('reuse already-complete kernel', slug); return True
    while True:
        r = _retry(kjob.push, slug, script, kernel_sources=[f'hosen42/{s}' for s in sources])
        if not r.get('hasError', True) and not r.get('errorNullable'):
            log('pushed', slug); time.sleep(60); return True
        log('push retry', slug, str(r)[:160]); time.sleep(180)


def wait(slug):
    while status(slug).lower() in ('queued', 'running', 'kernelworkerstatus.queued', 'kernelworkerstatus.running', ''):
        time.sleep(30)
    return status(slug)


def klog(slug):
    r = _retry(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    L = json.loads(r.get('log') or '[]')
    return ''.join(x['data'] for x in L), {f['fileName']: f['url'] for f in r.get('files', [])}


def fetch_small(slug, fname):
    _, files = klog(slug)
    if fname not in files:
        return None
    return urllib.request.urlopen(files[fname], timeout=300).read().decode()


def submit(slug, fname, desc, folder, upstreams):
    env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser('~/.kaggle/access_token')).read().strip())
    out = subprocess.run([os.path.join(ROOT, '.venv/bin/python'), '-W', 'ignore', os.path.join(HERE, 'submit.py'), slug, '1', desc, fname],
                         cwd=ROOT, env=env, capture_output=True, text=True).stdout
    m = re.search(r'complete\s+([0-9.]+)', out)
    log('submit', slug, fname, '->', out.strip().splitlines()[-1] if out.strip() else 'no output')
    subprocess.run(['python3', os.path.join(HERE, 'save_local.py'), folder, slug] + list(upstreams), cwd=ROOT, capture_output=True, text=True)
    return float(m.group(1)) if m else None


def apply(score, name, kernel, fname, thr):
    if score is None:
        log(name, 'NO SCORE - base unchanged'); return
    d = score - state['base_score']
    state['best_score'] = max(state['best_score'], score)
    keep = d >= thr
    log(f'{name}: LB {score:.5f} | delta vs chain base {d:+.5f} | rule >= {thr:+.4f} -> {"KEEP (new base)" if keep else "DROP (base unchanged)"}')
    if keep:
        state.update(base_kernel=kernel, base_file=fname, base_score=score)
    state['done'].append(dict(slot=name, score=score, delta=d, keep=keep)); save()


def write_variant(src, dst, repl):
    s = open(os.path.join(HERE, src)).read()
    for a, b in repl:
        n = len(re.findall(a, s, flags=re.M))
        assert n == 1, (src, a, n)
        s = re.sub(a, b, s, flags=re.M)
    open(os.path.join(HERE, dst), 'w').write(s)


def complete(k):
    return status(k).lower().endswith('complete')



OLD = ('tfb-t1v11', 'tfb-t1v12', 'tfb-t1v12s2', 'tfb-t1v12s3', 'tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7')
FDM_ALL = ('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11')


def skip(name, why):
    log(name, 'SKIPPED:', why); state['done'].append(dict(slot=name, score=None, keep=False)); save()


def ok_log(slug, marker):
    t, files = klog(slug)
    return (marker in t and 'Traceback' not in t), t, files


def recon_kernel(slug, src_kernel, src_file, do_recon, do_tv, gate, logr=0.5):
    write_variant('swap_fdrecon.py', f'swap_{slug}.py', [
        (r"^DO_RECON, DO_TV = .*$", f"DO_RECON, DO_TV = {do_recon}, {do_tv}"),
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{src_kernel}', '{src_file}'"),
        (r"^GATE, A = .*$", f"GATE, A = {gate}, 0.75"),
        (r"^LOGR_MAX = .*$", f"LOGR_MAX = {logr}")])
    push(slug, os.path.join(HERE, f'swap_{slug}.py'), [src_kernel])
    wait(slug)
    return ok_log(slug, 'DONE')


def t1mix_kernel(slug, w_fd, fdm):
    write_variant('swap_t1mix.py', f'swap_{slug}.py', [
        (r"^BASE_KERNEL, BASE_FILE, OUT_NAME = .*$", f"BASE_KERNEL, BASE_FILE, OUT_NAME = '{state['base_kernel']}', '{state['base_file']}', 'T1MIX'"),
        (r"^W_FD = .*$", f"W_FD = {w_fd}"),
        (r"^FD_MEMBERS = .*$", 'FD_MEMBERS = ' + repr(tuple(fdm)))])
    srcs = [state['base_kernel']] + [k + p for k in OLD + tuple(fdm) for p in 'ab']
    push(slug, os.path.join(HERE, f'swap_{slug}.py'), srcs)
    wait(slug)
    return ok_log(slug, 'T1MIX OK')


target = dt.datetime(2026, 10, 4, 0, 2)
while dt.datetime.utcnow() < target:
    time.sleep(60)
log('START chain oct4 | base', state['base_kernel'], state['base_file'], state['base_score'])
done = {d['slot'] for d in state['done']}
fdm = [k for k in FDM_ALL if complete(k + 'a') and complete(k + 'b')]
log('FD members complete:', fdm)

# ---------------- S1: T1 mix (0.5 FD) -> recon + TV
if 'S1_MIX' not in done:
    if len(fdm) < 2:
        skip('S1_MIX', f'only {len(fdm)} FD members')
    else:
        ok, t, _ = t1mix_kernel('tfb-d1m', 0.5, fdm)
        if not ok:
            skip('S1_MIX', 'mix kernel failed')
        else:
            ok, t, files = recon_kernel('tfb-d1', 'tfb-d1m', 'submission_T1MIX.csv', True, True, state['gate'])
            if ok and 'submission_FDRTV.csv' in files:
                sc = submit('tfb-d1', 'submission_FDRTV.csv', f'EXP_COMBO_23: T1 = 0.5 FD-feature bag ({len(fdm)}) + 0.5 old bag(8), then FD recon + TV',
                            '33_D1_T1MIX', ['tfb-d1m', state['base_kernel']])
                apply(sc, 'S1_MIX', 'tfb-d1', 'submission_FDRTV.csv', 0.0001)
            else:
                skip('S1_MIX', 'recon kernel failed')

# ---------------- S2: ongoing og8
if 'S2_OG8' not in done:
    if not complete('tfb-t2v20'):
        skip('S2_OG8', 'v20 not complete')
    else:
        write_variant('swap_queue.py', 'swap_d2.py', [
            (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'"),
            (r"^SRC_KERNEL, SRC_FILE, COND, OUT_NAME = .*$", "SRC_KERNEL, SRC_FILE, COND, OUT_NAME = 'tfb-t2v20', 'queue_submission_og8.csv', 'queue_ongoing', 'D2'")])
        push('tfb-d2', os.path.join(HERE, 'swap_d2.py'), [state['base_kernel'], 'tfb-t2v20'])
        wait('tfb-d2'); ok, t, _ = ok_log('tfb-d2', 'D2 OK')
        m = re.search(r'queue_ongoing rows changed share ([0-9.]+)', t); share = float(m.group(1)) if m else 1.0
        if ok and share <= 0.02:
            sc = submit('tfb-d2', 'submission_D2.csv', 'EXP_COMBO_24: chain base + ongoing rows from v20 og8 bag (np511r+id511r x 4 seeds)',
                        '34_D2_OG8', [state['base_kernel'], 'tfb-t2v20'])
            apply(sc, 'S2_OG8', 'tfb-d2', 'submission_D2.csv', 0.0)
        else:
            skip('S2_OG8', f'checks failed / share {share}')

# ---------------- S3: recon GATE 0.7 on the chain base (recon only)
if 'S3_G07' not in done:
    ok, t, files = recon_kernel('tfb-d3', state['base_kernel'], state['base_file'], True, False, 0.7)
    if ok and 'submission_FDR.csv' in files and 'EXCLUDED panels: []' in t:
        sc = submit('tfb-d3', 'submission_FDR.csv', 'EXP_COMBO_25: chain base + FD recon extended to v<0.7vf (recon only)', '35_D3_G07', [state['base_kernel']])
        apply(sc, 'S3_G07', 'tfb-d3', 'submission_FDR.csv', 0.0002)
        if state['done'][-1]['keep']:
            state['gate'] = 0.7; save()
    else:
        skip('S3_G07', 'kernel failed or a panel excluded at gate 0.7')

# ---------------- S4: FD-only T1 bag (W_FD 1.0) -> recon (kept gate) + TV
if 'S4_FDONLY' not in done:
    if len(fdm) < 3:
        skip('S4_FDONLY', f'only {len(fdm)} FD members')
    else:
        ok, t, _ = t1mix_kernel('tfb-d4m', 1.0, fdm)
        if not ok:
            skip('S4_FDONLY', 'mix kernel failed')
        else:
            ok, t, files = recon_kernel('tfb-d4', 'tfb-d4m', 'submission_T1MIX.csv', True, True, state['gate'])
            if ok and 'submission_FDRTV.csv' in files:
                sc = submit('tfb-d4', 'submission_FDRTV.csv', f'EXP_COMBO_26: T1 = FD-feature bag only ({len(fdm)}), then FD recon (gate {state["gate"]}) + TV',
                            '36_D4_FDONLY', ['tfb-d4m', state['base_kernel']])
                apply(sc, 'S4_FDONLY', 'tfb-d4', 'submission_FDRTV.csv', 0.0001)
            else:
                skip('S4_FDONLY', 'recon kernel failed')

# ---------------- S5: stronger recon
if 'S5_RECON2' not in done:
    if state['gate'] >= 0.7:
        gate, logr, desc = 0.8, 0.5, 'FD recon extended to v<0.8vf'
    else:
        gate, logr, desc = 0.6, 1.0, 'FD recon with wider clip |log r|<=1.0'
    ok, t, files = recon_kernel('tfb-d5', state['base_kernel'], state['base_file'], True, False, gate, logr)
    if ok and 'submission_FDR.csv' in files and 'EXCLUDED panels: []' in t:
        sc = submit('tfb-d5', 'submission_FDR.csv', f'EXP_COMBO_27: chain base + {desc} (recon only)', '37_D5_RECON2', [state['base_kernel']])
        apply(sc, 'S5_RECON2', 'tfb-d5', 'submission_FDR.csv', 0.0002)
    else:
        skip('S5_RECON2', 'kernel failed or panel excluded')

log('CHAIN DONE | final base', state['base_kernel'], state['base_file'], state['base_score'], '| best', state['best_score'])
