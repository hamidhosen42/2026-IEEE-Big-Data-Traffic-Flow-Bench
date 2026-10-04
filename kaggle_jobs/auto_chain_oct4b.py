"""Automated chain v2 for 2026-10-04 (resumable: state in auto_chain_oct4b_state.json; safe to restart at any time).
Base = tfb-c20 / submission_TV.csv (LB 0.87271). Every submission is a one-factor change on the current chain base.

  S1  T1 MIX: 0.5 * FD-feature bag + 0.5 * old bag-8, then FD recon + TV            keep if dLB >= +0.0001
  S2  ongoing og8 bag (v20)                                                         keep if dLB >= 0 (skip if changed share > 2%)
  S3, S4, S5  OVERRIDE SLOTS: each waits for kaggle_jobs/auto_chain_oct4b_override_<slot>.json until its deadline (UTC),
      then falls back to its default. Override formats:
        {"type": "queue", "src_kernel": "tfb-t2v21", "src_file": "queue_submission_fs2.csv", "cond": "queue_ongoing",
         "thr": 0.0005, "max_share": 0.25, "desc": "..."}
        {"type": "recon", "gate": 0.7, "a": 0.75, "logr": 0.5, "tv": [0.0075, 0.001, 0.02, 0.005], "thr": 0.0002, "desc": "..."}
        {"type": "t1mix", "w_fd": 0.5, "fd_members": ["tfb-t1v13", ...], "thr": 0.0001, "desc": "..."}
        {"type": "compose", "t1_kernel": "tfb-x", "t1_file": "submission_Y.csv", "thr": 0.0002, "desc": "..."}   (state rows from another kernel)
        {"type": "skip"}
      Defaults: S3 (09:00) recon gate 0.7; S4 (15:00) T1 mix with newly finished FD members, else recon |log r| <= 1.0;
                S5 (21:00) recon gate 0.8 if gate 0.7 was kept, else skipped.
Recon / TV are always rebuilt from the RAW Task 1 mix (state['raw_*']) and composed with the chain base's other rows."""
import json, os, re, subprocess, sys, time, urllib.request
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
_t0 = time.time()
while not os.path.exists(TOKEN_F):          # the Kaggle token file was removed on 2026-10-04 ~00:09 UTC; wait until it is back
    if time.time() - _t0 < 5:
        print('waiting for ~/.kaggle/access_token ...', flush=True)
    time.sleep(60)
import kjob  # noqa: E402

C = '2026-ieee-big-data-traffic-flow-bench'
LOG = open(os.path.join(HERE, 'auto_chain_oct4b.log'), 'a', buffering=1)
STATE_F = os.path.join(HERE, 'auto_chain_oct4b_state.json')
OLD = ('tfb-t1v11', 'tfb-t1v12', 'tfb-t1v12s2', 'tfb-t1v12s3', 'tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7')
FDM_ALL = ('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11', 'tfb-t1v13s12', 'tfb-t1v13s13')
TV_DEFAULT = [0.0075, 0.001, 0.02, 0.005]


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    msg = re.sub(r'KGAT_[A-Za-z0-9]+', 'KGAT_[REDACTED]', msg)
    print(msg, flush=True); LOG.write(msg + '\n')


state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {
    'base_kernel': 'tfb-c20', 'base_file': 'submission_TV.csv', 'base_score': 0.87271, 'best_score': 0.87271, 'done': [],
    'recon': {'gate': 0.6, 'a': 0.75, 'logr': 0.5, 'tv': TV_DEFAULT}, 'raw_kernel': 'tfb-c16', 'raw_file': 'submission_COMBO16.csv',
    'fd_used': [], 'versions': {}}


def save():
    tmp = STATE_F + '.tmp'
    json.dump(state, open(tmp, 'w'), indent=1); os.replace(tmp, STATE_F)


def _retry(fn, *a, **k):
    for i in range(40):
        try:
            return fn(*a, **k)
        except Exception as e:
            log('network retry', getattr(fn, '__name__', 'fn'), type(e).__name__, str(e)[:80]); time.sleep(30)
    raise RuntimeError('network failed 40 times')


def status(slug):
    return str(_retry(kjob.status, slug).get('status', ''))


def complete(k):
    return status(k).lower().endswith('complete')


def push(slug, script, sources):
    while True:
        r = _retry(kjob.push, slug, script, kernel_sources=[f'hosen42/{s}' for s in sources])
        if not r.get('hasError', True) and not r.get('errorNullable') and not r.get('http_error'):
            state['versions'][slug] = int(r.get('versionNumber') or 1); save()
            log('pushed', slug, 'version', state['versions'][slug]); time.sleep(75); return True
        log('push retry', slug, str(r)[:200]); time.sleep(180)


def wait(slug):
    while status(slug).lower() in ('queued', 'running', 'kernelworkerstatus.queued', 'kernelworkerstatus.running', ''):
        time.sleep(30)
    return status(slug)


def klog(slug):
    r = _retry(kjob.call, 'GET', f'/kernels/output?userName=hosen42&kernelSlug={slug}')
    if r.get('http_error'):
        return '', {}
    L = json.loads(r.get('log') or '[]')
    return ''.join(x['data'] for x in L), {f['fileName']: f['url'] for f in r.get('files', [])}


def ok_log(slug, marker):
    t, files = klog(slug)
    return (marker in t and 'Traceback' not in t), t, files


def wait_token():
    while not os.path.exists(TOKEN_F):
        log('token file missing - waiting'); time.sleep(120)


def list_subs():
    wait_token()
    tok = open(TOKEN_F).read().strip().splitlines()[0].strip()
    req = urllib.request.Request(f'https://www.kaggle.com/api/v1/competitions/submissions/list/{C}', headers={'Authorization': 'Bearer ' + tok})
    return json.load(urllib.request.urlopen(req, timeout=120))


def submit(slug, fname, desc, folder, upstreams):
    """Submit one kernel output file. Idempotent: a submission with the same description is never sent twice."""
    ver = state['versions'].get(slug, 1)
    prev = [x for x in _retry(list_subs) if (x.get('description') or '') == desc]
    ref, score = (prev[0].get('ref') if prev else None), None
    if ref is None:
        wait_token()
        env = dict(os.environ, KAGGLE_API_TOKEN=open(TOKEN_F).read().strip().splitlines()[0].strip())
        p = subprocess.run([os.path.join(ROOT, '.venv/bin/python'), '-W', 'ignore', os.path.join(HERE, 'submit.py'), slug, str(ver), desc, fname],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        m = re.search(r'submitted ref (\d+)', p.stdout)
        log('submit', slug, 'v', ver, fname, '->', (p.stdout.strip().splitlines() or ['no output'])[-1])
        if not m:
            again = [x for x in _retry(list_subs) if (x.get('description') or '') == desc]
            if not again:
                log('SUBMIT FAILED (nothing was submitted):', p.stderr.strip()[-300:]); return None
            ref = again[0].get('ref')
        else:
            ref = int(m.group(1))
            ms = re.search(r'complete\s+([0-9.]+)', p.stdout)
            score = float(ms.group(1)) if ms else None
    else:
        log('submission with this description already exists, ref', ref)
    for _ in range(90):
        if score is not None:
            break
        s = [x for x in _retry(list_subs) if str(x.get('ref')) == str(ref)]
        if s and s[0].get('status') == 'complete' and s[0].get('publicScore') not in (None, ''):
            score = float(s[0]['publicScore'])
        elif s and s[0].get('status') == 'error':
            log('submission error:', s[0].get('errorDescription')); return None
        else:
            time.sleep(30)
    subprocess.run(['python3', os.path.join(HERE, 'save_local.py'), folder, slug] + list(upstreams), cwd=ROOT, capture_output=True, text=True)
    return score


def apply(score, name, kernel, fname, thr, on_keep=None):
    if score is None:
        log(name, 'NO SCORE - base unchanged'); state['done'].append(dict(slot=name, score=None, keep=False)); save(); return False
    d = score - state['base_score']
    state['best_score'] = max(state['best_score'], score)
    keep = d >= thr
    log(f'{name}: LB {score:.5f} | delta vs chain base {d:+.5f} | rule >= {thr:+.4f} -> {"KEEP (new base)" if keep else "DROP (base unchanged)"}')
    if keep:
        state.update(base_kernel=kernel, base_file=fname, base_score=score)
        if on_keep:
            on_keep()
    state['done'].append(dict(slot=name, score=score, delta=d, keep=keep)); save()
    return keep


def skip(name, why):
    log(name, 'SKIPPED:', why); state['done'].append(dict(slot=name, score=None, keep=False, why=why)); save()


def write_variant(src, dst, repl):
    s = open(os.path.join(HERE, src)).read()
    for a, b in repl:
        n = len(re.findall(a, s, flags=re.M))
        assert n == 1, (src, a, n)
        s = re.sub(a, lambda _m, b=b: b, s, flags=re.M)
    open(os.path.join(HERE, dst), 'w').write(s)


def recon_kernel(slug, src_kernel, src_file, rc, do_recon=True, do_tv=True):
    tv = rc.get('tv') or TV_DEFAULT
    write_variant('swap_fdrecon.py', f'swap_{slug}.py', [
        (r"^DO_RECON, DO_TV = .*$", f"DO_RECON, DO_TV = {do_recon}, {do_tv}"),
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{src_kernel}', '{src_file}'"),
        (r"^GATE, A = .*$", f"GATE, A = {rc['gate']}, {rc['a']}"),
        (r"^LOGR_MAX = .*$", f"LOGR_MAX = {rc['logr']}"),
        (r"^TV_FREE, TV_FREE_A, TV_GATE, TV_GATE_A, TV_ITERS = .*$",
         f"TV_FREE, TV_FREE_A, TV_GATE, TV_GATE_A, TV_ITERS = {tv[0]}, {tv[1]}, {tv[2]}, {tv[3]}, 300")])
    push(slug, os.path.join(HERE, f'swap_{slug}.py'), [src_kernel])
    wait(slug)
    return ok_log(slug, 'DONE')


def compose_kernel(slug, t1_kernel, t1_file):
    write_variant('swap_compose.py', f'swap_{slug}.py', [
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'"),
        (r"^T1_KERNEL, T1_FILE, OUT_NAME = .*$", f"T1_KERNEL, T1_FILE, OUT_NAME = '{t1_kernel}', '{t1_file}', 'CMP'")])
    push(slug, os.path.join(HERE, f'swap_{slug}.py'), [state['base_kernel'], t1_kernel])
    wait(slug)
    return ok_log(slug, 'CMP OK')


# ------------------------------------------------------------------ actions
def act_t1mix(slot, pfx, w_fd, fdm, thr, desc, folder):
    fdm = list(fdm)
    write_variant('swap_t1mix.py', f'swap_{pfx}m.py', [
        (r"^BASE_KERNEL, BASE_FILE, OUT_NAME = .*$", f"BASE_KERNEL, BASE_FILE, OUT_NAME = '{state['base_kernel']}', '{state['base_file']}', 'T1MIX'"),
        (r"^W_FD = .*$", f"W_FD = {w_fd}"),
        (r"^FD_MEMBERS = .*$", 'FD_MEMBERS = ' + repr(tuple(fdm)))])
    push(f'{pfx}m', os.path.join(HERE, f'swap_{pfx}m.py'), [state['base_kernel']] + [k + p for k in OLD + tuple(fdm) for p in 'ab'])
    wait(f'{pfx}m'); ok, t, _ = ok_log(f'{pfx}m', 'T1MIX OK')
    if not ok:
        return skip(slot, 'mix kernel failed')
    for line in t.splitlines():
        if 'rms speed' in line or 'members' in line:
            log('  mix:', line.strip()[:160])
    ok, t, files = recon_kernel(pfx, f'{pfx}m', 'submission_T1MIX.csv', state['recon'])
    if not (ok and 'submission_FDRTV.csv' in files and 'EXCLUDED panels: []' in t):
        return skip(slot, 'recon kernel failed or a panel was excluded')
    sc = submit(pfx, 'submission_FDRTV.csv', desc, folder, [f'{pfx}m', state['base_kernel']])

    def on_keep():
        state.update(raw_kernel=f'{pfx}m', raw_file='submission_T1MIX.csv', fd_used=fdm)
    return apply(sc, slot, pfx, 'submission_FDRTV.csv', thr, on_keep)


def act_queue(slot, slug, src_kernel, src_file, cond, thr, max_share, desc, folder):
    if not complete(src_kernel):
        return skip(slot, f'{src_kernel} not complete')
    out_name = slug.replace('tfb-', '').upper()
    write_variant('swap_queue.py', f'swap_{slug}.py', [
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'"),
        (r"^SRC_KERNEL, SRC_FILE, COND, OUT_NAME = .*$", f"SRC_KERNEL, SRC_FILE, COND, OUT_NAME = '{src_kernel}', '{src_file}', '{cond}', '{out_name}'")])
    push(slug, os.path.join(HERE, f'swap_{slug}.py'), [state['base_kernel'], src_kernel])
    wait(slug); ok, t, _ = ok_log(slug, f'{out_name} OK')
    m = re.search(rf'{cond} rows changed share ([0-9.]+) \| positives before/after (\d+) (\d+)', t)
    share = float(m.group(1)) if m else 1.0
    log(f'  {slot}: {cond} rows changed share {share}' + (f' | positives {m.group(2)} -> {m.group(3)}' if m else ''))
    if not ok or share > max_share:
        return skip(slot, f'checks failed or changed share {share} > {max_share}')
    sc = submit(slug, f'submission_{out_name}.csv', desc, folder, [state['base_kernel'], src_kernel])
    return apply(sc, slot, slug, f'submission_{out_name}.csv', thr)


def act_recon(slot, pfx, rc, thr, desc, folder):
    """Rebuild recon + TV from the raw Task 1 mix with constants rc, compose with the chain base, submit."""
    ok, t, files = recon_kernel(f'{pfx}r', state['raw_kernel'], state['raw_file'], rc)
    if not (ok and 'submission_FDRTV.csv' in files):
        return skip(slot, 'recon kernel failed')
    if 'EXCLUDED panels: []' not in t:
        return skip(slot, 'a panel was excluded at these constants')
    for line in t.splitlines():
        if line.startswith('FDR:') or line.startswith('R: gate share'):
            log('  recon:', line.strip()[:170])
    ok, t, files = compose_kernel(pfx, f'{pfx}r', 'submission_FDRTV.csv')
    if not (ok and 'submission_CMP.csv' in files):
        return skip(slot, 'compose kernel failed')
    for line in t.splitlines():
        if 'state rows changed share' in line:
            log('  compose:', line.strip()[:170])
    sc = submit(pfx, 'submission_CMP.csv', desc, folder, [f'{pfx}r', state['raw_kernel'], state['base_kernel']])

    def on_keep():
        state['recon'] = dict(rc)
    return apply(sc, slot, pfx, 'submission_CMP.csv', thr, on_keep)


def act_compose(slot, pfx, t1_kernel, t1_file, thr, desc, folder):
    if not complete(t1_kernel):
        return skip(slot, f'{t1_kernel} not complete')
    ok, t, files = compose_kernel(pfx, t1_kernel, t1_file)
    if not (ok and 'submission_CMP.csv' in files):
        return skip(slot, 'compose kernel failed')
    sc = submit(pfx, 'submission_CMP.csv', desc, folder, [t1_kernel, state['base_kernel']])
    return apply(sc, slot, pfx, 'submission_CMP.csv', thr)


def fd_complete():
    return [k for k in FDM_ALL if complete(k + 'a') and complete(k + 'b')]


def run_override(slot, pfx, ov, folder):
    typ = ov.get('type')
    desc = f"{slot}: " + ov.get('desc', typ)
    if typ == 'skip':
        return skip(slot, 'override: skip')
    if typ == 'queue':
        return act_queue(slot, pfx, ov['src_kernel'], ov['src_file'], ov.get('cond', 'queue_ongoing'), ov.get('thr', 0.0), ov.get('max_share', 0.02), desc, folder)
    if typ == 'recon':
        rc = dict(state['recon']); rc.update({k: ov[k] for k in ('gate', 'a', 'logr', 'tv') if k in ov})
        return act_recon(slot, pfx, rc, ov.get('thr', 0.0002), desc, folder)
    if typ == 't1mix':
        return act_t1mix(slot, pfx, ov.get('w_fd', 0.5), ov.get('fd_members') or fd_complete(), ov.get('thr', 0.0001), desc, folder)
    if typ == 'compose':
        return act_compose(slot, pfx, ov['t1_kernel'], ov['t1_file'], ov.get('thr', 0.0002), desc, folder)
    return skip(slot, f'unknown override type {typ}')


def override_slot(slot, pfx, deadline, default_fn, folder):
    path = os.path.join(HERE, f'auto_chain_oct4b_override_{slot}.json')
    log(f'{slot}: waiting for {os.path.basename(path)} until {deadline:%Y-%m-%d %H:%M} UTC')
    while dt.datetime.utcnow() < deadline and not os.path.exists(path):
        time.sleep(30)
    if os.path.exists(path):
        try:
            ov = json.load(open(path))
        except Exception as e:
            log(slot, 'override unreadable', e); ov = None
        if ov:
            log(slot, 'override:', json.dumps(ov))
            return run_override(slot, pfx, ov, folder)
    log(slot, 'no override -> default')
    return default_fn()


# ------------------------------------------------------------------ run
target = dt.datetime(2026, 10, 4, 0, 2)
while dt.datetime.utcnow() < target:
    time.sleep(20)
log('START chain v2 | base', state['base_kernel'], state['base_file'], state['base_score'], '| done', [d['slot'] for d in state['done']])
done = {d['slot'] for d in state['done']}

if 'S1_MIX' not in done:
    fdm = fd_complete(); log('FD members complete:', fdm)
    if len(fdm) < 2:
        skip('S1_MIX', f'only {len(fdm)} FD members')
    else:
        act_t1mix('S1_MIX', 'tfb-e1', 0.5, fdm, 0.0001,
                  f'EXP_COMBO_23: T1 = 0.5 FD-feature bag ({len(fdm)}) + 0.5 old bag(8), then FD recon + TV', '33_E1_T1MIX')

if 'S2_OG8' not in done:
    act_queue('S2_OG8', 'tfb-e2', 'tfb-t2v20', 'queue_submission_og8.csv', 'queue_ongoing', 0.0, 0.02,
              'EXP_COMBO_24: chain base + ongoing rows from v20 og8 bag (np511r+id511r x 4 seeds)', '34_E2_OG8')


def default_s3():
    rc = dict(state['recon'], gate=0.7)
    return act_recon('S3', 'tfb-e3', rc, 0.0002, 'EXP_COMBO_25: FD recon extended to v<0.7vf (rebuilt from raw T1 mix, + TV)', '35_E3')


def default_s4():
    fdm = fd_complete()
    if len(fdm) > len(state.get('fd_used') or []) and len(fdm) >= 2:
        return act_t1mix('S4', 'tfb-e4', 0.5, fdm, 0.0001, f'EXP_COMBO_26: T1 = 0.5 FD-feature bag ({len(fdm)}) + 0.5 old bag(8), recon + TV', '36_E4')
    rc = dict(state['recon'], logr=1.0)
    return act_recon('S4', 'tfb-e4', rc, 0.0002, 'EXP_COMBO_26: FD recon with wider clip |log r|<=1.0 (from raw, + TV)', '36_E4')


def default_s5():
    if state['recon']['gate'] >= 0.7:
        rc = dict(state['recon'], gate=0.8)
        return act_recon('S5', 'tfb-e5', rc, 0.0002, 'EXP_COMBO_27: FD recon extended to v<0.8vf (from raw, + TV)', '37_E5')
    return skip('S5', 'no override and gate 0.7 was not kept')


if 'S3' not in done:
    override_slot('S3', 'tfb-e3', dt.datetime(2026, 10, 4, 9, 0), default_s3, '35_E3')
if 'S4' not in done:
    override_slot('S4', 'tfb-e4', dt.datetime(2026, 10, 4, 15, 0), default_s4, '36_E4')
if 'S5' not in done:
    override_slot('S5', 'tfb-e5', dt.datetime(2026, 10, 4, 21, 0), default_s5, '37_E5')

log('CHAIN DONE | final base', state['base_kernel'], state['base_file'], state['base_score'], '| best', state['best_score'])
