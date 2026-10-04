"""Automated chain for 2026-10-05 (resumable: state in auto_chain_oct5_state.json; safe to restart at any time).
Base = the final base of the 2026-10-04 chain (tfb-e5 / submission_E5.csv, LB 0.87961). One-factor changes on the chain base.
Five OVERRIDE SLOTS P1..P5: each waits for kaggle_jobs/auto_chain_oct5_override_<slot>.json until its deadline (UTC), then
falls back to its default (P2: T1 mix with all finished FD members if more than the base uses; the others: skip).
Override formats: as the 2026-10-04 chain (queue / recon / t1mix / compose / skip) plus
  {"type": "queue2", "parts": [["tfb-t2opv", "queue_submission_pv.csv", "queue_onset", "private"], ...], "thr": 0.0,
   "max_val_changed": 0, "desc": "..."}   multi-part queue swap by condition and split (swap_queue2.py); max_val_changed caps the
   number of changed validation (public) rows, 0 = private-only change (the public score must not move)."""
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
LOG = open(os.path.join(HERE, 'auto_chain_oct5.log'), 'a', buffering=1)
STATE_F = os.path.join(HERE, 'auto_chain_oct5_state.json')
OLD = ('tfb-t1v11', 'tfb-t1v12', 'tfb-t1v12s2', 'tfb-t1v12s3', 'tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7')
FDM_ALL = ('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11', 'tfb-t1v13s12', 'tfb-t1v13s13')
TV_DEFAULT = [0.0075, 0.001, 0.02, 0.005]


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    msg = re.sub(r'KGAT_[A-Za-z0-9]+', 'KGAT_[REDACTED]', msg)
    print(msg, flush=True); LOG.write(msg + '\n')


if os.path.exists(STATE_F):
    state = json.load(open(STATE_F))
else:
    prev = json.load(open(os.path.join(HERE, 'auto_chain_oct4b_state.json')))
    state = {k: prev[k] for k in ('base_kernel', 'base_file', 'base_score', 'best_score', 'recon', 'raw_kernel', 'raw_file', 'fd_used')}
    state.update(done=[], versions={})


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


def act_queue2(slot, slug, parts, thr, max_val_changed, desc, folder):
    parts = [list(p) for p in parts]
    srcs = sorted({p[0] for p in parts})
    for k in srcs:
        if not complete(k):
            return skip(slot, f'{k} not complete')
    out_name = slug.replace('tfb-', '').upper()
    write_variant('swap_queue2.py', f'swap_{slug}.py', [
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'"),
        (r"^PARTS = .*$", 'PARTS = ' + repr([tuple(p) for p in parts])),
        (r"^OUT_NAME = .*$", f"OUT_NAME = '{out_name}'")])
    push(slug, os.path.join(HERE, f'swap_{slug}.py'), [state['base_kernel']] + srcs)
    wait(slug); ok, t, _ = ok_log(slug, f'{out_name} OK')
    for line in t.splitlines():
        if line.startswith('PART') or line.startswith('TOTAL') or line.strip().startswith(('validation:', 'private:')):
            log(f'  {slot}:', line.strip()[:200])
    m = re.search(r'TOTAL changed rows: validation (\d+) \| private (\d+)', t)
    nval = int(m.group(1)) if m else 10 ** 9
    if not ok or nval > max_val_changed:
        return skip(slot, f'checks failed or validation rows changed {nval} > {max_val_changed}')
    sc = submit(slug, f'submission_{out_name}.csv', desc, folder, [state['base_kernel']] + srcs)
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
    if typ == 'queue2':
        return act_queue2(slot, pfx, ov['parts'], ov.get('thr', 0.0), ov.get('max_val_changed', 0), desc, folder)
    if typ == 'recon':
        rc = dict(state['recon']); rc.update({k: ov[k] for k in ('gate', 'a', 'logr', 'tv') if k in ov})
        return act_recon(slot, pfx, rc, ov.get('thr', 0.0002), desc, folder)
    if typ == 't1mix':
        return act_t1mix(slot, pfx, ov.get('w_fd', 0.5), ov.get('fd_members') or fd_complete(), ov.get('thr', 0.0001), desc, folder)
    if typ == 'compose':
        return act_compose(slot, pfx, ov['t1_kernel'], ov['t1_file'], ov.get('thr', 0.0002), desc, folder)
    return skip(slot, f'unknown override type {typ}')


def override_slot(slot, pfx, deadline, default_fn, folder):
    path = os.path.join(HERE, f'auto_chain_oct5_override_{slot}.json')
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
target = dt.datetime(2026, 10, 5, 0, 2)
while dt.datetime.utcnow() < target:
    time.sleep(20)
log('START chain oct5 | base', state['base_kernel'], state['base_file'], state['base_score'], '| done', [d['slot'] for d in state['done']])
done = {d['slot'] for d in state['done']}


def default_skip(slot):
    return lambda: skip(slot, 'no override')


def default_p2():
    fdm = fd_complete()
    if len(fdm) > len(state.get('fd_used') or []) and len(fdm) >= 2:
        return act_t1mix('P2', 'tfb-f2', 0.5, fdm, 0.0001, f'EXP_COMBO_F2: T1 = 0.5 FD-feature bag ({len(fdm)}) + 0.5 old bag(8), recon + TV', '39_F2')
    return skip('P2', 'no override and no new FD members')


SLOTS = [('P1', 'tfb-f1', dt.datetime(2026, 10, 5, 6, 0), default_skip('P1'), '38_F1'),
         ('P2', 'tfb-f2', dt.datetime(2026, 10, 5, 10, 0), default_p2, '39_F2'),
         ('P3', 'tfb-f3', dt.datetime(2026, 10, 5, 14, 0), default_skip('P3'), '40_F3'),
         ('P4', 'tfb-f4', dt.datetime(2026, 10, 5, 18, 0), default_skip('P4'), '41_F4'),
         ('P5', 'tfb-f5', dt.datetime(2026, 10, 5, 22, 0), default_skip('P5'), '42_F5')]
for slot, pfx, deadline, dfn, folder in SLOTS:
    if slot not in done:
        override_slot(slot, pfx, deadline, dfn, folder)

log('CHAIN DONE | final base', state['base_kernel'], state['base_file'], state['base_score'], '| best', state['best_score'])
