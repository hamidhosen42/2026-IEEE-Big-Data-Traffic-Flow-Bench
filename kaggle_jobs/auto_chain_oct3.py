"""Automated chain for 2026-10-03 (plan: research/PLAN_2026-10-03.md). Five one-factor submissions, each built on the
current chain base; KEEP rules decide whether the next slot builds on the new file.
  slot 1 T1 bag-N (all finished FTA members)      keep if dLB >= -0.0001
  slot 2 FD-gated recon (FDR) on the chain base   keep if dLB >= +0.0002
  slot 3 ongoing og4 bag (v18)                    keep if dLB >= 0 ; skip if ongoing changed share > 2%
  slot 4 onset bag-stack (v19) only if in-kernel gates pass (else T1 FD probe if its gate passes, else skipped)
  slot 5 TV smoothing on the chain base           keep if dLB >= +0.0002
Every submission is saved locally (save_local.py). Log: auto_chain_oct3.log."""
import json, os, re, subprocess, sys, time, urllib.request
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 'auto_chain_oct3.log'), 'a', buffering=1)
STATE_F = os.path.join(HERE, 'auto_chain_oct3_state.json')
REUSE = {'tfb-c16'}   # pushed before the previous session died; its output (8-member bag) is valid


def log(*a):
    msg = f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + '\n')


state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {
    'base_kernel': 'tfb-combo15', 'base_file': 'submission_COMBO15.csv', 'base_score': 0.86944, 'best_score': 0.86944, 'done': []}


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


# ------------------------------------------------------------------ wait for the UTC quota reset
target = dt.datetime(2026, 10, 3, 0, 2)
while dt.datetime.utcnow() < target:
    time.sleep(60)
log('START chain | base', state['base_kernel'], state['base_file'], state['base_score'])
done = {d['slot'] for d in state['done']}

# ------------------------------------------------------------------ slot 1: T1 bag of all finished members
if 'S1_T1BAG' not in done:
    cand = ['tfb-t1v11', 'tfb-t1v12', 'tfb-t1v12s2', 'tfb-t1v12s3', 'tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7']
    members = [k for k in cand if complete(k + 'a') and complete(k + 'b')]
    log('S1 members', members)
    write_variant('swap_t1bag7.py', 'swap_c16.py', [
        (r"^BASE_KERNEL, BASE_FILE, OUT_NAME = .*$", f"BASE_KERNEL, BASE_FILE, OUT_NAME = '{state['base_kernel']}', '{state['base_file']}', 'COMBO16'"),
        (r"^MEMBERS = .*$", 'MEMBERS = ' + repr(tuple(members)))])
    push('tfb-c16', os.path.join(HERE, 'swap_c16.py'), [state['base_kernel']] + [k + p for k in members for p in 'ab'])
    wait('tfb-c16'); t, _ = klog('tfb-c16')
    if 'COMBO16 OK' in t and 'Traceback' not in t:
        sc = submit('tfb-c16', 'submission_COMBO16.csv', f'EXP_COMBO_18: Task 1 bag of {len(members)} FTA members ({",".join(m.split("-")[-1] for m in members)})',
                    '28_COMBO16', [state['base_kernel']] + [k + p for k in members for p in 'ab'])
        apply(sc, 'S1_T1BAG', 'tfb-c16', 'submission_COMBO16.csv', -0.0001)
    else:
        log('S1 kernel failed checks; skipped'); state['done'].append(dict(slot='S1_T1BAG', score=None, keep=False)); save()

# ------------------------------------------------------------------ slot 2: FD-gated recon
if 'S2_FDR' not in done:
    write_variant('swap_fdrecon.py', 'swap_c17.py', [
        (r"^DO_RECON, DO_TV = .*$", "DO_RECON, DO_TV = True, True"),
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'")])
    push('tfb-c17', os.path.join(HERE, 'swap_c17.py'), [state['base_kernel']])
    wait('tfb-c17'); t, files = klog('tfb-c17')
    if 'DONE' in t and 'Traceback' not in t and 'submission_FDR.csv' in files:
        sc = submit('tfb-c17', 'submission_FDR.csv', 'EXP_COMBO_19: chain base + FD-gated reconciliation of Task 1 congested cells (v<0.6vf)',
                    '29_COMBO17_FDR', [state['base_kernel']])
        apply(sc, 'S2_FDR', 'tfb-c17', 'submission_FDR.csv', 0.0002)
    else:
        log('S2 kernel failed checks; skipped'); state['done'].append(dict(slot='S2_FDR', score=None, keep=False)); save()

# ------------------------------------------------------------------ slot 3: ongoing og4 bag (v18)
if 'S3_OG4' not in done:
    write_variant('swap_queue.py', 'swap_c18.py', [
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'"),
        (r"^SRC_KERNEL, SRC_FILE, COND, OUT_NAME = .*$", "SRC_KERNEL, SRC_FILE, COND, OUT_NAME = 'tfb-t2v18', 'queue_submission_og4.csv', 'queue_ongoing', 'COMBO18'")])
    push('tfb-c18', os.path.join(HERE, 'swap_c18.py'), [state['base_kernel'], 'tfb-t2v18'])
    wait('tfb-c18'); t, _ = klog('tfb-c18')
    m = re.search(r'queue_ongoing rows changed share ([0-9.]+)', t)
    share = float(m.group(1)) if m else 1.0
    if 'COMBO18 OK' in t and 'Traceback' not in t and share <= 0.02:
        sc = submit('tfb-c18', 'submission_COMBO18.csv', 'EXP_COMBO_20: chain base + ongoing rows from v18 og4 bag (np511r+id511r x 2 seeds)',
                    '30_COMBO18_OG4', [state['base_kernel'], 'tfb-t2v18'])
        apply(sc, 'S3_OG4', 'tfb-c18', 'submission_COMBO18.csv', 0.0)
    else:
        log('S3 skipped: checks failed or changed share', share); state['done'].append(dict(slot='S3_OG4', score=None, keep=False)); save()


# ------------------------------------------------------------------ slot 4: onset bag-stack (gated) or T1 FD probe (gated)
def onset_gate():
    if not complete('tfb-t2onv19'):
        return None, 'v19 not complete'
    t, _ = klog('tfb-t2onv19')
    if 'CONTROL PASS' not in t:
        return None, 'CONTROL not PASS'
    import io
    import pandas as pd
    csv = fetch_small('tfb-t2onv19', 'cv_onset_v19.csv')
    if not csv:
        return None, 'no cv csv'
    cv = pd.read_csv(io.StringIO(csv))
    for arm, need_d, need_z in (('ABC_st03', 0.004, 2.0), ('A0_st03', 0.002, 1.5)):
        p = cv[(cv.arm == arm) & (cv.set == 'pseudo')]; o = cv[(cv.arm == arm) & (cv.set == 'official')]
        if not len(p):
            continue
        p = p.iloc[0]; folds = sum(p[f'delta_f{f}'] > 0 for f in range(4))
        okoff = (not len(o)) or o.iloc[0].delta >= -0.025
        fm = re.search(rf'validation {re.escape(arm)}\s+windows changed\s+(\d+)\s*/\s*(\d+) \| cells changed\s+(\d+)', t)
        okfp = bool(fm) and int(fm.group(1)) <= 15 and int(fm.group(3)) <= 40
        log(f'onset gate {arm}: delta {p.delta:+.4f} z {p.z:.2f} folds+ {folds} official ok {okoff} footprint ok {okfp}')
        if p.delta >= need_d and p.z >= need_z and folds >= 3 and okoff and okfp:
            return f'queue_submission_{arm}.csv', arm
    return None, 'no arm passed'


def t1fd_gate():
    if not (complete('tfb-t1v13a') and complete('tfb-t1v13b')):
        return False, 'v13 not complete'
    vals = {}
    for k in ('tfb-t1v13a', 'tfb-t1v13b', 'tfb-t1v12s4a', 'tfb-t1v12s4b', 'tfb-t1v12s5a', 'tfb-t1v12s5b'):
        t, _ = klog(k)
        for mm in re.finditer(r'\] (\S+) VALIDATION-month internal holdout S_state \{.*?\'FTA\': ([0-9.]+)\}', t):
            vals.setdefault(mm.group(1), {})[k[:-1]] = float(mm.group(2))
    d = [v['tfb-t1v13'] - 0.5 * (v['tfb-t1v12s4'] + v['tfb-t1v12s5']) for v in vals.values() if {'tfb-t1v13', 'tfb-t1v12s4', 'tfb-t1v12s5'} <= set(v)]
    if len(d) < 8:
        return False, f'only {len(d)} panels comparable'
    mean, pos = sum(d) / len(d), sum(x > 0 for x in d)
    log(f'T1 FD gate: mean holdout delta {mean:+.4f} over {len(d)} panels, {pos} positive')
    return (mean >= 0.0010 and pos >= 8), f'mean {mean:+.4f} pos {pos}'


if 'S4' not in done:
    src, why = onset_gate(); log('S4 onset gate ->', src, why)
    if src:
        write_variant('swap_onset19.py', 'swap_c19.py', [
            (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'"),
            (r"^SRC_KERNEL, SRC_FILE, SRC_FALLBACK, COND, OUT_NAME = .*$",
             f"SRC_KERNEL, SRC_FILE, SRC_FALLBACK, COND, OUT_NAME = 'tfb-t2onv19', '{src}', '{src}', 'queue_onset', 'COMBO19'")])
        push('tfb-c19', os.path.join(HERE, 'swap_c19.py'), [state['base_kernel'], 'tfb-t2onv19'])
        wait('tfb-c19'); t, files = klog('tfb-c19')
        if 'submission_COMBO19.csv' in files and 'Traceback' not in t:
            sc = submit('tfb-c19', 'submission_COMBO19.csv', f'EXP_COMBO_21: chain base + onset rows from v19 {why} (CatBoost bag + stage-2; CV-gated)',
                        '31_COMBO19_ON19', [state['base_kernel'], 'tfb-t2onv19'])
            apply(sc, 'S4', 'tfb-c19', 'submission_COMBO19.csv', 0.0)
        else:
            log('S4 onset swap failed'); state['done'].append(dict(slot='S4', score=None, keep=False)); save()
    else:
        ok, why2 = t1fd_gate(); log('S4 T1-FD gate ->', ok, why2)
        if ok:
            write_variant('swap_t1fd.py', 'swap_c19fd.py', [
                (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'")])
            push('tfb-c19fd', os.path.join(HERE, 'swap_c19fd.py'), [state['base_kernel'], 'tfb-t1v13a', 'tfb-t1v13b'])
            wait('tfb-c19fd'); t, files = klog('tfb-c19fd')
            if 'submission_T1FD_PROBE.csv' in files and 'Traceback' not in t:
                sc = submit('tfb-c19fd', 'submission_T1FD_PROBE.csv', 'EXP_COMBO_21b: chain base T1 rows = 0.5 bag + 0.5 FD-feature member (v13)',
                            '31_COMBO19_T1FD', [state['base_kernel'], 'tfb-t1v13a', 'tfb-t1v13b'])
                apply(sc, 'S4', 'tfb-c19fd', 'submission_T1FD_PROBE.csv', 0.0002)
            else:
                log('S4 T1-FD swap failed'); state['done'].append(dict(slot='S4', score=None, keep=False)); save()
        else:
            log('S4 skipped: no gated candidate'); state['done'].append(dict(slot='S4', score=None, keep=False)); save()

# ------------------------------------------------------------------ slot 5: TV smoothing on the chain base
if 'S5_TV' not in done:
    write_variant('swap_fdrecon.py', 'swap_c20.py', [
        (r"^DO_RECON, DO_TV = .*$", "DO_RECON, DO_TV = False, True"),
        (r"^BASE_KERNEL, BASE_FILE = .*$", f"BASE_KERNEL, BASE_FILE = '{state['base_kernel']}', '{state['base_file']}'")])
    push('tfb-c20', os.path.join(HERE, 'swap_c20.py'), [state['base_kernel']])
    wait('tfb-c20'); t, files = klog('tfb-c20')
    if 'DONE' in t and 'Traceback' not in t and 'submission_TV.csv' in files:
        sc = submit('tfb-c20', 'submission_TV.csv', 'EXP_COMBO_22: chain base + 1-D TV smoothing of Task 1 density inside target runs',
                    '32_COMBO20_TV', [state['base_kernel']])
        apply(sc, 'S5_TV', 'tfb-c20', 'submission_TV.csv', 0.0002)
    else:
        log('S5 kernel failed checks; skipped'); state['done'].append(dict(slot='S5_TV', score=None, keep=False)); save()

log('CHAIN DONE | final base', state['base_kernel'], state['base_file'], state['base_score'], '| best', state['best_score'])
