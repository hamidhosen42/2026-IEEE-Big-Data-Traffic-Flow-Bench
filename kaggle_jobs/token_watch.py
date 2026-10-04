"""Watch ~/.kaggle/access_token. When its content changes (token rotated), check that the new token authenticates as hosen42
(no secret is ever printed), then restart the submission chain and the v21 helper so their in-memory state uses the new token.
The chain is resumable (state file + idempotent submissions by description), so a restart repeats at most one in-flight kernel.
Log: token_watch.log (token values are never written)."""
import hashlib, json, os, subprocess, time, urllib.request, urllib.error
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
LOG = open(os.path.join(HERE, 'token_watch.log'), 'a', buffering=1)
URL = 'https://www.kaggle.com/api/v1/competitions/submissions/list/2026-ieee-big-data-traffic-flow-bench'


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a) + '\n')


def token():
    try:
        return open(TOKEN_F).read().strip().splitlines()[0].strip()
    except Exception:
        return None


def fingerprint(t):
    return hashlib.sha256(t.encode()).hexdigest()[:8] if t else None


def check(t):
    try:
        L = json.load(urllib.request.urlopen(urllib.request.Request(URL, headers={'Authorization': 'Bearer ' + t}), timeout=90))
        return 'ok', (L[0].get('submittedBy') if L else None)
    except urllib.error.HTTPError as e:
        return f'http {e.code}', None
    except Exception as e:
        return f'error {type(e).__name__}', None


def restart():
    for name in ('tfbchain5',):
        subprocess.run(['screen', '-S', name, '-X', 'quit'], capture_output=True)
    subprocess.run(['pkill', '-f', 'auto_chain_oct5.py'], capture_output=True)
    time.sleep(3)
    q = '"/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"'
    subprocess.run(['screen', '-dmS', 'tfbchain5', 'bash', '-c', f'cd {q} && exec caffeinate -i -s /usr/bin/python3 auto_chain_oct5.py >> auto_chain_oct5.out 2>&1'])
    state_f = os.path.join(HERE, 'auto_chain_oct5_state.json')
    log('restarted chain oct5 | chain state exists:', os.path.exists(state_f))


cur = fingerprint(token())
log('watching token file, current fingerprint', cur)
while True:
    time.sleep(20)
    t = token()
    fp = fingerprint(t)
    if fp is None or fp == cur:
        continue
    time.sleep(5)
    t = token(); fp = fingerprint(t)
    res, who = check(t)
    log('token file changed: new fingerprint', fp, '| auth', res, '| account', who)
    if res == 'ok' and who == 'hosen42':
        restart()
        cur = fp
    else:
        log('new token not usable for hosen42 -> chain NOT restarted (will retry on the next change)')
        cur = fp
