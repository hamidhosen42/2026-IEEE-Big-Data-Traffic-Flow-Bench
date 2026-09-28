"""Download Kaggle competition files by path filter using the access token."""
import os, sys, json, re, urllib.request, urllib.parse, concurrent.futures as cf
C = '2026-ieee-big-data-traffic-flow-bench'
T = open(os.path.expanduser('~/.kaggle/access_token')).read().strip()
H = {'Authorization': 'Bearer ' + T}

def list_files():
    files, tok = [], ''
    while True:
        u = f'https://www.kaggle.com/api/v1/competitions/data/list/{C}?pageSize=200' + ('&pageToken=' + urllib.parse.quote(tok) if tok else '')
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H)))
        files += r['files']; tok = r.get('nextPageToken') or r.get('nextPageTokenNullable')
        if not tok or not r['files']: return files

def fetch(f, root='data', tries=5):
    for i in range(tries):
        try: return _fetch(f, root)
        except Exception as e: print('retry', i, f['name'], e, flush=True)
    print('FAILED', f['name'], flush=True); return 0

def _fetch(f, root):
    out = os.path.join(root, f['name'])
    if os.path.exists(out) and os.path.getsize(out) == f['totalBytes']: return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    u = f'https://www.kaggle.com/api/v1/competitions/data/download/{C}/' + urllib.parse.quote(f['name'], safe='')
    with urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60) as r, open(out + '.part', 'wb') as w:
        while chunk := r.read(1 << 20): w.write(chunk)
    os.replace(out + '.part', out); return f['totalBytes']

if __name__ == '__main__':
    pat = re.compile(sys.argv[1])
    todo = sorted([f for f in list_files() if pat.search(f['name'])], key=lambda f: f['totalBytes'])
    print(len(todo), 'files', sum(f['totalBytes'] for f in todo) / 1e6, 'MB', flush=True)
    with cf.ThreadPoolExecutor(12) as ex:
        n = sum(ex.map(fetch, todo))
    print('downloaded', n / 1e6, 'MB')
