"""Push a script to Kaggle as a private kernel with the competition data attached, poll it, fetch outputs."""
import json, os, sys, time, urllib.request, urllib.parse
T = open(os.path.expanduser('~/.kaggle/access_token')).read().strip()
H = {'Authorization': 'Bearer ' + T, 'Content-Type': 'application/json'}
USER = 'hosen42'
COMP = '2026-ieee-big-data-traffic-flow-bench'
API = 'https://www.kaggle.com/api/v1'


def call(method, path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:2000]}


def push(slug, script_path, title=None, gpu=False, kernel_sources=()):
    body = {'slug': f'{USER}/{slug}', 'newTitle': title or slug, 'text': open(script_path).read(),
            'language': 'python', 'kernelType': 'notebook' if script_path.endswith('.ipynb') else 'script', 'isPrivate': True, 'enableGpu': gpu,
            'enableTpu': False, 'enableInternet': False, 'datasetDataSources': [],
            'competitionDataSources': [COMP], 'kernelDataSources': list(kernel_sources), 'modelDataSources': [], 'categoryIds': []}
    return call('POST', '/kernels/push', body)


def status(slug):
    return call('GET', f'/kernels/status?userName={USER}&kernelSlug={slug}')


def output(slug, outdir):
    r = call('GET', f'/kernels/output?userName={USER}&kernelSlug={slug}')
    os.makedirs(outdir, exist_ok=True)
    if 'log' in r and r['log']:
        open(os.path.join(outdir, 'log.txt'), 'w').write(r['log'])
    for f in r.get('files', []):
        name, url = f.get('fileName'), f.get('url')
        if name and url:
            dest = os.path.join(outdir, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=600) as src:
                        data = src.read()
                    open(dest, 'wb').write(data)
                    break
                except Exception as e:
                    print('retry', name, attempt, e)
                    time.sleep(5)
    return [f.get('fileName') for f in r.get('files', [])], r.get('http_error')


if __name__ == '__main__':
    cmd, slug = sys.argv[1], sys.argv[2]
    if cmd == 'push':
        print(push(slug, sys.argv[3], kernel_sources=sys.argv[4:]))
    elif cmd == 'status':
        print(status(slug))
    elif cmd == 'wait':
        t0 = time.time()
        while True:
            s = status(slug); st = s.get('status')
            if st not in ('queued', 'running', 'QUEUED', 'RUNNING', 'KernelWorkerStatus.QUEUED', 'KernelWorkerStatus.RUNNING'):
                print(f'{time.time() - t0:.0f}s', s); break
            time.sleep(30)
    elif cmd == 'output':
        print(output(slug, sys.argv[3]))
