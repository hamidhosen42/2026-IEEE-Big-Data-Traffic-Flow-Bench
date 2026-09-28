"""save_local.py <folder> <submitted_slug> [upstream slugs...] : pull exact kernel sources into submissions/<folder>/ and refresh README."""
import json, os, sys, time, urllib.request
T = open(os.path.expanduser('~/.kaggle/access_token')).read().strip(); H = {'Authorization': 'Bearer ' + T}
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=120))
folder, slug, ups = sys.argv[1], sys.argv[2], sys.argv[3:]
d = f'submissions/{folder}'; os.makedirs(d, exist_ok=True)
for k in [slug] + ups:
    b = get(f'https://www.kaggle.com/api/v1/kernels/pull?userName=hosen42&kernelSlug={k}'); b = b.get('blob', b)
    name = f'{d}/{k}' if k == slug else f'{d}/upstream__{k}'
    if b.get('kernelType') == 'notebook':
        open(name + '.ipynb', 'w').write(b['source'])
    else:
        open(name + '.py', 'w').write(b['source'])
        nb = {'cells': [{'cell_type': 'markdown', 'id': 'c0', 'metadata': {}, 'source': f'Exact source of Kaggle kernel `{k}` (pulled from Kaggle).'},
                        {'cell_type': 'code', 'id': 'c1', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': b['source']}],
              'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
        json.dump(nb, open(name + '.ipynb', 'w'), indent=1)
    time.sleep(1)
subs = get('https://www.kaggle.com/api/v1/competitions/submissions/list/2026-ieee-big-data-traffic-flow-bench')
lines = ['# Submitted notebooks (kept locally)', '', '| Output file | Submission ref | Public LB | Description |', '|---|---|---:|---|']
for s in subs:
    lines.append(f"| {s.get('fileName','')} | {s.get('ref')} | {s.get('publicScore') or s.get('status')} | {s.get('description') or ''} |")
open('submissions/README.md', 'w').write('\n'.join(lines) + '\n')
print('saved', d)
