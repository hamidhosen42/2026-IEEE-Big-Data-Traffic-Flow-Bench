"""Rebuild all_submitted_notebooks/: one notebook per Kaggle submission (date order) + README index. Run after each day's chain.
Local folders submissions/NN_* are matched to submissions in date order (folder 13 never existed; #9/#10 share 04_PROBE_COND);
submissions made straight from a Kaggle notebook (#1, #3, #5) are pulled from that notebook; errored file uploads have no notebook."""
import glob, json, os, re, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import kjob  # noqa: E402
OUT = f'{ROOT}/all_submitted_notebooks'; os.makedirs(OUT, exist_ok=True)
S = []
for p in range(1, 10):
    r = kjob.call('GET', f'/competitions/submissions/list/2026-ieee-big-data-traffic-flow-bench?page={p}&pageSize=100')
    L = r if isinstance(r, list) else r.get('submissions', [])
    new = [x for x in L if x.get('ref') not in {y.get('ref') for y in S}]
    if not new:
        break
    S += new
S.sort(key=lambda x: x.get('date', ''))
folders = sorted(d for d in os.listdir(f'{ROOT}/submissions') if re.match(r'\d\d_', d))
early = {6: '01_PROBE_T2', 7: '02_SUB_A', 8: '03_FINAL_v2', 9: '04_PROBE_COND', 10: '04_PROBE_COND', 11: '05_OG_NOPRIOR', 12: '06_ON30',
         13: '07_T2_CATBOOST_ONSET', 14: '10_T1B_LOWRANK', 15: '08_V9_WEEKDAY_PRIOR', 16: '09_OG_PERS30', 17: '11_T1B_T2CAT', 18: '15_COMBO3',
         19: '12_COMBO2', 20: '16_COMBO4', 21: '17_COMBO5', 22: '14_COMBO2D'}
rest = [f for f in folders if int(f[:2]) >= 18]
fold = dict(early); fold.update({23 + i: f for i, f in enumerate(rest)})
for f in os.listdir(OUT):
    os.remove(f'{OUT}/{f}')
rows = []
for i, x in enumerate(S, 1):
    sc = x.get('publicScore') or x.get('status'); desc = (x.get('description') or x.get('errorDescription') or '').replace('|', '/')
    tag = f'LB{sc}' if x.get('status') == 'complete' else 'ERROR'
    src, kern, label = None, '', ''
    if i in fold:
        d = f'{ROOT}/submissions/{fold[i]}'
        src = [p for p in glob.glob(f'{d}/*.ipynb') if not os.path.basename(p).startswith('upstream__')][0]
        kern = os.path.basename(src)[:-6]; label = fold[i].split('_', 1)[1]
    elif '/code/hosen42/' in (x.get('url') or ''):
        kern = x['url'].split('/code/hosen42/')[1].split('?')[0]
        b = kjob.call('GET', f'/kernels/pull?userName=hosen42&kernelSlug={kern}'); b = b.get('blob', b)
        src = f'{OUT}/.tmp.ipynb'
        if b.get('kernelType') == 'notebook':
            open(src, 'w').write(b['source'])
        else:
            json.dump({'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': b['source']}],
                       'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}, open(src, 'w'))
        label = re.sub(r'[^A-Za-z0-9]+', '_', kern)[:40]; time.sleep(1)
    name = f'{i:02d}_{label}__{tag}.ipynb' if src else None
    if src:
        shutil.copy(src, f'{OUT}/{name}')
    rows.append(f"| {i} | {x['date'][:10]} | {name or '— (no notebook: file upload that errored)'} | `{kern}` | {sc} | {desc[:140]} |")
if os.path.exists(f'{OUT}/.tmp.ipynb'):
    os.remove(f'{OUT}/.tmp.ipynb')
best = max((float(x['publicScore']), i) for i, x in enumerate(S, 1) if x.get('status') == 'complete')
hdr = ['# All submitted notebooks', '',
       f'One notebook per Kaggle submission, in submission order ({len(S)} submissions; best public LB #{best[1]}, {best[0]:.5f}).',
       'Each file is the exact source of the kernel whose output was submitted. Swap kernels read upstream kernel outputs; their upstream '
       'sources are in `submissions/<folder>/upstream__*`. #1, #3 and #5 were submitted straight from Kaggle notebooks (current version).',
       'Rebuild: `.venv/bin/python kaggle_jobs/build_all_notebooks.py`.', '',
       '| # | Date (UTC) | Notebook | Kaggle kernel | Public LB | Description |', '|---:|---|---|---|---:|---|']
open(f'{OUT}/README.md', 'w').write('\n'.join(hdr + rows) + '\n')
print(len([f for f in os.listdir(OUT) if f.endswith('.ipynb')]), 'notebooks |', len(S), 'submissions')
