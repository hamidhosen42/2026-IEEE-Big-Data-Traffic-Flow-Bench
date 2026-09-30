"""Merge research/partials/*_resources.csv into research/resource_index.csv.

Dedup keys (any match merges rows): DOI, arXiv id, normalised canonical URL, normalised title, GitHub repo.
The merged row keeps the highest status (CODE_INSPECTED > DEEP_READ > SCREENED > DISCOVERED) and lists every
partial that found it in `found_by`.
"""
import glob, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RANK = {'DISCOVERED': 0, 'SCREENED': 1, 'DEEP_READ': 2, 'CODE_INSPECTED': 3}


def norm_url(u):
    u = str(u or '').strip().lower()
    u = re.sub(r'^https?://(www\.)?', '', u).split('#')[0].rstrip('/')
    u = re.sub(r'arxiv\.org/(abs|pdf|html)/([0-9.]+)(v\d+)?(\.pdf)?', r'arxiv:\2', u)
    return u


def keys(r):
    out = []
    for f in ('doi_or_arxiv', 'canonical_url', 'code_url'):
        v = str(r.get(f) or '').strip().lower()
        if not v or v == 'nan':
            continue
        m = re.search(r'10\.\d{4,9}/[^\s,;]+', v)
        if m:
            out.append('doi:' + m.group(0).rstrip('.'))
        m = re.search(r'(\d{4}\.\d{4,5})', v) if 'arxiv' in v or re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', v) else None
        if m:
            out.append('arxiv:' + m.group(1))
        if f == 'code_url':
            # repo key only for rows that are about the repo itself (canonical URL empty or the repo root),
            # so separate files / commits / papers that merely link a repo stay separate
            m = re.search(r'github\.com/([^/\s]+/[^/\s#?]+)', v)
            cu = norm_url(r.get('canonical_url'))
            if m and (not cu or cu == 'nan' or re.fullmatch(r'github\.com/[^/]+/[^/]+', cu)):
                out.append('gh:' + m.group(1).removesuffix('.git'))
        elif f == 'canonical_url':
            out.append('url:' + norm_url(v))
            m = re.fullmatch(r'github\.com/([^/]+/[^/]+)', norm_url(v))
            if m:
                out.append('gh:' + m.group(1).removesuffix('.git'))
    t = re.sub(r'[^a-z0-9]+', ' ', str(r.get('title') or '').lower()).strip()
    if len(t) > 25:
        out.append('title:' + t)
    return out


frames = []
for f in sorted(glob.glob(f'{HERE}/partials/*_resources.csv')):
    d = pd.read_csv(f, dtype=str).fillna('')
    d['found_by'] = os.path.basename(f)[0]
    frames.append(d)
A = pd.concat(frames, ignore_index=True)
A['status'] = A.status.str.strip().str.upper().where(A.status.str.strip().str.upper().isin(RANK), 'DISCOVERED')
raw = len(A)

# union-find over shared keys
parent = list(range(raw))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


seen = {}
for i, r in A.iterrows():
    for k in keys(r):
        if k in seen:
            parent[find(i)] = find(seen[k])
        else:
            seen[k] = i
A['grp'] = [find(i) for i in range(raw)]
A['rank'] = A.status.map(RANK)
rows = []
for g, d in A.groupby('grp', sort=False):
    d = d.sort_values('rank', ascending=False)
    best = d.iloc[0].copy()
    for c in A.columns:                               # fill blanks from the other copies
        if not str(best[c]).strip():
            vals = [v for v in d[c] if str(v).strip()]
            if vals:
                best[c] = vals[0]
    best['found_by'] = ''.join(sorted(set(d.found_by)))
    best['duplicates_merged'] = len(d) - 1
    best['source_ids'] = ';'.join(d.resource_id)
    rows.append(best)
M = pd.DataFrame(rows).drop(columns=['grp', 'rank'])
M['resource_id'] = [f'R{i + 1:04d}' for i in range(len(M))]
M.to_csv(f'{HERE}/resource_index.csv', index=False)
print('raw rows', raw, '-> unique', len(M))
print(M.status.value_counts().reindex(list(RANK)).fillna(0).astype(int).to_dict())
print('by partial (unique rows first found by):', M.found_by.str[0].value_counts().to_dict())
