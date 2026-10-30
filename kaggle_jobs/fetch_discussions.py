"""Dump the competition discussion forum (all topics + all comments) to discussions/*.md via Kaggle's web API (anonymous: the forum is public, and the old Bearer token returns 401 since 2026-10-29)."""
import json, os, re, time, html, urllib.request
def call(svc, body):
    req = urllib.request.Request(f'https://www.kaggle.com/api/i/{svc}', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    return json.load(urllib.request.urlopen(req, timeout=60))
FORUM = 11991069
topics, seen, page = [], set(), 1
while True:
    r = call('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': FORUM, 'page': page})
    new = [t for t in r.get('topics', []) if t['id'] not in seen]
    if not new: break
    for t in new: seen.add(t['id']); topics.append(t)
    page += 1; time.sleep(0.5)
    if page > 30: break
print(len(topics), 'topics')
def clean(s):
    s = re.sub(r'<br\s*/?>', '\n', s or ''); s = re.sub(r'</p>|</li>', '\n', s); s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s).strip()
index = ['# Competition discussion forum (dump)', '', f'Source: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion — fetched {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}', '',
         '| id | title | author | comments | last activity |', '|---|---|---|---:|---|']
for t in topics:
    tid = t['id']
    try:
        d = call('discussions.DiscussionsService/GetForumTopicById', {'forumTopicId': tid, 'includeComments': True})
    except Exception as e:
        print('fail', tid, e); continue
    ft = d.get('forumTopic', d)
    title = ft.get('name') or t.get('title', '')
    lines = [f"# {title}", '', f"https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/{tid}", '']
    def walk(msgs, depth=0):
        for m in msgs or []:
            au = (m.get('author') or {}).get('displayName', '?')
            when = (m.get('postDate') or '')[:16]
            body = clean(m.get('content') or m.get('rawMarkdown') or '')
            lines.append(f"{'>' * depth}**{au}** ({when}):\n\n{body}\n")
            walk(m.get('replies'), depth + 1)
    fm = ft.get('firstMessage') or {}
    lines.append(f"**{(fm.get('author') or {}).get('displayName', t.get('authorUser', {}).get('displayName', '?'))}** ({(fm.get('postDate') or '')[:16]}):\n\n{clean(fm.get('content') or fm.get('rawMarkdown') or '')}\n\n---\n")
    walk(ft.get('comments') or d.get('comments'))
    open(f'discussions/{tid}.md', 'w').write('\n'.join(lines))
    index.append(f"| [{tid}]({tid}.md) | {title} | {t.get('authorUser', {}).get('displayName', '')} | {t.get('commentCount', 0)} | {(t.get('lastCommentPostDate') or t.get('postDate') or '')[:10]} |")
    time.sleep(0.4)
open('discussions/README.md', 'w').write('\n'.join(index) + '\n')
print('done')
