"""submit.py <kernel_slug> <version> <description> ; submit a kernel's submission.csv and wait for the score."""
import sys, time, json, os, urllib.request
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest
C = '2026-ieee-big-data-traffic-flow-bench'
with KaggleClient() as k:
    r = ApiCreateCodeSubmissionRequest()
    r.competition_name = C; r.kernel_owner = 'hosen42'; r.kernel_slug = sys.argv[1]; r.kernel_version = int(sys.argv[2])
    r.file_name = sys.argv[4] if len(sys.argv) > 4 else 'submission.csv'; r.submission_description = sys.argv[3]
    ref = k.competitions.competition_api_client.create_code_submission(r).ref
print('submitted ref', ref, flush=True)
T = os.environ['KAGGLE_API_TOKEN']
for _ in range(60):
    time.sleep(30)
    req = urllib.request.Request(f'https://www.kaggle.com/api/v1/competitions/submissions/list/{C}', headers={'Authorization': 'Bearer ' + T})
    s = [x for x in json.load(urllib.request.urlopen(req)) if x.get('ref') == ref]
    if s and s[0].get('status') in ('complete', 'error'):
        print(s[0].get('status'), s[0].get('publicScore'), s[0].get('errorDescription')); break
