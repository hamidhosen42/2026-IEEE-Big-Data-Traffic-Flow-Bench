#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
for k in tfb-t1v12s2a tfb-t1v12s2b tfb-t1v12s3a tfb-t1v12s3b; do python3 kjob.py wait $k; done
for k in tfb-t1v12s2a tfb-t1v12s2b tfb-t1v12s3a tfb-t1v12s3b; do python3 kjob.py status $k; done
until python3 kjob.py push tfb-combo15 swap_combo15.py hosen42/tfb-combo13 hosen42/tfb-t1v11a hosen42/tfb-t1v11b hosen42/tfb-t1v12a hosen42/tfb-t1v12b hosen42/tfb-t1v12s2a hosen42/tfb-t1v12s2b hosen42/tfb-t1v12s3a hosen42/tfb-t1v12s3b | grep -q "'hasError': False"; do sleep 120; done
sleep 60; python3 kjob.py wait tfb-combo15
ok=$(python3 -c "
import sys,json; sys.path.insert(0,'.'); import kjob
r=kjob.call('GET','/kernels/output?userName=hosen42&kernelSlug=tfb-combo15')
t=''.join(x['data'] for x in json.loads(r.get('log') or '[]') if x['stream_name']=='stdout'); print(t)
print('OKFLAG' if 'COMBO15 OK' in t else 'FAILFLAG')")
echo "$ok"; cd ..
if echo "$ok" | grep -q OKFLAG; then
  .venv/bin/python -W ignore kaggle_jobs/submit.py tfb-combo15 1 "EXP_COMBO_17: COMBO13 + Task 1 bag of 4 FTA members (seeds 0-3)" submission_COMBO15.csv 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py 27_COMBO15 tfb-combo15 tfb-combo13 tfb-t1v11a tfb-t1v11b tfb-t1v12a tfb-t1v12b tfb-t1v12s2a tfb-t1v12s2b tfb-t1v12s3a tfb-t1v12s3b 2>&1 | tail -1
fi
echo AUTO_SUBMIT_OCT2_S5_DONE
