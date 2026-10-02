#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
until python3 kjob.py push tfb-combo13 swap_combo13.py hosen42/tfb-combo12 hosen42/tfb-t2v15 | grep -q "'hasError': False"; do sleep 180; done
echo "pushed $(date -u)"; sleep 60; python3 kjob.py wait tfb-combo13
ok=$(python3 -c "
import sys,json; sys.path.insert(0,'.'); import kjob
r=kjob.call('GET','/kernels/output?userName=hosen42&kernelSlug=tfb-combo13')
t=''.join(x['data'] for x in json.loads(r.get('log') or '[]') if x['stream_name']=='stdout'); print(t)
print('OKFLAG' if 'COMBO13 OK' in t else 'FAILFLAG')")
echo "$ok"; cd ..
if echo "$ok" | grep -q OKFLAG; then
  .venv/bin/python -W ignore kaggle_jobs/submit.py tfb-combo13 1 "EXP_COMBO_15: COMBO12 + ongoing rows from v15 np511r (prior-free 511 + ramp; probe vs id511r)" submission_COMBO13.csv 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py 25_COMBO13 tfb-combo13 tfb-combo12 tfb-t2v15 2>&1 | tail -1
fi
echo AUTO_SUBMIT_OCT2_S3_DONE
