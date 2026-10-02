#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
until grep -q "S1_DONE" kaggle_jobs/auto_submit_oct2.log 2>/dev/null; do sleep 60; done
cd kaggle_jobs; python3 kjob.py wait tfb-combo12
ok=$(python3 -c "
import sys,json; sys.path.insert(0,'.'); import kjob
r=kjob.call('GET','/kernels/output?userName=hosen42&kernelSlug=tfb-combo12')
t=''.join(x['data'] for x in json.loads(r.get('log') or '[]') if x['stream_name']=='stdout'); print(t)
print('OKFLAG' if 'COMBO12 OK' in t else 'FAILFLAG')")
echo "$ok"
cd ..
if echo "$ok" | grep -q OKFLAG; then
  .venv/bin/python -W ignore kaggle_jobs/submit.py tfb-combo12 1 "EXP_COMBO_14: COMBO11 + Task 1 bag (mean of v11 FTA seed0 and v12 FTA seed1)" submission_COMBO12.csv 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py 24_COMBO12 tfb-combo12 tfb-combo11 tfb-t1v11a tfb-t1v11b tfb-t1v12a tfb-t1v12b 2>&1 | tail -1
fi
echo AUTO_SUBMIT_OCT2_S2_DONE
