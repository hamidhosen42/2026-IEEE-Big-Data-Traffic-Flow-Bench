#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
pushwait() { until python3 kjob.py push "$1" "$2" | grep -q "'hasError': False"; do sleep 300; done; echo "pushed $1 $(date -u)"; sleep 60; }
pushwait tfb-oofr-smoke t2_oofR_smoke.py
python3 kjob.py wait tfb-oofr-smoke; python3 kjob.py output tfb-oofr-smoke out/oofR_smoke >/dev/null
if python3 -c "import json,sys;L=json.load(open('out/oofR_smoke/log.txt'));t=''.join(x['data'] for x in L);print(t[-2500:]);sys.exit(0 if 'OOF-R cache done' in t else 1)"; then
  pushwait tfb-oofr t2_oofR.py
  python3 kjob.py wait tfb-oofr; python3 kjob.py output tfb-oofr out/oofR >/dev/null
else echo SMOKE FAILED; fi
echo QUEUE_OOFR_DONE
