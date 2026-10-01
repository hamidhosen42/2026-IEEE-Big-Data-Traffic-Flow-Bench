#!/bin/bash
# push the pseudo-March gate smoke when a Kaggle CPU slot frees up; if it passes, push the full run the same way
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
pushwait() { # slug file
  until python3 kjob.py push "$1" "$2" hosen42/tfb-t1v8-full | grep -q "'hasError': False"; do sleep 300; done
  echo "pushed $1 $(date -u)"; sleep 60
}
pushwait tfb-pmarch-smoke t2_pseudo_march_smoke.py
python3 kjob.py wait tfb-pmarch-smoke; python3 kjob.py output tfb-pmarch-smoke out/pmarch_smoke >/dev/null
if python3 -c "import json,sys;L=json.load(open('out/pmarch_smoke/log.txt'));t=''.join(x['data'] for x in L);print(t[-3000:]);sys.exit(0 if 'pseudo-March gate done' in t else 1)"; then
  pushwait tfb-pmarch t2_pseudo_march.py
else echo SMOKE FAILED; fi
echo QUEUE_PMARCH_DONE
