#!/bin/bash
# priority queue for 2026-10-03 inputs (pushes when a Kaggle CPU session frees)
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
pushwait() { until python3 kjob.py push "$@" | grep -q "'hasError': False"; do sleep 240; done; echo "pushed $1 $(date -u)"; sleep 60; }
logok() { python3 -c "
import sys,json; sys.path.insert(0,'.'); import kjob
r=kjob.call('GET','/kernels/output?userName=hosen42&kernelSlug=$1')
t=''.join(x['data'] for x in json.loads(r.get('log') or '[]'))
print(t[-2500:]); sys.exit(0 if ('$2' in t and 'Traceback' not in t) else 1)"; }
# fdrecon smoke pushed manually 2026-10-02 06:33 UTC
# chain A: onset v19
( pushwait tfb-t2onv19-smoke t2_onset_v19_smoke.py hosen42/tfb-t2v13
  python3 kjob.py wait tfb-t2onv19-smoke
  if logok tfb-t2onv19-smoke CONTROL; then pushwait tfb-t2onv19 t2_onset_v19.py hosen42/tfb-t2v13; else echo "ONSET SMOKE FAILED"; fi ) > queue_oct3_onset.log 2>&1 &
# chain B: T1 v13 FD members
( pushwait tfb-t1v13-smoke t1_pipeline_v13_smoke.py hosen42/tfb-t2lab-full
  python3 kjob.py wait tfb-t1v13-smoke
  if logok tfb-t1v13-smoke "state rows"; then pushwait tfb-t1v13a t1_pipeline_v13a.py hosen42/tfb-t2lab-full; pushwait tfb-t1v13b t1_pipeline_v13b.py hosen42/tfb-t2lab-full; else echo "T1 V13 SMOKE FAILED"; fi ) > queue_oct3_t1.log 2>&1 &
wait
pushwait tfb-t1v12s7b t1_pipeline_v12s7b.py hosen42/tfb-t2lab-full
echo QUEUE_OCT3_DONE
