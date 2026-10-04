#!/bin/bash
# wait for the v21 smoke kernel; if its log is clean, push the full run (retrying while Kaggle slots are busy)
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
sleep 60
until s=$(python3 kjob.py status tfb-t2v22-smoke 2>/dev/null) && ! echo "$s" | grep -qi "running\|queued"; do sleep 30; done
echo "smoke status: $s $(date -u)"
python3 - <<'PY'
import sys, json
sys.path.insert(0, '.'); import kjob
r = kjob.call('GET', '/kernels/output?userName=hosen42&kernelSlug=tfb-t2v22-smoke')
L = json.loads(r.get('log') or '[]'); t = ''.join(x['data'] for x in L)
files = [f['fileName'] for f in r.get('files', [])]
need = ['queue_submission_fs2.csv', 'queue_submission_fs4.csv', 'queue_submission_fs_np.csv']
ok = ('Traceback' not in t) and all(n in files for n in need)
for line in t.splitlines():
    if any(k in line for k in ('trained', 'CONTROL', 'SKIPPED', 'unmapped', 'dupes', 'FOOTPRINT', 'Traceback', 'Error')):
        print(line[:220])
print('FILES', files)
print('SMOKE_OK' if ok else 'SMOKE_FAILED')
open('queue_v22.flag', 'w').write('ok' if ok else 'fail')
PY
if [ "$(cat queue_v22.flag)" = "ok" ]; then
  until python3 kjob.py push tfb-t2v22 t2_model_v22.py | grep -q "'hasError': False"; do sleep 120; done
  echo "PUSHED tfb-t2v22 full $(date -u)"
else
  echo "NOT PUSHED: smoke failed"
fi
