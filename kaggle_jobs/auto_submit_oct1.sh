#!/bin/bash
# 2026-10-01: submit COMBO6 at the UTC quota reset; then build COMBO7 when tfb-t2v13 finishes and submit it
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-10-01 00:02:00" +%s)
while [ $(date -u +%s) -lt $target ]; do sleep 60; done
submit() { # slug version file desc folder upstreams...
  .venv/bin/python -W ignore kaggle_jobs/submit.py "$1" "$2" "$4" "$3" 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py "$5" "$1" "${@:6}" 2>&1 | tail -1
}
submit tfb-combo6 1 submission_COMBO6.csv "EXP_COMBO_8: COMBO5 + T1 F (511 leaves, val-month holdout 0.9386 vs 0.9374)" 18_COMBO6 tfb-swap-v9 tfb-t1v8-full tfb-t2v12 tfb-t2v11
cd kaggle_jobs
python3 kjob.py wait tfb-t2v13
python3 kjob.py push tfb-combo7 swap_combo7.py hosen42/tfb-swap-v9 hosen42/tfb-t1v8-full hosen42/tfb-t2v12 hosen42/tfb-t2v11 hosen42/tfb-t2v13
sleep 60; python3 kjob.py wait tfb-combo7; python3 kjob.py output tfb-combo7 out/combo7 >/dev/null
python3 -c "import json;L=json.load(open('out/combo7/log.txt'));print(''.join(x['data'] for x in L if x['stream_name']=='stdout'))"
cd ..
if grep -q "COMBO7 OK" <(python3 -c "import json;L=json.load(open('kaggle_jobs/out/combo7/log.txt'));print(''.join(x['data'] for x in L if x['stream_name']=='stdout'))"); then
  submit tfb-combo7 1 submission_COMBO7.csv "EXP_COMBO_9: COMBO6 + ongoing LGB 511 leaves/lr0.03/1800 trees (t2v13)" 19_COMBO7 tfb-swap-v9 tfb-t1v8-full tfb-t2v13 tfb-t2v12 tfb-t2v11
else echo COMBO7 FAILED CHECKS; fi
echo AUTO_SUBMIT_OCT1_DONE
