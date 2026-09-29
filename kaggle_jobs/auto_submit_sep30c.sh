#!/bin/bash
# waits for the UTC quota reset, then submits the queued candidates and saves each notebook locally
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-09-30 00:35:00" +%s)
while [ $(date -u +%s) -lt $target ]; do sleep 60; done
submit() { # slug version file desc folder upstreams...
  .venv/bin/python -W ignore kaggle_jobs/submit.py "$1" "$2" "$4" "$3" 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py "$5" "$1" "${@:6}" 2>&1 | tail -1
}
submit tfb-combo5 1 submission_COMBO5.csv "EXP_COMBO_7: v9 + CatBoost onset + ongoing LGB 255 leaves + T1 v7 E" 17_COMBO5 tfb-swap-v9 tfb-t1v7-full tfb-t2v12 tfb-t2v11
echo AUTO_SUBMIT_SEP30C_DONE
