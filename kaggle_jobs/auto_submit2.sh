#!/bin/bash
# waits for the UTC quota reset, then submits the queued candidates and saves each notebook locally
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-09-29 00:20:00" +%s)
while [ $(date -u +%s) -lt $target ]; do sleep 60; done
submit() { # slug version file desc folder upstreams...
  .venv/bin/python -W ignore kaggle_jobs/submit.py "$1" "$2" "$4" "$3" 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py "$5" "$1" "${@:6}" 2>&1 | tail -1
}
submit tfb-swap-t1b 1 submission_T1B.csv "EXP_T1_n: 0.85309 + Task 1 low-rank SVD features (T1 v5 B)" 10_T1B_LOWRANK tfb-swap-v5 tfb-t1v5-full
submit tfb-swap-t1b 1 submission_T1B_T2cat.csv "EXP_COMBO_1: 0.85309 + T1 low-rank + T2 onset CatBoost" 11_T1B_T2CAT tfb-swap-v5 tfb-t1v5-full tfb-t2v10
echo AUTO_SUBMIT2_DONE
