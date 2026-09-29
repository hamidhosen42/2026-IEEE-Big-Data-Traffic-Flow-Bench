#!/bin/bash
# waits for the UTC quota reset, then submits the queued candidates and saves each notebook locally
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-09-29 00:02:00" +%s)
while [ $(date -u +%s) -lt $target ]; do sleep 60; done
submit() { # slug version file desc folder upstreams...
  .venv/bin/python -W ignore kaggle_jobs/submit.py "$1" "$2" "$4" "$3" 2>&1 | tail -2
  python3 kaggle_jobs/save_local.py "$5" "$1" "${@:6}" 2>&1 | tail -1
}
submit tfb-swap-v10 1 submission_main.csv "EXP_T2_p: onset CatBoost + ongoing prior-free LGB (else = 0.85309 pipeline)" 07_T2_CATBOOST_ONSET tfb-final tfb-t2v10
submit tfb-swap-v9 1 submission_main.csv "EXP_T2_m: v9 (weekday-class priors + broad ongoing keep), og_noprior recipe" 08_V9_WEEKDAY_PRIOR tfb-final tfb-t2v9-full
submit tfb-swap-v8 1 submission_og_pers30.csv "EXP_T2_n: og_noprior + 30% last-valid persistence on ongoing" 09_OG_PERS30 tfb-final tfb-t2v8
echo AUTO_SUBMIT_DONE
