#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench"
export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/access_token)
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-10-02 00:02:00" +%s)
while [ $(date -u +%s) -lt $target ]; do sleep 60; done
.venv/bin/python -W ignore kaggle_jobs/submit.py tfb-combo11 1 "EXP_COMBO_13: COMBO10 + Task 1 rows from T1 v11 FTA (transductive + ASM; holdout +0.0005 vs FT)" submission_COMBO11.csv 2>&1 | tail -2
python3 kaggle_jobs/save_local.py 23_COMBO11 tfb-combo11 tfb-combo10 tfb-t1v11a tfb-t1v11b 2>&1 | tail -1
echo AUTO_SUBMIT_OCT2_S1_DONE
