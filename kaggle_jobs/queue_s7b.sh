#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
until python3 kjob.py push tfb-t1v12s7b t1_pipeline_v12s7b.py hosen42/tfb-t2lab-full | grep -q "'hasError': False"; do sleep 300; done
echo "pushed s7b $(date -u)"
