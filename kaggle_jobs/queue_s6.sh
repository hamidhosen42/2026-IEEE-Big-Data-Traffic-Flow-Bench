#!/bin/bash
cd "/Users/md.hamidhosen/Documents/Kaggle/2026 IEEE Big Data-Traffic Flow Bench/kaggle_jobs"
for p in a b; do
  until python3 kjob.py push tfb-t1v12s6$p t1_pipeline_v12s6$p.py hosen42/tfb-t2lab-full | grep -q "'hasError': False"; do sleep 300; done
  echo "pushed s6$p $(date -u)"
done
