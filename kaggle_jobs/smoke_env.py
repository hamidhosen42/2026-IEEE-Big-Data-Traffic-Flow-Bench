import os, glob, json, time, pandas as pd
t0 = time.time()
roots = glob.glob('/kaggle/input/*')
print('inputs:', roots)
REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)
print('release roots:', REL)
REL = REL[0]
print(sorted(os.listdir(REL)))
for p in ['task2/D7_I405_N/train/window_index.csv', 'task2/D7_I405_N/validation/window_index.csv']:
    print('\n##', p); print(pd.read_csv(os.path.join(REL, p)).to_string())
h = pd.read_parquet(os.path.join(REL, 'task2/D7_I405_N/train/window_history.parquet'))
print('\n## window_history', h.shape); print(h.dtypes); print(h.head(5).to_string())
print('per-window rows', h.groupby('window_id').size().describe().to_dict())
q = pd.read_csv(os.path.join(REL, 'task2/D7_I405_N/train/sample_submission_queue.csv'))
print('\n## sample_submission_queue', q.shape); print(q.head(3).to_string())
for f in ['synthetic_link_counts.csv', 'synthetic_weak_prior.csv', 'sample_submission_path_flow.csv']:
    d = pd.read_csv(os.path.join(REL, 'task4/D7_I405_N/validation', f)); print('\n##', f, d.shape); print(d.head(3).to_string())
print('cpu', os.cpu_count(), 'elapsed', round(time.time() - t0, 1))
os.system('free -g; nproc; df -h /kaggle/working | tail -1; pip list 2>/dev/null | grep -iE "lightgbm|xgboost|catboost|scikit|torch |pandas|numpy"')
