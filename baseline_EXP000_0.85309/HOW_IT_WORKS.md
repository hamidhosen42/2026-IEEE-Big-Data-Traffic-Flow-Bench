# Submission 56643037 — Public LB 0.85309 (current best)

This submission was built from three Kaggle notebooks, run in this order:
1. `upstream__tfb-final.ipynb` (kernel hosen42/tfb-final, version 2). Produces the full base `submission.csv`:
   - Task 1: LightGBM, profile-free, gap-aware, with occupancy features.
   - Task 2: LightGBM blend70.
   - Task 4: NNLS on each split's own counts.
2. `upstream__tfb-t2v5.ipynb` (kernel hosen42/tfb-t2v5). Retrains the Task 2 models and writes `queue_submission_og_noprior.csv`.
   On ongoing windows it uses the prior-free model only. On onset windows it keeps the 70/30 blend.
3. `tfb-swap-v5.ipynb` (kernel hosen42/tfb-swap-v5). Takes (1), replaces only the Task 2 rows with (2), and writes `submission_og_noprior.csv`.
   That file was submitted.
