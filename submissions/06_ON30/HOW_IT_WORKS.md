# Submission 56645587 — Public LB 0.84917

Same pipeline as 05_OG_NOPRIOR (0.85309); only the onset windows differ.
1. `upstream__tfb-final.ipynb` (hosen42/tfb-final v2). Produces the base submission for Tasks 1, 2 and 4.
2. `upstream__tfb-t2v7.ipynb` (hosen42/tfb-t2v7). Retrains Task 2 and writes variants. The submitted variant `on30` uses:
   - ongoing windows: the prior-free model;
   - onset windows: a 30/70 main/prior-free blend.
3. `tfb-swap-v7.ipynb` (hosen42/tfb-swap-v7). Swaps the Task 2 rows into (1) and writes `submission_on30.csv`, which was submitted.

Lesson: onset needs the location/time priors (this variant lost 0.0039 against 0.85309).
