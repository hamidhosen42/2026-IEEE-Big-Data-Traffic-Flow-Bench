# Traffic Flow Bench: state, plan and verification brief

This brief is for a reviewer (human or AI) who has not seen the project. The goal is to verify the plan below and suggest experiments that can still help before the deadline.

- Competition: Kaggle "2026 IEEE Big Data – Traffic Flow Bench".
- Team: Hack2Publish (Kaggle user hosen42).
- Final submission deadline: Nov 7, 2026, 06:55 UTC.
- Daily limit: 5 submissions. We pick 2 final submissions.
- Final ranking uses the private leaderboard, which covers the April windows. The public leaderboard covers March.

## 1. Where we stand (Nov 3)

| Kernel | What it adds | Public LB |
|---|---|---|
| tfb-k7 (`submissions/74_K7`) | k5 + a flow-only low-rank tensor completion (LRTC) member in Task 1 (weight 0.35, ordinary rows), then FD reconciliation and TV smoothing | **0.88259** (best) |
| tfb-k8 (`submissions/75_K8`) | A hedge built on k7. April onset uses the production CatBoost with the train prior; April ongoing uses veto10; the March rows are unchanged | 0.88259 |
| k5 / k6 | Earlier versions | 0.88225 |

- The top public score is 0.92942, so we are about 0.047 behind.
- The current final picks are A = tfb-k7 and B = tfb-k8.

### Score formula

S_total = 0.35·S_state (Task 1) + 0.30·S_queue (Task 2) + 0.05·S_FD + 0.10·S_LWR + 0.20·S_ODME (Task 4).

The official scorers are included in `organizer_scoring/`.

- S_queue is a per-window IoU over the horizon cells T+5..T+30. The averages are taken in order: window → (panel, condition) → panel → family. Onset and ongoing windows each count for half.
- S_LWR is effectively the L1 accuracy of N = (q/v)·L at the hidden cells, because the boundary flux is projected onto the observations.

### Production pipeline (summary; the full code is in `consolidation/` and `submissions/`)

- **T1 (state):**
  - an FD bag and an FTA bag;
  - CatBoost 0.35 and XGBoost 0.35;
  - a blackout specialist at weight 0.75;
  - an LRTC flow member with ranks (8, 40, 10) at weight 0.35;
  - FD reconciliation and 1-D TV smoothing.
- **T2 (queue):**
  - March onset: sp4 CatBoost.
  - April onset: pv bag.
  - March ongoing: fs4occ.
  - April ongoing: pvogfv in pick A, veto10 in pick B.
  - The onset model predicts step 6 only, because 39 of the 40 official train onset windows have their first queue at step 6.
- **T4 (ODME):** NNLS.

## 2. Hard rules (the reviewer must check every idea against these)

From organizer posts 748789 and 749601, copied in `rules/`:

- Everything used for a window must be stamped at or before that window's T. This covers inputs, model fitting, and model or threshold selection.
- Task 2 is re-run by the organizer with all post-T rows removed, and the predictions must not change.
- April decisions may use train and March evidence only. Using the April leaderboard or April labels to choose anything is not allowed.
- Award track: one entry script, a README, the environment, and a PDF report, due Nov 10. The organizer re-runs the code and it must reproduce within 1 %. The award score is 85 % private LB and 15 % report.

## 3. Already tried and failed (do not re-propose without a new angle)

Seven experiment rounds have been run. The details are in `experiments/TRACKER.md`. Gates are pre-registered: the T1 J-proxy needs dJ ≥ +0.0003, a CI lower bound > 0, and at least 7 of 8 folds positive.

| Area | Idea | Result |
|---|---|---|
| T1 | kNN-day member | FAIL |
| T1 | LRTC rank ensemble | FAIL |
| T1 | Further members of the same kind | Lever exhausted |
| T2 | April ongoing with occupancy features (PVOCC) | pseudo-March +0.0031 [−0.0011, +0.0079], 4/8 folds; DO NOT ADOPT |
| T2 | Shock-gate redo | −0.06 on train, 0/8 |
| T2 | Online refits on same-split earlier days | Useless: the official windows sit in days 1–5 of each month |
| T2 | Onset boundary windows | 0/40 windows at the boundary; nothing to gain |
| T3 | LWR projection | Dropped earlier |

## 4. Ready but not run: k9

- **What it is:** Task 1 step M uses LRTC ranks (24, 40, 10) instead of (8, 40, 10), and blkVQ is applied at 0.15 on blackout rows.
- **Expected gain:** about +0.0002 on the public LB.
- **Chain:**
  1. Push `kaggle_jobs/swap_lrtc_k9.py` as tfb-k9 (source hosen42/tfb-k5d). Check that the M and B shares are above 0.97, and that the log shows "M OK", "B OK", "EXCLUDED panels: []" and "DONE".
  2. Push `kaggle_jobs/swap_k10.py` as tfb-k10 (sources: tfb-k9, tfb-t2osp, tfb-t2ogveto). Check for "TOTAL changed rows: validation 0" and "K10 OK".
  3. Run `kaggle_jobs/t4_validate_k9.py`. Check for "T4 VALIDATION PASSED".
  4. Submit both CSVs.

## 5. Plan for the remaining days

| Day | Action |
|---|---|
| Nov 3 (today, 5 subs) | Reviewer audits the k9 scripts (section 6). If they are clean, run the k9 chain and submit k9 and k10. |
| Nov 4–5 | Run one or two cheap experiments from section 7, offline first. Submit only those that pass their offline gate. |
| Nov 6 | Freeze. Choose the final 2 picks: A is the best validated public score, B is the hedge with a different April T2. Save the kernel sources to `submissions/`. |
| Nov 4–10 | If the team registered for awards: consolidated entry script, README, environment file and PDF report, using train and March evidence only. |

The rule for final picks is to never choose on public LB noise below +0.0005. Prefer the version whose gain was validated offline.

## 6. What the reviewer should verify

1. Diff `kaggle_jobs/swap_lrtc_k9.py` against `swap_lrtc_k7.py`, and diff `swap_k10.py` against `swap_k8.py`. Look for:
   - wrong row alignment by submission_id or task;
   - March and April rows being mixed;
   - any use of data after T in Task 2.
2. Confirm that k10 changes only the intended rows.
3. Check that the Task 2 code paths (`consolidation/tfb/stages`) read only rows at or before T for each window. This is the rerun test with post-T rows removed.
4. Sanity-check the final-pick logic in section 5 against the rules in section 2.

## 7. Candidate experiments not yet exhausted (ranked by expected value per hour)

1. **Task 4 / ODME (20 % weight).**
   - Only NNLS has been used so far.
   - Possible changes: regularised NNLS (ridge or entropy prior toward the historical OD), or a non-negative least-squares fit with a link-count weighting.
   - Validate on train with `organizer_scoring/task4/score_task4.py`. The validator is `kaggle_jobs/t4_validate*.py`.
2. **S_FD and S_LWR (15 % together).**
   - These depend on the T1 speed and flow consistency.
   - Check whether the FD reconciliation strength or the TV weight is tuned on train or March, and try a small grid offline.
3. **T1 blend weights.**
   - LRTC weight 0.35 and blackout specialist 0.75 could be re-fit on train and March folds as a constrained stacking. This is a gain only if the CI lower bound is > 0.
4. **High risk, private-only:** train April onset with extra windows built from the official March windows, labelled with T1-imputed queues. This cannot be checked on the public LB, so treat it as a hedge for pick B at most.

## Contents of this pack

| Folder | Contents |
|---|---|
| `submissions/` | Exact kernel sources of k7 and k8, and their upstream |
| `kaggle_jobs/` | The k9 chain, the k7/k8 swap scripts, and the T4 validators |
| `consolidation/` | The full, reproducible pipeline (award entry candidate) |
| `analysis/` | Lab README, the lab helper, and the Task 1 member lab code |
| `organizer_scoring/` | Official scorers |
| `rules/` | Organizer posts on time stamping |
| `experiments/TRACKER.md` | Every experiment with its numbers |

No competition data and no credentials are included.
