# 2026-IEEE-Big-Data-Traffic-Flow-Bench

Our solution and experiment log for the Kaggle competition
[**2026 IEEE Big Data – Traffic Flow Bench**](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench).
Kaggle team: *Md. Hamid Hosen* (`hosen42`).

**Best public LB: 0.85309.** We started from 0.53880, and the public base notebook scores 0.80939.

> The public leaderboard scores the validation month (March). The final ranking uses a separate private month (April), generated independently and with more incidents.

---

## The competition in one paragraph

Ten synthetic freeway corridors at 5-minute resolution, with four tasks scored together:

```
S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME
```

| Task | What is asked | Metric |
|---|---|---|
| 1 State | Reconstruct masked speed and flow (20/30/50% masks) | `0.54·(1−RMSE_v/25) + 0.46·(1−RMSE_q,lane/600)` |
| 2 Queue | Binary queue (speed ≤ 0.6·v_free) per link for T+5..T+30, from 60 min of history | Space-time IoU per window |
| 3 Physics | Computed from the Task 1 file: fundamental diagram + vehicle conservation | `(S_FD + 2·S_LWR)/3` |
| 4 ODME | Path flows from link counts and a weak prior | Composite of OD, link, deviation and attraction terms |

Official docs, evaluators and baselines: https://github.com/jacky850/trafficflowbench-public

---

## Leaderboard history

| Submission | What changed | Public LB |
|---|---|---:|
| Earlier copy of a public pipeline | – | 0.53880 |
| Base notebook *[0.809 LB] Physics Pipeline* (soukeaizenz) | reference | 0.80939 |
| PROBE_T2 | base + **our Task 2 model** (only queue rows changed) | 0.84357 |
| SUB_A | + our Task 1 LightGBM (gap-aware) | 0.84499 |
| FINAL v2 | + profile-free Task 1 model, occupancy features, speed cap | 0.84855 |
| **OG_NOPRIOR** | **+ prior-free model on ongoing queue windows** | **0.85309** |
| ON30 | onset with a more prior-free blend (rejected) | 0.84917 |

**Measured LB decomposition** (probe submissions that zero out one window type): onset IoU 0.743, ongoing IoU 0.747, S_queue 0.745. This was before OG_NOPRIOR raised ongoing by about 0.03.

---

## Pipeline

### Task 2: queue forecasting (largest gain, +0.034 on LB)
- **Window selector replica.**
  - Rules replicated from the organizers:
    - History is [T−60, T), with at least 70% of cells eligible.
    - A window is `ongoing` if the history holds ≥2 queued observations with ≥2 on one link; otherwise it is `onset`.
    - The horizon must contain a queue.
    - Persistence IoU must be ≤ 0.9.
    - Selection is greedy in time order, with 360-minute global spacing.
  - The replica reproduces all 40 official train onset windows. In every one, the queue first appears at T+30.
  - We used it to draw about 5,000 training windows from the 9 train months.
- **Features.** Only data before T, plus the masked view at slot T:
  - speed/v_cut and flow/capacity at several lags;
  - neighbour links ±1..3 and upstream/downstream minima in link bands;
  - distance to the nearest queue now and 15 minutes ago;
  - corridor queue share;
  - static link data;
  - a historical queue prior by link × time slot.
- **Models.** Two pooled LightGBM classifiers:
  - onset: T+30 rows only;
  - ongoing: cells near queue activity.
  - Each is also trained in a **prior-free** version, without the location and time features.
- **Decoding.** For each window, pick the top-m cells that maximise the expected-IoU surrogate `Σp_S / (|S| + Σp_¬S)`.
- **Blend.**
  - Onset: 0.7·main + 0.3·prior-free. Onset needs location priors; a more prior-free onset lost 0.004 on LB.
  - Ongoing: **prior-free only**, which is robust to the month's non-recurrent (incident) queues (+0.0045 on LB).

### Task 1 (+ Task 3): state reconstruction
- LightGBM on the **residual to per-station temporal interpolation**. Features:
  - temporal and spatial interpolation (spatial on **total** flow, ordered by milepost);
  - neighbour values at t−1, t and t+1;
  - occupancy;
  - gap distances;
  - static link data.
- Trained on train views with **simulated 90-minute blackouts** after each Task 2 origin. The validation and private months contain these dark gaps, about 1.9% of targets.
- The final model has **no weekday × time-of-day profile features**. It beat the profile model on a validation-month internal hold-out in 10/10 panels (0.9307 vs 0.9280; base 0.9141).
- Speed is capped at 1.05·v_free, and flow at ≥ 20 vph (well above the S_FD empty-road guard).

### Task 4: ODME
- Prior-regularised NNLS (λ = 0.05) on **each split's own released counts and own prior**. S_link is ≈ 0.9999, and S_ODME is ≈ 0.994 on LB according to other teams' probes.

---

## Validation (frozen folds)
- **Task 1:** train split, targets from the official template, 3 contiguous day blocks. There is also an internal hold-out on the **validation month itself**: 15% of the released cells are hidden, which is leakage-free and tracks the LB direction.
- **Task 3:** local LWR proxy `1 − Σ|ΔN_pred − ΔN_obs| / Σ|ΔN_obs|`. The organizer's boundary flows are projected onto the released observations, so this proxy tracks the official term.
- **Task 2:** 4 contiguous day blocks, scored on builder-faithful pseudo windows and on the 80 official train windows. Local CV is about 0.88, against about 0.75 on LB. The official persistence baseline shows the same ratio (0.82), so the gap is systematic (proxy truth and month shift), not model-specific.

## What did not work (all measured)
| Idea | Result |
|---|---|
| Structural Kalman/RTS for Task 1 | worse than interpolation |
| Spatial interpolation on per-lane flow | −0.055 S_state |
| Density model / density pull / L1-density reconciliation for Task 3 | no T1+T3 gain |
| Task 2 context features (same day, previous days) | ≈ 0 |
| Queue-block (shockwave) features, equal window weights | ≈ 0 |
| More prior-free onset | −0.004 LB |
| Hypotheses: wrong v_cut source, unsensored links unobserved | rejected by data |

Full log with every experiment: [`experiments/TRACKER.md`](experiments/TRACKER.md).

---

## Repository layout
```
TFB_final_notebook.ipynb   single end-to-end notebook (Task 2 → Task 1 → Task 4 → assembly + gates)
kaggle_jobs/               everything that ran on Kaggle
  kjob.py                  push / wait / fetch private Kaggle kernels via the API
  submit.py                submit a kernel output file and wait for the score
  save_local.py            pull the exact submitted kernel sources into submissions/
  t1_pipeline*.py          Task 1 versions (v4 = final)
  t2_model*.py             Task 2 versions (v2 = final models; v5–v9 variants)
  t2lab.py                 window-selector replica + rule baselines
  assemble.py              Task 4 + merge on submission_key + hard gates
  swap_*.py, probe_*.py    LB probes (replace or zero one task's rows)
  base/                    the untouched base notebook (EXP_000)
submissions/               exact source of every submitted notebook (+ README with refs/scores)
experiments/TRACKER.md     experiment tracker (ID, change, CV, LB, decision)
analysis/                  local experiments on one corridor (D12_I405_S)
notebooks/                 public competition notebooks studied (renamed with their scores)
top2_notebooks.zip         notebooks of the two best submissions
```

## How to run
1. **Kaggle (recommended):** create a notebook with the competition data attached and run `TFB_final_notebook.ipynb` top to bottom. It takes about 2.5 hours on CPU. Set `SMOKE = True` for a quick end-to-end check.
2. **From this machine:**
   - Push a kernel with `python3 kaggle_jobs/kjob.py push <slug> <file.py|.ipynb> [upstream kernels...]`, then `wait` and `output`.
   - Submit a kernel output with `KAGGLE_API_TOKEN=... python kaggle_jobs/submit.py <slug> <version> "<description>" [file]`.

**Not in this repo:** the competition data (Kaggle rules forbid redistribution), large kernel outputs and credentials. Keep this repository **private** until the competition ends. Near-identical predictions across teams can make both ineligible.

## Deadlines
- Registration e-mail to `trafficflowbench@gmail.com` by **Oct 25, 2026**.
- Final submission deadline **Nov 7, 2026 06:55 UTC**. Select 2 final submissions.
