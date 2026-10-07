# 2026 IEEE Big Data – TrafficFlowBench Deep Analysis

**Research date:** 2026-10-06  
**Competition:** 2026 IEEE Big Data-Traffic Flow Bench  
**Primary objective:** Maximize hidden Private Leaderboard / final-award robustness rather than overfit the visible validation month.

> Evidence labels used throughout: **OFFICIAL FACT**, **ORGANIZER/TOOLKIT FACT**, **VERIFIED PUBLIC RESULT**, **PARTICIPANT CLAIM**, **EXTERNAL RESEARCH EVIDENCE**, **HYPOTHESIS**, **RECOMMENDATION**.

---

## Executive Summary

TrafficFlowBench is unusually important because it is **not a single forecasting problem**. It is a four-task freeway-system benchmark:

1. **Task 1 — State reconstruction (35%)**  
   Recover masked **speed** and **flow** at five-minute resolution.

2. **Task 2 — Queue forecasting (30%)**  
   Given 60 minutes of history, predict queued links over the next 30 minutes.

3. **Task 3 — Physics consistency (15%)**  
   No independent output. The evaluator scores the Task-1 speed/flow reconstruction for fundamental-diagram plausibility and LWR conservation.

4. **Task 4 — OD/path-flow estimation (20%)**  
   Infer non-negative path flows from path-link incidence, counts, and a weak prior.

Composite:
`S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME`

The most important structural fact is that the visible **validation** leaderboard month and hidden **private** month were generated independently, with different incidents. Therefore aggressive tuning to Public LB is structurally unsafe.

A second critical fact is that the organizers **rebuilt the released data on 11 September 2026** after discovering two routes that leaked part of the Task-2 answer. Scores from before the rebuild are not comparable with current eligible scores.

The best verified public notebook located in this research pass is:

- **[0.809 LB TOP 50] Open-Sourcing Physics Pipeline — Public LB 0.80939**
- Other verified notebook scores found: **0.71846**, **0.63952**, **0.40218**.

A public GitHub repository claims an overall score around **0.90–0.93+** and cites a leaderboard top score of **0.92406**, but those values were **not independently verified** in this pass, so they must be treated as **PARTICIPANT CLAIM / UNVERIFIED**.

### Main strategic conclusion

The strongest path is not “use the fanciest deep model.” It is a **hybrid traffic-science + ML pipeline**:

- Task 1: spatio-temporal reconstruction with detector-neighbor structure + physically constrained post-processing.
- Task 3: optimize Task-1 outputs jointly for accuracy and conservation.
- Task 2: train directly on reconstructed training-window labels with shockwave/trend/topology features and optimize a **space-time IoU-aware thresholding strategy**.
- Task 4: use a strong regularized inverse solver, not an unnecessarily complex neural model.
- Final model selection: use train-derived pseudo-validation months/incidents, seed/fold stability, and cross-corridor robustness—not small Public-LB gains.

---

# 1. Competition Breakdown

## 1.1 Objective

**OFFICIAL FACT:** The benchmark asks whether a method can reconstruct, forecast, and explain a freeway traffic system across ten directional corridors.

The data are synthetic but generated from a calibrated traffic-flow model whose parameters came from real freeway detector data. No real detector record appears directly in the released competition data.

## 1.2 Real-world task

The four tasks form a practical Traffic Management Center workflow:

- fill broken/missing detector states;
- anticipate queue propagation;
- ensure the estimated state obeys traffic-flow physics;
- infer network demand/path flows from observed counts.

## 1.3 ML task types

- Task 1: multivariate spatio-temporal imputation/regression.
- Task 2: structured spatio-temporal binary prediction.
- Task 3: physics-constrained scoring / model-consistency problem.
- Task 4: constrained inverse problem / regularized optimization.

## 1.4 Data

**OFFICIAL FACT**

- Resolution: 5 minutes.
- Corridors: 10 directional panels.
- Families: D7_I10, D7_I210, D7_I405, D12_I5, D12_I405; each direction separately.
- Total history: 11 months.
- Splits:
  - train: 9 months / 273 days / labels available.
  - validation: 31 days / leaderboard.
  - private: 30 days / final ranking.
- Approximate unpacked release size reported by the public toolkit: ~13 GB.

Typical release root:
- `config/`
- `corridors/`
- `task1/`
- `task2/`
- `task4/`
- `README.md`
- `sample_submission.csv`
- `submission_key.csv`

## 1.5 Split behavior

**OFFICIAL FACT:** validation and private are independent generated months.

A public toolkit description reports:
- validation: 5 incidents;
- private: 7 incidents;
- incidents occur at different times, locations, and severities.

**Practical implication:** Public LB is an imperfect transfer check, not a safe tuning objective.

## 1.6 Submission completeness

**OFFICIAL FACT:** A task must contain its entire required row set. If any required row is missing, the **whole task scores 0**.

This is a severe operational risk. Submission validation should be a mandatory automated stage.

---

# 2. Exact Evaluation Metric

## 2.1 Composite

`S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME`

Each component is in [0,1].

Aggregation:
1. score each directional panel;
2. average two directions within the same corridor family;
3. average five corridor families equally.

This prevents a corridor from receiving double weight merely because it has two directions.

---

## 2.2 Task 1 — State reconstruction

For mask regime `r ∈ {R1,R2,R3}`:

`S_speed(r) = max(0, 1 - RMSE_speed(r)/25)`

`S_flow(r) = max(0, 1 - RMSE_flow_per_lane(r)/600)`

`S_state(r) = 0.54*S_speed(r) + 0.46*S_flow(r)`

`S_state = mean_r S_state(r)`

Masking rates:
- R1 = 20%
- R2 = 30%
- R3 = 50%

Only speed and flow are directly scored.
Occupancy is not a Task-1 submitted/scored channel.
Density is derived by the evaluator.

### Metric consequences

- Speed and flow need separate optimization.
- Flow must be judged per lane.
- A method that is excellent at speed but weak at flow is not sufficient.
- A global model should not let large-lane corridors dominate.
- Evaluate per regime and per corridor—not only overall RMSE.

---

## 2.3 Task 2 — Queue forecasting

Input:
- history through forecast origin T: 60 minutes.
- horizon: T+5 … T+30 minutes, six 5-minute steps.

Target:
- binary queued/not queued for each link/time point.

Queue threshold:
`speed <= 0.60 * free_speed`

Metric per window:
`IoU = |Q_pred ∩ Q_true| / |Q_pred ∪ Q_true|`

Conditions:
- `queue_onset`: no queue visible in history, queue appears in horizon.
- `queue_ongoing`: queue exists at origin.
- equal weight by condition.
- five windows per condition per corridor/split.

D12_I405_N/S are excluded from Task 2 only.

### Metric consequences

IoU punishes:
- overly large queue masks;
- conservative masks that miss propagation;
- thresholding tuned only for per-cell accuracy.

The model should explicitly optimize **space-time shape**, not pointwise accuracy.

---

## 2.4 Task 3 — Physics consistency

No separate Task-3 upload.

The evaluator uses Task-1 submitted values:

`k = q/v`

`N = k*L`

Conservation:
`N(t+dt)-N(t) = dt*(q_in + r_on - q_out - r_off)`

`S_LWR = max(0, 1 - sum|residual|/sum|rhs|)`

`S_physics = (1/3)S_FD + (2/3)S_LWR`

The toolkit notes that the FD term is mainly a validity check and the LWR term is more discriminative.

### Major strategic implication

A reconstruction method must be chosen using a **joint Task-1 + Task-3 objective**.
A visually smooth interpolation can lower state RMSE while violating mass conservation.

---

## 2.5 Task 4 — OD/path-flow estimation

Let:
- `f*` reference path flow
- `fhat` predicted path flow
- `A` path-link incidence
- `c` observed counts
- `b` weak prior

Components:

`S_od = max(0, 1 - sum|fhat-f*|/sum(f*))`

`S_link = max(0, 1 - sum|A fhat - c|/sum(c))`

`S_dev = exp(-|Dhat/Dstar - 1|)`

`S_attr = max(0, 1 - 0.5*L1(a_hat,a_star))`

`S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr`

Invalid IDs, mismatched zones, duplicate/missing paths, negative flows, or non-finite values can invalidate the corridor.

### Strategic consequence

Pure count fitting is not enough. An extreme inverse solution can improve `S_link` but damage `S_od`, `S_dev`, and `S_attr`.
The correct formulation is regularized, non-negative, and prior-aware.

---

# 3. Data Leakage and September 11 Rebuild

## 3.1 Confirmed event

**OFFICIAL FACT:** On 11 September 2026 the released data were rebuilt.

Reason: the organizers found **two distinct routes** through which part of the Task-2 answer could be read from the release rather than forecast.

The validation/private months were regenerated from new seeds and the data, solution file, and metric package were replaced together.

### Important consequences

- Pre-rebuild scores are not comparable to current scores.
- Only submissions after reopening on rebuilt data are award-eligible.
- Any public notebook using cached pre-rebuild files must be treated as contaminated until re-run.
- When ranking public notebooks, version date is essential.
- Any external output dataset must be checked for rebuild compatibility.

---

# 4. Public Leaderboard / Private Leaderboard Risk

## 4.1 Why the shuffle risk is high

Unlike a standard random holdout:
- public = one full generated month;
- private = another independently generated month;
- incidents differ;
- queue onset/propagation is rare and event-sensitive;
- Task 2 has high weight (30%);
- Task 3 magnifies systematic physical errors from Task 1.

Therefore a 0.002 Public-LB gain can easily be less meaningful than a smaller but stable improvement over multiple train-derived month holdouts.

## 4.2 Correct model-selection hierarchy

1. Train pseudo-month holdout performance.
2. Cross-corridor stability.
3. Incident-type robustness.
4. Seed/fold stability.
5. Physics score.
6. OOF ensemble diversity.
7. Public LB as a sanity check.

Not:
1. Public LB.
2. Public LB.
3. Public LB.

---

# 5. Recommended Validation Strategy

A classical random KFold is inappropriate because it leaks temporal/local traffic structure.

## Canonical validation design

### Level A — Rolling / blocked month holdouts

Construct pseudo-validation months from the nine training months.

Example:
- Fold A: months 1–7 train, month 8 validate.
- Fold B: months 1–8 train, month 9 validate.
- Optional additional earlier blocked windows.

### Level B — Incident-aware Task-2 holdout

For queue prediction:
- reconstruct the official Task-2 window-generation logic from train;
- split windows by time and event;
- keep whole events/windows together;
- no overlapping history/horizon between train and validation.

### Level C — Corridor generalization stress test

Leave-one-corridor-family-out or hold out selected directional panels to assess spatial transfer.

### Level D — Rebuild-safe final proxy

Freeze one untouched train month as a “private proxy” and never use it for threshold/blend tuning.

### Required logging

Track:
- overall composite proxy;
- Task 1 per regime / speed / flow;
- Task 2 by onset vs ongoing;
- Task 3 FD vs LWR;
- Task 4 all four sub-components;
- per-corridor scores;
- seed std.

---

# 6. Public Notebook Research

## Verified notebook scores found in this research pass

| Rank among found | Notebook | Author | Public LB | Runtime | Notes |
|---:|---|---|---:|---:|---|
| 1 | [0.809 LB TOP 50] Open-Sourcing Physics Pipeline | Sōsuke Aizen +2 | **0.80939** | 45m50s | Uses an additional dataset named `TFB Task2 LGBM + MU Cache`; submission output visible; strong candidate for lineage analysis |
| 2 | Traffic Flow Bench Pipeline | Lâm Huy | **0.71846** | 28m56s | Popular public pipeline; version 32/50 was indexed |
| 3 | Hydrodynamic Traffic Reconstruction & Queue ODME | Avik Das | **0.63952** | 30m58s, T4x2 | GitHub-linked; hydrodynamic framing |
| 4 | 2026 IEEE Big Data-Traffic Flow Bench/3 | Chaitanya Jamble | **0.40218** | 9m44s | Best historical version shown: 0.40651 V3 |

These are **VERIFIED PUBLIC RESULTS** because Kaggle indexed the score on the notebook page.

### Public-score interpretation

Do not infer that 0.80939 is currently the global best public method. It is merely the highest **independently verified public notebook score found in this pass**.

---

# 7. Participant Repository Claim — Use Carefully

A public GitHub repository titled:

**“2026 IEEE Big Data Cup: Traffic Flow Bench (Rank-1 ML Solution)”**

claims:

- Task 1 ~0.9402
- Task 2 ~0.7602–0.85+
- Task 3 ~0.90–0.95+
- Task 4 ~0.9998
- combined ~0.90–0.93+
- leaderboard top score 0.92406

It describes:
- Task 1: LightGBM + 2D spatio-temporal grid/interpolation.
- Task 2: shockwave-aware LightGBM.
- Task 3: triangular FD projection / LWR constraints.
- Task 4: bounded regularized L-BFGS-B.

**Status: PARTICIPANT CLAIM / UNVERIFIED.**

The architectural ideas are plausible and worth reproducing, but the reported scores must not be entered into a verified-ranking table until independently reproduced or matched to an eligible Kaggle submission.

---

# 8. Notebook Lineage / Family Hypothesis

## Family A — Official baseline descendants
Likely characteristics:
- official interpolation/state reconstruction;
- persistence queue baseline;
- official regularized ODME;
- modified post-processing.

## Family B — ML + physics pipelines
Likely characteristics:
- tree model or learned spatio-temporal state model;
- explicit FD/LWR refinement;
- engineered queue propagation model;
- regularized inverse solver.

The 0.80939 “Physics Pipeline” and the “Rank-1 ML Solution” claim appear conceptually close to this family, but code-level lineage must be checked before declaring copying/reuse.

## Family C — Hydrodynamic / traffic-theory-first
Likely characteristics:
- CTM/LWR/shockwave reasoning;
- physically motivated queue propagation;
- inverse traffic-flow estimation.

This family may generalize better to private incidents if calibrated conservatively.

---

# 9. Leakage Audit

## Confirmed historical leakage
Task-2 answer leakage existed pre-rebuild.

## Plausible current leakage routes to test

1. **Window overlap leakage**  
   Training a queue classifier on windows sharing time steps with validation.

2. **Future-feature leakage**  
   Any rolling/interpolation feature accidentally using values after forecast origin.

3. **Task-1 reconstruction leakage**  
   Using unmasked train truth differently from what will be available in evaluation.

4. **Cached dataset leakage**  
   Output datasets produced from the pre-11-September release.

5. **Template/ID leakage**  
   Learning deterministic behavior from ordering or generated IDs.

6. **Event leakage**  
   Same incident split across training/validation windows.

7. **Public-feedback leakage**  
   Tuning queue thresholds/blend weights repeatedly against Kaggle.

8. **ODME hidden-target leakage**  
   Any use of organizer-only flows or post-processed files containing hidden reference information.

---

# 10. Distribution Shift

The private month differs by:
- incident count;
- incident location;
- incident timing;
- incident severity;
- stochastic generation seed.

Recommended diagnostics:

- month classifier (train-month vs pseudo-private month);
- corridor/domain classifier;
- PSI/KS per speed/flow/occupancy/ramp feature;
- queue-event severity distribution;
- free-flow/congestion occupancy mix;
- missingness and `pct_observed` distributions;
- regime R1/R2/R3 performance separately;
- per-corridor link-length / lane / capacity normalization.

---

# 11. Task-by-Task Modeling Strategy

## Task 1 — State reconstruction

### Baseline ladder

**E0:** official baseline exact reproduction.

**E1:** two-axis interpolation:
- temporal interpolation per detector;
- spatial interpolation along ordered station/link positions;
- weekday/time-of-day historical profile;
- quality-aware weighting.

**E2:** LightGBM/CatBoost residual model over interpolation baseline.

Features:
- lag/lead values from allowed context;
- nearest upstream/downstream detector values;
- spatial distance;
- lanes;
- free-flow speed;
- capacity;
- jam density;
- occupancy;
- ramp on/off flows;
- time-of-day;
- weekday;
- mask regime;
- neighboring speed gradients;
- flow conservation residual features.

**E3:** graph/spatio-temporal neural model:
- DCRNN/STGCN/Graph WaveNet/MTGNN-like architecture.

Only pursue if E2 is saturated and compute is justified.

### Strong recommendation

Model residuals over a traffic-science baseline rather than predict absolute state from scratch.

---

## Task 2 — Queue forecasting

This is probably the largest remaining headroom because:
- 30% weight;
- queue onset is structurally hard;
- naive persistence cannot anticipate new queues.

### Features

Per link:
- speed / free-speed ratio;
- flow / capacity;
- occupancy;
- Δspeed 5/10/15/30 min;
- Δflow;
- Δoccupancy;
- upstream/downstream gradients;
- congestion-front position;
- distance to current queue front;
- ramp inflow surge;
- shockwave-speed proxy;
- time-of-day/weekday;
- bottleneck/link static features.

### Models

1. LightGBM / CatBoost classifier.
2. Sequence model per corridor.
3. Graph temporal model.
4. Hybrid shockwave extrapolation + classifier.

### IoU-aware inference

Do not simply threshold at 0.5.
Optimize:
- per-condition threshold;
- spatial dilation/erosion;
- temporal persistence;
- connected-component pruning;
- queue-front growth constraints.

Tune exclusively on train-generated pseudo-windows.

---

## Task 3 — Physics

Joint refinement of Task-1 predictions.

A safe post-processing objective could be:

`L = λ1 * deviation_from_ML_prediction
   + λ2 * FD_penalty
   + λ3 * conservation_penalty
   + λ4 * boundary/ramp consistency`

Constraints:
- non-negative flow;
- physically valid speed;
- capacity-aware flow;
- avoid artificially empty-road solutions.

The official evaluator already guards against degenerate low-flow submissions, so gaming physics is a poor strategy.

---

## Task 4 — ODME

Start from official non-negative least squares with prior regularization.

Then test:
- weighted NNLS;
- L-BFGS-B;
- sparse regularization;
- OD attraction regularization;
- corridor-specific λ;
- count reliability weighting.

Because the score includes prior deviation and attraction, select λ on train by the exact official Task-4 score—not only link residual.

---

# 12. External Research: High-Value Verified Resources

A complete 500-resource database was **not fabricated**. This research pass includes only sources independently located/verified. The field clearly contains far more than 500 related papers, but the strict requirement is “unique and relevant,” so no synthetic padding is used.

High-value verified items include:

1. **Official TrafficFlowBench public toolkit** — exact evaluators, baselines, schemas.
2. **TrafficFlowBench scoring specification** — normative metric implementation.
3. **Daganzo (1994), Cell Transmission Model** — foundational queue/shockwave dynamics.
4. **Daganzo (1995), CTM Part II: Network Traffic** — network CTM.
5. **DCRNN (Li et al., 2017)** — directed graph traffic forecasting.
6. **Variable-Length CTM (Canudas-de-Wit & Ferrara, 2018)** — congestion front as state.
7. **Location-Specific CTM** — heterogeneous freeway fundamental diagrams.
8. **Improved CTM freeway case study** — merge/diverge realism.
9. **Recent bidirectional Graph WaveNet traffic speed/volume work (2026)**.
10. **Context-aware STGCN with residual correction (2026)**.
11. **Origin-Destination Matrix Estimation Using Traffic Counts and Machine Learning (IEEE Access, 2026)**.
12. **Recent PatchTST-Graph traffic forecasting framework**.
13. **PDR-STGCN (2026)** — dynamic relational graph + periodicity.

See `external_resources.csv` for the verified resource table created in this pass.

---

# 13. Top 10 Experiments to Run

## E0 — Official baseline exact reproduction
**Priority:** 1  
**Why:** establishes scoring and submission correctness.  
**Keep criterion:** exact local score reproduction.

## E1 — Canonical blocked-month validation
**Priority:** 2  
Freeze folds before model tuning.

## E2 — Quality-aware spatial+temporal interpolation
**Priority:** 3  
Expected upside: strong Task-1 improvement with very low variance.

## E3 — LightGBM residual correction for Task 1
**Priority:** 4  
Train separate speed and flow residual models.

## E4 — Joint physics projection / conservation regularization
**Priority:** 5  
Evaluate `(0.35*S_state + 0.15*S_physics)` jointly.

## E5 — Train-derived queue-window dataset
**Priority:** 6  
Reproduce official onset/ongoing mechanics from train.

## E6 — Shockwave-aware LightGBM queue model
**Priority:** 7  
Optimize IoU on pseudo-windows.

## E7 — IoU-aware post-processing
**Priority:** 8  
Threshold + spatial/temporal shape tuning using OOF only.

## E8 — Regularized L-BFGS-B / weighted NNLS ODME
**Priority:** 9  
Optimize exact Task-4 metric on train.

## E9 — Alternative graph model
**Priority:** 10  
Only if decorrelated from tree/interpolation models.

## E10 — Small diverse final ensemble
Blend:
- interpolation/residual tree model;
- graph model;
- physics-refined variant;
using OOF predictions and frozen private proxy.

---

# 14. Approaches to Avoid

- random KFold across five-minute records;
- thresholds tuned directly to Public LB;
- pre-rebuild cached data/output;
- blind copying of a “rank-1” repository claim;
- optimizing Task 1 without monitoring Task 3;
- over-smoothed flow reconstructions;
- queue classifiers optimized for accuracy instead of IoU;
- predicting no queue broadly;
- huge ensembles of highly correlated interpolation variants;
- Task-4 solutions that only minimize `||Af-c||`;
- any pseudo-labeling based primarily on public leaderboard behavior.

---

# 15. Final Ensemble Recommendation

### Task 1 / 3
Two-model blend:
- Model A: robust interpolation + LightGBM residual correction.
- Model B: graph temporal model or alternative residual learner.
- Apply a conservative physics-aware projection.
- Blend weights determined from blocked OOF composite Task1+Task3.

### Task 2
- shockwave/gradient-feature LightGBM;
- one sequence/graph model if OOF correlation is sufficiently low;
- probability blend;
- IoU-aware structured post-processing.

### Task 4
- one strong prior-regularized constrained optimizer is likely preferable to an ensemble.
- optionally blend two λ settings if exact train score improves consistently.

---

# 16. Private-Leaderboard Strategy

Use three gates for every proposed experiment:

### Gate 1 — CV evidence
Must improve mean blocked-month score or materially improve worst-corridor behavior.

### Gate 2 — stability
Improvement must exceed seed/fold noise.

### Gate 3 — transfer plausibility
The change must be justified by traffic dynamics or robust ML behavior.

Public LB is only a fourth, weaker signal.

Decision framework:

- CV ↑, Public ↑ → KEEP candidate.
- CV ↑, Public slightly ↓ → often KEEP for private robustness.
- CV ↓, Public ↑ → likely public overfit; REJECT unless strong explanation.
- one fold only ↑ → NEEDS_MORE_EVIDENCE.

---

# 17. Execution Plan

## Stage 1 — Research
Deliverables:
- official metric implementation notes;
- version/rebuild checklist;
- public notebook lineage table.

## Stage 2 — Data audit
Deliverables:
- schema;
- mask statistics;
- corridor/link/station topology;
- missingness/quality report;
- incident/window extraction.

## Stage 3 — Validation
Deliverables:
- frozen folds;
- pseudo-private month;
- event-disjoint Task-2 windows.

## Stage 4 — Baseline
Deliverables:
- exact official baseline reproduction;
- submission validator;
- local scorer wrapper.

## Stage 5 — Controlled experiments
Deliverables:
- experiments.csv;
- one-variable-at-a-time ablations.

## Stage 6 — Strong single models
Deliverables:
- best Task-1/3 joint model;
- best Task-2 model;
- best Task-4 solver.

## Stage 7 — Diversity
Deliverables:
- OOF prediction correlation matrix;
- complementary candidate list.

## Stage 8 — Ensembling
Deliverables:
- OOF blend weights;
- IoU post-processing parameters.

## Stage 9 — Robustness
Deliverables:
- seed tests;
- month tests;
- corridor stress tests;
- incident severity tests.

## Stage 10 — Final submissions
Deliverables:
- conservative final;
- secondary diverse final;
- reproducible GitHub/zip + README + one entry script;
- technical report.

---

# 18. Awards / Timeline

Official competition information found in this pass:

- Kaggle final submission deadline: **November 6, 2026** (competition page governs exact cutoff).
- Award registration email deadline: **October 25, 2026**.
- Final code + PDF report: **November 10, 2026, 23:59 AoE**.
- Reproducibility review: November 11–17.
- Winners/selected teams: November 18.
- Camera-ready: November 20.
- IEEE BigData 2026: December 14–17, Phoenix, Arizona.

Awards:
- Gold $1,500
- Silver $1,000
- Bronze $500
- K–12 award $500

For awards, final score is reported as:
- 85% private leaderboard
- 15% report/review component

Ties are broken by higher private leaderboard score.

---

# 19. Source Links

Official / competition:
- https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench
- https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/evaluation
- https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules
- https://github.com/jacky850/trafficflowbench-public

Verified public notebooks:
- https://www.kaggle.com/code/soukeaizenz/0-809-lb-top-50-open-sourcing-physics-pipeline
- https://www.kaggle.com/code/lamhuy8904/traffic-flow-bench-pipeline
- https://www.kaggle.com/code/avikdas567/hydrodynamic-traffic-reconstruction-queue-odme
- https://www.kaggle.com/code/chaitanyajamble/2026-ieee-big-data-traffic-flow-bench-3

Participant repository (claims require independent verification):
- https://github.com/MAYANKSHARMA01010/2026-ieee-big-data-traffic-flow-bench

Research:
- https://arxiv.org/abs/1707.01926
- https://doi.org/10.1016/0191-2615(94)90002-7
- https://doi.org/10.1016/0191-2615(94)00022-R
- https://doi.org/10.1016/j.trc.2018.07.023
- https://doi.org/10.1109/access.2026.3673964

---

# 20. Final Conclusion

TrafficFlowBench rewards a competitor who understands **traffic dynamics, inverse problems, validation design, and leakage control** more than a competitor who simply trains a large neural network.

The highest-value near-term direction is:

1. reproduce the official toolkit exactly;
2. build blocked-month/event-disjoint validation;
3. strengthen Task 1 with interpolation + residual ML;
4. jointly improve Task 3 via conservation-aware refinement;
5. create true training queue windows and attack Task 2 with shockwave-aware features + IoU-aware post-processing;
6. solve Task 4 with exact metric-aware constrained optimization;
7. ensemble only models that are OOF-diverse;
8. choose final submissions using a frozen train-derived private proxy.

This is the most defensible route to a strong hidden/private result.
