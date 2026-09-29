# Prompt 02 — "Claude Prompt — আমাদের Top/Base Notebook ধরে 0.90+ Target Experiment" (verbatim summary, 2026-09-29)

Context given with the prompt:
- 0.85309 is the current best of the two uploaded variants.
- 0.84917: making onset a 30/70 main/prior-free blend lowered the score.
- So ongoing prior-free helps, but removing the location/time prior from onset hurts.
- Final notebook contents:
  - Task 1: LightGBM residual to temporal interpolation.
  - Task 2: LightGBM per condition + expected-IoU decoding.
  - Task 3: same outputs as Task 1.
  - Task 4: prior-regularized NNLS, λ = 0.05.
- The visible leaderboard top is about 0.92406, so 0.90+ looks technically reachable. Train-derived validation is needed rather than public-LB overfitting.

The prompt's phases:
- **PHASE 1:** analyze all notebooks deeply. Compare the best (0.85309) with the alternative (0.84917) and derive the hypothesis: onset benefits from time/location prior more than ongoing.
- **PHASE 2:** baseline lock. EXP_000 = the current best 0.85309 pipeline, kept as an immutable copy, plus an experiment copy and a tracker.
- **PHASE 3:** verify the score logic. S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME. Task 2 is the fastest iteration path. Do not optimize all tasks at once.
- **PHASE 4:** Task 2 deep audit. Cover:
  - the window selector and the onset/ongoing conditions;
  - 12-step history and 6-step horizon;
  - pseudo windows, official train windows, dense ongoing sampling and spacing;
  - masking at T;
  - features, priors vs prior-free, the expected-IoU decoder, training sample selection, feature importance and the blend.
- **PHASE 5:** optimize onset and ongoing separately.
  - Family A (onset):
    - A1 blend sweep 1.0 → 0.6.
    - A2 per-corridor blend.
    - A3 per-time-of-day blend.
    - A4 improved prior (link×TOD×DOW, ±15 min, smoothing, corridor and bottleneck priors).
    - A5 hazard model features.
    - A6 horizon-specific onset models T+5..T+30 ("high priority").
  - Family B (ongoing):
    - B1 propagation features (queue front/tail, length, components, gradients, bottleneck, lane drop).
    - B2 shockwave features.
    - B3 queue-boundary model.
    - B4 LightGBM + CatBoost ensemble.
- **PHASE 6:** decoder audit. Compare expected-IoU, fixed, optimized, horizon-, condition- and corridor-specific thresholds, and a structured contiguous decoder (morphology only if CV improves).
- **PHASE 7:** Task 2 validation. Compare 4 contiguous blocks with leave-month-period-out, incident-aware, bottleneck-aware and corridor holdouts.
- **PHASE 8:** Task 1 + 3. C1 regime-specific models, C2 low-rank auxiliary features, C3 spatial interpolation features at ±1..±8 links, C4 conservation-residual stage, C5 optimization-based correction.
- **PHASE 9:** Task 4. D1 λ sweep, D2 panel-specific λ, D3 destination-attraction regularization.
- **PHASE 10:** avoid blind deep learning (large GNN, Transformer, PINN, diffusion) until the tabular approach plateaus.
- **PHASE 11:** experiment order in batches 1–4.
- **PHASE 12:** strict experiment tracker.
- **PHASE 13:** smoke test before every full run.
- **PHASE 14:** leakage check. Task 2 may use only data at or before T.
- **PHASE 15:** public LB discipline. Submit only validated, stable improvements.
- **PHASE 16:** 0.90 roadmap. +0.04691 is needed: Task 2 +0.020..0.030, Task 1/3 +0.010..0.020, Task 4 +0.003..0.008 (a research target, not a guarantee).

Most important observation stated in the prompt: ongoing → prior-free is beneficial, onset → too much prior-free hurts. Treat onset as "recurrent location/time risk + current traffic instability" and ongoing as "physical spatial propagation".

Expected output:
- baseline, experiment table and best local CV;
- best submission candidate, the exact changes and why they should generalize;
- leakage audit and smoke-test result;
- final notebook and submission file;
- remaining experiments if the score is below 0.90.
