# Prompt 01 — Grandmaster research & experiment loop, target 0.90+ (2026-09-28)

User message: "akon tumi base notebook ke dore 0.90+ score koro. age tumi ja ja experiment kora dorkar koro. and must be validation and smoke test korba.. tumi ai related 500+ resourse findout koro than analysis kore experiment koro. amader score jeno improve hoiii..."

The attached prompt, section by section:

- **ROLE:** a senior Kaggle Grandmaster-level ML researcher, traffic-flow forecasting specialist and competition engineer. Start from the attached base notebook: understand → reproduce → diagnose → research → experiment → validate → improve → stress-test → finalize. **TARGET: 0.90+**, never fabricated, no leakage, no lucky splits; hidden-test generalization comes first.
- **0 Preserve the original notebook:**
  - keep it as an immutable baseline;
  - work on a copy;
  - record the pipeline and its score;
  - always compare base vs experiment vs best.
- **1 Read the whole base notebook first:** data loading, targets, IDs, features, preprocessing, validation, CV, metric, models, training, inference, post-processing, ensemble and submission.
- **2 Reproduce the baseline (EXP_000_BASELINE):**
  - record CV, folds, runtime, memory, features, config and prediction statistics;
  - if it cannot run, apply the minimal documented fix.
- **3 Verify the competition:**
  - check the task, target, metric, train/test relation, submission format, temporal and spatial structure, rules and external data;
  - separate VERIFIED FACT from INFERENCE from HYPOTHESIS.
- **4 Audit the base notebook:**
  - validation: random / chronological / grouped / blocked / spatial;
  - leakage of every kind, re-checked after every large gain;
  - features: useful, redundant, noisy, leaking;
  - models: under- or overfit, tuning, metric alignment.
- **5 Research 500+ high-relevance resources:**
  - traffic forecasting, spatiotemporal and GNN methods, Kaggle winning solutions, papers, GitHub, model families (LightGBM … AGCRN);
  - extract why each might work, expected gain, difficulty, compute, and leakage/validation risk;
  - produce a synthesis, not a link dump.
- **6 Data forensics:** shapes, missingness, distributions, shift, temporal coverage, entities, seasonality, autocorrelation and spatial correlation.
- **7 Trustworthy validation:** mimic the hidden test and **freeze the folds**.
- **8 Experiment backlog,** ranked by expected gain × confidence / compute cost.
- **9 Feature engineering:** time, lag, rolling, group, spatial and interaction features, leakage-safe only.
- **10 Model experiments:** start from the existing model; strong tabular baselines first; identical folds.
- **11 Hyperparameter optimization:** focus on the sensitive parameters; use Optuna when justified; tune on the competition metric.
- **12 Residual analysis** after every major improvement.
- **13 Ensembling:** mean, weighted, rank, fold, seed and stacking, using OOF weights and checking correlation.
- **14 Metric-specific optimization:** study the metric; validate any post-processing.
- **15 Mandatory smoke test** (PASS/FAIL) before any full run.
- **16 Controlled loop:** hypothesis → minimal change → smoke → full validation → same folds → leakage check → residuals → keep/reject.
- **17 Experiment tracker:** ID, change, CV, delta, fold std, runtime, smoke, leakage, decision; failures are recorded too.
- **18 Improvement gates:** smoke passes, CV improves, the gain is stable, no leakage, inference works, the submission works.
- **19 0.90+ verification:** try to falsify it (leakage, seeds, fold stability, OOF, train–valid gap).
- **20 Leaderboard discipline:** the public LB is not the validation set; track CV↔LB.
- **21 Compute management:** caching, fixed folds, early stopping, efficient dtypes, deterministic seeds.
- **22 Autonomous execution:** do not ask after each experiment; stop only for missing data or access.
- **23 Do not fake execution:** statuses EXECUTED / SMOKE-TESTED / FULLY VALIDATED / PROPOSED / FAILED / NOT EXECUTED.
- **24 Final model selection:** trustworthy CV, robustness, stability, leakage safety, reproducibility.
- **25 Final notebook:** structured config → submission, clean, and it runs **top → bottom without manual editing**.
- **26 Final smoke test,** then the full run.
- **27 Final submission sanity check:** rows, IDs, order, columns, no NaN or inf, plausible ranges.
- **28 Final report:**
  - baseline, research, experiments, improvements (base → change → score → delta), best solution and reliability;
  - **Original score / Best validated score / Absolute improvement / 0.90+ achieved: YES/NO / notebook status / submission status**.

Most important instruction: the base notebook is the reference implementation. Improve it scientifically (research/hypothesis → implementation → smoke test → validation → leakage check → keep/reject) toward the strongest leakage-free, reproducible solution.
