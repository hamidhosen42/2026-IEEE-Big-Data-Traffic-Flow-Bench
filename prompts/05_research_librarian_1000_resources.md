

<pasted_content id="6063">
# ROLE

তুমি একজন Senior ML Research Scientist, Kaggle Grandmaster-level Competition Engineer, Traffic Flow Forecasting Researcher এবং Research Librarian হিসেবে কাজ করবে।

আমাদের project:

**2026 IEEE Big Data — Traffic Flow Bench**

আমাদের বর্তমান strongest pipeline / notebook এবং previous experimental notebooks তোমাকে দেওয়া হয়েছে।

বর্তমান best known Public LB:

**0.85309**

Target:

# 0.90+

তোমার এই phase-এর প্রধান কাজ নতুন model implement করা নয়।

প্রথমে:

# এই competition এবং আমাদের existing solution-এর সঙ্গে সম্পর্কিত যত বেশি সম্ভব high-quality resource খুঁজে বের করো, organize করো, analyze করো, এবং সেখান থেকে genuinely new experiment hypothesis তৈরি করো।

---

# 1. প্রথমে আমাদের notebooks পড়ো

Internet research শুরু করার আগে সব provided notebook/file deeply inspect করো।

বিশেষভাবে বুঝো:

- Base pipeline
- Current best pipeline
- Task 1 implementation
- Task 2 implementation
- Task 3 physics handling
- Task 4 ODME
- Feature engineering
- Validation
- Pseudo-window generation
- Priors
- LightGBM
- Decoder
- Queue onset
- Queue ongoing
- Mask simulation
- Occupancy features
- Gap-aware features
- NNLS
- Submission generation

একটা internal map বানাও:

`Existing Method → Existing Experiment → Result → KEEP/DROP/PENDING`

Research থেকে এমন experiment recommend করবে না যেটা আমরা ইতিমধ্যে properly test করে reject করেছি, unless নতুন evidence দেখায় যে materially different implementation test করা উচিত।

---

# 2. আমাদের known experiment history preserve করো

বর্তমান knowledge:

### DROP

EXP_001/002:
T+5…T+25 onset modelling

কারণ official onset windows-এর queue মূলত T+30-এ শুরু হয়।

EXP_010/011:
queue boundary / shockwave version tested previously gave ~0 gain.

EXP_015:
window weighting gave no gain.

EXP_018:
tested LWR/density corrections did not improve combined result.

EXP_019:
simple λ sweep has low expected value because ODME is already near saturation.

### KEEP / promising

EXP_005/006:
day-aware/smoothed priors

EXP_009:
candidate radius improvement increased coverage.

### PENDING / investigate

EXP_003/004:
blend optimization

EXP_007/008:
hazard/gradient

EXP_012:
decoder

EXP_014:
CatBoost — but correlation with LightGBM ≈ 0.993

EXP_016:
Task1 regime-aware modelling

EXP_017:
low-rank auxiliary reconstruction

EXP_020:
final integration

Do not rediscover these facts and present them as new research.

We need ideas BEYOND them.

---

# 3. EXHAUSTIVE RESOURCE DISCOVERY

Conduct a broad systematic web/research search.

Target:

# 1000+ UNIQUE CANDIDATE RESOURCES

if publicly discoverable.

However:

Do NOT fabricate a resource count.

If only 742 legitimate unique relevant resources are found, report 742.

Never claim to have read 1000 papers if you only searched/indexed them.

Distinguish:

- DISCOVERED
- SCREENED
- READ ABSTRACT/SUMMARY
- DEEPLY ANALYZED
- CODE INSPECTED

---

# 4. RESOURCE TYPES

Search across all relevant source families.

## A. Competition-specific

Search extensively for:

- 2026 IEEE Big Data Traffic Flow Bench
- Kaggle competition pages
- Kaggle notebooks
- Kaggle discussions
- public submissions
- public solution notebooks
- high-scoring notebooks
- forks
- GitHub mirrors
- competition writeups
- participant discussions
- IEEE challenge documentation
- benchmark documentation
- organizer repositories
- official baselines
- leaderboard discussions
- solution explanations

Search variations of:

"Traffic Flow Bench"

"TrafficFlowBench"

"2026 IEEE Big Data Traffic Flow Bench"

"traffic-flow-bench"

"IEEE Big Data Traffic Flow"

and related identifiers.

---

# 5. TRAFFIC4CAST

Research Traffic4Cast deeply.

Search:

- Traffic4Cast 2019
- Traffic4Cast 2020
- Traffic4Cast 2021
- Traffic4Cast 2022
- Traffic4Cast 2023
- winning solutions
- top-3 solutions
- Kaggle notebooks
- GitHub repositories
- papers
- post-competition reports

Extract transferable techniques.

Do NOT assume techniques transfer automatically.

---

# 6. TRAFFIC STATE RECONSTRUCTION / MISSING DATA

Search literature on:

- traffic state reconstruction
- traffic data imputation
- missing traffic sensor data
- traffic speed imputation
- traffic flow imputation
- sparse traffic observations
- structured missingness
- block missingness
- spatiotemporal missing data
- sensor failure reconstruction
- freeway traffic reconstruction

Methods:

- linear interpolation
- spline
- PCHIP
- Kalman filtering
- Gaussian process
- Kriging
- matrix completion
- SoftImpute
- SVD
- PCA
- PPCA
- Robust PCA
- tensor completion
- Tucker
- CP decomposition
- TRMF
- Bayesian matrix factorization
- graph matrix completion
- graph regularization
- KNN imputation
- MissForest
- MICE
- autoencoder
- denoising autoencoder
- VAE
- GAN imputation
- diffusion imputation
- transformer imputation

Special attention:

structured masks similar to R1/R2/R3.

---

# 7. TRAFFIC FORECASTING

Search extensively:

- traffic speed forecasting
- traffic flow forecasting
- freeway forecasting
- sensor forecasting
- network traffic forecasting
- short-term traffic prediction

Include classical + modern methods.

---

# 8. GRAPH/SPATIOTEMPORAL MODELS

Research:

- STGCN
- DCRNN
- Graph WaveNet
- AGCRN
- MTGNN
- STFGNN
- ASTGCN
- GMAN
- STGODE
- D2STGNN
- PDFormer
- STAEformer
- STID
- MegaCRN
- DGCRN
- Graph Transformer
- adaptive adjacency
- dynamic graph
- spatial attention
- temporal graph network

Also inspect frameworks:

- LibCity
- BasicTS
- UCTB
- Torch Spatiotemporal
- PyTorch Geometric Temporal

For each determine whether it fits our corridor structure and data volume.

---

# 9. TIME-SERIES MODELS

Research:

- LightGBM
- XGBoost
- CatBoost
- HistGradientBoosting
- Random Forest
- ExtraTrees
- TCN
- LSTM
- GRU
- N-BEATS
- N-HiTS
- TFT
- Informer
- Autoformer
- FEDformer
- PatchTST
- TimesNet
- DLinear
- NLinear
- TiDE
- TSMixer
- iTransformer
- TimeMixer
- Chronos
- TimesFM
- Moirai

Do not recommend foundation models merely because they are popular.

Determine whether competition structure makes them useful.

---

# 10. QUEUE ONSET / CONGESTION FORMATION

This is extremely important.

Search:

- freeway queue prediction
- queue onset prediction
- congestion onset prediction
- traffic breakdown prediction
- bottleneck prediction
- recurrent bottleneck detection
- freeway breakdown probability
- pre-congestion detection
- congestion classification
- incident-related congestion
- capacity drop prediction
- queue length forecasting
- queue propagation forecasting

Look specifically for features available BEFORE breakdown.

Extract:

- speed-drop indicators
- occupancy
- density
- flow/capacity
- temporal derivatives
- volatility
- upstream/downstream gradient
- bottleneck structure
- traffic instability measures
- breakdown hazard models

---

# 11. TRAFFIC FLOW THEORY

Search:

- LWR
- kinematic wave theory
- Cell Transmission Model
- CTM
- fundamental diagram
- triangular fundamental diagram
- Newell simplified theory
- shockwave theory
- traffic conservation
- capacity drop
- bottleneck dynamics
- ramp flow
- traffic density estimation

But remember:

we already tested simple LWR/density correction with little/no gain.

Find substantially different approaches rather than repeating the same correction.

---

# 12. PHYSICS-INFORMED ML

Research:

- physics-informed traffic forecasting
- physics-informed neural networks traffic
- differentiable traffic simulation
- neural CTM
- hybrid traffic models
- conservation-aware neural networks
- differentiable LWR
- physics-guided graph neural networks

Identify approaches that could improve prediction without destroying Task1 RMSE.

---

# 13. OD MATRIX ESTIMATION

Research:

- OD matrix estimation
- ODME
- link-count OD estimation
- path flow estimation
- entropy maximization
- generalized least squares
- Bayesian OD estimation
- bilevel OD estimation
- stochastic OD estimation
- NNLS OD
- regularized inverse problems
- destination attraction estimation
- path-flow reconstruction

But remember:

our Task4 appears close to saturation.

Only recommend OD experiments with strong evidence of nontrivial upside.

---

# 14. IoU / STRUCTURED BINARY PREDICTION

Research:

- expected IoU optimization
- Jaccard optimization
- differentiable IoU loss
- Lovasz loss
- structured segmentation decoding
- probability threshold optimization
- calibrated probability decoding
- top-k optimization
- set prediction
- structured binary prediction

We currently use expected-IoU style decoding.

Find mathematically stronger alternatives.

---

# 15. PROBABILITY CALIBRATION

Research:

- Platt scaling
- isotonic regression
- beta calibration
- temperature scaling
- histogram calibration
- calibration under class imbalance
- calibration for IoU/Jaccard optimization

Check whether better calibration could improve Task2 decoder without retraining core models.

---

# 16. CLASS IMBALANCE

Research:

- focal loss
- weighted BCE
- balanced sampling
- hard-negative mining
- positive-unlabeled ideas
- rare-event prediction
- event-based sampling
- ranking losses

Queue onset is a rare-event problem.

Look for techniques relevant to rare breakdown events.

---

# 17. CHANGE-POINT / EVENT DETECTION

This is a new high-priority research direction.

Search:

- change point detection traffic
- Bayesian change point
- CUSUM congestion detection
- BOCPD
- regime switching
- Hidden Markov Models
- traffic regime detection
- breakdown precursor detection

Investigate whether onset T+30 can be predicted from pre-breakdown regime transitions.

---

# 18. SURVIVAL / HAZARD MODELS

Search:

- survival analysis traffic breakdown
- congestion hazard
- freeway breakdown hazard
- time-to-congestion
- Cox model traffic breakdown
- discrete-time hazard model

Potentially model:

`P(queue starts within next 30 minutes | history)`

instead of standard classification.

---

# 19. SELF-SUPERVISED LEARNING

Research:

- masked traffic modelling
- masked time-series pretraining
- self-supervised traffic representation
- contrastive traffic learning
- graph contrastive learning
- masked autoencoder traffic

This may be relevant because Task1 itself is a masking/reconstruction problem.

---

# 20. MULTI-TASK LEARNING

Search:

- joint speed-flow prediction
- joint speed-flow-density estimation
- traffic multi-task learning
- shared encoder traffic prediction
- reconstruction + forecasting joint training
- physics + prediction multi-task models

Potential opportunity:

Task1/Task2/Task3 share traffic state information.

---

# 21. ENSEMBLING

Research:

- OOF stacking
- blending
- residual ensemble
- heterogeneous ensembles
- regime-specific ensemble
- mixture-of-experts
- gating networks
- dynamic ensemble selection

Particularly:

`model choice conditional on missingness/regime/traffic state`.

---

# 22. MIXTURE OF EXPERTS

High-priority search:

- traffic mixture of experts
- regime-aware MoE
- congestion-state experts
- temporal regime gating
- sparse MoE time series

Possible experts:

- free flow
- near breakdown
- congested
- recovery

This could be more useful than simply R1/R2/R3 separate models.

---

# 23. DISTRIBUTION SHIFT / DOMAIN ADAPTATION

Public/private months differ.

Research:

- temporal domain adaptation
- traffic domain adaptation
- covariate shift
- test-time adaptation
- robust traffic forecasting
- invariant representation
- month-to-month transfer
- incident domain shift

Traffic4Cast domain-adaptation solutions deserve special attention.

---

# 24. VALIDATION RESEARCH

Search competition literature for:

- temporal cross-validation
- blocked CV
- rolling-origin CV
- event-based validation
- traffic incident holdout
- distribution-shift validation
- adversarial validation

We need CV that predicts private leaderboard behavior, not just pseudo-window performance.

---

# 25. SYNTHETIC DATA / AUGMENTATION

Research:

- traffic time-series augmentation
- mixup time series
- temporal jitter
- mask augmentation
- sensor dropout augmentation
- block masking
- graph masking
- rare congestion augmentation

Particularly useful for Task1 masks and Task2 rare onset.

---

# 26. RESIDUAL LEARNING

Current Task1 already uses residual learning.

Research stronger variants:

- boosting residual correction
- multi-stage residual model
- residual MoE
- bias correction
- cross-fitted residual model
- error-conditioned model

Find ideas beyond what current notebook already implements.

---

# 27. FEATURE DISCOVERY

Search literature for traffic features such as:

- occupancy
- density
- flow/capacity
- speed ratio
- demand/capacity
- coefficient of variation
- acceleration
- jerk
- volatility
- upstream/downstream difference
- temporal entropy
- spectral features
- autocorrelation
- wavelet features
- Fourier features
- seasonal decomposition
- congestion index
- bottleneck score
- traffic instability index

Map every useful feature to our available columns.

---

# 28. SIGNAL PROCESSING

Research:

- wavelets
- FFT
- spectral traffic analysis
- Savitzky-Golay
- Kalman smoother
- state-space model
- dynamic linear models
- Gaussian filtering

Look for low-cost auxiliary features rather than blindly replacing the pipeline.

---

# 29. OPTIMIZATION

Research:

- Optuna
- Bayesian optimization
- Hyperband
- successive halving
- population-based tuning

But prioritize high-impact structural changes over brute-force hyperparameter search.

---

# 30. CODE SEARCH

For every promising paper, search for:

- official GitHub
- author implementation
- reproducible repo
- PyTorch implementation
- benchmark implementation
- Kaggle implementation

Code availability matters.

A theoretically strong paper with no practical path should receive lower experiment priority.

---

# 31. RESOURCE DATABASE

Create:

`resource_index.csv`

Columns:

- resource_id
- title
- year
- authors
- resource_type
- source
- canonical_url
- DOI/arXiv ID
- code_url
- competition_task
- method_family
- dataset
- metric
- key_result
- relevance_to_our_pipeline
- novelty_vs_existing_pipeline
- implementation_cost
- compute_cost
- leakage_risk
- expected_value
- status

Status:

- DISCOVERED
- SCREENED
- DEEP_READ
- CODE_INSPECTED

---

# 32. DEDUPLICATION

Do not count the same paper multiple times because it exists on:

- arXiv
- IEEE
- ResearchGate
- GitHub
- author's webpage

Canonicalize by:

1. DOI
2. arXiv ID
3. exact title
4. repository identity

Report:

`raw discovered count`

and

`unique deduplicated count`.

---

# 33. QUALITY CONTROL

Prioritize:

### Tier A

- official competition resources
- peer-reviewed papers
- top conference papers
- official repositories
- winning competition solutions
- highly reproducible implementations

### Tier B

- arXiv with strong evidence/code
- credible technical reports
- established benchmark repositories

### Tier C

- Kaggle discussions
- blogs
- community implementations

Tier C can generate hypotheses but should not override stronger evidence.

---

# 34. RECENCY

Search both foundational and recent work.

Prioritize:

**2020–2026**

but include older foundational work where necessary:

- LWR
- CTM
- Kalman
- matrix completion
- OD estimation
- traffic breakdown theory

---

# 35. DO NOT JUST COLLECT LINKS

For every promising method answer:

1. What problem does it solve?
2. Why might it help OUR notebook?
3. Which current weakness does it address?
4. Has something equivalent already been tested?
5. What exact code change would be required?
6. What data would it need?
7. Is that information available at inference time?
8. Leakage risk?
9. Compute cost?
10. Expected upside?
11. How should we validate it?
12. What result would cause us to reject it?

---

# 36. CROSS-REFERENCE AGAINST OUR EXPERIMENT HISTORY

For each research idea label:

### NEW

Never tested.

### RELATED

Similar to something tested but materially different.

### DUPLICATE

Already tested.

### CONTRADICTED

Our experiments already provide evidence against it.

### HIGH-VALUE RETEST

Previously tested, but literature suggests our implementation may have missed an important component.

Do not recommend DUPLICATE ideas.

---

# 37. FIND NEGATIVE RESULTS TOO

Do not search only for successful methods.

Find evidence showing:

- methods that fail under high missingness
- GNN limitations
- PINN instability
- matrix completion limitations
- domain-shift failures
- over-smoothing
- calibration problems
- leaderboard overfitting

Negative evidence saves compute.

---

# 38. RESEARCH → EXPERIMENT CONVERSION

After resource mining, create:

`research_to_experiments.md`

For each experiment:

### EXP_NEW_001

Research basis:
[resources]

Current weakness:
[...]

Hypothesis:
[...]

Exact change:
[...]

Affected task:
Task1 / Task2 / Task3 / Task4

Expected metric:
[...]

Smoke test:
[...]

Validation:
[...]

Compute:
Low / Medium / High

Leakage risk:
Low / Medium / High

Expected gain:
Low / Medium / High

Confidence:
Low / Medium / High

Decision threshold:
[...]

---

# 39. PRIORITIZATION FORMULA

Rank experiments approximately using:

`Priority = Expected Gain × Evidence Strength × Novelty × Confidence / (Compute Cost × Leakage Risk)`

Do not rank by novelty alone.

---

# 40. FIND AT LEAST 50 GENUINELY NEW EXPERIMENT IDEAS

After research, generate at least:

# 50 non-duplicate candidate experiments

Then reduce them to:

### Top 30
worth considering.

Then:

### Top 15
worth implementing.

Then:

### Top 5
run immediately.

Do NOT force 50 if legitimate distinct hypotheses cannot be supported.

Quality > artificial count.

---

# 41. SEARCH FOR "CHEAP GAINS"

Explicitly identify experiments that require:

- no retraining
- decoder only
- calibration only
- feature transformation only
- blending only
- post-processing only
- cached OOF predictions

These should be tested early.

---

# 42. SEARCH FOR "HIGH-UPSIDE STRUCTURAL GAINS"

Separately identify experiments involving:

- regime-aware reconstruction
- mixture-of-experts
- masked pretraining
- graph reconstruction
- low-rank + ML hybrid
- change-point onset detection
- survival/hazard onset model
- domain adaptation
- multi-task state/queue learning

These can become the next generation pipeline.

---

# 43. SPECIAL INVESTIGATION — WHY ARE TOP TEAMS >0.90?

Do not speculate about private implementations.

Use only public evidence.

Investigate whether publicly visible high-performing approaches suggest advantages from:

- better Task1
- better Task2
- physics
- OD
- ensembling
- validation
- priors
- synthetic generator exploitation
- temporal structure

Clearly separate:

FACT

from

INFERENCE.

---

# 44. COMPETITION GENERATOR ANALYSIS

If public competition code/documentation describes synthetic generation, deeply analyze it.

Determine:

- temporal generation process
- incident process
- bottleneck process
- masking process
- queue generation
- OD generation
- noise model
- missingness model
- public/private differences

A competition-specific generative insight may be more valuable than a sophisticated generic model.

Do NOT exploit hidden/private information or violate competition rules.

---

# 45. REPRODUCIBILITY

For every top experiment identify:

- package requirements
- GPU/CPU
- approximate runtime
- memory
- code source
- license if relevant
- implementation difficulty

Prefer reproducible methods.

---

# 46. DO NOT FABRICATE

Never invent:

- paper
- URL
- DOI
- GitHub repository
- score
- experiment result
- leaderboard result

Every resource must have a verifiable source.

If access is unavailable, label:

`DISCOVERED — NOT VERIFIED`

Do not claim it was analyzed.

---

# 47. DO NOT STOP AT 500

Continue searching until saturation.

Saturation means:

several consecutive search families produce almost no genuinely new method families or competition-relevant hypotheses.

Target 1000+ unique candidates if feasible.

But relevance is more important than hitting a number.

---

# 48. FINAL RESEARCH REPORT

Produce:

## A. Resource statistics

Raw resources discovered:

Unique after deduplication:

Deeply screened:

Deep-read:

Code inspected:

Year distribution:

Resource-type distribution:

---

## B. Research map

Organize findings into:

- Task1 reconstruction
- Task2 onset
- Task2 ongoing
- Task3 physics
- Task4 ODME
- validation
- calibration
- ensemble
- domain adaptation
- generator-specific insights

---

## C. What our current pipeline already does well

Explain why 0.85309 is strong.

---

## D. Current bottlenecks

Identify evidence-supported remaining weaknesses.

---

## E. Rejected research directions

List methods that should NOT consume compute and explain why.

---

## F. Top 50 new experiments

Ranked.

---

## G. Top 15 implementation candidates

Provide detailed implementation plans.

---

## H. TOP 5 NEXT EXPERIMENTS

These must be immediately actionable against our current best notebook.

For each include exact:

- file/cell/function to modify
- feature/model change
- parameters
- validation
- smoke test
- KEEP/DROP rule

---

# 49. IMPORTANT WORKING PRINCIPLE

Do not ask:

"What is the most advanced traffic model?"

Ask:

"What modification has the highest probability of improving OUR CURRENT 0.85309 pipeline under THIS competition metric and private-test distribution?"

That distinction is critical.

---

# 50. FINAL OBJECTIVE

We are not doing a literature review for academic interest.

Research has one purpose:

# Find overlooked, evidence-backed experiments capable of moving our validated solution from ~0.853 toward 0.90+.

Research broadly.

Filter aggressively.

Cross-reference everything against our notebooks.

Avoid already-failed ideas.

Extract actionable techniques.

And finish with a concrete experiment queue rather than a pile of links.
</pasted_content id="6063">
