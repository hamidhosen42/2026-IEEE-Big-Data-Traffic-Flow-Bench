# ROUND 8 — Task 2 onset: seed bagging, exact expected-IoU decoding, precursor columns

Written 2026-11-06 11:48 +06 by the orchestrating session, BEFORE any round-8 model, feature or number exists.
Scripts must assert the sha1 of this file (printed in VERDICT) and must not change it. Any deviation from this text is reported
as a deviation, never silently applied.

## Why (evidence available before writing)
- Production onset (consolidation/tfb/stages/onset_sp.py:304) is ONE CatBoost (600 it, depth 8, lr 0.08, random_seed 0); the April
  pv bag (onset_pv.py:416) is 3 CatBoosts that all use random_seed 0. EXP_T2_PHASE2 measured a per-seed IoU spread of about
  ±0.02 on the 214 pseudo-March onset windows, i.e. production carries single-seed variance.
- The production decoder maximises sum(top p) / (m + sum(rest p)), a ratio of expectations. Onset windows have a median of about
  3 true cells, where E[ratio] and ratio of E differ most.
- The onset rows use slot-T speed but not slot-T flow; plarmuseau's public analysis reports downstream density (AUC 0.76 at 30 min)
  and upstream accumulation (0.73) as the earliest onset precursors.

## Evaluation sets (identical to EXP_T2_PHASE2)
- E1 = the 214 pseudo-March onset windows (new scenario). E1 is evidence for APRIL rows only: pseudo-March windows lie after the
  official March windows' T, so they may not select anything for March rows.
- E2 = train OOF offlike onset windows, 4 folds pooled (about 2,105 windows). E2 is the only evidence allowed for MARCH rows.
- Score = per-window IoU averaged window -> (panel) -> overall exactly as phase_report; window-bootstrap 95 % CI (2,000 draws);
  "non-tie panel" = panel with |seed-mean Δ| > 1e-9.

## Model recipe
Production-like lab onset recipe of EXP_T2_PHASE2 (69 base columns, step-6 CatBoost). Seed s = CatBoost random_seed s; every other
setting unchanged. A bag of K = mean of the K seeds' probabilities, then decoding. Reference R0 = mean IoU over the single seeds of
the same set (expected IoU of a production-like single model); seed 0 alone is also reported.

## Arm BAG — bag of 10 seeds vs single seeds
Replicates: seeds 0–9 and 10–19 (E1, reuse the cached EXP_T2_PHASE / PHASE2 base models); E2: seeds 0–9 per fold (train what is
missing).
- PASS-April: pooled E1 Δ(bag − single-seed mean) ≥ +0.005 and CI lower bound > 0; each replicate ≥ +0.002; ≥ 70 % of non-tie
  panels positive and none below −0.02; E2 Δ ≥ −0.002.
- PASS-March: E2 Δ ≥ +0.002, CI lower bound > 0, ≥ 6/8 panels non-negative.

## Arm DEC — exact expected-IoU decoding (on the BAG probabilities)
Cells independent Bernoulli(p); candidates = top-m prefixes, m = 1..#cells; choose argmax_m E[|S∩T| / |S∪T|] computed exactly
(Poisson-binomial DP inside and outside S). Variants: (a) raw bag probabilities; (b) isotonic-calibrated probabilities, the map for
fold f fitted on the E2 OOF predictions of the other folds (E1 uses the map fitted on all E2 folds). Δ is against the production
decoder on the same probabilities.
- PASS-April: E1 Δ ≥ +0.005, CI lower bound > 0, ≥ 70 % of non-tie panels positive; E2 Δ ≥ −0.002.
- PASS-March: E2 Δ ≥ +0.003, CI lower bound > 0, ≥ 6/8 panels non-negative.
If both variants pass, the one with the larger E2 Δ is proposed (E2 is the scenario-free ruler).

## Arm FEAT — precursor columns (data ≤ T only)
Base columns plus: (i) slot-T flow per lane / capacity per lane, with the production slot-T visibility simulation (62 %);
(ii) slot-T density k = q/v per lane over k_crit per lane, own link and first downstream link; (iii) mean first-downstream-link
density ratio over T−15..T; (iv) upstream accumulation Σ_{links within 3 km upstream} k·L over Σ k_crit·L; (v) flow change across
the last :00 / :30 boundary at or before T (mean of the ≤ 3 slots after it minus the 3 before it, per lane / capacity per lane).
Compared as bag-10 FEAT vs bag-10 base with the same seeds (replicates 0–9 and 10–19), production decoder (or DEC if DEC passed).
- PASS-April: pooled E1 Δ ≥ +0.008, CI lower bound > 0, each replicate ≥ +0.003, ≥ 70 % of non-tie panels positive, none below
  −0.02; E2 Δ ≥ −0.002.
- PASS-March: E2 Δ ≥ +0.003, CI lower bound > 0, ≥ 6/8 panels non-negative.

## Triage (no gate, reported)
On E1 windows with zero overlap under the base bag: rank of the true T+30 link(s) in the bag probability (share with a true link at
rank 2–3 vs not in the top 5), and mean |SHAP| (CatBoost ShapValues) per feature group (prior / position-time vs state).

## Deployment rule
An arm is proposed for production only if it PASSES; an April-only pass changes only April onset rows, March rows change only with
PASS-March. Order: BAG, then DEC on the BAG probabilities, then FEAT on BAG(+DEC). The public leaderboard is not used for any
decision. No April labels, no April data beyond what the production kernels already read for April rows.
