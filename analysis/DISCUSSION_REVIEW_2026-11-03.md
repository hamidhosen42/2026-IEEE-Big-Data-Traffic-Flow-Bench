# Discussion review against the current 0.88281 pipeline

Reviewed: 2026-11-03, Asia/Dhaka. Scope: analysis only; no training, kernel push, submission, or changes to the selected predictions.

## Finding

The 19 locally archived discussion topics contain useful, organizer-confirmed Task 2 information, but **no newly demonstrated method that takes the current pipeline above 0.90**. The strongest disclosed advantages are already represented in the current solution. This is an evidence assessment, not a claim that 0.90 is impossible.

In particular, the **0.93** reported in discussion 742328 is an onset-only IoU on held-out training dates, not a composite public leaderboard score. The **0.9** in discussion 742761 is a question about S_queue, not an achieved result or a proven ceiling.

Baseline: `consolidation/README.md` and `consolidation/LINEAGE_K9.md` identify A3 = tfb-k9 and B3 = tfb-k10, both public LB **0.88281**. The root README's 0.85309 and the starting point in `research/SCORE_GAP_ANALYSIS.md` are historical, not current.

## Evidence and freshness

- Read all 19 topic files, including their archived replies, under `discussions/`. The index says the forum was fetched on **2026-11-03 16:41 UTC / 22:41 Bangladesh time**.
- Cross-checked the claims against the current stage implementations, experiment tracker, selector replay, transfer analysis, capacity-scenario analysis, and recent experiment results through October 30.
- Web search found the official competition pages. Direct web reads of the detailed discussion pages were unavailable or empty; a separate anonymous API refresh failed because shell network DNS was unavailable. Therefore the detailed discussion evidence here is the dated local archive, not an independently refreshed live forum.
- Official current rules corroborate the online/offline distinction and the September rebuild: <https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules>.
- Organizer statements, participant claims, measured project results, and new hypotheses are distinguished below. No gain is assigned to an untested hypothesis.

## What the useful threads actually offer

| Thread | Evidence | Implication | Status against our solution |
|---|---|---|---|
| [742350](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350), [742328](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742328) | Organizer confirms all 120 onset windows have empty first five horizon steps and queue only at T+30; conditioning on this is allowed | Predict location at step 6, rather than spreading positives over six steps | **Used.** `to_matrix` uses `[6]` for onset. Participant's 0.57 → 0.93 is a train-onset result; it is not fresh headroom over our baseline |
| [745700](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/745700) | Organizer allows observations stamped T, plus conditioning on the published 360-minute spacing and maximum-persistence-IoU selection rule; only origins at or before T may be used | Origin-slot features and selector-faithful training matter | **Substantially used.** Slot T is read; F03 uses faithful candidates and run-start weighting. Exact-360-minute regime switching was tested and did not establish a public gain |
| [749278](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749278) | Organizer says Task 2 history spans are protected from Task 1 masking; train has unmasked states | Ordinary masked calendar windows do not match official history coverage | **Known and partly addressed.** Raw train histories and selection-faithful sets exist. Pseudo-March remains an imperfect proxy, especially for small onset effects |
| [748789](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748789) | Earlier same-split observations may refit a model; all released March traffic may inform April forecasts, but not the reverse | Same-scenario adaptation is legitimate and more promising for private than early public windows | **Used.** Scenario prior sp4, March-augmented onset pv and ongoing pvogf. A3/B3 already hedge this assumption |
| [748784](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748784) | Organizer confirms all 80 official train windows are in June; labels use underlying state, not noisy observations; wider train windows are reasonable | Official-train-only validation is too narrow; observed-cell proxy IoU is not the exact official metric | **Known.** Nine-month faithful train sets exist; noisy/partial labels remain a validation limitation |
| [749601](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749601) | Looking at released descriptive metadata is allowed. Traffic after T must not enter fitting, selection, or values in a forecast; organizers will truncate data after each T and rerun | A method must be causal end to end, not merely at feature extraction | Current provenance/truncation work is relevant. Full-March pseudo-label diagnostics cannot by themselves justify selecting a method for earlier March windows |
| [742761](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742761) | Organizer rejects neither achievable scores nor forecasting difficulty; labels themselves have no observation-noise floor | Do not treat the participant's onset cap near 0.7 as a theorem | **No algorithm disclosed.** Noise-free labels do not imply that future state is identifiable from permitted history |
| [743819](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/743819) | Organizer specifies milepost/link_id ordering, incoming/outgoing connectivity, ramp attachments; disconnected detector-only topology is expected | Do not force a connected graph or infer scorer order from from_node/to_node | **Known/audited.** No newly demonstrated 0.01719 gain; distinguish scorer topology from empirically learned propagation direction |
| [742980](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742980) | Participant shares 0.809 pipeline, regularized NNLS, soft FD projection; abrupt onset defeats simple high-speed suppression | Good baseline lessons, especially ODME scaling; simplistic onset suppression can fail | **Already surpassed.** Our pipeline descends from this baseline, with richer reconstruction and queue models |
| [742068](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068), [742910](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742910) | Task 2 uses traffic at or before T; Task 1 is offline; synthetic dates do not match real traffic archives | Offline T1 reconstruction and post-T ramps cannot supply online queue forecasts. External traffic is not an answer source | **Contract clarification, not new signal** |

Remaining topics were also read:

| Thread | Relevance to a 0.90 method |
|---|---|
| 740522, 740536 | Historical disclosures of direct horizon access and overlapping-history access; both led to repairs/rebuild. These are not remaining forecasting methods |
| 740844 | September 11 rebuild changed the scored traffic and IDs; earlier public scores are incomparable. Do not infer present attainability from an old high score |
| 744187, 749480 | Award process, registration, reproducibility and reporting; no modeling method |
| 741898 | Older student-award clarification, superseded by the later award announcement; no modeling method |
| 743549 | Secondary-account submission issue; no modeling method |

## Exact score budget from today's baseline

The competition weights are:

`S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`.

With other outputs unchanged:

`gap = 0.90000 - 0.88281 = 0.01719`.

| Changed part | Required absolute component improvement |
|---|---:|
| Queue, both conditions averaged | **+0.05730 S_queue** |
| Onset alone | **+0.11460 onset IoU** |
| Ongoing alone | **+0.11460 ongoing IoU** |
| State alone | +0.04911 S_state |

The effective weights from the local Task 3 specification are 0.05 FD and 0.10 LWR. That would require +0.17190 LWR if that term alone closed the gap. These are arithmetic requirements, not feasibility bounds.

The historical queue decomposition in `TRANSFER_ANALYSIS_2026-10-08.md` estimates onset ≈0.765 and ongoing ≈0.863. The current picks retain the same public queue block. **If those approximate estimates are accurate**, S_queue is about 0.814 and would need to reach about **0.8713** with the other terms held fixed. Illustrative combinations:

| Onset | Ongoing | Mean S_queue | Approximate total under that decomposition |
|---:|---:|---:|---:|
| 0.7650 | 0.8630 | 0.8140 | 0.88281 |
| 0.8200 | 0.9200 | 0.8700 | 0.89961 |
| 0.8300 | 0.9200 | 0.8750 | 0.90111 |

These totals are scenarios, not measured candidate scores. Current exact component scores are not available from the composite leaderboard alone.

There are 40 windows per condition on public. Under the balanced aggregation, one window contributes `0.15 / 40 = 0.00375` times its IoU. Closing 0.01719 through queues means **4.584 units of summed per-window IoU improvement**, e.g. roughly ten windows improving by 0.4584 each. It is a substantial behavioral change, not a few marginal threshold edits.

## Which apparently promising ideas have already been tested?

Cross-checks: `experiments/TRACKER.md`, `analysis/TRANSFER_ANALYSIS_2026-10-08.md`, `analysis/CAPACITY_SCENARIO_2026-10-11.md`, and the corresponding local result files.

- **Selector fidelity:** F03 already produced public +0.00474. It is in the baseline, so it cannot be added again to the score budget.
- **360-minute spacing flag:** EXP_T2_G72 switched models on the flagged windows; public delta was −0.00022. This does not disprove all selector modeling, but it gives no remaining demonstrated gain.
- **Same-split early-month adaptation:** sp4 gave +0.00104 public. More elaborate LODO and mix9 variants lost −0.00496 and −0.00144. Large later-month proxy gains do not establish a gain on the first five March days.
- **More ongoing structure:** per-step models, step-6 specialists and queue/free splits did not retain their smaller-model gains at 511 leaves (EXP_T2_OGCAP).
- **Extent calibration is not unexplored:** `extcal_lib.py` already defines per-step Platt/isotonic calibration, inside/outside-queue calibration, a Poisson size model and size-rescaling arms. A fresh proposal must distinguish itself from these experiments rather than advertise them as new.
- **Same-day context:** OGCLOCK showed train gains, but new-scenario gains were small and seed-sensitive; it was not shipped.
- **Physics/shock filtering:** EXP_T2_SHOCK failed the transfer criteria. Later occupancy/vehicle-count features and raw space-time patch arms also failed the stated deployment gates (EXP_T2_PHYS / PATCH).
- **Occupancy for private pvogf:** the recent PVOCC verdict has `adopt: false` for both arms. Its many-day +0.0031 IoU estimate has a confidence interval crossing zero, and the private-footprint estimate is only about +0.00015 total. It is not evidence for +0.01719.
- **Onset label noise:** EXP_onset_noise found only 0.2% of positive onset cells within ±2 km/h of the cutoff in its sample. Label denoising cannot currently be credited with fixing the main onset gap.
- **Capacity ×1.2 correction:** the capacity-scenario study found no corresponding train/evaluation miscalibration. The published factor is not an unused score multiplier.
- **More T1 models and smoothing:** several additions are already in K9. Recent gains are much smaller than the remaining gap; T4 is already near saturation according to the existing probes. These observations do not constitute an absolute upper bound on T1/3.

## What is still worth investigating, without calling it a 0.90 solution?

### 1. A joint distribution over queue sites and extent, conditioned on selection

**Hypothesis, untested in this review.** Current `t2_occ.py:decode` ranks cell probabilities and maximizes the proxy `sum(top p) / (m + sum(rest p))`. It does not explicitly infer competing whole-window queue trajectories.

A distinct experiment would model candidate onset sites and ongoing queue-boundary trajectories jointly, using pre-origin evidence, then choose a prediction by average IoU across plausible trajectories. Condition the trajectory distribution on the published onset/persistence selector. Compare against the actual production bag, not a weaker single model.

This differs from simply narrowing predictions to one or two sites, per-step size scaling, or removing cells with a shock-speed rule; variants of those approaches have already been tested. It targets the two error types identified in the transfer study:

- Onset: 57–78% of loss in the analyzed new-scenario proxies comes from zero-overlap windows, i.e. wrong location.
- Ongoing: the study's per-step size oracle recovers about +0.05–0.06 IoU, indicating useful extent information could matter. That oracle is unavailable at inference and is not a demonstrated achievable gain.

**Important inference limit:** selection requires true-state persistence IoU ≤0.9. Our observed/filled queue at T is noisy and incomplete. Forcing the final prediction's IoU against that proxy below 0.9 is not logically guaranteed to improve it. The selector should inform a probabilistic model and be validated, not imposed as an assumed answer constraint on a noisy reference. Even a Bayes-optimal prediction need not satisfy every property of each possible true trajectory.

Because F03 training already conditions on selection, the remaining benefit may be small. There is no measured gain for this proposal.

### 2. Treat early-public and later-private evidence separately

**Evidence-backed direction, mostly already used.** March's first official windows have very little same-scenario past; April has the preceding month. The organizer explicitly allows that history. The existing A3/B3 split is consequently sensible, but it is already part of the selected solution.

Any new adaptation experiment should use expanding chronological prefixes with protected-history-like inputs, and assess queue location and extent separately. Whole-March diagnostics may inform April method selection, but cannot supply post-origin traffic to earlier March fitting/selection. Held-out future labels may score a frozen backtest; they must not be used to fit or choose that same earlier forecast.

### 3. Improve the ruler before believing a small gain

**Evidence-backed validation need.** Protected histories, early-origin sampling, partially observed proxy targets and latent official targets all matter. Reproduce the history-coverage distribution and rolling month-start evaluation, then require improvement across panels/scenario holdouts. Use retrospective FM-PM as a diagnostic; its whole-March outputs alone are not causal evidence for selecting an earlier March forecast.

The transfer study estimates onset public-probe variability around 0.003–0.005 total for the tested changes. Public scoring is deterministic for a fixed file; this variability concerns sampling and choosing among variants, not server randomness. Repeated public gains of a few ten-thousandths should not be extrapolated into a reliable private gain or a route to 0.90.

## Decision

**No discussion-derived 0.90+ candidate is demonstrated.** The useful disclosed advantages are mostly already consumed. The remaining plausible research focus is Task 2's site/extent prediction with a faithful chronological evaluation. A joint trajectory model is a distinguishable hypothesis, but it has no measured gain and should not be presented as a promised 0.90 solution.

No production code, model outputs or final picks were changed during this review.

## Fact-check of the subsequently pasted alternative analysis

The pasted analysis correctly identifies the onset timing, inclusive-T input rule, March-to-April adaptation permission and topology contract. Its assertion that a compliant 0.90+ method has been established does not follow from those facts.

1. **Historical component estimates are presented as current ceilings.** The 0.940–0.945 state and roughly 0.778 queue figures originate in the 0.86217-era score-gap analysis. They are neither current exact measurements nor mathematical upper bounds. In particular, the 0.530 LWR ceiling is not established across all panels by the cited local prototypes.
2. **The claimed non-queue maximum is inconsistent with its own table.** Using its stated maxima gives `0.35*0.950 + 0.05*0.991 + 0.10*0.530 + 0.20*0.995 = 0.63405`, not 0.631. Under that assumed table, reaching 0.90 requires S_queue = 0.8865; under 0.631 it requires approximately 0.89667. Neither assumption proves attainability or a true ceiling.
3. **The proposed roadmap's stated minimum targets do not suffice.** Onset 0.85 and ongoing 0.88 give S_queue = 0.865. Combined with the table's assumed best non-queue score, the total is **0.89355**. Under the illustrative current decomposition above it is **0.89811**. Higher achieved scores could cross 0.90, but the stated thresholds alone do not establish that.
4. **The onset 0.93 is a participant's held-out train result.** The organizer confirmed the timing property and its permitted use, not that every model or scored split reaches that IoU. Our pipeline already uses the timing property.
5. **Direction reversal is not an untested fix.** `experiments/TRACKER.md:89` records EXP_021 reversing W/S panels and rejecting it after a negative overall pseudo-window delta. Further, `analysis/t2lab/ogphys_lib.py` records an October 30 data check finding the stable queue head at the high index on all eight synthetic panels. The pasted geographic-direction rule must not be imposed without validating the synthetic dynamics; the later corrected occGd arm also failed its combined deployment gate.
6. **Existing baseline components cannot be counted as future gains.** Transductive training, CatBoost/LightGBM members, ODME regularization and richer FD-aware reconstruction are already present. A generic 15% FD blend is a historical baseline setting, not evidence of improvement over K9.

Assessment: the alternative text summarizes several useful facts, but supplies no new measured candidate or justified score budget that overturns the review's conclusion. Task 2 remains the strongest research priority; calling it the only possible route or declaring 0.90 achievable from these experiments overstates the evidence.
