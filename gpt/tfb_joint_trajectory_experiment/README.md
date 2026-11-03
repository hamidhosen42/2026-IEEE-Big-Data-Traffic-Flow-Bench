# TrafficFlowBench Task 2 — correlated-trajectory decoder (RESEARCH ONLY)

**Created:** 2026-11-03. **No real competition data was supplied or executed.**
This experiment does NOT beat the Kaggle leader, does NOT create a competition CSV,
and does NOT alter tfb-k9, tfb-k10 or your current two final picks.

## The hypothesis

Existing production `consolidation/tfb/stages/t2_occ.py:decode` selects independent
cells from P6 by maximizing `sum(top p) / (m + sum(rest p))`. However true queue
cells are correlated spatially and over time. This new *post-processing* arm:

1. Collects observed TRAIN queue trajectories from days STRICTLY earlier than
   the evaluated origin and outside its held-out training fold.
2. Translates candidate trajectory templates within the same directional panel
   to 3 most likely anchor links (zero padded, no wrapping).
3. Retains 64 most plausible candidates by marginal-probability similarity.
4. Builds a probability-weighted posterior over the candidate whole trajectories.
5. Selects the candidate with best posterior **expected full-window IoU**.
6. Keeps the exact baseline top-m output unless the posterior expected improvement
   exceeds 0.005; never reads current window's post-T truth at prediction time.

It does NOT claim calibrated posterior probabilities or proof of physical correctness.
The 0.02 temperature, 3 anchors, 128 bank templates and 0.005 decision margin
are **hypothesis settings**, not tuned or proven optimal. Do NOT choose these
values using April labels or April leaderboard results.

## Locally completed

- 6 unit/synthetic tests passed for the decoder, zero-padded translation, baseline
  equivalence, no-horizon-truth dependency and past-day/fold bank exclusion.
- The actual OOF runner passed a 64-window / two-panel synthetic integration test.
- Two-panel synthetic fixture correctly FAILED the >=8-panel research gate.
- 0 actual TrafficFlowBench train windows evaluated; 0 Kaggle submissions made.

## Important: which score is comparable?

Your reported A3/B3 public composite is **0.88281**, vs screenshot top **0.93124**.
A strict win needs **0.93125**, gap **+0.04844**. If only S_queue changed, it
would have to rise by +0.1614667. For a *single* queue condition, the total-score
coefficient is 0.15, so this experiment's measured `delta_condition_iou`
corresponds to `0.15 * delta_condition_iou` only if it generalizes and nothing
else changes. The official public April hidden scores cannot be inferred here.

## Real OOF experiment (requires your existing competition data and caches)

Unzip the package into any folder. Place the three .py files wherever you like;
the script reads only existing TRAIN lab caches and does NOT import the macOS
absolute-path-bound `lab.py` file.

From the *root of your own repository*, assuming the cache exists:

```bash
python /path/to/run_oof_experiment.py \
  --cache-root data/t2local/cache \
  --arm A_np255 \
  --set offlike \
  --bank-set faith3 \
  --condition queue_ongoing \
  --output analysis/t2lab/JOINT_OFFLIKE.json
```

Required cached files for **every panel**:

```
data/t2local/cache/oof/<ARM>/probs__offlike__<PANEL>.npz
     keys P6, done, d, t (original OOF probability output)
data/t2local/cache/rows/<PANEL>__offlike__queue_ongoing/
     aux.npz with truth, ev; win.pkl with d,t,fold
data/t2local/cache/rows/<PANEL>__faith3__queue_ongoing/
     aux.npz with truth, ev; win.pkl with d,t,fold
```

The `A_np255` example is an **old lab baseline**, *not the current k9/k10
production fs4occ bag*. To establish incremental value against the actual
current submission, first export equivalent OOF marginal P6 from **the current
production bag** (same folds, same eligibility, same decoder), and pass that
arm name. Any win versus `A_np255` alone is insufficient to promote the arm.

The runner outputs JSON plus `<output>.windows.csv`, including:
- exact eligible-cell per-window space-time IoU (baseline and alternative),
- panel/family-averaged IoU and per-panel deltas,
- day-block bootstrap confidence interval,
- changed-window count, a pre-specified diagnostic gate, and the deployment gate.

**Research gate (not an automatic Kaggle submit)**:
`>=8 panels`, at least `7` positive panels, aggregated condition IoU
`>=+0.01`, day-block bootstrap `95% lower > 0`; all on TRAIN OOF only.
This would correspond to at most an estimated +0.0015 composite from the
one condition before domain-shift uncertainty, far short of +0.04844.

## Critical non-leakage limits

- The TRAIN template bank filters `bank_day < eval_day` and `bank_fold != eval_fold`.
- The existing `analysis/t2lab/lab.py:run_arm` OOF model itself may be fitted
  using *future TRAIN folds relative to the simulated OOF origin*. Thus existing
  cache OOF scores **do not prove full end-to-end online causality**.
- The code is a decoder lab and has not been integrated into production.
- For award eligibility, repeat model training chronologically; use only train
  histories known before origin T; run the organizer's **post-T truncation replay**
  to ensure *exact queue predictions* are unchanged.
- Do NOT train, tune, select, filter or choose any parameters using April
  labels, April leaderboard outputs or its post-T observations.
- Do NOT train this decoder on March outputs when backtesting earlier March
  windows. Train labels may be used for March; March history available before
  an April origin may be used for April.
- `truth` stored in current lab RowCache is observed eligible proxy truth,
  NOT the hidden exact official truth. Synthetic tests are solely software tests.

## Recommended practical decision

Keep k9/k10 frozen. Run one train-only diagnostic against *current production*
OOF P6 if those caches can be provided before the deadline. If it fails, do
not spend a Kaggle submission. If it passes, do a chronological replay and
March-style transfer check **before** building any submission. The available
artifacts alone cannot certify that this wins against 0.93124.

## Smoke tests

```bash
python test_joint_decoder.py
python smoke_integration.py
```

Requires `numpy`, `pandas`; no Kaggle credentials and no external networking.
