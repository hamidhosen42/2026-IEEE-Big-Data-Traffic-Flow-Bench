# t2lab: local Task 2 lab (queue forecast)

Local experiment harness on the real Task 2 panels exported to `data/t2local/t2data_<PANEL>.npz`
(8 panels, 273 train days each). Everything here is new code under `analysis/t2lab/`; no project file is modified.

| file | purpose |
|---|---|
| `lab.py` | importable library (below) |
| `run_heavy.py` | 3-slot semaphore, required for every training run |
| `replay_selector.py` | selector reproduction table (official train windows vs rule variants) |
| `prep_cache.py` | builds window sets + row caches for all panels (about 2 min in total) |
| `baseline_A.py` | arm A = production ongoing recipe at local scale |
| `parity_float16.py` | measures the only lossy step of the export (speed stored as float16) |
| `review_parity.py` | independent review: feature parity with a second compile of the production functions, online rule, fold safety |
| `review_selector.py` | independent review: plain-loop selector replay vs `replay_table.csv`, `fill_time` check |
| `review_scoring.py` | independent review: aggregation / decode by hand, organizer `score_window`, bootstrap sanity |
| `review_sets.py`, `review_data.py` | independent review: window-set facts, run_arm plumbing smoke (both conditions), export facts |
| `logs/` | run logs (`review_*.log` = the review runs) |

## Production code is reused, not rewritten

`lab.py` parses `kaggle_jobs/t2_model_v20.py` with `ast`, keeps only the top-level function definitions and the
constants `H, F, SPACING, PARAMS, PRIOR_COLS, ID_DROP, PANELS`, and executes just those (the Kaggle main code never runs).
They are available as `lab.prod.<name>` and re-exported: `window_features, add_ramp, prior_table, prior_feats, make_rows,
to_matrix, ongoing_keep, decode, iou, established, build_windows`. `lab.selftest(panel)` checks:

* the functions are compiled from the production file itself (line numbers match) and no main-code global exists;
* `cut == 0.6 * vf`, `E == isfinite(V)`, `Q == E & (V <= cut)`;
* the official `history_coverage` column equals `E[history].mean()` (max abs difference 4.4e-5 = 4-decimal rounding);
* the vectorised selector with the production rule returns exactly `build_windows(global_spacing=True)`;
* cached rows + fold prior + keep mask are bit-identical to `to_matrix(make_rows(P, wins, prior_k), cond, None, ongoing_keep(R))`.

Run: `.venv/bin/python analysis/t2lab/lab.py D7_I210_W`.

## The FAITHFUL selector rule

Measured by `replay_selector.py` (greedy replay from 2030-06-01 with the 5 + 5 quota, all 8 panels,
40 official onset + 40 official ongoing train origins; see `logs/replay_selector.log`):

| filled state | persistence state at T | IoU cells | onset exact | ongoing exact | ongoing within 5 min |
|---|---|---|---:|---:|---:|
| any | observed (missing = free) = production `build_windows` | observed | 39 | 14 | 16 |
| `hb` (time 3, space 2, time 12) | filled | observed (E) | 39 | 34 | 38 |
| `hb` | filled | all filled cells | 39 | 32 | 36 |
| `t3` (time 3) | filled | E / all | 39 | 25 / 25 | 32 / 33 |
| `t12` (time 12) | filled | E / all | 39 | 33 / 35 | 38 / 38 |
| **`t36` (time 36)** | **filled** | E / **all** | 39 | 34 / **36** | 39 / **39** |

Scan order (production two-pass "onset first" versus one interleaved chronological scan) gives identical picks in every
row: the official train windows cannot distinguish the two.

**FAITHFUL = `lab.RULE_FAITHFUL`** = `dict(pers_state='fill', cells='all', fill='t36', pers_max=0.9)` with `FAITHFUL_SCAN = 'interleaved'`:

* candidate origins: slots 12..281 of a day, history coverage `E[T-12:T].mean() >= 0.7`;
* condition from the visible history only: ongoing iff some link has >= 2 queued observations in `[T-60, T)`, else onset;
* onset: an observed queued cell in `T+1..T+6` (using the filled state here makes onset reproduction worse: 28/40);
* ongoing: a queued cell of the filled state in the horizon and
  `IoU(QF[T+1..T+6], QF[T] repeated) <= 0.9` over all filled cells, where `QF = VF <= cut` and `VF` = speed linearly
  interpolated along time within the day for gaps up to 36 slots (no spatial fill);
* greedy: one chronological scan, a pick needs >= 72 slots (360 min) to every earlier pick, 5 per condition.

Why: every link is score-eligible and `future_coverage` is 1.0 in the window index, so the organizer evaluates the
persistence cap on the full true state; 23.6% of train cells are simply unobserved. Reading missing cells at T as free
(production) lets stable queues through. `lab.RULE_SANDBOX` (`hb`, E cells) is the earlier one-panel sandbox rule,
`lab.RULE_PROD` the production behaviour. The remaining 4 misses are 1-2 slot shifts (threshold noise of the proxy).

The filled state is for window selection and diagnostics only (it uses future values). It is never a feature.

## Window sets: `lab.window_set(panel, name)`

DataFrame `d, t, cond, fold, pers_obs, pers_fill, faithful, runstart`; `fold = min(d * 4 // D, 3)` (production blocks).

| name | content |
|---|---|
| `dense<cap>` | (a) production training windows `build_windows(global_spacing=False, dense_ongoing=True)`, ongoing subsampled to `<cap>` with `rng(0)` as production (`dense1500` cached; `dense3000` = Kaggle setting, built on first use) |
| `faith`, `faith3` | (b) every faithful ongoing candidate; `faith3` = those on the production `t % 3 == 0` grid |
| `runstart` | (b) faithful run-starts: candidate at `t` whose `t-1` is not a candidate |
| `offlike` | (c) official-like evaluation windows: faithful greedy restarted at 00:00 of every train day (5 + 5 quota, 360 min), de-duplicated; 2,992 ongoing windows |
| `pseudo` | (d) current-style pseudo windows `build_windows(global_spacing=True)`; 2,058 ongoing windows |
| `official` | (e) the 40 + 40 official train windows (all in fold 0) |

`lab.register_window_set(panel, name, df)` stores a custom list (columns `d, t, cond`) under a new name.

## Rows: `rc = lab.rows(panel, name, cond='queue_ongoing')`

Built once with production `make_rows` / `to_matrix` and cached. `rc.X` (memmap, all production columns, prior columns
holding the all-train prior), `rc.cols, rc.y, rc.e, rc.wi, rc.li, rc.h, rc.fold, rc.win`, `rc.prior5[n, k, 3]`
(`prior, prior_now, prior_alldays` from `prior_table(P, block != k)`, `k = 4` = all days), `rc.keep5[n, k]`
(production `ongoing_keep` under prior `k`), `rc.truth / rc.ev [W, 6, L]`, `rc.qlast [W, L]`.
`rc.matrix(mask, cols, k)` returns the matrix with fold-safe priors substituted. Column sets: `lab.colset('np' | 'id' | 'full')`.

## Training, scoring

* `lab.train_matrix(train, k, cols, weight_fn=...)`: rows of windows outside fold `k`; `train` is a set name or
  `[(set, weight), ...]`; `weight_fn(win_df, set_name, panel)` returns per-window weights (0 drops the window).
* `lab.run_arm(arm, train, evals, cols, params, n_trees, folds, weight_fn, seeds)`: OOF LightGBM arm, returns one row per
  held-out window (`iou`, per-step `s1..s6`, error split, selection flags) and saves results and probabilities.
* `lab.predict_set(models, rc, k, cols)` + `lab.score_probs(rc, P6)`: score any probabilities `[W, 6, L]` with the
  production decode (expected-IoU top-m over the 6 x L cells; cells outside the keep region are 0).
* `lab.aggregate(df)`: window -> panel -> family -> mean for one condition; `lab.s_queue(df)` for both conditions.
* `lab.paired_bootstrap(dfA, dfB)`: B - A, windows resampled within panel (each frame must hold ONE evaluation set and one
  row per window, otherwise it raises); `lab.step_profile`, `lab.error_split`, `lab.summary`.
* `lab.split_rows(P, 'validation' | 'private')`: production-shaped rows of the released windows (features <= T only).

## Cache layout (`data/t2local/cache/`)

```
fill_<PANEL>_<kind>.npz                      filled speed cube (hb, t36)
windows/<PANEL>__<set>.pkl                   window DataFrames (denseall = uncapped production list)
rows/<PANEL>__<set>__<cond>/                 X.npy, aux.npz, win.pkl, meta.json   (about 9 GB with faith3)
oof/<arm>/results.pkl, config.json, probs__<set>__<PANEL>.npz   (P6 float16, done mask, d, t)
replay_table.csv
```

Caches are keyed by name only: after changing `RULE_FAITHFUL` or the production file, delete `windows/` and `rows/`.

## Example (save as `my_arm.py`, run through the semaphore)

```python
import lab, numpy as np
def wfn(win, set_name, panel):                      # per-window weights
    if set_name == 'dense1500':
        return np.ones(len(win))
    return np.where(win.runstart, 1.0, 0.2)         # faithful windows: run-starts 1, the rest 0.2
df = lab.run_arm('B_dense_plus_faith', train=[('dense1500', 1.0), ('faith3', 1.0)],
                 evals=('official', 'offlike', 'pseudo'), cols='np',
                 params=lab.lgb_params(num_leaves=255, learning_rate=0.05, min_data_in_leaf=40),
                 n_trees=500, folds=(0, 1), weight_fn=wfn)
print(lab.summary(df).round(4))
base = lab.load_arm('A_np255'); base = base[base.fold.isin([0, 1])]
for s in ('official', 'offlike', 'pseudo'):
    print(s, lab.paired_bootstrap(base[base.set == s], df[df.set == s]))
```

```
PY=".venv/bin/python"; "$PY" analysis/t2lab/run_heavy.py -- "$PY" analysis/t2lab/my_arm.py > analysis/t2lab/logs/my_arm.log 2>&1
```

Note: `faith3` overlaps `dense1500` (about half of the dense windows are faithful) and windows of an evaluation set that
fall in a training fold may also be training windows; held-out folds are always clean because folds are day blocks.

## Baseline (arm `A_np255`, OOF, ongoing; `logs/baseline_A_np255.log`)

Prior-free `np` columns (58, ramp features included), 255 leaves, lr 0.05, min_data_in_leaf 40, 500 trees, `dense1500`,
about 3.3 M training rows and 65 s per fold on 4 threads.

| set | n | IoU (aggregated) | T+5 | T+10 | T+15 | T+20 | T+25 | T+30 | missed new | kept cleared | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official 40 | 40 | 0.8525 | 0.935 | 0.904 | 0.891 | 0.913 | 0.831 | 0.681 | 0.072 | 0.056 | 0.020 |
| official-like (`offlike`) | 2992 | 0.8387 | 0.937 | 0.910 | 0.876 | 0.855 | 0.831 | 0.699 | 0.056 | 0.064 | 0.041 |
| current-style (`pseudo`) | 2058 | 0.8780 (plain mean 0.9225) | 0.956 | 0.939 | 0.867 | 0.846 | 0.849 | 0.816 | 0.023 | 0.074 | 0.025 |
| persistence (qlast), official / offlike / pseudo | | 0.7501 / 0.7212 / 0.8112 | | | | | | | | | |

Inside `pseudo`: faithful subset 0.747 (n 738), stable subset (filled persistence > 0.9) 0.966 (n 1320).
`D7_I10_W` has only 4 pseudo ongoing windows (0.625), which is why the aggregated pseudo number sits below the plain mean.

## Independent review (2026-10-03, `logs/review_*.log`)

All checks passed; one defect was fixed (`paired_bootstrap` silently merged duplicate windows when a frame held several
evaluation sets: 1659 "windows" from 1291 rows; it now raises on multi-set or duplicate-key frames).

* Feature parity (`review_parity.py`, D7_I10_W and D12_I5_S, 76 checks): the production functions compiled a second way
  (source truncated before the main block, real imports) give bit-identical matrices, labels, eligibility flags and
  (window, link, step) meta to the row cache for `offlike`, `official` and `dense1500` under fold priors 0, 2 and the
  all-train prior; `colset('np'|'id')` are pure column subsets (58 / 64 of 69).
* Online rule (perturbation per window): no feature changes when data after T, data before T-60, or ramp flows at T or
  later are altered; ramp flows in T-3..T-1 reach only `ramp_*`; ramp flows in T-12..T-4 reach only `ramp_on_surge`
  (the surge baseline uses [T-60, T-15), still strictly before T); the slot-T value reaches only the masked-slot
  features (`vr_T, has_T, vr_last, q_last` and their neighbour / distance derivatives). No selection-side column exists in X.
* Fold safety: `prior_table(block != k)` is unchanged when fold-k days are zeroed; `train_matrix(k)` returns exactly the
  rows with `fold != k`; train and eval day sets are disjoint; `predict_set(k)` fills only fold-k windows and exactly the
  `keep5[:, k]` region.
* Selector (`review_selector.py`, D7_I10_W, D12_I5_S, D7_I210_E): a plain-loop implementation of the stats, candidates
  and both greedy scans reproduces `replay_table.csv` exactly (per panel, rule and scan) and `lab.greedy` picks are
  identical in every case; `selector_stats` equals per-slot slicing for cov, est, horizon flags and all three persistence
  definitions. In these 3 panels t36/E scores 13 and t36/all 12 ongoing exact matches (8-panel totals 34 vs 36: the
  two rules differ on D12_I5_N (2 vs 5) and D7_I210_E (5 vs 4) only), so the E/all choice remains weakly supported.
* Scoring (`review_scoring.py`): `aggregate` and `s_queue` match hand-computed values on an unbalanced toy frame;
  `decode` equals the brute-force maximiser of the expected-IoU proxy on 50 random 10-cell cases; `score_probs` equals
  the organizer's `repo/src/task2/score_task2.score_window` and the cached `results.pkl` on 60 windows of arm A_np255
  (max abs diff 0, from the float16-stored probabilities); per-step and error-split identities hold; bootstrap of
  B = A + 0.01 gives diff = lo = hi = 0.01 and diff always equals `aggregate(B) - aggregate(A)`.
* Baseline (`logs/review_baseline_f0.log`): a fresh fold-0 run of arm A reproduces 0.8525 / 0.8385 / 0.9030
  (official / offlike / pseudo) and the identical per-panel official-40 table: the pipeline is deterministic.
* Sets (`review_sets.py`): no duplicate windows in any set; 36 of the 40 official ongoing origins are inside `offlike`
  (7 inside `pseudo`); 0 offlike and 1 official ongoing window have filled persistence > 0.9; 30 of 40 official ongoing
  windows are run-starts; `run_arm` works for `queue_onset` as well (plumbing smoke only, no onset baseline measured).

## Caveats

* The export stores the ramp cubes (`train_ON / train_OFF`, validation / private ON / OFF) as float16 too (flows in vph,
  resolution 2-4 vph above 2048 vph); the lab casts them to float32. Effect on ramp features not measured.
* Real released windows: history coverage 0.755-0.770 (train 0.764) and slot-T visible share 0.44-0.59 per panel
  (validation mean 0.52, private mean 0.52) against 0.47 (= 0.62 x 0.764) in the train-side simulation; `has_T` share on
  `split_rows('validation')` ongoing rows is 0.54 vs 0.49 locally (D7_I10_W).

* Truth proxy = observed queue state on observed cells (76.4% of cells); the real scorer uses the noise-free state on all cells.
* Speeds are float16 in the export: on D12_I405_S (raw files local) 10 of 2.58 M observed cells change queue state
  (3.9e-6) and the pseudo-window list is unchanged. Raw train files of the Task 2 panels are not on this machine, so the
  loader is not compared byte-for-byte with the Kaggle loader on those panels.
* Slot-T visibility is simulated exactly as production (`rng(len(wins))`, 62% visible), so it depends on the window list.
* Onset replay is 39/40 locally (D12_I5_S: official 14:35, replay 14:45), not 40/40.
