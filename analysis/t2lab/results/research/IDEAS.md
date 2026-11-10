# Research ideas for Task 2 (and one for Task 1), 2026-10-04

Written by the research agent. Read-only work: no model was trained and no project file was changed except this one.
Base: public LB 0.87961 (tfb-e5), onset about 0.765, ongoing about 0.85.

## Headline

**Our prior columns include the training row's own day, and all three are tied to one 5-minute slot** (two of them
also to the day class). Three public pipelines build these frequencies leave-one-day-out or out-of-fold; ours does not.
The effect is a hard rule the model learns for free: "prior is exactly 0, so no queue". It is true for every
training row by construction and false in the new scenario.

| Fact (read-only diagnostics on `data/t2local`, labels only from train or from released March cells) | Value |
|---|---|
| Train onset rows: positive cells with slot prior = 0 | 0% in every panel (by construction) |
| Train onset rows: negative cells with slot prior = 0 | 75% to 99.7% per panel |
| Same positives under a leave-one-day-out prior | 0.9% to 9.1% per panel have prior 0 (136 cells) |
| March pseudo-onset positive cells whose train slot prior is 0 | D7_I10_W 88%, D7_I405_S 41%, others 0 to 4% |
| Official validation onset windows where the train slot prior at T+30 is 0 on every link | 4 of 40, all D7_I10_W (03-01 13:25, 03-02 13:35, 03-03 05:30, 03-04 05:25) |
| Official onset windows with max slot prior < 0.05 / < 0.2 | validation 6 / 11, private 2 / 10, official train 0 / 6 |
| Saved lab arm (production-like onset, no ramp), 214 pseudo-March windows | "blind" windows (max slot prior < 0.05; 19, of which 18 on D7_I10_W): IoU 0.41, 53% score exactly 0; other windows 0.77 |
| Same, blind windows in March days 1-5 (public-like, no same-scenario days yet) | 0.00 for base and for s8 (n = 2) |
| Queued cell-observations sitting in zero-train-prior cells (matters for a prior-using ongoing model) | March: D7_I10_W 13.5%, D12_I5_S 11.3%, D7_I405_S 8.4%, D7_I10_E 7.2%. April: D7_I10_W 18.8%, D7_I405_S 17.3%, D7_I10_E 10.7%, D7_I210_E 9.4%, D12_I5_S 9.2%. Train with self-inclusion: 0% |

The March onsets on the failing panels are at the **same bottleneck links as train, at other times**:

- D7_I10_W: train onsets sit at links 24-26 (link 26 is in 81% of the 134 train onset events). The queue starts at
  14:35-15:15 in 123 of them (earliest 14:35) and never before 07:00. In March the same links start queueing at
  13:55-14:25 (origins 13:25-13:55), 15 to 45 minutes earlier than train ever saw, and on weekdays also near 05:55
  (origins 05:20-05:35). Official origins: validation 03-01 13:25, 03-02 13:35, 03-03 05:30, 03-04 05:25; private
  04-07 05:20.
- D12_I5_N: March onsets are at links 103-105 (train onset-site share 0.57; the second train site, 236-238, has 0.25).
  The slot prior ranks another link first in 23% of the March events.
- D7_I405_S: March uses site 32-39 in 28 of 28 events and site 11-15 in 3; in train both are frequent (0.45 and 0.39).
- D12_I5_S: the main train site 43-45 (0.48-0.54) never breaks first in March; links 10-15 and 89 do. A site prior
  alone fails here.

"Is the top-ranked candidate link inside the true first block?" on March pseudo-onset events (candidates = links in
more than 2% of train onset events, visible labels only, ties broken at random):

| Panel | n | slot prior (ours) | time-free site share | prior max over ±60 min, both day classes | flow / link's own pre-onset flow |
|---|---:|---:|---:|---:|---:|
| D7_I10_W | 18 | 0.32 | 1.00 | 0.78 | 0.17 |
| D7_I405_S | 28 | 0.57 | 1.00 | 0.61 | 0.96 |
| D12_I5_N | 22 | 0.77 | 0.96 | 0.41 | 0.96 |
| D12_I5_S | 26 | 0.58 | 0.04 | 0.54 | 0.73 |
| D7_I210_W | 38 | 0.90 | 0.40 | 0.64 | 0.50 |
| D7_I10_E, D7_I210_E, D7_I405_N | 55 | 1.00 | 1.00 | 1.00 | 1.00 |
| Blind windows only | 18 | 0.37 | 0.94 | 0.78 | 0.17 |

This is a ranking check among candidate links, not an IoU; it overstates absolute hit rates.
No single score wins everywhere, so these belong in the model as features, with honest priors, not as a rule.

Scripts and outputs of these diagnostics are in the session scratchpad
`/private/tmp/claude-501/-Users-md-hamidhosen-Documents-Kaggle-2026-IEEE-Big-Data-Traffic-Flow-Bench/b036a5d8-355a-4104-b35c-e5f21802f7bf/scratchpad/`
(`diag_lodo.py`, `diag_site.py`, `diag_flow2.py`, `diag_offprior.py`, `diag_blind_iou.py`, `diag_events_w.py`,
`diag_og_prior0.py`, `diag_rsreason.py`, `diag_rsreason2.py`, `diag_gap.py`). They only read the npz files and saved results.

Legality: every proposed feature is built from the train split and from data timestamped at or before T. The March
event lists above are decision support, as in the existing pseudo-March gate.

---

## Ideas, ranked (1 = best expected gain × credibility)

### 1. Leave-one-day-out (or out-of-fold) queue priors in every training row that carries a prior

- **Source.**
  - `notebooks/0.85_LB__yeashusemwal__trees-unet-cnn.ipynb`: "Profiles that include the target day leak ...
    Rebuilding them leave-one-day-out made local scores drop and leaderboard scores rise".
  - happyc0der/trafficflowbench `NOTES.md` (public LB 0.845): Task 2 v1 uses a "leave-one-day-out queue frequency".
  - Hisernberg `hb_TASK2_ANALYSIS.md` section 4: "models and time-of-day profiles for a fold never see its days";
    training rows carry fold-excluded profiles, released windows the full-train profile.
- **Reported effect.** yeashusemwal: public LB +0.008 from Task 2 and +0.004 from Task 1. The other two give no ablation.
- **Already tested by us?** No.
  - `prior_table(P, np.ones(D))` feeds the training rows in `t2_onset_sp.py` (line 296), `t2_model_v22.py` (line 425)
    and `t2_onset_pv.py`; CV uses `prior_table(P, block != k)`, which still contains each training row's own day.
  - EXP_T2_m says "fold-safe priors": that covers evaluation rows only.
  - EXP_T2_oprior used a parity split, for a different prior.
  - EXP_T2_j, EXP_T2_k and EXP_T2_OGSP dropped the priors instead of fixing them. EXP_T2_OGSP measured the full
    ongoing model with the train prior at −0.08 on pseudo-March; in-distribution the full model was better
    (0.916 vs 0.908, EXP_T2_j).
- **Experiment (about 1 h).**
  - Per row (day d, slot s, link l, class c):
    `prior = (Qsum[c][s, l] − Q[d, s, l]) / max(Esum[c][s, l] − E[d, s, l], 1)`, sums over the days allowed for that
    fit. Same for `prior_now` (slot t) and `prior_alldays` (all classes). Row keys: `rc.win.d/t[rc.wi]`, `rc.li`, `rc.h`.
  - **A. Onset.** New `exp_onset_lodo.py` from `exp_onset_splitprior.py` with `NORAMP=1`.
    - Arms: A0 = saved `results/splitprior/cat_onset_noramp.cbm`; L1 = leave-one-day-out; L7 = leave-one-week-out
      (`(d // 7) % 4`, guards against the two-value artefact of pure leave-one-out). Same CatBoost parameters.
    - Evaluation 1: the 214 pseudo-March onset windows with base, s4 and s8 priors; table all / d1-5 / d6-15 / d16+,
      per panel, and split by blind (max train slot prior < 0.05) and low (< 0.2); `lab.paired_bootstrap` against A0.
    - Evaluation 2 (train labels): fold-0 hold-out as in `exp_onset_wrongscen.py` (522 official-like onset windows),
      fit on folds 1-3 with self-inclusive versus leave-one-out priors, fold-safe prior at evaluation; also report the
      windows whose fold-safe max slot prior is < 0.2. The `early` track already has the self-inclusive fold model and
      the bundles (`early_train_f0.py`, `early_lib.train_bundle` / `march_bundle`); only the three columns change.
    - Keep if pseudo-March (base prior) gains at least +0.02 with a CI above 0, D7_I10_W and D7_I405_S both rise,
      and the fold-0 hold-out loses no more than 0.003.
  - **B. Ongoing full member.** F03 rows (`exp_firstcross_lib`), `cols='full'`, 255 leaves, 500 trees.
    - Arms: full with the current prior versus full with leave-one-day-out priors.
    - Evaluation: official-like OOF, and pseudo-March `offlike_m` / `runstart_m` through `exp_ongoing_splitprior.py`
      (train prior and s4): np+id, full alone, np+id+full, 0.5·npid + 0.5·full.
    - Keep if np+id+full beats np+id by at least +0.005 on both March sets (6 of 8 panels) with no OOF loss.
  - Optional arm in A: panel-balanced row weights (D7_I10_W holds 3% of positive cells and 1/8 of the score).

### 2. Time-free onset-site share and time-tolerant priors for onset

- **Source.** Hisernberg sections 5 and 11: the onset prior `oprior` has ±60 min, ±90 min, weekday-class and **all-day**
  variants; "The location prior helps on the non-recurrent onsets too, because it includes all-day and weekday-class
  variants". Section 19: the March misses "cluster in early-morning windows (04:40–05:40: D7_I10_W 003/005 ...)".
- **Reported effect.** Hisernberg CV: onset 0.709 → 0.713; non-recurrent onset slice 0.240 → 0.265-0.275. No LB ablation.
- **Already tested by us?** No. Our three prior columns are all specific to one slot (two also to the day class).
  EXP_T2_oprior tested only a ±60 min, class-specific onset prior with a parity split (−0.003 to −0.016). No all-day or
  tolerant variant was tried.
- **Features** (out-of-fold as in idea 1):
  - `site_on`: share of other days' onset events (first observed queue at T+30) that contain the link, all times;
  - `prior_tol12`, `prior_tol6`: maximum of the slot prior over ±12 / ±6 slots and both day classes;
  - `prior_day`: share of days with any queued observation on the link;
  - distance in links to the nearest link with `site_on` ≥ 0.1, and the rank of `site_on` in the corridor.
- **Experiment (about 1 h, same harness as idea 1).**
  - Arms: L1, L1 + site, L1 + site + tolerant.
  - Extra arm without retraining: production probabilities, but in blind windows the decoded set is replaced by the
    top `site_on` link(s). Blindness needs only the train prior and T.
  - The `early` track already scans wider same-scenario boxes (±4 to ±24 slots) at inference with the unchanged
    model. Rerun its best design (M8) on the retrained model to see whether the two gains add.
  - Keep if the 19 blind windows gain at least +0.15 IoU, all windows gain at least +0.02 over L1, and no panel loses
    more than 0.03 (watch D12_I5_S and D7_I210_W).
  - A clean LB probe afterwards changes only the validation windows with max slot prior < 0.05 (6 windows).

### 3. Shift the clock of the priors in training (prior-timing jitter), or drop them at random

- **Source.** Our diagnostics above, plus Hisernberg section 11 ("queues at links 24-26, the train bottleneck, but at
  weekdays and times when train almost never queues there"). The lab track `pvon` emulates a shift along the links; the
  observed March shift is mostly in time and day class.
- **Reported effect.** None published; this is our own proposal.
- **Already tested by us?** No. EXP_T2_OSP_IDCOLS only dropped tod / pos / dow for every row (slightly worse).
- **Experiment (about 1 h, on top of idea 2).**
  - One augmented copy per train onset window: δ drawn from ±6 to ±18 slots, day class flipped with probability 0.25.
    Look up the three slot priors and the tolerant priors at slot + δ and that class, set `tod += δ` and `dow` to a day
    of that class. State features and labels stay. Weight of the copies 0.5 and 1.0.
  - Variant: for 30% of the windows set the three slot priors to 0 and `tod` to NaN.
  - Evaluation: pseudo-March (all, blind, low, per panel), fold-0 train hold-out, and the wrong-scenario set of
    `exp_onset_wrongscen.py`.
  - Keep if pseudo-March gains at least +0.01 over the idea-2 model and the train hold-out loses less than 0.01.
  - Use it for the March rows and the train-only hedge file; the same-scenario April file has a reliable prior.

### 4. Day-so-far "episode clock" features for the ongoing np / id models

- **Source.** happyc0der `NOTES.md` (2026-09-22): `run_q, since_first_q, frac_q_today, min_r_today` per link and
  `corr_age, nq_peak, nq_rel_peak, since_peak, nq_d6/12/24` per corridor, from the masked layer at slots ≤ T.
  yeashusemwal: "Three hours of same-day context".
- **Reported effect.** happyc0der: ongoing CV 0.9114 → 0.9155 (+0.0041 ± 0.0006; `run_q` gives a quarter of it);
  organizer ongoing windows 0.8810 → 0.8910 with 3 seeds. yeashusemwal: public LB +0.0004.
- **Already tested by us?** Only in another setting. EXP_T2_e added day context to the full model (which had tod and
  priors) on the old window population: ≈ 0, rejected. The production np and id models have no clock at all (both drop
  tod and dow; `q_count` stops at 60 min). The `ogfeat` track uses the 60-minute history only.
- **Why now.** F03 np255 out-of-fold by window type (official-like, 2,992 windows):

  | Window type | Share | IoU |
  |---|---:|---:|
  | Run-start of a queue that was already established at T−5 | 53% | 0.941 |
  | Mid-run window (not a run-start; picked when the 360-min spacing ends) | 24% | 0.811 |
  | Run-start of a newly established queue (median 3 queued links in the history) | 23% | 0.781 |

  Mid-run windows are the ones that need to know how old the queue is and whether the peak has passed.
- **Experiment (about 1.5 h).**
  - Features from slots < T−12 of the same day: train side on an emulated masked layer (visible share 0.5 / 0.7 / 0.8
    of eligible cells, as `pvon_lib`), released windows from `validation_Vm` / `private_Vm`.
  - Add them to F03 np255 and id255 the way `ogfeat` adds columns.
  - Evaluation: official and official-like OOF by window type, pseudo-March `offlike_m` and `runstart_m`.
  - Keep if official-like gains at least +0.004 with a CI above 0, mid-run windows gain at least +0.01, and
    pseudo-March is not negative.

### 5. Flow relative to each site's own pre-breakdown flow (onset physics)

- **Source.** Breakdown-probability literature: product-limit and Weibull breakdown curves as a function of flow
  (Brilon, Geistefeldt and Regler; Berkeley ITS, breakdown probability in speed and occupancy). Hisernberg section 12
  `ph_*` features. Release config: an hour before a queue the road "carries 6912 vph against 4620".
- **Reported effect.** Hisernberg v4 → v5: onset CV +0.008, non-recurrent slice +0.018 to +0.020, March LB onset +0.012
  (inside LB noise).
- **Already tested by us?** Generic versions only: EXP_T2_H (hazard and gradient features, noise). EXP_NEW_034
  (breakdown probability) was proposed and never run. New here: the denominator is the link's own median flow 30-45 min
  before its train onsets, which makes sites comparable and does not depend on the clock.
- **Evidence.** Top-1 table above: 0.73 on D12_I5_S (slot prior 0.58), 0.96 on D7_I405_S (0.57), but 0.17 on D7_I10_W.
- **Experiment (about 45 min, on the idea-2 model).** Features: the ratio for the link and for the mean of the 3 links
  on each side, its 15-minute slope, its rank in the window; medians out-of-fold. Keep if D12_I5_S and D7_I405_S gain
  at least +0.05 on pseudo-March with no loss overall.

### 6. Task 1: transductive blackout rows

- **Source.** `hb_EXPERIMENTS.md`, "Test-month blackout pseudo-holdout" (members fullPD, fullPD2).
- **Reported effect.** Blackout speed RMSE 7.33 / 6.89 → 6.42 / 5.91 (March / April), flow per lane 81.2 / 72.0 →
  76.1 / 63.5; mean of two members 6.27 / 5.75. Expected S_total +0.0004 to +0.0006. Hisernberg published no LB result.
- **Already tested by us?** No. `t1_pipeline_v13a.py` darkens train days and adds i.i.d. hidden cells of the target
  months (EXP_T1_trans); no simulated blackout inside the target months. EXP_NEW_007 was proposed, never run.
  Blackout targets are about 1.9% of targets but carry roughly a quarter of the squared speed error (estimate from
  Hisernberg's RMSE of 6.9 km/h on blackout cells against 1.5 on regular cells).
- **Experiment.** Kaggle T1 kernel variant (a first read is possible in `analysis/t1lab`).
  - Draw queue-like origins in the validation and private masked months, at least 18 slots from released blackouts.
  - Blank T+1..T+18 for all stations and use the released cells inside as training rows.
  - Hold out half of the simulated blackouts. Keep if blackout speed RMSE falls by 8% or more, then probe on the LB.

### 7. Released inputs at or before T that no model uses yet

- **Source.** Organizer ruling 742068; Hisernberg section 5 ("the origin slot T (ratio, flow, neighbours ...)").
- **Reported effect.** Hisernberg: slot-T group "small but positive in the model"; no separate number for flow.
- **Already tested by us?** No. Slot T enters as speed only (`VT`); `FT` is exported and unused. Ramp flows use
  T−15..T−5 only. Occupancy is used nowhere in Task 2 and is not in the local export. Occupancy is an independent
  density reading: log(k_occ / (q/v)) has sd 0.09 in free flow (D12_I405_S, 10 days).
- **Experiment (about 45 min for the first two).** Add flow / capacity at T (with a visibility flag) and ramp flows at
  T to F03 np/id and to the onset model; evaluate as in ideas 1 and 4. Occupancy needs an export first (extend
  `export_t2x.py`). Expected gain is small (a few thousandths of IoU). Discussion 745700 (is data at T allowed?) is
  unanswered, so keep the no-T variant alive.

### 8. A corridor-level network for onset, not for ongoing

- **Source.** yeashusemwal notebook; happyc0der `NOTES.md` (2026-09-25).
- **Reported effect.** yeashusemwal: CNN for onset +0.0019 public LB, CNN for ongoing −0.0049. happyc0der: CNN for
  ongoing 0.8997 vs LightGBM 0.9230, blends negative.
- **Already tested by us?** No onset network. The `cnn` lab track builds an ongoing member, which two public sources
  report as a loss.
- **Experiment (about 1.5 h, reusing `cnn_lib.py`).** One output at T+30, trained on the onset rows with the idea-1 and
  idea-2 prior channels; blend 50/50 with CatBoost; evaluate on pseudo-March and the fold-0 hold-out. Low priority:
  onset changes have a poor LB record, so only probe if blind windows and the hold-out both improve.

---

## Notes

**Sources with nothing new.**
- `k1lst1x/trafficflowbench-2026`: a README, one architecture diagram and a course proposal. No code, no method.
- `NA__sachith7__when-does-temporal-repair-fail.ipynb`: Task 1 baselines only. One small point: our Task 1 features
  interpolate inside a day and `shift()` returns NaN at day edges; stitching consecutive days is worth about +0.0001.
- `0.7269_LB__shuangranse__solution-writeup.ipynb`: rule-based Task 2 and Kalman Task 1, superseded.
- Other GitHub repositories found: emanmusheer2008-glitch (0.470), leigaadik (copy of the organizer repo),
  MAYANKSHARMA01010 (pushed 09-22, not inspected).
- Discussions: no organizer statement after 2026-09-28. 745700 is unanswered.

**Hisernberg items checked and closed.**
- The local copies of both notes equal the GitHub files byte for byte; the repository has no entry after 2026-09-28.
- Eligibility-aware decoder (`t2/elig.py`): not needed. `future_coverage` is 1.0 for every official window, so every
  horizon cell is scored.
- Label fix with an imputer: not applicable. We train on observed cells only; their bias came from time interpolation.
- Stage-2 stacking, importance-weighted training, set-size bias, site-commit decoders: failed for them or for us.
- Extra onset events without the 360-minute spacing: only 286 more first-of-run events (+14%). Not worth it.

**Work already in progress in the lab (not listed as ideas).**
- Spacing flag `g72` (track `ogmodel`): validation has 8 of 40 ongoing windows exactly 360 min after the previous pick,
  private 12 of 40, official train 9 of 40. Possible refinement: split run-starts by the observable reason. When the
  history was already established at T−5 and coverage was fine, the hidden reason is the persistence cap, and the
  filled persistence at T is 0.871 ± 0.030 on the 23 official windows of that kind.
- Ineligible cells that carry values (`allfin`, `export_t2x.py`); model structure and step calibration (`ogmodel`);
  ongoing features from the 60-minute history (`ogfeat`); same-scenario prior designs for the first days of a month,
  model unchanged (`early`); private-regime emulation (`pvon`, `pvog`). The `pvon` shift rolls links; idea 3 says the
  real shift is mostly in time. None of these tracks retrains a model with honest priors or adds train-derived
  time-free site features.

**Other.**
- A U-Net member for Task 1 gave yeashusemwal +0.0010 LB at a 30% blend; their trees are weaker than ours.
- The 4 validation onset windows with an all-zero slot prior are 10% of the public onset score: 0.015 of the LB if they
  moved from 0 to 1, about 0.005 if they reached 0.6 from 0.25.
