Team: Md. Hamid Hosen, Md. Foysal Emon Shanto, Md. Esfer Abdus Sami.

## Summary

| | Public (March) | Private (April) |
|---|---:|---:|
| Starting public notebook | 0.80939 | – |
| Final picks (`tfb-k9`, `tfb-k10`) | **0.88281** | **0.87590** (rank 83) |

Most of our gain came from Task 2 (about +0.053 public), then Task 1 (+0.016), the Task 3 post-processing (+0.002) and one
combined step (+0.003). Probes that zeroed one block of the final file measured public onset IoU 0.779, ongoing IoU 0.845
(S_queue 0.812) and S_ODME ≈ 0.994. Everything we changed went through day-blocked cross-validation, a pseudo-March evaluation
and, from October on, gates written down before the result was computed.

## Task 1 — state reconstruction

Every member predicts **residuals** of speed/v_f and flow per lane with respect to a per-station linear interpolation in time.
Features include the interpolation itself, neighbour stations at the same instant, per-day rank-3/rank-5 SVD reconstructions of
the interpolated fields, occupancy, averages along ±80 and ±18 km/h characteristic waves, and FD-implied densities and flows.
Profile (weekday × time) features were dropped: the profile-free model was better on 10/10 panels.

What mattered most:
- **Transductive rows.** Task 1 is offline, so in 2–4 rounds we hid 20 % of the released cells of the target months and used
  them as extra training targets: +0.0017 public.
- **Capacity.** 511 leaves, 1,500 trees and 4× the rows (+0.0011), then bags of 8 + 6 LightGBM members.
- **Diverse members, each added only after a gate** (ΔJ ≥ +0.0003 after post-processing, CI > 0, ≥ 7/8 panel-months): a CatBoost
  member (+0.00071), an XGBoost member (+0.00056) and an EM low-rank tensor completion of the month's own flow cube
  (days × 288 slots × stations, ranks (24, 40, 10)), blended at 0.35 on flow (+0.00054). The LRTC member gained mostly through
  S_LWR: averaging with an estimator of a different kind removes part of the per-cell noise.
- **A blackout specialist.** About 1.9 % of target rows lie in the 90 minutes after a Task 2 forecast origin, where the whole
  mainline is dark. A model trained on blackouts simulated exactly where the official selector puts them (with ramp flows, which
  stay visible) takes 75 % weight there.

## Task 3 — physics, only through Task 1

S_LWR is computed against organizer-held boundary fluxes and, locally, truth plus 1 km/h of noise already drops it from 1.0 to
about 0.89. It is set by the per-cell noise of the reconstructed density, not by the flux balance. So we improved the density
directly, with two post-processing steps that work on any reconstruction:
- **FD-gated reconciliation.** In predicted-congested cells (v < 0.6 v_f) the per-lane density is moved towards the congested FD
  branch implied by the predicted speed; 75 % of the correction goes to speed. +0.00167 public.
- **1-D total variation.** Each temporal run of target cells per station is TV-denoised in density, anchored to the released
  cells on both sides. +0.00026 public.

Conservation-aware projections, density interpolation post-processing (27 settings) and Kalman smoothing all failed: on these
panels the station-to-station flux estimate is 10–28× noisier than the signal.

## Task 2 — queue forecasting (data ≤ T only)

**Cell classifier + expected-IoU decoder.** One row per (link, horizon step), features from the 60-minute history, the released
row at T, static link attributes and train-split location/time priors. Probabilities are decoded per window by choosing the
top-m cells that maximise Σ_top p / (m + Σ_rest p). This first model alone took the public notebook from 0.809 to 0.844.

**Onset is a one-step problem.** Every onset queue first appears at T+30 (organizer post 742350), so we predict step 6 only, with
a CatBoost classifier. For March rows the location prior is blended with a *same-scenario* prior from released March days
strictly before the window's day (+0.00104 — the only onset change besides CatBoost that survived the leaderboard).

**Selection-faithful training for ongoing windows** — our largest single gain after the first model. We wrote a replica of the
official window selector. Our first replica reproduced only 14/40 official ongoing origins; evaluating the persistence cap
(IoU ≤ 0.9) on a time-filled state instead reproduced 36/40 (39/40 within 5 minutes). Training on windows selected the same way
the official ones are lifted official-like OOF IoU from 0.839 to 0.875–0.881 and the public score by **+0.0047**. Using all finite
released values as inputs (including ineligible-but-valued cells) and 13 occupancy features added +0.001.

**April and the two final picks.** March and April come from a new scenario relative to train (re-drawn ramp demand, a different
set of active bottlenecks). Whether April shares March's scenario was the one question our train and March evidence could not
settle, and the bet is two-sided: models trained with released-March windows win by +0.15 onset IoU if it does, and lose
0.13–0.14 if it does not. So we submitted one file per hypothesis, identical except for 200 April queue rows:
- `tfb-k9`: April onset and ongoing models trained on train + released-March windows, with the same-scenario prior.
- `tfb-k10`: train-only April models (a hedge).

Both choices used only train and March evidence; nothing was selected with April data or the public leaderboard.

## Task 4 — ODME

The organizer's construction: min ‖Af − c‖² + 0.05 ‖f − b‖², f ≥ 0, solved as an NNLS on the stacked system. A λ sweep and IRLS
gave nothing; S_ODME ≈ 0.994 for everyone using it.

## Validation — what kept us from fooling ourselves

- **Leaderboard noise is large for Task 2.** With 40 windows per condition, one ongoing probe has an SD of ±0.0012–0.0027 and one
  onset probe ±0.003–0.005. A public delta below +0.0001 was never accepted as a gain.
- **Onset never transferred.** Six onset changes that passed CV or pseudo-March then lost on the public LB, one by −0.013.
  An onset window has a median of about 3 queued cells, so one flipped cell moves its IoU by 0.3–0.6.
- **Footprint-matched pseudo-March (FM-PM)** — pricing only the official windows whose decision changes, with paired deltas of
  similar lab windows — tracked ongoing LB changes (MAE 0.0005) where plain OOF did not.
- **Pre-registration.** From October, every Task 1 round and every late Task 2 test had its gate written and timestamped before
  any number existed. Task 1 lab gains transferred to the LB at about 0.3–0.6×.
- **The organizer's post-T rerun, done ourselves.** We reran every Task 2 producer with all rows after each window's origin
  deleted (three deletion styles, ~4 million compared cells): 0 changed predictions. It also exposed crashes of the original code
  on empty inputs, which we guarded.

## What did not work

- Onset: LODO priors, early-month mixture priors, CatBoost stacking, all-finite inputs, seed bags (+0.003, CI includes 0),
  exact expected-IoU decoding, precursor columns (slot-T flow, downstream density, upstream accumulation; −0.014), half-hour phase
  columns (+0.018 on 5 seeds, +0.003 on 15 fresh seeds).
- Simulation: a cell-transmission model to generate training scenarios failed its realism check — in this generator the released
  ramp flows barely move the mainline, and onsets landed in the right place in 62 % of windows against 98 % for a location prior.
- Ongoing: 103 extra history features (good offline, −0.0012 public), calibration, shock-wave gates, reversed link order in W/S
  panels (−0.003), raw space-time patches, persistence blends.
- Task 1: a day-grid U-Net (speed RMSE 2.5 vs 1.7 for LightGBM), Kalman/RTS smoothing (−0.019), ExtraTrees, ridge, kNN-day members.

The biggest open gap for us is the *location* of onset queues: in new-scenario proxies, most of the onset loss comes from windows
whose predicted cells miss the true queue entirely, and about 65 % of the onset model's attribution is location/clock identity,
which changes with the scenario.

## Reproducibility

One entry script (`run_all.py --pick A3|B3`) rebuilds either pick from the release; every stage is the original Kaggle kernel
with only its I/O lines changed. On Kaggle, the consolidated code rebuilt pick A with the same public score (0.88281) and pick B
byte for byte. The full recipe needs about 145 Kaggle CPU-hours, dominated by the Task 1 members.

Thanks to the organizers for an unusually careful benchmark and for answering every rules question on the forum, and to the
authors of the public notebooks we started from.
