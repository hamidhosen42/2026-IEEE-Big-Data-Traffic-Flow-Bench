# External sweep 2026-11-04: prioritized experiments

Read-only research. Nothing was pushed, submitted, committed or trained. Fetched pages were treated as data, and no downloaded code was run.
Context read: TRACKER (all rows, English original in analysis/markdown_originals_2026-11-03.zip), DISCUSSION_REVIEW_2026-11-03,
CAPACITY_SCENARIO (onset is a choice among train-recurrent candidates: 90 % within ±1 link / ±30 min of prior ≥ 0.05),
TRANSFER_ANALYSIS (onset location ≈ 0.02 LB of loss; public onset probes have SD 0.003–0.005), SIMAUG verdict, and t2_model_v20.window_features.

## Bottom line
- **Nothing found publicly beats us.** The only other TrafficFlowBench repo, Agnuxo1/traffic-flow-hrc-2026 (Apache-2.0), reports a local queue IoU of about 0.667, well below ours. The Hugging Face org "TrafficBenchV1" is unrelated (a 2025 dataset). No organizer code newer than 11-03 turned up.
- **The literature offers no ready-made onset-location method that fits this setup.** Breakdown-probability and bottleneck-activation work targets one known bottleneck with real loop data, using flow/capacity ratios, Weibull curves and ANNs. The location problem here is a choice among the candidate bottlenecks of the current scenario, and our features (fr = F/cap, cap ratios, priors) already contain the classic predictors.
- **Realistic upside before Nov 6 is small.** Expect about +0.000 to +0.003 S_total, mostly private and mostly onset. No public probe can read an onset change. Keep A/B frozen unless an item passes its pre-registered gate by a wide margin.

Conversions: 1 onset (or ongoing) IoU point × 0.15 = S_total. So +0.01 onset IoU is worth +0.0015.

## (A) Feasible before Nov 6 (in priority order)

### A1. Exact expected-IoU decoding (replacing the E[I]/E[U] proxy), with a site-mixture variant for onset
- Sources: Nowozin, CVPR 2014, https://microsoft.com/en-us/research/wp-content/uploads/2016/02/nowozin2014intersectionoverunion.pdf ; Ahmed et al., ICCV 2015, https://openaccess.thecvf.com/content_iccv_2015/papers/Ahmed_Optimizing_Expected_Intersection-Over-Union_ICCV_2015_paper.pdf
- **Idea.** `decode` maximises sum(top p) / (m + sum(rest p)), which is the EIoEU approximation. Ahmed et al. show this approximation is biased for small sets, which onset windows are (step 6 only, a few cells). Arms:
  - (a) Exact E[IoU] under independent cells, computed by Monte-Carlo (2–5k Bernoulli draws per window) or a cardinality DP, then searched over m.
  - (b) Site mixture. Group candidate cells into contiguous sites, draw the active-site set from site probabilities (max cell p per site, or a small site classifier), and pick the prediction with the best mean IoU over the draws. This includes predicting one site versus hedging over two.
- **Target.** Onset zero-overlap windows. On a two-site toss-up, the IoU-optimal choice differs from the proxy's.
- **New vs TRACKER.** EXP_T2_D tested thresholds and gap rules, EXP_T2_calib tested calibration, and EXP_T2_JOINT was an ongoing-only trajectory bank. None used exact EIoU, and none used an onset site mixture.
- **Rules.** Post-processing of ≤ T probabilities only. Compliant.
- **Effort.** 2–3 h CPU on the existing caches: onset cat8 OOF (oofX), pseudo-March 214-window onset probabilities (splitprior), pv/sp4 April caches. No retraining.
- **Gate (pre-register first).** The onset arm passes only if all four hold:
  - pseudo-March many-day ΔIoU ≥ +0.005;
  - train OOF ΔIoU ≥ 0;
  - ≥ 6/8 panels non-negative;
  - window-bootstrap 95 % CI lower bound > 0.
  - Ongoing arm (a): FM-PM ≥ +0.0005 on both footprints, the existing rule.
- **Expected gain.** +0.000 to +0.0015 S_total. The evidence is weak: the proxy is often close to exact for large sets, and calibration was already fine.

### A2. Onset precursor pack: slot-T flow, block-step, discharge saturation, section accumulation
- Sources: Bertini & Cassidy cumulative curves, https://its.berkeley.edu/publications/some-observed-queue-discharge-features-freeway-bottleneck-downstream-merge ; automated bottleneck ID from 5-min loops, https://https-sage-cnpereading-com-443.webvpn1.xju.edu.cn/doi/10.3141/1867-06 (TRR 1867) ; robust bottleneck ID (Jin et al.), https://files.topslab.wisc.edu/publications/2010/A%20Robust%20Bottleneck%20Identification%20Method%20using%20Noisy%20and%20Inconsistent%20Fixed-Point%20Detector%20Data%20(10-1374).pdf ; ANN breakdown with merge conditions, https://doaj.org/article/ad2f6877ff254b27af38a7a0dcc1fe7e
- **Idea.** Four groups of physically motivated features the onset model does not have:
  - (i) Flow at slot T. `make_rows` passes VT (speed at T) but never the flow at T; data stamped T is allowed.
  - (ii) Block step: fr_last − fr_l1 and fr at the last :00/:30 boundary versus before it. Onsets cluster 30 min after the demand-block change, and fr_slope (mean of the last 3 minus the first 3) blurs this.
  - (iii) Discharge saturation: q at T−1..T over the train-measured discharge of known heads (capscen heads.csv; discharge is identical across splits). A bottleneck active between detectors pins downstream flow at discharge before any detector drops below 0.6 vf.
  - (iv) Section accumulation: cumulative Σ(q_up − q_dn) over [T−60, T] per link pair, Bertini's excess accumulation.
- **Target.** Picking the right candidate site, in both public and private windows. It is the only listed item that could help March days 1–5, where no same-scenario history exists.
- **New vs TRACKER.** EXP_T2_H (speed deltas, density over critical) was inside noise. EXP_T2_PHASE was clock phase only. Groups (i), (iii) and (iv) are not in the feature list. Group (ii) partly overlaps fr_slope.
- **Rules.** Only data ≤ T is used. Discharge comes from train, which is allowed.
- **Effort.** 3–5 h lab CPU: onset CatBoost retrain at production settings (cat8, NORAMP, v13 features + pack), folds 0+1, then pseudo-March. Use 15+ fresh seeds; PHASE2 showed ±0.02 per-seed noise.
- **Gate.** Same as A1, plus mean over ≥ 15 fresh seeds and an official-40 train-analog sign that is not negative. If it passes, ship it only in the private file (pick A April onset) unless the public analogs are also positive.
- **Expected gain.** +0.000 to +0.003 S_total. The evidence is weak to negative: SIMAUG found activation "seeded by hidden factors", and EXP_T2_H failed.

### A3. kNN analog-day onset location prior for April (pick A only)
- Sources: Habtemichael & Cetin, TR-C 2015, k-NN with similar traffic patterns, https://www.bohrium.com/paper-details/short-term-traffic-flow-rate-forecasting-based-on-identifying-similar-traffic-patterns/814596193983135745-3753 ; Smith & Demetsky nonparametric regression, https://trid.trb.org/View/352691
- **Idea.** sp4 averages all earlier same-scenario days per (link, slot, day class).
  - Instead, weight each earlier day (March + April days < d, and same-day earlier hours are off-limits) by how closely its corridor state on [T−60, T] matches today's (speed and flow vectors, recency-weighted distance).
  - Use the weighted queue indicator at T+30 as a new prior column, or blend it with sp4.
  - The underlying fact: 84–89 % of new-scenario onsets sit where a previous same-split day also queued, so the open question is which day-type the current day resembles.
- **Target.** Onset location for private windows only. There is no lever on public: March days 1–5 have 0–4 analog days.
- **New vs TRACKER.** EXP_T1_R7 used kNN-day for T1 (dropped). EXP_SCEN_LLR and MIX9 work at scenario level, not day-analog. EXP_T2_PV2 used event-location columns, which were too sparse; kNN uses every day's state, not just events.
- **Rules.** Only days fully before d, at slot ≤ T−1, are used. Compliant with "April may use train + March".
- **Effort.** 3–4 h lab, then 1 kernel (≈ 1–2 h CPU). A pick-A change also needs T4 validation and a hand-over to the team member.
- **Gate.** Many-day pseudo-March (days ≥ 10, 161-window late-month set), with the sp4 blend as control:
  - ΔIoU ≥ +0.01;
  - ≥ 6/8 panels non-negative;
  - CI lower bound > 0;
  - EXP_T2_OSP_DOWNSIDE re-run (train-like month) loss ≤ 0.005.
- **Expected gain.** +0.000 to +0.003 private S_total, public 0. Evidence is moderate for the mechanism and weak for any increment over sp4.

### A4. LATC / LCR as a Task 1 diversity member
- Sources: transdim (MIT), https://github.com/xinychen/transdim ; LCR paper, https://arxiv.org/abs/2212.01529 (code at github.com/xinychen/LCR, licence not verified)
- **Idea.** Our LRTC is an EM truncated-SVD over 3 unfoldings. LATC adds an autoregressive temporal-variation term, and LCR adds a circulant-nuclear-norm plus Laplacian temporal smoothness term. Both target smooth local dynamics that SVD lacks, which may help S_LWR (temporal ΔN consistency).
- **New vs TRACKER.** LATC was listed as "P" (proposed) in FINAL_RESEARCH_REPORT row 14 and never run.
- **Rules.** Task 1 is offline. Compliant.
- **Effort.** 3–4 h: re-implement in NumPy (about 100 lines, no code download), with the pilot rank/λ chosen on non-evaluation cells as in ROUND 4. Then 2–3 h for a swap kernel.
- **Gate.** The t1div harness, a new ROUND 8 written before any numbers:
  - J-proxy ≥ +0.0005 on stand-in S5 (k9);
  - ≥ 7/10 panels non-negative;
  - both evaluation months positive.
- **Expected gain.** +0.0000 to +0.0003. LRTC follow-ups have shrunk each round (+0.00034, then +0.00022).

### A5. (Optional, report-and-risk) SHAP attribution of the onset model on zero-overlap windows
- Source: precursor attribution with ST-GNN Shapley, https://arxiv.org/pdf/2504.17109
- **Idea.** Take TreeSHAP from CatBoost on the onset windows the model misses (wrong site). Check whether the loss is driven by prior columns (scenario mismatch) or by state columns. The answer tells whether A2 or A3 has any chance.
- **Effort.** 1 h, run before A2/A3.
- **Gain.** 0 directly. Its value is triage, plus a report figure.

## (B) Report / analysis value only (not before Nov 6)
| Idea | Sources | Why only report |
|---|---|---|
| Breakdown-probability curves (Weibull / product-limit) per train bottleneck vs fr | https://www.bohrium.com/paper-details/development-and-evaluation-of-methods-for-constructing-breakdown-probability-models/813223788782026761-13367 ; https://repository.lib.ncsu.edu/server/api/core/bitstreams/b3f4b99d-e996-4694-946a-be746e477d21/content | Quantifies how stochastic activation is, a natural explanation of the onset ceiling. Not a score lever: fr is already a feature |
| Bertini cumulative-curve map of active bottleneck sets per split (train vs March vs April) | Bertini & Cassidy (link above) | A strong figure for the award report's "scenario shift" section (complements CAPACITY_SCENARIO §1.3) |
| Physics-informed TSE (PIDL LWR/CTM, ADD-PINN, PI-DeepONet) for Task 1 | https://www.arxiv.org/pdf/2101.06580 ; https://arxiv.org/pdf/2605.08028 ; https://www.emergentmind.com/papers/2508.08002 | Built for sparse sensors and over-smooth shocks. Ours is dense with GBDT at S_state ≈ 0.94, and the U-Net was already dropped. Hours of GPU training per panel. Related-work paragraph only |
| Online test-time adaptation (ADCSD, FORESEE, TTT thesis) | https://arxiv.org/pdf/2401.04148v1.pdf ; https://arxiv.org/pdf/2602.21757 ; https://opus.lib.uts.edu.au/bitstream/10453/179545/1/thesis.pdf | Point-forecast MAE drift correction. Our analogue (scenario prior, March-augmented training, extent calibration EXP_T2_EXTCAL) is done, and EXTCAL says not to retry scalar calibration. Cite as framing |
| Lovász-softmax / IoU-surrogate training | https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf | Needs a neural model; GBDT production cannot use it cheaply. Future work |
| Traffic4cast 2022 (sparse loop counters → congestion classes) | https://arxiv.org/pdf/2211.07031v1.pdf ; https://arxiv.org/pdf/2211.09984v1.pdf ; github.com/skandium/t4c22 | Second place used LightGBM on PCA-reduced counters, which supports our GBDT plus priors choice. No new trick for a set/IoU metric |

## Resources NOT worth trying (and why)
- **Time-series foundation models (Chronos/Chronos-Bolt, TimesFM 2.x, Moirai, Lag-Llama).** These are univariate or covariate-light context forecasters. Onset windows have a clean history and the queue is triggered after T by demand the model cannot see, so a zero-shot speed forecast at T+30 is near persistence. They would also need weights attached as a Kaggle dataset. Licences are unverified here: Chronos and TimesFM are Apache-2.0 as far as I know, while Moirai 1.x weights are CC-BY-NC, an award-package risk. Energy benchmarks also show overconfidence: https://arxiv.org/html/2602.10848
- **OpenCity / UniST / LargeST / PEMS checkpoints.** OpenCity (https://arxiv.org/abs/2408.10269, weights via HKUDS/OpenCity) is pretrained on real PEMS/CN graphs for MAE point forecasts. It does not match the synthetic generator, the 0.6 vf set label, or the 8-corridor topology, it needs a GPU, and fine-tuning data is just our train. No UniST source was even retrievable. There is no plausible gain over 511-leaf GBDT.
- **Diffusion or transformer imputers (CSDI, PriSTI, ImputeFormer) for Task 1.** These need a GPU, and the U-Net precedent was negative. T1 diversity gains are now ≤ 0.0003.
- **CTM/METANET simulation augmentation.** This is already NO-GO (EXP_T2_SIMAUG: location hit 0.618 vs prior 0.976).
- **Agnuxo1/traffic-flow-hrc-2026 (reservoir computing).** Its self-reported queue holdout IoU of about 0.667 is below ours, and no reservoir gain was shown.
- **HF TrafficBenchV1 and fmops/ai-traffic-flows.** Unrelated to this competition.
- **Generic set/conformal size control.** EXP_T2_EXTCAL already covers Platt, isotonic and size models.

## Recommendation for the remaining 2 days
1. Run A5 (1 h) and then A1 (2–3 h). Both work on caches only, with no kernel.
2. Run A2 or A3 only if A5 shows that state columns, not priors, drive the misses.
3. Change a final pick only on a gate pass of at least 2× the threshold. Otherwise keep k9/k10 (0.88281) and put A5 and the (B) analyses into the Nov 10 award report.
4. Do not spend public submissions on onset variants: probe SD is 0.003–0.005, above any expected gain.
