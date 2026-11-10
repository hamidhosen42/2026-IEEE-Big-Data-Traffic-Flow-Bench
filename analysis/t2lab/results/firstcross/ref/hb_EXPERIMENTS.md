# Experiment log

All numbers are local. S_state uses the official formula on the train holdout (days 243–272). The LWR
proxy is `evaluate.s_lwr_proxy`. J = 0.35·S_state + 0.10·S_LWR is the part of S_total that Task 1
controls; S_FD is about constant.

## Task 1 / Task 3

| # | Setup | S_state | LWR proxy | Note |
|---|---|---|---|---|
| E0 | historical mean (official baseline), D7_I10_W | 0.832 | 0.070 | full train, no blackouts |
| E0 | temporal linear interpolation, D7_I10_W | 0.932 | 0.534 | |
| E1 | per-panel LightGBM residual on interpolation, 500k rows, D7_I10_W | 0.9507 | 0.585 | holdout, no blackouts |
| E2 | + L1 density model, q = v·k̂ | 0.9478 | 0.607 | J +0.0014 |
| E3 | blackout cells, D7_I10_W: interpolation vs gap-LGB | speed RMSE 5.81 → 5.36 | | regular cells about 1.1 |
| P1 | pooled 10-panel LightGBM (150k reg/panel), sampled holdout with dark share reweighted | **0.9385** family mean | | raw |
| P1 | same, reconcile a=0.25 | 0.9348 | | costs mostly dark-cell flow |
| P1-full | full-coverage holdout, realistic blackout density, 4 panels | raw → recon a=0.25 | LWR +0.013 to +0.081 | J +0.0010 to +0.0062 on every panel → **adopt a=0.25** |

Pooled model holdout RMSE:
- regular cells: speed 1.54, flow 30.6/lane, density (Huber) 0.567
- dark cells: speed 6.90, flow 65.9/lane, density 3.21

Error budget (regular cells):
- free flow: 97% of cells, 74% of speed SSE and 91% of flow SSE (noise floor)
- queued: 2% of cells, 22% of speed SSE

## Physics facts
- Congested observations lie on the congested branch of the released triangular FD: median deviation
  0.0%, IQR ±2.4%. In free flow, v ≈ v_f (std 2.5 including the transition zone).
- Observed |ΔN|/N per step is 4.4%. The topology-based flux correlates with ΔN at only 0.06, so the
  organizer's flux must absorb the noise, which is why the proxy is valid.

## Task 2 (agent report, trafficflow/docs/TASK2_ANALYSIS.md)
CV S_queue on 5,083 selector-replicated windows (official aggregation):

| Method | S_queue (sim) |
|---|---|
| persistence (official) | 0.285 |
| persistence_fill | 0.353 |
| rules | 0.663 |
| lgb_v2 | 0.788 |
| **lgb_v3** | **0.795** (onset 0.713, ongoing 0.877) |

## Task 4 (agent report, trafficflow/docs/TASK4_ANALYSIS.md)
The L2 projection of the split prior onto the split counts reproduces the published baseline S_ODME
(0.8357 vs 0.8359). Expected S_ODME is about 1.000.

## Expected S_total (local estimates)
| Task | Estimate | Weighted |
|---|---|---|
| state | ~0.935 | 0.327 |
| queue | ~0.75–0.79 | 0.225–0.237 |
| physics | ~(0.98 + 2·0.56)/3 ≈ 0.70 | 0.105 |
| ODME | ~1.0 | 0.200 |
| **total** | | **≈ 0.857–0.869** |

That compares with the best post-rebuild public score of 0.879. Our previous best was 0.80855.

## Leaderboard log (public = validation month, March 2031)

| Date (UTC) | Submission | Contents | Public | Rank | Note |
|---|---|---|---|---|---|
| 2026-09-22 | earlier account subs (v1–v5) | various, not from this pipeline | best 0.80855 | 38th/108 post-rebuild | baseline for comparison |
| 2026-09-23 13:42 | **A** `A_full_t1full1r25_t2lgbv3_t4l2proj.zip` | T1 pooled LGB full1 + density recon a=0.25; T2 lgb_v3; T4 L2 projection | **0.85204** | 21/145 overall, **13/115 post-rebuild** | +0.043 over previous best; about 0.005–0.017 below the local estimate (0.857–0.869); post-rebuild top 0.88153 |
| 2026-09-23 13:47 | **P1** probe_odme_only | T4 only (state, queue zeroed) | 0.19876 | – | **S_ODME = 0.9938.** The Task 4 projection hypothesis holds; the most Task 4 can still add is 0.0012 total |
| 2026-09-23 13:51 | **P2** probe_state_odme | A with queue zeroed | 0.62747 | – | A − P2 = 0.30·S_queue → **S_queue = 0.7486** (CV 0.795); P2 − P1 = 0.35·S_state + 0.15·S_phys = 0.42871 (local 0.432) → **S_phys ≈ 0.67–0.68** |
| 2026-09-23 13:57 | **P3** probe_onset_zeroed | A with onset windows zeroed | 0.74919 | – | onset = 2·(A − P3)/0.30 = **0.686** (CV 0.713); ongoing = **0.812** (CV 0.877) → most of the Task 2 transfer loss is in ongoing windows |
| 2026-09-23 16:23 | **B1** `B1_gate06_t2v4robust.zip` | A + density reconciliation gated to v<0.6·v_f; T2 lgb_v4_robust (ongoing = 50/50 blend of v3 and the no-location model; onset identical to v3) | **0.85732** | 14/146 overall, **8/116 post-rebuild** | +0.0053 vs A (local expectation +0.002 to +0.004: gating +0.0016, ongoing robustness +0.001 to +0.003). The Task 1/3 and Task 2 parts of the gain are not separated on the LB; post-rebuild top 0.88153 |
| 2026-09-24 12:02 | **S1 C1** `C1_gate06_t2v5.zip` | B1 with queue v5 (279 queue cells) | **0.85975** | – | +0.00243 vs B1 → ΔS_queue = **+0.0081** (CV +0.005). Base → C1 |
| 2026-09-24 12:08 | **S2 D1** `D1_gate06_t2v6.zip` | C1 with v6 onset trained on corrected labels (42 onset cells) | **0.86428** | – | +0.00453 vs C1 → ΔS_queue = +0.0151, **onset +0.030** on March. Base → D1 |
| 2026-09-24 12:11 | **S3 D2** `D2_gate06a75_t2v6.zip` | D1 with gate split a = 0.75 (266,914 dense-traffic state rows) | **0.86492** | 14/158 overall, **8/128 post-rebuild** | +0.00064 vs D1, **exactly the local prediction (+0.0006)**. Base → D2; post-rebuild top 0.88318 |
| 2026-09-24 14:40 | **S4 E1** `E1_fd_gate06a75_t2v6.zip` | D2 with Task 1 state from the FD-feature models `full3` (state rows only) | **0.86591** | 14/160 overall, **8/130 post-rebuild** | +0.00099 vs D2 (local +0.0012). Base → E1; post-rebuild top 0.88318 |
| 2026-09-24 14:42 | **S5 P4** probe `P4_E1_onset_zeroed.zip` | E1 with onset windows zeroed (311 cells) | 0.75669 | – | **onset = 2·(E1 − P4)/0.30 = 0.728** on March (A: 0.686) |
| 2026-09-25 08:18 | **F1** probe `F1_onsetb05.zip` | E1 with the v6 onset re-decoded at logit bias +0.5 (+15 onset cells; 12 in 8 validation windows) | 0.86356 | – | −0.00235 → **March onset −0.016**: larger onset sets hurt |
| 2026-09-25 08:44 | **F2** `F2_onset_site2.zip` | E1 with the onset decoded by `site2_lo.05_r.5` (commit to 1–2 sites; −11 hedge cells, 9 in 7 validation windows) | 0.86418 | 9/140 post-rebuild (E1) | −0.00173 → **March onset −0.012**: fewer hedges hurt too. Base stays E1 |
| 2026-09-25 09:44 | **G1** `G1_tvsmooth.zip` | E1 + TV smoothing of the density inside runs of target cells (5.1M state rows, mean \|Δv\| 0.0066 km/h, \|Δq\| 11 veh/h) | **0.86651** | 9/140 post-rebuild | **+0.00060, exactly the local J prediction (+0.00062).** Base → G1 |
| 2026-09-25 10:25 | **G2** `G2_onset_v8stack.zip` | G1 with onset = v8 `seeds9_stack03` (stage-2 stacking 0.3 + 9-seed stage 1; 15 onset cells, 11 in 4 validation windows) | **0.86711** | – | **+0.00060** → March onset about +0.004 (CV +0.0035 hybrid / +0.0024 old). Base → G2 |
| 2026-09-25 10:16 | **G3** `G3_ongoing_stack.zip` | G2 with ongoing = v9 stage-2 stacking (217 ongoing cells, 139 in validation) | 0.86361 | – | **−0.00350 → March ongoing −0.023**, against CV +0.0065 ± 0.0009. Failed transfer; base stays G2 |
| 2026-09-26 00:09 | **H2** `H2_t1ens34.zip` | G2 with Task 1 state = mean of full3 and full4 (seed ensemble; state rows only) | **0.86777** | 10/151 post-rebuild | **+0.00066** (local J +0.00077). Base → H2 |
| 2026-09-26 00:11 | H1b `H1b_og_shrink_on_ens34.zip` | H2 with ongoing v10 (stage 2 may only remove v5 cells where recurrence ≥ 0.05; 77 cells) | 0.86629 | – | **−0.00148** → March ongoing −0.010. It passed the new Task 2 gate (plain +0.0029, both weighted CVs +, footprint clean) and still failed. **Ongoing stacking line dropped** |
| 2026-09-26 00:16 | P5 probe `P5_smooth_f3.zip` | H2 with TV smoothing ×3 | 0.86736 | – | −0.00041 (local −0.00062) → the official Task 3 truth behaves like the train truth; the smoothing strength is at or near its optimum |
| 2026-09-26 00:20 | P6 probe `P6_H2_queue_zeroed.zip` | H2 with the queue zeroed | 0.63144 | – | **S_queue(H2) = 0.7878** exactly: onset 0.732, ongoing 0.843. Task 1+3 = 0.43268 |
| 2026-09-27 06:22 | **H3** `H3_t1ens345.zip` | H2 with Task 1 state = mean of full3, full4, full5 (third seed; state rows only) | **0.86799** | – | **+0.00022, local J +0.00023**: exact. Base → H3 |
| 2026-09-27 07:00 | G7 `G7_ongoing_v7.zip` | H3 with ongoing = v7 (the v5 recipe retrained on hybrid labels; +100/−45 cells in 30/40 validation windows) | 0.86779 | – | **−0.00020 → March ongoing −0.0013.** CV was +0.0032 (hybrid truth) / +0.0006 (old truth). The label fix that gave onset +0.030 does nothing for ongoing. **Ongoing label work closed.** Base stays H3 |
| 2026-09-27 09:48 | **H5w** `H5w_t1ramp_dark.zip` | H3 with Task 1 regular rows = mean(full3, 4, 5, 7), blackout rows = full7 (ramp-flow member) | **0.86838** | – | **+0.00039** (local J +0.00081). Adopted, base → H5w. First Task 1 change where the proxy is off by more than 0.0002: the holdout gives blackouts to D12_I405_N, which has none in the test, and the blackout part of the gain transferred at about half |
| 2026-09-27 09:58 | H6 `H6_dark8.zip` | H5w with blackout rows = mean(full7, full8) (dark-only booster) | 0.86839 | – | **+0.00001** (local +0.00016). Not adopted. Blackout-row gains transfer unreliably (below) |
| 2026-09-27 11:10 | G8 `G8_ongoing_v11.zip` | H5w with ongoing = v11 (v5 blend, og_v3 / og_v3_noloc at capacity p3; +32/−45 cells in 28/40 validation windows) | 0.86798 | – | **−0.00041 → March ongoing −0.0027**, against CV +0.0053 ± 0.0005, shift-weighted +0.0050 (validation) / +0.0052 (private), footprint clean. Passed the full Task 2 gate and lost. Not adopted; robust list for the final pick |
| 2026-09-28 00:10 | **H7** `H7_reg9.zip` | H5w with Task 1 regular rows = 0.5·full9 (boosted regular member) + 0.5·mean(full3, 4, 5, 7) | **0.86864** | – | **+0.00026** vs H5w (local +0.00053). Adopted, base → H7. Transfer about 50%: the S_state part (+0.00022) came through, the LWR part (+0.00031, a better density model in congested cells) mostly did not |
| 2026-09-28 03:05 | H8 `H8_reg10.zip` | H7 with regular rows = 0.5·full9 + 0.5·full10 (two boosted seeds, old members dropped) | 0.86868 | – | **+0.00004** vs H7 (local +0.00026: S_state +0.00013, LWR +0.00013). Not adopted (bar +0.0001). Even the S_state part did not transfer |
| 2026-09-28 05:05 | **H9P** `H9P_transductive.zip` | H7 with Task 1 regular rows = fullP (transductive member: train + observed March/April cells) | **0.87093** | 14/post-rebuild | **+0.00229** vs H7 (pseudo-holdout +0.0019 from S_state alone). Adopted, base → H9P. The largest Task 1 step since E1 |
| 2026-09-28 06:48 | **H10P** `H10P_transductive2.zip` | H9P with regular rows = 0.35·fullP + 0.65·fullP2 (2× transductive rows) | **0.87202** | – | **+0.00109** vs H9P (pseudo-holdout +0.00054 from S_state). Adopted, base → H10P. The LB gain is about 2× the S_state-only pseudo figure, as for H9P |
| 2026-09-28 08:35 | **H11P** `H11P_transductive3.zip` | H10P with regular rows = 0.5·fullP2 + 0.5·fullP3 (fullP3: 3× transductive rows, seed 13) | **0.87249** | – | **+0.00047** vs H10P (pseudo +0.00025 from S_state; the ~2× LB factor held). Adopted, base → H11P |

### Decomposition of A (0.85204), exact from the probes
| Task | Weighted | Task score | Local estimate |
|---|---|---|---|
| ODME | 0.19876 | 0.994 | 1.000 |
| queue | 0.22457 | 0.749 (onset 0.686, ongoing 0.812) | 0.795 (0.713 / 0.877) |
| state + physics | 0.42871 | S_state ≈ 0.935–0.94, S_phys ≈ 0.67–0.68 | 0.432 |

Where the headroom is:
- queue: +0.1 S_queue is worth +0.030 total. Ongoing robustness to incidents comes first.
- physics: +0.1 S_phys is worth +0.015. The density model and blackout cells are the levers.
- state: about +0.005 at most.
- ODME: done.

## Gated density reconciliation (2026-09-23)
Per speed band, the L1 density error (D12_I5_S holdout) shows where the density model is better:

| v/v_f | q/v | density model | FD(v) |
|---|---|---|---|
| < 0.4 | 2.99 | **1.50** | 1.77 |
| 0.4–0.6 | 3.40 | 2.82 | **2.64** |
| 0.6–0.8 | **4.83** | 5.10 | 5.50 |
| 0.8–0.9 | **2.85** | 3.73 | 5.94 |

The density model wins only in dense traffic, so reconciliation is gated to predicted v < 0.6·v_f.
Full-coverage holdout, realistic blackouts:

| Panel | J recon-all (A) | J gate 0.6 | Δ |
|---|---|---|---|
| D12_I5_S | 0.38307 | 0.38529 | +0.0022 |
| D7_I10_W | 0.38704 | 0.38767 | +0.0006 |
| D7_I405_S | 0.37741 | 0.37928 | +0.0019 |
| D12_I405_N | 0.39010 | 0.39184 | +0.0017 |

Gating improves J on 4/4 panels (mean +0.0016 total) and is adopted: `make_submission --gate 0.6`.
Adding FD density in 0.4–0.6·v_f changes J by −0.0002 to +0.0005, which is noise, so it is not adopted.

## Task 1/3 v2 round (2026-09-23 evening)
Full-coverage holdout, realistic blackouts, gate 0.6. J = 0.35·S_state + 0.10·S_LWR (the Task 1+3 part of S_total).

| Panel | hold1 (submitted) | hold2 (all new) | + longer blackout speed/flow only | + L1-leaning density only |
|---|---|---|---|---|
| D12_I5_S | **0.38529** | 0.38422 | 0.38530 | 0.38419 |
| D7_I10_W | **0.38767** | 0.38747 | 0.38753 | 0.38760 |
| D7_I405_S | 0.37928 | 0.37900 | **0.37938** | 0.37890 |
| D12_I405_N | 0.39184 | 0.39157 | **0.39195** | 0.39144 |

- The L1-leaning density model (Huber α 0.2, 127 leaves) is worse on 4/4 panels, so it is **rejected**. The near-L2 density (α 1) is better for S_LWR.
- Longer-trained blackout models: mean +0.00002 total, which is noise and not worth a retrain.

**Speed/flow split inside the density gate.** LWR is unchanged, since q/v = k̂ for any a:

| Panel | a = 0.25 (A, B1) | a = 0.75 | a = 1.0 |
|---|---|---|---|
| D12_I5_S | 0.93180 | **0.93429** | 0.93429 |
| D7_I10_W | 0.93996 | 0.94091 | **0.94109** |
| D7_I405_S | 0.91588 | **0.91845** | 0.91819 |
| D12_I405_N | 0.94800 | **0.94934** | 0.94930 |

a = 0.75 improves S_state on 4/4 panels (mean +0.0018, about +0.0006 total). In queues, flow sits near discharge capacity while speed carries the uncertainty. **Adopted** (`--recon-a 0.75 --gate 0.6`).

## Task 2 v5 (2026-09-23 evening, agent)
CV on sim / official windows. Details in trafficflow/docs/TASK2_ANALYSIS.md section 12.

| Slice | v4 (in B1) | v5 |
|---|---|---|
| overall | 0.7971 / 0.8233 | **0.8022 / 0.8299** |
| onset | 0.7133 | **0.7210** |
| ongoing | 0.8809 | **0.8833** |
| non-recurrent onset / ongoing (recur < 0.05) | 0.275 / 0.588 | **0.295 / 0.599** |

## Candidates for 2026-09-24 (single-factor chain, each passes 65/65 checks)
| File | Differs from | Change | Expected Δ |
|---|---|---|---|
| `C1_gate06_t2v5.zip` | B1 | Task 2 v4 → v5 (279 queue cells) | about +0.0015 (CV × 0.30), possibly more on March |
| `C2_gate06a75_t2v5.zip` | C1 | gate split a 0.25 → 0.75 (266,914 dense-traffic state cells) | about +0.0006 |

## Task 1 with FD (fundamental-diagram) features, hold3 (2026-09-24)
Column-based FD features (`TFB_FD=1`, `t1_pipeline.add_fd`, verified identical to the native path): congested-branch flow implied by speed, speed implied by flow, and q/v density for the interpolated/previous/next values, plus `fd_w`, `fd_kj` and v/v_f.

Holdout RMSE, hold1 → hold3:

| Model | hold1 | hold3 |
|---|---|---|
| speed | 1.543 | 1.515 |
| flow | 30.57 | 30.19 |
| density | 0.567 | 0.541 |
| dark speed | 6.903 | 6.888 |
| dark flow | 65.86 | 65.70 |
| dark density | 3.21 | 3.19 |

Full-coverage holdout, gate 0.6, a = 0.75:

| Panel | J hold1 | J hold3 | Δ | LWR hold1 → hold3 |
|---|---|---|---|---|
| D12_I5_S | 0.38618 | 0.38810 | +0.0019 | 0.5916 → 0.6068 |
| D7_I10_W | 0.38800 | 0.38862 | +0.0006 | 0.5868 → 0.5905 |
| D7_I405_S | 0.38044 | 0.38215 | +0.0017 | 0.5896 → 0.5985 |
| D12_I405_N | 0.39232 | 0.39293 | +0.0006 | 0.6005 → 0.6055 |

Better on 4/4 panels (mean +0.0012 total), with gains from both the speed/flow and the density models. **Adopted.** Full-data fit `full3` is in progress.

## Day summary 2026-09-24: B1 0.85732 → E1 0.86591 (+0.0086)
Every step changed a single, locally validated factor. Each Task 1 delta matched its holdout prediction to within 0.0002.

| Step | Change | LB Δ | Local prediction |
|---|---|---|---|
| S1 | queue v5 (physics + shockwave features) | +0.00243 (ΔS_queue +0.0081) | CV +0.005 × 0.30 |
| S2 | onset v6 (trained on corrected labels) | +0.00453 (onset +0.030) | CV onset +0.005 to +0.012 |
| S3 | gate split a = 0.75 | +0.00064 | +0.0006 |
| S4 | Task 1 FD-feature models | +0.00099 | +0.0012 |

**Decomposition of E1 (0.86591).** Only the gating share of B1 is estimated locally; everything else is exact from probes and single-factor deltas.

| Task | Weighted | Task score |
|---|---|---|
| ODME | 0.19876 | 0.994 |
| Task 1 + 3 | ≈ 0.4319 | 0.4287 (A) + gating 0.0016 + split 0.0006 + FD 0.0010 |
| queue | ≈ 0.2352 | S_queue ≈ 0.784: **onset 0.728 (exact)**, ongoing ≈ 0.840 |

The ongoing retrain on corrected labels (v7) was rejected: +0.0006 in the conservative evaluation, and the non-recurrent slice got worse. The label bias sits at queue formation, not inside established queues.

**Next lever: onset.**
- March onset is 0.728, against about 0.76 in CV (old truth) and about 0.86 (corrected truth), so March onsets transfer worst. Diagnose March onset windows next (site recurrence, time of day, incident-like sites) with history-only features.
- Each +0.1 onset is worth +0.015 total. The gap to the leader is 0.0173.

## Decoding calibration (2026-09-25)
**Onset, v6 hybrid OOF on re-drawn windows** (`python -m trafficflow.t2.calib cv` / `decoders`, T2_WORK=t2h):

| logit bias b | hybrid sim | hybrid off | old sim | old off | rec<0.05 | rec<0.2 | cells/window |
|---|---|---|---|---|---|---|---|
| −0.5 | 0.8563 | 0.8841 | 0.7844 | 0.7943 | 0.394 | 0.691 | 3.92 |
| **0 (v6)** | **0.8589** | 0.8841 | **0.7871** | 0.7943 | 0.412 | 0.704 | 4.00 |
| +0.25 | 0.8594 | 0.8841 | 0.7869 | 0.7943 | 0.418 | 0.707 | 4.04 |
| +0.5 | 0.8588 | 0.8928 | 0.7862 | 0.8038 | 0.419 | 0.707 | 4.07 |
| +1.0 | 0.8554 | 0.8970 | 0.7824 | 0.8088 | 0.434 | 0.713 | 4.16 |

- Site-commit decoders cost −0.001 to −0.002. `site2_lo.05_r.5` is −0.0010 ± 0.0012 hybrid and −0.0009 ± 0.0011 old (paired, 94 windows changed: 44 better, 50 worse).
- The contiguous-range and anchor decoders lose heavily (0.795 and 0.562 hybrid).

**On the LB, both directions lose on March:**
- b = +0.5 adds hedge cells: onset −0.016.
- `site2` removes them: onset −0.012.
- **Conclusion:** the v6 top-m decoder at b = 0 is at the March optimum. Onset gains have to come from better probabilities.
- The added cells were mostly extra sites in uncertain windows (D12_I5_N, D12_I5_S, D7_I405_S). Removing the model's own hedges also hurts, so the hedges it picks do hit.

**Ongoing, v5 blend OOF, original windows, old truth** (`calib cv_ongoing`):

| b | −0.5 | −0.25 | **0** | +0.25 | +0.5 | +1.0 |
|---|---|---|---|---|---|---|
| sim | 0.8797 | 0.8827 | **0.8833** | 0.8820 | 0.8788 | 0.8659 |
| off | 0.8790 | 0.8866 | 0.8906 | 0.8950 | 0.8960 | 0.8889 |

- The official train windows prefer larger sets for both conditions. That preference did not transfer to the LB for onset, so it is an artefact of our truth on those windows (5-minute selector shifts), not a property of the official truth.
- No ongoing bias probe was spent.

## Task 3: TV density smoothing (2026-09-25, agent; details in trafficflow/docs/T3_SMOOTHING.md)
**The method:**
- Total-variation smoothing of the per-lane density k = q/v inside each run of consecutive target cells on a link. Small increments inside a run are set to zero; queue fronts are kept.
- The threshold is τ × the run's mean density: free-flow 0.0075, gate (v < 0.6·v_f) 0.02, blackout 0.05. Weak L1 anchors tie the run ends to the observed neighbours.
- Outside the gate only the flow moves; inside it, the a = 0.75 split applies.

**Holdout J with hold3, gate 0.6, a 0.75:**

| | D12_I5_S | D7_I10_W | D7_I405_S | D12_I405_N | mean ΔJ |
|---|---|---|---|---|---|
| baseline J | 0.38810 | 0.38862 | 0.38215 | 0.39293 | |
| TV J | 0.38877 | 0.38930 | 0.38262 | 0.39358 | **+0.00062 (4/4)** |
| LWR baseline → TV | 0.6068 → 0.6132 | 0.5905 → 0.5967 | 0.5985 → 0.6032 | 0.6055 → 0.6116 | +0.0058 |

- Parameters were chosen on 2 panels and confirmed on the other 2.
- Quadratic smoothing gives +0.00018. Moving speed instead of flow gives +0.00021. Density blends are negative.
- Gate 0.7 adds +0.00004 (noise); a = 1.0 fails.

**Where the LWR loss sits (baseline):**
- isolated target cells 46–48%;
- runs of 2–3 cells 36–38%;
- runs of 4+ cells 9–10%;
- blackout runs 5–9%.

Only the within-run transitions (about 29% of the loss) can be smoothed. **LB: G1 +0.00060 against the local +0.00062.**

## Day summary 2026-09-25: E1 0.86591 → G2 0.86711 (+0.0012), rank 9 of 140 post-rebuild
| Step | Change | LB Δ | Local prediction | Verdict |
|---|---|---|---|---|
| F1 | onset logit bias +0.5 | −0.00235 | CV flat, off-windows + | decoder at its optimum |
| F2 | onset site-commit decoder | −0.00173 | CV −0.0010 ± 0.0012 | decoder at its optimum |
| G1 | Task 3 TV density smoothing | +0.00060 | +0.00062 | adopted |
| G2 | onset v8 stacking + seeds | +0.00060 | about +0.0005 | adopted |
| G3 | ongoing v9 stage-2 stacking | −0.00350 | +0.0010 (CV mix) to +0.0018 (reweighted) | failed transfer |

- **Tasks 1/3.** The local proxy stays exact to within 0.0002.
- **Task 2 onset.** Transfers roughly as predicted.
- **Task 2 ongoing.** Gains that exist only on train months don't transfer. On March the stack extended growing queues on D7_I10_E and reshuffled small queues.
- **Next step.** Make the Task 2 gate shift-aware: importance-weighted CV toward the validation and private window distributions, checked against today's four LB outcomes before it is trusted.

## 2026-09-26 chain: G2 0.86711 → H2 0.86777
**Exact decomposition of H2 (0.86777), from P6 and P1:**

| Task | Weighted | Task score |
|---|---|---|
| ODME | 0.19876 | 0.994 |
| Task 1+3 | 0.43268 | S_state ≈ 0.94, S_phys ≈ 0.69 (S_LWR ≈ 0.55) |
| queue | 0.23633 | S_queue 0.7878: onset 0.732 (P4 + G2 Δ), ongoing 0.843 |

**Lessons**
- **Task 1/3.** Local J keeps predicting the LB to within 0.0002: H2 +0.00066 vs +0.00077; P5 −0.00041 vs −0.00062.
- **Ongoing.** Stage-2 re-scoring hurts March even when it may only remove cells (H1b −0.010) or passes the shift-weighted CV and footprint checks. The March ongoing level (0.843) matches the validation-weighted CV (0.849), but no train-based evaluation predicts the *direction* of ongoing edits. From now on, ongoing changes are LB probes first.
- **Leaderboard.** KTK 0.90033 (+0.012 on 25 Sep), gichang 0.89173, Inocchi 0.88635. We are 10th of 151 post-rebuild, gap 0.0326.
  - Where a 0.033 gap could come from: +0.1 S_queue = +0.030; +0.1 S_LWR = +0.010 (ceiling 0.955 against our ~0.55).

## Leaderboard 2026-09-27 00:10 UTC
We are 14th of 170 post-rebuild (20th overall) with H2 = 0.86777. The top post-rebuild teams are KTK 0.90038, gichang 0.89628, Inocchi 0.88635,
Giorgio Ottoboni 0.88458 and Lukas 0.88382 (gap 0.0326). No new candidates were ready on 26 Sep, and the 26 Sep evening sweep ran in plan mode, so nothing ran.

## Task 1 ramp-flow member hold7 (2026-09-27)
On/off-ramp flows (`rflow`, `rvalid`) are released even inside mainline blackouts (80% valid). `TFB_RAMP=1` adds per-link
ramp features (own link, ±3 links up/downstream, previous/next slot, net inflow, and on dark rows the change in net inflow
since the blackout started). Seed 4, otherwise the full4 recipe.

- **Equal-weight gate** (`seed_member.sh`, ens345 → ens3457): ΔJ +0.00012 / +0.00021 / +0.00053 / +0.00071, mean **+0.00039**, 4/4. PASS.
- The member is much better on blackout cells (dark speed RMSE 6.11 vs 6.89), so an equal weight dilutes it. Per-kind weights
  (`t1_weighted_eval.py`; the ramp member gets w_reg on regular cells, w_dark on dark cells, the others share the rest):

| scheme | D12_I405_N | D12_I5_S | D7_I10_W | D7_I405_S | mean J | ΔJ vs ens345 | panels up |
|---|---|---|---|---|---|---|---|
| ens345 (base) | 0.39439 | 0.39001 | 0.39021 | 0.38364 | 0.38956 | – | – |
| equal (0.25 each) | 0.39451 | 0.39022 | 0.39074 | 0.38435 | 0.38996 | +0.00039 | 4 |
| dark 0.5 | 0.39463 | 0.39027 | 0.39109 | 0.38476 | 0.39019 | +0.00062 | 4 |
| dark 0.75 | 0.39476 | 0.39026 | 0.39139 | 0.38498 | 0.39035 | +0.00078 | 4 |
| **dark 1.0** | 0.39474 | 0.39008 | 0.39140 | 0.38527 | 0.39037 | **+0.00081** | 4 |
| reg 0.5 / dark 0.75 | 0.39458 | 0.39007 | 0.39124 | 0.38482 | 0.39018 | +0.00061 | 4 |
| reg 0.5 / dark 1.0 | 0.39455 | 0.38988 | 0.39125 | 0.38511 | 0.39020 | +0.00064 | 3 |

**Chosen for H5:** regular cells = equal mean of full3/4/5/7, dark cells = full7 alone (`t1_pipeline ensw --w-reg 0.25 --w-dark 1.0`).
Expected LB +0.0008. Weighting regular cells towards the ramp member does not help, so its gain is in blackouts.

## Task 1 dark-only booster hold8 (2026-09-27)
Blackout models only (`TFB_KINDS=dark`), ramp features, 127 leaves, lr 0.1, up to 4000 rounds, 200k rows per panel,
seed 5. The hold7 blackout models had all stopped at the 3000-round cap. hold8's dark_speed early-stopped at 2699
rounds with RMSE 5.98, against 6.11 for hold7.

Holdout J with regular cells = mean(hold3, 4, 5, 7) and the blackout rows below (`t1_weighted_eval kinds`):

| blackout rows | D12_I405_N | D12_I5_S | D7_I10_W | D7_I405_S | mean J | ΔJ | dark speed RMSE (same panel order) |
|---|---|---|---|---|---|---|---|
| hold7 (= H5w) | 0.39474 | 0.39008 | 0.39140 | 0.38527 | 0.39037 | – | 5.94 / 6.58 / 6.12 / 8.01 |
| hold8 | 0.39451 | 0.39055 | 0.39141 | 0.38539 | 0.39047 | +0.00009 | 5.80 / 6.19 / 6.22 / 7.91 |
| **mean(hold7, hold8)** | 0.39471 | 0.39046 | 0.39147 | 0.38548 | 0.39053 | **+0.00016** | 5.78 / 6.27 / 6.08 / 7.81 |

- Gate passed with mean(hold7, hold8), up on 3/4 panels.
- The one loss is D12_I405_N (−0.00002), which has no blackouts in validation/private (no Task 2 there).
  On the three panels with test blackouts the gain is about +0.0002.
- H6 = H5w with blackout rows = mean(full7, full8).

### Why the blackout gains transferred badly (holdout rescored with and without blackout cells)

| change | local ΔJ, all cells | local ΔJ, blackout cells excluded | LB Δ |
|---|---|---|---|
| H3 → H5w | +0.00081 | +0.00008 | +0.00039 |
| H5w → H6 | +0.00016 | 0 | +0.00001 |

- Blackout cells do count on the LB. Without them, H5w would have gained only about +0.0001.
- Their gains transfer unreliably: about 50% for H5w's ramp member and about 5% for H6's booster.
- The holdout has only 10 blackouts per panel (40 in all, one of them on D12_I405_N, which has none in the test).
  Blackout ΔJ estimates therefore rest on a few events, while regular-cell estimates use hundreds of thousands of cells.
- **Rule from now on:** discount blackout-only ΔJ to about 1/3 when predicting the LB. Before investing in blackout
  models again, widen the holdout to all holdout-period origins so the dark estimate is precise.

## Day summary 2026-09-27: H2 0.86777 → H5w 0.86838 (+0.00061)

| # | Submission | Public | Δ vs best | Decision |
|---|---|---|---|---|
| S1 | H3: third Task 1 seed | 0.86799 | +0.00022 (local +0.00023) | adopted |
| S2 | G7: ongoing retrained on hybrid labels | 0.86779 | −0.00020 | rejected; ongoing label work closed |
| S3 | **H5w: ramp-flow member, blackout-weighted** | **0.86838** | +0.00039 (local +0.00081) | **adopted (best)** |
| S4 | H6: dark-only booster on blackout rows | 0.86839 | +0.00001 (local +0.00016) | rejected |
| S5 | G8: ongoing capacity p3 (v11) | 0.86798 | −0.00041 (CV +0.005, full gate passed) | rejected; robust list |

**Lessons**
- **Regular-cell Task 1 changes transfer, blackout-cell changes do not reliably.** The holdout has only 10
  blackouts per panel, so dark ΔJ is noisy: H5w transferred about 50% of its dark part, H6 about 5%.
- **Ongoing CV improvements do not transfer to March.** Four of the last five ongoing changes lost on the LB
  (G3 −0.023, H1b −0.010, G7 −0.0013, G8 −0.0027 in ongoing IoU): stacking, cell removal, labels and now capacity.
  All passed plain CV, and H1b and G8 passed the shift-weighted and footprint checks too. Only v5 (+0.016) transferred.
  The March ongoing windows reward something our CV does not measure. Until that is understood, ongoing edits
  have no reliable local gate.

## Task 1 regular booster hold9 (2026-09-27; H7 for 28 Sep)
Regular-cell models only (`TFB_KINDS=reg`, `reg_member.sh`), seed 6, 300k rows per panel (the other members use
150k), lr 0.05, up to 6000 rounds, no ramp features. Single-model holdout RMSE against hold7:

| model | hold7 (best iter) | hold9 (best iter) | change |
|---|---|---|---|
| speed | 1.5067 (970) | 1.4388 (3649) | −4.5% |
| flow per lane | 30.16 (1110) | 29.37 (4533) | −2.6% |
| density | 0.5558 (3000, cap) | 0.5167 (5999, cap) | −7.0%, still at the cap |

Holdout J, blackout rows fixed (= hold7) so only the regular change is measured (`t1_weighted_eval regs`):

| regular rows | D12_I405_N | D12_I5_S | D7_I10_W | D7_I405_S | mean J | ΔJ | panels up |
|---|---|---|---|---|---|---|---|
| mean(hold3, 4, 5, 7) (= H5w) | 0.39474 | 0.39008 | 0.39140 | 0.38527 | 0.39037 | – | – |
| add hold9 (equal weight) | 0.39502 | 0.39043 | 0.39161 | 0.38558 | 0.39066 | +0.00029 | 4 |
| **0.5·hold9 + 0.5·mean(old)** | **0.39526** | **0.39081** | **0.39174** | **0.38579** | **0.39090** | **+0.00053** | **4** |
| hold9 alone | 0.39520 | 0.39084 | 0.39150 | 0.38574 | 0.39082 | +0.00045 | 4 |

- The regular speed RMSE of the ensemble drops from 1.248 / 1.732 / 1.311 / 2.044 to 1.218 / 1.692 / 1.301 / 2.002.
- More data plus a slower learning rate beats seed averaging: one boosted model is better than the 4-seed ensemble.
- Next steps are more boosted seeds, and more rounds for the density model, which is still at the cap.
- H7 = H5w with regular rows = 0.5·full9 + 0.5·mean(full3, 4, 5, 7). Expected LB about +0.0005, since regular-cell changes transfer exactly.

H7 build (27 Sep 17:28): `H7_reg9.zip`, 65/65 checks. Against H5w, 6,613,327 regular state rows change (mean |Δv| 0.091 km/h,
|Δq| 9.7 veh/h); blackout rows, queue and ODME are identical. Predicting with the boosted models is slow: about 3 h for
the test rows when two predictions share the CPU.

## Leaderboard 2026-09-27 21:15 UTC
We are 15th post-rebuild with H5w = 0.86838 (the raw best, H6 = 0.86839, is the same within noise). The top post-rebuild
teams are KTK 0.90085, gichang 0.89628, George Daniel Gherasim 0.89145 (new), Inocchi 0.88697 and Giorgio Ottoboni 0.88548;
the gap to #1 is 0.03246. Two container restarts today (about 13:00 and 17:40–21:14) cost the hold10 run; it
restarted at 21:15.

## Second boosted regular seed hold10 (2026-09-27 23:19; H8 for 28 Sep)
Seed 7 with the hold9 settings, except that the density model may run to 10000 rounds. Single-model holdout RMSE: speed 1.4412
(hold9 1.4388), flow 29.32 (29.37), density **0.5047** (0.5167; best iteration 9989, still near the cap).

Holdout J, blackout rows fixed (= hold7), reference = the H7 scheme:

| regular rows | D12_I405_N | D12_I5_S | D7_I10_W | D7_I405_S | mean J | ΔJ | panels up |
|---|---|---|---|---|---|---|---|
| H7: 0.5·hold9 + 0.5·mean(hold3, 4, 5, 7) | 0.39526 | 0.39081 | 0.39174 | 0.38579 | 0.39090 | – | – |
| 0.25·hold9 + 0.25·hold10 + 0.5·old | 0.39535 | 0.39093 | 0.39181 | 0.38591 | 0.39100 | +0.00010 | 4 |
| 0.35·hold9 + 0.35·hold10 + 0.3·old | 0.39551 | 0.39112 | 0.39186 | 0.38601 | 0.39112 | +0.00022 | 4 |
| **0.5·hold9 + 0.5·hold10 (old members dropped)** | **0.39556** | **0.39124** | **0.39181** | **0.38602** | **0.39116** | **+0.00026** | **4** |

- Regular speed RMSE (same panel order): 1.218 / 1.692 / 1.301 / 2.002 → 1.203 / 1.660 / 1.301 / 1.982.
- The old 150k-row members no longer add anything once there are two boosted seeds.
- Next: a third boosted seed, and the density model beyond 10000 rounds.
- H8 = H5w with regular rows = 0.5·full9 + 0.5·full10 (blackout rows full7). Local ΔJ vs H7 +0.00026; vs H5w about +0.0008.

### Why H7 transferred at ~50% (local ΔJ split into its S_state and LWR parts)

| change | local ΔJ | from S_state (0.35·ΔS) | from LWR (0.10·ΔLWR) | LB Δ | reading |
|---|---|---|---|---|---|
| H2 (seed averaging) | +0.00077 | +0.00024 | +0.00053 | +0.00066 | the LWR part transferred at about 80% |
| H7 (boosted member, better density model) | +0.00053 | +0.00022 | +0.00031 | +0.00026 | the LWR part transferred at about 15% |
| H8 (second boosted seed), prediction | +0.00026 | +0.00013 | +0.00013 | +0.00013 to +0.00026 | mixed: seed averaging plus density |

- LWR gains from variance reduction or smoothing (H2, G1) transfer.
- LWR gains from a more accurate density model in congested (gated) cells mostly don't. The February holdout is more
  congested than March on some panels (D12_I5_N, D12_I5_S), and the gated cells are where the density model acts.
- **Proxy rule from now on:** predict Task 1 LB Δ as 0.35·ΔS_state plus the LWR part discounted to about 15% when it
  comes from density accuracy.

### Capacity gains do not transfer; variance reduction does (28 Sep)

| change | kind | local ΔJ | LB Δ | transfer |
|---|---|---|---|---|
| H2: mean of 2 seeds | variance reduction | +0.00077 | +0.00066 | ~85% |
| H3: 3rd seed | variance reduction | +0.00023 | +0.00022 | ~95% |
| G1: TV smoothing | post-processing | +0.00062 | +0.00060 | ~100% |
| H7: boosted member (2× rows, lr 0.05) at 0.5 | capacity / data | +0.00053 | +0.00026 | ~50% |
| H8: 2 boosted seeds, old members dropped | capacity / data | +0.00026 | +0.00004 | ~15% |

The holdout is the last 30 train days, which share the train simulation's demand draw. March and April are
independent draws ("their own demand draws and incident schedules", DATA.md). More capacity or more rows fit
train-draw-specific structure that the holdout rewards and the test months do not. Seed averaging and smoothing only
remove variance, so they transfer.

**Proxy rule:** discount capacity/data-driven local gains to about 30%; count variance-reduction and post-processing
gains in full.

**Consequence for the protocol:** a holdout that is an independent draw would measure transfer directly. None
exists in train, but the observed (non-target) cells of March/April could serve as a transductive check (backlog).

## Test-month pseudo-holdout (2026-09-28, `t1_pseudo.py`)
In each panel, 20k observed, eligible, non-target, non-blackout cells of March and 20k of April are hidden. They are
predicted the way a regular target is: panel built as for the test rows, features recomputed with the cells hidden.
Each scheme's regular rows are scored with the official S_state formula (gated reconciliation, no TV smoothing).
Only released data is used; the cells are about 1% of the observed cells.

| step | train holdout, local ΔJ | pseudo March, 0.35·ΔS_state | pseudo April | LB (March) |
|---|---|---|---|---|
| H5w → H7 | +0.00053 | +0.00013 (9/10 panels) | +0.00010 (9/10) | +0.00026 |
| H7 → H8 | +0.00026 | −0.00001 | 0.00000 | +0.00004 |

- The pseudo-holdout gets both steps right. It measures the S_state part only, which is about half of H7's LB gain; the rest is LWR/smoothing.
- The train holdout overstated both steps.
- March regular-cell RMSE: speed 1.87 → 1.86, flow per lane 31.9 → 31.7 (H5w → H7).
- **Gate from now on for Task 1 regular-cell changes:** the pseudo-holdout on both months, not the train holdout.

## Transductive Task 1 member fullP (2026-09-28)
Training rows are the usual train rows plus 90k observed cells per month and panel from March and April
(`t1_pseudo.build_train_rows`, `TFB_PSEUDO=1`). They are sampled disjoint from the 40k evaluation cells. Features are
built in rounds of 30k hidden cells (about 5% extra masking), and the labels are the released observed values.
Config: standard regular models (lr 0.1, 150k train rows per panel, rounds from hold7), seed 11, no ramp features.

Pseudo-holdout S_state (regular rows, gated reconciliation), mean over 10 panels:

| scheme | March | April | 0.35·Δ vs H7 (March / April) | panels up vs full3 |
|---|---|---|---|---|
| full3 / full4 / full5 (single standard models) | 0.93368 / 0.93376 / 0.93377 | 0.93211 / 0.93259 / 0.93274 | – | – |
| H7 (0.5·full9 + 0.125·full3/4/5/7) | 0.93522 | 0.93406 | 0 | 10 / 10 |
| **fullP alone** | **0.94069** | **0.93943** | **+0.0019 / +0.0019** | **10 / 10** |
| 0.5·full9 + 0.5·fullP | 0.93928 | 0.93812 | +0.0014 / +0.0014 | 10 / 10 |
| H7 with fullP added at 0.1 | 0.93625 | 0.93508 | +0.0004 / +0.0004 | 10 / 10 |

- A single transductive model beats every train-only ensemble by a wide margin, in both months and on every panel.
- This is the first Task 1 change of this size since the FD features (E1).
- Leakage checks:
  - evaluation cells are excluded from the training rows;
  - evaluation cells are hidden when their own features are built;
  - no feature carries the absolute date or day index, so the model can only learn the test months' general
    relations (demand level, neighbour structure), not memorise cells.
- Task 1 is offline reconstruction; the ≤ T rule is Task 2's.
- H9P = H7 with regular rows = fullP; blackout rows stay full7.

### More transductive data (fullP2: 2× the test-month rows, seed 12)
Pseudo-holdout, 0.35·ΔS_state vs fullP alone (March / April, panels up):

| scheme | March | April |
|---|---|---|
| fullP2 alone | +0.00043 (10) | +0.00056 (9) |
| mean(fullP, fullP2) | +0.00050 (10) | +0.00061 (10) |
| **0.35·fullP + 0.65·fullP2** | **+0.00054 (10)** | **+0.00067 (10)** |

- More test-month rows keep helping (1× → 2×: about +0.0005), and averaging adds a little.
- H10P = H9P with the weighted pair.
- A third disjoint set is built; fullP3 (3×, seed 13) is chained after H10P (`/home/user/work/chain_H11P.sh`).
- 733 MB were freed by deleting `/home/user/work/t2/_xev_11368.npy`, a temp file the killed p4 CV left behind.

### 3× transductive rows (fullP3, seed 13) and the day's transductive curve
Pseudo-holdout vs H10P, 0.35·ΔS_state (March / April, panels up): fullP3 alone +0.00013 (7) / +0.00008 (8);
**0.5·fullP2 + 0.5·fullP3 +0.00024 (10) / +0.00026 (9)**; 0.2·P + 0.35·P2 + 0.45·P3 +0.00023 (10) / +0.00026 (10).
Returns from more rows alone are flattening (1× → 2× gave +0.0005, 2× → 3× +0.0001); averaging members adds the rest.

| step | pseudo (S_state only) | LB Δ | LB / pseudo |
|---|---|---|---|
| H7 → H9P (1×) | +0.0019 | +0.00229 | 1.2 |
| H9P → H10P (2×) | +0.00054 | +0.00109 | 2.0 |
| H10P → H11P (3×) | +0.00025 | +0.00047 | 1.9 |

The LB gain includes an LWR part (better density in the test months) that the pseudo-holdout does not measure, so the
pseudo figure is a conservative lower bound for transductive changes.

## Test-month blackout pseudo-holdout (2026-09-28, `t1_pseudo.py dark`)
Queue-like origins are found in the observed March/April data with the selector replica on the masked view, away from
the released blackouts. At most 40 per month and panel are kept, and rows T+1..T+18 are blanked at each, as in the
release. Features are built for the eligible observed cells in those rows. Origins alternate between an evaluation half
and a training half (disjoint events): 800 simulated blackouts in all, 1.18M blackout cells, 585k of them for
evaluation. The history rows T−12..T−1 stay in the masked view; in a real window they are fully observed, so the context
is slightly harder.

Baseline blackout RMSE (cell-weighted over panels):

| model | speed, March / April | flow per lane, March / April |
|---|---|---|
| full7 (current blackout rows, ramp-flow member) | 7.33 / 6.89 | 81.2 / 72.0 |
| hold7 (same, train days < 243) | 7.35 / 6.95 | 81.1 / 72.4 |

**fullPD** (full7's settings plus the training half of these blackouts, `TFB_PSEUDO_DARK=1`, seed 15):

| model | speed, March / April | flow per lane, March / April |
|---|---|---|
| full7 | 7.33 / 6.89 | 81.2 / 72.0 |
| **fullPD** | **6.42 / 5.91** (−12% / −14%) | **76.1 / 63.5** (−6% / −12%) |
| 0.5·full7 + 0.5·fullPD | 6.72 / 6.21 | 77.2 / 66.2 |

- **Expected LB gain, from the test's blackout share (about 2.1% of targets):** speed RMSE per regime 2.108 → 2.045
  (ΔS_state about +0.0014 on the 8 blackout panels), and a smaller gain on flow. S_total about +0.0004 to +0.0006.
- H12P = H11P with blackout rows = fullPD.
- fullPB (boosted config plus 3 row sets) was OOM-killed at 10.3 GB while fullPD trained alongside. Run it alone.

### Second simulated-blackout set (fullPD2, 2026-09-28)
`dark2`: up to 80 more queue-like origins per month and panel, disjoint from set 1 (≥ 18 slots away), all used for
training; 1,103 more blackouts in 9 panels. fullPD2 = full7's settings, seed 16, trained on the set-1 training half
plus dark2. Evaluation is set 1's evaluation half, as before.

| blackout rows | speed, March / April | flow per lane, March / April |
|---|---|---|
| full7 (in H11P) | 7.33 / 6.89 | 81.2 / 72.0 |
| fullPD (in H12P) | 6.42 / 5.91 | 76.1 / 63.5 |
| fullPD2 | 6.30 / 5.77 | 75.9 / 62.8 |
| **mean(fullPD, fullPD2)** | **6.27 / 5.75** | **75.3 / 62.3** |

H13P = H12P with blackout rows = mean(fullPD, fullPD2): about 2.5% lower blackout speed RMSE than H12P, worth about +0.0001.
