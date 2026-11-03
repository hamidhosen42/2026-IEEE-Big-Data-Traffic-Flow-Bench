# Single-entry reproduction of the final picks — design (2026-10-09)

Goal: one entry script (plus README and environment) that the organizers can rerun to reproduce the selected
submission within 1 % of its private score (about 0.0088). Code and PDF report are due Nov 10
(`research/organizer_verified_facts.md` item 11).

> **Update 2026-11-03: the final picks are now A3 = `tfb-k9` and B3 = `tfb-k10` (both public 0.88281).** They differ from
> A2 / B2 only in the Task 1 state rows (3 more members + LRTC steps M / B); every queue and odme row is the same. See **§8**
> for the delta to this design (stages, runtime, RAM, determinism, open items) and `LINEAGE_K9.md` for the lineage.
> §0–§7 below describe A2 / B2 and stay valid for everything A3 / B3 share with them.

Target recipe: **A2 = tfb-j4**, i.e. tfb-i4 with the April ongoing rows of tfb-t2ogpvogf. Optional recipe:
**B = tfb-j3**, the hedge. Internal target: stay within about **0.002** of the selected file, so that hardware
nondeterminism still has room inside the 1 %.

Companion files:
- `lineage_A2.json`: graph of 88 kernels, versions, Docker images, runtimes, peak RAM, recipes and hidden-input hashes.
- `lineage_src/`: exact sources of all 88 kernels. 21 were added today: tfb-h1…tfb-j3, t1ramp/ramp2/bspec, t2ogfa/b/c, t2ogveto, t2ogpvogf.
  The other 67 sources are byte-identical to the G4 trace.
- `LINEAGE_G4.md`: the older G4 map.

---

## 0. Summary

- **What decides the plan: Task 2 private rows are lumpy, Task 1 rows are smooth.**
  - One official window is worth 0.00375 per unit of IoU (8 panels × 5 windows; `analysis/TRANSFER_ANALYSIS_2026-10-08.md`).
  - So one flipped cell in an onset window (median 3 cells, a flip moves IoU by 0.3–0.6) costs **0.001–0.002 of private score**. That is the whole 0.002 budget.
  - Task 1 changes average over 3.28 M private state rows. Their effect is predictable, and it is the same on public and private, because the same models produce both months.
  - **Rule:** reproduce every T2 private row and T4 exactly, and simplify Task 1 only. T1 is 90 % of the compute: 96 of about 107 Kaggle CPU-hours.
- **Lean recipe (lean-1, recommended).**
  - Task 1:
    - old FTA bag 8 → 4 (keep v12s4–s7);
    - FD bag 6 → 4 (v13, s10, s11, t1ramp; drop t1dark and t1ramp2);
    - t1bspec on the blackout rows (W 0.75) unchanged;
    - recon + TV unchanged.
  - T2 and T4 exactly as A2.
  - Expected private Δ vs the A2 file: **−0.0002 to −0.0003** (range −0.0005 to 0) from T1, plus up to about ±0.0003 from T2/T1 nondeterminism. Worst plausible case, one onset window flipping: ≈ 0.0025, still about 3.5× inside 1 %.
- **Runtime (8 cores / 32 GB, Linux x86-64).**
  - Lean-1 A2: about **42–47 h** (≈ 2 days; Kaggle equivalent ≈ 72 CPU-hours at 4 vCPU).
  - Full A2: about 63–68 h.
  - B with lean T1: about 49 h. A2 and B from one run (shared stages): about 55 h.
- **Main risks.** Each is detailed in §4.6 and §7.
  1. Porting bugs. Mitigation: member-level parity runs.
  2. Two Kaggle images were used: A (Python 3.12) for the 8 old FTA members, v13, the pseudo windows and T4; B (Python 3.13) for everything else. Package versions were never logged.
  3. T2 tie and nondeterminism flips.
  4. 32 GB RAM is borderline: T1 members peaked at 26.6–28.0 GB on Kaggle.
  5. A2's producer tfb-t2ogpvogf is still running today. Its veto count decides whether pvo (2.4 h) is needed.
  6. Nov 7–10 is too short to validate after the private LB. Both A2 and B must be validated before Nov 7.

---

## 1. Updated lineage

### 1.1 Chain since G4 (traced with `GET /kernels/pull`, kernelDataSources)

Each step below swaps a subset of rows; all other rows are carried unchanged and asserted:

| Step | Kernel | What changed |
|---|---|---|
| 1 | `tfb-g4` | queue2 on g3: private onset ← tfb-t2opv `pv`; private ongoing ← tfb-t2ogpvaf `pvo` |
| 2 | `tfb-h1d` → `tfb-h1` | raw T1 = 0.5·FD3 + 0.5·old8; blackout rows 0.25 mix + 0.75 t1ramp; recon + TV |
| 3 | `tfb-h5d` → `tfb-h5` | FD bag 5 (+ t1dark, t1ramp) on all rows |
| 4 | `tfb-i2d` → `tfb-i2` | FD bag 6 (+ t1ramp2); blackout member = mean(ramp, ramp2) |
| 5 | `tfb-i4d` → **`tfb-i4`** | blackout member → t1bspec (a + b + c). **A**, public 0.88176 |
| 6a | **`tfb-j3`** | queue2 on i4: private onset ← osp `base`; private ongoing ← ogveto `veto10`. **B**, public 0.88176; 0 validation rows / 369 private rows changed |
| 6b | **`tfb-j4`** (planned, chain oct9 Q4, `p9b_planner.py`) | queue2 on i4: private ongoing ← tfb-t2ogpvogf `queue_submission_pvogfv.csv` (fallback `pvogf.csv`); max_val_changed 0. **A2** |

- The T1 swaps (i4d etc.) **recompute every state row** from the member files. The base file contributes only its queue and path_flow rows.
- Queue rows have therefore been fixed since g4, apart from the j3/j4 swaps. path_flow has been fixed since tfb-final.

### 1.2 Effective producers per row block

Submission = 6,980,503 rows:
- state 6,735,795 (validation 3,451,376, private 3,284,419);
- queue 174,000 (43,500 per condition × split);
- odme 70,708.

| Rows | A (i4) | A2 (j4) | B (j3) | Recipe (constants) |
|---|---|---|---|---|
| **T1 state**, validation + private | `tfb-i4` on `tfb-i4d` | = A | = A | see T1 recipe below |
| **Blackout rows** (subset of T1): 128,465 = 1.91 % | `tfb-i4d` | = A | = A | timestamp in (origin, origin + 90 min] of any official T2 window of the panel (validation or private). Value = 0.25·raw mix + 0.75·t1bspec (parts a/b/c cover disjoint panels) |
| **Onset, March** (validation) | `tfb-t2osp` `sp4` (via e5) | = A | = A | CatBoost (600 it, depth 8, lr 0.08, `thread_count 4`, `random_seed 0`) on train onset rows (v13 features, `rng(0)` permutation of 310,782 rows). The 3 prior columns are blended per cell with the scenario prior of March days < d: slot ±2, weekday class, a = n/(n+4). Expected-IoU top-m decode |
| **Onset, April** (private) | `tfb-t2opv` `pv` | = A | `tfb-t2osp` `base` | **pv:** mean of 3 CatBoost models (same params) on train rows + released-March onset windows, March weights 10 / 20 / 40; April features carry the sp4 blend. 159 positives. **base:** same CatBoost, train prior. 157 positives; 48 rows differ from pv |
| **Ongoing, March** | `tfb-t2occ` `fs4occ` (via g3) | = A | = A | F03 bag of 4 LightGBM: np511 + id511 × seed pair {default, (1, 11, 21)}. 511 leaves, min_data 40, lr 0.03, 1800 trees, `num_threads 4`. Selection-faithful windows (t%3 weight 0.3, run-starts 1.0; fill ≤ 36 slots, persistence cap 0.9; MAX_OG 3000); all-finite inputs + 13 occupancy features; mean probability → expected-IoU decode |
| **Ongoing, April** | `tfb-t2ogpvaf` `pvo` | **`tfb-t2ogpvogf` `pvogfv`** | `tfb-t2ogveto` `veto10` | **pvo:** 0.25 npB + 0.25 idB + 0.5 fullB (511 leaves, min_data 40, lr 0.05, 900 trees, default seeds, 4 threads). Trained on F03 train rows + released-March faithful windows (weight × 5), all-finite inputs; fullB gets the scenario-prior blend (K 4). **pvogfv:** pvo + 103 ogfeat columns in all 3 members; a window keeps the pvo rows if it adds > 10 cells outside the queue at T. **veto10:** fs4ogf (mean of 4 ogfeat models, t2ogfa + t2ogfb) unless the window adds > 10 cells outside qlast vs fs4occ (private: 1 window vetoed) |
| **T4 path_flow** | `tfb-final` cell 7 | = A | = A | `scipy.optimize.nnls` on [A_meas; √0.05·I] · f = [counts; √0.05·prior], per panel and split |

**T1 recipe (A, A2 and B are identical).** Producer `tfb-i4` = `swap_fdrecon.py` applied to `tfb-i4d` = `t1_t1dark_swap.py`.

- **Ordinary rows:** raw = 0.5·mean(FD bag) + 0.5·mean(old FTA bag), i.e. **W_FD 0.5**.
  - Old FTA bag = v11, v12, v12s2–s7 (8 members).
  - FD bag = v13, v13s10, v13s11, t1dark, t1ramp, t1ramp2 (6 members).
- **Blackout rows:** **W_DARK 0.75** toward t1bspec.
- **Recon:**
  - GATE 0.6, A 0.75, LOGR_MAX 0.5;
  - link exclusion when w ∉ [8, 40] km/h or kj/lane ∉ [60, 200] (10 links excluded);
  - panel exclusion when the median exceeds 2 %, the IQR is outside ±5 % or n < 200 (0 panels excluded);
  - 3.98 % of rows changed.
- **TV:** free 0.0075 / anchor 0.001, gate 0.02 / anchor 0.005, 300 iterations, runs break at gaps > 5 min.

### 1.3 Task 1 members (all LightGBM regression)

Settings shared by every member:
- two models per panel, speed/vf and flow/lane residual to temporal interpolation;
- 511 leaves, min_data 100, lr 0.04, 1500 trees, max_bin 63, ff 0.8, bf 0.8, `num_threads 4`;
- MAX_ROWS 600k, train rows capped at 4× that;
- transductive rows: released target-month cells hidden 20 % per round;
- RNG re-seeded **inside the panel loop** (checked), so per-panel outputs do not depend on the a/b split.

| Member | Code family | Row rng | Rounds (rng base) | TX cap | Holdout | LGB seeds | Image | Kaggle s (a + b [+ c]) | Peak GB (a / b / c) |
|---|---|---:|---|---:|---:|---|---|---|---|
| v11 | FTA (also trains unused FT) | 0 | 2 (1000) | 4× | 0.15 | default | A | 20200 + 14196 (FTA-only ≈ 20.5k) | not logged |
| v12 | FTA | 1 | 2 (2000) | 4× | 0.15 | 1/11/21 | A | 11811 + 8656 | – |
| v12s2 | FTA | 2 | 2 (3000) | 4× | 0.15 | 2/12/22 | A | 11009 + 5380 | – |
| v12s3 | FTA | 3 | 2 (4000) | 4× | 0.15 | 3/13/23 | A | 11126 + 8361 | – |
| v12s4 | FTA | 4 | 3 (5000) | 6× | 0.15 | 4/14/24 | A | 12545 + 8874 | – |
| v12s5 | FTA | 5 | 3 (6000) | 6× | 0.15 | 5/15/25 | A | 9577 + 9085 | – |
| v12s6 | FTA | 6 | 3 (7000) | 6× | **0.001** (no holdout) | 6/16/26 | A | 12930 + 8746 | – |
| v12s7 | FTA | 7 | 4 (8000) | 8× | 0.15 | 7/17/27 | A | 13794 + 9666 | – |
| v13 | FD (+20 FD columns) | 9 | 3 (10000) | 6× | 0.15 | 9/19/29 | **A** | 14046 + 10716 | 26.6 / 19.3 |
| v13s10 | FD | 10 | 3 (11000) | 6× | 0.15 | 10/20/30 | B | 14041 + 7293 | 26.9 / 19.7 |
| v13s11 | FD | 11 | 3 (12000) | 6× | 0.15 | 11/21/31 | B | 15267 + 7690 | 27.0 / 19.7 |
| t1dark | DARK (+ simulated-blackout rows, DARK_SEED 1300) | 13 | 3 (14000) | 6× | 0.15 | 13/23/33 | B | 15334 + 11850 | 27.1 / 19.6 |
| t1ramp | RAMP (= dark + 9 ramp columns, same seed) | 13 | 3 (14000) | 6× | 0.15 | 13/23/33 | B | 16717 + 12194 | 27.9 / 20.8 |
| t1ramp2 | RAMP, seed 17 (DARK_SEED 1700) | 17 | 3 (18000) | 6× | 0.15 | 17/27/37 | B | 16249 + 11655 | 27.9 / 21.0 |
| t1bspec | BSPEC (+ faithful blackouts, 30 timing columns, no transductive rounds) | 13 | – | 6× | 0.15 | 13/23/33 | B | 14035 + 13266 + 4541 | 28.0 / 25.0 / 20.0 |

- Halves: a = panels[:5], b = panels[5:]. bspec: a = panels[:4], b = panels[5:], **c = D7_I405_N only**, with MAX_ROWS 420k and 60 % of the train days (day index mod 5 ∈ {0, 1, 3}), because of the 30 GB OOM.
  - **c must be reproduced with exactly these reductions**, even on a bigger machine.
- Within one code family the sources differ only in constants: 6–16 code lines (ast diff, docstrings removed).
- Across families the extra feature code is 67 lines (FTA→FD), 81 (FD→DARK), 83 (DARK→RAMP) and 184 (RAMP→BSPEC).
- **Hidden input of every member:** `tfb-t2lab-full` `pseudo_<panel>.npz`. These are the train pseudo-window origins (d, t, cond) used for the simulated 90-min training gaps; bspec also uses cond.
  - They are deterministic: `build_windows(P, cap=None)` of `kaggle_jobs/t2lab.py`, 203 s.
  - Expected window counts and SHA-1 hashes are in `lineage_A2.json → hidden_dependencies`. Example: D7_I10_E 494, D7_I210_E 715, D7_I405_N 728 windows.
  - D12_I405_N/S have no Task 2 windows, so they have no file.

### 1.4 Other hidden or optional inputs

| Input | Used by | Entry-script action |
|---|---|---|
| tfb-t2occ `t2v26occ_probs.npz` (qlast) + `fs4occ` | t2ogveto (B veto10) | produce in the ongoing-validation stage and cache |
| tfb-t2ogpvaf `pvo` | pvogfv veto (A2) | needed only if the full pvogf run vetoes ≥ 1 private window (smoke: 0/10). Record the count when tfb-t2ogpvogf finishes |
| tfb-t2v22af `fs4af` | pvaf and pvogf copy their **validation** rows from it; the kernels assert it exists | drop. Validation ongoing comes from fs4occ, and the final file never uses these rows |
| tfb-t2v13, tfb-e3, tfb-f5, tfb-t2v21 | CONTROL lines only (try/except) | drop |

---

## 2. Component table

Runtime is the last log timestamp on Kaggle CPU (4 vCPU, 30 GB). Peak RAM is `peak RSS` as printed by the kernel. "est." means it was not logged and is estimated from matrix sizes.

| # | Component | Source (`kaggle_jobs/` = `lineage_src/`) | Inputs (hidden in bold) | Kaggle runtime | Peak RAM | Seeds / threads | Img |
|---|---|---|---|---|---|---|---|
| 1 | Pseudo windows | `t2lab.py` (tfb-t2lab-full) | train mainline_states, topology, fd_parameters, train window_index | 203 s (builder only ≈ 100 s) | small | none (deterministic) | A |
| 2 | Old FTA members × 8 | `t1_pipeline_v11a/b`, `v12*a/b` | release (train full + masked, val/private masked, task1 templates, fd_parameters) + **pseudo npz** | 4.6–6.5 h per member (v11 9.6 h incl. FT); 45.0 h total (v11 FTA-only) | not logged; ≈ v13 (26.6 / 19.3) | §1.3; LGB 4 threads | A |
| 3 | FD members × 3 | `t1_pipeline_v13a/b`, `v13s10*`, `v13s11*` | same | 5.9–6.9 h each; 19.2 h | 26.6–27.0 / 19.3–19.7 | §1.3 | A (v13), B |
| 4 | t1dark / t1ramp / t1ramp2 | `t1_t1dark_member_*`, `t1_ramp_member_*`, `t1_ramp2_member_*` | + task2 window_index (official origins), ramp_states, ramp_attachment_map | 7.6 / 8.0 / 7.8 h | 27.1–27.9 / 19.6–21.0 | §1.3 | B |
| 5 | t1bspec | `t1_bspec_member_a/b/c` | same | 3.9 + 3.7 + 1.3 = 8.8 h | 28.0 / 25.0 / 20.0 | seed 13, DARK_N 48, DARK_SEED 1300 | B |
| 6 | T1 mix + blackout override | `t1_t1dark_swap.py` (tfb-i4d) | member `state_submission_FTA.csv`, submission_key, window_index | 444 s | est. ≈ 10 GB | none | B |
| 7 | FD recon + TV | `swap_fdrecon.py` (tfb-i4) | raw mix, fd_parameters, masked layers (TV anchors, exclusion diagnostics) | 499 s | est. 6–10 GB | none | B |
| 8 | T2 onset: base, sp4, pv | `t2_onset_pv.py` (tfb-t2opv); `t2_onset_sp.py` (tfb-t2osp) gives base/sp4 identically | release (task2 window_index/history/templates, masked months for the scenario prior, train) | 523 s (+ 333 s) | est. < 8 GB | CatBoost seed 0, thread_count 4, `rng(0)` permutation. CONTROL vs t2v13 = 1.000000 (exact across images A→B) | B |
| 9 | Ongoing March, fs4occ | `t2_occ_v26.py` (tfb-t2occ) | release | 19,630 s (4 models × 72–80 min) | est. 12–16 GB (X 11.8 M × 77 float32) | default seeds / (1, 11, 21); 4 threads; window subsample `default_rng(0)`, `default_rng(1)`, slot-T draw `default_rng(len(wins))` (same in pvaf / pvogf / ogfeat) | B |
| 10 | Ongoing April, pvo | `t2_ongoing_pvaf.py` (tfb-t2ogpvaf) | release (+ **fs4af**, validation copy only) | 8,647 s (3 × 37–46 min) | est. 12–16 GB | default seeds; 4 threads | B |
| 11 | Ongoing April, pvogf / pvogfv (A2) | `t2_ongoing_pvogf.py` (tfb-t2ogpvogf) | release + **pvo** (veto) (+ fs4af validation copy) | **running**; est. 4–5.5 h | est. 20–26 GB (X ≈ 8.8 GB) | default seeds; 4 threads | B |
| 12 | Ongoing April, veto10 (B) | `t2_ogfeat_v27a/b.py`, `t2_ogfeat_comb.py`, `t2_ogveto.py` | release + **fs4occ, qlast** | 18,352 + 22,927 + 26 + 26 s (11.5 h) | est. 20–25 GB (X 8.71 GB logged) | seed 0 pair / (1, 11, 21); 4 threads | B |
| 13 | T4 NNLS | `TFB_final_notebook.ipynb` cell 7 (tfb-final) | path_set, path_link_incidence, task4 counts + prior | ≈ 800 s of the 8,497 s kernel | small | deterministic (scipy) | A |
| 14 | Queue swaps / assembly | `swap_queue2.py` (j3, j4) | – | 97 s | small | none | B |

Sums:
- Full A2 ≈ 96.4 h (T1) + 0.3 + 0.15 + 5.45 + 4–5.5 (+ 2.4 for pvo) + 0.25 ≈ **107–111 Kaggle-hours**.
- B ≈ 96.4 + 0.3 + 0.15 + 5.45 + 11.5 + 0.25 ≈ 114 h.

---

## 3. Simplification options and estimated private-score impact

### 3.1 How the estimates are made

- **T1 options.** Each one is priced from the LB delta measured when that member was *added* (TRACKER). The bag-variance term is scaled by the member's current weight.
  - Example: old bag 4 → 8 alone gave +0.00065 (c16). Inside today's 0.5 / 0.5 mix, the idiosyncratic variance term changes by 0.25·(1/4 − 1/8) σ², a quarter of the original 0.125 σ². So 8 → 4 costs ≈ 0.00016.
  - The 4 dropped members (2 transductive rounds) are also ≈ 0.0005 holdout S_state worse each than the kept 3–4-round members, which buys back ≈ +0.00004.
  - T1 public deltas transfer to private: same models, millions of cells.
- **T2 options.** Expected values come from OOF or FM-PM. The *realized* private change has an SD of 0.0012–0.0027 for an ongoing change and 0.003–0.005 for an onset change. Any T2 private-row change is therefore excluded unless it saves a lot of time and its Jaccard bound (§4.6) is ≤ 0.0005.

### 3.2 Options

| # | Option | Saves (Kaggle h) | Est. private Δ | Evidence | Type | Decision |
|---|---|---:|---|---|---|---|
| S1 | Old FTA bag 8 → 4: keep v12s4–s7, drop v11, v12, s2, s3 | 21.3 | **−0.0001** (−0.0002 to 0) | 1→2 +0.0008 (COMBO12), 2→4 +0.00045 (COMBO15), 4→8 +0.00065 (c16) | smooth | **lean-1** |
| S2 | Old FTA bag 8 → 2 (s6, s7) | 32.5 | −0.0004 (−0.0006 to −0.0002) | same | smooth | lean-2 |
| S3 | No old bag (W_FD 1.0, FD-only) | 45.0 | sign unknown (−0.0006 to +0.0003) | the mix beat old8 by +0.00063 (e1); FD members +0.0014 holdout each; no LB probe | smooth | only after a public probe |
| S4 | Drop t1dark from the FD bag (t1ramp = t1dark + ramp columns, same seed and rows) | 7.6 | −0.00005 | h5 (+dark +ramp) +0.00016 | smooth | **lean-1** (also removes the DARK code family) |
| S5 | Drop t1ramp2 (second ramp seed) | 7.8 | −0.00005 to −0.0001 | i2 +0.00011, part of it from the blackout role that bspec now holds | smooth | **lean-1** |
| S6 | Also drop t1ramp (FD bag = 3) | 8.0 | −0.0001 | h5 | smooth | no |
| S7 | Drop the 3rd FD member s11 | 6.4 | −0.00005 to −0.0001 | 4th FD member +0.00002 (F1) | smooth | lean-2 |
| S8 | Drop t1bspec | 8.8 | −0.0002; −0.0009 without any blackout member | i4 +0.00020, h1 +0.00037, t1dark +0.00029 | smooth | no |
| S9 | Drop TV / recon | 0.1 | −0.00026 / −0.0017 | c20 / c17 | smooth | no (minutes) |
| S10 | March ongoing fs4occ → fs2occ (seed-0 pair) | 2.7 | **0 private**; public −0.0004 ± 0.0015 | e4 fs2→fs4 +0.0004 (noise level) | public only | optional; keep for a clean public check |
| S11 | Drop pvo (A2 veto reference) | 2.4 | 0 if the full run vetoes 0 windows; otherwise ±0.002 per window | smoke 0/10 vetoed | lumpy | conditional on the tfb-t2ogpvogf log |
| S12 | B: ogfeat bag 4 → 2 (seed 0) inside veto10 | 6.4 | −0.0001 expected; ±0.001–0.002 realized | OOF bag of 4 vs pair +0.0003 IoU | lumpy | only if the private Jaccard bound ≤ 0.0005 |
| S13 | pv bag 3 → 1 (w 20) | 0.03 | lumpy | lab: bag ≈ single model | lumpy | no (no saving) |
| S14 | v11 without its unused FT models | 3.8 | **0** (FT and FTA are separate `lgb.train` calls with no shared RNG) | code | none | yes in any recipe that keeps v11 |

### 3.3 Recipes

| Recipe | T1 members | Private T2 / T4 | Est. private Δ vs selected | Kaggle h | 8-core / 32 GB wall h |
|---|---|---|---|---:|---:|
| full-A2 | 8 old + 6 FD + bspec | exact | 0 ± 0.0003 (nondeterminism) | 107–111 | 63–68 |
| **lean-1 A2** | old v12s4–s7; FD v13, s10, s11, t1ramp; bspec | exact | **−0.0002 to −0.0003** (± 0.0004) | 70–74 | **42–47** |
| lean-2 A2 | old s6, s7; FD v13, s10, t1ramp; bspec | exact | −0.0006 (−0.0010 to −0.0003) | 53–57 | 32–36 |
| lean-1 B | lean-1 T1 | exact (ogfeat × 4, veto10) | as lean-1 | ≈ 77 | ≈ 49 |
| lean-1 A2 + B | shared | both | – | 81–85 | ≈ 55 |

**Lean-1 A2 runtime on 8 cores / 32 GB (stages run one after another).**

Memory allows only one T1 member at a time (peak 26.6–28 GB), and pvogf or ogfeat (est. 20–26 GB) also run alone.

| Stage | Kaggle h | Threads | Wall h (assumptions) |
|---|---:|---:|---:|
| pseudo windows | 0.05 | 1 | 0.05 |
| T1 lean-1 members (9) | 59.7 | 8 | 33–35 (×1.75: LightGBM training is about 80 % of member time) |
| T1 mix + recon / TV | 0.26 | 1 | 0.2 |
| T2 onset (base, sp4, pv) | 0.15 | 4 | 0.12 |
| T2 March ongoing fs4occ | 5.45 | 4 | 4.5 (×1.2, faster cores at the same thread count) |
| T2 April ongoing pvogf (+ pvo) | 4–5.5 (+ 2.4) | 4 | 3.5–4.6 (+ 2.0) |
| T4 + assembly | 0.27 | 1 | 0.25 |
| **Total** | **70–74** | | **≈ 42–47 (≈ 2 days)** |

- With ≥ 64 GB, two jobs can run concurrently, which cuts wall time by about 35–40 %.
- The ×1.75 factor for 8 threads is an assumption. Measure it on the first parity member (§5).

**Decision procedure (no new training needed).**
- The lean T1 files are just different averages of existing member outputs. So, before any porting:
  1. Push one swap kernel that builds the lean-1, lean-2 and FD-only raw mixes from the existing member outputs, then runs `swap_fdrecon` on each.
  2. Print RMS(Δspeed) and RMS(Δflow) vs `tfb-i4` per split × regime, separately for blackout and ordinary rows.
  3. Optionally use one public slot on lean-1. Its queue rows equal A2's, so the public delta *is* the T1 effect, which transfers to private.
- Freeze lean-1 if the public Δ ≥ −0.0005. Otherwise take lean-1 + t1ramp2 or full.

**Measured (Kaggle parity kernel `tfb-par-lean1`, 2026-10-09).** Lean-1 mix built from the existing member outputs, then recon + TV, compared with the full-recipe files. All 6,735,795 state rows; split rows: validation 3,451,376 / private 3,284,419; blackout rows 64,720 / 63,745.

| Comparison | Rows | Speed RMS (km/h) | Speed max \|Δ\| | Flow RMS (vph) | Flow max \|Δ\| |
|---|---|---:|---:|---:|---:|
| raw mix vs `tfb-i4d` | all | 0.092 | 11.6 | 7.13 | 260 |
| | blackout, validation / private | 0.150 / 0.146 | 4.1 / 3.8 | 6.07 / 5.28 | 125 / 82 |
| | ordinary, validation / private | 0.089 / 0.092 | 11.6 / 9.1 | 7.11 / 7.21 | 260 / 249 |
| **final (recon + TV) vs `tfb-i4`** | **all** | **0.098** | 34.2 | **6.81** | 808 |
| | blackout, validation / private | 0.223 / 0.219 | 22.4 / 20.9 | 8.49 / 7.98 | 677 / 722 |
| | ordinary, validation / private | 0.095 / 0.092 | 34.2 / 23.3 | 6.74 / 6.82 | 771 / 808 |

- **Bound check.** The loose bound |ΔS_speed| ≤ RMS(Δv)/25 and |ΔS_flow| ≤ RMS(Δq)/600 (§4.7) gives, with weights 0.35·(0.54, 0.46):
  - speed: 0.35·0.54·0.098/25 ≈ 0.0007;
  - flow: 0.35·0.46·6.8/600 ≈ 0.0018.
  - This is a loose worst-case bound (sum ≈ 0.0026), not an estimate. The RMS above measures how far lean-1 moves from the full recipe, not how much worse it is.
  - The expected Δ therefore remains the §3.2 estimate (−0.0002 to −0.0003) until the public probe measures it.
- **Blackout rows.** They move about 2.3× more in speed RMS after recon + TV (0.22 vs 0.095). The blackout member (t1bspec) is the same in both recipes, so this comes from the 0.25 raw-mix share and from recon/TV gating on those rows.
- **Status.** The public probe of lean-1 (one slot) is still the deciding test.

---

## 4. Entry-script architecture

### 4.1 Layout

Approach: **vendored kernel code behind one orchestrator**. The kernels are not rewritten:
- Within a code family the members differ only in constants (§1.3). So each family is one vendored template. Constants come from the config, and only I/O lines are patched.
- A test diffs every vendored file against `lineage_src/` and allows only the patched lines to differ.

```
tfb_entry.py                 # the single entry point (CLI); can also be shipped as one file: python -m zipapp
tfb/
  io.py                      # data-root resolution, sorted file lists (same sort keys as the kernels), cache paths, atomic writes
  windows.py                 # pseudo windows  (t2lab.py build_windows, cap=None)
  t1/fta.py fd.py ramp.py bspec.py   # vendored member families (+ dark.py only for the full recipe)
  t1/combine.py              # t1_t1dark_swap.py mix + blackout override; swap_fdrecon.py recon() / tv() verbatim
  t2/onset.py                # t2_onset_pv.py (base, sp4, pv in one run; B uses base)
  t2/ongoing_val.py          # t2_occ_v26.py (fs4occ + qlast npz)
  t2/ongoing_pv.py           # t2_ongoing_pvogf.py (pvogf, optional pvo-veto) and t2_ongoing_pvaf.py (pvo)
  t2/ongoing_b.py            # t2_ogfeat_v27a/b + comb + t2_ogveto (B only)
  t4.py                      # NNLS cell of tfb-final
  assemble.py                # pick-specific row selection, sample order, hard gates (+ t4_validate.py checks)
  compare.py                 # row-level comparison and score bounds vs a reference submission
recipes/{A2_lean1,A2_full,B_lean1,B_full}.yaml
README.md  ENVIRONMENT.md  requirements.txt  Dockerfile
```

CLI:

```
python tfb_entry.py --data-root /data/kaggle_public --work-dir ./work --recipe recipes/A2_lean1.yaml \
       [--stages all|pseudo,t1,t1post,onset,ongoing,t4,assemble,compare] [--only-member v13 --only-panels D7_I10_E,...] \
       [--threads 8] [--reference /path/submission_J4.csv]
```

### 4.2 Stages and cached intermediates

Each stage writes `work/<stage>/…` atomically (temporary file, then rename) plus a `.done.json`. That file records the hashes of the recipe, code, inputs and environment. A rerun skips stages that are done, so a crash costs at most one member-panel.

| Stage | Output (cache) | Needs |
|---|---|---|
| 0 env | `work/manifest.json` (package versions, CPU flags, thread settings, data file count / size, SHA-256 of submission_key, sample_submission, window_index files) | – |
| 1 pseudo | `pseudo/pseudo_<panel>.npz`, checked against the hashes in `lineage_A2.json` | data |
| 2 t1 member × panel | `t1/<member>/<panel>.parquet` (validation + private template rows: speed, flow) and `t1_valholdout` row | 1 |
| 3 t1post | `t1/raw_mix.parquet`, `t1/state_final.parquet` (recon + TV) | 2 |
| 4 onset | `t2/onset_{base,sp4,pv}.parquet` + probabilities | data |
| 5 ongoing | `t2/fs4occ.parquet` + `qlast.npz`; `t2/pvogf[v].parquet` (+ `pvo`); B: `fs4ogf`, `veto10` | data (+ 5a for B / pvo) |
| 6 t4 | `t4/path_flow.parquet` | data |
| 7 assemble | `submission.csv` + `.sha256` + gate report | 3–6 |
| 8 compare (ours only) | `compare/report.json` | 7 + reference file |

**Assembly replaces the swap chain** with a direct, pick-specific selection by window condition from window_index:

| Rows | Source |
|---|---|
| validation onset | sp4 |
| private onset | pv (A2) / base (B) |
| validation ongoing | fs4occ |
| private ongoing | pvogfv (A2) / veto10 (B) |
| state | T1 final |
| odme | NNLS |

Kept gates:
- ids equal sample_submission in order;
- finite values, ≥ 0;
- queue ∈ {0, 1};
- flow < 50 vph share < 20 % per panel × regime;
- the t4_validate path checks.

### 4.3 Recipe config (excerpt of `A2_lean1.yaml`)

```yaml
pick: A2
threads: {t1: 8, t2_private: 4, t2_validation: 4, catboost: 4}   # t2 private producers stay at the Kaggle value
t1:
  old_fta: [v12s4, v12s5, v12s6, v12s7]
  fd:      [v13, v13s10, v13s11, t1ramp]
  blackout_member: t1bspec
  w_fd: 0.5
  w_dark: 0.75
  blackout_minutes: 90
  members:            # constants of lineage_src, checked by a test against the ast of each original kernel
    v12s4:  {family: fta,  row_rng: 4,  tr_rounds: 3, tr_rng_base: 5000,  tx_cap: 6, holdout: 0.15, lgb_seeds: [4, 14, 24]}
    v13:    {family: fd,   row_rng: 9,  tr_rounds: 3, tr_rng_base: 10000, tx_cap: 6, holdout: 0.15, lgb_seeds: [9, 19, 29]}
    t1ramp: {family: ramp, row_rng: 13, tr_rounds: 3, tr_rng_base: 14000, tx_cap: 6, holdout: 0.15, lgb_seeds: [13, 23, 33], dark_seed: 1300}
    t1bspec: {family: bspec, row_rng: 13, lgb_seeds: [13, 23, 33], dark_n: 48, dark_seed: 1300,
              panel_overrides: {D7_I405_N: {max_rows: 420000, train_day_mod5_keep: [0, 1, 3]}}}
  recon: {gate: 0.6, a: 0.75, logr_max: 0.5, w_range: [8, 40], kjl_range: [60, 200], excl: [0.02, 0.05, 200]}
  tv: {free: 0.0075, free_anchor: 0.001, gate: 0.02, gate_anchor: 0.005, iters: 300, run_gap_min: 5}
t2:
  onset: {validation: sp4, private: pv, k_blend: 4, slot_halfwidth: 2, march_weights: [10, 20, 40],
          catboost: {iterations: 600, depth: 8, learning_rate: 0.08, random_seed: 0}}
  ongoing: {validation: fs4occ, private: pvogfv, veto_g: 10, need_pvo: auto,   # auto = from the recorded veto count
            f03: {w_rs: 1.0, w_dense: 0.3, fill_gap: 36, pers_max: 0.9, max_og: 3000},
            big: {leaves: 511, min_data: 40, lr: 0.03, trees: 1800}, pv: {leaves: 511, min_data: 40, lr: 0.05, trees: 900, wm: 5, k_blend: 4}}
t4: {lambda: 0.05}
```

### 4.4 Paths

- **Data root.** One argument, `--data-root`: the folder holding `submission_key.csv`, `sample_submission.csv` and `corridors/`, `task1/`, `task2/`, `task4/`.
  - It replaces every `glob('/kaggle/input/**/kaggle_public')`.
  - Release check: 9,688 files, 12.39 GB, all created 2026-09-11.
  - Disk needed: about 13 GB of data + about 10 GB of cache.
- **Cross-kernel inputs.** Every `/kaggle/input/**/<kernel>/**/<file>` read becomes a work-dir cache path. `/kaggle/working` becomes the stage folder.
- **Sort keys.** Keep each kernel's own sort key: the full path for `read_split`, the basename date for `masked_month` / `masked_cube`.

### 4.5 Environment

Two Kaggle images were used (`lineage_A2.json → images`):

| Image | Digest | Python | Kernels |
|---|---|---|---|
| A | `gcr.io/kaggle-images/python@sha256:dafd4ce5…3142755c40b9` | 3.12 | the 8 old FTA members, v13, t2v13, t2lab-full (pseudo windows), tfb-final (T4) |
| B | `gcr.io/kaggle-images/python@sha256:81b1e7c4…4b6b601cd84a` | 3.13 | everything else: v13s10/s11, dark/ramp/ramp2/bspec, all T2 producers, all swaps |

- Package versions were **not printed in any log**.
- For reference, the local lab venv is Python 3.9 with lightgbm 4.6.0, catboost 1.2.10, numpy 2.0.2, pandas 2.3.3, pyarrow 21.0.0. That is *not* Kaggle's environment.

How to pin (choose image B: it ran every T2 producer, every swap and 6 of the 15 T1 members, 4 of the 9 lean-1 members):
1. `docker pull gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a`, then `pip freeze` inside it. This needs no Kaggle push.
   - Write `requirements.txt` with lightgbm, catboost, numpy, pandas, scipy, pyarrow, scikit-learn at the exact versions.
   - The `Dockerfile` is `FROM` that digest. It is the reference environment for the organizers.
   - A slim venv with the same pins is the light alternative.
2. Stage 0 of every parity kernel prints `importlib.metadata.version(...)` for these packages. This confirms the pins on Kaggle itself.
3. Image-A components are re-trained in image B.
   - CatBoost already reproduced exactly from A to B (CONTROL v13 onset agreement 1.000000).
   - The LightGBM drift of the old members is measured in parity (§4.6, level 1). Its effect is smooth and small: at most a "new seed" per member, ≈ ±0.00005.

### 4.6 Determinism checklist

1. **Seeds.**
   - Every `np.random.default_rng(<const>)` call site keeps its constant and its call order (per panel; the per-split offsets `+ (split == 'private')`).
   - LightGBM seed triples as in §1.3; "default" means no seed key, as in v11, the seed-0 pairs and pv*. Never add a seed key.
   - CatBoost `random_seed 0`. No global `np.random` state.
   - `PYTHONHASHSEED=0`. Audit for `list(set(...))` iteration in order-sensitive paths.
2. **Threads.**
   - `num_threads` / `thread_count` = 4 for every T2 producer, as on Kaggle.
   - T1 members may use 8: their effect is smooth. Keep 4 if time allows exactness.
   - `OMP_NUM_THREADS` set explicitly; `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, since scipy NNLS and numpy linalg must not depend on BLAS threading.
3. **LightGBM.**
   - Same version as image B; params byte-identical to the kernels; no `deterministic` or `force_col_wise` (that would change results vs the originals).
   - Log the automatic col-wise / row-wise choice: `verbose=0` on the first model.
   - Known Kaggle-to-Kaggle rebuild agreement: 0.99994 of rows (t2v21 CONTROL cur2 vs t2v20: ≈ 5 of 87,000 ongoing rows; positives 10,165 vs 10,164).
4. **Data order.**
   - Same panel order, window order and column order: LightGBM `feature_fraction_seed` picks columns by index.
   - float32 cubes as in the kernels; sorted globs with the original keys.
5. **Decode ties.** `np.argsort(-p)` (quicksort; numpy may dispatch to SIMD sorts) is order-ambiguous only when p ties at the top-m cut. `compare.py` counts the windows with `p[order[j]] == p[order[j+1]]`. Zero means the decode is order-free.
6. **Hardware.** Linux x86-64 for the official rerun. No Apple Silicon: FMA and SIMD differences. Record CPU flags (AVX2 / AVX-512) in the manifest.
7. **bspec part c.** Keep the reduced settings for D7_I405_N whatever the RAM.
8. **CSV.** Write with pandas defaults (full precision), as the kernels did.

### 4.7 Validation plan

Levels 1–3 can run on Kaggle with the original kernels attached as sources, so nothing large is downloaded. The same image means porting bugs are isolated from environment drift.

| Level | Test | Pass criterion |
|---|---|---|
| 0 static | config ↔ kernel constants (ast of `lineage_src/*.py`); vendored file diff limited to I/O lines; pseudo-window hashes | exact |
| 1 member parity (Kaggle) | each rerun member × panel vs its original `state_submission_FTA.csv`: RMS / max \|Δspeed\|, \|Δflow\| | image-B members: 0, or ≤ 0.01 km/h and 0.5 vph RMS (calibrate with one repeat run). Image-A members: report the drift |
| 2 stage parity (Kaggle) | T1 final vs `tfb-i4` state rows (full recipe); onset / ongoing vs osp / opv / t2occ / pvogf / veto10 rows; path_flow vs tfb-final | queue rows identical (agreement ≥ 0.9999, no window with Jaccard < 1 on **private onset**); path_flow max \|Δ\| < 1e-6 relative |
| 3 file vs file | rerun `submission.csv` vs the Kaggle file (`tfb-j4` submission_J4.csv, or i4 / j3) | see the bounds below |
| 4 public LB (optional, 1 slot) | submit the rerun (or the lean mix built from existing outputs) | public Δ ≥ −0.0005 vs 0.88176 |
| 5 private (after Nov 7) | if Kaggle opens late submissions, submit the rerun file and read its private score directly | ≤ 0.002 from the selected file |

**Level-3 metrics.** Computed separately for validation and private, and for onset and ongoing, so that private is judged on its own.

- **Queue.** Per-window Jaccard J_w between the rerun and the reference; number of windows not identical; positives.
  - Jaccard distance is a metric, so per window |IoU(rerun, truth) − IoU(ref, truth)| ≤ 1 − J_w.
  - Hence |ΔS_total| from T2 ≤ 0.15 · [mean_w(1 − J_w)_onset + mean_w(1 − J_w)_ongoing], using the official equal weight per window: 40 per condition and split.
  - If some windows differ, also price them with the FM-PM gate for an expected value.
- **State.** RMS, P99 and max of Δspeed and Δflow per split × regime × panel, with blackout rows separately.
  - Hard bound per regime: |ΔRMSE| ≤ RMS(Δ). So |ΔS_speed| ≤ RMS(Δv)/25 and |ΔS_flow| ≤ RMS(Δq)/600, aggregated regime → panel → family and weighted 0.35·(0.54, 0.46). This is loose and is for reporting only.
  - The expected T1 Δ comes from level 4 or the §3 estimates.
  - Also report the exact S_FD of both files (released cells + predictions, `repo/src/task3/score_task3.py` functions) and the temporal-LWR proxy Δ.
- **odme.** max |Δpath_flow|.

**Acceptance for the official recipe (private):**
- T2 bound ≤ 0.0005;
- expected T1 Δ ≥ −0.0005;
- T4 exact;
- total expected |Δ| ≤ 0.001;
- worst-case total ≤ 0.002.

### 4.8 LightGBM reruns are not bit-reproducible (Kaggle parity round 2, 2026-10-10)

**Measured.** Everything else reproduced exactly:
- **EXACT:** onset (CatBoost), T4, the swaps, the pseudo windows (16/16 arrays).
- **NUMERIC:** chaint1 state (≤ 9e-13).

The LightGBM ongoing reruns drift from the originals:

| Rerun | Changed cells | Detail | Probabilities |
|---|---|---|---|
| tfb-t2occ fs4occ | 22 of 87,000 ongoing | validation 11 in 8 windows; private 11 in 9 windows, min J 0.944 | 64/96 arrays differ, max 0.136 |
| tfb-t2ogpvogf pvogf / pvogfv | 24 private, in 11 windows | min J 0.867; positives 4767 = 4767 | max 0.20 |

**Why.** The originals ran LightGBM with `num_threads=4`, `verbose=-1`, and neither `deterministic` nor `force_row_wise/col_wise`.
- LightGBM then picks row- or col-wise histograms by a silent timing test.
- It also sums the multithreaded histograms in no fixed order.
- Kaggle sessions land on different CPUs (round 1 logged both AMD EPYC 7B12 and Intel Xeon 2.20 GHz).
- A 511-leaf, 1800-tree model amplifies one different split into probability changes of 0.1–0.2 on a few cells. Decode then flips about 1 cell in 4,000.

The original path was never logged, so **no setting can reproduce the original files exactly**. On one machine both settings repeat: local SMOKE test, 4 runs in parallel, switch on and off, all identical.

**Private-score bound.** The Jaccard distance is a metric, so per window |ΔIoU| ≤ 1 − J. With 0.00375 per window-IoU unit:
- **A2 (pvogf private rows, 24 cells / 11 windows, about 119 queued cells per window):**
  - Σ(1 − J) ≈ 0.13 (the min-J window) + 22/119 ≈ 0.32, so |ΔS| ≲ **0.0012**.
  - Pessimistic (all 11 windows at J = 0.867): 0.0055.
  - The expected value is about 0, because flips sit on the decode boundary and go both ways. SD ≈ 0.00375 · ½ · √Σ(1 − J)² ≈ **0.0003**.
- **B2:**
  - fs4occ private (11 cells / 9 windows) enters veto10 only through qlast and the vetoed window.
  - fs4ogf (ogfa/ogfb) is not measured yet; assume pvogf-like, about 0.001–0.0013.
- Validation-row drift moves the public score only.

**Against the rules.** Even the pessimistic case (0.0055) is inside the 1 % rule (about 0.0088). The realistic bound (0.0012) uses about 60 % of our ±0.002 budget; add lean-1 (expected −0.0003). Onset, T4 and the swaps add nothing; T1 adds NUMERIC noise (members pending).

**Setting adopted (`tfb/config.py LGB_REPRO`).**
- The Task 2 LightGBM stages (`t2_occ`, `t2_pvogf`, `t2_ogfeat`) add `deterministic=True, force_row_wise=True` to their params, with `num_threads=4` kept.
- With that, every rerun on the same hardware, LightGBM version and thread count writes the same file. LightGBM does not promise identical results across CPUs or builds.
- **Trade-off:** the entry script then writes, every time, a file that differs from the submitted one by about the drift above (expected 0, realistic bound ≈ 0.0012 private).
- `--set <stage>.lgb_repro=null` restores the original behaviour.
- Task 1 members keep the original setting. Their drift is smooth, and deterministic mode would add runtime to about 60 h of training.
- Parity kernels `tfb-par-pvogfrep` / `tfb-par-occrep` measure run-to-run repeatability across Kaggle sessions.

**Option for the team: submit the consolidated output before Nov 6.**
- Build A2 (and B2) with the entry script on Kaggle (repro setting, chosen recipe), submit both, and select them. The selected submission is then the reproducible file: the organizers' rerun matches it exactly if their platform matches Kaggle, and differs by T1 noise only otherwise.
- **Cost:**
  - compute: lean-1 about 85 Kaggle-h, full about 130 h, i.e. 1–1.5 days at 5 sessions, competing with the parity runs;
  - 2 submission slots;
  - the public score moves by about ±0.001 (noise, expected 0);
  - the new files must be validated before Nov 3 (code freeze).
- **Worth it mostly if lean-1 is adopted.** Lean-1 alone moves the rerun by an expected −0.0003, and a lean-1 submission removes that gap.
- Without resubmitting, the 1 % rule still holds with a wide margin. This is the team's call.

---

## 5. Timeline (Oct 9 → Nov 10)

| Dates | Work | Exit criterion |
|---|---|---|
| Oct 9 | Chain oct9 Q4 builds A2 = tfb-j4 once tfb-t2ogpvogf is clean. Record its runtime, the private-window **veto count** (decides need_pvo) and peak RAM. Rerun tfb-t4val on j4 | j4 submitted, public 0.88176 |
| Oct 10–12 | Decision kernel from existing outputs: lean-1 / lean-2 / FD-only mixes + recon/TV, RMS vs i4. Optional public probe of lean-1. B: bound for fs2ogf-in-veto10. Pull image B and pin versions | lean recipe frozen |
| Oct 12–19 | Vendor and patch the code families (T1 fta / fd / ramp / bspec [/ dark], combine, T2 onset / ongoing, T4, assemble, compare); level-0 tests; local smoke on a data subset (SMOKE switches as in the kernels) | level 0 green; end-to-end smoke |
| Oct 20–26 | Kaggle parity runs (levels 1–2): about 19 lean-1 member parts (4–5 sessions in parallel, about 2 days), T2 stages, T4, assemble → Kaggle-built rerun file; level 3 vs j4 and j3. **Oct 25: registration e-mail** (team, members, affiliations, contact) | level 1–3 pass for A2 and B |
| Oct 27–Nov 2 | Full run on 8 cores / 32 GB Linux x86-64 in the pinned Docker image (data via one bulk download, or a cloud VM). Measure the real speed factor; level 3 vs the Kaggle rerun and the reference files. Draft the report | the off-Kaggle rerun passes the acceptance criteria |
| Nov 3–5 | Code freeze; README (runtime, hardware, stage resume), ENVIRONMENT, Docker. No final-pick changes after Nov 3 unless they are re-validated | release candidate |
| Nov 6–7 | Kaggle final deadline (organizers say Nov 6; the competition page says 2026-11-07 06:55 UTC; use the earlier one). Select A2 + B | final picks locked |
| Nov 7–9 | Private LB → identify the selected submission (A2 or B). Fill the report numbers; optional late-submission check (level 5). Final cached rerun of that recipe | report final |
| Nov 10 | Submit code + PDF | – |

---

## 6. Report outline (15 % of the award score)

1. **Problem and metric.**
   - 4 tasks with weights state 0.35, queue 0.30, physics 0.15 (LWR 0.10 + FD 0.05), ODME 0.20.
   - Per-window IoU aggregation; the ≤ T rule for Task 2; Task 1 is offline.
2. **Validation methodology.**
   - Frozen day-block folds; replica of the official window selector (onset 39/40; the selection-faithful replica 36/40 vs 14/40); validation-month internal holdout.
   - Pseudo-March; the **FM-PM gate** (footprint-matched pseudo-March, MAE 0.0005 vs 0.0013–0.0015 for OOF / plain pseudo-March).
   - LB noise floor (ongoing ±0.0012–0.0027, onset ±0.003–0.005); the lab→LB transfer table; pre-registered keep rules; why onset probes cannot be read.
3. **Methods.**
   1. **Same-scenario prior.** Validation and private share one scenario seed. Per-cell blend a = n/(n+k) with the released days before T. sp4 onset: LB +0.00104. April models with released-March windows (pv, pvo / pvogf). The final-pick logic is A2 (same-scenario bet) vs B (hedge), given the two-sided ±0.14 onset lab risk.
   2. **Selection-faithful ongoing training (F03).** Persistence cap on the time-filled state; LB +0.0047. All-finite inputs (+0.0009), occupancy, ogfeat with a growth veto for April. The ogfeat March failure was traced to scenario-specific growth calls.
   3. **T2 onset.** T+30-only CatBoost (organizer fact: onset queues appear only at T+30), expected-IoU decode.
   4. **T1 transductive members.** Residual LightGBM on temporal interpolation; transductive rows from released target-month cells; ASM wave features; low-rank; FD-implied features; seed / round bags. **Blackout specialist:** selection-faithful simulated blackouts, 30 timing columns, ramp features (ramps are released inside mainline blackouts); W_DARK 0.75.
   5. **FD-gated reconciliation + 1-D TV** (T3): +0.00167 / +0.00026.
   6. **T4.** Prior-regularized NNLS (λ 0.05), saturated.
4. **Results.**
   - LB progression 0.80939 → 0.88176 with per-component deltas (§3 evidence).
   - Negative results: 8 onset failures, ogfeat on March, Kalman, calibration, LWR projection, the persistence blend.
5. **Reproducibility.**
   - Entry script, lean recipe and its measured Δ, runtime and hardware, determinism.
   - Rerun-vs-submission table (level 3), including the Jaccard bound.
6. **Limitations.** 40 windows per condition; scenario shift; compute.

---

## 7. Open items and risks

- **tfb-t2ogpvogf (A2 producer) is running on Oct 9.** When it finishes, record runtime, peak RAM and the private veto count:
  - 0 vetoes → pvogfv = pvogf, and pvo is dropped from the A2 recipe;
  - ≥ 1 veto → pvo (2.4 h) stays.
  - Then rerun the lineage trace to add tfb-j4 to `lineage_A2.json` (method: BFS over `kernelDataSources` via `kjob.call('GET', '/kernels/pull?…')`, logs via `/kernels/output`, log text only).
- **Porting bugs** are the largest real risk to the 1 % rule. Level 1 member parity on Kaggle is the guard.
- **LightGBM drift** (§4.8): the T2 ongoing reruns differ from the originals by about 1 cell in 4,000 (A2 private bound ≈ 0.0012). The entry script uses `LGB_REPRO` for repeatable reruns. Submitting the consolidated output is optional insurance.
- **Environment.** No single environment reproduces both images. Image-A members will drift smoothly, while T2 private producers ran on B. Pin B's versions from the image digest.
- **T2 lumpiness.** One onset flip ≈ 0.001–0.002 private. Requirements: exact CatBoost (proven A→B), LightGBM with the same version and 4 threads, the decode-tie audit. The FM-PM gate and the Jaccard bound quantify any residue.
- **RAM.** 32 GB is the minimum (T1 member peak 28.0 GB); 48–64 GB is comfortable. bspec part c must keep its reduced settings.
- **Which pick is selected** is known only on Nov 7, so A2 and B must both pass levels 1–3 before then. B adds the ogfeat bag (11.5 Kaggle-h).
- **Data access.** Local data is incomplete (186 MB of the 12.4 GB release; per-file downloads hit HTTP 429). Use Kaggle-hosted parity, plus one bulk download or a cloud VM for the off-Kaggle run.

---

## 8. Final picks A3 = tfb-k9 / B3 = tfb-k10 (update 2026-11-03)

### 8.1 What changed against A2 / B2

| Block | A2 / B2 | A3 / B3 |
|---|---|---|
| Task 1 members | 15 LightGBM (8 old FTA, 6 FD, t1bspec), 31 parts | + `tfb-t1catb`, `tfb-t1catb2` (CatBoost, 2 seed sets), `tfb-t1xgb` (XGBoost), 4 parts each: 43 parts |
| Task 1 combine | `tfb-i4d`: 0.5 FD + 0.5 old, blackout 0.25 / 0.75 t1bspec | `tfb-k5d`: same, with 0.65 / 0.35 mean(catb, catb2) and then 0.65 / 0.35 xgb before the blackout override |
| Task 1 post-processing | `tfb-i4`: R (FD recon) + T (1-D TV) | `tfb-k9`: M (flow LRTC (24, 40, 10), W 0.35, ordinary rows) + B (speed (8, 20, 20) / flow (8, 40, 10) LRTC, W 0.15, blackout rows) + the same R + T |
| Queue rows | A2: osp sp4 / occ fs4occ / opv pv / pvogfv; B2: … / osp base / veto10 | identical (A3 inherits tfb-j4's rows through k3d → k3 → k5d → k9; B3 = tfb-k10 = the B2 private parts on tfb-k9) |
| odme rows | tfb-final | identical |
| Public LB | 0.88176 / 0.88176 | 0.88281 / 0.88281 |

The LRTC steps use only the released cells of the target month (Task 1 is offline, no ≤ T rule); they never touch queue rows
(asserted in the kernel). So the Task 2 rule and the Task 2 post-T rerun test are exactly those of A2 / B2.

### 8.2 Entry-script changes

- `tfb/config.py`: `PICKS = ('A3', 'B3', 'A2', 'B2')`, `PICK_GEN` (j / k generation), `K_EXTRA`, `QUAD_PANELS`, 3 new `T1_MEMBERS`,
  `T1_MIX_K5D`, `LRTC_K9`, `QUEUE_PARTS['A3'/'B3']` (= A2 / B2), `ASSEMBLE_BASE`, `ASSEMBLE_MODULE`, `PICK_KERNEL` /
  `PICK_REFERENCE` for A3 / B3, plan per generation (`t1_mix_k5d`, `t1_lrtc` instead of `t1_mix`, `t1_recon`).
- New vendored stages (only `# [TFB]` lines differ from the originals): `t1_mix_k5d.py` (tfb-k5d), `t1_lrtc.py` (tfb-k9),
  `assemble_k10.py` (tfb-k10; its "no March row changed" gate is switched by `march_from_parts`, because the T4 skeleton
  base has no March queue rows and the validation parts fill them), templates `t1_catb.py`, `t1_xgb.py`.
- A3's assembly writes into the virtual folder `tfb-k9-asm` (`submission_K9.csv`); its reference is `tfb-k9/submission_FDRTV.csv`.
- Default pick of `run_all.py`: A3.
- `bundle.py`: 5 light parity kernels (`tfb-par-k5d`, `-k9`, `-asma3`, `-asmb3`, `-chaint1k`) + 12 member parity kernels.

### 8.3 Runtime and RAM

| | Kaggle CPU-h | Wall (8 cores / 32 GB, est.) | Peak RAM |
|---|---:|---:|---:|
| 12 new member parts | 35.8 (catb 15.7, catb2 15.8, xgb 4.2) | ≈ 30–36 h (4 threads fixed; two in parallel need ≥ 64 GB) | 26.1 GB (parts a) |
| `t1_mix_k5d` + `t1_lrtc` | ≈ 0.6 | ≈ 0.5 h | ≈ 10–12 GB (est.) |
| **full A3** | 143–147 | ≈ 93–104 h | 28.0 GB (unchanged maximum) |
| lean-1 A3 (old v12s4–s7, FD v13/s10/s11/ramp, bspec + the 3 new members) | 106–110 | ≈ 72–83 h | 28.0 GB |

The lean-1 delta (−0.0002 … −0.0003 private for A2) has not been measured for A3; with 3 strong new members in the mix the
loss from thinning the LightGBM bags is probably smaller, but the selected file should be rebuilt with **full**.

### 8.4 Determinism and exactness

- Locally (`tests/test_k9_parity.py`, real D12_I405_S April release, synthetic key): the vendored k5d / k9 / k10 stages
  reproduce the original kernel sources byte for byte through the runner; a second run is byte-identical.
- CatBoost / XGBoost: fixed seeds and 4 threads as in the originals; expected EXACT on the same package versions and CPU
  family, otherwise smooth drift (a new member seed moves the private score by ≈ 0.00005).
- LRTC (`numpy.linalg.eigh`): last-bit differences across BLAS builds / CPUs are possible (NUMERIC; ≤ 1e-9 relative is
  expected, far below the 1 % tolerance).
- A3's final file is EXACT by value but not byte-identical to tfb-k9's: one more pandas CSV cycle in `assemble` re-formats
  about 6 % of the floats in the last printed digit (parsed values identical). This is cosmetic and is not changed (changing
  the order of stages would only move the extra cycle to K5D, before the LRTC).
- Images: the k-generation kernels ran on Kaggle's default image of 2026-10-10 … 11-03, not on image B; their package
  versions were not logged. The member parity kernels print them (`ENV` lines).

### 8.5 Open items for Nov 10 (A3 / B3)

1. Kaggle parity of the light A3 / B3 stages (5 kernels, ≈ 2.1 Kaggle-h, parallel): `tfb-par-k9` and `tfb-par-k5d` first.
2. Re-push `tfb-par-t4` / `tfb-par-chaint1` (path_flow fix; open since milestone 1).
3. Member parity of the 12 catb / catb2 / xgb parts (38 Kaggle-h, ≈ 10 h wall at 5 sessions) and the milestone-2 parity
   that is still open (pseudo, LightGBM members, Task 2 ongoing incl. the LGB_REPRO repeatability runs).
4. **Task 2 post-T deletion rerun** (organizer check): delete every post-T row of each Task 2 window from a copy of the
   release, rerun the Task 2 stages, require identical queue files. Same stages for all four picks; ≈ 0.5 day for the
   deleter + ≈ 22 Kaggle-h (or ≈ 12 h on 8 cores).
5. Off-Kaggle full A3 run for the rerun-vs-submission table and the measured wall time (≈ 4 days; only possible if started
   by Nov 5 — otherwise report the Kaggle parity results and the projected runtime).
6. ENVIRONMENT.md: pin `catboost` / `xgboost` from the parity `ENV` lines; README / report numbers for k9 / k10 runtimes.
7. PDF report (§6 outline + CatBoost / XGBoost members, LRTC steps, LB 0.88176 → 0.88281): 1–1.5 days.
