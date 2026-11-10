# Score gap analysis: 0.86217 → 0.90

**Formula:** S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME, where S_physics = S_FD/3 + 2·S_LWR/3.

**Effective weights:**

| Component | Weight |
|---|---:|
| S_state | 0.35 |
| S_queue | 0.30 |
| **S_LWR** | **0.10** |
| S_FD | 0.05 |
| S_ODME | 0.20 |

## 1. Component estimates at 0.86217

| Component | Estimate | How it was obtained | Confidence |
|---|---:|---|---|
| S_ODME | 0.994 | LB probes (ours, and Hisernberg's 0.9938) | High |
| Onset IoU | ≈ 0.758 | Probe: 0.743 at 0.84855, plus the LB deltas since (CatBoost +0.004, half of the v9 prior gain) | Medium |
| Ongoing IoU | ≈ 0.797 | Probe: 0.747, plus prior-free +0.030, v9 half +0.011, 127 leaves +0.007, 255 leaves +0.002 | Medium |
| S_queue | ≈ 0.778 | Mean of onset and ongoing. Check: the sum of T2 LB deltas, 0.0099 = 0.30 × (0.778 − 0.745). | Medium |
| S_state | ≈ 0.940–0.945 | Val-month holdout 0.9374 (harder: an extra 15% hidden) | Medium |
| S_FD | ≈ 0.99 | Truth rows give 0.991 locally. FD is evaluated on all rows, 50–80% of which are released truth. | Medium–High |
| S_LWR | ≈ 0.50–0.52 | Solved from the LB: 0.35·S_state + 0.15·S_phys = 0.4300. With S_state 0.940 → S_phys 0.674 → S_LWR 0.515. | Medium |

**Probe arithmetic.** Zeroing one condition's predictions sets that half of every panel to IoU 0, so ΔLB = 0.30 × 0.5 × IoU.
- Onset: 0.84855 − 0.73711 → IoU 0.743.
- Ongoing: 0.84855 − 0.73657 → IoU 0.747.

## 2. Realistic headroom per task

| Task | Current strength | Realistic upside (component) | Total-score upside | Evidence | Confidence |
|---|---:|---:|---:|---|---|
| Task 1 (S_state) | 0.94 | +0.005 … +0.010 | **+0.002 … +0.0035** | A 3-point smoothed-truth oracle (unreachable, uses the cell's own value) gives S_state ≈ 0.950. Gains shrink with each model (+0.0024, then +0.0012). | Medium |
| Task 3 (S_LWR) | 0.51 | +0.00 … +0.03 | **+0.000 … +0.003** | Local prototype (D12_I405_S, 40/20-day split). Temporal interpolation 0.523; residual-HGB model **0.520–0.524** (S_state 0.924 → 0.936, but LWR is unchanged); density blend or run-smoothing +0.001 … +0.009 S_LWR (score +0.0005); same-instant high-pass neighbour features 0.516 (worse). The 0.65 'smoothed-truth oracle' uses the cell's own truth and is **not reachable**. S_LWR is pinned by white observation noise at the target cells. Hisernberg's TV smoothing: +0.0006 LB. | Medium–High |
| Task 3 (S_FD) | 0.99 | ≈ 0 | 0 | Truth S_FD is 0.991. | High |
| Task 2 onset | 0.758 | +0.02 … +0.05 | +0.003 … +0.0075 | Our CV is 0.89 but the LB is 0.743–0.76. The participant claim of a "cap near 0.7" is already exceeded. The organizer says labels have no noise floor. | Low–Medium |
| Task 2 ongoing | 0.797 | +0.05 … +0.09 | **+0.0075 … +0.0135** | Hisernberg's public ongoing is 0.843, with the same data and rules. Our CV is 0.92. Possible fix: the direction bug (topology audit). | Medium |
| Task 4 | 0.994 | ≤ +0.001 | ≤ +0.0002 | Saturated; we already use the organizer's method. | High |

**Sum of realistic upside:** ≈ +0.010 … +0.024 → **0.872 … 0.886**. On this estimate 0.90 is not reachable through Task 1 or Task 3.

**Consequence (INFERENCE, arithmetic):**
- Task 1, Task 3 and Task 4 are close to saturation for everyone. Task 3 is pinned by noise, and Task 4 is the organizer's own method.
- A post-rebuild 0.90187 (KTK) therefore implies S_queue ≈ 0.778 + (0.0397 − ~0.004) / 0.30 ≈ **0.89**.
- **The frontier gap is almost entirely Task 2.** Frontier teams need about 0.11 more S_queue than we have: ongoing and onset both ≈ 0.89, not just better Task 1 reconstruction.

**Required improvement per component if one lever did it all** (+0.0378):

| Lever | Required change | Feasible? |
|---|---|---|
| S_queue | +0.126 (to 0.904) | Unrealistic |
| S_LWR | +0.378 (to 0.89) | Impossible: above the noise ceiling |
| S_state | +0.108 | Impossible |

**So no single task gets us to 0.90.** The path is additive.

## 3. Task 2 ceiling questions

- **Q1. How much can Task 2 still add?** Realistically **+0.010 … +0.021 total**, with ongoing the larger part. The frontier arithmetic (section 2) implies top teams have about +0.033 over us here.
  - At the Hisernberg level (ongoing 0.843, onset 0.758): +0.007.
  - At ongoing 0.88 and onset 0.80: +0.019.
- **Q2. Is Task 2 near diminishing returns?**
  - **Onset: partly.** It is already above the participant-claimed cap of 0.7. What remains is deciding which links queue, and the selection rule gives no further free structure.
  - **Ongoing: no.** A public pipeline reports 0.843 against our ≈ 0.80. We also found a direction inconsistency in the pooled Task 2 model that affects exactly the spatial-propagation features ongoing depends on.
- **Q3. Would +0.05 S_queue be enough?**
  - It gives +0.015 total, reaching about 0.877. **Not enough on its own.**
  - Even with Task 1 (+0.002 … +0.0035) and Task 3 (≤ +0.003), it stays below 0.90. **0.90 needs roughly +0.11 S_queue.**

## 4. Scenario budget

These are targets that tell us where experiments must come from, **not claimed gains**.

| Scenario | Task 1 | Task 3 LWR | Task 2 | Task 4 | Total | Requires |
|---|---:|---:|---:|---:|---:|---|
| A (realistic) | +0.003 | +0.001 | +0.012 | 0 | 0.878 | transductive T1, direction fix + ongoing identity |
| B (strong) | +0.0035 | +0.002 | +0.020 | 0 | 0.883 | A at the top end, plus ongoing ≈ 0.88 |
| C (0.90) | +0.004 | +0.003 | +0.031 | 0 | 0.900 | S_queue ≈ 0.88: ongoing ≈ 0.92 **and** onset ≈ 0.84 |

## 5. Allocation decision

The evidence says the opposite of shifting compute to Task 1 / Task 3:
- Task 3 is noise-pinned; the local prototype shows no reachable lever beyond about +0.003.
- Task 1 has at most about +0.0035 left.
- **Task 2 must get most of the effort, and it must be structural rather than tuning.** Candidates:
  - the direction bug (running);
  - link identity for ongoing (running);
  - features that decide *which* links queue at T+30 in onset: bottleneck propensity × live state, and ranking.
- Task 1 keeps two cheap, well-evidenced items: transductive rows (running) and seed bagging.
