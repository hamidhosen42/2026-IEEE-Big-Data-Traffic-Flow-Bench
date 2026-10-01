# Organizer-verified facts

I read every thread in full, not just the titles. The dump was refreshed on 2026-09-30 at 18:48 UTC and is stored in `discussions/*.md`.

**Tags:**
- **[ORG]** — ORGANIZER CONFIRMED: an organizer post or reply, or the organizer's public scorer code.
- **[PART]** — PARTICIPANT CLAIM.
- **[OURS]** — OUR OBSERVATION: measured by us.
- **[INF]** — INFERENCE.

## Task 2 rules (online task)

1. **[ORG] 742068 (09-19).** A forecast with origin T may use released data timestamped **≤ T** and nothing after it, in any file. This explicitly allows:
   - earlier times on day D;
   - **earlier days of the same split**;
   - the full training split;
   - `mainline_states_masked` and `ramp_states`.

   The following are out of scope:
   - ramp values inside the horizon;
   - mainline observations after the buffer.
2. **[ORG] 742068.** Task 1 is offline. Any released observation in the split may be used, at any time, including after the target cell. This makes our transductive Task 1 (EXP_NEW_001) explicitly legal.
3. **[ORG] 742910 (09-28).** The timestamp rule applies to the released files. Public external data is allowed under the Rules' External data section and must be disclosed in the finals. Any resource meant to identify hidden answers is prohibited.
4. **[ORG] 742350 (09-21).** In all 120 `queue_onset` windows, across all three splits, the queue appears **only at T+30**, and T+5..T+25 are empty. This comes from the selector keeping the earliest origin whose horizon contains a queue. Conditioning on the `condition` column is within the rules. `queue_ongoing` windows are not affected.
   - Quote: "What Task 2 still asks, and what will separate the leaderboard, is **which links queue**."
5. **[PART] 742328.** On the train onset windows, a model trained only on T+30 scored IoU 0.93, against 0.57 for a model spread over all steps. The windows are small, with a median of 3 queued cells.
   - **[OURS]** agrees: our CV onset IoU is ≈ 0.89, against an LB onset of 0.743 (probe on 09-28).
6. **[ORG] 742761 (09-28).** The organizer does not publish achievable targets.
   - "A perfect forecast scores 1.0."
   - "The labels are computed from the **underlying state** rather than from the noisy observations, so the labels themselves carry no noise floor."
7. **[PART] 742761.** "Onset windows cap out near 0.7" using only data ≤ T. The organizer neither confirmed nor refuted this.
   - **[OURS]:** our LB onset IoU is 0.743 (measured by probe), above that claimed cap.
8. **[ORG] 740536 / 740522.** The Task 2 horizon cells, and later a path through `window_history.parquet`, were readable in earlier releases. Both were closed, and then the val/private months were **rebuilt from new seeds** (740844).
   - **[PART] 740522:** at the time, the best genuine forecaster was about 0.5 S_queue, and "the recent jump of the top of the leaderboard" happened then.

## Rebuild and leaderboard

9. **[ORG] 740844 (09-11).**
   - Validation and private were regenerated from new seeds; **train is identical cell for cell**.
   - Scores from before 09-11 refer to values that no longer exist, and only post-rebuild submissions count for awards.
   - The Task 4 prior bug was fixed, so Task 4 scores read higher.
   - An empty submission now scores 0.
10. **[ORG] 744187 (09-28).**
    - The organizer is asking Kaggle to remove pre-rebuild scores; the leaderboard rules are unchanged.
    - **[PART] 740844 (Musa Peker, 09-13):** "most of the top entries have not been resubmitted since the rebuild".
    - **[INF]:** the 0.91–0.92 entries are probably pre-rebuild or leak-era scores. The credible post-rebuild frontier is about 0.90: KTK 0.90187, gichang 0.89702.

## Awards (744187)

11. **[ORG]** Key dates and requirements:
    - Registration by e-mail by **Oct 25, 23:59 AoE**, with team name, members' usernames, full names, affiliations and e-mails, and a primary contact.
    - Final Kaggle submission deadline: Nov 6.
    - Code and PDF report by **Nov 10**. The code must be a single entry script with a README and environment details.
    - Organizers rerun the code, and it must reproduce the selected submission **within 1% of its private score**.
    - Award score = 85% private LB + 15% report. The K–12 award is for K–12 students only.

## Topology (743819)

12. **[ORG] 743819 (09-28).** The benchmark **never uses from_node / to_node**. Instead:
    - longitudinal order = **milepost** (the Task 3 scorer sorts by milepost, then link_id);
    - connectivity for conservation = **incoming_link_ids / outgoing_link_ids**, restricted to the links present;
    - ramps = **ramp_attachment_map.csv**.

    Multiple components are expected. They come from connector (CONN-) links and from mainline links without detectors.
13. **[OURS] Topology audit, all 10 panels (`research/topology_audit.md`):**
    - `order_index` equals milepost rank exactly (Spearman 1.000) in every panel;
    - there is 1 station per link and no milepost ties;
    - links.csv v_f equals fd_parameters v_f.
    - **The travel direction differs by panel:**
      - in E/N panels the upstream neighbour has the *lower* milepost;
      - in W/S panels it has the *higher* one.

## Task 3 scorer (public code `repo/src/task3/score_task3.py`)

14. **[ORG code]** How S_LWR is computed:
    - It is **temporal per link**: dN = N(t+1) − N(t), where N = (q/max(v,1))·L on the reconstructed field (targets = our Task 1 values, the rest = released truth).
    - rhs = dt·(inflow + on_ramp − outflow − off_ramp), using the **organizer-held boundary flux projected onto the released observations**.
    - E = Σ|dN − rhs| / Σ|rhs|, and S_LWR = 1 − min(1, E).
    - In mode A, transitions count only where the attached ramp cells are valid.
15. **[ORG code]** S_FD is 1 − Σ|q − q_FD(k)| / Σ|q| per lane, using fd_parameters for all links. Its floor rule sets S_FD to 0 if more than 20% of submitted flows are below 50 vph.
16. **[ORG code comment]** "the conservation signal is 0.75% of the accumulated total … a topology-derived flux misses the simulated one by about 0.22%".
    - **[OURS]** local sensitivity measurements, D12_I405_S train, 60 days:

      | Prediction at target cells | S_LWR proxy |
      |---|---:|
      | Temporal interpolation | 0.52 |
      | Truth + 1 km/h speed noise | ≈ 0.89 |
      | Truth + 1% flow noise | ≈ 0.89 |
      | 3-point smoothed truth (strong oracle, includes the cell's own value) | **0.65** |

    - Truth S_FD = 0.991.

## Task 4

17. **[ORG code]** `repo/src/task4/build_task4_odme_artifacts.py` is NNLS on [A; √0.05·I] with the split's own prior, which is exactly our method.
    - **[PART] Hisernberg's docs (A0060):** the truth is consistent with a ridge fit to the prior, and S_ODME is about 0.994 for everyone who uses it.
