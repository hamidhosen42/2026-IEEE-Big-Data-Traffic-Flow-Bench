# Lineage of the best submission (tfb-g4, public LB 0.88092 = final candidate A)

Traced on 2026-10-06 from the Kaggle kernel metadata (`kernelDataSources`), starting at `tfb-g4`.
- 67 kernels are reachable. The full graph is in `lineage_g4.json`, and the exact source of every kernel is in `lineage_src/<slug>.py`.
- Most of these kernels are swap or compose steps. Each step replaced some rows and left the rest alone.
- The table below shows which kernel finally produced each block of rows. Those kernels are what a single-entry script must reproduce.

## Effective sources per row block

| Rows | Split | Final producer | Recipe (kernels to reproduce) |
|---|---|---|---|
| Task 1 state (speed, flow) | val + private | `tfb-g1` (recon + TV) on `tfb-g1d` (T1 mix + blackout override) | 0.5 × mean(FTA members v11, v12, v12s2..s7 = 8 old) + 0.5 × mean(FD members v13, v13s10, v13s11). Blackout rows (90 min after each official Task 2 origin) get 0.5 mix + 0.5 `tfb-t1dark`. Then FD-gated recon + TV (`swap_fdrecon.py` constants in the chain state). Each member = `t1_pipeline_v1x{a,b}.py`, two panel halves, kernel source `tfb-t2lab-full` (pseudo_<panel>.npz). |
| Task 2 onset | validation (March) | `tfb-t2osp` sp4 | CatBoost onset v13 (train split) with the three prior columns blended with the March days < d (a = n/(n+4), slots ±2). |
| Task 2 onset | private (April) | `tfb-t2opv` pv | Bag of 3 CatBoost models (train + released-March onset windows, weights 10/20/40), scenario-prior blend. **Bet A only**; hedge B uses `tfb-t2osp` base (train prior). |
| Task 2 ongoing | validation | `tfb-t2occ` fs4occ | F03 bag (np511 + id511 × 2 seeds) with all-finite inputs + 13 occupancy features. |
| Task 2 ongoing | private | `tfb-t2ogpvaf` pvo | 0.25 npB + 0.25 idB + 0.5 fullB (F03 + March faithful windows ×5, all-finite, scenario prior). **Bet A only**; hedge B uses `tfb-t2v22af` fs4af. |
| Task 4 path_flow | val + private | `tfb-final` (unchanged through every swap) | Prior-regularised NNLS, λ = 0.05, on each split's own counts and prior. |

## Hedge B (tfb-g5)

B equals G4 with two blocks replaced:
- private onset ← `tfb-t2osp` queue_submission_base.csv (v13 with the train prior);
- private ongoing ← `tfb-t2v22af` fs4af.

Public score: identical (0.88092).

## Compute (Kaggle CPU, observed)

| Component | Time |
|---|---|
| T1 member (one half) | several hours each; 11 members × 2 halves + t1dark × 2 |
| v22af / t2occ (4 × 511-leaf models) | 4.9 h / ~6 h |
| ogpvaf | ~2.4 h |
| opv / osp (CatBoost onset) | ~10 min |
| recon + TV, swaps | minutes |

## Notes for the single-entry script (awards)

**Requirements:**
- Rules: one entry script plus README and environment. The organizers rerun it, and it must reproduce the selected submission within 1 % of its private score.
- Code and report are due by Nov 10. Registration e-mail is due by Oct 25.

**Determinism:**
- LightGBM / CatBoost with fixed seeds and thread counts were reproducible across our reruns (e.g., the v13 CatBoost CONTROL gave 1.000000 agreement).
- Keep `thread_count` and `num_threads` fixed in the entry script.

**Simplification to check before writing it:**
- Fewer T1 members.
- The 1 % tolerance on the private score is generous compared with the member-bag effects (each about 0.0005).
