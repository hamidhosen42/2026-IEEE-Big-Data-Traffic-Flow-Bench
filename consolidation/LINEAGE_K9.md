# Lineage of the final picks A3 = tfb-k9 and B3 = tfb-k10 (both public LB 0.88281, submitted 2026-11-03)

Traced on 2026-11-03 with `kaggle kernels pull -m` (metadata `kernel_sources` + source) from the two roots back to the
kernels that `consolidation/` already reproduces (`tfb-i4`, `tfb-j4`'s parts, the Task 2 producers, the 31 Task 1 member
parts of `lineage_A2.json`).
- 18 kernels are new upstream of the picks, plus 2 reference kernels (`tfb-k5`, `tfb-k7`). Graph, runtimes, peak RAM and
  SHA-1: `lineage_K9.json`. Exact sources: `lineage_src/<slug>.py` (all byte-identical to `submissions/` and `kaggle_jobs/`, checked with `cmp`).
- `tests/test_static.py` re-checks the SHA-1 of every source, every constant of the vendored stages and the base chain.

## The chain from the previous picks

```
tfb-j4 (= A2, 0.88176) ──base──> tfb-k3d ──> tfb-k3 ──base──> tfb-k5d ──base──> tfb-k9 (= A3, 0.88281) ──base──> tfb-k10 (= B3, 0.88281)
            31 old members ──────────┤                  ├── 31 old members           │                     ├── tfb-t2osp  base
            t1catb a-d, t1catb2 a-d ─┘                  ├── t1catb a-d, t1catb2 a-d  │                     └── tfb-t2ogveto veto10
                                                        └── t1xgb a-d                └ (released data only: LRTC of the month)
```

| Kernel | Kernel sources | What it changes | Rows that reach the picks | Source (`lineage_src/`) = local file |
|---|---|---|---|---|
| `tfb-t1catb{a,b,c,d}` | `tfb-t2lab-full` | **new Task 1 member**: CatBoost residual model (depth 8 × 2000 it + depth 10 × 1000 it, averaged; lr 0.12, l2 10, Bernoulli 0.85, border 64, `random_seed` 20261005, 4 threads) on the 94 v13 FD columns; v13 rows (row rng 9, transductive rng 10000+), ≤ 2.4 M rows (rng 21). Panels: a = D7_I405_N, D7_I10_W; b = D7_I210_E, D7_I210_W, D12_I405_S; c = D12_I5_N, D12_I5_S; d = D7_I10_E, D7_I405_S, D12_I405_N | `state_submission_FTA.csv` → k3d / k5d | `tfb-t1catba..d.py` = `kaggle_jobs/t1_catb_member_a..d.py` |
| `tfb-t1catb2{a,b,c,d}` | `tfb-t2lab-full` | second CatBoost seed set: `CB_SEED` 31, `random_seed` 20261016, row rng 19, transductive rng 12000+ | same | `tfb-t1catb2a..d.py` = `kaggle_jobs/t1_catb2_member_a..d.py` |
| `tfb-k3d` | 31 old member parts, 8 catb parts, `tfb-j4` | T1 mix = tfb-i4d mix, then 0.65 mix + 0.35 mean(catb, catb2), then the blackout override; base `tfb-j4/submission_J4.csv` | **only queue / path_flow rows** (as the base of k3 → k5d) | `tfb-k3d.py` = `kaggle_jobs/t1_catb2_swap.py` |
| `tfb-k3` | `tfb-k3d` | recon + TV (tfb-i4 code) = old pick of 10-16 (0.88201) | **only queue / path_flow rows** | `tfb-k3.py` = `kaggle_jobs/swap_fdrecon_k3.py` |
| `tfb-t1xgb{a,b,c,d}` | `tfb-t2lab-full` | **new Task 1 member**: XGBoost hist (max_depth 9, eta 0.06, 1500 rounds, subsample / colsample 0.8, min_child_weight 50, lambda 10, max_bin 256, seed 20261016, 4 threads); rows and parts as t1catb | `state_submission_FTA.csv` → k5d | `tfb-t1xgba..d.py` = `kaggle_jobs/t1_xgb_member_a..d.py` |
| `tfb-k5d` | 31 old parts, 8 catb parts, 4 xgb parts, `tfb-k3` | T1 mix K5D: i4d mix → 0.65 mix + 0.35 mean(catb, catb2) → 0.65 mix + 0.35 xgb → blackout rows 0.25 mix + 0.75 t1bspec. **Every state row is recomputed**; queue / path_flow rows copied from the base (asserted) | all state rows (before M/B/R/T); queue rows of tfb-j4 | `tfb-k5d.py` = `kaggle_jobs/t1_div2_swap.py` |
| `tfb-k9` | `tfb-k5d` | step **M**: flow LRTC of the released month (per panel × split, ranks (24, 40, 10), 60 EM iterations, station-mean normalised), `q ← max(0.65 q + 0.35 max(q_lrtc, 20), 20)` on ordinary state rows; step **B**: blackout rows `v ← clip(0.85 v + 0.15 v_lrtc(8,20,20), 3, 1.05 vf)`, `q ← max(0.85 q + 0.15 q_lrtc(8,40,10), 20)`; steps **R** (FD-gated recon) and **T** (1-D TV) verbatim from tfb-i4 / tfb-k5. Queue / path_flow rows asserted unchanged | **pick A3** = `submission_FDRTV.csv` | `tfb-k9.py` = `kaggle_jobs/swap_lrtc_k9.py` = `submissions/76_K9/tfb-k9.py` |
| `tfb-k10` | `tfb-k9`, `tfb-t2osp`, `tfb-t2ogveto` | queue swap (tfb-j3 code + 4 gates): private onset ← osp `base`, private ongoing ← ogveto `veto10`; asserts that no March row changed | **pick B3** = `submission_K10.csv` | `tfb-k10.py` = `kaggle_jobs/swap_k10.py` = `submissions/77_K10/tfb-k10.py` |
| `tfb-j4` | `tfb-i4`, `tfb-t2ogpvogf` | (already reproduced as A2) | its queue / path_flow rows = the A3 queue / path_flow rows | `tfb-j4.py` = `kaggle_jobs/swap_tfb-j4.py` |
| `tfb-k5` (reference) | `tfb-k5d` | R + T on k5d (0.88225) = k9 without M, B | – | `tfb-k5.py` |
| `tfb-k7` (reference) | `tfb-k5d` | M with ranks (8, 40, 10), no B, + R + T (0.88259) | – | `tfb-k7.py` |

Not upstream: `tfb-k1d`, `tfb-k1` (one CatBoost seed; replaced by k3d / k3), `tfb-k2`, `tfb-k4`, `tfb-k6`, `tfb-k8` (hedges of older picks).

## Effective producers per row block

| Rows | Split | A3 = tfb-k9 | B3 = tfb-k10 | vs A2 / B2 |
|---|---|---|---|---|
| Task 1 state | val + private | tfb-k9 M + B + R + T on tfb-k5d (18 members: 8 old FTA, 6 FD, t1bspec, 2 × CatBoost, XGBoost) | = A3 | **changed** (A2/B2: tfb-i4 R + T on tfb-i4d, 15 members) |
| Task 2 onset | validation | tfb-t2osp sp4 | = A3 | same |
| Task 2 ongoing | validation | tfb-t2occ fs4occ | = A3 | same |
| Task 2 onset | private | tfb-t2opv pv | tfb-t2osp base | same as A2 / B2 |
| Task 2 ongoing | private | tfb-t2ogpvogf pvogfv | tfb-t2ogveto veto10 | same as A2 / B2 |
| Task 4 path_flow | val + private | tfb-final (unchanged through every kernel) | = A3 | same |

So the new picks differ from A2 / B2 only in the Task 1 state rows; every queue and odme row is the same.

## How the entry script reproduces it (stages, `tfb/config.py`)

- `t1m:<part>` for the 12 new parts: templates `tfb/stages/t1_catb.py` (catb + catb2; constants `cb_seed`, `cb_random_seed`, `row_rng`, `rng_base`, `lgb_seeds` of the dead `pF` line, panels) and `tfb/stages/t1_xgb.py` (panels only). `tests/test_templates.py` renders them for all 12 parts and compares the AST with each original.
- `t1_mix_k5d` = `tfb-k5d` (base: the T4 skeleton instead of tfb-k3, as `t1_mix` replaced tfb-i2 for tfb-i4d).
- `t1_lrtc` = `tfb-k9` (base: our tfb-k5d).
- `assemble` (module `assemble_k10` = `tfb-k10`) on our tfb-k9 FDRTV, with the 4 condition × split parts of A2 (A3) or B2 (B3). A3 writes into the virtual kernel folder `tfb-k9-asm` (the original chain had no separate A3 assembly; its queue rows came through j4 → k3d → k3 → k5d → k9).

## Compute (Kaggle CPU, observed)

| Component | Runtime | Peak RSS |
|---|---|---|
| t1catb a / b / c / d | 3.5 / 4.1 / 3.2 / 4.9 h (15.7 h) | 26.1 / 22.0 / 19.3 / 10.9 GB |
| t1catb2 a / b / c / d | 3.4 / 4.2 / 3.5 / 4.8 h (15.8 h) | 26.1 / 22.2 / 19.2 / 10.9 GB |
| t1xgb a / b / c / d | 1.0 / 1.2 / 0.8 / 1.2 h (4.2 h) | 26.0 / 22.2 / 19.2 / 10.8 GB |
| tfb-k5d | ≤ 15.6 min (push → complete) | not logged (≈ tfb-i4d + 12 files) |
| tfb-k9 | not traced; expected 16–20 min (tfb-k7, one LRTC per panel-month: 771 s; k9 runs three) | ≈ tfb-k5 + < 0.4 GB |
| tfb-k10 | not traced (tfb-j3, same code: 97 s) | small |
