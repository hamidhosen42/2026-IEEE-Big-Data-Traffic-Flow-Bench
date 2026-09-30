# Prompts given during this project

The user's prompts and experiment plans, kept verbatim, plus a record of what was done with each proposal.

| File | Date | What it asked for |
|---|---|---|
| [01_grandmaster_0.90_prompt.md](01_grandmaster_0.90_prompt.md) | 2026-09-28 | Full research → reproduce → validate → improve loop on the base notebook, targeting 0.90+ (28 phases) |
| [02_top_base_notebook_prompt.md](02_top_base_notebook_prompt.md) | 2026-09-29 | Edit the current best (0.85309) pipeline; Task 2 onset/ongoing separately; phases 1–16 |
| [03_EXP001-020_plan.md](03_EXP001-020_plan.md) | 2026-09-29 | Concrete EXP_001–EXP_020 table |
| [04_beat_0.87249.md](04_beat_0.87249.md) | 2026-09-29 | Beat 0.87249 (team Kragglenote2 Forwork); read and store the discussion forum |
| [05_research_librarian_1000_resources.md](05_research_librarian_1000_resources.md) | 2026-09-30 | 1000+ resource research → `research/` (resource_index.csv 2,114 unique, research_to_experiments.md, FINAL_RESEARCH_REPORT.md) |

## Outcome of every proposed experiment (measured, see experiments/TRACKER.md)

| Proposal | Result | Decision |
|---|---|---|
| Onset: train/predict T+5..T+25 (EXP_001/002, A6) | 39/40 official onset windows first queued at T+30, 1 has no observed queue; 96.9% of 2,128 pseudo windows; organizer confirmed 120/120 (discussion 742350) | rejected on evidence |
| Onset blend sweep (EXP_003, A1) | 0.8/0.2 ≈ +0.001 local; 30/70 = −0.0039 on LB | noise; keep 70/30 or CatBoost |
| Per-panel / per-TOD onset blend (A2/A3) | no nested-CV gain | rejected |
| Weekday-class prior (EXP_005, A4) | **LB 0.85639 (+0.0033)** | **kept** |
| Onset-start prior (Hisernberg-style `oprior`) | onset −0.003..−0.016 in 4/4 pseudo folds | rejected |
| Hazard / gradient features (EXP_007/008, A5) | inside noise | rejected |
| Ongoing keep radius / broad keep (EXP_009) | coverage 97.6–99.8%, part of v9 | kept (in v9) |
| Queue-boundary / shockwave features (EXP_010/011, B1/B2) | ≈ 0 | rejected |
| Decoder alternatives: thresholds, contiguity, morphology (EXP_012/013) | all ≤ expected-IoU; morphology −0.01..−0.015 | rejected |
| CatBoost (EXP_014, B4) | onset **+0.004..+0.014 in 4/4 folds, LB +0.0006**; ongoing correlation 0.997, no gain | kept for onset |
| Window weights / sampling (EXP_015) | no gain | rejected |
| Task 1 per-regime models (EXP_016, C1) | −0.002 on the validation-month holdout | rejected |
| Task 1 low-rank auxiliary features (EXP_017, C2) | **+0.0036 validation-month holdout; LB +0.0015** | **kept** |
| Task 1 → Task 3 LWR / density correction (EXP_018, C4/C5) | no T1+T3 gain (density pull, L1 density) | rejected |
| Task 4 λ sweep (EXP_019, D1–D3) | public probes by other teams: λ 5e-4..0.5 give the same solution, S_ODME ≈ 0.994 | not worth a run |
| Link order: milepost vs order_index (discussion 743819) | identical in all 8 Task 2 panels | no change needed |
