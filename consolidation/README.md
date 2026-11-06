# TrafficFlowBench final picks — reproduction code (team hosen42)

> **Draft (milestone 3, 2026-11-03): final picks A3 = `tfb-k9` / B3 = `tfb-k10`.** Every stage of all four picks is ported.
> - The new stages of A3 / B3 (`t1_mix_k5d`, `t1_lrtc`, `assemble_k10`, members `t1_catb` / `t1_xgb`) reproduce their original
>   kernels byte for byte locally on one real panel-month (`tests/test_k9_parity.py`); their Kaggle parity kernels are generated, not run yet.
> - Milestone 1 (light stages of A2 / B2) passed Kaggle parity on 2026-10-09. The path_flow fix is to be re-confirmed with `tfb-par-t4` / `tfb-par-chaint1`.
> - The milestone-2 parity kernels (pseudo windows, Task 1 members, Task 2 ongoing) are generated and dry-run locally, and are waiting for their Kaggle runs.
> - Open items for the Nov 10 package: see *Open items* at the end.

One entry script rebuilds either final submission of the 2026 IEEE Big Data Cup – TrafficFlowBench competition from the
released data:

| Pick | Kaggle kernel | Public LB | What differs |
|---|---|---:|---|
| **A3** (final A) | `tfb-k9` | **0.88281** | Task 1: 18 members (A2's 15 + 2 CatBoost seeds + 1 XGBoost, `tfb-k5d`), then LRTC steps M + B, recon, TV. April onset = `tfb-t2opv` pv; April ongoing = `tfb-t2ogpvogf` pvogfv |
| **B3** (final B, hedge) | `tfb-k10` | **0.88281** | = A3 state rows; April onset = `tfb-t2osp` base; April ongoing = `tfb-t2ogveto` veto10 |
| A2 (previous A) | `tfb-j4` | 0.88176 | Task 1: 15 members (`tfb-i4d`), recon, TV; queue rows = A3's |
| B2 (previous B) | `tfb-j3` | 0.88176 | = A2 state rows; queue rows = B3's |

A3 and B3 share every state row (Task 1/3), every March queue row and every odme row (Task 4). Their queue rows equal A2's /
B2's block for block; only the state rows changed. The lineage of the new picks is in `LINEAGE_K9.md` / `lineage_K9.json`
(18 new kernels); the plan and risk analysis are in `DESIGN.md` (§8 for A3 / B3). `lineage_A2.json`, `lineage_K9.json` and
`lineage_src/` hold the 108 original Kaggle kernels behind the four picks.

## Quick start

```bash
# 1. environment: the Kaggle Python image B (Python 3.13); see ENVIRONMENT.md
# 2. data: the full competition release (9,688 files, 12.4 GB); --data-root is its kaggle_public folder
python run_all.py --data-root /data/kaggle_public --work-dir ./work --pick A3 --recipe full --threads 8
# result: ./work/submission_A3_full.csv  (+ .sha256), ./work/manifest.json
python run_all.py --data-root /data/kaggle_public --work-dir ./work --pick B3 --recipe full --threads 8
# reuses every shared stage (all Task 1 stages, t4, onset_sp, ongoing_val); only the B3 ongoing stages and the assembly run
```

| Option | Meaning |
|---|---|
| `--data-root` | the competition's `kaggle_public` folder (`submission_key.csv`, `sample_submission.csv`, `corridors/`, `task1/`, `task2/`, `task4/`) |
| `--work-dir` | cache of every stage: `<work>/kernels/<original kernel>/<original file>`, `stage.log` and `.done.json` per stage |
| `--pick {A3,B3,A2,B2}` | which submission to build (default A3; A2 / B2 = the previous picks) |
| `--recipe {full,lean1}` | Task 1 member bags: **full** = 8 old FTA + 6 FD members + t1bspec (+ t1catb, t1catb2, t1xgb for A3 / B3) = exact; **lean1** = old v12s4–s7, FD v13 / s10 / s11 / t1ramp + t1bspec (+ the 3 k members for A3 / B3) (DESIGN §3.3: expected private Δ −0.0002 to −0.0003 measured for A2, about 21 Kaggle-hours less; not measured for A3) |
| `--stages` | `all` (default), `light` (t4, onset_sp, onset_pv, the two Task 1 combine stages of the pick, assemble), or a comma list (`t4,onset_sp,...`) |
| `--threads` | LightGBM threads of the Task 1 LightGBM members (milestone 2). The CatBoost / XGBoost members of A3 / B3 keep their literal 4 threads, and every Task 2 and Task 4 stage keeps the thread count it had on Kaggle (4), because a queue flip there costs 0.001–0.002 of private score |
| `--env-threads` | OMP / BLAS threads of every stage process (default 4, the Kaggle value) |
| `--force` | rerun the selected stages even if cached |
| `--list` | print the stage plan and exit |

A rerun skips every stage whose `.done.json` fingerprint still matches (stage code, constants, input file sizes and
mtimes). It reruns everything downstream of a changed input. A crashed stage leaves no marker, so at most one stage is lost.

## Stages

Each stage is the original Kaggle kernel, vendored into `tfb/stages/` with only its I/O and recipe-constant lines changed.
Within one Task 1 family the 2–16 parts differ only in constants, so each family is one template. The constants come from `tfb/config.py` (`t1_part_cfg`).
`tests/test_templates.py` renders every template with each part's constants and checks that the result equals that part's original source as an AST, for all 31 parts and ogfa/ogfb.
Every changed line ends in `# [TFB]`, and `tests/test_static.py` enforces this. A stage writes the files of the kernel it
reproduces into `<work>/kernels/<kernel>/`. It reads upstream results through `TFB.kfile(<kernel>, <file>)` instead of
`/kaggle/input/**/<kernel>/**/<file>`.

| Stage | Reproduces | Source (`lineage_src/`) | Writes | Kaggle runtime | Peak RAM | Status |
|---|---|---|---|---:|---:|---|
| `pseudo` | `tfb-t2lab-full` | `tfb-t2lab-full.py` (`build_windows(P, cap=None)`) | `pseudo_<panel>.npz` (self-checked: counts + SHA-1) | 203 s | small | M2 |
| `t1m:<part>` (`t1_members` = all parts of the recipe) | 15 members × parts a/b(/c): 31 kernels (lean1: 19) | 6 family templates `t1_{fta,fta_v11,fd,dark,ramp,bspec}.py` from `tfb-t1v12s4a`, `v11a`, `v13a`, `darka`, `rampa`, `bspeca` | `state_submission_FTA.csv` | 1.3–5.6 h per part | 26.6–28.0 GB | M2 |
| `t1m:<part>` of A3 / B3 only | t1catb, t1catb2, t1xgb × parts a–d: 12 kernels | templates `t1_catb.py` (from `tfb-t1catba`), `t1_xgb.py` (from `tfb-t1xgba`) | `state_submission_FTA.csv` | catb 3.2–4.9 h, xgb 0.8–1.2 h per part | 10.8–26.1 GB | M3 |
| `t4` | `tfb-final` cell 7 | `tfb-final.py` (notebook) | `submission.csv` skeleton (odme rows = NNLS, state / queue rows 0), `odme_path_flow.csv` | ≈ 15 min | small | **M1** |
| `onset_sp` | `tfb-t2osp` | `tfb-t2osp.py` | `queue_submission_{base,sp2,sp4,sp8}.csv` | 333 s | < 8 GB (est.) | **M1** |
| `onset_pv` (A2) | `tfb-t2opv` | `tfb-t2opv.py` | `queue_submission_{base,sp4,pv,pvb}.csv` | 523 s | < 8 GB (est.) | **M1** |
| `ongoing_val` | `tfb-t2occ` | `tfb-t2occ.py` (`t2_occ.py`) | `queue_submission_fs4occ.csv`, `t2v26occ_probs.npz` | 5.5 h | 12–16 GB (est.) | M2 |
| `ongoing_pvogf` (A2) | `tfb-t2ogpvogf` | `tfb-t2ogpvogf.py` (`t2_pvogf.py`) | `queue_submission_pvogf.csv` (= pvogfv on private rows: 0/40 vetoes, so the pvo reference is not needed) | 4.6 h | 20–26 GB (est.) | M2 |
| `ogfeat_a/b`, `ogfeat_comb`, `ogveto` (B2) | `tfb-t2ogfa/b/c`, `tfb-t2ogveto` | template `t2_ogfeat.py` (part a / b), `t2_ogfc.py`, `t2_ogveto.py` | `queue_submission_veto10.csv` | 5.1 + 6.4 h + 1 min | 20–25 GB (est.) | M2 |
| `t1_mix` | `tfb-i4d` | `tfb-i4d.py` (`t1_t1dark_swap.py`) | `submission_T1DARK.csv` | 444 s | ≈ 10 GB (est.) | **M1** |
| `t1_recon` | `tfb-i4` | `tfb-i4.py` (`swap_fdrecon.py`) | `submission_{FDR,FDRTV,TV}.csv` | 499 s | 6–10 GB (est.) | **M1** |
| `assemble` | `tfb-j4` (A2) / `tfb-j3` (B2) | `tfb-j3.py` (`swap_queue2.py`) | `submission_J4.csv` / `submission_J3.csv` | 97 s | small | **M1** |
| `t1_mix_k5d` (A3 / B3) | `tfb-k5d` | `tfb-k5d.py` (`t1_div2_swap.py`) | `submission_K5D.csv` | ≤ 15.6 min | ≈ 10–12 GB (est.) | M3 |
| `t1_lrtc` (A3 / B3) | `tfb-k9` | `tfb-k9.py` (`swap_lrtc_k9.py`) | `submission_LRTC.csv`, `submission_FDRTV.csv` | 16–20 min (est.) | ≈ 7–11 GB (est.) | M3 |
| `assemble` (A3 / B3; module `assemble_k10`) | `tfb-k9` file (A3, folder `tfb-k9-asm`) / `tfb-k10` (B3) | `tfb-k10.py` (`swap_k10.py`) | `submission_K9.csv` / `submission_K10.csv` | ≈ 2 min | small | M3 |

**How the swap chain is replaced.** (A3 / B3: `t1_mix_k5d` and `t1_lrtc` take the places of `t1_mix` and `t1_recon`.)
- `t4` writes a skeleton submission in `sample_submission` order.
- `t1_mix` recomputes every state row on that skeleton: raw = 0.5·mean(FD bag) + 0.5·mean(old FTA bag). Blackout rows get 0.25·raw + 0.75·t1bspec.
  A3 / B3 (`t1_mix_k5d`): raw ← 0.65·raw + 0.35·mean(t1catb, t1catb2), then raw ← 0.65·raw + 0.35·t1xgb, then the same blackout override.
- `t1_recon` applies the FD-gated reconciliation and 1-D TV. A3 / B3 (`t1_lrtc`): first step M (flow LRTC (24, 40, 10) of the
  released month, 0.35 on ordinary rows) and step B (speed + flow LRTC, 0.15 on blackout rows), then the same R and T.
- `assemble` fills the queue rows with one part per (condition × split) block:
  - March onset ← osp `sp4`;
  - March ongoing ← occ `fs4occ`;
  - April onset ← opv `pv` (A2) or osp `base` (B2);
  - April ongoing ← `pvogfv` (A2) or `veto10` (B2).
- `assemble` asserts:
  - every queue row is covered;
  - ids are in `sample_submission` order;
  - all values are finite and ≥ 0;
  - queue values are binary;
  - the share of state flows below 50 vph is ≤ 20 % per panel × regime.
- Every constant lives in `tfb/config.py`.

## Runtime and hardware

| Run (8 cores / 32 GB, Linux x86-64, sequential stages) | Kaggle CPU-hours | Wall time |
|---|---:|---:|
| **full A3** (the selected file, exact recipe) | 143–147 | **≈ 93–104 h** |
| full A3 + B3 (shared stages run once) | 155–159 | ≈ 100–112 h |
| lean-1 A3 | 106–110 | ≈ 72–83 h |
| lean-1 A2 | 70–74 | ≈ 42–47 h |
| full A2 | 107–111 | 63–68 h |
| lean-1 B2 | ≈ 77 | ≈ 49 h |
| lean-1 A2 + B2 (shared stages run once) | 81–85 | ≈ 55 h |
| light stages only (`--stages light`, given member / ongoing outputs) | ≈ 0.7 | ≈ 0.5 h |

- **A3 / B3 vs A2.** The 12 CatBoost / XGBoost member parts add 35.8 Kaggle CPU-hours (catb 15.7 h, catb2 15.8 h, xgb 4.2 h).
  They keep their literal 4 threads (`--threads` does not apply), so on 8 cores they gain little from the faster machine:
  the wall-time rows above add ≈ 30–36 h for them (estimate; two can run side by side with ≥ 64 GB RAM). `t1_mix_k5d` +
  `t1_lrtc` add ≈ 20 min over `t1_mix` + `t1_recon`.
- **CPU / OS.** Linux x86-64, at least 8 cores. Apple Silicon is not supported for the official rerun (FMA / SIMD differences). The CPU flags are recorded in `manifest.json`.
- **RAM.**
  - 32 GB is the minimum: a Task 1 member peaked at 28.0 GB on Kaggle (A3 / B3 CatBoost / XGBoost part a: 26.1 GB), so members run one at a time.
  - 48–64 GB is comfortable, and two jobs can then run concurrently (about 35–40 % less wall time).
  - bspec part c (D7_I405_N) keeps its reduced settings whatever the RAM.
- **Disk.** 13 GB of data, about 10 GB of stage cache, and about 0.5 GB per submission-sized CSV (around 8 of them).
- The wall-time factors are DESIGN §3.3 estimates. They will be measured on the first member parity run.

## Determinism

- Seeds, thread counts and operation order are those of the original kernels:
  - CatBoost `random_seed 0`, `thread_count 4`;
  - LightGBM seed triples per member, `num_threads 4` for every Task 2 producer;
  - every `np.random.default_rng(<const>)` call site is kept unchanged.
- Each stage runs in its own process with:
  - `PYTHONHASHSEED=0`;
  - `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 4` (the Kaggle values);
  - the stage folder as the working directory (`catboost_info/` lands there, as in `/kaggle/working`).
- CSV round trips are the same as in the Kaggle chain (pandas defaults). `t4` writes path_flow at its CSV fixed point, which is the chain's value.
- **LightGBM** (DESIGN §4.8): the original kernels did not fix LightGBM's histogram path, so their exact files cannot be reproduced.
  - The Task 2 LightGBM stages use `deterministic=True, force_row_wise=True` (`tfb/config.py LGB_REPRO`) so that reruns are repeatable.
  - Expected drift vs the submitted files: a few dozen of 87,000 ongoing cells. Private bound ≈ 0.0012, expected about 0.
  - Task 1 members keep the original setting (smooth NUMERIC drift).
- Locally, a second run of every light stage reproduces the first byte for byte (see *Local smoke results*).
- **A3 / B3.** CatBoost (`random_seed` 20261005 / 20261016, `thread_count` 4) and XGBoost (`seed` 20261016, `nthread` 4, hist)
  are run exactly as in the originals; the catb / xgb members train no LightGBM. The LRTC of `t1_lrtc` uses `numpy.linalg.eigh`
  (LAPACK / BLAS): another BLAS build or CPU can move its values in the last bits (smooth, NUMERIC level); the runner keeps
  the BLAS threads at 4.
- **A3 text vs values.** `tfb-k9` wrote its file straight from memory; the entry script's A3 file passes one more pandas
  write / read cycle in `assemble` (A2 / B2 / B3 have that cycle in their original chains too). Every parsed value is
  identical (local test: 0 of 166,527 rows differ), but about 6 % of the lines are re-formatted in the last printed digit, so
  the A3 file is EXACT by value, not byte-identical. B3 is byte-identical.

## Development and parity testing

```bash
python tests/test_static.py                       # level 0: vendored diffs only on [TFB] lines; constants = original kernels
python tests/test_templates.py                    # level 0: each family template rendered per part == the original part (31 + 2)
python tests/test_csv_roundtrip.py                # the path_flow CSV fixed point (reproduces the 2026-10-09 chaint1 failure)
python tests/test_k9_parity.py /tmp/k9par         # level 1 local: t1_mix_k5d / t1_lrtc / assemble_k10 vs the ORIGINAL tfb-k5d / k9 / k10
                                                  # sources on one real panel-month (D12_I405_S April, synthetic key); ~30 s
python tests/make_fixture.py /tmp/fx              # miniature release + stand-in member / ongoing outputs (local files only)
python run_all.py --data-root /tmp/fx/kaggle_public --work-dir /tmp/fx/work --pick A2 --recipe full \
       --stages light --smoke --seed-kernel-outputs /tmp/fx/kernels
python run_all.py --data-root /tmp/fx/kaggle_public --work-dir /tmp/fx/work --pick B2 --recipe full --stages light --smoke
python bundle.py                                  # -> kaggle_parity/<slug>.py + SOURCES.md (kernel sources, push commands)
python tests/dryrun_parity.py /tmp/fx             # runs every generated kernel against a fake /kaggle/input
```

- `--seed-kernel-outputs DIR` (development only) copies `DIR/<kernel>/...` into the work dir. This covers stages that are not ported yet, for example downloaded Kaggle outputs.
- `--input-root DIR[:glob]` searches such a folder without copying it.
- `--smoke` turns on the kernels' own SMOKE switches (2 panels, 90 train days). It is for tests only.

**Kaggle parity (level 1–2 of DESIGN §4.7).** `bundle.py` writes one single-file kernel per check into `kaggle_parity/`:
- the needed `tfb` modules are inlined byte for byte;
- each kernel prints the image's package versions (`ENV` lines);
- it reruns the stage on the competition data and the original upstream outputs, attached as kernel sources;
- it compares the result with the original kernel output and prints:
  - `PARITY <check> <block> status=EXACT|NUMERIC|DIFF|FAIL ...` lines;
  - a final `PARITY SUMMARY verdict=PASS|CLOSE|FAIL` line.

Queue rows and path_flow must be EXACT. For state rows the kernel reports max |Δ| and RMS per split and for blackout / ordinary rows separately. `kaggle_parity/SOURCES.md` lists the sources and the push command of each kernel.

## Local results for A3 / B3 (2026-11-03)

| Test | Result |
|---|---|
| `tests/test_static.py` | OK: 23 vendored stages differ from their originals only on `[TFB]` lines; every A3 / B3 constant = the literals of tfb-k5d / k9 / k10; base chain j4 → k3d → k3 → k5d → k9 with queue-row asserts; SHA-1 of 108 sources |
| `tests/test_templates.py` | OK: 43 member parts (31 A2 / B2 + 12 catb / catb2 / xgb) rendered == original AST; ogfa / ogfb |
| `tests/test_k9_parity.py` | OK: original tfb-k5d / tfb-k9 / tfb-k10 vs the vendored stages through the runner: K5D, LRTC, FDRTV, K10 **byte-identical**; production assembly (zero-queue base + 4 parts) A3 == tfb-k9 FDRTV on every value, B3 == tfb-k10 byte-identical; runner cache OK; steps M / B changed 166,313 flows and 2,029 speeds (non-trivial) |
| generated parity kernels `tfb-par-k5d`, `-k9`, `-asma3`, `-asmb3`, dry run on the same inputs | PASS (EXACT on every block) |
| fixture smoke (`make_fixture.py`, `--smoke`) | t1_mix_k5d done on the 10-panel fixture; members `t1m:tfb-t1catba`, `t1m:tfb-t1catb2d`, `t1m:tfb-t1xgbb` done, SMOKE gates OK (4–5 s each). `t1_lrtc` cannot run on the fixture: its "LRTC member looks broken" gate needs member values that follow the released cells (the fixture's are random) |

## Local smoke results (A2 / B2)

**Fixture.** `tests/make_fixture.py` builds the fixture from local files only:
- D7_I10_E and D12_I5_S: fake release rebuilt from the lab exports, plus occupancy and real train window indexes;
- D12_I405_S: the real local files;
- D7_I10_W: a station-level clone of D12_I405_S with the real D7_I10_W windows, for the member SMOKE runs;
- 408,000 state rows, 29,520 queue rows, 4,842 odme rows.

**Runs.** All stages run with `--smoke`, which turns on the kernels' own SMOKE switches.

| Stages | Result |
|---|---|
| light A2 / B2 (t4, onset_sp, onset_pv, t1_mix, t1_recon, assemble) | all done (1–20 s each); B2 reuses every cached stage |
| pseudo | done; builder matches 7 of the 10 official train windows (D7_I10_E) |
| ongoing_val, ongoing_pvogf (A2) | done (35 s / 51 s); pvogf validation rows = fs4occ; no pvo → pvogf only |
| ogfeat_a / ogfeat_b / ogfeat_comb / ogveto (B2) | done; ogfc control "recomputed vs ogfa / ogfb files: agreement 1.000000" |
| members `t1m:` v13a, v12s4a, v11a, darka, rampa, bspeca, bspecc | done (7–11 s each), all SMOKE gates OK; bspec c keeps 13/21 train days (60 %) |

**Local dry run of the generated parity kernels** (`tests/dryrun_parity.py`; stand-in originals = earlier smoke outputs):
- 25 of the 49 kernels ran: 23 returned PASS (byte-identical reruns) and lean1 returned CLOSE by design. The other 24 were skipped because no local stand-in exists for them (member parts not smoke-run).
- `tfb-par-v12s4a` returned CLOSE: 4 of 42,499 flow values differed by 1.1e-13 vph (one ulp) between two runs on this Mac. That is threading or summation-order noise, and it is harmless for state rows.

## Open items for the Nov 10 package (A3 / B3)

| Item | Why | Estimate |
|---|---|---|
| Kaggle parity of the light A3 / B3 stages: `tfb-par-k5d`, `tfb-par-k9`, `tfb-par-asma3`, `tfb-par-asmb3`, `tfb-par-chaint1k` (`kaggle_parity/SOURCES.md`) | level 1–2 on the real data and the original outputs (no submission_key.csv locally) | ≈ 2.1 Kaggle-h, one session each, all in parallel → ≈ 1.5 h wall + reading |
| Re-push `tfb-par-t4` / `tfb-par-chaint1` (path_flow fix) | still open from milestone 1; the A3 path_flow rows are tfb-j4's | 1.1 Kaggle-h |
| Kaggle parity of the 12 catb / catb2 / xgb member parts (`tfb-par-catb*`, `-xgb*`) | the CatBoost / XGBoost versions of the original image were not logged; a rerun on today's image shows whether the members are EXACT or drift (expected: smooth, ≈ "a new seed" at most) | 38 Kaggle-h, 5 sessions in parallel → ≈ 10 h wall |
| Milestone-2 parity still not run (pseudo, Task 1 LightGBM members, Task 2 ongoing incl. LGB_REPRO repeatability) | unchanged from A2 / B2; A3 / B3 use the same stages | ≈ 140 Kaggle-h (see SOURCES.md) → 2–3 days wall at 5 sessions |
| **Task 2 post-T deletion rerun test** | the organizer reruns Task 2 with every row after each window's T removed; predictions must not change. A3 / B3 use the same Task 2 stages as A2 / B2 (`onset_sp`, `onset_pv`, `ongoing_val`, `ongoing_pvogf` / `ogfeat_*`, `ogveto`), so one test covers all four picks: build a copy of the release with the post-T rows of every Task 2 window removed, rerun the Task 2 stages, compare the queue files (EXACT required; LightGBM stages with LGB_REPRO) | 0.5 day to write the deleter (per window: masked / ramp / occupancy layers after T) + ≈ 22 Kaggle-h of Task 2 stages (or 12 h on 8 cores) |
| Off-Kaggle full run of A3 (8 cores / ≥ 32 GB Linux x86-64, Docker image B) | measures the real wall time / RAM of the 18-member recipe and gives the rerun-vs-submission table | ≈ 4–4.5 days wall (full A3; B3 + 0.5 day) – starts too late for Nov 10 unless begun by Nov 5; otherwise report the Kaggle parity results and the projected runtime |
| Package versions of the k-generation images | catb / xgb / k5d / k9 / k10 ran on Kaggle's default image of 10-10 … 11-03 (not image B); record `ENV` lines of the parity kernels and pin `catboost`, `xgboost` in `ENVIRONMENT.md` | with the parity runs |
| PDF report (15 % of the award score) | DESIGN §6 outline + the A3 additions (CatBoost / XGBoost members, LRTC steps M / B, LB 0.88176 → 0.88281) | 1–1.5 days |
| Measure k9 / k10 runtimes | not traced (expected 16–20 min / ≈ 2 min) | from the parity logs |
