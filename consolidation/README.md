# TrafficFlowBench final picks — reproduction code (team hosen42)

> **Draft (milestone 2).** Every stage of both picks is ported.
> - Milestone 1 (light stages) passed Kaggle parity on 2026-10-09. The path_flow fix is to be re-confirmed with `tfb-par-t4` / `tfb-par-chaint1`.
> - The milestone-2 parity kernels (pseudo windows, Task 1 members, Task 2 ongoing) are generated and dry-run locally, and are waiting for their Kaggle runs.

One entry script rebuilds either final submission of the 2026 IEEE Big Data Cup – TrafficFlowBench competition from the
released data:

| Pick | Kaggle kernel | Public LB | What differs |
|---|---|---:|---|
| **A2** | `tfb-j4` | 0.88176 | April onset rows = `tfb-t2opv` pv bag; April ongoing rows = `tfb-t2ogpvogf` pvogfv |
| **B2** | `tfb-j3` | 0.88176 | April onset rows = `tfb-t2osp` base; April ongoing rows = `tfb-t2ogveto` veto10 |

Both picks share every state row (Task 1/3), every March queue row and every odme row (Task 4). The plan, lineage and risk
analysis are in `DESIGN.md`. `lineage_A2.json` and `lineage_src/` hold the 88 original Kaggle kernels behind the picks.

## Quick start

```bash
# 1. environment: the Kaggle Python image B (Python 3.13); see ENVIRONMENT.md
# 2. data: the full competition release (9,688 files, 12.4 GB); --data-root is its kaggle_public folder
python run_all.py --data-root /data/kaggle_public --work-dir ./work --pick A2 --recipe lean1 --threads 8
# result: ./work/submission_A2_lean1.csv  (+ .sha256), ./work/manifest.json
python run_all.py --data-root /data/kaggle_public --work-dir ./work --pick B2 --recipe lean1 --threads 8
# reuses every shared stage; only the B2-specific stages and the assembly run
```

| Option | Meaning |
|---|---|
| `--data-root` | the competition's `kaggle_public` folder (`submission_key.csv`, `sample_submission.csv`, `corridors/`, `task1/`, `task2/`, `task4/`) |
| `--work-dir` | cache of every stage: `<work>/kernels/<original kernel>/<original file>`, `stage.log` and `.done.json` per stage |
| `--pick {A2,B2}` | which final submission to build |
| `--recipe {full,lean1}` | Task 1 member bags: **full** = 8 old FTA + 6 FD members + t1bspec (exact); **lean1** = old v12s4–s7, FD v13 / s10 / s11 / t1ramp + t1bspec (DESIGN §3.3: expected private Δ −0.0002 to −0.0003, about 21 Kaggle-hours less) |
| `--stages` | `all` (default), `light` (the milestone-1 stages), or a comma list (`t4,onset_sp,...`) |
| `--threads` | LightGBM threads of the Task 1 members (milestone 2). Every Task 2 and Task 4 stage keeps the thread count it had on Kaggle (4), because a queue flip there costs 0.001–0.002 of private score |
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
| `t4` | `tfb-final` cell 7 | `tfb-final.py` (notebook) | `submission.csv` skeleton (odme rows = NNLS, state / queue rows 0), `odme_path_flow.csv` | ≈ 15 min | small | **M1** |
| `onset_sp` | `tfb-t2osp` | `tfb-t2osp.py` | `queue_submission_{base,sp2,sp4,sp8}.csv` | 333 s | < 8 GB (est.) | **M1** |
| `onset_pv` (A2) | `tfb-t2opv` | `tfb-t2opv.py` | `queue_submission_{base,sp4,pv,pvb}.csv` | 523 s | < 8 GB (est.) | **M1** |
| `ongoing_val` | `tfb-t2occ` | `tfb-t2occ.py` (`t2_occ.py`) | `queue_submission_fs4occ.csv`, `t2v26occ_probs.npz` | 5.5 h | 12–16 GB (est.) | M2 |
| `ongoing_pvogf` (A2) | `tfb-t2ogpvogf` | `tfb-t2ogpvogf.py` (`t2_pvogf.py`) | `queue_submission_pvogf.csv` (= pvogfv on private rows: 0/40 vetoes, so the pvo reference is not needed) | 4.6 h | 20–26 GB (est.) | M2 |
| `ogfeat_a/b`, `ogfeat_comb`, `ogveto` (B2) | `tfb-t2ogfa/b/c`, `tfb-t2ogveto` | template `t2_ogfeat.py` (part a / b), `t2_ogfc.py`, `t2_ogveto.py` | `queue_submission_veto10.csv` | 5.1 + 6.4 h + 1 min | 20–25 GB (est.) | M2 |
| `t1_mix` | `tfb-i4d` | `tfb-i4d.py` (`t1_t1dark_swap.py`) | `submission_T1DARK.csv` | 444 s | ≈ 10 GB (est.) | **M1** |
| `t1_recon` | `tfb-i4` | `tfb-i4.py` (`swap_fdrecon.py`) | `submission_{FDR,FDRTV,TV}.csv` | 499 s | 6–10 GB (est.) | **M1** |
| `assemble` | `tfb-j4` (A2) / `tfb-j3` (B2) | `tfb-j3.py` (`swap_queue2.py`) | `submission_J4.csv` / `submission_J3.csv` | 97 s | small | **M1** |

**How the swap chain is replaced.**
- `t4` writes a skeleton submission in `sample_submission` order.
- `t1_mix` recomputes every state row on that skeleton: raw = 0.5·mean(FD bag) + 0.5·mean(old FTA bag). Blackout rows get 0.25·raw + 0.75·t1bspec.
- `t1_recon` applies the FD-gated reconciliation and 1-D TV.
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
| lean-1 A2 | 70–74 | **≈ 42–47 h** |
| full A2 | 107–111 | 63–68 h |
| lean-1 B2 | ≈ 77 | ≈ 49 h |
| lean-1 A2 + B2 (shared stages run once) | 81–85 | ≈ 55 h |
| light stages only (`--stages light`, given member / ongoing outputs) | ≈ 0.7 | ≈ 0.5 h |

- **CPU / OS.** Linux x86-64, at least 8 cores. Apple Silicon is not supported for the official rerun (FMA / SIMD differences). The CPU flags are recorded in `manifest.json`.
- **RAM.**
  - 32 GB is the minimum: a Task 1 member peaked at 28.0 GB on Kaggle, so members run one at a time.
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

## Development and parity testing

```bash
python tests/test_static.py                       # level 0: vendored diffs only on [TFB] lines; constants = original kernels
python tests/test_templates.py                    # level 0: each family template rendered per part == the original part (31 + 2)
python tests/test_csv_roundtrip.py                # the path_flow CSV fixed point (reproduces the 2026-10-09 chaint1 failure)
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

## Local smoke results

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
