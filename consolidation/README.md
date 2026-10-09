# TrafficFlowBench final picks — reproduction code (team hosen42)

> **Draft (milestone 1).** The framework and the light stages are ported and smoke-tested. The heavy stages (pseudo
> windows, the Task 1 LightGBM members and the Task 2 ongoing LightGBM producers) are registered but **not ported yet**
> (milestone 2). Until then, their outputs must be provided as files (see *Development*).

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
Every changed line ends in `# [TFB]`, and `tests/test_static.py` enforces this. A stage writes the files of the kernel it
reproduces into `<work>/kernels/<kernel>/`. It reads upstream results through `TFB.kfile(<kernel>, <file>)` instead of
`/kaggle/input/**/<kernel>/**/<file>`.

| Stage | Reproduces | Source (`lineage_src/`) | Writes | Kaggle runtime | Peak RAM | Status |
|---|---|---|---|---:|---:|---|
| `pseudo` | `tfb-t2lab-full` | `tfb-t2lab-full.py` (`build_windows(P, cap=None)`) | `pseudo_<panel>.npz` | 203 s | small | M2 |
| `t1_members` | 15 members × parts a/b(/c) | `tfb-t1v11a.py` … `tfb-t1bspecc.py` | `state_submission_FTA.csv` | 4.6–9.6 h per member | 26.6–28.0 GB | M2 |
| `t4` | `tfb-final` cell 7 | `tfb-final.py` (notebook) | `submission.csv` skeleton (odme rows = NNLS, state / queue rows 0), `odme_path_flow.csv` | ≈ 15 min | small | **M1** |
| `onset_sp` | `tfb-t2osp` | `tfb-t2osp.py` | `queue_submission_{base,sp2,sp4,sp8}.csv` | 333 s | < 8 GB (est.) | **M1** |
| `onset_pv` (A2) | `tfb-t2opv` | `tfb-t2opv.py` | `queue_submission_{base,sp4,pv,pvb}.csv` | 523 s | < 8 GB (est.) | **M1** |
| `ongoing_val` | `tfb-t2occ` | `tfb-t2occ.py` | `queue_submission_fs4occ.csv`, `t2v26occ_probs.npz` | 5.5 h | 12–16 GB (est.) | M2 |
| `ongoing_pvogf` (A2) | `tfb-t2ogpvogf` | `tfb-t2ogpvogf.py` | `queue_submission_pvogfv.csv` | 4.6 h | 20–26 GB (est.) | M2 |
| `ogfeat_a/b`, `ogfeat_comb`, `ogveto` (B2) | `tfb-t2ogfa/b/c`, `tfb-t2ogveto` | same names | `queue_submission_veto10.csv` | 5.1 + 6.4 h + 1 min | 20–25 GB (est.) | M2 |
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
- CSV round trips are the same as in the Kaggle chain (pandas defaults). Floats therefore pass the same parse/print sequence.
- Locally, a second run of every light stage reproduces the first byte for byte (see *Local smoke results*).

## Development and parity testing

```bash
python tests/test_static.py                       # level 0: vendored diffs only on [TFB] lines; constants = original kernels
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

## Local smoke results (milestone 1, fixture: 3 panels, 381,170 rows)

| Stage | Local time | Peak RSS | Notes |
|---|---:|---:|---|
| t4 | 7 s | 0.5 GB | 0 unmatched odme keys |
| onset_sp | 7 s | 0.7 GB | CONTROL lookups (tfb-t2v13, tfb-e3) are absent and skipped, as designed |
| onset_pv | 19 s | 0.85 GB | `pv == sp4 on validation rows: True` |
| t1_mix | 5–8 s | 0.9 GB | blackout share 2.07 % (the kernel gate needs 0.5–5 %) |
| t1_recon | 7 s | 1.0 GB | 0 panels excluded; R and T exercised |
| assemble A2 / B2 | 1 s | 0.4 GB | 29,520 / 29,520 queue rows covered |

All 11 generated parity kernels were dry-run against a fake `/kaggle/input`:
- 10 returned PASS. Every rerun was byte-identical to the first run on this machine.
- `tfb-par-lean1` returned CLOSE, as expected: the lean recipe differs from the full one by design.
