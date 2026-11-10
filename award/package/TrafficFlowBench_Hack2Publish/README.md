# TrafficFlowBench (2026 IEEE Big Data Cup) — final package of team Hack2Publish

Kaggle team: **Hack2Publish** (Kaggle user `hosen42`). Members: Md. Hamid Hosen (East Delta University), Md. Foysal Emon Shanto
(Jashore University of Science and Technology), Md. Esfer Abdus Sami (North South University).

Selected final submissions (both public 0.88281; private 0.87590, rank 83):

| Pick | Kaggle notebook | Rebuild command |
|---|---|---|
| A3 | `tfb-k9` | `python code/run_all.py --data-root <kaggle_public> --work-dir ./work --pick A3 --recipe full` |
| B3 | `tfb-k10` | `python code/run_all.py --data-root <kaggle_public> --work-dir ./work --pick B3 --recipe full` |

`<kaggle_public>` is the competition release folder (with `submission_key.csv`, `sample_submission.csv`, `corridors/`, `task1/`,
`task2/`, `task4/`). The output is `./work/submission_A3_full.csv` (or `_B3_`) with a `.sha256` and `manifest.json`.

## Contents

| Path | What it is |
|---|---|
| `report/TrafficFlowBench_Hack2Publish_report.pdf` | the PDF report: method, analysis and results across the four tasks |
| `code/` | the reproduction code. **Single entry script: `code/run_all.py`.** Start with `code/README.md` (quick start, stage table, options, status at submission) |
| `code/requirements.txt`, `code/ENVIRONMENT.md` | Python 3.13 and package versions; hardware, threads and determinism notes |
| `code/tfb/` | the pipeline package: one module per stage, vendored from the original Kaggle kernels with only I/O lines changed (marked `# [TFB]`) |
| `code/lineage_src/`, `code/LINEAGE_K9.md`, `code/lineage_*.json` | the 108 original Kaggle kernels behind the picks and their lineage |
| `code/kaggle_parity/` | single-file Kaggle kernels that rerun stages and compare them with the original outputs (`SOURCES.md`) |
| `code/tests/` | static, template and CSV round-trip tests (`python code/tests/test_static.py` etc.) |
| `t2_postT_test/` | our rerun of Task 2 with every row after each window's origin removed (scripts and result tables; 0 changed cells) |

## Compute

All stages ran on Kaggle CPU sessions (4 vCPU, about 30 GB RAM). The full recipe of pick A3 needs about 145 Kaggle CPU-hours,
dominated by the Task 1 member models; `code/README.md` lists every stage with its runtime and lets the stages run separately
or in parallel. Task 2 uses only data stamped at or before each window's origin.
