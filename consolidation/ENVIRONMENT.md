# Environment (draft, milestone 1)

## Reference environment: Kaggle Python image B

Every Task 2 producer, every swap and 6 of the 15 Task 1 members (4 of the 9 lean-1 members) ran in image B. It is the
reference environment for the organizers (DESIGN §4.5).

| Image | Digest | Python | Original kernels that ran in it |
|---|---|---|---|
| **B (reference)** | `gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a` | 3.13 | v13s10 / s11, t1dark / ramp / ramp2 / bspec, all Task 2 producers (osp, opv, occ, ogpvaf, ogpvogf, ogfa/b/c, ogveto), all swaps (i4d, i4, j3, j4) |
| A | `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9` | 3.12 | the 8 old FTA members, v13, t2v13, t2lab-full (pseudo windows), tfb-final (Task 4) |

The image-A components are rerun in image B:
- CatBoost already reproduced exactly from A to B: the CONTROL agreement against t2v13 onset was 1.000000.
- The Task 4 NNLS (scipy) and the old LightGBM members are measured by the parity kernels: `tfb-par-t4` and member parity (milestone 2).
- The expected effect is smooth, at most "a new seed" per member (about ±0.00005).

## Packages to pin

No log recorded the package versions. Fill this table from either source, and prefer the first:

1. **`pip freeze` inside image B** (no Kaggle push needed):
   ```bash
   docker pull gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a
   docker run --rm gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a \
       sh -c "python --version; pip freeze | grep -iE '^(numpy|pandas|scipy|lightgbm|catboost|pyarrow|scikit-learn)=='"
   ```
2. **The `ENV packages ...` line of every parity kernel log.** Every kernel in `kaggle_parity/` prints `importlib.metadata` versions and CPU flags first.
   - Caution: a kernel pushed today runs on Kaggle's *current* default image, which may be newer than B.
   - Compare the image of each parity run with B before reading a non-EXACT result as a porting bug.
   - The image of a run can be read with the same `GET /kernels/pull` trace used for `lineage_A2.json`.

Measured on 2026-10-09: the `ENV` lines of all 11 milestone-1 parity kernels. That image reproduced every image-B
stage byte for byte: t1mix, t1recon, asma2/b and chaint2a2/b gave identical files, and onset sp/pv was EXACT.

| Package | Kaggle image of the parity runs (2026-10-09) | Image B (original runs) | Used by |
|---|---|---|---|
| python | 3.13.15 | 3.13 (minor not logged) | all |
| numpy | 2.1.3 | TBD (`pip freeze`) | all |
| pandas | 2.3.3 | TBD | all (the default CSV float parser matters: see "CSV round trips" below) |
| scipy | 1.16.3 | TBD | `t4` (`scipy.optimize.nnls`; EXACT vs tfb-final, which ran in image A) |
| lightgbm | 4.6.0 | TBD | Task 1 members, Task 2 ongoing |
| catboost | 1.2.10 | TBD | `onset_sp`, `onset_pv` (EXACT) |
| xgboost | not imported by the parity runs of 10-09 | – | `t1_xgb` (A3 / B3 member `tfb-t1xgb`; local 2.1.4). The k-generation kernels (catb / xgb / k5d / k9 / k10) ran on Kaggle's default image of 2026-10-10 … 11-03, not image B: pin from the `ENV` lines of `tfb-par-catb*` / `tfb-par-xgb*` |
| pyarrow | 23.0.1 | TBD | parquet reads |
| scikit-learn | 1.6.1 | TBD | not imported by any stage |

The parity image ran on 4 vCPU (AMD EPYC 7B12 or Intel Xeon 2.20 GHz; AVX2 + FMA, no AVX-512) with 31.3 GB RAM.
Until image B's own `pip freeze` is available, **pin the versions in the second column**: they are proven to reproduce
the original outputs for every stage tested so far.

**CSV round trips.** pandas' default float parser (`read_csv`, `float_precision=None`) is not correctly rounded.
- One write + read cycle can move a value by an ulp, and repeated cycles converge only after several steps.
- The Kaggle chain wrote path_flow through 26+ such cycles. The `t4` stage therefore writes the fixed point of that cycle, which is exactly the chain's value (`tests/test_csv_roundtrip.py`).
- Keep pandas' default parser: do not switch to `float_precision='round_trip'` or pyarrow CSV parsing anywhere.

For reference, the local lab environment is **not** the Kaggle environment. It was used only for the local smoke runs:
Python 3.9.6 (macOS arm64), numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, lightgbm 4.6.0, catboost 1.2.10, pyarrow 21.0.0,
scikit-learn 1.6.1.

## Container (draft)

```dockerfile
FROM gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a
COPY consolidation /opt/tfb
ENV PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
WORKDIR /opt/tfb
ENTRYPOINT ["python", "run_all.py"]
# docker run --rm -v /data/kaggle_public:/data:ro -v $PWD/work:/work <image> --data-root /data --work-dir /work --pick A2 --recipe lean1 --threads 8
```

A slim virtual environment with the same pins (`requirements.txt`, written once the TBD versions are filled) is the light
alternative. It needs Python 3.13 on Linux x86-64.

## Runtime settings applied by the runner

These are set for every stage process; they do not have to be set by hand:

| Variable | Value | Why |
|---|---|---|
| `PYTHONHASHSEED` | 0 | no hash-order dependence |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` | `--env-threads`, default 4 | the Kaggle machines had 4 vCPU; BLAS reductions keep Kaggle's thread count |
| LightGBM `num_threads` / CatBoost `thread_count` | 4, inside the vendored kernels | exactly as on Kaggle for every Task 2 producer; Task 1 members take `--threads` (milestone 2) |
| working directory | the stage's output folder | as `/kaggle/working`; CatBoost writes `catboost_info/` there |

`manifest.json` in the work dir records the package versions, CPU model and flags (AVX2 / AVX-512), core count, memory,
thread settings and a data-root summary (file count, size) for every run.

## Hardware

- **CPU / OS.** Linux x86-64 with at least 8 cores; no Apple Silicon for the official rerun.
- **RAM.** 32 GB minimum (a Task 1 member peaks at 28.0 GB); 48–64 GB is comfortable.
- **Disk.** About 30 GB free: 13 GB of data plus the stage cache and submission-sized CSVs.
