# Today's kernels (2026-11-03): k9 chain

Run in this order; each one must finish (status complete) before the next, because it reads the previous outputs.
Settings for all: CPU, internet OFF, competition data attached, the kernel outputs below attached as inputs.

| # | Kernel | Script | Input kernels | Must appear in the log |
|---|---|---|---|---|
| 1 | tfb-k9 | swap_lrtc_k9.py | hosen42/tfb-k5d | M OK (share > 0.97), B OK (share > 0.97), 'EXCLUDED panels: []', 'DONE | excluded []', submission_FDRTV.csv + submission_LRTC.csv |
| 2 | tfb-k10 | swap_k10.py | hosen42/tfb-k9, hosen42/tfb-t2osp, hosen42/tfb-t2ogveto | 'TOTAL changed rows: validation 0 | private 200', 'K10 OK 6980503' |
| 3 | tfb-t4val | t4_validate_k9.py | hosen42/tfb-k9, hosen42/tfb-k10 | 'T4 VALIDATION PASSED' |
| 4 | tfb-guard | guard_k9k10.py | hosen42/tfb-k7, hosen42/tfb-k8, hosen42/tfb-k9, hosen42/tfb-k10 | every line PASS, 'FINAL: CROSS-GENERATION SUBMISSION GUARD PASSED' |

Submit (only if all 4 pass): tfb-k9 `submission_FDRTV.csv` (candidate A) and tfb-k10 `submission_K10.csv` (candidate B).
Safety gates:
- Confirm the exact success messages above in each kernel log and its Kaggle status is **Complete**; a file existing in the output directory alone is **not** sufficient.
- `tfb-t4val` now raises `AssertionError` and fails its run if **any** Task 4 validation issue is found (including an empty ODME key); only the exact `T4 VALIDATION PASSED` line means it passed.
- `tfb-k9` can write intermediate `submission_LRTC.csv` (and potentially `submission_FDR.csv`) before a later step fails. Do **not** submit an intermediate CSV: require the final `submission_FDRTV.csv`, the `DONE | excluded []` log, and a complete run. Its assertions protect output invariants, but cannot guarantee zero CSVs after failure.
- `tfb-k10` writes `submission_K10.csv` near the end after its hard gates; require its exact success log and a complete run.
- `tfb-guard` checks cross-generation CSV row and task invariants. It **does not** rerun Task 2 sources with post-T input rows removed; the organizer's post-T time-safety requirement remains a separate test for award-track reproducibility.
- If any kernel fails, stop the chain, fix the cause, and rerun the dependent kernels using the new **completed** output versions; do not submit partial or stale outputs.
Push with the CLI from a folder: `kaggle kernels push -p <folder>` (each folder has kernel-metadata.json).
