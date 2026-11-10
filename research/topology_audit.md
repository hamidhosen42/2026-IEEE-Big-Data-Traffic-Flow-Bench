# Topology audit (EXP_021 prerequisite)

- **Run:** 2026-10-01, Kaggle kernel `hosen42/tfb-topo-audit`, script `kaggle_jobs/topology_audit.py`.
- **Data:** per-link rows in [`topology_audit_links.csv`](topology_audit_links.csv) (link, sensored upstream 1–3, sensored downstream 1–3, ramps, milepost rank, spacing, lanes). Per-panel summary in [`topology_audit_summary.csv`](topology_audit_summary.csv).
- **Contract** (organizer, discussion 743819): order = milepost, then link_id; conservation connectivity = incoming/outgoing_link_ids; ramps = ramp_attachment_map.
- Up/downstream neighbours were found by walking incoming/outgoing through latent (sensor-less) mainline links. Connectors are treated as boundaries.

## Per-panel summary

| Panel | T2 | Dir | Sensored links | Spearman(order_index, milepost) | Upstream has **higher** milepost | Milepost neighbours adjacent in topology | Lane drops / adds | Median / max spacing (km) |
|---|---|---|---:|---:|---:|---|---|---|
| D7_I10_E | ✓ | E | 146 | 1.000 | 0.000 (85 checks) | 83/145 | 19 / 19 | 0.26 / 3.08 |
| D7_I10_W | ✓ | W | 107 | 1.000 | **1.000** (28) | 28/106 | 14 / 17 | 0.34 / 2.49 |
| D7_I210_E | ✓ | E | 297 | 1.000 | 0.000 (290) | 288/296 | 9 / 7 | 0.17 / 3.89 |
| D7_I210_W | ✓ | W | 81 | 1.000 | **1.000** (15) | 15/80 | 11 / 9 | 0.40 / 4.33 |
| D7_I405_N | ✓ | N | 352 | 1.000 | 0.000 (332) | 332/351 | 17 / 19 | 0.10 / 1.50 |
| D7_I405_S | ✓ | S | 109 | 1.000 | **1.000** (15) | 15/108 | 19 / 21 | 0.29 / 1.72 |
| D12_I5_N | ✓ | N | 258 | 1.000 | 0.004 (256) | 256/257 | 21 / 22 | 0.18 / 1.80 |
| D12_I5_S | ✓ | S | 100 | 1.000 | **1.000** (30) | 30/99 | 22 / 24 | 0.38 / 2.09 |
| D12_I405_N | – | N | 89 | 1.000 | 0.014 (69) | 68/88 | 14 / 16 | 0.33 / 7.77 |
| D12_I405_S | – | S | 44 | 1.000 | **1.000** (10) | 10/43 | 14 / 11 | 0.31 / 7.59 |

Every panel also has:
- 1 station per link;
- no milepost ties;
- links.csv v_f equal to fd_parameters v_f.

## Findings

1. **Order.** `order_index` equals milepost rank in every panel, and our Task 1 milepost sort equals the scorer's order. There are no wrong link orders, reversed segments or station/link mismatches. **Task 1 and Task 3 ordering are correct.**
2. **Direction (NEW, matters for Task 2).**
   - In E/N panels, traffic flows toward **increasing** milepost / order_index.
   - In W/S panels it flows toward **decreasing** order_index.
   - Task 2 pools all 8 panels in one model and names its features by array direction:
     - `vr_min_dn*` / `vr_min_up*`;
     - `vr_n±k`, `q_n±k`, `fr_n±k`;
     - `dq_hi*` / `dq_lo*`;
     - `cap_ratio_hi/lo`.
   - So "downstream" means physically downstream in 4 panels and **upstream** in the other 4.
   - The prior-free ongoing model has no `pos`, lanes or capacity features that could let it tell the two cases apart.
   - Queues spill back upstream, so these are exactly the features ongoing relies on.
   - **Fix:** reverse the link order for W/S panels, so index + 1 is always downstream.
   - **Test:** `tfb-oofF` (same windows, folds and seeds as oofX; P255 ongoing plus CatBoost onset) against the oofX controls.
   - Task 1 is not affected: each panel has its own model, so the direction is fixed within a model.
3. **Connectivity.**
   - In W/S panels only 10–30% of milepost-adjacent sensored links are connected through the released topology; in E/N panels it is 57–99%.
   - This matches the organizer's note that the topology is fragmented by connector links and latent mainline links. It does not affect our milepost-based features.
   - For LWR the scorer uses organizer-held flux, not our topology.
4. **Physical distance.**
   - Station spacing varies from 0.1 to 7.8 km.
   - Task 1 and Task 2 neighbour features ignore distance: `pos` is a linspace, and ±k is an index offset.
   - Candidate feature: the km gap to each neighbour (EXP_032).
5. **Ramps and lane drops.**
   - Each panel has 47–116 ramps and 9–22 lane drops, none of which our features use.
   - Ramp flows are released. For Task 2, ramp values up to and including T are allowed. For Task 1, all ramp values are allowed.
   - Candidate features: EXP_032 and EXP_033.
