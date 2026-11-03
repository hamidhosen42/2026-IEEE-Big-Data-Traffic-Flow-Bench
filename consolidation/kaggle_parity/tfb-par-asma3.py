# ======================================================================================================================
# GENERATED FILE - DO NOT EDIT. Built by consolidation/bundle.py from the tfb package
#   tfb 0.1.0-m1 | modules tree sha1 f139673c3969c7a23c7a6a83b0f6e2406466eb30 | generated 2026-11-03 02:09
# Parity kernel tfb-par-asma3 (A3 required, ~8 min on Kaggle CPU): queue assembly A3 (4 condition x split parts = the tfb-j4 rows, on the original tfb-k9 file) vs tfb-k9 submission_FDRTV.csv
# Stages ['assemble'] | pick A3 | recipe full | overrides {}
# Kernel sources to attach (5, plus the competition data):
#   hosen42/tfb-k9 hosen42/tfb-t2osp hosen42/tfb-t2occ hosen42/tfb-t2opv hosen42/tfb-t2ogpvogf
# Push: python kaggle_jobs/kjob.py push tfb-par-asma3 consolidation/kaggle_parity/tfb-par-asma3.py hosen42/tfb-k9 hosen42/tfb-t2osp hosen42/tfb-t2occ hosen42/tfb-t2opv hosen42/tfb-t2ogpvogf
# Output: PARITY lines + PARITY SUMMARY in the log; /kaggle/working/parity_<slug>.json; stage outputs in /kaggle/working/tfb_work/
# ======================================================================================================================
import glob, json, os, sys, tempfile, time

SPEC = json.loads(r'''{"slug": "tfb-par-asma3", "priority": "A3 required", "minutes": 8, "stages": ["assemble"], "pick": "A3", "recipe": "full", "overrides": {}, "desc": "queue assembly A3 (4 condition x split parts = the tfb-j4 rows, on the original tfb-k9 file) vs tfb-k9 submission_FDRTV.csv", "sources": ["tfb-k9", "tfb-t2osp", "tfb-t2occ", "tfb-t2opv", "tfb-t2ogpvogf"], "checks": [{"type": "submission", "ours": ["tfb-k9-asm", "submission_K9.csv"], "orig": ["tfb-k9", "submission_FDRTV.csv"], "tasks": ["state", "queue", "odme"], "label": "asm_A3_vs_tfb-k9"}], "tfb_version": "0.1.0-m1", "tree_sha1": "f139673c3969c7a23c7a6a83b0f6e2406466eb30", "generated": "2026-11-03 02:09"}''')

MODULES = {}
MODULES['tfb/__init__.py'] = '''"""TrafficFlowBench final-pick reproduction package (2026 IEEE Big Data Cup, team hosen42).

tfb/config.py     recipes of the two final picks (every constant of the selected submission)
tfb/runtime.py    path layer: data root, per-stage output folder, "kernel output X" lookup
tfb/runner.py     stage plan, cached intermediates, one subprocess per stage
tfb/stages/*.py   the original Kaggle kernels, vendored with only their I/O and constant lines patched (marked [TFB])
tfb/parity.py     row-level comparison against the original kernel outputs (used by the Kaggle parity kernels)
"""
__version__ = '0.1.0-m1'
'''
MODULES['tfb/config.py'] = '''"""Recipes of the two final picks: the single source of truth for every constant the entry script uses.

Every value is copied from the kernel that produced the selected submission (sources in consolidation/lineage_src/, chain
states in kaggle_jobs/auto_chain_oct8b_state.json / auto_chain_oct9_state.json, plan in consolidation/DESIGN.md).
tests/test_static.py checks them against the ast of the original kernels; do not edit a value without re-running it.

Picks (final selection 2026-11-03: A3 / B3; A2 / B2 = the previous picks, kept working)
  A3 = tfb-k9 (selected candidate A, public 0.88281): tfb-k5d (T1 mix + 2 CatBoost seeds + XGBoost member + blackout override)
       -> steps M (flow LRTC (24, 40, 10), W 0.35, ordinary rows), B (blkVQ@0.15, blackout rows), R (FD recon), T (1-D TV);
       queue rows = those of tfb-j4 (= A2's, carried through tfb-k3d -> tfb-k3 -> tfb-k5d -> tfb-k9).
  B3 = tfb-k10 (hedge B, public 0.88281): tfb-k9 + April onset rows of tfb-t2osp (base) + April ongoing rows of tfb-t2ogveto (veto10).
  A2 = tfb-j4 (previous candidate A, 0.88176): tfb-i4 + April ongoing rows of tfb-t2ogpvogf (pvogfv).
  B2 = tfb-j3 (previous hedge B, 0.88176): tfb-i4 + April onset rows of tfb-t2osp (base) + April ongoing rows of tfb-t2ogveto (veto10).
A3 / B3 share every state, validation-queue and odme row (as A2 / B2 do). Queue rows: A3 == A2, B3 == B2 (every block).
Lineage of the new picks: consolidation/LINEAGE_K9.md, lineage_K9.json.
"""

PANELS_ALL = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
              'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
PANELS_T2 = PANELS_ALL[:8]                       # D12_I405_N/S have no Task 2 windows

PICKS = ('A3', 'B3', 'A2', 'B2')
# generation of the Task 1 chain: 'j' = tfb-i4d -> tfb-i4 (A2 / B2), 'k' = tfb-k5d -> tfb-k9 (A3 / B3)
PICK_GEN = {'A2': 'j', 'B2': 'j', 'A3': 'k', 'B3': 'k'}
A_PICKS = ('A2', 'A3')                           # picks whose April queue rows are the same-scenario bet (opv pv + pvogfv)
RECIPES = ('full', 'lean1')

# ------------------------------------------------------------------ Task 1 member bags (tfb-i4d constants; DESIGN 1.2, 3.3)
OLD_FTA_FULL = ('tfb-t1v11', 'tfb-t1v12', 'tfb-t1v12s2', 'tfb-t1v12s3', 'tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7')
FD_FULL = ('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11', 'tfb-t1dark', 'tfb-t1ramp', 'tfb-t1ramp2')
DARK_MEMBER = 'tfb-t1bspec'                      # '+'-joined = mean of several members (tfb-i2d used 'tfb-t1ramp+tfb-t1ramp2')

T1_BAGS = {
    'full': dict(old_fta=OLD_FTA_FULL, fd=FD_FULL, dark=DARK_MEMBER),
    # lean-1 (DESIGN 3.2 S1 + S4 + S5): old FTA 8 -> 4 (3-4 transductive rounds), FD 6 -> 4 (drop t1dark, t1ramp2)
    'lean1': dict(old_fta=('tfb-t1v12s4', 'tfb-t1v12s5', 'tfb-t1v12s6', 'tfb-t1v12s7'),
                  fd=('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11', 'tfb-t1ramp'), dark=DARK_MEMBER),
}


# k generation (A3 / B3; tfb-k5d constants): two CatBoost seeds (mean) and one XGBoost member, each in 4 panel parts a-d.
# Every recipe keeps all three (they are the k-generation change; lean1 only thins the LightGBM bags as for A2 / B2).
K_EXTRA = dict(catb_member='tfb-t1catb+tfb-t1catb2', catb_parts='abcd', w_catb=0.35,
               cand_member='tfb-t1xgb', cand_parts='abcd', w_cand=0.35)
QUAD_PANELS = {'a': ['D7_I405_N', 'D7_I10_W'], 'b': ['D7_I210_E', 'D7_I210_W', 'D12_I405_S'],   # parts of t1catb / t1catb2 / t1xgb
               'c': ['D12_I5_N', 'D12_I5_S'], 'd': ['D7_I10_E', 'D7_I405_S', 'D12_I405_N']}


def member_parts(member):
    """kernel parts that hold one member's state_submission_FTA.csv (halves for memory; bspec has a third part; the
    CatBoost / XGBoost members of the k generation have four)."""
    if member in ('tfb-t1catb', 'tfb-t1catb2', 'tfb-t1xgb'):
        return tuple(f'{member}{p}' for p in 'abcd')
    return (f'{member}a', f'{member}b', f'{member}c') if member == 'tfb-t1bspec' else (f'{member}a', f'{member}b')


def k_members():
    return tuple(K_EXTRA['catb_member'].split('+')) + tuple(K_EXTRA['cand_member'].split('+'))


def t1_member_kernels(recipe, pick='A2'):
    bag = T1_BAGS[recipe]
    out = []
    ms = bag['old_fta'] + bag['fd'] + tuple(bag['dark'].split('+'))
    if PICK_GEN[pick] == 'k':
        ms += k_members()
    for m in ms:
        out.extend(member_parts(m))
    return list(dict.fromkeys(out))


# T1 mix + blackout override (stage t1_mix = tfb-i4d, t1_t1dark_swap.py)
T1_MIX = dict(
    w_fd=0.5,                                    # raw = W_FD * mean(FD bag) + (1 - W_FD) * mean(old FTA bag)
    w_dark=0.75,                                 # blackout rows: (1 - W_DARK) * raw + W_DARK * dark member
    out_name='T1DARK',
    # The base file only contributes the row order and the queue / path_flow rows (every state row is recomputed).
    # tfb-i4d used tfb-i2/submission_FDRTV.csv; the entry script uses the T4 skeleton (queue rows 0, filled by assemble).
    base=('tfb-final', 'submission.csv'),
)

# T1 mix of the k generation (stage t1_mix_k5d = tfb-k5d, kaggle_jobs/t1_div2_swap.py): the tfb-i4d mix, then
# mix <- 0.65 mix + 0.35 mean(t1catb, t1catb2), then mix <- 0.65 mix + 0.35 t1xgb, then the blackout override (W_DARK 0.75).
# Base: tfb-k5d read tfb-k3/submission_FDRTV.csv (queue / path_flow rows only; every state row is recomputed) -> T4 skeleton.
T1_MIX_K5D = dict(T1_MIX, out_name='K5D', **K_EXTRA)

# FD-gated reconciliation + 1-D TV (stage t1_recon = tfb-i4, swap_fdrecon.py; chain state "recon")
RECON_TV = dict(
    do_recon=True, do_tv=True,
    gate=0.6, a=0.75, logr_max=0.5,
    tv_free=0.0075, tv_free_a=0.001, tv_gate=0.02, tv_gate_a=0.005, tv_iters=300,
    run_gap_ns=5 * 60 * 10 ** 9,                 # a TV run breaks where consecutive target timestamps are > 5 min apart
    w_range=(8.0, 40.0), kjl_range=(60.0, 200.0),
    excl_med=0.02, excl_iqr=0.05, excl_min_n=200,
    base=('tfb-i4d', 'submission_T1DARK.csv'),
)

# steps M + B + R + T of tfb-k9 (stage t1_lrtc = kaggle_jobs/swap_lrtc_k9.py); R and T use the RECON_TV constants (verbatim
# copies of the tfb-i4 / tfb-k5 code). Writes submission_LRTC.csv (M + B) and submission_FDRTV.csv (the candidate).
LRTC_K9 = dict(
    w_lrtc=0.35, lrtc_ranks=(24, 40, 10), lrtc_iters=60,   # step M: flow LRTC on ordinary state rows, q <- max(0.65 q + 0.35 q_lrtc, 20)
    w_blk=0.15, blk_v_ranks=(8, 20, 20), blk_q_ranks=(8, 40, 10),   # step B: blackout rows, both channels
    lrtc_min_obs=200,
    base=('tfb-k5d', 'submission_K5D.csv'),
)

# ------------------------------------------------------------------ Task 2 onset (stages onset_sp = tfb-t2osp, onset_pv = tfb-t2opv)
# Literal constants stay inside the vendored kernels; listed here for the report and checked by tests/test_static.py.
ONSET = dict(
    catboost=dict(iterations=600, depth=8, learning_rate=0.08, thread_count=4, random_seed=0),
    subsample_rows=1_200_000, subsample_rng=0, max_og=3000, spacing=72,
    ks_sp=(2, 4, 8),                             # tfb-t2osp variants sp2 / sp4 / sp8
    k_blend=4, w_march=(10, 20, 40),             # tfb-t2opv pv bag
)

# ------------------------------------------------------------------ Task 2 ongoing (milestone 2)
ONGOING = dict(
    # tfb-t2ogpvogf full run (2026-10-09, 16,399 s): "private ongoing windows vetoed 0/40" -> pvogfv == pvogf on every
    # private row, so the pvo veto reference (tfb-t2ogpvaf, 2.4 Kaggle-h) is not needed for A2 (DESIGN S11).
    need_pvo=False,
    veto_g=10,
)

# ------------------------------------------------------------------ Task 4 (stage t4 = tfb-final cell 7)
T4 = dict(lam=0.05)

# ------------------------------------------------------------------ LightGBM reproducibility (Task 2 ongoing stages)
# The original kernels ran LightGBM with num_threads=4, verbose=-1 and neither `deterministic` nor `force_row_wise/col_wise`:
# LightGBM then picks row- or col-wise histograms by a timing test and sums multithreaded histograms in no fixed order, so
# a rerun is not bit-repeatable (Kaggle parity 2026-10-10: occ 22 / pvogf 24 queue cells of 87,000 changed). The entry
# script fixes the code path so that every rerun on the same hardware / version / threads gives the same file. It cannot
# reproduce the original file exactly (its own path was never logged). None = original behaviour. Task 1 members keep the
# original setting (their effect is smooth; deterministic mode would add runtime to the 60 h of member training).
LGB_REPRO = dict(deterministic=True, force_row_wise=True)

# ------------------------------------------------------------------ queue assembly (stage assemble = swap_queue2.py)
# One part per (condition, split) block; together they cover every queue row (the T4 skeleton's queue rows are 0).
# Provenance of each block in the Kaggle chain:
#   validation onset  <- tfb-t2osp sp4      tfb-e5 (queue_onset, all splits); private onset later replaced by g4 / j3
#   validation ongoing<- tfb-t2occ fs4occ   tfb-g3
#   private onset     <- tfb-t2opv pv       tfb-g4 (A2)   | tfb-t2osp base     tfb-j3 (B2)
#   private ongoing   <- tfb-t2ogpvogf pvogfv tfb-j4 (A2) | tfb-t2ogveto veto10 tfb-j3 (B2)
# 'a|b' = first file that exists (tfb-j4 planner fallback: pvogf when the veto file is missing; identical rows, 0 vetoes).
_VAL_PARTS = [('tfb-t2osp', 'queue_submission_sp4.csv', 'queue_onset', 'validation'),
              ('tfb-t2occ', 'queue_submission_fs4occ.csv', 'queue_ongoing', 'validation')]
QUEUE_PARTS = {
    'A2': _VAL_PARTS + [('tfb-t2opv', 'queue_submission_pv.csv', 'queue_onset', 'private'),
                        ('tfb-t2ogpvogf', 'queue_submission_pvogfv.csv|queue_submission_pvogf.csv', 'queue_ongoing', 'private')],
    'B2': _VAL_PARTS + [('tfb-t2osp', 'queue_submission_base.csv', 'queue_onset', 'private'),
                        ('tfb-t2ogveto', 'queue_submission_veto10.csv', 'queue_ongoing', 'private')],
}
# the original swap kernels (only the private parts; validation rows were already in tfb-i4) - for tests / parity
QUEUE_PARTS_ORIGINAL = {
    'A2': [('tfb-t2ogpvogf', 'queue_submission_pvogfv.csv', 'queue_ongoing', 'private')],
    'B2': [('tfb-t2osp', 'queue_submission_base.csv', 'queue_onset', 'private'),
           ('tfb-t2ogveto', 'queue_submission_veto10.csv', 'queue_ongoing', 'private')],
}
# A3 / B3: the queue rows of tfb-k9 are tfb-j4's (bases j4 -> k3d -> k3 -> k5d -> k9; no kernel of that chain touched a
# queue row), and tfb-k10 = tfb-k9 + the private parts of tfb-j3 / tfb-k8. So every block equals A2 / B2.
QUEUE_PARTS['A3'] = list(QUEUE_PARTS['A2'])
QUEUE_PARTS['B3'] = list(QUEUE_PARTS['B2'])
QUEUE_PARTS_ORIGINAL['A3'] = []                  # tfb-k9 has no assembly step (queue rows inherited from tfb-j4)
QUEUE_PARTS_ORIGINAL['B3'] = list(QUEUE_PARTS_ORIGINAL['B2'])   # = tfb-k10 PARTS
# tfb-k9's own submission_FDRTV.csv is the A3 file; its assembly is a stage of the entry script only, so it writes into a
# virtual kernel folder 'tfb-k9-asm' (parity reference: tfb-k9/submission_FDRTV.csv).
PICK_KERNEL = {'A2': ('tfb-j4', 'J4'), 'B2': ('tfb-j3', 'J3'), 'A3': ('tfb-k9-asm', 'K9'), 'B3': ('tfb-k10', 'K10')}
PICK_REFERENCE = {'A2': ('tfb-j4', 'submission_J4.csv'), 'B2': ('tfb-j3', 'submission_J3.csv'),
                  'A3': ('tfb-k9', 'submission_FDRTV.csv'), 'B3': ('tfb-k10', 'submission_K10.csv')}
ASSEMBLE = dict(base=('tfb-i4', 'submission_FDRTV.csv'), require_full_cover=True)
ASSEMBLE_BASE = {'A2': ('tfb-i4', 'submission_FDRTV.csv'), 'B2': ('tfb-i4', 'submission_FDRTV.csv'),
                 'A3': ('tfb-k9', 'submission_FDRTV.csv'), 'B3': ('tfb-k9', 'submission_FDRTV.csv')}
# A2 / B2 use the tfb-j3 code (stage module assemble), A3 / B3 the tfb-k10 code (assemble_k10 = the same swap plus 4 gates)
ASSEMBLE_MODULE = {'A2': 'assemble', 'B2': 'assemble', 'A3': 'assemble_k10', 'B3': 'assemble_k10'}

# ------------------------------------------------------------------ Task 1 members (DESIGN 1.3)
# Common LightGBM: 511 leaves, min_data 100, lr 0.04, 1500 trees, max_bin 63, ff 0.8, bf 0.8, num_threads 4, MAX_ROWS 600k.
# family -> template module tfb/stages/t1_<family>.py (rendered with a part's constants it equals that part's original source;
# tests/test_templates.py checks all 31 parts). tx_cap = cap of the transductive rows in units of MAX_ROWS.
T1_MEMBERS = {
    'tfb-t1v11':    dict(family='fta_v11', row_rng=0,  rounds=2, rng_base=1000,  tx_cap=4, holdout=0.15,  lgb_seeds=None,         image='A'),
    'tfb-t1v12':    dict(family='fta',   row_rng=1,  rounds=2, rng_base=2000,  tx_cap=4, holdout=0.15,  lgb_seeds=(1, 11, 21),  image='A'),
    'tfb-t1v12s2':  dict(family='fta',   row_rng=2,  rounds=2, rng_base=3000,  tx_cap=4, holdout=0.15,  lgb_seeds=(2, 12, 22),  image='A'),
    'tfb-t1v12s3':  dict(family='fta',   row_rng=3,  rounds=2, rng_base=4000,  tx_cap=4, holdout=0.15,  lgb_seeds=(3, 13, 23),  image='A'),
    'tfb-t1v12s4':  dict(family='fta',   row_rng=4,  rounds=3, rng_base=5000,  tx_cap=6, holdout=0.15,  lgb_seeds=(4, 14, 24),  image='A'),
    'tfb-t1v12s5':  dict(family='fta',   row_rng=5,  rounds=3, rng_base=6000,  tx_cap=6, holdout=0.15,  lgb_seeds=(5, 15, 25),  image='A'),
    'tfb-t1v12s6':  dict(family='fta',   row_rng=6,  rounds=3, rng_base=7000,  tx_cap=6, holdout=0.001, lgb_seeds=(6, 16, 26),  image='A'),
    'tfb-t1v12s7':  dict(family='fta',   row_rng=7,  rounds=4, rng_base=8000,  tx_cap=8, holdout=0.15,  lgb_seeds=(7, 17, 27),  image='A'),
    'tfb-t1v13':    dict(family='fd',    row_rng=9,  rounds=3, rng_base=10000, tx_cap=6, holdout=0.15,  lgb_seeds=(9, 19, 29),  image='A'),
    'tfb-t1v13s10': dict(family='fd',    row_rng=10, rounds=3, rng_base=11000, tx_cap=6, holdout=0.15,  lgb_seeds=(10, 20, 30), image='B'),
    'tfb-t1v13s11': dict(family='fd',    row_rng=11, rounds=3, rng_base=12000, tx_cap=6, holdout=0.15,  lgb_seeds=(11, 21, 31), image='B'),
    'tfb-t1dark':   dict(family='dark',  row_rng=13, rounds=3, rng_base=14000, tx_cap=6, holdout=0.15,  lgb_seeds=(13, 23, 33), image='B', dark_seed=1300),
    'tfb-t1ramp':   dict(family='ramp',  row_rng=13, rounds=3, rng_base=14000, tx_cap=6, holdout=0.15,  lgb_seeds=(13, 23, 33), image='B', dark_seed=1300),
    'tfb-t1ramp2':  dict(family='ramp',  row_rng=17, rounds=3, rng_base=18000, tx_cap=6, holdout=0.15,  lgb_seeds=(17, 27, 37), image='B', dark_seed=1700),
    'tfb-t1bspec':  dict(family='bspec', row_rng=13, rounds=3, rng_base=14000, tx_cap=6, holdout=0.15,  lgb_seeds=(13, 23, 33), image='B', dark_seed=1300,
                         panel_overrides={'D7_I405_N': dict(max_rows=420_000, train_day_mod5_keep=(0, 1, 3))}),
    # k generation (A3 / B3): v13 rows (row rng / transductive rng / holdout), model swapped; 4 parts (QUAD_PANELS).
    # CatBoost thread_count 4 / XGBoost nthread 4 stay literal in the templates (--threads does not apply; no LightGBM is trained:
    # the pF line is dead code kept from tfb-t1v13a). image: not traced (Kaggle default image of 2026-10-10 / 10-16).
    'tfb-t1catb':   dict(family='catb',  row_rng=9,  rounds=3, rng_base=10000, tx_cap=6, holdout=0.15,  lgb_seeds=(9, 19, 29),  image=None,
                         cb_seed=21, cb_random_seed=20261005),
    'tfb-t1catb2':  dict(family='catb',  row_rng=19, rounds=3, rng_base=12000, tx_cap=6, holdout=0.15,  lgb_seeds=(19, 39, 49), image=None,
                         cb_seed=31, cb_random_seed=20261016),
    'tfb-t1xgb':    dict(family='xgb',   row_rng=9,  rounds=3, rng_base=10000, tx_cap=6, holdout=0.15,  lgb_seeds=(9, 19, 29),  image=None),
}
DARK_N = 48
# Kaggle runtime (s) and peak RSS (GB) of every original part (lineage_A2.json), for scheduling the parity kernels
T1_PART_RUNTIME_S = {
    'tfb-t1v11a': 20200, 'tfb-t1v11b': 14196, 'tfb-t1v12a': 11811, 'tfb-t1v12b': 8656, 'tfb-t1v12s2a': 11009, 'tfb-t1v12s2b': 5380,
    'tfb-t1v12s3a': 11126, 'tfb-t1v12s3b': 8361, 'tfb-t1v12s4a': 12545, 'tfb-t1v12s4b': 8874, 'tfb-t1v12s5a': 9577, 'tfb-t1v12s5b': 9085,
    'tfb-t1v12s6a': 12930, 'tfb-t1v12s6b': 8746, 'tfb-t1v12s7a': 13794, 'tfb-t1v12s7b': 9666, 'tfb-t1v13a': 14046, 'tfb-t1v13b': 10716,
    'tfb-t1v13s10a': 14041, 'tfb-t1v13s10b': 7293, 'tfb-t1v13s11a': 15267, 'tfb-t1v13s11b': 7690, 'tfb-t1darka': 15334, 'tfb-t1darkb': 11850,
    'tfb-t1rampa': 16717, 'tfb-t1rampb': 12194, 'tfb-t1ramp2a': 16249, 'tfb-t1ramp2b': 11655, 'tfb-t1bspeca': 14035, 'tfb-t1bspecb': 13266,
    'tfb-t1bspecc': 4541,
    # k generation (last log timestamp of each part, kaggle_jobs/k1_followup.log, k3_followup.log, k5_followup.log); peak RSS
    # catb / catb2 / xgb parts a 26.1 / b 22.2 / c 19.3 / d 10.9 GB
    'tfb-t1catba': 12692, 'tfb-t1catbb': 14811, 'tfb-t1catbc': 11556, 'tfb-t1catbd': 17471,
    'tfb-t1catb2a': 12226, 'tfb-t1catb2b': 15019, 'tfb-t1catb2c': 12549, 'tfb-t1catb2d': 17180,
    'tfb-t1xgba': 3502, 'tfb-t1xgbb': 4317, 'tfb-t1xgbc': 2953, 'tfb-t1xgbd': 4459,
}


def t1_part_panels(member, part):
    if T1_MEMBERS[member]['family'] in ('catb', 'xgb'):
        return list(QUAD_PANELS[part])
    if T1_MEMBERS[member]['family'] == 'bspec':
        return {'a': PANELS_ALL[:4], 'b': PANELS_ALL[5:], 'c': ['D7_I405_N']}[part]
    return {'a': PANELS_ALL[:5], 'b': PANELS_ALL[5:]}[part]


def t1_part_cfg(kernel, threads=4):
    """constants of one member part (kernel slug, e.g. 'tfb-t1v12s4a') for its family template."""
    member, part = kernel[:-1], kernel[-1]
    m = T1_MEMBERS[member]
    panels = t1_part_panels(member, part)
    ov = {}
    for p in panels:
        ov.update(m.get('panel_overrides', {}).get(p, {}) if len(panels) == 1 else {})
    return dict(member=member, part=part, family=m['family'], panels=list(panels), rounds=m['rounds'], row_rng=m['row_rng'],
                rng_base=m['rng_base'], tx_cap=m['tx_cap'], holdout=m['holdout'], lgb_seeds=m['lgb_seeds'],
                dark_n=DARK_N, dark_seed=m.get('dark_seed'), max_rows=ov.get('max_rows', 600_000),
                train_day_mod5_keep=ov.get('train_day_mod5_keep'), threads=int(threads),
                **{k: m[k] for k in ('cb_seed', 'cb_random_seed') if k in m})


PSEUDO_EXPECTED = {   # tfb-t2lab-full outputs: (windows, onset, ongoing, sha1(json([[d, t, cond], ...]))) - counts = lineage_A2.json
    'D7_I10_E': (494, 265, 229, '15de06a99c8ddd3e087d05a7bc7917149aaa6e33'),
    'D7_I10_W': (138, 134, 4, 'c60153bacc2b908a332681f1e0681acefe23af7a'),
    'D7_I210_E': (715, 276, 439, '948787a55c488812f53387c6bc4edcb789ed394a'),
    'D7_I210_W': (697, 310, 387, '1d86321888dc6a1f723899303545cbad5e8923d8'),
    'D7_I405_N': (728, 353, 375, '74aac66ff5287a6035a30c102076a8f90e3810fa'),
    'D7_I405_S': (492, 271, 221, '04ea5e775fc2a6cf0e780f382f0b6bb10bccab5f'),
    'D12_I5_N': (449, 255, 194, '021a82439570d618ea2000f193708c2607a23d43'),
    'D12_I5_S': (474, 264, 210, '735269728e36cf59990ad07f2700c65076ef5eb8'),
}

# ------------------------------------------------------------------ stages
# kernel = the original Kaggle kernel whose outputs the stage reproduces; its outputs live in <work>/kernels/<kernel>/.
# milestone 2 stages are registered so that the plan, the dependency checks and --seed-kernel-outputs already work.
STAGES = {
    't4':          dict(module='t4_nnls',   kernel='tfb-final', outputs=['submission.csv', 'odme_path_flow.csv'], m=1),
    'onset_sp':    dict(module='onset_sp',  kernel='tfb-t2osp', outputs=['queue_submission_base.csv', 'queue_submission_sp2.csv',
                                                                         'queue_submission_sp4.csv', 'queue_submission_sp8.csv'], m=1),
    'onset_pv':    dict(module='onset_pv',  kernel='tfb-t2opv', outputs=['queue_submission_base.csv', 'queue_submission_sp4.csv',
                                                                         'queue_submission_pv.csv', 'queue_submission_pvb.csv'], m=1),
    't1_mix':      dict(module='t1_mix',    kernel='tfb-i4d',   outputs=['submission_T1DARK.csv'], m=1),
    't1_recon':    dict(module='t1_recon',  kernel='tfb-i4',    outputs=['submission_FDR.csv', 'submission_FDRTV.csv', 'submission_TV.csv'], m=1),
    't1_mix_k5d':  dict(module='t1_mix_k5d', kernel='tfb-k5d', outputs=['submission_K5D.csv'], m=1),                    # A3 / B3
    't1_lrtc':     dict(module='t1_lrtc',   kernel='tfb-k9',    outputs=['submission_LRTC.csv', 'submission_FDRTV.csv'], m=1),  # A3 / B3
    'assemble':    dict(module='assemble',  kernel=None,        outputs=None, m=1),          # kernel / output from PICK_KERNEL, module ASSEMBLE_MODULE
    # ---- milestone 2
    'pseudo':        dict(module='pseudo', kernel='tfb-t2lab-full', outputs=[f'pseudo_{p}.npz' for p in PANELS_T2], m=2),
    't1_members':    dict(module=None, kernel=None, outputs=None, m=2),          # expands to one 't1m:<part kernel>' stage per part
    'ongoing_val':   dict(module='t2_occ', kernel='tfb-t2occ',
                          outputs=['queue_submission_fs4occ.csv', 'queue_submission_fs2occ.csv', 'queue_submission_fs_np.csv', 't2v26occ_probs.npz'], m=2),
    'ongoing_pvaf':  dict(module=None, kernel='tfb-t2ogpvaf', outputs=['queue_submission_pvo.csv'], m=2),   # not needed (0/40 vetoes); not ported
    'ongoing_pvogf': dict(module='t2_pvogf', kernel='tfb-t2ogpvogf', outputs=['queue_submission_pvogf.csv', 'queue_submission_pvogf_npid.csv'], m=2),
    'ogfeat_a':      dict(module='t2_ogfeat', kernel='tfb-t2ogfa', outputs=['queue_submission_fs2ogf.csv', 't2v27a_probs.npz'], m=2),
    'ogfeat_b':      dict(module='t2_ogfeat', kernel='tfb-t2ogfb', outputs=['queue_submission_fs2ogf1.csv', 't2v27b_probs.npz'], m=2),
    'ogfeat_comb':   dict(module='t2_ogfc', kernel='tfb-t2ogfc', outputs=['queue_submission_fs4ogf.csv'], m=2),
    'ogveto':        dict(module='t2_ogveto', kernel='tfb-t2ogveto', outputs=['queue_submission_veto10.csv', 't2ogveto_windows.csv'], m=2),
}
T1_PART_OUTPUTS = ['state_submission_FTA.csv', 'state_submission.csv', 't1_valholdout.csv']
# production references of the ongoing kernels (the originals read these files from other kernels):
#   pvogf copies its VALIDATION rows from REF_V22 (tfb-t2v22af fs4af in the original; the final file never uses them, so the
#   entry script points it at fs4occ). The pvo veto reference is optional (need_pvo False): without it no pvogfv is written
#   and assemble uses pvogf (identical private rows: 0/40 windows were vetoed).
ONGOING_REFS = dict(ref_v22=('tfb-t2occ', 'queue_submission_fs4occ.csv'))
LIGHT_STAGES = ['t4', 'onset_sp', 'onset_pv', 't1_mix', 't1_recon', 't1_mix_k5d', 't1_lrtc', 'assemble']


def plan(pick, recipe):
    """stage order of a full run (milestone 2 stages included). t4 precedes t1_mix: it writes the base skeleton."""
    t2 = ['onset_sp'] + (['onset_pv'] if pick in A_PICKS else []) + ['ongoing_val']
    if pick in A_PICKS:
        t2 += (['ongoing_pvaf'] if ONGOING['need_pvo'] else []) + ['ongoing_pvogf']
    else:
        t2 += ['ogfeat_a', 'ogfeat_b', 'ogfeat_comb', 'ogveto']
    t1 = ['t1_mix', 't1_recon'] if PICK_GEN[pick] == 'j' else ['t1_mix_k5d', 't1_lrtc']
    return ['pseudo', 't1_members', 't4'] + t2 + t1 + ['assemble']   # 't1_members' -> expand()


def is_member_stage(stage):
    return stage.startswith('t1m:')


def stage_module(stage, pick='A2'):
    if is_member_stage(stage):
        return 't1_' + T1_MEMBERS[stage[4:-1]]['family']
    if stage == 'assemble':
        return ASSEMBLE_MODULE[pick]
    return STAGES[stage]['module']


def stage_cfg(stage, pick, recipe, threads=4):
    """the constants a stage file reads through TFB.cfg (JSON-serialisable)."""
    if is_member_stage(stage):
        return t1_part_cfg(stage[4:], threads)
    if stage == 't1_mix':
        bag = T1_BAGS[recipe]
        return dict(T1_MIX, members=list(bag['old_fta']), fd_members=list(bag['fd']), dark_member=bag['dark'], recipe=recipe)
    if stage == 't1_recon':
        return dict(RECON_TV)
    if stage == 't1_mix_k5d':
        bag = T1_BAGS[recipe]
        return dict(T1_MIX_K5D, members=list(bag['old_fta']), fd_members=list(bag['fd']), dark_member=bag['dark'], recipe=recipe)
    if stage == 't1_lrtc':
        return dict(RECON_TV, **LRTC_K9)                # base = LRTC_K9['base'] (tfb-k5d)
    if stage == 'assemble':
        kern, name = PICK_KERNEL[pick]
        cfg = dict(ASSEMBLE, base=ASSEMBLE_BASE[pick], parts=[list(p) for p in QUEUE_PARTS[pick]], out_name=name, pick=pick)
        if ASSEMBLE_MODULE[pick] == 'assemble_k10':
            cfg['march_from_parts'] = True          # the base has no March queue rows (T4 skeleton); validation parts fill them
        return cfg
    if stage == 't4':
        return dict(T4)
    if stage in ('onset_sp', 'onset_pv'):
        return dict(ONSET)
    if stage == 'pseudo':
        return dict(expected={k: list(v) for k, v in PSEUDO_EXPECTED.items()})
    if stage == 'ongoing_val':
        return dict(lgb_repro=dict(LGB_REPRO))
    if stage == 'ongoing_pvogf':
        return dict(ref_v22=list(ONGOING_REFS['ref_v22']), veto_g=ONGOING['veto_g'], lgb_repro=dict(LGB_REPRO))
    if stage == 'ogfeat_a':
        return dict(part='a', tag='v27a', lgb_repro=dict(LGB_REPRO))
    if stage == 'ogfeat_b':
        return dict(part='b', tag='v27b', lgb_repro=dict(LGB_REPRO))
    return {}


def stage_kernel(stage, pick):
    if is_member_stage(stage):
        return stage[4:]
    if stage == 'assemble':
        return PICK_KERNEL[pick][0]
    return STAGES[stage]['kernel']


def stage_outputs(stage, pick, smoke=False):
    if is_member_stage(stage):
        return list(T1_PART_OUTPUTS)
    if stage == 'pseudo' and smoke:                  # the kernel's SMOKE switch: PANELS[:2]
        return [f'pseudo_{p}.npz' for p in PANELS_T2[:2]]
    if stage == 'assemble':
        return [f'submission_{PICK_KERNEL[pick][1]}.csv']
    return list(STAGES[stage]['outputs'])


T1_SMOKE_PANELS = ['D12_I405_S', 'D7_I10_W']      # the member kernels' own SMOKE panel override (fta: both, other families: D7_I10_W)


def stage_inputs(stage, pick, recipe, cfg=None, smoke=False):
    """(kernel, file) pairs a stage reads besides the competition data (optional CONTROL inputs are not listed)."""
    cfg = cfg if cfg is not None else stage_cfg(stage, pick, recipe)
    if is_member_stage(stage):
        return [('tfb-t2lab-full', f'pseudo_{p}.npz') for p in (T1_SMOKE_PANELS if smoke else cfg['panels']) if p in PANELS_T2]
    if stage == 't1_mix':
        ins = [tuple(cfg['base'])]
        for m in cfg['members'] + cfg['fd_members']:
            ins += [(k, 'state_submission_FTA.csv') for k in member_parts(m)[:2]]
        for m in cfg['dark_member'].split('+'):
            ins += [(k, 'state_submission_FTA.csv') for k in member_parts(m)[:2]]   # part c is optional (as in the kernel)
        return ins
    if stage == 't1_mix_k5d':
        ins = [tuple(cfg['base'])]
        for m in cfg['members'] + cfg['fd_members']:
            ins += [(k, 'state_submission_FTA.csv') for k in member_parts(m)[:2]]
        for key_m, key_p in (('catb_member', 'catb_parts'), ('cand_member', 'cand_parts')):
            for m in cfg[key_m].split('+'):
                ins += [(f'{m}{p}', 'state_submission_FTA.csv') for p in cfg[key_p]]
        for m in cfg['dark_member'].split('+'):
            ins += [(k, 'state_submission_FTA.csv') for k in member_parts(m)[:2]]
        return ins
    if stage in ('t1_recon', 't1_lrtc'):
        return [tuple(cfg['base'])]
    if stage == 'assemble':
        return [tuple(cfg['base'])] + [(p[0], p[1]) for p in cfg['parts']]
    if stage == 'ongoing_pvogf':
        return [tuple(cfg['ref_v22'])]
    if stage == 'ogfeat_comb':
        return [('tfb-t2ogfa', 't2v27a_probs.npz'), ('tfb-t2ogfb', 't2v27b_probs.npz'),
                ('tfb-t2ogfa', 'queue_submission_fs2ogf.csv'), ('tfb-t2ogfb', 'queue_submission_fs2ogf1.csv')]
    if stage == 'ogveto':
        return [('tfb-t2occ', 'queue_submission_fs4occ.csv'), ('tfb-t2ogfc', 'queue_submission_fs4ogf.csv'), ('tfb-t2occ', 't2v26occ_probs.npz')]
    return []


def expand(stages, recipe, pick='A2'):
    """replace 't1_members' by the member-part stages of the recipe (and pick generation)."""
    out = []
    for st in stages:
        out += [f't1m:{k}' for k in t1_member_kernels(recipe, pick)] if st == 't1_members' else [st]
    return out
'''
MODULES['tfb/runtime.py'] = '''"""Path layer shared by every stage: competition data root, the stage's output folder and "kernel output X" lookup.

The vendored kernels read upstream results with  glob('/kaggle/input/**/<kernel>/**/<file>', recursive=True)[0]  and write
to /kaggle/working.  Here those become  TFB.kfile(<kernel>, <file>)  and  TFB.out:
  - every stage writes into <work>/kernels/<kernel it reproduces>/ (e.g. t1_recon -> <work>/kernels/tfb-i4/), and
  - kfile() looks for <work>/kernels/<kernel>/<file> first, then in the extra input roots (parity kernels on Kaggle add
    /kaggle/input in 'glob' mode, i.e. exactly the original lookup; local development can add a folder of downloaded outputs).
A file name 'a.csv|b.csv' means: the first alternative that exists; a kernel name 'orig:<kernel>' skips the work dir.
The stage subprocess gets its context as JSON (tfb/stage_exec.py sets `ctx` before the stage module runs).
"""
import glob
import json
import os


class Ctx:
    """attributes: rel, out, work_dir, kernels_dir, input_roots [{root, mode: flat|glob}], stage, kernel, pick, recipe,
    smoke, threads, cfg (stage constants from tfb/config.py, after overrides)."""

    def __init__(self, d):
        self.__dict__.update(d)
        self.cfg = d.get('cfg') or {}
        self.input_roots = d.get('input_roots') or []

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def _candidates(self, kernel, fname):
        if kernel.startswith('orig:'):                 # parity overrides: the original kernel output, never our rerun
            kernel = kernel[5:]
        else:
            yield 'work', os.path.join(self.kernels_dir, kernel, fname)
        for r in self.input_roots:
            if r.get('mode', 'flat') == 'flat':
                yield 'flat', os.path.join(r['root'], kernel, fname)
            else:
                yield 'glob', f"{r['root']}/**/{kernel}/**/{fname}"

    def kglob(self, kernel, fname):
        """all matches of the first alternative / root that has one (sorted for glob roots); [] if none."""
        for alt in str(fname).split('|'):
            for kind, p in self._candidates(kernel, alt):
                if kind == 'glob':
                    hits = sorted(glob.glob(p, recursive=True))
                    if hits:
                        return hits
                elif os.path.exists(p):
                    return [p]
        return []

    def kfile(self, kernel, fname):
        hits = self.kglob(kernel, fname)
        if not hits:
            where = [p for _, p in self._candidates(kernel, str(fname).split('|')[0])]
            raise FileNotFoundError(f'kernel output {kernel}/{fname} not found (looked in {where}); run the stage that '
                                    f'produces it first or seed it with --seed-kernel-outputs')
        return hits[0]

    def kexists(self, kernel, fname):
        return bool(self.kglob(kernel, fname))


ctx = None          # set by tfb.stage_exec before the stage module is executed


def find_kernel_output(kernels_dir, input_roots, kernel, fname):
    """kfile() outside a stage (runner dependency checks, parity comparisons)."""
    return Ctx(dict(kernels_dir=kernels_dir, input_roots=input_roots)).kglob(kernel, fname)
'''
MODULES['tfb/runner.py'] = '''"""Stage orchestration: plan, dependency checks, cached intermediates, one subprocess per stage.

Every stage writes <work>/kernels/<kernel>/... (the files the original Kaggle kernel wrote) plus .done.json. The marker
holds a fingerprint of the stage code, its constants, the smoke flag and the size / mtime of every input file; a rerun
skips a stage whose fingerprint is unchanged, and reruns everything downstream of a changed input. A stage that crashes
leaves no marker, so its partial files are never trusted.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from . import config as C
from . import envinfo
from .runtime import find_kernel_output

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the folder that holds tfb/ and run_all.py
STAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stages')


class StageError(RuntimeError):
    pass


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _sig(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, int(st.st_mtime_ns)]


class Runner:
    def __init__(self, data_root, work_dir, pick='A2', recipe='lean1', threads=4, env_threads=4, smoke=False,
                 input_roots=(), overrides=None, force=False, log=print):
        assert pick in C.PICKS and recipe in C.RECIPES, (pick, recipe)
        self.data_root = os.path.abspath(data_root)
        self.work_dir = os.path.abspath(work_dir)
        self.kernels_dir = os.path.join(self.work_dir, 'kernels')
        self.pick, self.recipe, self.threads, self.env_threads, self.smoke = pick, recipe, int(threads), int(env_threads), bool(smoke)
        self.input_roots = [dict(r) for r in input_roots]
        self.overrides = overrides or {}
        self.force = force
        self.log = log
        os.makedirs(self.kernels_dir, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def resolve(self, kernel, fname):
        return find_kernel_output(self.kernels_dir, self.input_roots, kernel, fname)

    def cfg(self, stage):
        cfg = C.stage_cfg(stage, self.pick, self.recipe, self.threads)
        cfg.update(self.overrides.get(stage, {}))
        return cfg

    def write_manifest(self):
        d = envinfo.collect(self.data_root)
        d.update(pick=self.pick, recipe=self.recipe, threads=self.threads, env_threads=self.env_threads, smoke=self.smoke,
                 time=time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(os.path.join(self.work_dir, 'manifest.json'), 'w') as f:
            json.dump(d, f, indent=1)
        envinfo.print_env(d)
        if not os.path.exists(os.path.join(self.data_root, 'submission_key.csv')):
            self.log(f'WARNING: {self.data_root}/submission_key.csv not found - is --data-root the kaggle_public folder?')
        return d

    def seed(self, src_dir):
        """development only: copy (hard-link when possible) downloaded kernel outputs <src_dir>/<kernel>/... into the work dir."""
        n = 0
        for kernel in sorted(os.listdir(src_dir)):
            s = os.path.join(src_dir, kernel)
            d = os.path.join(self.kernels_dir, kernel)
            if not os.path.isdir(s) or os.path.exists(d):
                continue
            for dp, _, fs in os.walk(s):
                rel = os.path.relpath(dp, s)
                os.makedirs(os.path.join(d, rel), exist_ok=True)
                for f in fs:
                    a, b = os.path.join(dp, f), os.path.join(d, rel, f)
                    try:
                        os.link(a, b)
                    except OSError:
                        shutil.copy2(a, b)
            with open(os.path.join(d, '.seeded'), 'w') as f:
                f.write(os.path.abspath(s) + '\\n')
            n += 1
        self.log(f'[tfb] seeded {n} kernel output folders from {src_dir}')

    # ------------------------------------------------------------------ one stage
    def run_stage(self, stage):
        module = C.stage_module(stage, self.pick)
        kernel = C.stage_kernel(stage, self.pick)
        outs = C.stage_outputs(stage, self.pick, smoke=self.smoke)
        if module is None:                                           # not ported (optional stage): accept provided outputs only
            miss = [(kernel, o) for o in outs if not self.resolve(kernel, o)]
            if miss:
                raise StageError(f'stage {stage} is not ported and its outputs are missing: {miss} - provide them with '
                                 f'--seed-kernel-outputs / --input-root')
            self.log(f'[tfb] {stage:12s} provided (not ported; outputs found)')
            return 'provided'
        cfg = self.cfg(stage)
        inputs = C.stage_inputs(stage, self.pick, self.recipe, cfg, smoke=self.smoke)
        miss = [(k, f) for k, f in inputs if not self.resolve(k, f)]
        if miss:
            raise StageError(f'stage {stage}: missing inputs {miss[:8]}{" ..." if len(miss) > 8 else ""}')
        in_paths = [self.resolve(k, f)[0] for k, f in inputs]
        out_dir = os.path.join(self.kernels_dir, kernel)
        os.makedirs(out_dir, exist_ok=True)
        src = os.path.join(STAGE_DIR, module + '.py')
        fp_obj = dict(stage=stage, code=_sha1_file(src), runtime=_sha1_file(os.path.join(os.path.dirname(STAGE_DIR), 'runtime.py')),
                      cfg=cfg, smoke=self.smoke, inputs=[_sig(p) for p in in_paths], data=os.path.abspath(self.data_root),
                      threads=self.threads if C.is_member_stage(stage) else None)
        fp = hashlib.sha1(json.dumps(fp_obj, sort_keys=True, default=str).encode()).hexdigest()
        done_p = os.path.join(out_dir, '.done.json')
        if not self.force and os.path.exists(done_p):
            done = json.load(open(done_p))
            if done.get('fingerprint') == fp and all(self.resolve(kernel, o) for o in outs):
                self.log(f'[tfb] {stage:12s} cached ({done.get("runtime_s")} s on {done.get("finished")})')
                return 'cached'
        if os.path.exists(done_p):
            os.remove(done_p)
        ctx = dict(rel=self.data_root, out=out_dir, work_dir=self.work_dir, kernels_dir=self.kernels_dir, input_roots=self.input_roots,
                   stage=stage, module=module, kernel=kernel, pick=self.pick, recipe=self.recipe, smoke=self.smoke,
                   threads=self.threads, cfg=cfg)
        ctx_p = os.path.join(out_dir, '.tfb_ctx.json')
        with open(ctx_p, 'w') as f:
            json.dump(ctx, f, indent=1)
        env = os.environ.copy()
        nt = str(self.env_threads)
        env.update(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', OMP_NUM_THREADS=nt, OPENBLAS_NUM_THREADS=nt, MKL_NUM_THREADS=nt,
                   VECLIB_MAXIMUM_THREADS=nt, PYTHONPATH=PKG_ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
        self.log(f'[tfb] {stage:12s} RUN -> {out_dir}')
        t0 = time.time()
        # output is relayed line by line (a Kaggle script kernel only logs the parent's stdout) and kept in stage.log
        p = subprocess.Popen([sys.executable, '-m', 'tfb.stage_exec', ctx_p], env=env, cwd=out_dir,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        with open(os.path.join(out_dir, 'stage.log'), 'w') as lf:
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
        p.stdout.close()
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status) if hasattr(os, 'waitstatus_to_exitcode') else (status >> 8)
        dt = time.time() - t0
        peak_gb = ru.ru_maxrss / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20)
        if p.returncode != 0:
            raise StageError(f'stage {stage} failed with exit code {p.returncode} after {dt:.0f}s (log above)')
        missing = [o for o in outs if not self.resolve(kernel, o)]
        if missing:
            raise StageError(f'stage {stage} finished but did not write {missing}')
        done = dict(fingerprint=fp, stage=stage, kernel=kernel, outputs={o: os.path.getsize(self.resolve(kernel, o)[0]) for o in outs},
                    inputs=[os.path.relpath(p_, self.work_dir) if p_.startswith(self.work_dir) else p_ for p_ in in_paths],
                    runtime_s=round(dt, 1), peak_rss_gb=round(peak_gb, 2), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                    pick=self.pick, recipe=self.recipe, smoke=self.smoke)
        with open(done_p, 'w') as f:
            json.dump(done, f, indent=1)
        self.log(f'[tfb] {stage:12s} done in {dt:.0f}s | peak RSS {peak_gb:.2f} GB')
        return 'ran'

    # ------------------------------------------------------------------ plan
    def run(self, stages):
        self.write_manifest()
        res = {}
        for st in stages:
            res[st] = self.run_stage(st)
        if 'assemble' in stages:
            self.finalize()
        return res

    def finalize(self):
        kernel, name = C.PICK_KERNEL[self.pick]
        src = self.resolve(kernel, f'submission_{name}.csv')[0]
        dst = os.path.join(self.work_dir, f'submission_{self.pick}_{self.recipe}{"_SMOKE" if self.smoke else ""}.csv')
        shutil.copyfile(src, dst)
        digest = sha256_file(dst)
        with open(dst + '.sha256', 'w') as f:
            f.write(f'{digest}  {os.path.basename(dst)}\\n')
        self.log(f'[tfb] FINAL {dst} sha256 {digest}')
        return dst


def parse_stages(arg, pick, recipe):
    if arg in (None, '', 'all'):
        return C.expand(C.plan(pick, recipe), recipe, pick)
    if arg == 'light':
        return [s for s in C.LIGHT_STAGES if s in C.plan(pick, recipe)]
    out = C.expand([s.strip() for s in arg.split(",") if s.strip()], recipe, pick)
    bad = [s for s in out if s not in C.STAGES and not (C.is_member_stage(s) and s[4:-1] in C.T1_MEMBERS)]
    if bad:
        raise SystemExit(f'unknown stage(s) {bad}; known: {list(C.STAGES)} and t1m:<member part kernel>')
    return out
'''
MODULES['tfb/stage_exec.py'] = '''"""Run one stage in this process:  python -m tfb.stage_exec <ctx.json>

Sets tfb.runtime.ctx, changes into the stage's output folder (as /kaggle/working was the CWD of the original kernel, so
CatBoost's catboost_info/ lands there too) and executes the vendored stage module as __main__.
The runner starts every stage in a fresh subprocess, so memory is released between stages and a crash costs one stage.
"""
import os
import runpy
import sys
import time


def _darwin_rss_in_kb():
    """macOS reports ru_maxrss in bytes, Linux (Kaggle, the reference platform) in KiB; the Task 1 kernels print / gate peak
    RSS assuming KiB. Local development on macOS only: present the value in KiB (no effect on Linux)."""
    import resource
    orig = resource.getrusage

    class _RU:
        def __init__(self, ru):
            self._ru = ru

        def __getattr__(self, k):
            v = getattr(self._ru, k)
            return v // 1024 if k == 'ru_maxrss' else v

    resource.getrusage = lambda who: _RU(orig(who))


def main(ctx_path):
    if sys.platform == 'darwin':
        _darwin_rss_in_kb()
    import tfb.runtime as rt
    rt.ctx = rt.Ctx.load(ctx_path)
    os.makedirs(rt.ctx.out, exist_ok=True)
    os.chdir(rt.ctx.out)
    t0 = time.time()
    print(f'[tfb] stage {rt.ctx.stage} -> {rt.ctx.out} | pick {rt.ctx.pick} | recipe {rt.ctx.recipe} | smoke {rt.ctx.smoke}', flush=True)
    runpy.run_module(f'tfb.stages.{rt.ctx.module}', run_name='__main__', alter_sys=True)
    print(f'[tfb] stage {rt.ctx.stage} finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
'''
MODULES['tfb/envinfo.py'] = '''"""Stage 0: environment and data manifest (package versions, CPU flags, thread settings, data-root summary)."""
import hashlib
import json
import os
import platform
import sys

PACKAGES = ('numpy', 'pandas', 'scipy', 'lightgbm', 'catboost', 'pyarrow', 'scikit-learn')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONHASHSEED')


def versions():
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:                                            # pragma: no cover
        return {}
    out = {}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def cpu():
    info = dict(machine=platform.machine(), processor=platform.processor(), nproc=os.cpu_count())
    try:
        with open('/proc/cpuinfo') as f:
            txt = f.read()
        flags = set()
        for line in txt.splitlines():
            if line.startswith('model name') and 'model' not in info:
                info['model'] = line.split(':', 1)[1].strip()
            if line.startswith('flags'):
                flags = set(line.split(':', 1)[1].split())
        info['flags'] = sorted(f for f in flags if f in ('avx', 'avx2', 'fma', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'sse4_2'))
    except OSError:
        info['flags'] = None
    try:
        with open('/proc/meminfo') as f:
            info['mem_gb'] = round(int(f.readline().split()[1]) / 2 ** 20, 1)
    except (OSError, ValueError, IndexError):
        pass
    return info


def data_summary(data_root):
    n, size = 0, 0
    for dp, _, fs in os.walk(data_root):
        for f in fs:
            n += 1
            try:
                size += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    out = dict(root=os.path.abspath(data_root), files=n, gb=round(size / 1e9, 3))
    for f in ('sample_submission.csv', 'submission_key.csv', 'config/synthetic_release_v1.json'):
        p = os.path.join(data_root, f)
        out[f] = os.path.getsize(p) if os.path.exists(p) else None
    p = os.path.join(data_root, 'config', 'synthetic_release_v1.json')
    if os.path.exists(p):
        out['release_config_sha256'] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


def collect(data_root=None):
    d = dict(python=sys.version.split()[0], platform=platform.platform(), packages=versions(), cpu=cpu(),
             thread_env={k: os.environ.get(k) for k in THREAD_ENV})
    if data_root:
        d['data'] = data_summary(data_root)
    return d


def print_env(d=None):
    d = d or collect()
    print('ENV python', d['python'], '|', d['platform'], flush=True)
    print('ENV packages', json.dumps(d['packages']), flush=True)
    print('ENV cpu', json.dumps(d['cpu']), flush=True)
    print('ENV threads', json.dumps(d['thread_env']), flush=True)
'''
MODULES['tfb/parity.py'] = '''"""Row-level comparison of a rerun against the original Kaggle kernel outputs. Prints one 'PARITY ...' line per block.

Status per block:
  EXACT    every value identical (required for queue rows and path_flow)
  NUMERIC  state / path_flow differ by less than 1e-6 (absolute for state, relative for path_flow) - floating-point drift
  DIFF     state rows differ more (reported with max |d| and RMS; expected for the lean recipe)
  FAIL     queue rows differ, ids / row sets differ, or path_flow differs beyond NUMERIC
  SKIP     the original file is not attached
summary() prints the overall verdict: PASS (all EXACT), CLOSE (only NUMERIC / DIFF state blocks), FAIL otherwise.
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

RESULTS = []


def _line(check, block, status, **kw):
    s = f'PARITY {check} {block} status={status}' + ''.join(f' {k}={v}' for k, v in kw.items())
    print(s, flush=True)
    RESULTS.append(dict(check=check, block=block, status=status, **{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in kw.items()}))


_CACHE = {}


def _key_info(rel):
    """per key row (index submission_id): split ('validation' / 'private', from the key's split column or the month) and,
    for state rows, the blackout flag of tfb-i4d (timestamp in (origin, origin + 90 min] of an official window of the panel)."""
    if rel is None:
        return None
    if 'info' not in _CACHE:
        k = pd.read_csv(f'{rel}/submission_key.csv', dtype=str,
                        usecols=lambda c: c in ('submission_id', 'task', 'split', 'panel', 'timestamp', 'departure_time'))
        if 'split' in k.columns and k.split.notna().all():
            sp = k.split.str.strip().str.lower().replace({'val': 'validation', 'public': 'validation', 'test': 'private'}).to_numpy()
        else:
            t = k.timestamp.fillna(k['departure_time'] if 'departure_time' in k.columns else k.timestamp).fillna('')
            sp = np.where(t.str[:7] == '2031-03', 'validation', np.where(t.str[:7] == '2031-04', 'private', 'unknown'))
        inb = np.zeros(len(k), bool)
        st = np.flatnonzero((k.task == 'state').to_numpy())
        ts = pd.to_datetime(k.timestamp.to_numpy()[st], utc=True)
        for panel, idx in pd.Series(k.panel.to_numpy()[st]).groupby(k.panel.to_numpy()[st]).indices.items():
            tsp = ts[idx]
            hit = np.zeros(len(idx), bool)
            for split in ('validation', 'private'):
                wi_p = f'{rel}/task2/{panel}/{split}/window_index.csv'
                if os.path.exists(wi_p):
                    for o in pd.to_datetime(pd.read_csv(wi_p).forecast_origin, utc=True):
                        hit |= np.asarray((tsp > o) & (tsp <= o + pd.Timedelta(minutes=90)))
            inb[st[idx[hit]]] = True
        _CACHE['info'] = pd.DataFrame(dict(split=sp, blackout=inb), index=k.submission_id.astype('int64').to_numpy())
    return _CACHE['info']


def _windows(rel):
    """window_id -> (split, condition) from the released window indexes."""
    if rel is None:
        return None
    if 'win' not in _CACHE:
        w = []
        for f in sorted(glob.glob(f'{rel}/task2/*/*/window_index.csv')):
            sp = f.split('/')[-2]
            if sp != 'train':
                w.append(pd.read_csv(f, dtype=str)[['window_id', 'condition']].assign(split=sp))
        _CACHE['win'] = pd.concat(w).drop_duplicates('window_id').set_index('window_id') if w else None
    return _CACHE['win']


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def _stat(d):
    nd = int((d != 0).sum())
    mx = float(np.nanmax(np.abs(d))) if len(d) else 0.0
    rms = float(np.sqrt(np.nanmean(d.astype(np.float64) ** 2))) if len(d) else 0.0
    return nd, mx, rms


def compare_submission(ours, orig, check, tasks=('state', 'queue', 'odme'), rel=None):
    """two submission files (submission_id, task, speed_kmh, flow_vph, queue_pred, path_flow)."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    a = pd.read_csv(ours)
    b = pd.read_csv(orig)
    same = len(a) == len(b) and np.array_equal(a.submission_id.to_numpy(), b.submission_id.to_numpy())
    if same:
        _line(check, 'ids', 'EXACT', rows=len(a), bytes_identical=_sha256(ours) == _sha256(orig))
    else:
        _line(check, 'ids', 'FAIL', rows_ours=len(a), rows_orig=len(b), note='aligned on submission_id for the value checks')
        b = b.merge(a[['submission_id']], on='submission_id', how='inner')
        a = a.merge(b[['submission_id']], on='submission_id', how='inner')
    if not (a.task.to_numpy() == b.task.to_numpy()).all():
        _line(check, 'task', 'FAIL', n_diff=int((a.task.to_numpy() != b.task.to_numpy()).sum()))
    info = _key_info(rel) if rel is not None else None
    if info is not None:
        info = info.reindex(b.submission_id.to_numpy())
        sp, bo = info.split.to_numpy(), info.blackout.fillna(False).to_numpy(bool)
    else:
        sp = bo = None
    for task in tasks:
        m = (b.task == task).to_numpy()
        if task == 'state':
            for col in ('speed_kmh', 'flow_vph'):
                d = a[col].to_numpy()[m] - b[col].to_numpy()[m]
                nd, mx, rms = _stat(d)
                st = 'EXACT' if nd == 0 else ('NUMERIC' if mx < 1e-6 else 'DIFF')
                _line(check, f'state.{col}', st, rows=int(m.sum()), n_diff=nd, max_abs=f'{mx:.4g}', rms=f'{rms:.4g}')
                if nd and sp is not None:                 # DESIGN 4.7: per split, blackout and ordinary rows separately
                    for s in ('validation', 'private'):
                        for rows_name, rsel in (('all', np.ones(int(m.sum()), bool)), ('blackout', bo[m]), ('ordinary', ~bo[m])):
                            ms = (sp[m] == s) & rsel
                            nd2, mx2, rms2 = _stat(d[ms])
                            _line(check, f'state.{col}.{s}.{rows_name}', 'EXACT' if nd2 == 0 else ('NUMERIC' if mx2 < 1e-6 else 'DIFF'),
                                  rows=int(ms.sum()), n_diff=nd2, max_abs=f'{mx2:.4g}', rms=f'{rms2:.4g}')
        elif task == 'queue':
            qa = a.queue_pred.to_numpy()[m].astype(int)
            qb = b.queue_pred.to_numpy()[m].astype(int)
            nd = int((qa != qb).sum())
            kw = dict(rows=int(m.sum()), n_diff=nd, pos_ours=int(qa.sum()), pos_orig=int(qb.sum()))
            if sp is not None:
                for s in ('validation', 'private'):
                    kw[f'n_diff_{s}'] = int(((qa != qb) & (sp[m] == s)).sum())
            _line(check, 'queue', 'EXACT' if nd == 0 else 'FAIL', **kw)
        elif task == 'odme':
            pa = a.path_flow.to_numpy()[m].astype(np.float64)
            pb = b.path_flow.to_numpy()[m].astype(np.float64)
            d = pa - pb
            nd, mx, _ = _stat(d)
            relmax = float(np.max(np.abs(d) / np.maximum(np.abs(pb), 1e-12))) if len(d) else 0.0
            st = 'EXACT' if nd == 0 else ('NUMERIC' if relmax < 1e-6 else 'FAIL')
            _line(check, 'odme.path_flow', st, rows=int(m.sum()), n_diff=nd, max_abs=f'{mx:.4g}', max_rel=f'{relmax:.4g}')


def compare_queue_file(ours, orig, check, rel=None):
    """two queue files (window_id, timestamp, link_id, queue_pred): exact equality, per (split, condition) block."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    a = pd.read_csv(ours, dtype={'window_id': str, 'link_id': str})
    b = pd.read_csv(orig, dtype={'window_id': str, 'link_id': str})
    for x in (a, b):
        x['ts'] = pd.to_datetime(x.timestamp, utc=True)
    m = b[['window_id', 'ts', 'link_id', 'queue_pred']].merge(a[['window_id', 'ts', 'link_id', 'queue_pred']], on=['window_id', 'ts', 'link_id'],
                                                              how='outer', suffixes=('_orig', '_ours'), indicator=True)
    unmatched = int((m._merge != 'both').sum())
    m = m[m._merge == 'both']
    qa, qb = m.queue_pred_ours.astype(int).to_numpy(), m.queue_pred_orig.astype(int).to_numpy()
    nd = int((qa != qb).sum())
    _line(check, 'queue', 'EXACT' if nd == 0 and unmatched == 0 else 'FAIL', rows=len(m), unmatched=unmatched, n_diff=nd,
          pos_ours=int(qa.sum()), pos_orig=int(qb.sum()))
    win = _windows(rel)
    if win is not None:
        m = m.assign(split=m.window_id.map(win.split).fillna('?').to_numpy(), cond=m.window_id.map(win.condition).fillna('?').to_numpy(),
                     d=(qa != qb), a=qa, b=qb)
        for (s, c), g in m.groupby(['split', 'cond']):
            jw = []                                       # Jaccard of every window that differs (1 - J bounds |dIoU|)
            for wid in g.loc[g.d, 'window_id'].unique():
                w_all = g[g.window_id == wid]
                uni = int(((w_all.a == 1) | (w_all.b == 1)).sum())
                inter = int(((w_all.a == 1) & (w_all.b == 1)).sum())
                jw.append(1.0 if uni == 0 else inter / uni)
            # score bound (DESIGN 4.7): Jaccard distance is a metric, so per window |dIoU| <= 1 - J; one official window is worth
            # 0.15 / 40 = 0.00375 of the total score per IoU unit -> |dS| <= 0.00375 * sum_w (1 - J_w) for this block
            _line(check, f'queue.{s}.{c}', 'EXACT' if not g.d.any() else 'FAIL', rows=len(g), n_diff=int(g.d.sum()),
                  windows_diff=len(jw), min_jaccard=f'{min(jw):.4f}' if jw else '1', sum_1mJ=f'{sum(1 - j for j in jw):.4f}',
                  score_bound=f'{0.00375 * sum(1 - j for j in jw):.5f}', pos_ours=int(g.a.sum()), pos_orig=int(g.b.sum()))


STATE_KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']


def compare_state_file(ours, orig, check):
    """Task 1 member output (panel, timestamp, station_id, link_id, mask_regime, speed_kmh, flow_vph): row set, then speed /
    flow max |d| and RMS overall, per split (timestamp month) and per panel."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    dt = {'station_id': str, 'link_id': str, 'mask_regime': str, 'panel': str, 'timestamp': str}
    a, b = pd.read_csv(ours, dtype=dt), pd.read_csv(orig, dtype=dt)
    same_order = len(a) == len(b) and (a[STATE_KEYS].to_numpy() == b[STATE_KEYS].to_numpy()).all()
    if not same_order:
        m = b.merge(a, on=STATE_KEYS, how='outer', suffixes=('_o', '_r'), indicator=True)
        unmatched = int((m._merge != 'both').sum())
        m = m[m._merge == 'both']
        a = m[STATE_KEYS].assign(speed_kmh=m.speed_kmh_r, flow_vph=m.flow_vph_r)
        b = m[STATE_KEYS].assign(speed_kmh=m.speed_kmh_o, flow_vph=m.flow_vph_o)
    else:
        unmatched = 0
    _line(check, 'rows', 'EXACT' if same_order else ('FAIL' if unmatched else 'NUMERIC'), rows_ours=len(a), unmatched=unmatched,
          same_order=same_order, bytes_identical=_sha256(ours) == _sha256(orig))
    split = np.where(b.timestamp.str[:7].to_numpy() == '2031-03', 'validation', 'private')
    for col in ('speed_kmh', 'flow_vph'):
        d = a[col].to_numpy(np.float64) - b[col].to_numpy(np.float64)
        nd, mx, rms = _stat(d)
        _line(check, f'state.{col}', 'EXACT' if nd == 0 else ('NUMERIC' if mx < 1e-6 else 'DIFF'), rows=len(d), n_diff=nd,
              max_abs=f'{mx:.4g}', rms=f'{rms:.4g}')
        if nd:
            for s in ('validation', 'private'):
                ms = split == s
                nd2, mx2, rms2 = _stat(d[ms])
                _line(check, f'state.{col}.{s}', 'EXACT' if nd2 == 0 else ('NUMERIC' if mx2 < 1e-6 else 'DIFF'), rows=int(ms.sum()),
                      n_diff=nd2, max_abs=f'{mx2:.4g}', rms=f'{rms2:.4g}')
            for p, idx in pd.Series(np.arange(len(d))).groupby(b.panel.to_numpy()).indices.items():
                nd2, mx2, rms2 = _stat(d[idx])
                _line(check, f'state.{col}.{p}', 'EXACT' if nd2 == 0 else ('NUMERIC' if mx2 < 1e-6 else 'DIFF'), rows=len(idx),
                      n_diff=nd2, max_abs=f'{mx2:.4g}', rms=f'{rms2:.4g}')


def compare_npz(ours, orig, check):
    """every array of two .npz files: integer / bool / string arrays must be EXACT; float arrays report max |d|
    (probabilities: the decoded queue files are the decisive check)."""
    if not orig or not os.path.exists(orig):
        _line(check, '-', 'SKIP', orig=orig)
        return
    if not os.path.exists(ours):
        _line(check, '-', 'FAIL', note='the rerun did not write this file', ours=ours)
        return
    A, B = np.load(ours, allow_pickle=True), np.load(orig, allow_pickle=True)
    ka, kb = set(A.files), set(B.files)
    _line(check, 'keys', 'EXACT' if ka == kb else 'FAIL', n_ours=len(ka), n_orig=len(kb), only_ours=sorted(ka - kb)[:5], only_orig=sorted(kb - ka)[:5])
    worst, n_exact, n_float_diff, bad = 0.0, 0, 0, []
    for k in sorted(ka & kb):
        x, y = A[k], B[k]
        if x.shape != y.shape:
            bad.append(f'{k}:shape'); continue
        if x.dtype.kind == 'f' and y.dtype.kind == 'f':
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            m = float(np.nanmax(d)) if d.size else 0.0
            if m == 0 and np.array_equal(np.isnan(x), np.isnan(y)):
                n_exact += 1
            else:
                n_float_diff += 1; worst = max(worst, m)
        elif np.array_equal(x, y):
            n_exact += 1
        else:
            bad.append(k)
    st = 'FAIL' if bad else ('EXACT' if n_float_diff == 0 else ('NUMERIC' if worst < 1e-6 else 'DIFF'))
    _line(check, 'arrays', st, arrays=len(ka & kb), exact=n_exact, float_diff=n_float_diff, max_abs=f'{worst:.4g}', mismatched=bad[:6])


def summary(save=None):
    st = [r['status'] for r in RESULTS]
    counts = {s: st.count(s) for s in ('EXACT', 'NUMERIC', 'DIFF', 'FAIL', 'SKIP')}
    verdict = 'FAIL' if counts['FAIL'] else ('PASS' if counts['NUMERIC'] + counts['DIFF'] == 0 else 'CLOSE')
    print(f'PARITY SUMMARY verdict={verdict} ' + ' '.join(f'{k}={v}' for k, v in counts.items()), flush=True)
    if save:
        with open(save, 'w') as f:
            json.dump(dict(verdict=verdict, counts=counts, results=RESULTS), f, indent=1)
    return verdict
'''
MODULES['tfb/stages/__init__.py'] = '''"""Vendored Kaggle kernels, one module per stage. Each module is the original kernel source (consolidation/lineage_src/)
with only its I/O and recipe-constant lines changed; every changed or added line ends in a '[TFB]' comment and
tests/test_static.py enforces that. The modules run top to bottom as __main__ inside tfb.stage_exec (never import them)."""
'''
MODULES['tfb/stages/assemble_k10.py'] = '''"""K10 = FINAL CANDIDATE B (hedge) on tfb-k9 (EXP_T1_K9_PREP, prepared 2026-10-30, NOT pushed): identical to swap_k8.py (K8 = hedge B on
tfb-k7) except the base, so K10 - K9 has exactly the April queue rows of K8 - K7 = K6 - K5 (April onset = production CatBoost with the
train prior, April ongoing = veto10; March rows unchanged).

(swap_k8.py docstring follows)
K8 = FINAL CANDIDATE B (hedge) on tfb-k7 (prepared 2026-10-30, NOT pushed): identical to swap_k6.py (K6 = hedge B on tfb-k5)
except the base, so K8 - K7 has exactly the April queue rows of K6 - K5 (April onset = production CatBoost with the train prior,
April ongoing = veto10; March rows unchanged).

(swap_k6.py docstring follows)
Multi-part queue swap: start from the exact base file; for every entry of PARTS replace the queue rows of one condition
(queue_onset / queue_ongoing / any) and one split (validation / private / any) with the rows of SRC_FILE from SRC_KERNEL.
Optional 5th element of a part: 'g72' / 'nog72' / 'any' = only windows whose previous official window of the same panel and
split is exactly 360 min earlier (g72) / all other windows (nog72); computed from the window index (origins only).
Everything else is untouched. Edit BASE / PARTS / OUT_NAME before pushing."""
import glob, numpy as np, pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-k10.py (= kaggle_jobs/swap_k10.py, swap_queue2 + gates); patched lines end in [TFB]
CFG = TFB.cfg  # [TFB] tfb/config.py ASSEMBLE_BASE[pick] + QUEUE_PARTS[pick]: one part per (condition, split) block
BASE_KERNEL, BASE_FILE = CFG['base'][0], CFG['base'][1]  # [TFB] original base tfb-k9/submission_FDRTV.csv
PARTS = [tuple(p) for p in CFG['parts']]  # [TFB]
OUT_NAME = CFG['out_name']  # [TFB]
REL = TFB.rel  # [TFB]
base = pd.read_csv(TFB.kfile(BASE_KERNEL, BASE_FILE))  # [TFB]
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
kq = key[key.task == 'queue'][['submission_id', 'window_id', 'timestamp', 'link_id']].copy()
kq['timestamp'] = pd.to_datetime(kq.timestamp, utc=True)
wins = []
for f in glob.glob(f'{REL}/task2/*/*/window_index.csv'):
    sp = f.split('/')[-2]
    if sp == 'train':
        continue
    w = pd.read_csv(f, dtype=str)
    org = pd.to_datetime(w.forecast_origin, utc=True)
    gs = ((org - pd.Timestamp('2030-01-01', tz='UTC')).dt.total_seconds() // 300).astype(np.int64).to_numpy()
    o = np.argsort(gs, kind='stable'); gap = np.full(len(gs), 10 ** 9, np.int64); gap[o[1:]] = np.diff(gs[o])
    wins.append(w[['window_id', 'condition']].assign(split=sp, g72=(gap == 72)))
wins = pd.concat(wins).drop_duplicates(); assert wins.window_id.is_unique, 'window_id with conflicting condition/split'
print('g72 windows per split and condition:', wins.groupby(['split', 'condition']).g72.sum().to_dict())
kq = kq.merge(wins, on='window_id', how='left', validate='many_to_one'); assert kq.condition.notna().all() and kq.split.notna().all()
isq = (base.task == 'queue').to_numpy(); sid = base.loc[isq, 'submission_id'].to_numpy()
cur = base.loc[isq, 'queue_pred'].to_numpy().astype(int); new = cur.copy()
covered = np.zeros(len(sid), bool)  # [TFB] queue rows replaced by some part (the T4 skeleton's queue rows are 0)
for part in PARTS:
    src_kernel, src_file, cond, split = part[:4]
    gsel = part[4] if len(part) > 4 else 'any'
    q = pd.read_csv(TFB.kfile(src_kernel, src_file), dtype={'window_id': str, 'link_id': str})  # [TFB]
    q['timestamp'] = pd.to_datetime(q.timestamp, utc=True)
    m = kq.merge(q, on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one'); assert m.queue_pred.notna().all()
    assert set(m.submission_id.astype('int64')) == set(sid) and m.submission_id.is_unique, 'queue ids of key and base differ'
    src = pd.Series(m.queue_pred.astype(int).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
    sel = np.ones(len(m), bool)
    if cond != 'any':
        sel &= (m.condition == cond).to_numpy()
    if split != 'any':
        sel &= (m.split == split).to_numpy()
    if gsel != 'any':
        sel &= (m.g72 if gsel == 'g72' else ~m.g72).to_numpy()
    mask = pd.Series(sel, index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
    assert not pd.isna(src).any() and not pd.isna(mask).any(), 'unmapped queue rows after reindex'
    before = new.copy(); new = np.where(mask, src, new)
    covered |= mask  # [TFB]
    print(f'PART {src_kernel}/{src_file} cond={cond} split={split} windows={gsel}: rows {int(mask.sum())} | changed {int((new != before).sum())} '
          f'share {float((new[mask] != before[mask]).mean()):.5f} | positives {int(before[mask].sum())} -> {int(new[mask].sum())}')
    for sp in ('validation', 'private'):
        ms = pd.Series((m.split == sp).to_numpy(), index=m.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
        print(f'  {sp}: changed rows {int(((new != before) & ms).sum())}')
val = pd.Series((kq.split == 'validation').to_numpy(), index=kq.submission_id.astype('int64').to_numpy()).reindex(sid).to_numpy()
print('TOTAL changed rows: validation', int(((new != cur) & val).sum()), '| private', int(((new != cur) & ~val).sum()))
assert CFG['march_from_parts'] or not ((new != cur) & val).any(), 'K10 changed March validation queue rows'  # [TFB] the skeleton base has no March queue rows; validation parts fill them
out = base.copy(); out.loc[isq, 'queue_pred'] = new
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert out.loc[isq, 'queue_pred'].isin([0, 1]).all()
# [TFB] added final gates (tfb-final cell 7 hard gates + full queue coverage), then the writer of the kernel
print('queue rows covered by a part:', int(covered.sum()), 'of', len(covered), '| positives per split:', {sp: int(new[(val if sp == 'validation' else ~val)].sum()) for sp in ('validation', 'private')})  # [TFB]
assert covered.all() or not CFG.get('require_full_cover'), f'{int((~covered).sum())} queue rows not covered by any part'  # [TFB]
assert (out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']] >= 0).all().all(), 'negative values'  # [TFB]
_ks = key[key.task == 'state'][['submission_id', 'panel', 'mask_regime']].assign(submission_id=lambda d: d.submission_id.astype('int64'))  # [TFB]
_st = _ks.merge(out.loc[out.task == 'state', ['submission_id', 'flow_vph']], on='submission_id', how='left', validate='one_to_one')  # [TFB]
assert _st.flow_vph.notna().all(), 'state key rows missing from the file'  # [TFB]
_low = _st.assign(low=_st.flow_vph < 50).groupby(['panel', 'mask_regime']).low.mean()  # [TFB]
assert (_low <= 0.2).all(), f'S_FD empty-road guard would fire: {_low[_low > 0.2]}'  # [TFB]
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'{TFB.out}/submission_{OUT_NAME}.csv', index=False)  # [TFB]
print(OUT_NAME, 'OK', len(out))
'''

def _materialize():
    root = tempfile.mkdtemp(prefix='tfb_pkg_')
    for rel_path, src in MODULES.items():
        p = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write(src)
    return root


def main():
    t0 = time.time()
    kin = os.environ.get('TFB_KAGGLE_INPUT', '/kaggle/input')
    kwd = os.environ.get('TFB_KAGGLE_WORKING', '/kaggle/working')
    smoke = os.environ.get('TFB_PARITY_SMOKE') == '1'            # local dry runs on the miniature fixture only
    pkg = _materialize()
    sys.path.insert(0, pkg)
    from tfb import envinfo, parity
    from tfb.runner import Runner
    print(f"PARITY KERNEL {SPEC['slug']} | tfb {SPEC['tfb_version']} tree {SPEC['tree_sha1'][:12]} | stages {SPEC['stages']} | pick {SPEC['pick']} | recipe {SPEC['recipe']}", flush=True)
    envinfo.print_env()
    rels = sorted(glob.glob(f'{kin}/**/kaggle_public', recursive=True))
    assert rels, f'competition data (kaggle_public) not found under {kin}'
    rel = rels[0]
    missing = [s for s in SPEC['sources'] if not glob.glob(f'{kin}/**/{s}', recursive=True)]
    print('data root', rel, '| kernel sources found', len(SPEC['sources']) - len(missing), 'of', len(SPEC['sources']),
          ('| MISSING ' + ' '.join(missing)) if missing else '', flush=True)
    work = os.path.join(kwd, 'tfb_work')
    r = Runner(rel, work, pick=SPEC['pick'], recipe=SPEC['recipe'], threads=4, env_threads=4, smoke=smoke,
               input_roots=[dict(root=kin, mode='glob')], overrides=SPEC['overrides'])
    r.run(SPEC['stages'])
    for c in SPEC['checks']:
        ours = os.path.join(work, 'kernels', c['ours'][0], c['ours'][1])
        hits = sorted(glob.glob(f"{kin}/**/{c['orig'][0]}/**/{c['orig'][1]}", recursive=True))
        orig = hits[0] if hits else None
        print(f"--- check {c['label']}: ours {ours} | original {orig}", flush=True)
        if c['type'] == 'submission':
            parity.compare_submission(ours, orig, c['label'], tasks=c['tasks'], rel=rel)
        elif c['type'] == 'state':
            parity.compare_state_file(ours, orig, c['label'])
        elif c['type'] == 'npz':
            parity.compare_npz(ours, orig, c['label'])
        else:
            parity.compare_queue_file(ours, orig, c['label'], rel=rel)
    parity.summary(save=os.path.join(kwd, f"parity_{SPEC['slug']}.json"))
    print(f'PARITY KERNEL DONE in {time.time() - t0:.0f}s', flush=True)


main()
