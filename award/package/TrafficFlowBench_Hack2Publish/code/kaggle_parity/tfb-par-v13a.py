# ======================================================================================================================
# GENERATED FILE - DO NOT EDIT. Built by consolidation/bundle.py from the tfb package
#   tfb 0.1.0-m1 | modules tree sha1 2d93478f05fb210ba51e523f14957a9a2f40bc7f | generated 2026-11-03 02:09
# Parity kernel tfb-par-v13a (lean1 + full, ~246 min on Kaggle CPU): Task 1 member part tfb-t1v13a (template t1_fd, original image A, 3.9 h on Kaggle): state_submission_FTA.csv vs the original part
# Stages ['t1m:tfb-t1v13a'] | pick A2 | recipe full | overrides {}
# Kernel sources to attach (2, plus the competition data):
#   hosen42/tfb-t2lab-full hosen42/tfb-t1v13a
# Push: python kaggle_jobs/kjob.py push tfb-par-v13a consolidation/kaggle_parity/tfb-par-v13a.py hosen42/tfb-t2lab-full hosen42/tfb-t1v13a
# Output: PARITY lines + PARITY SUMMARY in the log; /kaggle/working/parity_<slug>.json; stage outputs in /kaggle/working/tfb_work/
# ======================================================================================================================
import glob, json, os, sys, tempfile, time

SPEC = json.loads(r'''{"slug": "tfb-par-v13a", "priority": "lean1 + full", "minutes": 246, "stages": ["t1m:tfb-t1v13a"], "pick": "A2", "recipe": "full", "overrides": {}, "desc": "Task 1 member part tfb-t1v13a (template t1_fd, original image A, 3.9 h on Kaggle): state_submission_FTA.csv vs the original part", "sources": ["tfb-t2lab-full", "tfb-t1v13a"], "checks": [{"type": "state", "ours": ["tfb-t1v13a", "state_submission_FTA.csv"], "orig": ["tfb-t1v13a", "state_submission_FTA.csv"], "label": "v13a_FTA"}], "tfb_version": "0.1.0-m1", "tree_sha1": "2d93478f05fb210ba51e523f14957a9a2f40bc7f", "generated": "2026-11-03 02:09"}''')

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
MODULES['tfb/stages/t1_fd.py'] = '''"""Task 1 v13 (FD-feature bagging member, 3 transductive rounds, cap 6x rows, 15% val holdout kept): seed 7.
v13 notes: identical to v12s5 (same panel split per half, data loading, output names, gates) except
  (1) seed 7 wherever it is free: train-row subsample rng(7), transductive hidden-cell rounds rng(8000+...), LightGBM seed 7
      (holdout rng(7) on the val-month cube unchanged, so the holdout cells are the same as s4/s5);
  (2) features(): + 20 FD-implied columns (congested-branch density / flow implied by 8 speed views, flow-implied speed,
      density ratio, w, k_jam per lane), exactly as scratchpad t1loc/fdfeat2.py.
  Gate vs s4/s5 of the same half on t1_valholdout.csv (val_FTA): KEEP mean delta >= +0.0010 and >= 8/10 panels better;
  DROP delta < +0.0005 or >= 3 panels worse.
v12 notes: Task 1 v12 (bagging member): identical to v11 FTA except seed 1 everywhere it is free: train-row subsample rng(1),
transductive hidden-cell rounds rng(2000+...), LightGBM seed 1. Holdout (rng 7) unchanged. Averaged with v11 FTA in the swap kernel.
Only FTA is trained (FT dropped) to save time.

v11 notes: Task 1 v11: transductive rows (v9, LB +0.0017) + ASM features (v10, holdout +0.0005). Compared on the val-month holdout:
FT (transductive, no ASM; = v9 FT) vs FTA (transductive + ASM). Both fitted on train rows + the same transductive rows.

v10 notes: ASM/GASM characteristic-kernel features (research C A1): mask-normalised kernel averages of the
visible speed / per-lane flow along 4 wave speeds (+-80 km/h free-flow, +-18 km/h congested; both signs because the
travel direction vs milepost order is not assumed), +-5 slots x +-5 stations with real milepost distance.
Compared on the val-month holdout: F (v8 features) vs FA (+ ASM features). No transductive rows here (TR_ROUNDS = 0).

v9 notes: transductive rows. Task 1 is offline, so released (observed) cells of the validation and private months
are hidden in random rounds and used as extra training rows (features from the hidden view, target = released value).
The validation-month internal holdout (15%, seed 7) is hidden in every view and never used as a target.
Compared on that holdout: F (train rows only, = v8 F) vs FT (F + transductive rows).

Task 1 (+ Task 3 proxy) pipeline on Kaggle.

Validation: train split, targets from the official task1 train template, 3 contiguous day blocks (frozen).
Methods scored on identical cells:
  temporal   per-station linear interpolation in time
  base       the base notebook's recipe (temporal 85/15 spatial, density roll, soft FD, conservation smoothing)
  lgbm       LightGBM on neighbour/interp/profile features, residual to temporal interpolation
Final: LightGBM refit on all train targets, applied to the validation and private templates.
"""
import glob, json, os, re, time, gc, resource
import numpy as np
import pandas as pd
import lightgbm as lgb
from tfb.runtime import ctx as TFB  # [TFB] family template from consolidation/lineage_src/tfb-t1v13a.py; patched lines end in [TFB]
CFG = TFB.cfg  # [TFB] per-part constants (tfb/config.py t1_part_cfg); rendered with them this file equals the original part

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = TFB.out  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
PANELS = list(CFG['panels'])  # [TFB] the panels of this part (a / b / c)
SMOKE = TFB.smoke  # [TFB] False in every real run
N_TREES, MAX_ROWS = (60, 150_000) if SMOKE else (300, 600_000)
if SMOKE:
    PANELS = ['D7_I10_W']
FAMILY = lambda p: p.rsplit('_', 1)[0]
RATE = {'R1': .2, 'R2': .3, 'R3': .5}
TR_ROUNDS, TR_HIDE = CFG['rounds'], 0.20  # [TFB]
ASM_WAVES, ASM_K, ASM_SIGMA, ASM_TAU = (80.0, -80.0, 18.0, -18.0), 5, 0.6, 5.0


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def peak_gb():
    """peak resident set size of this process (Linux ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


# ------------------------------------------------------------------ loading
def read_split(panel, split, layer):
    fs = sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/{layer}/*/*.parquet'))
    if SMOKE:   # same calendar days in every layer: every 9th day
        keep = sorted({os.path.basename(f)[-18:-8] for f in fs})[::9]
        fs = [f for f in fs if os.path.basename(f)[-18:-8] in keep]
    parts = []
    for f in fs:
        d = pd.read_parquet(f, columns=['date', 'timestamp', 'station_id', 'link_id', 'milepost',
                                        'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
        m = re.search(r'mask_regime=(R\\d)', f)
        d['regime'] = m.group(1) if m else ''
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


class Cube:
    """(day, 288, station) arrays for one panel/split; stations in milepost order."""

    def __init__(self, masked, stations, full=None):
        self.days = sorted(masked.date.unique())
        self.reg = masked.drop_duplicates('date').set_index('date').regime.reindex(self.days).to_numpy()
        self.S = len(stations)
        sidx = pd.Index(stations)
        shape = (len(self.days), 288, self.S)

        def put(df, col, fill=np.nan, dtype=np.float32):
            a = np.full(shape, fill, dtype=dtype)
            ts = pd.to_datetime(df.timestamp, utc=True)
            di = pd.Index(self.days).get_indexer(df.date)
            ok = di >= 0
            ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            si = sidx.get_indexer(df.station_id)
            ok &= si >= 0
            a[di[ok], ti[ok], si[ok]] = df[col].to_numpy()[ok]
            return a
        self.mv = put(masked, 'speed_kmh'); self.mq = put(masked, 'flow_vph'); self.mo = put(masked, 'occupancy')
        self.melig = put(masked, 'is_score_eligible', 0, np.int8).astype(bool)
        self.dow = pd.to_datetime(pd.Series(self.days)).dt.dayofweek.to_numpy()
        if full is not None:
            self.v = put(full, 'speed_kmh'); self.q = put(full, 'flow_vph')
            self.elig = put(full, 'is_score_eligible', 0, np.int8).astype(bool) & np.isfinite(self.v)
        self.put = put


def template_mask(panel, split, cube, stations):
    t = pd.read_csv(f'{REL}/task1/{panel}/{split}/sample_submission_state.csv', dtype=str)
    t['date'] = t.timestamp.str[:10]
    ts = pd.to_datetime(t.timestamp, utc=True)
    di = pd.Index(cube.days).get_indexer(t.date)
    si = pd.Index(stations).get_indexer(t.station_id)
    ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    ok = (di >= 0) & (si >= 0)
    m = np.zeros((len(cube.days), 288, len(stations)), bool)
    m[di[ok], ti[ok], si[ok]] = True
    return m, t, (di, ti, si, ok)


# ------------------------------------------------------------------ features
def interp_axis(a, axis, limit=None):
    b = np.moveaxis(a, axis, -1)
    sh = b.shape
    flat = b.reshape(-1, sh[-1]).astype(np.float64)
    out = flat.copy()
    x = np.arange(sh[-1])
    for i in range(flat.shape[0]):
        y = flat[i]; ok = np.isfinite(y)
        if ok.sum() == 0 or ok.all():
            continue
        f = np.interp(x, x[ok], y[ok])
        if limit is not None:
            idx = np.where(ok)[0]; pos = np.searchsorted(idx, x)
            dl = x - idx[np.clip(pos - 1, 0, len(idx) - 1)]; dr = idx[np.clip(pos, 0, len(idx) - 1)] - x
            f = np.where(np.minimum(np.abs(dl), np.abs(dr)) <= limit, f, np.nan)
        out[i] = np.where(ok, y, f)
    return np.moveaxis(out.reshape(sh), -1, axis).astype(np.float32)


def shift(a, dt=0, ds=0):
    """value at (t+dt, s+ds); NaN outside."""
    out = np.full_like(a, np.nan)
    T, S = a.shape[1], a.shape[2]
    ts_src = slice(max(dt, 0), T + min(dt, 0)); ts_dst = slice(max(-dt, 0), T + min(-dt, 0))
    ss_src = slice(max(ds, 0), S + min(ds, 0)); ss_dst = slice(max(-ds, 0), S + min(-ds, 0))
    out[:, ts_dst, ss_dst] = a[:, ts_src, ss_src]
    return out


def gap_dist(obs):
    T = obs.shape[1]; fin = np.isfinite(obs)
    t = np.arange(T)[None, :, None]
    prev = np.maximum.accumulate(np.where(fin, t, -10 ** 6), axis=1)
    nxt = np.minimum.accumulate(np.where(fin, t, 10 ** 6)[:, ::-1], axis=1)[:, ::-1]
    return np.minimum(t - prev, 99).astype(np.float32), np.minimum(nxt - t, 99).astype(np.float32)


def base_recipe(c, st, hv, hq):
    """The base notebook's Task 1 recipe, re-implemented on the station cube (milepost order)."""
    lanes = st.lanes.to_numpy()[None, None, :]; vf = st.vf.to_numpy()[None, None, :]
    cap = st.cap.to_numpy()[None, None, :]; kj = st.k_jam.to_numpy()[None, None, :]
    t12v = interp_axis(c.mv, 1, 12); t12q = interp_axis(c.mq / lanes, 1, 12)
    s8v = interp_axis(t12v, 2, 8); s8q = interp_axis(t12q, 2, 8)
    pv = np.where(np.isfinite(t12v), 0.85 * t12v + 0.15 * np.where(np.isfinite(s8v), s8v, t12v),
                  np.where(np.isfinite(s8v), 0.9 * s8v + 0.1 * hv, hv))
    pq = np.where(np.isfinite(t12q), 0.85 * t12q + 0.15 * np.where(np.isfinite(s8q), s8q, t12q),
                  np.where(np.isfinite(s8q), 0.9 * s8q + 0.1 * hq / lanes, hq / lanes)) * lanes
    pv = np.clip(pv, 5, vf * 1.05); pq = np.maximum(pq, 50)
    k = pq / np.maximum(pv, 1)
    kp = np.pad(k, ((0, 0), (1, 1), (0, 0)), mode='edge')
    pq = np.clip(0.85 * pq + 0.15 * (kp[:, :-2] + kp[:, 1:-1] + kp[:, 2:]) / 3 * pv, 50, cap * 1.15)
    kc = cap / vf; w = cap / np.maximum(kj - kc, 1); k = pq / np.maximum(pv, 1)
    qfd = np.where(k <= kc, vf * k, w * np.maximum(kj - k, 0))
    pq = np.where((qfd > 0) & (qfd < cap * 1.3), 0.85 * pq + 0.15 * qfd, pq)
    for _ in range(5):
        for j in range(pq.shape[2] - 1):
            cc = 0.2 * (pq[:, :, j + 1] - pq[:, :, j]); pq[:, :, j] += cc; pq[:, :, j + 1] -= cc
    return np.clip(pv, 5, vf * 1.05), np.clip(pq, 50, cap * 1.15)


def lowrank(a, rank=3):
    """per-day rank-k SVD reconstruction of a (day, 288, S) matrix (NaN filled with column means)."""
    out = np.full_like(a, np.nan)
    for d in range(a.shape[0]):
        m = a[d].astype(np.float64)
        if not np.isfinite(m).any():
            continue
        cm = np.nanmean(m, 0); cm = np.where(np.isfinite(cm), cm, np.nanmean(m))
        m = np.where(np.isfinite(m), m, cm[None, :])
        mu = m.mean(0, keepdims=True); U, sv, Vt = np.linalg.svd(m - mu, full_matrices=False)
        out[d] = (U[:, :rank] * sv[:rank]) @ Vt[:rank] + mu
    return out.astype(np.float32)


def lowrank_days(a, rank=3):
    """per-station rank-k SVD across the split's days: (day x 288) matrix per station -> the month's own daily pattern."""
    out = np.full_like(a, np.nan)
    for s_ in range(a.shape[2]):
        m = a[:, :, s_].astype(np.float64)
        if not np.isfinite(m).any() or m.shape[0] < 3:
            continue
        cm = np.nanmean(m, 0); cm = np.where(np.isfinite(cm), cm, np.nanmean(m))
        m = np.where(np.isfinite(m), m, cm[None, :])
        mu = m.mean(0, keepdims=True); U, sv, Vt = np.linalg.svd(m - mu, full_matrices=False)
        out[:, :, s_] = (U[:, :rank] * sv[:rank]) @ Vt[:rank] + mu
    return out.astype(np.float32)


NEW_LR = ('lr5_v', 'lr5_q', 'lr_o', 'lrd_v', 'lrd_q')


def features(c, st, hv, hq):
    lanes = st.lanes.to_numpy()[None, None, :]; vf = st.vf.to_numpy()[None, None, :]
    shape = c.mv.shape
    tv = interp_axis(c.mv, 1); tq = interp_axis(c.mq, 1)
    tv = np.where(np.isfinite(tv), tv, hv); tq = np.where(np.isfinite(tq), tq, hq)
    sv = interp_axis(c.mv, 2); sq = interp_axis(c.mq, 2)
    sv = np.where(np.isfinite(sv), sv, tv); sq = np.where(np.isfinite(sq), sq, tq)
    gp, gn = gap_dist(c.mv)
    nv = c.mv / vf; nq = c.mq / lanes
    F = {'tv': tv / vf, 'tq': tq / lanes, 'sv': sv / vf, 'sq': sq / lanes, 'hv': hv / vf, 'hq': hq / lanes,
         'tk': tq / np.maximum(tv, 1) / lanes, 'gp': gp, 'gn': gn,
         'tod': np.broadcast_to(np.arange(288, dtype=np.float32)[None, :, None], shape),
         'dow': np.broadcast_to(c.dow.astype(np.float32)[:, None, None], shape),
         'lanes': np.broadcast_to(lanes.astype(np.float32), shape), 'vf': np.broadcast_to(vf.astype(np.float32), shape),
         'pos': np.broadcast_to(np.linspace(0, 1, shape[2], dtype=np.float32)[None, None, :], shape),
         'rate': np.broadcast_to(np.array([RATE.get(r, .3) for r in c.reg], np.float32)[:, None, None], shape)}
    for ds in (-2, -1, 1, 2):
        for dt in (-1, 0, 1):
            F[f'v{ds:+d}{dt:+d}'] = shift(nv, dt, ds); F[f'q{ds:+d}{dt:+d}'] = shift(nq, dt, ds)
    for dt in (-2, -1, 1, 2):
        F[f'v0{dt:+d}'] = shift(nv, dt, 0); F[f'q0{dt:+d}'] = shift(nq, dt, 0)
    for ds in (-1, 0, 1):
        for dt in (-1, 0, 1):
            if (ds, dt) != (0, 0):
                F[f'o{ds:+d}{dt:+d}'] = shift(c.mo, dt, ds)
    F['to'] = interp_axis(c.mo, 1); F['so'] = interp_axis(c.mo, 2)
    # neighbour-anchored ratio transfer
    for ds in (-1, 1):
        nb = shift(nq, 0, ds)
        with np.errstate(all='ignore'):
            r = interp_axis(np.log(nq) - np.log(nb), 1)
        F[f'rq{ds:+d}'] = nb * np.exp(r)
    F['lr_v'] = lowrank(F['tv']); F['lr_q'] = lowrank(F['tq'])
    F['lr5_v'] = lowrank(F['tv'], 5); F['lr5_q'] = lowrank(F['tq'], 5); F['lr_o'] = lowrank(F['to'])
    F['lrd_v'] = lowrank_days(F['tv']); F['lrd_q'] = lowrank_days(F['tq'])
    F.update(asm_feats(nv, nq, st.milepost.to_numpy()))
    n12 = len(F)
    assert n12 == V12_NFEAT, f'v12 feature count changed: {n12} != {V12_NFEAT}'
    F.update(fd_feats(F, st, shape))
    assert len(F) == n12 + N_FD, f'FD feature count {len(F) - n12} != {N_FD}'
    return F, tv, tq


V12_NFEAT = 78     # 15 base + 24 v/q nb + 8 v0/q0 + 8 occ + to/so + 2 rq + 7 low-rank + 12 ASM
FD_SPEEDS = ('tv', 'sv', 'v-1+0', 'v+1+0', 'v0-1', 'v0+1', 'asm_v-18', 'asm_v+18')
FD_COLS = [f'fd{kind}_{nm}' for nm in FD_SPEEDS for kind in 'kq'] + ['fdv_tq', 'fdk_ratio', 'fd_w', 'fd_kj']
N_FD = len(FD_COLS)
assert N_FD == 20


def fd_feats(F, st, shape):
    """FD-implied features (= scratchpad t1loc/fdfeat2.py). Per station, per lane:
    w = cap / max(k_jam - k_crit, 1e-6) (congested wave speed), kj = k_jam.
    For each speed view vv (km/h = normalised view * vf): congested-branch density w*kj/(max(vv,1)+w) and flow w*kj*vv/(max(vv,1)+w);
    fdv_tq = per-lane interp flow / max(kj - tq/w, 0.5) / vf; fdk_ratio = tk / fdk_tv; fd_w, fd_kj. NaN views stay NaN."""
    ln = st.lanes.to_numpy().astype(np.float64)
    capS = st.cap.to_numpy() / ln; kcS = st.critical_density.to_numpy() / ln; kjS = st.k_jam.to_numpy() / ln
    w_ = (capS / np.maximum(kjS - kcS, 1e-6))[None, None, :]; kj_ = kjS[None, None, :]
    vf = st.vf.to_numpy().astype(np.float64)[None, None, :]
    out = {}
    with np.errstate(all='ignore'):
        for nm in FD_SPEEDS:
            vv = np.asarray(F[nm], np.float64) * vf
            den = np.clip(vv, 1, None) + w_
            out[f'fdk_{nm}'] = (w_ * kj_ / den).astype(np.float32)
            out[f'fdq_{nm}'] = (w_ * kj_ * vv / den).astype(np.float32)
        tqn = np.asarray(F['tq'], np.float64)
        out['fdv_tq'] = (tqn / np.clip(kj_ - tqn / w_, 0.5, None) / vf).astype(np.float32)
        out['fdk_ratio'] = (np.asarray(F['tk'], np.float64) / out['fdk_tv']).astype(np.float32)
    out['fd_w'] = np.broadcast_to(w_.astype(np.float32), shape)
    out['fd_kj'] = np.broadcast_to(kj_.astype(np.float32), shape)
    assert list(out) == FD_COLS
    return out


def asm_feats(nv, nq, x):
    """ASM-style kernel smoothing along characteristics. x: milepost (km) per station, in cube station order."""
    D, T, S = nv.shape
    out = {}
    num = {(c, ch): np.zeros(nv.shape, np.float32) for c in ASM_WAVES for ch in 'vq'}
    den = {c: np.zeros(nv.shape, np.float32) for c in ASM_WAVES}
    xi = np.asarray(x, np.float64)
    for ds in range(-ASM_K, ASM_K + 1):
        dx = np.full(S, np.nan)
        if ds >= 0:
            dx[:S - ds] = xi[ds:] - xi[:S - ds]
        else:
            dx[-ds:] = xi[:S + ds] - xi[-ds:]
        for dt in range(-ASM_K, ASM_K + 1):
            if ds == 0 and dt == 0:
                continue
            sv = shift(nv, dt, ds); sq = shift(nq, dt, ds)
            ok = np.isfinite(sv) & np.isfinite(sq)
            sv = np.where(ok, sv, 0).astype(np.float32); sq = np.where(ok, sq, 0).astype(np.float32); okf = ok.astype(np.float32)
            for c in ASM_WAVES:
                w = np.exp(-np.abs(dx) / ASM_SIGMA - np.abs(dt * 5.0 - dx / c * 60.0) / ASM_TAU)
                w = np.nan_to_num(w, nan=0.0).astype(np.float32)[None, None, :]
                num[c, 'v'] += sv * w; num[c, 'q'] += sq * w; den[c] += okf * w
    for c in ASM_WAVES:
        with np.errstate(all='ignore'):
            out[f'asm_v{c:+.0f}'] = np.where(den[c] > 1e-6, num[c, 'v'] / den[c], np.nan).astype(np.float32)
            out[f'asm_q{c:+.0f}'] = np.where(den[c] > 1e-6, num[c, 'q'] / den[c], np.nan).astype(np.float32)
        out[f'asm_w{c:+.0f}'] = den[c]
    return out


def stack(F, mask):
    return np.column_stack([np.asarray(F[k])[mask] for k in F]).astype(np.float32), list(F)


# ------------------------------------------------------------------ scoring
def s_state(v, q, tv_, tq_, lanes, reg):
    out = {}
    for r in ('R1', 'R2', 'R3'):
        m = reg == r
        if m.sum() == 0:
            continue
        rv = np.sqrt(np.mean((v[m] - tv_[m]) ** 2)); rq = np.sqrt(np.mean(((q[m] - tq_[m]) / lanes[m]) ** 2))
        out[r] = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
    return float(np.mean(list(out.values()))), out


def lwr_proxy(c, st, tmask, pv, pq):
    L = st.length_km.to_numpy()[None, None, :]
    vv = np.where(tmask, pv, c.v); qq = np.where(tmask, pq, c.q)
    N = qq / np.clip(vv, 1, None) * L; Nt = c.q / np.clip(c.v, 1, None) * L
    ok = c.elig[:, :-1] & c.elig[:, 1:]
    dN = N[:, 1:] - N[:, :-1]; dNt = Nt[:, 1:] - Nt[:, :-1]
    out = {}
    for r in ('R1', 'R2', 'R3'):
        m = ok & (c.reg == r)[:, None, None]
        if m.sum():
            out[r] = max(0, 1 - np.nansum(np.abs(dN - dNt)[m]) / np.nansum(np.abs(dNt)[m]))
    return float(np.mean(list(out.values())))




def predict_variants(Xm, tvn, tqn, vf_, ln_, reg_, MODELS, ixC):
    """returns {variant: (speed, flow)} for rows of Xm (all cols); each model carries its own column index."""
    out = {}
    for nm, (mv, mq, ix) in MODELS.items():
        out[nm] = (np.clip((tvn + mv.predict(Xm[:, ix])) * vf_, 3, 1.05 * vf_), np.maximum((tqn + mq.predict(Xm[:, ix])) * ln_, 20))
    return out

# ------------------------------------------------------------------ main
GAP = 18


def gap_origins(panel, days):
    fs = TFB.kglob('tfb-t2lab-full', f'pseudo_{panel}.npz')  # [TFB] hidden input: the pseudo-window stage
    if not fs:
        return []
    z = np.load(fs[0], allow_pickle=True)
    all_days = sorted(os.path.basename(f)[-18:-8].replace('_', '-') for f in
                      glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    pos = {dd: i for i, dd in enumerate(days)}
    return [(pos[all_days[d]], int(t)) for d, t in zip(z['d'], z['t']) if all_days[d] in pos]


def darken(c, origins):
    mv, mq, mo = c.mv.copy(), c.mq.copy(), c.mo.copy(); gapm = np.zeros(mv.shape, bool)
    for d, t in origins:
        mv[d, t + 1:t + 1 + GAP] = np.nan; mq[d, t + 1:t + 1 + GAP] = np.nan; mo[d, t + 1:t + 1 + GAP] = np.nan
        gapm[d, t + 1:t + 1 + GAP] = True
    return mv, mq, mo, gapm


def split_gap_share(panel, split, stations):
    wi_p = f'{REL}/task2/{panel}/{split}/window_index.csv'
    tpl = pd.read_csv(f'{REL}/task1/{panel}/{split}/sample_submission_state.csv', usecols=['timestamp'])
    if not os.path.exists(wi_p):
        return 0.0
    wi = pd.read_csv(wi_p); ts = pd.to_datetime(tpl.timestamp, utc=True); ing = np.zeros(len(ts), bool)
    for o in pd.to_datetime(wi.forecast_origin, utc=True):
        ing |= (ts > o) & (ts <= o + pd.Timedelta(minutes=5 * GAP))
    return float(ing.mean())


results = []
subs = []
val_rows = []
params = dict(objective='regression', learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, max_bin=63,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=CFG['threads'])  # [TFB] Kaggle: 4
for panel in PANELS:
    tp = time.time()
    fdp = pd.read_csv(f'{REL}/corridors/{panel}/network/fd_parameters.csv', dtype={'link_id': str}).set_index('link_id')
    tr_full = read_split(panel, 'train', 'mainline_states')
    tr_mask = read_split(panel, 'train', 'mainline_states_masked')
    st = tr_full.drop_duplicates('station_id').set_index('station_id')[['link_id', 'milepost']].sort_values('milepost')
    st = st.join(fdp.rename(columns={'free_speed_kmh': 'vf', 'capacity_vph': 'cap'})[['lanes', 'length_km', 'vf', 'cap', 'critical_density', 'k_jam']], on='link_id')
    stations = st.index.tolist()
    c = Cube(tr_mask, stations, full=tr_full); del tr_full, tr_mask; gc.collect()
    lanes3 = np.broadcast_to(st.lanes.to_numpy()[None, None, :], c.mv.shape)
    # profile from released (masked-layer) train observations: weekday x slot x station
    hv = np.full((7, 288, len(stations)), np.nan, np.float32); hq = hv.copy()
    for w in range(7):
        sel = c.dow == w
        with np.errstate(all='ignore'):
            hv[w] = np.nanmean(c.mv[sel], 0); hq[w] = np.nanmean(c.mq[sel], 0)
    hv = np.where(np.isfinite(hv), hv, np.nanmean(hv)); hq = np.where(np.isfinite(hq), hq, np.nanmean(hq))
    tmask, _, _ = template_mask(panel, 'train', c, stations)
    tmask &= c.elig
    assert tmask.sum() > 0, 'no Task 1 targets mapped'
    origins = gap_origins(panel, c.days)
    gshare = split_gap_share(panel, 'validation', stations)
    mv0, mq0 = c.mv, c.mq
    c.mv, c.mq, c.mo, gapm = darken(c, origins)                     # train on the darkened view
    F, tv, tq = features(c, st, hv[c.dow], hq[c.dow])
    X, cols = stack(F, tmask)
    # hard gates (v13): feature count = v12 + 20, no all-NaN FD column on the training rows
    assert len(cols) == V12_NFEAT + N_FD and cols[-N_FD:] == FD_COLS, (len(cols), cols[-N_FD:])
    fd_fin = {k: float(np.isfinite(X[:, cols.index(k)]).mean()) for k in FD_COLS}
    assert all(v > 0 for v in fd_fin.values()), f'all-NaN FD columns: {[k for k, v in fd_fin.items() if v == 0]}'
    log(panel, 'features', len(cols), '(v12', V12_NFEAT, '+ FD', N_FD, ') FD finite share min',
        round(min(fd_fin.values()), 4), {k: round(v, 3) for k, v in fd_fin.items() if v < 0.99})
    in_gap = gapm[tmask]
    log(panel, 'gap origins', len(origins), 'targets in simulated gaps', int(in_gap.sum()),
        'validation gap share of targets', round(gshare, 4))
    vf3 = np.broadcast_to(st.vf.to_numpy()[None, None, :], c.mv.shape)
    yv = (c.v / vf3)[tmask] - F['tv'][tmask]; yq = (c.q / lanes3)[tmask] - F['tq'][tmask]
    dayid = np.broadcast_to(np.arange(len(c.days))[:, None, None], c.mv.shape)[tmask]
    block = np.minimum(dayid * 3 // len(c.days), 2)                      # 3 contiguous blocks (frozen)
    reg_t = np.broadcast_to(c.reg[:, None, None], c.mv.shape)[tmask]
    rng = np.random.default_rng(CFG['row_rng'])  # [TFB]
    NOPROF = {'hv', 'hq', 'tod', 'dow'}
    ixD = [cols.index(k) for k in cols if k not in NOPROF]
    ixF = [i for i in ixD if not cols[i].startswith('asm')]; posF = [ixD.index(i) for i in ixF]
    idx4 = np.arange(len(X))
    if len(idx4) > 4 * MAX_ROWS:
        idx4 = rng.choice(idx4, 4 * MAX_ROWS, replace=False)
    Xtr, yvtr, yqtr = X[idx4][:, ixD], yv[idx4], yq[idx4]
    del X, F, c, yv, yq; gc.collect()
    # ---------------- target-month cubes + transductive rows
    CS, HOLD = {}, {}
    TX, TYV, TYQ = [], [], []
    for split in ('validation', 'private'):
        sm = read_split(panel, split, 'mainline_states_masked')
        cs = Cube(sm, stations); del sm
        CS[split] = cs
        if not TR_ROUNDS:
            HOLD[split] = (cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq) & (np.random.default_rng(7).random(cs.mv.shape) < CFG['holdout'])) if split == 'validation' else np.zeros(cs.mv.shape, bool)  # [TFB]
            continue
        obs = cs.melig & np.isfinite(cs.mv) & np.isfinite(cs.mq)
        hold = np.zeros(cs.mv.shape, bool)
        if split == 'validation':
            hold = obs & (np.random.default_rng(7).random(cs.mv.shape) < CFG['holdout'])  # [TFB]
        HOLD[split] = hold
        vf3s = np.broadcast_to(st.vf.to_numpy()[None, None, :], cs.mv.shape)
        ln3s = np.broadcast_to(st.lanes.to_numpy()[None, None, :], cs.mv.shape)
        for r in range(TR_ROUNDS):
            hid = obs & ~hold & (np.random.default_rng(CFG['rng_base'] + 10 * r + (split == 'private')).random(cs.mv.shape) < TR_HIDE)  # [TFB]
            cv = Cube.__new__(Cube); cv.__dict__.update(cs.__dict__)
            gone = hid | hold
            cv.mv = np.where(gone, np.nan, cs.mv); cv.mq = np.where(gone, np.nan, cs.mq); cv.mo = np.where(gone, np.nan, cs.mo)
            Fr, _, _ = features(cv, st, hv[cv.dow], hq[cv.dow])
            Xr = np.column_stack([np.asarray(Fr[cols[i]])[hid] for i in ixD]).astype(np.float32)
            TX.append(Xr); TYV.append((cs.mv / vf3s)[hid] - Fr['tv'][hid]); TYQ.append((cs.mq / ln3s)[hid] - Fr['tq'][hid])
            del Fr, cv; gc.collect()
        log(panel, split, 'transductive rows so far', sum(len(t) for t in TX))
    pF = dict(params, num_leaves=511, min_data_in_leaf=100, learning_rate=0.04, seed=CFG['lgb_seeds'][0], bagging_seed=CFG['lgb_seeds'][1], feature_fraction_seed=CFG['lgb_seeds'][2])  # [TFB]
    NT = 100 if SMOKE else 1500
    TX = np.concatenate(TX); TYV = np.concatenate(TYV); TYQ = np.concatenate(TYQ)
    if len(TX) > CFG['tx_cap'] * MAX_ROWS:  # [TFB]
        sel = rng.choice(len(TX), CFG['tx_cap'] * MAX_ROWS, replace=False); TX, TYV, TYQ = TX[sel], TYV[sel], TYQ[sel]  # [TFB]
    XA = np.concatenate([Xtr, TX]); YV = np.concatenate([yvtr, TYV]); YQ = np.concatenate([yqtr, TYQ])
    del TX, TYV, TYQ, Xtr; gc.collect()
    MODELS = {'FTA': (lgb.train(pF, lgb.Dataset(XA, YV), NT), lgb.train(pF, lgb.Dataset(XA, YQ), NT), ixD)}
    imp = pd.Series(MODELS['FTA'][0].feature_importance('gain'), index=[cols[i] for i in ixD]).sort_values(ascending=False)
    log(panel, 'FTA speed top features', imp.head(10).round(0).to_dict())
    log(panel, 'train+transductive rows', len(XA))
    del XA, YV, YQ; gc.collect()
    for split in ('validation', 'private'):
        cs = CS[split]
        if split == 'validation':
            # internal holdout on the validation month itself: hide 15% of released eligible cells (same cells as v8)
            hold = HOLD[split]
            hv_true, hq_true = cs.mv[hold].copy(), cs.mq[hold].copy()
            ch = Cube.__new__(Cube); ch.__dict__.update(cs.__dict__)
            ch.mv = np.where(hold, np.nan, cs.mv); ch.mq = np.where(hold, np.nan, cs.mq); ch.mo = np.where(hold, np.nan, cs.mo)
            Fh, tvh, tqh = features(ch, st, hv[ch.dow], hq[ch.dow])
            Xh = np.column_stack([np.asarray(Fh[k])[hold] for k in cols]).astype(np.float32)
            vfh = np.broadcast_to(st.vf.to_numpy()[None, None, :], ch.mv.shape)[hold]
            lnh = np.broadcast_to(st.lanes.to_numpy()[None, None, :], ch.mv.shape)[hold]
            regh = np.broadcast_to(ch.reg[:, None, None], ch.mv.shape)[hold]
            bvh, bqh = base_recipe(ch, st, hv[ch.dow], hq[ch.dow])
            cand = {'temporal': (tvh[hold], tqh[hold]), 'base': (bvh[hold], bqh[hold])}
            cand.update(predict_variants(Xh, Fh['tv'][hold], Fh['tq'][hold], vfh, lnh, regh, MODELS, ixD))
            vh = {k: round(s_state(a, b, hv_true, hq_true, lnh, regh)[0], 4) for k, (a, b) in cand.items()}
            row_v = dict(panel=panel, family=FAMILY(panel), **{f'val_{k}': v for k, v in vh.items()})
            val_rows.append(row_v); log(panel, 'VALIDATION-month internal holdout S_state', vh)
            del Fh, Xh, ch
        tm, tpl, (di, ti, si, ok) = template_mask(panel, split, cs, stations)
        Fs, tvs, tqs = features(cs, st, hv[cs.dow], hq[cs.dow])
        Xs = np.column_stack([np.asarray(Fs[k])[di[ok], ti[ok], si[ok]] for k in cols]).astype(np.float32)
        bad = [k for k in FD_COLS if not np.isfinite(Xs[:, cols.index(k)]).any()]
        assert not bad, f'{panel} {split}: all-NaN FD columns on template rows {bad}'
        vfs = st.vf.to_numpy()[si[ok]]; lns = st.lanes.to_numpy()[si[ok]]
        regs = tpl.mask_regime.to_numpy()[ok]
        V3 = predict_variants(Xs, Fs['tv'][di[ok], ti[ok], si[ok]], Fs['tq'][di[ok], ti[ok], si[ok]], vfs, lns, regs, MODELS, ixD)
        out = tpl.loc[ok, ['timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
        out.insert(0, 'panel', panel)
        for vname, (pvs, pqs) in V3.items():
            out[f'speed_{vname}'] = pvs; out[f'flow_{vname}'] = pqs
        out['speed_kmh'] = out.speed_FTA; out['flow_vph'] = out.flow_FTA
        miss = int((~ok).sum())
        log(panel, split, 'template rows', len(tpl), 'predicted', len(out), 'unmapped', miss,
            'speed range', np.round([out.speed_kmh.min(), out.speed_kmh.max()], 1).tolist(),
            'flow<50 share', round(float((out.flow_vph < 50).mean()), 4))
        subs.append(out)
        del Fs, Xs, cs, CS[split]; gc.collect()
    log(panel, f'panel time {time.time() - tp:.0f}s', f'peak RSS {peak_gb():.2f} GB')

V = pd.DataFrame(val_rows); V.to_csv(f'{OUT}/t1_valholdout.csv', index=False)
print('\\n===== validation-month internal holdout (family-mean S_state) =====')
print(V.groupby('family')[[c for c in V.columns if c.startswith('val_')]].mean().mean().round(4).to_string())
print('\\n===== per-panel val-month holdout S_state (gate: compare val_FTA with s4/s5 of the same half) =====')
print(V[['panel'] + [c for c in V.columns if c.startswith('val_')]].to_string(index=False))
print('mean val_FTA', round(float(V.val_FTA.mean()), 5))
S = pd.concat(subs, ignore_index=True)
base_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
for vname in ('FTA',):
    S[base_cols].assign(speed_kmh=S[f'speed_{vname}'], flow_vph=S[f'flow_{vname}']).to_csv(f'{OUT}/state_submission_{vname}.csv', index=False)
S[base_cols + ['speed_kmh', 'flow_vph']].to_csv(f'{OUT}/state_submission.csv', index=False)
log('state rows', len(S), 'NaN', int(S[['speed_kmh', 'flow_vph']].isna().sum().sum()))
log(f'peak RSS {peak_gb():.2f} GB')
if SMOKE:
    assert peak_gb() < 25, f'peak RSS {peak_gb():.2f} GB >= 25 GB'
    log('SMOKE gates OK: features', V12_NFEAT + N_FD, 'no all-NaN FD columns, peak RSS < 25 GB')
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
