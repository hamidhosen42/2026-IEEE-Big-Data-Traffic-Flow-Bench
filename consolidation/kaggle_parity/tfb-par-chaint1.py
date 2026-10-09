# ======================================================================================================================
# GENERATED FILE - DO NOT EDIT. Built by consolidation/bundle.py from the tfb package
#   tfb 0.1.0-m1 | modules tree sha1 391b08e76040726ad078c2fc2f85c10d40f7e4b4 | generated 2026-10-10 00:21
# Parity kernel tfb-par-chaint1 (re-push (fix), ~45 min on Kaggle CPU): production wiring of the Task 1 / Task 4 rows: t4 skeleton -> t1_mix -> t1_recon; state + path_flow EXACT vs tfb-j4
# Stages ['t4', 't1_mix', 't1_recon'] | pick A2 | recipe full | overrides {}
# Kernel sources to attach (32, plus the competition data):
#   hosen42/tfb-t1v11a hosen42/tfb-t1v11b hosen42/tfb-t1v12a hosen42/tfb-t1v12b hosen42/tfb-t1v12s2a
#   hosen42/tfb-t1v12s2b hosen42/tfb-t1v12s3a hosen42/tfb-t1v12s3b hosen42/tfb-t1v12s4a hosen42/tfb-t1v12s4b
#   hosen42/tfb-t1v12s5a hosen42/tfb-t1v12s5b hosen42/tfb-t1v12s6a hosen42/tfb-t1v12s6b hosen42/tfb-t1v12s7a
#   hosen42/tfb-t1v12s7b hosen42/tfb-t1v13a hosen42/tfb-t1v13b hosen42/tfb-t1v13s10a hosen42/tfb-t1v13s10b
#   hosen42/tfb-t1v13s11a hosen42/tfb-t1v13s11b hosen42/tfb-t1darka hosen42/tfb-t1darkb hosen42/tfb-t1rampa
#   hosen42/tfb-t1rampb hosen42/tfb-t1ramp2a hosen42/tfb-t1ramp2b hosen42/tfb-t1bspeca hosen42/tfb-t1bspecb
#   hosen42/tfb-t1bspecc hosen42/tfb-j4
# Push: python kaggle_jobs/kjob.py push tfb-par-chaint1 consolidation/kaggle_parity/tfb-par-chaint1.py hosen42/tfb-t1v11a hosen42/tfb-t1v11b hosen42/tfb-t1v12a hosen42/tfb-t1v12b hosen42/tfb-t1v12s2a hosen42/tfb-t1v12s2b hosen42/tfb-t1v12s3a hosen42/tfb-t1v12s3b hosen42/tfb-t1v12s4a hosen42/tfb-t1v12s4b hosen42/tfb-t1v12s5a hosen42/tfb-t1v12s5b hosen42/tfb-t1v12s6a hosen42/tfb-t1v12s6b hosen42/tfb-t1v12s7a hosen42/tfb-t1v12s7b hosen42/tfb-t1v13a hosen42/tfb-t1v13b hosen42/tfb-t1v13s10a hosen42/tfb-t1v13s10b hosen42/tfb-t1v13s11a hosen42/tfb-t1v13s11b hosen42/tfb-t1darka hosen42/tfb-t1darkb hosen42/tfb-t1rampa hosen42/tfb-t1rampb hosen42/tfb-t1ramp2a hosen42/tfb-t1ramp2b hosen42/tfb-t1bspeca hosen42/tfb-t1bspecb hosen42/tfb-t1bspecc hosen42/tfb-j4
# Output: PARITY lines + PARITY SUMMARY in the log; /kaggle/working/parity_<slug>.json; stage outputs in /kaggle/working/tfb_work/
# ======================================================================================================================
import glob, json, os, sys, tempfile, time

SPEC = json.loads(r'''{"slug": "tfb-par-chaint1", "priority": "re-push (fix)", "minutes": 45, "stages": ["t4", "t1_mix", "t1_recon"], "pick": "A2", "recipe": "full", "overrides": {}, "desc": "production wiring of the Task 1 / Task 4 rows: t4 skeleton -> t1_mix -> t1_recon; state + path_flow EXACT vs tfb-j4", "sources": ["tfb-t1v11a", "tfb-t1v11b", "tfb-t1v12a", "tfb-t1v12b", "tfb-t1v12s2a", "tfb-t1v12s2b", "tfb-t1v12s3a", "tfb-t1v12s3b", "tfb-t1v12s4a", "tfb-t1v12s4b", "tfb-t1v12s5a", "tfb-t1v12s5b", "tfb-t1v12s6a", "tfb-t1v12s6b", "tfb-t1v12s7a", "tfb-t1v12s7b", "tfb-t1v13a", "tfb-t1v13b", "tfb-t1v13s10a", "tfb-t1v13s10b", "tfb-t1v13s11a", "tfb-t1v13s11b", "tfb-t1darka", "tfb-t1darkb", "tfb-t1rampa", "tfb-t1rampb", "tfb-t1ramp2a", "tfb-t1ramp2b", "tfb-t1bspeca", "tfb-t1bspecb", "tfb-t1bspecc", "tfb-j4"], "checks": [{"type": "submission", "ours": ["tfb-i4", "submission_FDRTV.csv"], "orig": ["tfb-j4", "submission_J4.csv"], "tasks": ["state", "odme"], "label": "chaint1_vs_tfb-j4"}], "tfb_version": "0.1.0-m1", "tree_sha1": "391b08e76040726ad078c2fc2f85c10d40f7e4b4", "generated": "2026-10-10 00:21"}''')

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

Picks
  A2 = tfb-j4 (selected candidate A): tfb-i4 + April ongoing rows of tfb-t2ogpvogf (pvogfv).
  B2 = tfb-j3 (hedge B): tfb-i4 + April onset rows of tfb-t2osp (base) + April ongoing rows of tfb-t2ogveto (veto10).
Both share every state, validation-queue and odme row.
"""

PANELS_ALL = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
              'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
PANELS_T2 = PANELS_ALL[:8]                       # D12_I405_N/S have no Task 2 windows

PICKS = ('A2', 'B2')
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


def member_parts(member):
    """kernel parts that hold one member's state_submission_FTA.csv (halves for memory; bspec has a third part)."""
    return (f'{member}a', f'{member}b', f'{member}c') if member == 'tfb-t1bspec' else (f'{member}a', f'{member}b')


def t1_member_kernels(recipe):
    bag = T1_BAGS[recipe]
    out = []
    for m in bag['old_fta'] + bag['fd'] + tuple(bag['dark'].split('+')):
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
PICK_KERNEL = {'A2': ('tfb-j4', 'J4'), 'B2': ('tfb-j3', 'J3')}
ASSEMBLE = dict(base=('tfb-i4', 'submission_FDRTV.csv'), require_full_cover=True)

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
}


def t1_part_panels(member, part):
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
                train_day_mod5_keep=ov.get('train_day_mod5_keep'), threads=int(threads))


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
    'assemble':    dict(module='assemble',  kernel=None,        outputs=None, m=1),          # kernel / output from PICK_KERNEL
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
LIGHT_STAGES = ['t4', 'onset_sp', 'onset_pv', 't1_mix', 't1_recon', 'assemble']


def plan(pick, recipe):
    """stage order of a full run (milestone 2 stages included). t4 precedes t1_mix: it writes the base skeleton."""
    t2 = ['onset_sp'] + (['onset_pv'] if pick == 'A2' else []) + ['ongoing_val']
    if pick == 'A2':
        t2 += (['ongoing_pvaf'] if ONGOING['need_pvo'] else []) + ['ongoing_pvogf']
    else:
        t2 += ['ogfeat_a', 'ogfeat_b', 'ogfeat_comb', 'ogveto']
    return ['pseudo', 't1_members', 't4'] + t2 + ['t1_mix', 't1_recon', 'assemble']   # 't1_members' -> expand()


def is_member_stage(stage):
    return stage.startswith('t1m:')


def stage_module(stage):
    if is_member_stage(stage):
        return 't1_' + T1_MEMBERS[stage[4:-1]]['family']
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
    if stage == 'assemble':
        kern, name = PICK_KERNEL[pick]
        return dict(ASSEMBLE, parts=[list(p) for p in QUEUE_PARTS[pick]], out_name=name, pick=pick)
    if stage == 't4':
        return dict(T4)
    if stage in ('onset_sp', 'onset_pv'):
        return dict(ONSET)
    if stage == 'pseudo':
        return dict(expected={k: list(v) for k, v in PSEUDO_EXPECTED.items()})
    if stage == 'ongoing_pvogf':
        return dict(ref_v22=list(ONGOING_REFS['ref_v22']), veto_g=ONGOING['veto_g'])
    if stage == 'ogfeat_a':
        return dict(part='a', tag='v27a')
    if stage == 'ogfeat_b':
        return dict(part='b', tag='v27b')
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
    if stage == 't1_recon':
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


def expand(stages, recipe):
    """replace 't1_members' by the member-part stages of the recipe."""
    out = []
    for st in stages:
        out += [f't1m:{k}' for k in t1_member_kernels(recipe)] if st == 't1_members' else [st]
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
        module = C.stage_module(stage)
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
        return C.expand(C.plan(pick, recipe), recipe)
    if arg == 'light':
        return [s for s in C.LIGHT_STAGES if s in C.plan(pick, recipe)]
    out = C.expand([s.strip() for s in arg.split(',') if s.strip()], recipe)
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
            _line(check, f'queue.{s}.{c}', 'EXACT' if not g.d.any() else 'FAIL', rows=len(g), n_diff=int(g.d.sum()),
                  windows_diff=len(jw), min_jaccard=f'{min(jw):.4f}' if jw else '1', pos_ours=int(g.a.sum()), pos_orig=int(g.b.sum()))


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
MODULES['tfb/stages/t4_nnls.py'] = '''"""Assemble the upload file: Task 1 + Task 2 from upstream kernels, Task 4 solved here, joined on submission_key.

Task 4 = the official prior-regularised NNLS on each split's OWN released counts and OWN weak prior
(lambda = 0.05), which is what the base notebook ships.
Hard gates (the kernel fails rather than writing a bad file): exact row count and id order versus
sample_submission.csv, zero unmatched keys in every task, no NaN/inf, queue in {0,1}, non-negative
values, and fewer than 20% of state flows below 50 vph in every panel x regime.
"""
import glob, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-final.py (notebook) cell 7; patched lines end in [TFB]
SMOKE = False  # [TFB] cell 1 of tfb-final; strict gates always (the skeleton must match every odme key row)
# [TFB] In the entry script this stage writes the SKELETON submission: odme rows = the NNLS below (verbatim), state and queue
# [TFB] rows 0. t1_mix fills the state rows, assemble fills the queue rows and re-applies the state hard gates.

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = f'{TFB.out}/submission.csv'  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S',
          'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
KEYS = {'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
        'queue': ['window_id', 'timestamp', 'link_id'], 'odme': ['panel', 'departure_time', 'path_id']}
VALUES = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def find(name):
    return TFB.kfile('tfb-final', name)  # [TFB] unused in the entry script (no upstream state / queue tables)


# ---------------- Task 4
rows = []
for panel in PANELS:
    net = f'{REL}/corridors/{panel}/network'
    paths = pd.read_csv(f'{net}/path_set.csv', dtype={'path_id': str})
    inc = pd.read_csv(f'{net}/path_link_incidence.csv', dtype=str)
    pids = paths.path_id.tolist(); lids = inc.link_id.drop_duplicates().tolist()
    A = np.zeros((len(lids), len(pids)))
    A[inc.link_id.map({l: i for i, l in enumerate(lids)}), inc.path_id.map({p: i for i, p in enumerate(pids)})] = 1
    for split in ('validation', 'private'):
        c = pd.read_csv(f'{REL}/task4/{panel}/{split}/synthetic_link_counts.csv', dtype={'link_id': str})
        pr = pd.read_csv(f'{REL}/task4/{panel}/{split}/synthetic_weak_prior.csv', dtype={'path_id': str})
        b = pr.set_index('path_id').reindex(pids).path_flow.fillna(0).to_numpy()
        meas = np.array([l in set(c.link_id) for l in lids])
        cnt = c.set_index('link_id').reindex(np.array(lids)[meas])['count'].to_numpy()
        lam = 0.05
        f, _ = nnls(np.vstack([A[meas], np.sqrt(lam) * np.eye(len(pids))]), np.r_[cnt, np.sqrt(lam) * b])
        s_link = 1 - np.abs(A[meas] @ f - cnt).sum() / cnt.sum()
        dep = str(pr.departure_time.iloc[0])
        rows.append(pd.DataFrame({'panel': panel, 'departure_time': dep, 'path_id': pids, 'path_flow': f}))
        log(panel, split, 'paths', len(pids), 'S_link(own counts)', round(s_link, 6), 'dev/prior', round(np.abs(f - b).sum() / b.sum(), 4))
odme = pd.concat(rows, ignore_index=True)
odme.to_csv(f'{TFB.out}/odme_path_flow.csv', index=False)  # [TFB] the Task 4 table itself (raw NNLS output; parity / reuse)
# [TFB] pandas' default CSV float parser is not correctly rounded, so one to_csv + read_csv cycle g() is not idempotent:
# [TFB] some values move by an ulp on each of the first cycles. In the Kaggle chain path_flow went through 26+ cycles
# [TFB] (tfb-final -> ... -> tfb-j4) and ended on fixed points of g (tfb-i4 asserts g(x) == x). Writing that fixed point
# [TFB] here makes every later cycle of the entry script a no-op and reproduces the chain's values.
import io  # [TFB]


def _csv_cycle(v):  # [TFB] one write + read with pandas defaults, exactly as every swap kernel did
    return pd.read_csv(io.StringIO(pd.DataFrame({'path_flow': v}).to_csv(index=False)))['path_flow'].to_numpy()  # [TFB]


_pf = odme.path_flow.to_numpy(np.float64)  # [TFB]
_first = _csv_cycle(_pf)  # [TFB] = what tfb-final's own submission.csv parses to
for _it in range(1, 101):  # [TFB]
    _nx = _csv_cycle(_pf)  # [TFB]
    if np.array_equal(_nx, _pf):  # [TFB]
        break  # [TFB]
    _pf = _nx  # [TFB]
assert np.array_equal(_csv_cycle(_pf), _pf), f'path_flow has no CSV fixed point after {_it} cycles ({int((_csv_cycle(_pf) != _pf).sum())} values)'  # [TFB]
log('path_flow CSV fixed point after', _it - 1, 'cycles | values moved vs the first parse (tfb-final file):', int((_pf != _first).sum()),  # [TFB]
    '| max |d|', float(np.abs(_pf - _first).max()))  # [TFB]
odme['path_flow'] = _pf  # [TFB]

tables = {'odme': odme}  # [TFB] state / queue tables come from later stages; their rows stay 0 in the skeleton
for task, fr in tables.items():
    fr = fr[KEYS[task] + VALUES[task]].copy()
    for col in KEYS[task]:
        fr[col] = pd.to_datetime(fr[col], utc=True) if col == 'timestamp' else fr[col].astype(str)
    assert not fr.duplicated(KEYS[task]).any(), f'{task}: duplicate keys'
    tables[task] = fr.set_index(KEYS[task])
    log(task, 'rows', len(fr))

# ---------------- merge on submission_key (streamed, template order)
sample_ids = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
parts, gaps = [], {t: 0 for t in tables}
for chunk in pd.read_csv(f'{REL}/submission_key.csv', chunksize=1_000_000, dtype=str):
    chunk['timestamp'] = pd.to_datetime(chunk.timestamp, utc=True, errors='coerce')
    out = pd.DataFrame({'submission_id': chunk.submission_id.astype('int64').to_numpy(), 'task': chunk.task.to_numpy()})
    for col in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'):
        out[col] = 0.0
    for task, table in tables.items():
        m = (out.task == task).to_numpy()
        if not m.any():
            continue
        found = table.reindex(pd.MultiIndex.from_frame(chunk.loc[m, KEYS[task]]))
        for col in VALUES[task]:
            v = found[col].to_numpy(); gaps[task] += int(np.isnan(v).sum())
            out.loc[m, col] = np.nan_to_num(v)
    parts.append(out)
sub = pd.concat(parts, ignore_index=True)
log('unmatched keys per task', gaps)

# ---------------- hard gates
if SMOKE:
    print('SMOKE: unmatched keys expected (panel/day subsets):', gaps)
else:
    assert all(v == 0 for v in gaps.values()), f'unmatched keys: {gaps}'
assert len(sub) == len(sample_ids) and (sub.submission_id.to_numpy() == sample_ids).all(), 'row set/order differs from sample_submission'
num = sub[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']]
assert np.isfinite(num.to_numpy()).all(), 'NaN/inf present'
assert (num >= 0).all().all(), 'negative values'
assert sub.loc[sub.task == 'queue', 'queue_pred'].isin([0, 1]).all(), 'non-binary queue_pred'
# [TFB] the empty-road guard on state flows (< 50 vph share <= 20 % per panel x regime) runs in the assemble stage
sub['queue_pred'] = sub.queue_pred.astype(int)
sub.to_csv(OUT, index=False)
log('WROTE', OUT, len(sub), 'rows')
print(sub.groupby('task')[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].describe().T.round(3).to_string())
'''
MODULES['tfb/stages/t1_mix.py'] = '''"""t1dark swap = T1 MIX swap + BLACKOUT-ROW override (raw, BEFORE FD recon/TV).
T1 rows = W_FD * mean(FD-feature members v13) + (1-W_FD) * mean(old FTA bag) (= swap_t1mix.py); then on the blackout rows
(state rows whose timestamp lies in (origin, origin + 90 min] of an official Task 2 window of that panel, validation or
private; about 2% of the Task 1 targets) T1 = (1 - W_DARK) * T1 + W_DARK * t1dark member (kernels <DARK_MEMBER>a/b =
kaggle_jobs/t1_t1dark_member_a/b.py, trained with simulated-blackout transductive rows). Regular rows are unchanged.
Previous notes (swap_t1mix.py):
T1 MIX swap (raw, BEFORE FD recon/TV): T1 rows = W_FD * mean(FD-feature members v13) + (1-W_FD) * mean(old FTA bag).
Queue and Task 4 rows come from the base file unchanged. The output is meant to be passed through swap_fdrecon (DO_RECON+DO_TV)."""
import glob, os, numpy as np, pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-i4d.py; patched lines end in [TFB]
CFG = TFB.cfg  # [TFB] constants of the recipe (tfb/config.py T1_MIX, T1_BAGS)
BASE_KERNEL, BASE_FILE, OUT_NAME = CFG['base'][0], CFG['base'][1], CFG['out_name']  # [TFB]
W_FD = CFG['w_fd']  # [TFB]
W_DARK = CFG['w_dark']  # [TFB]
DARK_MEMBER = CFG['dark_member']  # [TFB]
FD_MEMBERS = tuple(CFG['fd_members'])  # [TFB]
MEMBERS = tuple(CFG['members'])  # [TFB]
REL = TFB.rel  # [TFB]
base = pd.read_csv(TFB.kfile(BASE_KERNEL, BASE_FILE))  # [TFB]
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
ks = key[key.task == 'state'][['submission_id', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].copy()
ks['timestamp'] = pd.to_datetime(ks.timestamp, utc=True)
iss = (base.task == 'state').to_numpy(); sid = base.loc[iss, 'submission_id'].to_numpy()


def load(kernels, fname):
    st = pd.concat([pd.read_csv(TFB.kfile(k, fname), dtype={'station_id': str, 'link_id': str})  # [TFB]
                    for k in kernels], ignore_index=True)
    st['timestamp'] = pd.to_datetime(st.timestamp, utc=True)
    m = ks.merge(st, on=['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'], how='left', validate='one_to_one')
    assert m.speed_kmh.notna().all() and m.flow_vph.notna().all(), 'unmatched state keys'
    idx = m.submission_id.astype('int64').to_numpy()
    return (pd.Series(m.speed_kmh.to_numpy(), index=idx).reindex(sid).to_numpy(),
            pd.Series(m.flow_vph.to_numpy(), index=idx).reindex(sid).to_numpy())


M = [load((f'{k}a', f'{k}b'), 'state_submission_FTA.csv') for k in MEMBERS]
MF = [load((f'{k}a', f'{k}b'), 'state_submission_FTA.csv') for k in FD_MEMBERS]
print('old members', len(M), '| FD members', len(MF), FD_MEMBERS, '| W_FD', W_FD)
vo, qo = np.mean([m[0] for m in M], 0), np.mean([m[1] for m in M], 0)
vf_, qf_ = np.mean([m[0] for m in MF], 0), np.mean([m[1] for m in MF], 0)
print('rms speed old-bag vs FD-bag', round(float(np.sqrt(np.mean((vo - vf_) ** 2))), 3), '| flow', round(float(np.sqrt(np.mean((qo - qf_) ** 2))), 2))
print('info: base (post FDR/TV) vs old bag rms speed', round(float(np.sqrt(np.mean((base.loc[iss, 'speed_kmh'].to_numpy() - vo) ** 2))), 3))
out = base.copy()
out.loc[iss, 'speed_kmh'] = W_FD * vf_ + (1 - W_FD) * vo; out.loc[iss, 'flow_vph'] = W_FD * qf_ + (1 - W_FD) * qo
# ---- t1dark: blackout rows (timestamp in (origin, origin + 90 min] of an official window of the panel) <- dark member
def _parts(k):                                      # a, b and an optional part c (panel split for memory)
    return tuple(x for x in (f'{k}a', f'{k}b', f'{k}c') if x != f'{k}c' or TFB.kglob(x, 'state_submission_FTA.csv'))  # [TFB]


_D = [load(_parts(k), 'state_submission_FTA.csv') for k in DARK_MEMBER.split('+')]   # '+'-joined = mean of members
vd, qd = np.mean([x[0] for x in _D], 0), np.mean([x[1] for x in _D], 0)
print('dark members', DARK_MEMBER.split('+'))
kk = ks.set_index(ks.submission_id.astype('int64')).reindex(sid)
assert kk.panel.notna().all(), 'state rows without a key'
inb = np.zeros(len(sid), bool)
for panel in kk.panel.unique():
    pm = (kk.panel == panel).to_numpy(); tsp = kk.timestamp[pm]; hit = np.zeros(int(pm.sum()), bool)
    for split in ('validation', 'private'):
        wi_p = f'{REL}/task2/{panel}/{split}/window_index.csv'
        if os.path.exists(wi_p):
            for o in pd.to_datetime(pd.read_csv(wi_p).forecast_origin, utc=True):
                hit |= ((tsp > o) & (tsp <= o + pd.Timedelta(minutes=90))).to_numpy()
    inb[np.flatnonzero(pm)[hit]] = True
assert 0.005 < inb.mean() < 0.05, f'blackout share {inb.mean():.4f} outside the expected range'
v_mix = out.loc[iss, 'speed_kmh'].to_numpy(); q_mix = out.loc[iss, 'flow_vph'].to_numpy()
print('t1dark blackout rows', int(inb.sum()), 'share of state rows', round(float(inb.mean()), 4), '| W_DARK', W_DARK,
      '| rms dark vs mix on blackout rows: speed', round(float(np.sqrt(np.mean((vd[inb] - v_mix[inb]) ** 2))), 3),
      'flow', round(float(np.sqrt(np.mean((qd[inb] - q_mix[inb]) ** 2))), 2),
      '| regular rows: speed', round(float(np.sqrt(np.mean((vd[~inb] - v_mix[~inb]) ** 2))), 3))
out.loc[iss, 'speed_kmh'] = np.where(inb, (1 - W_DARK) * v_mix + W_DARK * vd, v_mix)
out.loc[iss, 'flow_vph'] = np.where(inb, (1 - W_DARK) * q_mix + W_DARK * qd, q_mix)
isq = (base.task == 'queue').to_numpy()
assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
out['queue_pred'] = out.queue_pred.astype(int)
out.to_csv(f'{TFB.out}/submission_{OUT_NAME}.csv', index=False)  # [TFB]
print(OUT_NAME, 'OK', len(out))
'''
MODULES['tfb/stages/t1_recon.py'] = '''"""FDRECON / TV swap (Task 1/3 post-processing on the exact base file; no training). Derived from the local draft
scratchpad/t1loc/swap_fdrecon_tv.py with the data loading, output naming and hard gates of swap_combo15.py.

Step R (FD-gated reconciliation, DO_RECON): at state rows with v < GATE*vf, move the per-lane density q/v onto the released congested
FD branch implied by the predicted speed, k_FD(v) = w*kj/(v+w), w = cap/(kj-kc) (all per lane). FD parameters are looked up by
(panel, link_id) by AVERAGING the station rows of fd_parameters.csv per link, then kc = cap/vf and kj = max(kj, 1.05 kc), exactly like
repo/src/task3/score_task3.py network_parameters(). r = clip(kFD/k, e^-0.5, e^0.5); v' = clip(v r^-0.75, 3, 1.05 vf); q' = max(q r^0.25, 20).
Step T (1-D TV, DO_TV): Hisernberg-style TV smoothing of per-lane density inside every maximal temporal run of target cells per station
(a run breaks where consecutive target timestamps are more than 5 min apart; no UTC-date grouping). Free cells tau 0.0075 / anchor 0.001,
gate cells tau 0.02 / anchor 0.005, each scaled by the run's mean density, 300 dual iterations; anchors = released eligible cells of the
same split at t0-5min / tend+5min. Gate cells split the density change 0.75 to speed, free cells put it all on flow.
Panel exclusion rule (fixed before any LB result): on released eligible congested cells (v < 0.6 vf) of mainline_states_masked
(validation + private pooled), rel = (q/lanes/v - kFD(v))/kFD(v); a panel is EXCLUDED from R and T when |median(rel)| > 2%, or the IQR
is not inside [-5%, +5%], or it has fewer than EXCL_MIN_N such cells.
Outputs (one run): submission_FDR.csv (R only), submission_FDRTV.csv (R then T), submission_TV.csv (T only).
Local evidence (scratchpad, D12_I405_S, real train targets, 2-member bag): R dJ +0.0013, R+T +0.0017 (S_state +0.001, proxy LWR +0.013).
Queue and Task 4 rows untouched (asserted byte-identical)."""
import glob, re, numpy as np, pandas as pd
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-i4.py; patched lines end in [TFB]
CFG = TFB.cfg  # [TFB] constants of the recipe (tfb/config.py RECON_TV)

SMOKE = False                                   # True: diagnostics run on COMBO15, outputs prefixed smoke_ (never submit them)
DO_RECON, DO_TV = CFG['do_recon'], CFG['do_tv']  # [TFB]
BASE_KERNEL, BASE_FILE = CFG['base'][0], CFG['base'][1]  # [TFB]
if SMOKE:
    BASE_KERNEL, BASE_FILE = 'tfb-combo15', 'submission_COMBO15.csv'
PREFIX = 'smoke_submission' if SMOKE else 'submission'
GATE, A = CFG['gate'], CFG['a']  # [TFB]
LOGR_MAX = CFG['logr_max']  # [TFB]
TV_FREE, TV_FREE_A, TV_GATE, TV_GATE_A, TV_ITERS = CFG['tv_free'], CFG['tv_free_a'], CFG['tv_gate'], CFG['tv_gate_a'], CFG['tv_iters']  # [TFB]
RUN_GAP_NS = CFG['run_gap_ns']  # [TFB] a TV run breaks where consecutive target timestamps are > 5 min apart
W_RANGE, KJL_RANGE = tuple(CFG['w_range']), tuple(CFG['kjl_range'])  # [TFB] km/h, veh/km/lane
EXCL_MED, EXCL_IQR, EXCL_MIN_N = CFG['excl_med'], CFG['excl_iqr'], CFG['excl_min_n']  # [TFB]
CHANGED_EXPECT = (0.013, 0.022)                 # expected share of state rows changed by R (warning only)
OUT_DIR = TFB.out  # [TFB]
REL = TFB.rel  # [TFB]
BASE_PATH = TFB.kfile(BASE_KERNEL, BASE_FILE)  # [TFB]
print('SMOKE', SMOKE, '| DO_RECON', DO_RECON, '| DO_TV', DO_TV, '| base', BASE_PATH, flush=True)
assert DO_RECON or DO_TV, 'nothing to do'

base = pd.read_csv(BASE_PATH)
key = pd.read_csv(f'{REL}/submission_key.csv', dtype=str)
iss = (base.task == 'state').to_numpy(); sid = base.loc[iss, 'submission_id'].to_numpy().astype('int64')
kcols = [c for c in ('submission_id', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'split') if c in key.columns]
ks = key[key.task == 'state'][kcols].copy()
ks['submission_id'] = ks.submission_id.astype('int64')
assert ks.submission_id.is_unique and len(ks) == len(sid), 'state rows of base and key differ'
ks = ks.set_index('submission_id').loc[sid].reset_index()              # row order = base state rows (positional index 0..N-1)
ks['station_id'] = ks.station_id.astype(str); ks['link_id'] = ks.link_id.astype(str)
ts = pd.to_datetime(ks.timestamp, utc=True)
tsns = ts.dt.tz_convert(None).to_numpy('datetime64[ns]').astype('int64')
mon = ts.dt.strftime('%Y-%m').to_numpy()
month_split = np.where(mon == '2031-03', 'validation', np.where(mon == '2031-04', 'private', 'train'))
SPLIT_MAP = {'validation': 'validation', 'val': 'validation', 'public': 'validation', 'private': 'private', 'test': 'private', 'train': 'train'}
if 'split' in ks.columns:
    sp = ks.split.astype(str).str.strip().str.lower().map(SPLIT_MAP)
    print('split from key column | unmapped', int(sp.isna().sum()), '| agreement with timestamp month', round(float((sp.fillna('?').to_numpy() == month_split).mean()), 5))
    ks['split'] = sp.fillna(pd.Series(month_split, index=ks.index)).to_numpy()
else:
    print('split from timestamp month (no split column in submission_key)')
    ks['split'] = month_split
if 'mask_regime' in ks.columns:
    ks['mask_regime'] = ks.mask_regime.astype(str).str.replace('mask_regime=', '', regex=False)
    print('mask_regime from key:', ks.mask_regime.value_counts().to_dict())
print('state rows per split:', ks.split.value_counts().to_dict(), flush=True)
for (panel, split) in ks[['panel', 'split']].drop_duplicates().itertuples(index=False):
    assert glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked'), f'no released dir for {panel}/{split}'
PANELS = sorted(ks.panel.unique())


# ---------------- FD parameters per (panel, link_id): average the station rows per link, as the scorer does
def fd_params(panel):
    f = pd.read_csv(f'{REL}/corridors/{panel}/network/fd_parameters.csv', dtype={'link_id': str})
    have = [c for c in ('free_speed_kmh', 'capacity_vph', 'lanes', 'k_jam') if c in f.columns]
    assert {'free_speed_kmh', 'capacity_vph', 'lanes'} <= set(have), f'{panel}: fd_parameters.csv lacks columns, has {list(f.columns)}'
    m = f.groupby('link_id', as_index=False)[have].mean()
    m = m.merge(f.groupby('link_id').size().rename('n_rows').reset_index(), on='link_id')
    m['lanes'] = m.lanes.clip(lower=1.0)
    m['critical_density'] = m.capacity_vph / m.free_speed_kmh.clip(lower=1)    # recomputed from the averages, never averaged
    if 'k_jam' not in m.columns:
        m['k_jam'] = np.nan
    m['k_jam'] = m.k_jam.fillna(100.0 * m.lanes)
    m['k_jam'] = np.maximum(m.k_jam, 1.05 * m.critical_density)
    m['capl'] = m.capacity_vph / m.lanes; m['kcl'] = m.critical_density / m.lanes; m['kjl'] = m.k_jam / m.lanes
    m['w'] = m.capl / np.maximum(m.kjl - m.kcl, 1e-6)
    m['panel'] = panel
    return m


FD = pd.concat([fd_params(p) for p in PANELS], ignore_index=True)
unit_fail = []
for panel in PANELS:
    used = set(ks.loc[ks.panel == panel, 'link_id'])
    m = FD[(FD.panel == panel) & FD.link_id.isin(used)]
    print(f'{panel:12s} FD links used {len(m):3d}/{len(used):3d} (multi-station links {int((m.n_rows > 1).sum())}) | per lane: '
          f'cap [{m.capl.min():.0f},{m.capl.max():.0f}] kc [{m.kcl.min():.1f},{m.kcl.max():.1f}] kj [{m.kjl.min():.1f},{m.kjl.max():.1f}] '
          f'| w [{m.w.min():.1f},{m.w.max():.1f}] km/h | vf [{m.free_speed_kmh.min():.1f},{m.free_speed_kmh.max():.1f}]', flush=True)
    bw = m[(m.w < W_RANGE[0]) | (m.w > W_RANGE[1]) | ~np.isfinite(m.w)]
    bk = m[(m.kjl < KJL_RANGE[0]) | (m.kjl > KJL_RANGE[1]) | ~np.isfinite(m.kjl)]
    if len(bw):
        unit_fail.append(f'{panel}: w outside {W_RANGE} km/h on links {bw.link_id.tolist()[:10]} (values {bw.w.round(2).tolist()[:10]})')
    if len(bk):
        unit_fail.append(f'{panel}: kj/lane outside {KJL_RANGE} veh/km on links {bk.link_id.tolist()[:10]} (values {bk.kjl.round(1).tolist()[:10]})')
# links with implausible FD parameters (e.g. per-lane capacity ~7,700 vph -> w > 200 km/h) are EXCLUDED link-wise (left untouched);
# a panel where more than 5% of its links fail still aborts the run.
bad = ((FD.w < W_RANGE[0]) | (FD.w > W_RANGE[1]) | ~np.isfinite(FD.w) | (FD.kjl < KJL_RANGE[0]) | (FD.kjl > KJL_RANGE[1]) | ~np.isfinite(FD.kjl))
share = bad.groupby(FD.panel).mean()
print('links excluded for implausible FD parameters:', int(bad.sum()), '| per panel share:', share[share > 0].round(3).to_dict())
for line in unit_fail:
    print('  excluded ->', line)
if (share > 0.05).any():
    raise RuntimeError('FD UNIT CHECK FAILED (>5% of links in a panel) - not writing any submission: ' + str(share[share > 0.05].to_dict()))
FD = FD[~bad].copy()
print('FD unit checks passed after link-wise exclusion: w in', W_RANGE, 'km/h, kj/lane in', KJL_RANGE, 'veh/km')

p = ks[['panel', 'link_id']].merge(FD, on=['panel', 'link_id'], how='left', validate='many_to_one')
hasfd = p.lanes.notna().to_numpy()
print('state rows without FD parameters (left untouched):', int((~hasfd).sum()))
ln = p.lanes.fillna(1.0).to_numpy(float); vf = p.free_speed_kmh.fillna(1e9).to_numpy(float)
kjl = p.kjl.fillna(0.0).to_numpy(float); w = p.w.fillna(1.0).to_numpy(float)


def kfd_of(v, w_, kj_):
    return w_ * kj_ / (np.clip(v, 1, None) + w_)


# ---------------- released eligible cells per (panel, split) (diagnostic + TV anchors)
REL_CACHE = {}


def released(panel, split):
    if (panel, split) not in REL_CACHE:
        parts = []
        for f in sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*.parquet')):
            d = pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
            d = d[(d.is_score_eligible == 1) & d.speed_kmh.notna() & d.flow_vph.notna()]
            parts.append(d[['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph']])
        if parts:
            d = pd.concat(parts, ignore_index=True)
            d['station_id'] = d.station_id.astype(str); d['link_id'] = d.link_id.astype(str)
            d['ts'] = pd.to_datetime(d.timestamp, utc=True).dt.tz_convert(None).to_numpy('datetime64[ns]').astype('int64')
            d = d.drop(columns='timestamp').merge(FD.loc[FD.panel == panel, ['link_id', 'lanes', 'free_speed_kmh', 'kjl', 'w']], on='link_id', how='left')
            d = d[d.lanes.notna()].drop_duplicates(['station_id', 'ts'])
            d['k'] = d.flow_vph / d.lanes / d.speed_kmh.clip(lower=1)
        else:
            d = pd.DataFrame(columns=['station_id', 'link_id', 'speed_kmh', 'flow_vph', 'ts', 'lanes', 'free_speed_kmh', 'kjl', 'w', 'k'])
        REL_CACHE[(panel, split)] = d
    return REL_CACHE[(panel, split)]


# ---------------- diagnostic + exclusion rule (fixed before any LB result)
EXCLUDED = []
print(f'diagnostic on released eligible congested cells (v < {GATE} vf): rel = (k_obs - kFD(v_obs)) / kFD(v_obs)')
for panel in PANELS:
    pooled = []
    for split in ('validation', 'private'):
        d = released(panel, split)
        if len(d) == 0:
            print(f'  {panel:12s} {split:10s} no released cells'); continue
        g = d.speed_kmh.to_numpy() < GATE * d.free_speed_kmh.to_numpy()
        kf = kfd_of(d.speed_kmh.to_numpy(float), d.w.to_numpy(float), d.kjl.to_numpy(float))
        rel = (d.k.to_numpy(float)[g] - kf[g]) / kf[g]
        pooled.append(rel)
        if len(rel):
            print(f'  {panel:12s} {split:10s} cells {len(d):7d} gate share {g.mean():.4f} n_gate {len(rel):6d} | median {np.median(rel):+.4f} '
                  f'IQR [{np.percentile(rel, 25):+.4f}, {np.percentile(rel, 75):+.4f}]')
        else:
            print(f'  {panel:12s} {split:10s} cells {len(d):7d} gate share 0')
    rel = np.concatenate(pooled) if pooled else np.zeros(0)
    if len(rel) < EXCL_MIN_N:
        EXCLUDED.append(panel); why = f'n_gate {len(rel)} < {EXCL_MIN_N}'
    else:
        med, q25, q75 = np.median(rel), np.percentile(rel, 25), np.percentile(rel, 75)
        bad = abs(med) > EXCL_MED or q25 < -EXCL_IQR or q75 > EXCL_IQR
        why = f'pooled n {len(rel)} median {med:+.4f} IQR [{q25:+.4f}, {q75:+.4f}]'
        if bad:
            EXCLUDED.append(panel)
    print(f'  {panel:12s} -> {"EXCLUDED" if panel in EXCLUDED else "kept"} ({why})', flush=True)
print('EXCLUDED panels:', EXCLUDED)
APPLY = hasfd & ~ks.panel.isin(EXCLUDED).to_numpy()
print('state rows eligible for R/T:', int(APPLY.sum()), '/', len(APPLY))
if not APPLY.any():
    print('WARNING: every panel excluded (or no FD) - all outputs equal the base; do NOT submit them')

v0 = base.loc[iss, 'speed_kmh'].to_numpy(float); q0 = base.loc[iss, 'flow_vph'].to_numpy(float)
vf_ok = APPLY & np.isfinite(v0) & np.isfinite(q0)


# ---------------- step R: FD-gated reconciliation
def recon(v, q):
    g = vf_ok & (v < GATE * vf)
    k = q / ln / np.clip(v, 1, None)
    r = np.where(g, kfd_of(v, w, kjl) / np.maximum(k, 1e-9), 1.0)
    r = np.exp(np.clip(np.log(np.maximum(r, 1e-9)), -LOGR_MAX, LOGR_MAX))
    vn = np.where(g, np.clip(v * r ** (-A), 3, 1.05 * vf), v)
    qn = np.where(g, np.maximum(q * r ** (1 - A), 20.0), q)
    assert (vn[~g] == v[~g]).all() and (qn[~g] == q[~g]).all(), 'R changed a non-gate row'
    print(f'R: gate share {g.mean():.4f} | median |log r| in gate {np.median(np.abs(np.log(r[g]))) if g.any() else 0:.4f}', flush=True)
    return vn, qn


# ---------------- step T: TV inside temporal runs of target cells
def tv_solve(kk, aL, aR, lam, lama, iters):
    R, n = kk.shape
    hasL, hasR = np.isfinite(aL), np.isfinite(aR)
    aL0, aR0 = np.where(hasL, aL, 0.0), np.where(hasR, aR, 0.0)
    lamL, lamR = np.where(hasL, lama, 0.0), np.where(hasR, lama, 0.0)
    u = np.zeros((R, max(n - 1, 0))); uL = np.zeros(R); uR = np.zeros(R)

    def prim():
        x = kk.copy()
        if n > 1:
            x[:, :-1] += u; x[:, 1:] -= u
        x[:, 0] -= uL; x[:, -1] -= uR
        return x
    for _ in range(iters):
        x = prim()
        if n > 1:
            u = np.clip(u + 0.2 * (x[:, 1:] - x[:, :-1]), -lam[:, None], lam[:, None])
        uL = np.clip(uL + 0.2 * (x[:, 0] - aL0), -lamL, lamL); uR = np.clip(uR + 0.2 * (x[:, -1] - aR0), -lamR, lamR)
    return prim()


def tv(v, q, tag):
    vn, qn = v.copy(), q.copy()
    K = q / ln / np.clip(v, 1, None)
    G = v < GATE * vf
    for (panel, split), grp in ks[vf_ok].groupby(['panel', 'split'], sort=True):
        o = grp.assign(_t=tsns[grp.index.to_numpy()]).sort_values(['station_id', '_t'], kind='mergesort')
        rows = o.index.to_numpy(); st = o.station_id.to_numpy(); t = o._t.to_numpy()
        same = st[1:] == st[:-1]
        assert not (same & (t[1:] == t[:-1])).any(), f'{panel}/{split}: duplicate (station, timestamp) target rows'
        new = np.ones(len(rows), bool); new[1:] = ~same | (t[1:] - t[:-1] > RUN_GAP_NS)
        starts = np.flatnonzero(new); lens = np.diff(np.append(starts, len(rows)))
        rel = released(panel, split)
        kobs = pd.Series(rel.k.to_numpy(float), index=pd.MultiIndex.from_arrays([rel.station_id.to_numpy(), rel.ts.to_numpy()])) if len(rel) else None

        def anchor(stn, tt):
            if kobs is None:
                return np.full(len(stn), np.nan)
            return kobs.reindex(pd.MultiIndex.from_arrays([stn, tt])).to_numpy(float)
        aLall = anchor(st[starts], t[starts] - RUN_GAP_NS)
        aRall = anchor(st[starts + lens - 1], t[starts + lens - 1] + RUN_GAP_NS)
        nchg = 0
        for n in np.unique(lens):
            sel = lens == n
            idx = rows[starts[sel][:, None] + np.arange(n)[None, :]]
            kk = K[idx]; gg = G[idx]
            mk = np.maximum(kk.mean(1), 1e-6)
            xf = tv_solve(kk, aLall[sel], aRall[sel], TV_FREE * mk, TV_FREE_A * mk, TV_ITERS)
            xg = tv_solve(kk, aLall[sel], aRall[sel], TV_GATE * mk, TV_GATE_A * mk, TV_ITERS)
            kn = np.where(gg, xg, xf)
            rr = np.exp(np.clip(np.log(np.maximum(kn, 1e-9) / np.maximum(kk, 1e-9)), -LOGR_MAX, LOGR_MAX))
            a = np.where(gg, A, 0.0)
            chg = rr != 1.0
            vv, qq = v[idx], q[idx]
            vn[idx] = np.where(chg & gg, np.clip(vv * rr ** (-a), 3, 1.05 * vf[idx]), vv)
            qn[idx] = np.where(chg, np.maximum(qq * rr ** (1 - a), 20.0), qq)
            nchg += int(chg.sum())
        print(f'T[{tag}] {panel:12s} {split:10s} rows {len(rows):7d} runs {len(starts):6d} (max len {lens.max()}) | anchors L {np.isfinite(aLall).mean():.3f} '
              f'R {np.isfinite(aRall).mean():.3f} | rows changed {nchg} | rms dq {np.sqrt(np.mean((qn[rows] - q[rows]) ** 2)):.3f}', flush=True)
    assert (vn[~vf_ok] == v[~vf_ok]).all() and (qn[~vf_ok] == q[~vf_ok]).all(), 'T changed a row outside the applied panels'
    return vn, qn


# ---------------- write + hard gates (swap_combo15 asserts + byte-identical queue/path controls)
sample = pd.read_csv(f'{REL}/sample_submission.csv', usecols=['submission_id']).submission_id.to_numpy()
isq = (base.task == 'queue').to_numpy()
assert np.array_equal(base.queue_pred.to_numpy(), np.round(base.queue_pred.to_numpy())), 'base queue_pred not integer'
flow_low_base = float((q0 < 50).mean())


def write(vn, qn, name):
    chg = (vn != v0) | (qn != q0)
    dv, dq = np.abs(vn - v0)[chg], np.abs(qn - q0)[chg]
    share = float(chg.mean())
    print(f'{name}: state rows changed {share:.4%} ({int(chg.sum())}) | median |dv| {np.median(dv) if len(dv) else 0:.3f} km/h '
          f'| median |dq| {np.median(dq) if len(dq) else 0:.2f} vph | max |dv| {dv.max() if len(dv) else 0:.2f} | max |dq| {dq.max() if len(dq) else 0:.1f}')
    if name == 'FDR' and not (CHANGED_EXPECT[0] <= share <= CHANGED_EXPECT[1]):
        print(f'WARNING: FDR changed share {share:.4%} outside expected {CHANGED_EXPECT}')
    for pn in PANELS:
        m = (ks.panel == pn).to_numpy()
        print(f'   {pn:12s} changed {chg[m].mean():.4%}', end='')
    print()
    out = base.copy()
    out.loc[iss, 'speed_kmh'] = vn; out.loc[iss, 'flow_vph'] = qn
    assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()
    assert np.array_equal(out.queue_pred.to_numpy(), base.queue_pred.to_numpy()), 'queue_pred differs from base'
    assert np.array_equal(out.path_flow.to_numpy(), base.path_flow.to_numpy(), equal_nan=True), 'path_flow differs from base'
    assert np.array_equal(out.loc[~iss, ['speed_kmh', 'flow_vph']].to_numpy(), base.loc[~iss, ['speed_kmh', 'flow_vph']].to_numpy(), equal_nan=True)
    assert len(out) == len(sample) and (out.submission_id.to_numpy() == sample).all()
    assert np.isfinite(out[['speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']].to_numpy()).all()
    assert (out[['speed_kmh', 'flow_vph', 'path_flow']] >= 0).all().all()
    assert float((out.loc[iss, 'flow_vph'] < 50).mean()) <= flow_low_base + 0.001
    out['queue_pred'] = out.queue_pred.astype(int)
    path = f'{OUT_DIR}/{PREFIX}_{name}.csv'
    out.to_csv(path, index=False)
    chk = pd.read_csv(path, usecols=['submission_id', 'queue_pred', 'path_flow'])
    assert (chk.submission_id.to_numpy() == sample).all()
    assert np.array_equal(chk.queue_pred.to_numpy(), base.queue_pred.to_numpy().astype(int)), 'written queue_pred differs from base'
    assert np.array_equal(chk.path_flow.to_numpy(), base.path_flow.to_numpy(), equal_nan=True), 'written path_flow differs from base'
    print(path, 'OK', len(out), flush=True)


if DO_RECON:
    v1, q1 = recon(v0, q0)
    write(v1, q1, 'FDR')
    if DO_TV:
        v2, q2 = tv(v1, q1, 'FDRTV')
        write(v2, q2, 'FDRTV')
if DO_TV:
    v3, q3 = tv(v0, q0, 'TV')
    write(v3, q3, 'TV')
print('DONE | excluded', EXCLUDED)
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
