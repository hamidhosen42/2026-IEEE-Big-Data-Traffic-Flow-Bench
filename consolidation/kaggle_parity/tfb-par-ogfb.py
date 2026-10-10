# ======================================================================================================================
# GENERATED FILE - DO NOT EDIT. Built by consolidation/bundle.py from the tfb package
#   tfb 0.1.0-m1 | modules tree sha1 ab91b0a8252fe73dd82392b4f4e4cbbbcd6b2b34 | generated 2026-10-10 11:39
# Parity kernel tfb-par-ogfb (B2, ~397 min on Kaggle CPU): ogfeat seed-1 pair (tfb-t2ogfb, template t2_ogfeat part b): fs2ogf1 exactly + probabilities
# Stages ['ogfeat_b'] | pick B2 | recipe full | overrides {}
# Kernel sources to attach (1, plus the competition data):
#   hosen42/tfb-t2ogfb
# Push: python kaggle_jobs/kjob.py push tfb-par-ogfb consolidation/kaggle_parity/tfb-par-ogfb.py hosen42/tfb-t2ogfb
# Output: PARITY lines + PARITY SUMMARY in the log; /kaggle/working/parity_<slug>.json; stage outputs in /kaggle/working/tfb_work/
# ======================================================================================================================
import glob, json, os, sys, tempfile, time

SPEC = json.loads(r'''{"slug": "tfb-par-ogfb", "priority": "B2", "minutes": 397, "stages": ["ogfeat_b"], "pick": "B2", "recipe": "full", "overrides": {}, "desc": "ogfeat seed-1 pair (tfb-t2ogfb, template t2_ogfeat part b): fs2ogf1 exactly + probabilities", "sources": ["tfb-t2ogfb"], "checks": [{"type": "queue", "ours": ["tfb-t2ogfb", "queue_submission_fs2ogf1.csv"], "orig": ["tfb-t2ogfb", "queue_submission_fs2ogf1.csv"], "label": "ogfb_queue_submission_fs2ogf1"}, {"type": "npz", "ours": ["tfb-t2ogfb", "t2v27b_probs.npz"], "orig": ["tfb-t2ogfb", "t2v27b_probs.npz"], "label": "ogfb_t2v27b_probs"}], "tfb_version": "0.1.0-m1", "tree_sha1": "ab91b0a8252fe73dd82392b4f4e4cbbbcd6b2b34", "generated": "2026-10-10 11:39"}''')

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
MODULES['tfb/stages/t2_ogfeat.py'] = '''"""v27a (track ogfeat, E1 of analysis/LEVERS_2026-10-07.md) = v26 (tfb-t2occ: F03 np511 + id511, all-finite inputs, 13 occupancy
features) + 98 + 5 OGFEAT columns (lab arm all2 = g1b corridor context, g2 per-link kinematics + kx step extrapolations,
g3 demand / supply at the queue ends, g1c corridor flow / speed level and trend, g2b per-link volatility / smoothed state).
The new columns use exactly the inputs of the v26 history features: the all-finite 60-min history [T-60, T), the all-finite
masked slot-T value and the ramp columns of add_ramp (slots < T).  Nothing after T; no tod / dow added.
Feature block copied verbatim from analysis/t2lab/ogfeat_lib.py by analysis/t2lab/ogfeat_make_kernel_v27.py.
Seed split (Kaggle time limit): PART 'a' trains fs_np + fs_id (seed 0) -> fs2ogf, fs_npogf; PART 'b' trains fs_np1 + fs_id1
(seed 1) -> fs2ogf1.  Training rows are identical in both parts; kaggle_jobs/t2_ogfeat_comb.py averages the 4 float32
probability sets in the v26 fs4occ order and decodes -> fs4ogf.  The v20-style CUR rows of v26 (only logged) are not built.
Outputs: queue_submission_<variant>.csv (onset rows 0), t2v27a_probs.npz (float32 ongoing probabilities), t2v27a_footprint.csv,
t2v27a_trainsets.csv.  CONTROL / footprint: vs tfb-t2occ queue_submission_fs4occ.csv (ALARM above 3 % changed validation
ongoing rows).

v26 notes: v26 = v22af (F03 bag: np511 + id511, seeds 0 and 1, ALL-FINITE INPUTS) + 13 OCCUPANCY features.
Occupancy is an independent density reading of the same detector, co-masked with speed / flow (every cell with a speed value
carries it; a blanked cell loses speed, flow and occupancy together).  The features use exactly the inputs of the history
features: the 60-min history [T-60, T) of every cell with a speed value and the masked slot-T value (train: visible exactly
where the simulated slot-T speed is visible, i.e. the same rng draw as v22af; released windows: window_history.parquet and
mainline_states_masked at T).  Per-link calibration from the train split only: ocrit = occupancy that best separates queued
(v <= 0.6 v_free) from free cells; c = median log(o * lanes * v / q) in free flow.  Features: o_last, or_last, or_T, or_mean3,
or_slope, or_max12, or_n-3..or_n+3 (neighbours' or_last), dens_dev  (or = o / ocrit).  Labels / selection / weights unchanged.
Lab (analysis/t2lab/occ_*.py; train OOF on 2,992 official-like windows, 4 folds, rows identical to the all-finite baseline,
feature parity of this file's occ_calib / occ_features vs the lab exact): np+id +0.0028 (CI +0.0013..+0.0041, 6/8 panels),
seed-1 replicate +0.0025 (+0.0011..+0.0039, 7/8), bag of 4 (np, id x seeds 0, 1) +0.0031 (+0.0018..+0.0044, 7/8; same-arm
seed noise about 0.0005); pseudo-March (new seed) bag of 4: offlike_m +0.0017 (-0.0031..+0.0063), runstart_m +0.0009
(-0.0025..+0.0041), faith3_m +0.0071 (+0.0046..+0.0099).  Training cost about +20 % (13 more columns).
Variant fs4occ -> queue_submission_fs4occ.csv.  CONTROL: footprint vs the attached tfb-t2v22af queue_submission_fs4af.csv.
"""
import glob, json, os, time, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from tfb.runtime import ctx as TFB  # [TFB] vendored from consolidation/lineage_src/tfb-t2ogfa.py; patched lines end in [TFB]

T0 = time.time()
REL = TFB.rel  # [TFB]
OUT = TFB.out  # [TFB]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SMOKE = TFB.smoke  # [TFB] False in every real run (True: D7_I10_E + D12_I5_S, few trees)
if SMOKE:
    PANELS = ['D7_I10_E', 'D12_I5_S']
H, F, SPACING = 12, 6, 72
N_BIG = 40 if SMOKE else 1800               # trees of the 511-leaf ongoing models (v20: 1800)
MAX_OG = 300 if SMOKE else 3000             # dense ongoing windows per panel (CUR, and the non-run-start part of F03)
FILL_GAP, PERS_MAX = 36, 0.9                # faithful rule: time fill over interior gaps <= 36 slots, persistence cap 0.9
W_RS, W_DENSE = 1.0, 0.3                    # window weights: run-starts / everything else in F03 and H
TIME_LIMIT_S = 10.5 * 3600                  # v27: a model is skipped if its predicted end exceeds this (Kaggle CPU limit 12 h)
ONGOING = 'queue_ongoing'
ALLFIN = True                                # features from all finite released values (labels / selection: eligible cells)
OCC = True                                   # occupancy features (same history / slot-T inputs as the speed features)
NB = (-3, -2, -1, 1, 2, 3)
OCC_COLS = ['o_last', 'or_last', 'or_T', 'or_mean3', 'or_slope', 'or_max12'] + [f'or_n{k:+d}' for k in NB] + ['dens_dev']
PART = TFB.cfg['part']  # [TFB] 'a' = seed-0 pair (tfb-t2ogfa), 'b' = seed-1 pair (tfb-t2ogfb)
TAG = TFB.cfg['tag']  # [TFB] 'v27a' / 'v27b'
OG_STATIC = ['c_q3', 'c_q6', 'c_q9', 'c_q12', 'c_d3', 'c_d6', 'c_d12', 'c_rel6', 'c_n0', 'c_new3', 'c_clr3', 'c_new6', 'c_clr6', 'c_first', 'c_peak', 'k_run', 'k_ntr', 'k_qf6', 'k_dvr3', 'k_dvr6', 'k_vrmax6', 'k_sdlo0', 'k_sdhi0', 'k_cs0', 'k_nq7_0', 'k_nqhi8_0', 'k_nqlo8_0', 'k_sdlo3', 'k_sdhi3', 'k_cs3', 'k_nq7_3', 'k_nqhi8_3', 'k_nqlo8_3', 'k_sdlo6', 'k_sdhi6', 'k_cs6', 'k_nq7_6', 'k_nqhi8_6', 'k_nqlo8_6', 'k_vlo3', 'k_vhi3', 'k_vlo6', 'k_vhi6', 'e_side', 'e_dist', 'e_size', 'e_rel', 'e_n3', 'e_n6', 'e_x3', 'e_x6', 'e_vr0', 'e_dvr3', 'e_dvr6', 'e_fr0', 'e_in_fr0', 'e_in_dfr3', 'e_out_fr0', 'e_out_dfr3', 'e_in_vr0', 'e_in_dvr3', 'e_out_vr0', 'e_out_dvr3', 'e_ramp_tail', 'e_ramp_head', 'e_off_head', 'e_ramp_in', 'e_surge_head', 'c2_fr0', 'c2_dfr3', 'c2_dfr6', 'c2_dfr11', 'c2_vr0', 'c2_dvr3', 'c2_dvr6', 'c2_dvr12', 'c2_vq0', 'c2_dvq3', 'c2_dvq6', 'c2_frq0', 'c2_dfrq3', 'c2_slow0', 'c2_dslow3', 'k2_vstd', 'k2_vmin', 'k2_vmax', 'k2_z', 'k2_vm5', 'k2_dvm5_3', 'k2_dvm5_6', 'k2_fm5', 'k2_dfm5_3', 'k2_fr_up', 'k2_dfr_up3', 'k2_fr_dn', 'k2_dfr_dn3', 'k2_vr_up', 'k2_vr_dn']
OG_STEP = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']
OG_ADD_PROD = []          # production columns re-admitted to the np / id sets (none)
CAPD = 30.0
BIG = 10 ** 6


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ data
def load_panel(panel):
    net = f'{REL}/corridors/{panel}/network'
    topo = pd.read_csv(f'{net}/lwr_mainline_topology.csv', dtype={'link_id': str}).sort_values('order_index').reset_index(drop=True)
    links = topo.link_id.tolist(); li = {l: i for i, l in enumerate(links)}
    lk = pd.read_csv(f'{net}/links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
    vf = lk.free_speed_kmh.fillna(105).to_numpy(); cut = 0.6 * vf
    cap = lk.capacity_vph.fillna(7200).to_numpy(); lanes = lk.lanes.fillna(4).to_numpy()
    length = lk.length_km.fillna(0.5).to_numpy()
    files = sorted(glob.glob(f'{REL}/corridors/{panel}/train/mainline_states/*/*.parquet'))
    if SMOKE:
        files = files[:90]
    days = [os.path.basename(f)[-18:-8].replace('_', '-') for f in files]
    D, L = len(files), len(links)
    V = np.full((D, 288, L), np.nan, np.float32); Fq = V.copy()
    Vx = V.copy() if ALLFIN else None; Fx = V.copy() if ALLFIN else None       # ineligible cells that carry a value
    O = V.copy() if OCC else None                                          # occupancy of every cell with a speed value
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'] + (['occupancy'] if OCC else []))
        x = x[x.speed_kmh.notna()]
        ts = pd.to_datetime(x.timestamp, utc=True)
        x = x.assign(t=(ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(), l=x.link_id.astype(str).map(li)).dropna(subset=['l'])
        if OCC:
            g = x[x.occupancy.notna()].groupby(['t', 'l']).occupancy.mean().reset_index()
            O[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.occupancy.to_numpy()
        el = (x.is_score_eligible == 1).to_numpy()
        g = x[el].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
        V[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
        Fq[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
        if ALLFIN:
            g = x[~el].groupby(['t', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
            Vx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.speed_kmh.to_numpy()
            Fx[d, g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    E = np.isfinite(V); Q = E & (V <= cut[None, None, :])
    dow = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    ON, OFF = ramp_cube(panel, 'train', li, days)
    P = dict(panel=panel, links=links, li=li, cut=cut, vf=vf, cap=cap, lanes=lanes, length=length,
             days=days, dow=dow, V=V, Fq=Fq, E=E, Q=Q, ON=ON, OFF=OFF)
    if ALLFIN:                                                             # FEATURE cubes: eligible value, else the ineligible value
        P['Vf'] = np.where(E, V, Vx); P['Ff'] = np.where(E, Fq, Fx)
        log(panel, 'all-finite inputs: eligible share', round(float(E.mean()), 4), '| + ineligible valued', round(float(np.isfinite(Vx).mean()), 4))
    if OCC:
        Va = P.get('Vf', V)
        O = np.where(np.isfinite(Va), O, np.nan).astype(np.float32)          # same visibility as the speed feature cube
        P['O'] = O
        P['ocrit'], P['occ_c'], n_est = occ_calib(Va, P.get('Ff', Fq), O, cut, vf, lanes)
        log(panel, 'occupancy: finite share', round(float(np.isfinite(O).mean()), 4), '| speed-cube share', round(float(np.isfinite(Va).mean()), 4),
            '| ocrit estimated links', n_est, 'of', L, 'q10/50/90', np.quantile(P['ocrit'], [0.1, 0.5, 0.9]).round(3).tolist())
    return P


def occ_calib(Va, Fa, O, cut, vf, lanes, min_q=20, min_free=20, min_ff=100):
    """per-link calibration from train cells: ocrit = occupancy threshold that best separates queued (v <= cut) from free cells
    (midpoint of the best gap; links without enough of both: median of the estimated links); c = median log(o * lanes * v / q)
    over free-flow cells (v > 0.85 v_free)."""
    L = O.shape[2]; oc = np.full(L, np.nan); c = np.full(L, np.nan); est = np.zeros(L, bool)
    for l in range(L):
        v = Va[:, :, l]; m = np.isfinite(v) & np.isfinite(O[:, :, l])
        vv, oo, qq = v[m], O[:, :, l][m], Fa[:, :, l][m]
        qd = vv <= cut[l]
        if qd.sum() >= min_q and (~qd).sum() >= min_free:
            s = np.argsort(oo, kind='stable'); os_, qs = oo[s], qd[s]
            err = np.cumsum(qs) + ((~qs).sum() - np.cumsum(~qs)); i = int(np.argmin(err))
            oc[l] = 0.5 * (os_[i] + os_[min(i + 1, len(os_) - 1)]); est[l] = True
        ff = (vv > 0.85 * vf[l]) & np.isfinite(qq) & (qq > 0) & (oo > 0)
        if ff.sum() >= min_ff:
            c[l] = np.median(np.log(oo[ff] * lanes[l] * vv[ff] / qq[ff]))
    oc[~est] = np.nanmedian(oc[est]) if est.any() else 0.3
    c[~np.isfinite(c)] = np.nanmedian(c) if np.isfinite(c).any() else -4.3
    return oc.astype(np.float32), c.astype(np.float32), int(est.sum())


def occ_features(Oh, OT, Vh, Fh, P):
    """13 occupancy features [W, L] from the occupancy history [W, 12, L], the slot-T occupancy [W, L] (NaN = not visible) and
    the speed / flow history of the same cells (density agreement)."""
    lanes = P['lanes']; ocb = P['ocrit'][None, :]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')                                    # all-NaN slices -> NaN (as the lab), no warning spam
        of = ffill_time(Oh)
        olast = np.where(np.isfinite(OT), OT, of[:, -1])
        ok = np.isfinite(Oh) & np.isfinite(Vh) & np.isfinite(Fh) & (Oh > 0) & (Fh > 0) & (Vh > 0)
        lr = np.where(ok, np.log(np.where(ok, Oh * lanes[None, None, :] * Vh / np.where(ok, Fh, 1.0), 1.0)), np.nan)
        raw = dict(oT=OT, olast=olast, omean3=np.nanmean(Oh[:, -3:], 1), ofirst3=np.nanmean(Oh[:, :3], 1), omax12=np.nanmax(Oh, 1),
                   lr3=np.nanmean(lr[:, -3:], 1))
        raw = {k: np.asarray(v, np.float32) for k, v in raw.items()}
        orl = raw['olast'] / ocb
        out = {'o_last': raw['olast'], 'or_last': orl, 'or_T': raw['oT'] / ocb, 'or_mean3': raw['omean3'] / ocb,
               'or_slope': (raw['omean3'] - raw['ofirst3']) / ocb, 'or_max12': raw['omax12'] / ocb}
        for k in NB:
            out[f'or_n{k:+d}'] = shift_links(orl, k)
        out['dens_dev'] = raw['lr3'] - P['occ_c'][None, :]
    return {k: np.asarray(out[k], np.float32) for k in OCC_COLS}


def ramp_cube(panel, split, li, days):
    """released on/off-ramp flows summed onto the attached mainline link: [D, 288, L], NaN = not observed / not attached."""
    net = f'{REL}/corridors/{panel}/network'
    D, L = len(days), len(li)
    ON = np.full((D, 288, L), np.nan, np.float32); OFF = ON.copy()
    amap = pd.read_csv(f'{net}/ramp_attachment_map.csv', dtype=str)
    amap = amap[amap.nearest_mainline_link_id.isin(li)]
    rl = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id.map(li)))
    dpos = {dd: i for i, dd in enumerate(days)}
    for f in sorted(glob.glob(f'{REL}/corridors/{panel}/{split}/ramp_states/**/*.parquet', recursive=True)):
        r = pd.read_parquet(f, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'pct_observed', 'is_score_eligible'])
        r = r[(r.is_score_eligible == 1) & r.flow_vph.notna() & (r.pct_observed >= 75) & r.ramp_link_id.isin(rl)]
        if r.empty:
            continue
        ts = pd.to_datetime(r.timestamp, utc=True)
        r = r.assign(d=ts.dt.strftime('%Y-%m-%d').map(dpos), t=(ts.dt.hour * 12 + ts.dt.minute // 5), l=r.ramp_link_id.map(rl),
                     on=r.ramp_type.str.upper().isin(['OR', 'ON'])).dropna(subset=['d'])
        for flag, A in ((True, ON), (False, OFF)):
            g = r[r.on == flag].groupby(['d', 't', 'l']).flow_vph.sum().reset_index()
            A[g.d.astype(int).to_numpy(), g.t.to_numpy(), g.l.astype(int).to_numpy()] = g.flow_vph.to_numpy()
    return ON, OFF


def add_ramp(Fe, ON, OFF, d, t, cap):
    """ramp features from slots t-3..t-1 (strictly before the forecast origin T)."""
    capl = cap[None, :]
    with np.errstate(all='ignore'):
        on3 = np.nanmean(ON[d[:, None], t[:, None] + np.arange(-3, 0)[None]], 1) / capl
        off3 = np.nanmean(OFF[d[:, None], t[:, None] + np.arange(-3, 0)[None]], 1) / capl
        on12 = np.nanmean(ON[d[:, None], t[:, None] + np.arange(-H, -3)[None]], 1) / capl
    Fe['ramp_on3'] = on3; Fe['ramp_off3'] = off3; Fe['ramp_on_surge'] = on3 - on12
    for nm, a in (('on', on3), ('off', off3)):
        a0 = np.nan_to_num(a)
        Fe[f'ramp_{nm}_pm3'] = sum(shift_links(a0, k) for k in (-3, -2, -1, 1, 2, 3)).astype(np.float32)


# ============================================================================================ kernel-portable block
def og_ffill(a):
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def og_dist(q, direction):
    """distance in links to the nearest True at lower (-1) / higher (+1) index, the link itself included; BIG if none."""
    L = q.shape[-1]; idx = np.arange(L)
    if direction < 0:
        return idx - np.maximum.accumulate(np.where(q, idx, -BIG), axis=-1)
    return np.minimum.accumulate(np.where(q, idx, BIG)[..., ::-1], axis=-1)[..., ::-1] - idx


def og_close(q, gap=2):
    """fill runs of <= gap non-queued links that lie between two queued links."""
    return q | ((og_dist(q, -1) + og_dist(q, 1) - 1) <= gap)


def og_rsum(A, lo, hi):
    """sum and count of the finite values A[w, lo..hi] (inclusive, clipped to the corridor; empty when lo > hi)."""
    W, L = A.shape; ok = np.isfinite(A)
    cs = np.zeros((W, L + 1), np.float64); cs[:, 1:] = np.cumsum(np.where(ok, A, 0.0), 1)
    cn = np.zeros((W, L + 1), np.int64); cn[:, 1:] = np.cumsum(ok, 1)
    lo_ = np.clip(lo, 0, L); hi_ = np.maximum(np.clip(hi + 1, 0, L), lo_)
    return (np.take_along_axis(cs, hi_, 1) - np.take_along_axis(cs, lo_, 1),
            np.take_along_axis(cn, hi_, 1) - np.take_along_axis(cn, lo_, 1))


def og_rmean(A, lo, hi):
    s, n = og_rsum(A, lo, hi)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def og_edges(q):
    """closed queue state and edge distances (corridor ends count as free): qc, dql, dqh (to a queued link, BIG if none),
    dfl, dfh (to a non-queued link)."""
    L = q.shape[-1]; idx = np.arange(L)
    qc = og_close(q)
    dql, dqh = og_dist(qc, -1), og_dist(qc, 1)
    dfl = np.minimum(og_dist(~qc, -1), idx + 1); dfh = np.minimum(og_dist(~qc, 1), L - idx)
    return qc, dql, dqh, dfl, dfh


def og_extra(P, Vh, Fh, VT, Fe):
    """extra ongoing features; Vh, Fh [W, 12, L], VT [W, L] exactly as passed to window_features, Fe = its output after
    add_ramp (only ramp_on3 / ramp_off3 / ramp_on_surge are read).  Returns dict name -> [W, L] float32.
    Time index of the 13-slot array: 12 = T (masked-layer value), 12 - k = T - 5k minutes."""
    with np.errstate(all='ignore'):
        cut = P['cut']; W, _, L = Vh.shape; idx = np.arange(L)[None, :]
        vr = np.concatenate([Vh / cut[None, None, :], (VT / cut[None, :])[:, None, :]], 1)    # same dtype / ops as production vr, vT
        fr = Fh / P['cap'][None, None, :]
        vf = og_ffill(vr); vb = og_ffill(vf[:, ::-1])[:, ::-1]                 # forward fill, then the leading gap backwards
        frf = og_ffill(fr); frb = og_ffill(frf[:, ::-1])[:, ::-1]
        qraw = np.isfinite(vr) & (np.nan_to_num(vr, nan=9.0) <= 1.0)
        Q = np.nan_to_num(vb, nan=9.0) <= 1.0                                   # [W, 13, L] filled queue state
        X = {}
        bc = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], L, 1)
        # ---------------- g1b corridor context
        sh = Q.mean(2)                                                          # [W, 13]
        X['c_q3'] = bc(sh[:, 9]); X['c_q6'] = bc(sh[:, 6]); X['c_q9'] = bc(sh[:, 3]); X['c_q12'] = bc(sh[:, 0])
        X['c_d3'] = bc(sh[:, 12] - sh[:, 9]); X['c_d6'] = bc(sh[:, 12] - sh[:, 6]); X['c_d12'] = bc(sh[:, 12] - sh[:, 0])
        X['c_rel6'] = bc((sh[:, 12] - sh[:, 6]) / (sh[:, 12] + sh[:, 6] + 1e-6))
        X['c_n0'] = bc(Q[:, 12].sum(1))
        X['c_new3'] = bc((Q[:, 12] & ~Q[:, 9]).sum(1)); X['c_clr3'] = bc((~Q[:, 12] & Q[:, 9]).sum(1))
        X['c_new6'] = bc((Q[:, 12] & ~Q[:, 6]).sum(1)); X['c_clr6'] = bc((~Q[:, 12] & Q[:, 6]).sum(1))
        anyq = qraw.any(2)
        X['c_first'] = bc(np.where(anyq.any(1), 12 - np.argmax(anyq, 1), -1))  # slots since the first visible queued cell
        X['c_peak'] = bc(sh[:, 12] / (sh.max(1) + 1e-6))
        # ---------------- g2 per-link kinematics
        same = (Q == Q[:, 12:13])[:, ::-1]
        X['k_run'] = np.where(same.all(1), 13, np.argmin(same, 1))              # slots the link has been in its current state
        X['k_ntr'] = (Q[:, 1:] != Q[:, :-1]).sum(1)
        X['k_qf6'] = Q[:, 7:].mean(1)
        X['k_dvr3'] = vb[:, 12] - vb[:, 9]; X['k_dvr6'] = vb[:, 12] - vb[:, 6]
        X['k_vrmax6'] = np.nanmax(np.where(np.isfinite(vb[:, 7:]), vb[:, 7:], -9.0), 1)
        X['k_vrmax6'] = np.where(X['k_vrmax6'] < -8, np.nan, X['k_vrmax6'])
        sd = {}; E0 = None
        for lag, tag in ((12, '0'), (9, '3'), (6, '6')):
            qc, dql, dqh, dfl, dfh = og_edges(Q[:, lag])
            if lag == 12:
                E0 = (qc, dql, dqh, dfl, dfh)
            sd['lo' + tag] = np.where(qc, np.minimum(dfl, CAPD), -np.minimum(dql, CAPD))   # > 0 inside: links to the tail edge
            sd['hi' + tag] = np.where(qc, np.minimum(dfh, CAPD), -np.minimum(dqh, CAPD))   # > 0 inside: links to the head edge
            X['k_sdlo' + tag] = sd['lo' + tag]; X['k_sdhi' + tag] = sd['hi' + tag]
            X['k_cs' + tag] = np.where(qc, np.minimum(dfl + dfh - 1, 3 * CAPD), 0)
            qf = Q[:, lag].astype(np.float32)
            X['k_nq7_' + tag] = og_rsum(qf, idx - 3, idx + 3)[0]
            X['k_nqhi8_' + tag] = og_rsum(qf, idx + 1, idx + 8)[0]
            X['k_nqlo8_' + tag] = og_rsum(qf, idx - 8, idx - 1)[0]
        X['k_vlo3'] = sd['lo0'] - sd['lo3']; X['k_vhi3'] = sd['hi0'] - sd['hi3']
        X['k_vlo6'] = sd['lo0'] - sd['lo6']; X['k_vhi6'] = sd['hi0'] - sd['hi6']
        # ---------------- g3 demand / supply at the ends of the reference cluster
        qc, dql, dqh, dfl, dfh = E0
        use_hi = ~qc & (dqh <= dql) & (dqh < BIG // 2)
        use_lo = ~qc & ~use_hi & (dql < BIG // 2)
        a_h = np.clip(idx + dqh, 0, L - 1); b_h = a_h + np.take_along_axis(dfh, a_h, 1) - 1
        b_l = np.clip(idx - dql, 0, L - 1); a_l = b_l - np.take_along_axis(dfl, b_l, 1) + 1
        A = np.where(qc, idx - dfl + 1, np.where(use_hi, a_h, a_l)); B = np.where(qc, idx + dfh - 1, np.where(use_hi, b_h, b_l))
        ok = qc | use_hi | use_lo
        size = (B - A + 1).astype(np.float32)
        nanify = lambda v: np.where(ok, v, np.nan)
        X['e_side'] = np.where(qc, 0, np.where(use_hi, 1, np.where(use_lo, -1, np.nan)))
        X['e_dist'] = np.where(qc, 0, np.minimum(np.minimum(dql, dqh), 2 * CAPD))
        X['e_size'] = nanify(np.minimum(size, 3 * CAPD))
        X['e_rel'] = np.where(qc, (idx - A) / np.maximum(size - 1, 1), np.nan)
        q12, q9, q6 = (Q[:, k].astype(np.float32) for k in (12, 9, 6))
        X['e_n3'] = nanify(og_rsum(q9, A, B)[0] / size); X['e_n6'] = nanify(og_rsum(q6, A, B)[0] / size)
        n_now = og_rsum(q12, A - 5, B + 5)[0]
        X['e_x3'] = nanify(og_rsum(q9, A - 5, B + 5)[0] - n_now); X['e_x6'] = nanify(og_rsum(q6, A - 5, B + 5)[0] - n_now)
        v12, v9, v6 = vb[:, 12], vb[:, 9], vb[:, 6]; f11, f8 = frb[:, 11], frb[:, 8]
        m0 = og_rmean(v12, A, B)
        X['e_vr0'] = nanify(m0); X['e_dvr3'] = nanify(m0 - og_rmean(v9, A, B)); X['e_dvr6'] = nanify(m0 - og_rmean(v6, A, B))
        X['e_fr0'] = nanify(og_rmean(f11, A, B))
        i0 = og_rmean(f11, A - 3, A - 1); o0 = og_rmean(f11, B + 1, B + 3)
        X['e_in_fr0'] = nanify(i0); X['e_in_dfr3'] = nanify(i0 - og_rmean(f8, A - 3, A - 1))
        X['e_out_fr0'] = nanify(o0); X['e_out_dfr3'] = nanify(o0 - og_rmean(f8, B + 1, B + 3))
        iv = og_rmean(v12, A - 3, A - 1); ov = og_rmean(v12, B + 1, B + 3)
        X['e_in_vr0'] = nanify(iv); X['e_in_dvr3'] = nanify(iv - og_rmean(v9, A - 3, A - 1))
        X['e_out_vr0'] = nanify(ov); X['e_out_dvr3'] = nanify(ov - og_rmean(v9, B + 1, B + 3))
        ron = np.asarray(Fe['ramp_on3'], np.float32); rof = np.asarray(Fe['ramp_off3'], np.float32)
        rsg = np.asarray(Fe['ramp_on_surge'], np.float32)
        X['e_ramp_tail'] = nanify(og_rsum(ron, A - 3, A + 1)[0]); X['e_ramp_head'] = nanify(og_rsum(ron, B - 1, B + 3)[0])
        X['e_off_head'] = nanify(og_rsum(rof, B - 1, B + 3)[0]); X['e_ramp_in'] = nanify(og_rsum(ron, A, B)[0])
        X['e_surge_head'] = nanify(og_rsum(rsg, B - 1, B + 3)[0])
    return {k: np.asarray(v, np.float32) for k, v in X.items()}


def og_extra2(P, Vh, Fh, VT, Fe):
    """round-2 features (same inputs as og_extra): g1c corridor flow / speed level and trend (c2_*), g2b per-link volatility and
    spatially smoothed speed / flow with trends (k2_*)."""
    with np.errstate(all='ignore'):
        cut = P['cut']; W, _, L = Vh.shape; idx = np.arange(L)[None, :]
        vr = np.concatenate([Vh / cut[None, None, :], (VT / cut[None, :])[:, None, :]], 1)
        fr = Fh / P['cap'][None, None, :]
        vb = og_ffill(og_ffill(vr)[:, ::-1])[:, ::-1]; frb = og_ffill(og_ffill(fr)[:, ::-1])[:, ::-1]
        Q = np.nan_to_num(vb, nan=9.0) <= 1.0
        X = {}
        bc = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], L, 1)
        nm = lambda a, m=None: (np.where(np.isfinite(a) & (True if m is None else m), a, 0.0).sum(1)
                                / np.maximum((np.isfinite(a) & (True if m is None else m)).sum(1), 1))
        # ---------------- g1c corridor level / trend of flow and speed
        f0 = nm(frb[:, 11]); v0 = nm(vb[:, 12]); q12 = Q[:, 12]
        X['c2_fr0'] = bc(f0); X['c2_dfr3'] = bc(f0 - nm(frb[:, 8])); X['c2_dfr6'] = bc(f0 - nm(frb[:, 5])); X['c2_dfr11'] = bc(f0 - nm(frb[:, 0]))
        X['c2_vr0'] = bc(v0); X['c2_dvr3'] = bc(v0 - nm(vb[:, 9])); X['c2_dvr6'] = bc(v0 - nm(vb[:, 6])); X['c2_dvr12'] = bc(v0 - nm(vb[:, 0]))
        vq = nm(vb[:, 12], q12)
        X['c2_vq0'] = bc(vq); X['c2_dvq3'] = bc(vq - nm(vb[:, 9], q12)); X['c2_dvq6'] = bc(vq - nm(vb[:, 6], q12))
        fq = nm(frb[:, 11], q12)
        X['c2_frq0'] = bc(fq); X['c2_dfrq3'] = bc(fq - nm(frb[:, 8], q12))
        slow = lambda a: ((a > 1.0) & (a <= 1.3)).mean(1)
        X['c2_slow0'] = bc(slow(vb[:, 12])); X['c2_dslow3'] = bc(slow(vb[:, 12]) - slow(vb[:, 9]))
        # ---------------- g2b per-link volatility, smoothed speed / flow and trends
        okv = np.isfinite(vr); n = okv.sum(1)
        mu = np.where(okv, vr, 0.0).sum(1) / np.maximum(n, 1)
        sdv = np.sqrt(np.where(okv, (vr - mu[:, None, :]) ** 2, 0.0).sum(1) / np.maximum(n - 1, 1))
        X['k2_vstd'] = np.where(n >= 3, sdv, np.nan)
        X['k2_vmin'] = np.where(n > 0, np.where(okv, vr, 9.0).min(1), np.nan); X['k2_vmax'] = np.where(n > 0, np.where(okv, vr, -9.0).max(1), np.nan)
        X['k2_z'] = (vb[:, 12] - 1.0) / (np.where(n >= 3, sdv, 0.1) + 0.03)
        m0 = og_rmean(vb[:, 12], idx - 2, idx + 2)
        X['k2_vm5'] = m0; X['k2_dvm5_3'] = m0 - og_rmean(vb[:, 9], idx - 2, idx + 2); X['k2_dvm5_6'] = m0 - og_rmean(vb[:, 6], idx - 2, idx + 2)
        g0 = og_rmean(frb[:, 11], idx - 2, idx + 2)
        X['k2_fm5'] = g0; X['k2_dfm5_3'] = g0 - og_rmean(frb[:, 8], idx - 2, idx + 2)
        u0 = og_rmean(frb[:, 11], idx - 6, idx - 1); d0 = og_rmean(frb[:, 11], idx + 1, idx + 6)
        X['k2_fr_up'] = u0; X['k2_dfr_up3'] = u0 - og_rmean(frb[:, 8], idx - 6, idx - 1)
        X['k2_fr_dn'] = d0; X['k2_dfr_dn3'] = d0 - og_rmean(frb[:, 8], idx + 1, idx + 6)
        X['k2_vr_up'] = og_rmean(vb[:, 12], idx - 6, idx - 1); X['k2_vr_dn'] = og_rmean(vb[:, 12], idx + 1, idx + 6)
    return {k: np.asarray(v, np.float32) for k, v in X.items()}


OG_STEP_ALL = ['kx_lo3', 'kx_hi3', 'kx_lo6', 'kx_hi6', 'kx_min3']


def og_step(S, h):
    """step-dependent columns from the static ones (S: name -> array) and the step h (array or scalar, 1..6)."""
    h = np.asarray(h, np.float32)
    lo3 = S['k_sdlo0'] + h / 3.0 * S['k_vlo3']; hi3 = S['k_sdhi0'] + h / 3.0 * S['k_vhi3']
    return {'kx_lo3': lo3, 'kx_hi3': hi3, 'kx_lo6': S['k_sdlo0'] + h / 6.0 * S['k_vlo6'], 'kx_hi6': S['k_sdhi0'] + h / 6.0 * S['k_vhi6'],
            'kx_min3': np.minimum(lo3, hi3)}
# ============================================================================================ end of portable block


def established(Qh):
    return Qh.sum() >= 2 and (Qh.sum(0) >= 2).any()


def iou(truth, pred, mask):
    t, p = truth[mask], pred[mask]; u = (t | p).sum()
    return 1.0 if u == 0 else (t & p).sum() / u


def build_windows(P, global_spacing=True, dense_ongoing=False):
    D = len(P['days']); out = []; taken = []
    for cond in ('queue_onset', 'queue_ongoing'):
        if not global_spacing:
            taken = []
        for d in range(D):
            for t in range(H, 288 - F):
                g = d * 288 + t
                if not (dense_ongoing and cond == 'queue_ongoing') and any(abs(g - x) < SPACING for x in taken[-400:]):
                    continue
                if dense_ongoing and cond == 'queue_ongoing' and t % 3:
                    continue
                Qf = P['Q'][d, t + 1:t + 1 + F]
                if not Qf.any() or P['E'][d, t - H:t].mean() < 0.7:
                    continue
                if (cond == 'queue_ongoing') != established(P['Q'][d, t - H:t]):
                    continue
                if cond == 'queue_ongoing' and iou(Qf, np.repeat(P['Q'][d, t][None], F, 0), P['E'][d, t + 1:t + 1 + F]) > 0.9:
                    continue
                out.append((d, t, cond)); taken.append(g); taken.sort()
    return out


# ------------------------------------------------------------------ faithful ongoing selection (TRAIN windows only; never a feature)
def fill_time(a, maxgap):
    """linear interpolation along time within each day for interior gaps <= maxgap slots ([D, 288, L])."""
    out = a.copy(); Dd, Tt, Ll = a.shape; idx = np.arange(Tt)
    for d in range(Dd):
        okd = np.isfinite(a[d])
        for l in np.where(okd.any(0) & ~okd.all(0))[0]:
            ok = okd[:, l]
            if ok.sum() < 2:
                continue
            y = a[d, :, l]
            yi = np.interp(idx, idx[ok], y[ok])
            prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6)); nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
            m = (~ok) & (nxt - prev - 1 <= maxgap) & (prev >= 0) & (nxt < Tt)
            out[d, m, l] = yi[m]
    return out


def faithful_candidates(P):
    """(cand, pers): [D, 288] faithful ongoing candidate map and filled persistence IoU (NaN outside slots 12..281).
    Uses the train future (time-filled state): offline train-window construction only."""
    Q, E = P['Q'], P['E']; D, _, L = Q.shape
    TS = np.arange(H, 288 - F)
    VF = fill_time(P['V'], FILL_GAP); MF = np.isfinite(VF); QF = MF & (VF <= P['cut'][None, None, :])
    del VF
    cE = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(E.sum(2), 1)], 1)
    cov = (cE[:, TS] - cE[:, TS - H]) / float(H * L)
    cQ = np.concatenate([np.zeros((D, 1, L), np.int16), np.cumsum(Q, 1, dtype=np.int16)], 1)
    est = ((cQ[:, TS] - cQ[:, TS - H]) >= 2).any(2)
    aqf = np.concatenate([np.zeros((D, 1), np.int64), np.cumsum(QF.any(2), 1)], 1)
    hq = (aqf[:, TS + F + 1] - aqf[:, TS + 1]) > 0
    inter = np.zeros((D, len(TS)), np.int32); union = inter.copy()
    S = QF[:, H:288 - F]
    for h in range(1, F + 1):
        Th = QF[:, H + h:288 - F + h]; M = MF[:, H + h:288 - F + h]
        inter += (Th & S & M).sum(2); union += ((Th | S) & M).sum(2)
    pers_ts = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    cand = np.zeros((D, 288), bool); pers = np.full((D, 288), np.nan)
    cand[:, H:288 - F] = (cov >= 0.7) & est & hq & (pers_ts <= PERS_MAX)
    pers[:, H:288 - F] = pers_ts
    return cand, pers


def pq(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return 'n=0'
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return f'n={len(x)} q10/25/50/75/90 ' + '/'.join(f'{v:.3f}' for v in q) + f' share>0.9 {np.mean(x > 0.9):.3f} share>0.8 {np.mean(x > 0.8):.3f}'


def select_windows(P):
    """P['trainw'] = v20 training windows (onset + dense ongoing, identical draw); P['exw'] = run-starts + faithful t%3 rest."""
    tw = build_windows(P, global_spacing=False, dense_ongoing=True)
    rng = np.random.default_rng(0)
    og = [w for w in tw if w[2] == 'queue_ongoing']; on = [w for w in tw if w[2] == 'queue_onset']
    n_og_all = len(og)
    if len(og) > MAX_OG:
        og = [og[i] for i in sorted(rng.choice(len(og), MAX_OG, replace=False))]
    P['trainw'] = on + og
    cand, pers = faithful_candidates(P)
    prev = np.zeros_like(cand); prev[:, 1:] = cand[:, :-1]
    rs = cand & ~prev
    rest = cand & ~rs & (np.arange(288) % 3 == 0)[None, :]
    rd, rt = np.where(rs); fd, ft = np.where(rest); n_rest_all = len(fd)
    if n_rest_all > MAX_OG:
        k = np.sort(np.random.default_rng(1).choice(n_rest_all, MAX_OG, replace=False)); fd, ft = fd[k], ft[k]
    g = np.r_[rd * 288 + rt, fd * 288 + ft].astype(np.int64); isrs = np.r_[np.ones(len(rd), bool), np.zeros(len(fd), bool)]
    o = np.argsort(g, kind='stable'); g, isrs = g[o], isrs[o]
    cur_g = {w[0] * 288 + w[1] for w in og}
    P['exw'] = [(int(x // 288), int(x % 288), ONGOING) for x in g]
    P['ex_rs'] = isrs; P['ex_incur'] = np.array([int(x) in cur_g for x in g], bool)
    dc = np.array([w[0] for w in P['trainw']]); tc = np.array([w[1] for w in P['trainw']])
    isog = np.array([w[2] == ONGOING for w in P['trainw']])
    P['cur_rs'] = isog & rs[dc, tc]
    cur_faith = isog & cand[dc, tc]
    P['cand'], P['rs'], P['pers'] = cand, rs, pers
    st = dict(panel=P['panel'], days=len(P['days']), cur_onset=len(on), cur_og_all=n_og_all, cur_og=len(og),
              cur_og_faithful=int(cur_faith.sum()), cur_og_runstart=int(P['cur_rs'].sum()),
              faith_all_slots=int(cand.sum()), runstarts=len(rd), faith3_rest_all=n_rest_all, faith3_rest_used=len(fd),
              F03_windows=len(g), H_windows=len(og) + int((isrs & ~P['ex_incur']).sum()),
              runstarts_already_in_cur=int((isrs & P['ex_incur']).sum()))
    log(P['panel'], 'windows', json.dumps({k: v for k, v in st.items() if k != 'panel'}))
    log(P['panel'], 'filled persistence | CUR ongoing:', pq(pers[dc[isog], tc[isog]]))
    log(P['panel'], 'filled persistence | run-starts :', pq(pers[rd, rt]))
    log(P['panel'], 'filled persistence | F03 rest   :', pq(pers[fd, ft]))
    P['pers_sets'] = dict(cur=pers[dc[isog], tc[isog]], rs=pers[rd, rt], rest=pers[fd, ft])
    return st


def read_official(panel, P):
    wi = pd.read_csv(f'{REL}/task2/{panel}/train/window_index.csv')
    return [(P['days'].index(pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d')),
             pd.Timestamp(r.forecast_origin).hour * 12 + pd.Timestamp(r.forecast_origin).minute // 5, r.condition)
            for r in wi.itertuples() if pd.Timestamp(r.forecast_origin).strftime('%Y-%m-%d') in P['days']]


# ------------------------------------------------------------------ features
def ffill_time(a):
    """forward-fill NaNs along axis 1 (time) of [W, T, L]."""
    out = a.copy()
    for i in range(1, a.shape[1]):
        m = np.isnan(out[:, i]); out[:, i][m] = out[:, i - 1][m]
    return out


def shift_links(a, k):
    out = np.full_like(a, np.nan)
    if k > 0: out[..., :-k] = a[..., k:]
    elif k < 0: out[..., -k:] = a[..., :k]
    else: out[:] = a
    return out


def dist_to_queued(q, direction):
    """distance in links to the nearest queued link at higher (+1) or lower (-1) index; 99 if none. q: [W, L] bool."""
    W, L = q.shape; out = np.full((W, L), 99.0, np.float32)
    idx = np.arange(L)
    for w in range(W):
        pos = np.where(q[w])[0]
        if len(pos) == 0:
            continue
        if direction > 0:
            j = np.searchsorted(pos, idx, side='left'); ok = j < len(pos)
            out[w, ok] = pos[j[ok]] - idx[ok]
        else:
            j = np.searchsorted(pos, idx, side='right') - 1; ok = j >= 0
            out[w, ok] = idx[ok] - pos[j[ok]]
    return out


def window_features(P, Vh, Fh, tod, dow, prior, VT):
    """Vh, Fh: [W, 12, L] history (NaN = missing). Returns dict name -> [W, L] and filled queue state at T."""
    cut = P['cut'][None, None, :]; cap = P['cap'][None, None, :]
    vr = Vh / cut; fr = Fh / cap
    vrf = ffill_time(vr); frf = ffill_time(fr)
    q_hist = np.nan_to_num(vr <= 1.0, nan=0).astype(bool) & np.isfinite(vr)
    vT = VT / P['cut'][None, :]
    last = np.where(np.isfinite(vT), vT, vrf[:, -1]); qlast = np.nan_to_num(last, nan=9) <= 1.0
    q15 = np.nan_to_num(vrf[:, -4], nan=9) <= 1.0
    W, L = last.shape
    Fe = {'vr_last': last, 'vr_raw_last': vr[:, -1], 'vr_l1': vrf[:, -2], 'vr_l2': vrf[:, -3], 'vr_l3': vrf[:, -4],
          'vr_l6': vrf[:, -7], 'vr_l11': vrf[:, 0], 'vr_min3': np.nanmin(np.where(np.isfinite(vr[:, -3:]), vr[:, -3:], 9), 1),
          'vr_mean12': np.nanmean(np.where(np.isfinite(vr), vr, np.nan), 1),
          'vr_slope': np.nanmean(vr[:, -3:], 1) - np.nanmean(vr[:, :3], 1),
          'fr_last': frf[:, -1], 'fr_mean3': np.nanmean(fr[:, -3:], 1), 'fr_slope': np.nanmean(fr[:, -3:], 1) - np.nanmean(fr[:, :3], 1),
          'q_last': qlast.astype(np.float32), 'q_count': q_hist.sum(1).astype(np.float32),
          'q_age': np.where(q_hist.any(1), 11 - np.argmax(q_hist[:, ::-1], 1) * 0 - (11 - np.argmax(q_hist[:, ::-1], 1)), 12).astype(np.float32),
          'miss_last': np.isnan(vr[:, -1]).astype(np.float32), 'vr_T': vT, 'has_T': np.isfinite(vT).astype(np.float32)}
    Fe['q_age'] = np.where(q_hist.any(1), np.argmax(q_hist[:, ::-1], 1), 12).astype(np.float32)
    for k in (-3, -2, -1, 1, 2, 3):
        Fe[f'vr_n{k:+d}'] = shift_links(last, k); Fe[f'q_n{k:+d}'] = shift_links(qlast.astype(np.float32), k)
        Fe[f'fr_n{k:+d}'] = shift_links(frf[:, -1], k)
    for a, b in ((1, 3), (4, 8), (9, 16)):
        for sgn, nm in ((1, 'dn'), (-1, 'up')):
            stack = np.stack([shift_links(np.nan_to_num(last, nan=9), sgn * k) for k in range(a, b + 1)])
            Fe[f'vr_min_{nm}{a}_{b}'] = np.nanmin(np.where(np.isfinite(stack), stack, 9), 0)
    Fe['dq_hi'] = dist_to_queued(qlast, 1); Fe['dq_lo'] = dist_to_queued(qlast, -1)
    Fe['dq_hi15'] = dist_to_queued(q15, 1); Fe['dq_lo15'] = dist_to_queued(q15, -1)
    q30 = np.nan_to_num(vrf[:, -7], nan=9) <= 1.0
    Fe['dq_hi30'] = dist_to_queued(q30, 1); Fe['dq_lo30'] = dist_to_queued(q30, -1)
    Fe['share_q'] = np.repeat(qlast.mean(1, keepdims=True), L, 1); Fe['share_q15'] = np.repeat(q15.mean(1, keepdims=True), L, 1)
    Fe['n_q_hist_win'] = np.repeat(q_hist.any(1).sum(1, keepdims=True).astype(np.float32), L, 1)
    Fe['pos'] = np.broadcast_to(np.linspace(0, 1, L, dtype=np.float32)[None], (W, L))
    Fe['lanes'] = np.broadcast_to(P['lanes'][None].astype(np.float32), (W, L))
    Fe['cap_lane'] = np.broadcast_to((P['cap'] / P['lanes'])[None].astype(np.float32), (W, L))
    Fe['cap_ratio_hi'] = np.broadcast_to((np.r_[P['cap'][1:], P['cap'][-1]] / P['cap'])[None].astype(np.float32), (W, L))
    Fe['cap_ratio_lo'] = np.broadcast_to((np.r_[P['cap'][0], P['cap'][:-1]] / P['cap'])[None].astype(np.float32), (W, L))
    Fe['length'] = np.broadcast_to(P['length'][None].astype(np.float32), (W, L))
    Fe['tod'] = np.repeat(tod[:, None].astype(np.float32), L, 1); Fe['dow'] = np.repeat(dow[:, None].astype(np.float32), L, 1)
    return Fe, qlast


def prior_table(P, day_mask):
    """P(queued | link, slot) by day class: [0]=weekday, [1]=weekend, [2]=all days (train days in day_mask)."""
    out = []
    wkend = P['dow'] >= 5
    for m in (day_mask & ~wkend, day_mask & wkend, day_mask):
        E = P['E'][m]; Q = P['Q'][m]
        with np.errstate(all='ignore'):
            out.append(Q.sum(0) / np.maximum(E.sum(0), 1))
    return np.stack(out).astype(np.float32)          # [3, 288, L]


def prior_feats(prior, dow, tod, h):
    cls = (np.asarray(dow) >= 5).astype(int); slot = np.minimum(np.asarray(tod) + h, 287)
    return prior[cls, slot], prior[cls, np.asarray(tod)], prior[2, slot]


def make_rows(P, wins, prior, with_labels=True):
    """Stack per-(window, link, step) rows. onset -> step 6 only."""
    if not wins:
        return None
    d = np.array([w[0] for w in wins]); t = np.array([w[1] for w in wins]); cond = [w[2] for w in wins]
    idx = t[:, None] + np.arange(-H, 0)[None]
    Vh = P.get('Vf', P['V'])[d[:, None], idx]; Fh = P.get('Ff', P['Fq'])[d[:, None], idx]
    # slot T itself is only visible through the Task-1-masked layer (~50-66% of cells): simulate that
    rng = np.random.default_rng(len(wins))
    VT = np.where(rng.random((len(d), P['V'].shape[2])) < 0.62, P['V'][d, t], np.nan)
    if 'Vf' in P:                                                          # ineligible-valued cells are never Task-1-masked
        VT = np.where(P['E'][d, t], VT, P['Vf'][d, t])
    Fe, qlast = window_features(P, Vh, Fh, t, P['dow'][d], prior, VT)
    add_ramp(Fe, P['ON'], P['OFF'], d, t, P['cap'])
    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # v27: ogfeat columns
    Fe.update({k: OGX[k] for k in OG_STATIC})
    occ = None
    if 'O' in P:                                                           # occupancy: history cells as Vh, slot T where VT is visible
        Oh = P['O'][d[:, None], idx]; OT = np.where(np.isfinite(VT), P['O'][d, t], np.nan).astype(np.float32)
        occ = occ_features(Oh, OT, Vh, Fh, P)
    W, L = qlast.shape
    rows = []
    for h in range(1, F + 1):
        slot = np.minimum(t + h, 287)
        feats = {k: v for k, v in Fe.items()}
        feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(prior, P['dow'][d], t, h)
        if occ is not None:                                                # appended after the production columns (lab column order)
            feats.update(occ)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # v27: step-dependent columns
        lab = P['Q'][d, np.minimum(t + h, 287)] if with_labels else None
        ev = P['E'][d, np.minimum(t + h, 287)] if with_labels else None
        rows.append((h, feats, lab, ev))
    return dict(d=d, t=t, cond=np.array(cond), rows=rows, qlast=qlast, W=W, L=L)


def to_matrix(R, cond, cols=None, keep=None):
    """Flatten rows for one condition into X, y, (w_idx, l_idx, h)."""
    sel = np.where(R['cond'] == cond)[0]
    Xs, ys, es, meta = [], [], [], []
    steps = [6] if cond == 'queue_onset' else range(1, 7)
    for h, feats, lab, ev in R['rows']:
        if h not in steps:
            continue
        cols = cols or list(feats)
        M = np.stack([np.asarray(feats[c])[sel] for c in cols], -1)      # [w, L, f]
        wi, lidx = np.meshgrid(sel, np.arange(R['L']), indexing='ij')
        m = np.ones(wi.shape, bool) if keep is None else keep[sel]
        Xs.append(M[m]); meta.append(np.stack([wi[m], lidx[m], np.full(m.sum(), h)], 1))
        if lab is not None:
            ys.append(lab[sel][m]); es.append(ev[sel][m])
    X = np.concatenate(Xs).astype(np.float32); meta = np.concatenate(meta)
    y = np.concatenate(ys) if ys else None; e = np.concatenate(es) if es else None
    return X, y, e, meta, cols


def ongoing_keep(R, radius=12):
    """keep cells near queue activity: within `radius` links of a queue now / 15 / 30 min ago,
    or slow now (<= 1.3 x v_cut in the last 15 min), or historically queued at this slot (>= 5%)."""
    f = R['rows'][0][1]
    near = lambda a, b: np.minimum(f[a], f[b]) <= radius
    k = near('dq_hi', 'dq_lo') | near('dq_hi15', 'dq_lo15') | near('dq_hi30', 'dq_lo30')
    k |= np.nan_to_num(f['vr_min3'], nan=9) <= 1.3
    k |= np.nan_to_num(f['prior_now'], nan=0) >= 0.05
    return k


def decode(p):
    """choose the top-m cells maximizing sum(top p) / (m + sum(rest p)); at least one cell."""
    order = np.argsort(-p); ps = p[order]; cs = np.cumsum(ps); tot = cs[-1]
    m = np.arange(1, len(p) + 1)
    j = np.argmax(cs / (m + (tot - cs)))
    out = np.zeros(len(p), bool); out[order[:j + 1]] = True
    return out


PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=4)
if TFB.cfg['lgb_repro'] is not None:  # [TFB] reproducibility switch (tfb/config.py LGB_REPRO): deterministic=True, force_row_wise=True;
    PARAMS.update(TFB.cfg['lgb_repro'])  # [TFB] None = the original setting (auto row/col-wise choice by timing, not repeatable)


PRIOR_COLS = {'prior', 'prior_now', 'prior_alldays', 'pos', 'tod', 'dow', 'lanes', 'cap_lane', 'length', 'cap_ratio_hi', 'cap_ratio_lo'}
ID_DROP = {'prior', 'prior_now', 'prior_alldays', 'tod', 'dow'}


def predict_windows(R, models, cols):
    """probabilities [W, 6, L] for all windows in R."""
    P6 = np.zeros((R['W'], F, R['L']), np.float32)
    for cond, mdl in models.items():
        keep = ongoing_keep(R) if cond == 'queue_ongoing' else None
        X, _, _, meta, _ = to_matrix(R, cond, cols, keep)
        if len(X):
            P6[meta[:, 0], meta[:, 2] - 1, meta[:, 1]] = mdl.predict(X)
    return P6


# ------------------------------------------------------------------ training rows
def panel_rows(P, cols=None):
    """ongoing training rows of one panel: CUR (exactly as v20: same windows, same slot-T mask draw, same keep mask) and EX
    (run-starts + faithful rest). Returns cols and two dicts(X, y, rs[, incur])."""
    P['prior_all'] = prior_table(P, np.ones(len(P['days']), bool))
    R = make_rows(P, P['exw'], P['prior_all'])                              # v27: the v20-style CUR rows are not built
    X, y, e, meta, cols = to_matrix(R, ONGOING, cols, ongoing_keep(R))
    ex = dict(X=X[e], y=y[e], rs=P['ex_rs'][meta[:, 0]][e], incur=P['ex_incur'][meta[:, 0]][e])
    del R, X; gc.collect()
    return cols, ex


# ------------------------------------------------------------------ released windows (validation / private)
def load_split(P, panel, split):
    """features of the released windows exactly as v20: 60-min history, masked slot-T value, ramp slots < T. No labels."""
    wi = pd.read_csv(f'{REL}/task2/{panel}/{split}/window_index.csv')
    hist = pd.read_parquet(f'{REL}/task2/{panel}/{split}/window_history.parquet')
    tpl = pd.read_csv(f'{REL}/task2/{panel}/{split}/sample_submission_queue.csv', dtype=str)
    W, L = len(wi), len(P['links'])
    Vh = np.full((W, H, L), np.nan, np.float32); Fh = Vh.copy(); Oh = Vh.copy()
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    hist = hist[((hist.is_score_eligible == 1) | ALLFIN) & hist.speed_kmh.notna()]
    org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
    ts = pd.to_datetime(hist.timestamp, utc=True)
    step = ((ts - hist.window_id.map(org)).dt.total_seconds() // 300).astype(int) + H
    ok = (step >= 0) & (step < H) & hist.link_id.isin(P['li'])
    hh = hist[ok].assign(s=step[ok].values, w=hist[ok].window_id.map(wpos).values, l=hist[ok].link_id.map(P['li']).values)
    g = hh.groupby(['w', 's', 'l'])[['speed_kmh', 'flow_vph']].mean().reset_index()
    Vh[g.w, g.s, g.l] = g.speed_kmh; Fh[g.w, g.s, g.l] = g.flow_vph
    if OCC:                                                                # occupancy of the same history cells
        go = hh[hh.occupancy.notna()].groupby(['w', 's', 'l']).occupancy.mean().reset_index()
        Oh[go.w, go.s, go.l] = go.occupancy
        Oh = np.where(np.isfinite(Vh), Oh, np.nan).astype(np.float32)
    tod = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
    dow = org.dt.dayofweek.to_numpy()
    VT = np.full((W, L), np.nan, np.float32); OT = VT.copy()
    for date, grp in wi.groupby('date'):
        fs = glob.glob(f'{REL}/corridors/{panel}/{split}/mainline_states_masked/*/*{date.replace("-", "_")}.parquet')
        if not fs:
            continue
        m = pd.read_parquet(fs[0], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'] + (['occupancy'] if OCC else []))
        m = m[((m.is_score_eligible == 1) | ALLFIN) & m.speed_kmh.notna()]
        for r in grp.itertuples():
            x = m[m.timestamp == r.forecast_origin]
            x = x[x.link_id.isin(P['li'])]
            xs = x.groupby('link_id').speed_kmh.mean()
            VT[wpos[r.window_id], xs.index.map(P['li']).to_numpy()] = xs.to_numpy()
            if OCC:
                xo = x[x.occupancy.notna()].groupby('link_id').occupancy.mean()
                OT[wpos[r.window_id], xo.index.map(P['li']).to_numpy()] = xo.to_numpy()
    log(panel, split, 'slot-T visible share', round(float(np.isfinite(VT).mean()), 3),
        'history filled share', round(float(np.isfinite(Vh).mean()), 3))
    Fe, qlast = window_features(P, Vh, Fh, tod, dow, P['prior_all'], VT)
    wdays = sorted(wi.date.unique())
    ONs, OFFs = ramp_cube(panel, split, P['li'], wdays)
    add_ramp(Fe, ONs, OFFs, wi.date.map({dd: i for i, dd in enumerate(wdays)}).to_numpy(), tod, P['cap'])
    log(panel, split, 'ramp on3 finite share', round(float(np.isfinite(Fe['ramp_on3']).mean()), 3))
    OGX = og_extra(P, Vh, Fh, VT, Fe); OGX.update(og_extra2(P, Vh, Fh, VT, Fe))          # v27: ogfeat columns
    Fe.update({k: OGX[k] for k in OG_STATIC})
    occ = None
    if OCC:
        OT = np.where(np.isfinite(VT), OT, np.nan).astype(np.float32)
        log(panel, split, 'occupancy: history finite share', round(float(np.isfinite(Oh).mean()), 3), '(speed', round(float(np.isfinite(Vh).mean()), 3),
            ') | slot-T finite share', round(float(np.isfinite(OT).mean()), 3), '(speed', round(float(np.isfinite(VT).mean()), 3), ')')
        occ = occ_features(Oh, OT, Vh, Fh, P)
    R = dict(d=np.zeros(W, int), t=tod, cond=wi.condition.to_numpy(), qlast=qlast, W=W, L=L, rows=[])
    for h in range(1, F + 1):
        feats = dict(Fe); feats['h'] = np.full((W, L), h, np.float32)
        feats['prior'], feats['prior_now'], feats['prior_alldays'] = prior_feats(P['prior_all'], dow, tod, h)
        if occ is not None:
            feats.update(occ)
        feats.update({k: v for k, v in og_step(OGX, h).items() if k in OG_STEP})          # v27: step-dependent columns
        R['rows'].append((h, feats, None, None))
    return wi, tpl, R


VARIANTS = {'fs2ogf': ('fs_np', 'fs_id'),                              # seed-0 pair + ogfeat (PART a; Q4 candidate if b is late)
            'fs4ogf': ('fs_np', 'fs_id', 'fs_np1', 'fs_id1'),          # bag of 4 (only when PART == 'all'; else t2_ogfeat_comb.py)
            'fs2ogf1': ('fs_np1', 'fs_id1'),                           # seed-1 pair (PART b)
            'fs_npogf': ('fs_np',)}


def predict_split(P, panel, split, models, foot, probs):
    """decoded masks for every available variant mapped onto the queue template; footprint rows vs cur2 appended to `foot`."""
    wi, tpl, R = load_split(P, panel, split)
    W, L = R['W'], R['L']
    is_og = (wi.condition == ONGOING).to_numpy()
    P6 = {name: predict_windows(R, {ONGOING: mdl}, cols) for name, (mdl, cols) in models.items()}
    for name, p in P6.items():
        probs[f'{panel}__{split}__{name}'] = p[is_og].astype(np.float32)          # v27: float32 (exact combine)
    probs[f'{panel}__{split}__window_id'] = wi.window_id.to_numpy()[is_og].astype(str)
    probs[f'{panel}__{split}__qlast'] = R['qlast'][is_og]
    preds = {}
    for v, parts in VARIANTS.items():
        if not all(m in P6 for m in parts):
            continue
        p = P6[parts[0]] if len(parts) == 1 else (1.0 / len(parts)) * sum([P6[m] for m in parts[1:]], P6[parts[0]])
        pr = np.zeros((W, F, L), bool)                                    # onset windows stay 0 (not produced by this kernel)
        for i in np.where(is_og)[0]:
            pr[i] = decode(p[i].ravel()).reshape(F, L)
        preds[v] = pr
    for v, pr in preds.items():
        for i in np.where(is_og)[0]:
            a, b = pr[i], preds[next(iter(preds))][i]
            row = dict(panel=panel, split=split, variant=v, window_id=wi.window_id.iloc[i], changed=bool((a != b).any()),
                       jacc_cur2=float((a & b).sum() / max((a | b).sum(), 1)), size_T=int(R['qlast'][i].sum()))
            for h in range(F):
                row[f'chg_h{h + 1}'] = int((a[h] != b[h]).sum()); row[f'add_h{h + 1}'] = int((a[h] & ~b[h]).sum())
                row[f'size_h{h + 1}'] = int(a[h].sum())
            foot.append(row)
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
    tpl['w'] = tpl.window_id.map(wpos); tpl['l'] = tpl.link_id.map(P['li'])
    tts = pd.to_datetime(tpl.timestamp, utc=True)
    tpl['h'] = ((tts - tpl.window_id.map(org)).dt.total_seconds() // 300).astype(int)
    okr = tpl.w.notna() & tpl.l.notna() & tpl.h.between(1, F)
    for name, pr in preds.items():
        col = f'queue_pred_{name}'
        tpl[col] = 0
        tpl.loc[okr, col] = pr[tpl.w[okr].astype(int), tpl.h[okr].astype(int) - 1, tpl.l[okr].astype(int)].astype(int)
    tpl['is_og'] = tpl.window_id.map(dict(zip(wi.window_id, is_og))).fillna(False).astype(bool)
    log(panel, split, 'windows', W, 'ongoing', int(is_og.sum()), 'template rows', len(tpl), 'unmapped', int((~okr).sum()),
        'positive share (ongoing rows)', {n: round(float(tpl.loc[tpl.is_og, f'queue_pred_{n}'].mean()), 4) for n in preds})
    return tpl[['window_id', 'timestamp', 'link_id', 'is_og'] + [f'queue_pred_{n}' for n in preds]], list(preds)


def print_footprint(foot):
    """footprint of every variant against cur2 on the official ongoing windows (predictions only, no labels)."""
    fp = pd.DataFrame(foot)
    if fp.empty:
        return fp
    hs = range(1, F + 1)
    for split, g in fp.groupby('split'):
        b0 = g.variant.iloc[0]
        log(f'FOOTPRINT {split} (vs {b0}): ongoing windows {g[g.variant == b0].shape[0]} | mean queued cells at T (q_last) '
            f'{g[g.variant == b0].size_T.mean():.2f}')
        for v, x in g.groupby('variant', sort=False):
            log(f'  {v:8s} windows changed {int(x.changed.sum())}/{len(x)} | mean Jaccard vs {b0} {x.jacc_cur2.mean():.4f}'
                f' | cells changed per step h1..h6 {[int(x[f"chg_h{h}"].sum()) for h in hs]} (added {[int(x[f"add_h{h}"].sum()) for h in hs]})'
                f' | mean predicted size per step {[round(float(x[f"size_h{h}"].mean()), 2) for h in hs]}')
        pv = g.pivot_table(index='panel', columns='variant', values='changed', aggfunc='sum')
        log(f'  windows changed per panel ({split}):', pv.to_dict('index'))
    return fp


def control_check(S, names):
    """FOOTPRINT of every variant vs tfb-t2occ/queue_submission_fs4occ.csv (current March ongoing production), by split.
    The models differ by design (new features); ALARM if more than 3 % of the validation ongoing rows change."""
    ref = TFB.kglob('tfb-t2occ', 'queue_submission_fs4occ.csv')  # [TFB] optional FOOTPRINT reference
    if not ref:
        log('FOOTPRINT: skipped (tfb-t2occ queue_submission_fs4occ.csv not attached)'); return None
    r = pd.read_csv(ref[0], dtype={'window_id': str, 'timestamp': str, 'link_id': str})
    m = S.merge(r.rename(columns={'queue_pred': 'ref'}), on=['window_id', 'timestamp', 'link_id'], how='left')
    log('FOOTPRINT: reference rows matched', int(m.ref.notna().sum()), 'of', len(m), '| unmatched ongoing rows', int((m.is_og & m.ref.isna()).sum()))
    m = m[m.is_og & m.ref.notna()]
    m['split'] = np.where(m.window_id.isin(VAL_IDS), 'validation', 'private')
    out = {}
    for name in names:
        for split, x in m.groupby('split'):
            a = x[f'queue_pred_{name}'].to_numpy().astype(int); b = x.ref.to_numpy().astype(int)
            chg = x.assign(c=(a != b)).groupby('window_id').c.any(); sh = float((a != b).mean())
            flag = ' | ALARM: more than 3 % of the validation ongoing rows changed' if (split == 'validation' and sh > 0.03) else ''
            log(f'FOOTPRINT {name} vs tfb-t2occ fs4occ | {split}: windows changed {int(chg.sum())}/{len(chg)} | rows changed {int((a != b).sum())}/{len(a)}'
                f' ({sh:.4%}) | Jaccard of queued cells {(a & b).sum() / max((a | b).sum(), 1):.4f} | positives {int(a.sum())} ref {int(b.sum())}{flag}')
            out[(name, split)] = sh
    return out


VAL_IDS = set()


# ------------------------------------------------------------------ main
def main():
    panels = {}; stats = []
    for panel in PANELS:
        P = load_panel(panel)
        stats.append(select_windows(P))
        off = [w for w in read_official(panel, P) if w[2] == ONGOING]
        if off:
            od = np.array([w[0] for w in off]); ot = np.array([w[1] for w in off])
            log(panel, f'official train ongoing windows {len(off)}: faithful {int(P["cand"][od, ot].sum())}, run-starts {int(P["rs"][od, ot].sum())},'
                f' filled persistence {np.round(P["pers"][od, ot], 3).tolist()}')
            stats[-1].update(official_og=len(off), official_faithful=int(P['cand'][od, ot].sum()), official_runstart=int(P['rs'][od, ot].sum()))
        panels[panel] = P
    st = pd.DataFrame(stats); st.to_csv(f'{OUT}/t2{TAG}_trainsets.csv', index=False)
    log('TRAIN SETS total:', st.drop(columns=['panel']).sum().to_dict())
    for k, nm in (('cur', 'CUR ongoing'), ('rs', 'run-starts'), ('rest', 'F03 rest')):
        log(f'filled persistence ALL panels | {nm}:', pq(np.concatenate([P['pers_sets'][k] for P in panels.values()])))

    # ---- training rows (ongoing only)
    cols = None; EX = []
    for panel, P in panels.items():
        cols, ex = panel_rows(P, cols)
        log(panel, f'rows EX {len(ex["y"])} (pos {ex["y"].mean() if len(ex["y"]) else float("nan"):.3f};'
            f' run-start rows {int(ex["rs"].sum())}, pos {ex["y"][ex["rs"]].mean() if ex["rs"].any() else float("nan"):.3f}) | cols {len(cols)}')
        EX.append(ex)
        for k in ('V', 'Fq', 'E', 'Q', 'ON', 'OFF', 'cand', 'rs', 'pers', 'Vf', 'Ff', 'O'):  # only network vectors + prior (+ ocrit / occ_c) from here on
            P.pop(k, None)
        gc.collect()
    cat = lambda L_, k: np.concatenate([x[k] for x in L_])
    Xe, ye, re_ = cat(EX, 'X'), cat(EX, 'y').astype(np.float32), cat(EX, 'rs')
    del EX; gc.collect()
    cols_np = [x for x in cols if x not in PRIOR_COLS or x in OG_ADD_PROD]; ix_np = [cols.index(x) for x in cols_np]
    cols_id = [x for x in cols if x not in ID_DROP or x in OG_ADD_PROD]; ix_id = [cols.index(x) for x in cols_id]
    w_f = np.where(re_, W_RS, W_DENSE).astype(np.float32)
    new_cols = [x for x in cols if x in OG_STATIC or x in OG_STEP]
    assert len(new_cols) == len(OG_STATIC) + len(OG_STEP), 'a new column is missing from the matrix'
    assert all(c in cols_np and c in cols_id for c in new_cols) and all(c in cols_np and c in cols_id for c in OCC_COLS)
    log(f'features: all {len(cols)}, np {len(cols_np)}, id {len(cols_id)} | new {len(new_cols)} | Xe {Xe.shape} {Xe.nbytes / 1e9:.2f} GB')
    log('new feature finite share / mean (first 200k rows):', {c: (round(float(np.isfinite(Xe[:200000, cols.index(c)]).mean()), 3),
                                                              round(float(np.nanmean(Xe[:200000, cols.index(c)])), 3)) for c in new_cols})
    log(f'POSITIVE RATES | F03 rows {len(ye)} pos {ye.mean():.4f} weighted {np.average(ye, weights=w_f) if len(ye) else float("nan"):.4f}'
        f' (run-start rows {int(re_.sum())} pos {ye[re_].mean() if re_.any() else float("nan"):.4f}; rest pos {ye[~re_].mean() if (~re_).any() else float("nan"):.4f})')
    P511 = dict(PARAMS, num_leaves=511, min_data_in_leaf=40, learning_rate=0.03)   # = v20 og_big / og_id (seed 0)
    models = {}; rate = [None]

    def fit(name, getX, y, w, cset, params=None):
        n = len(y)
        if rate[0] is not None and time.time() - T0 + rate[0] * n * 1.15 > TIME_LIMIT_S:
            log(f'SKIPPED model {name}: predicted {rate[0] * n / 3600:.2f} h would pass the time limit'); return
        t1 = time.time()
        X = getX()
        mdl = lgb.train(params or P511, lgb.Dataset(X, y, weight=w), N_BIG)
        del X; gc.collect()
        dt = time.time() - t1; rate[0] = max(rate[0] or 0.0, dt / max(n, 1))
        models[name] = (mdl, cset)
        log(f'model {name} trained: {n} rows x {len(cset)} cols, {N_BIG} trees, {dt / 60:.1f} min, weighted {w is not None}')

    S1 = dict(P511, seed=1, bagging_seed=11, feature_fraction_seed=21)
    if PART in ('a', 'all'):
        fit('fs_np', lambda: Xe[:, ix_np], ye, w_f, cols_np)
        fit('fs_id', lambda: Xe[:, ix_id], ye, w_f, cols_id)
    if PART in ('b', 'all'):
        fit('fs_np1', lambda: Xe[:, ix_np], ye, w_f, cols_np, S1)
        fit('fs_id1', lambda: Xe[:, ix_id], ye, w_f, cols_id, S1)
    log('PART', PART, 'models trained:', list(models))
    del Xe; gc.collect()

    # ---- released windows
    subs = []; foot = []; probs = {}; names = None
    for panel, P in panels.items():
        for split in ('validation', 'private'):
            tpl, names = predict_split(P, panel, split, models, foot, probs)
            if split == 'validation':
                VAL_IDS.update(tpl.window_id.unique().tolist())
            subs.append(tpl)
    S = pd.concat(subs, ignore_index=True)
    for name in names:
        S[['window_id', 'timestamp', 'link_id', f'queue_pred_{name}']].rename(columns={f'queue_pred_{name}': 'queue_pred'}).to_csv(
            f'{OUT}/queue_submission_{name}.csv', index=False)
    np.savez_compressed(f'{OUT}/t2{TAG}_probs.npz', **probs)
    log('queue rows', len(S), 'dupes', int(S.duplicated(['window_id', 'timestamp', 'link_id']).sum()), 'ongoing rows', int(S.is_og.sum()),
        'files', [f'queue_submission_{n}.csv' for n in names], '| onset rows are 0 in every file')
    fp = print_footprint(foot)
    if len(fp):
        fp.to_csv(f'{OUT}/t2{TAG}_footprint.csv', index=False)
    control_check(S, names)
    log('done')


main()
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
