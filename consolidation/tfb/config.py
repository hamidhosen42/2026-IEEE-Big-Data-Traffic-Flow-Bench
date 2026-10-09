"""Recipes of the two final picks: the single source of truth for every constant the entry script uses.

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
