# ======================================================================================================================
# GENERATED FILE - DO NOT EDIT. Built by consolidation/bundle.py from the tfb package
#   tfb 0.1.0-m1 | modules tree sha1 99d458e7d1189636c1d24f80748fe494bb8fd761 | generated 2026-10-09 22:41
# Parity kernel tfb-par-t4 (required, ~20 min on Kaggle CPU): Task 4 NNLS (tfb-final cell 7) -> skeleton; path_flow must equal tfb-final and both final files exactly
# Stages ['t4'] | pick A2 | recipe full | overrides {}
# Kernel sources to attach (3, plus the competition data):
#   hosen42/tfb-final hosen42/tfb-j4 hosen42/tfb-j3
# Push: python kaggle_jobs/kjob.py push tfb-par-t4 consolidation/kaggle_parity/tfb-par-t4.py hosen42/tfb-final hosen42/tfb-j4 hosen42/tfb-j3
# Output: PARITY lines + PARITY SUMMARY in the log; /kaggle/working/parity_<slug>.json; stage outputs in /kaggle/working/tfb_work/
# ======================================================================================================================
import glob, json, os, sys, tempfile, time

SPEC = json.loads(r'''{"slug": "tfb-par-t4", "priority": "required", "minutes": 20, "stages": ["t4"], "pick": "A2", "recipe": "full", "overrides": {}, "desc": "Task 4 NNLS (tfb-final cell 7) -> skeleton; path_flow must equal tfb-final and both final files exactly", "sources": ["tfb-final", "tfb-j4", "tfb-j3"], "checks": [{"type": "submission", "ours": ["tfb-final", "submission.csv"], "orig": ["tfb-final", "submission.csv"], "tasks": ["odme"], "label": "t4_vs_tfb-final"}, {"type": "submission", "ours": ["tfb-final", "submission.csv"], "orig": ["tfb-j4", "submission_J4.csv"], "tasks": ["odme"], "label": "t4_vs_tfb-j4"}, {"type": "submission", "ours": ["tfb-final", "submission.csv"], "orig": ["tfb-j3", "submission_J3.csv"], "tasks": ["odme"], "label": "t4_vs_tfb-j3"}], "tfb_version": "0.1.0-m1", "tree_sha1": "99d458e7d1189636c1d24f80748fe494bb8fd761", "generated": "2026-10-09 22:41"}''')

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

# ------------------------------------------------------------------ Task 1 members (milestone 2; DESIGN 1.3)
# Common LightGBM: 511 leaves, min_data 100, lr 0.04, 1500 trees, max_bin 63, ff 0.8, bf 0.8, num_threads 4, MAX_ROWS 600k.
T1_MEMBERS = {
    'tfb-t1v11':    dict(family='fta',   row_rng=0,  rounds=2, rng_base=1000,  tx_cap=4, holdout=0.15,  lgb_seeds=None,         image='A'),
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
    'tfb-t1bspec':  dict(family='bspec', row_rng=13, rounds=0, rng_base=None,  tx_cap=6, holdout=0.15,  lgb_seeds=(13, 23, 33), image='B', dark_seed=1300,
                         dark_n=48, panel_overrides={'D7_I405_N': dict(max_rows=420_000, train_day_mod5_keep=(0, 1, 3))}),
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
    # ---- milestone 2 (not ported yet)
    'pseudo':        dict(module=None, kernel='tfb-t2lab-full', outputs=[f'pseudo_{p}.npz' for p in PANELS_T2], m=2),
    't1_members':    dict(module=None, kernel=None, outputs=['state_submission_FTA.csv'], m=2),   # one kernel per member part
    'ongoing_val':   dict(module=None, kernel='tfb-t2occ',     outputs=['queue_submission_fs4occ.csv'], m=2),   # + t2v26occ_probs.npz (qlast) for B2
    'ongoing_pvaf':  dict(module=None, kernel='tfb-t2ogpvaf',  outputs=['queue_submission_pvo.csv'], m=2),
    'ongoing_pvogf': dict(module=None, kernel='tfb-t2ogpvogf', outputs=['queue_submission_pvogfv.csv|queue_submission_pvogf.csv'], m=2),
    'ogfeat_a':      dict(module=None, kernel='tfb-t2ogfa',    outputs=['queue_submission_fs2ogf.csv'], m=2),
    'ogfeat_b':      dict(module=None, kernel='tfb-t2ogfb',    outputs=['queue_submission_fs2ogf1.csv'], m=2),
    'ogfeat_comb':   dict(module=None, kernel='tfb-t2ogfc',    outputs=['queue_submission_fs4ogf.csv'], m=2),
    'ogveto':        dict(module=None, kernel='tfb-t2ogveto',  outputs=['queue_submission_veto10.csv'], m=2),
}
LIGHT_STAGES = ['t4', 'onset_sp', 'onset_pv', 't1_mix', 't1_recon', 'assemble']


def plan(pick, recipe):
    """stage order of a full run (milestone 2 stages included). t4 precedes t1_mix: it writes the base skeleton."""
    t2 = ['onset_sp'] + (['onset_pv'] if pick == 'A2' else []) + ['ongoing_val']
    if pick == 'A2':
        t2 += (['ongoing_pvaf'] if ONGOING['need_pvo'] else []) + ['ongoing_pvogf']
    else:
        t2 += ['ogfeat_a', 'ogfeat_b', 'ogfeat_comb', 'ogveto']
    return ['pseudo', 't1_members', 't4'] + t2 + ['t1_mix', 't1_recon', 'assemble']


def stage_cfg(stage, pick, recipe):
    """the constants a stage file reads through TFB.cfg (JSON-serialisable)."""
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
    return {}


def stage_kernel(stage, pick):
    if stage == 'assemble':
        return PICK_KERNEL[pick][0]
    return STAGES[stage]['kernel']


def stage_outputs(stage, pick):
    if stage == 'assemble':
        return [f'submission_{PICK_KERNEL[pick][1]}.csv']
    return list(STAGES[stage]['outputs'])


def stage_inputs(stage, pick, recipe, cfg=None):
    """(kernel, file) pairs a stage reads besides the competition data."""
    cfg = cfg if cfg is not None else stage_cfg(stage, pick, recipe)
    if stage == 't1_mix':
        bag = T1_BAGS[recipe]
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
    return []
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
        cfg = C.stage_cfg(stage, self.pick, self.recipe)
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
        spec = C.STAGES[stage]
        kernel = C.stage_kernel(stage, self.pick)
        outs = C.stage_outputs(stage, self.pick)
        if spec['module'] is None:                                   # milestone 2: accept provided outputs only
            kernels = C.t1_member_kernels(self.recipe) if stage == 't1_members' else [kernel]
            miss = [(k, o) for k in kernels for o in outs if not self.resolve(k, o)]
            if miss:
                raise StageError(f'stage {stage} is not ported yet (milestone 2) and its outputs are missing: {miss[:6]}'
                                 f'{" ..." if len(miss) > 6 else ""} - provide them with --seed-kernel-outputs / --input-root')
            self.log(f'[tfb] {stage:12s} provided (milestone-2 stage; {len(kernels)} kernel(s) found)')
            return 'provided'
        cfg = self.cfg(stage)
        inputs = C.stage_inputs(stage, self.pick, self.recipe, cfg)
        miss = [(k, f) for k, f in inputs if not self.resolve(k, f)]
        if miss:
            raise StageError(f'stage {stage}: missing inputs {miss[:8]}{" ..." if len(miss) > 8 else ""}')
        in_paths = [self.resolve(k, f)[0] for k, f in inputs]
        out_dir = os.path.join(self.kernels_dir, kernel)
        os.makedirs(out_dir, exist_ok=True)
        src = os.path.join(STAGE_DIR, spec['module'] + '.py')
        fp_obj = dict(stage=stage, code=_sha1_file(src), runtime=_sha1_file(os.path.join(os.path.dirname(STAGE_DIR), 'runtime.py')),
                      cfg=cfg, smoke=self.smoke, inputs=[_sig(p) for p in in_paths], data=os.path.abspath(self.data_root),
                      threads=self.threads if stage == 't1_members' else None)
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
                   stage=stage, module=spec['module'], kernel=kernel, pick=self.pick, recipe=self.recipe, smoke=self.smoke,
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
        return C.plan(pick, recipe)
    if arg == 'light':
        return [s for s in C.LIGHT_STAGES if s in C.plan(pick, recipe)]
    out = [s.strip() for s in arg.split(',') if s.strip()]
    bad = [s for s in out if s not in C.STAGES]
    if bad:
        raise SystemExit(f'unknown stage(s) {bad}; known: {list(C.STAGES)}')
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


def main(ctx_path):
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
odme.to_csv(f'{TFB.out}/odme_path_flow.csv', index=False)  # [TFB] the Task 4 table itself (parity / reuse)

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
        else:
            parity.compare_queue_file(ours, orig, c['label'], rel=rel)
    parity.summary(save=os.path.join(kwd, f"parity_{SPEC['slug']}.json"))
    print(f'PARITY KERNEL DONE in {time.time() - t0:.0f}s', flush=True)


main()
