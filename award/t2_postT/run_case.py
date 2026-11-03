"""Post-T deletion test, step 3: run the Task 2 producers of the final picks on one release tree.

  run_case.py REL WORK [stages=onset_sp,onset_pv,ongoing_val,ogfeat_a,ogfeat_b,ogfeat_comb,ogveto,ongoing_pvogf] [smoke=1] [vendor=vendor|vendor_r]

vendor   = verbatim copies (+ [POSTT] panel override); vendor_r = + the [POSTT-R1..R6] empty-input guards (vendor_r_patches.diff),
           shown to give byte-identical outputs on the full tree (results_noise.csv).

Each stage runs in a fresh subprocess (as consolidation/tfb/stage_exec.py does) on the vendored, otherwise unchanged copies in
award/t2_postT/vendor/tfb/stages (only change: POSTT_PANELS env override of the panel list, lines tagged [POSTT]).
Outputs land in WORK/kernels/<kernel>/ (the chained stages read their inputs from there, as on Kaggle). The stage constants
(cfg) are those of consolidation/tfb/config.py stage_cfg(): LGB_REPRO = deterministic + force_row_wise for the LightGBM stages.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REL, WORK = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
kw = dict(a.split('=') for a in sys.argv[3:])
LGB_REPRO = dict(deterministic=True, force_row_wise=True)
STAGES = {
    'onset_sp':      ('onset_sp', 'tfb-t2osp', {}),
    'onset_pv':      ('onset_pv', 'tfb-t2opv', {}),
    'ongoing_val':   ('t2_occ', 'tfb-t2occ', dict(lgb_repro=LGB_REPRO)),
    'ogfeat_a':      ('t2_ogfeat', 'tfb-t2ogfa', dict(part='a', tag='v27a', lgb_repro=LGB_REPRO)),
    'ogfeat_b':      ('t2_ogfeat', 'tfb-t2ogfb', dict(part='b', tag='v27b', lgb_repro=LGB_REPRO)),
    'ogfeat_comb':   ('t2_ogfc', 'tfb-t2ogfc', {}),
    'ogveto':        ('t2_ogveto', 'tfb-t2ogveto', {}),
    'ongoing_pvogf': ('t2_pvogf', 'tfb-t2ogpvogf', dict(ref_v22=['tfb-t2occ', 'queue_submission_fs4occ.csv'], veto_g=10, lgb_repro=LGB_REPRO)),
}
order = kw.get('stages', ','.join(STAGES)).split(',')
smoke = kw.get('smoke', '1') == '1'
os.makedirs(f'{WORK}/kernels', exist_ok=True); os.makedirs(f'{WORK}/logs', exist_ok=True)
RUNNER = r'''
import os, runpy, sys
sys.path.insert(0, sys.argv[2])
import tfb.runtime as rt
rt.ctx = rt.Ctx.load(sys.argv[1])
os.makedirs(rt.ctx.out, exist_ok=True); os.chdir(rt.ctx.out)
runpy.run_module('tfb.stages.' + rt.ctx.module, run_name='__main__', alter_sys=True)
'''
for st in order:
    module, kernel, cfg = STAGES[st]
    out = f'{WORK}/kernels/{kernel}'
    ctx = dict(rel=REL, out=out, work_dir=WORK, kernels_dir=f'{WORK}/kernels', input_roots=[], stage=st, kernel=kernel,
               module=module, pick='postT', recipe='postT', smoke=smoke, threads=4, cfg=cfg)
    cp = f'{WORK}/logs/{st}.ctx.json'
    with open(cp, 'w') as f:
        json.dump(ctx, f, indent=1)
    t0 = time.time()
    with open(f'{WORK}/logs/{st}.log', 'w') as lg:
        rc = subprocess.call([sys.executable, '-c', RUNNER, cp, f'{HERE}/' + kw.get('vendor', 'vendor')], stdout=lg, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OGVETO_LOCAL=''))
    print(f'[run_case] {st} rc={rc} {time.time() - t0:.0f}s -> {out}', flush=True)
    if rc != 0:
        print(f'[run_case] STAGE FAILED: {st} (see {WORK}/logs/{st}.log)', flush=True)
        sys.exit(rc)
print('[run_case] DONE', WORK, flush=True)
