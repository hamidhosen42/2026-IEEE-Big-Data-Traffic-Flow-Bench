"""Post-T deletion test: one case = truncate the release at CUT, rerun the producers, compare with the full-data reference.

  cut_job.py FULL_ROOT REF_WORK CASES_DIR CUT NAME RESULTS_CSV [mode=keep] [stages=...] [smoke=1]

FULL_ROOT = folder holding kaggle_public/ (the full tree); REF_WORK = run_case.py work dir of the full tree (same stages / smoke /
POSTT_PANELS). The truncated tree is deleted after the run (outputs + logs stay in CASES_DIR/NAME/work).
"""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FULL, REF, CASES, CUT, NAME, RES = sys.argv[1:7]
kw = dict(a.split('=') for a in sys.argv[7:])
mode = kw.get('mode', 'keep')
py = sys.executable
case = f'{CASES}/{NAME}'
os.makedirs(case, exist_ok=True)
tree = f'{case}/tree'
if os.path.exists(tree):
    shutil.rmtree(tree)
subprocess.check_call([py, f'{HERE}/truncate.py', FULL, tree, CUT, f'mode={mode}'])
args = [py, f'{HERE}/run_case.py', f'{tree}/kaggle_public', f'{case}/work', f'smoke={kw.get("smoke", "1")}']
for k in ('stages', 'vendor'):
    if k in kw:
        args.append(f'{k}={kw[k]}')
rc = subprocess.call(args)
if rc == 0:
    subprocess.check_call([py, f'{HERE}/compare.py', f'{FULL}/kaggle_public', REF, f'{case}/work', CUT, NAME, RES])
else:
    with open(RES + '.failures', 'a') as f:
        f.write(f'{NAME}\t{CUT}\t{mode}\trc={rc}\tsee {case}/work/logs\n')
    print(f'[cut_job] {NAME}: producer run FAILED rc={rc}', flush=True)
shutil.rmtree(tree, ignore_errors=True)
sys.exit(rc)
