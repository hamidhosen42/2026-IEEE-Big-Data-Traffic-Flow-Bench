#!/usr/bin/env python3
"""Generate the single-file Kaggle PARITY kernels: consolidation/kaggle_parity/<slug>.py (+ SOURCES.md, sources.json).

  python bundle.py                 # all specs
  python bundle.py --only tfb-par-t4,tfb-par-t1mix

Each generated kernel inlines the needed tfb modules byte for byte (as string literals that it writes to a temporary
package folder at start-up), then on Kaggle:
  1. prints the environment (ENV lines: package versions, CPU flags) - this is how the image-B pins are confirmed;
  2. runs the listed stages through the same runner as run_all.py: data root = the attached competition data, upstream
     kernel outputs = /kaggle/input (the original lookup  /kaggle/input/**/<kernel>/**/<file>), outputs ->
     /kaggle/working/tfb_work/kernels/<kernel>/;
  3. compares every listed output with the ORIGINAL kernel output attached as a kernel source and prints PARITY lines
     (queue rows and path_flow must be EXACT; state rows report max |d| and RMS), a PARITY SUMMARY line, and writes
     /kaggle/working/parity_<slug>.json.
Local dry run of a generated kernel:  TFB_KAGGLE_INPUT=<dir> TFB_KAGGLE_WORKING=<dir> [TFB_PARITY_SMOKE=1] python <file>
The kernels are never pushed by this script.
"""
import argparse
import ast
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tfb import config as C        # noqa: E402
import tfb                         # noqa: E402

OUT_DIR = os.path.join(HERE, 'kaggle_parity')
USER = 'hosen42'
CORE = ['tfb/__init__.py', 'tfb/config.py', 'tfb/runtime.py', 'tfb/runner.py', 'tfb/stage_exec.py', 'tfb/envinfo.py', 'tfb/parity.py',
        'tfb/stages/__init__.py']


def sub(ours, orig, tasks, label):
    return dict(type='submission', ours=list(ours), orig=list(orig), tasks=list(tasks), label=label)


def queue(ours, orig, label):
    return dict(type='queue', ours=list(ours), orig=list(orig), label=label)


def state(ours, orig, label):
    return dict(type='state', ours=list(ours), orig=list(orig), label=label)


def npz(ours, orig, label):
    return dict(type='npz', ours=list(ours), orig=list(orig), label=label)


def same(kernel, files, label_prefix, kind=queue):
    """rerun of ONE original kernel: each listed output against the same file of the original"""
    return [kind((kernel, f), (kernel, f), f'{label_prefix}_{f.rsplit(".", 1)[0]}') for f in files]


ALL = ('state', 'queue', 'odme')
MEMBERS_FULL = C.t1_member_kernels('full')
MEMBERS_LEAN1 = C.t1_member_kernels('lean1')
ONSET_SP = ['base', 'sp2', 'sp4', 'sp8']
ONSET_PV = ['base', 'sp4', 'pv', 'pvb']

SPECS = [
    dict(slug='tfb-par-t4', priority='re-push (fix)', minutes=20, stages=['t4'], pick='A2', recipe='full', overrides={},
         desc='Task 4 NNLS (tfb-final cell 7) -> skeleton at the CSV fixed point of path_flow: EXACT vs tfb-j4 / tfb-j3 required '
              '(vs tfb-final NUMERIC expected: ~390 values the 26-cycle chain moved by an ulp)',
         sources=['tfb-final', 'tfb-j4', 'tfb-j3'],
         checks=[sub(('tfb-final', 'submission.csv'), ('tfb-final', 'submission.csv'), ['odme'], 't4_vs_tfb-final'),
                 sub(('tfb-final', 'submission.csv'), ('tfb-j4', 'submission_J4.csv'), ['odme'], 't4_vs_tfb-j4'),
                 sub(('tfb-final', 'submission.csv'), ('tfb-j3', 'submission_J3.csv'), ['odme'], 't4_vs_tfb-j3')]),
    dict(slug='tfb-par-onsetsp', priority='required', minutes=10, stages=['onset_sp'], pick='B2', recipe='full', overrides={},
         desc='Task 2 onset CatBoost + same-scenario prior (tfb-t2osp): base / sp2 / sp4 / sp8 queue files exactly',
         sources=['tfb-t2osp'],
         checks=[queue(('tfb-t2osp', f'queue_submission_{v}.csv'), ('tfb-t2osp', f'queue_submission_{v}.csv'), f'onsetsp_{v}') for v in ONSET_SP]),
    dict(slug='tfb-par-onsetpv', priority='required', minutes=15, stages=['onset_pv'], pick='A2', recipe='full', overrides={},
         desc='Task 2 onset pv bag (tfb-t2opv): base / sp4 / pv / pvb exactly; base and sp4 also against tfb-t2osp',
         sources=['tfb-t2opv', 'tfb-t2osp'],
         checks=[queue(('tfb-t2opv', f'queue_submission_{v}.csv'), ('tfb-t2opv', f'queue_submission_{v}.csv'), f'onsetpv_{v}') for v in ONSET_PV]
         + [queue(('tfb-t2opv', f'queue_submission_{v}.csv'), ('tfb-t2osp', f'queue_submission_{v}.csv'), f'onsetpv_{v}_vs_osp') for v in ('base', 'sp4')]),
    dict(slug='tfb-par-t1mix', priority='required', minutes=15, stages=['t1_mix'], pick='A2', recipe='full',
         overrides={'t1_mix': {'base': ['orig:tfb-i4d', 'submission_T1DARK.csv']}},
         desc='T1 mix + blackout override (tfb-i4d) from the 31 original member outputs; every row vs tfb-i4d',
         sources=MEMBERS_FULL + ['tfb-i4d'],
         checks=[sub(('tfb-i4d', 'submission_T1DARK.csv'), ('tfb-i4d', 'submission_T1DARK.csv'), ALL, 't1mix_vs_tfb-i4d')]),
    dict(slug='tfb-par-t1recon', priority='required', minutes=20, stages=['t1_recon'], pick='A2', recipe='full', overrides={},
         desc='FD recon + 1-D TV (tfb-i4) on the original tfb-i4d file; FDR / FDRTV / TV vs tfb-i4',
         sources=['tfb-i4d', 'tfb-i4'],
         checks=[sub(('tfb-i4', f'submission_{v}.csv'), ('tfb-i4', f'submission_{v}.csv'), ALL, f't1recon_{v}') for v in ('FDRTV', 'FDR', 'TV')]),
    dict(slug='tfb-par-asma2', priority='required', minutes=8, stages=['assemble'], pick='A2', recipe='full', overrides={},
         desc='queue assembly A2 (4 condition x split parts on the original tfb-i4 file) vs tfb-j4 submission_J4.csv',
         sources=['tfb-i4', 'tfb-t2osp', 'tfb-t2occ', 'tfb-t2opv', 'tfb-t2ogpvogf', 'tfb-j4'],
         checks=[sub(('tfb-j4', 'submission_J4.csv'), ('tfb-j4', 'submission_J4.csv'), ALL, 'asm_A2_vs_tfb-j4')]),
    dict(slug='tfb-par-asmb2', priority='required', minutes=8, stages=['assemble'], pick='B2', recipe='full', overrides={},
         desc='queue assembly B2 (4 condition x split parts on the original tfb-i4 file) vs tfb-j3 submission_J3.csv',
         sources=['tfb-i4', 'tfb-t2osp', 'tfb-t2occ', 'tfb-t2ogveto', 'tfb-j3'],
         checks=[sub(('tfb-j3', 'submission_J3.csv'), ('tfb-j3', 'submission_J3.csv'), ALL, 'asm_B2_vs_tfb-j3')]),
    dict(slug='tfb-par-chaint1', priority='re-push (fix)', minutes=45, stages=['t4', 't1_mix', 't1_recon'], pick='A2', recipe='full', overrides={},
         desc='production wiring of the Task 1 / Task 4 rows: t4 skeleton -> t1_mix -> t1_recon; state + path_flow EXACT vs tfb-j4',
         sources=MEMBERS_FULL + ['tfb-j4'],
         checks=[sub(('tfb-i4', 'submission_FDRTV.csv'), ('tfb-j4', 'submission_J4.csv'), ['state', 'odme'], 'chaint1_vs_tfb-j4')]),
    dict(slug='tfb-par-chaint2a2', priority='integration', minutes=30, stages=['onset_sp', 'onset_pv', 'assemble'], pick='A2', recipe='full', overrides={},
         desc='A2 queue rows with the onset stages rerun (ongoing rows from the original producers) vs tfb-j4, every row',
         sources=['tfb-i4', 'tfb-t2occ', 'tfb-t2ogpvogf', 'tfb-j4'],
         checks=[sub(('tfb-j4', 'submission_J4.csv'), ('tfb-j4', 'submission_J4.csv'), ALL, 'chaint2_A2_vs_tfb-j4')]),
    dict(slug='tfb-par-chaint2b2', priority='integration', minutes=20, stages=['onset_sp', 'assemble'], pick='B2', recipe='full', overrides={},
         desc='B2 queue rows with the onset stage rerun (ongoing rows from the original producers) vs tfb-j3, every row',
         sources=['tfb-i4', 'tfb-t2occ', 'tfb-t2ogveto', 'tfb-j3'],
         checks=[sub(('tfb-j3', 'submission_J3.csv'), ('tfb-j3', 'submission_J3.csv'), ALL, 'chaint2_B2_vs_tfb-j3')]),
    dict(slug='tfb-par-lean1', priority='optional', minutes=30, stages=['t1_mix', 't1_recon'], pick='A2', recipe='lean1',
         overrides={'t1_mix': {'base': ['orig:tfb-i4d', 'submission_T1DARK.csv']}},
         desc='DESIGN 3.3 decision data: lean-1 mix (19 member outputs) + recon/TV; state RMS vs the full-recipe tfb-i4d / tfb-i4 (DIFF expected)',
         sources=MEMBERS_LEAN1 + ['tfb-i4d', 'tfb-i4'],
         checks=[sub(('tfb-i4d', 'submission_T1DARK.csv'), ('tfb-i4d', 'submission_T1DARK.csv'), ['state'], 'lean1_raw_vs_full_tfb-i4d'),
                 sub(('tfb-i4', 'submission_FDRTV.csv'), ('tfb-i4', 'submission_FDRTV.csv'), ['state'], 'lean1_final_vs_full_tfb-i4')]),
]


# ------------------------------------------------------------------ milestone 2: one original kernel per parity kernel
def member_spec(part):
    short = part[len('tfb-t1'):]
    rt = C.T1_PART_RUNTIME_S[part]
    fam = C.T1_MEMBERS[part[:-1]]['family']
    return dict(slug=f'tfb-par-{short}', priority='lean1 + full' if part in MEMBERS_LEAN1 else 'full only', minutes=int(rt / 60) + 12,
                stages=[f't1m:{part}'], pick='A2', recipe='full', overrides={},
                desc=f'Task 1 member part {part} (template t1_{fam}, original image {C.T1_MEMBERS[part[:-1]]["image"]}, {rt / 3600:.1f} h on Kaggle): '
                     f'state_submission_FTA.csv vs the original part',
                sources=['tfb-t2lab-full', part],
                checks=[state((part, 'state_submission_FTA.csv'), (part, 'state_submission_FTA.csv'), f'{short}_FTA')])


SPECS_M2 = [
    dict(slug='tfb-par-pseudo', priority='lean1 + full', minutes=8, stages=['pseudo'], pick='A2', recipe='full', overrides={},
         desc='pseudo windows (tfb-t2lab-full, hidden input of every Task 1 member): d / t / cond arrays exactly; the stage also self-checks counts + SHA-1',
         sources=['tfb-t2lab-full'],
         checks=same('tfb-t2lab-full', [f'pseudo_{p}.npz' for p in C.PANELS_T2], 'pseudo', npz)),
    dict(slug='tfb-par-occ', priority='A2 + B2', minutes=int(19630 / 60) + 15, stages=['ongoing_val'], pick='A2', recipe='full', overrides={},
         desc='March ongoing (tfb-t2occ): fs4occ / fs2occ / fs_np queue files exactly + probabilities and qlast (t2v26occ_probs.npz)',
         sources=['tfb-t2occ'],
         checks=same('tfb-t2occ', ['queue_submission_fs4occ.csv', 'queue_submission_fs2occ.csv', 'queue_submission_fs_np.csv'], 'occ')
         + same('tfb-t2occ', ['t2v26occ_probs.npz'], 'occ', npz)),
    dict(slug='tfb-par-pvogf', priority='A2', minutes=int(16399 / 60) + 15, stages=['ongoing_pvogf'], pick='A2', recipe='full',
         overrides={'ongoing_pvogf': {'ref_v22': ['tfb-t2v22af', 'queue_submission_fs4af.csv']}},
         desc='April ongoing of A2 (tfb-t2ogpvogf) with the original references attached (fs4af validation copy, pvo veto): pvogf / pvogf_npid / pvogfv exactly + probabilities',
         sources=['tfb-t2ogpvogf', 'tfb-t2v22af', 'tfb-t2ogpvaf'],
         checks=same('tfb-t2ogpvogf', ['queue_submission_pvogf.csv', 'queue_submission_pvogf_npid.csv', 'queue_submission_pvogfv.csv'], 'pvogf')
         + same('tfb-t2ogpvogf', ['t2pvogf_probs.npz'], 'pvogf', npz)),
    dict(slug='tfb-par-ogfa', priority='B2', minutes=int(18352 / 60) + 15, stages=['ogfeat_a'], pick='B2', recipe='full', overrides={},
         desc='ogfeat seed-0 pair (tfb-t2ogfa, template t2_ogfeat part a): fs2ogf / fs_npogf exactly + probabilities',
         sources=['tfb-t2ogfa'],
         checks=same('tfb-t2ogfa', ['queue_submission_fs2ogf.csv', 'queue_submission_fs_npogf.csv'], 'ogfa') + same('tfb-t2ogfa', ['t2v27a_probs.npz'], 'ogfa', npz)),
    dict(slug='tfb-par-ogfb', priority='B2', minutes=int(22927 / 60) + 15, stages=['ogfeat_b'], pick='B2', recipe='full', overrides={},
         desc='ogfeat seed-1 pair (tfb-t2ogfb, template t2_ogfeat part b): fs2ogf1 exactly + probabilities',
         sources=['tfb-t2ogfb'],
         checks=same('tfb-t2ogfb', ['queue_submission_fs2ogf1.csv'], 'ogfb') + same('tfb-t2ogfb', ['t2v27b_probs.npz'], 'ogfb', npz)),
    dict(slug='tfb-par-ogfc', priority='B2', minutes=8, stages=['ogfeat_comb'], pick='B2', recipe='full', overrides={},
         desc='fs4ogf = mean of the 4 ogfeat models (tfb-t2ogfc) from the original ogfa / ogfb outputs: exactly',
         sources=['tfb-t2ogfa', 'tfb-t2ogfb', 'tfb-t2ogfc'],
         checks=same('tfb-t2ogfc', ['queue_submission_fs4ogf.csv'], 'ogfc')),
    dict(slug='tfb-par-ogveto', priority='B2', minutes=8, stages=['ogveto'], pick='B2', recipe='full', overrides={},
         desc='veto10 (tfb-t2ogveto) from the original fs4occ / qlast / fs4ogf: exactly',
         sources=['tfb-t2occ', 'tfb-t2ogfc', 'tfb-t2ogveto'],
         checks=same('tfb-t2ogveto', ['queue_submission_veto10.csv'], 'ogveto')),
] + [member_spec(p) for p in C.t1_member_kernels('full')]
SPECS = SPECS + SPECS_M2


def modules_for(spec):
    mods = list(CORE)
    for st in spec['stages']:
        m = C.stage_module(st)
        if m:
            mods.append(f'tfb/stages/{m}.py')
    return list(dict.fromkeys(mods))


def tree_sha1(paths):
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.encode())
        h.update(open(os.path.join(HERE, p), 'rb').read())
    return h.hexdigest()


def literal(src):
    assert src.endswith('\n')
    return "'''" + src.replace('\\', '\\\\').replace("'''", "\\'\\'\\'") + "'''"


RUNTIME_MAIN = r'''

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
'''


def generate(spec):
    mods = modules_for(spec)
    sha = tree_sha1(mods)
    full = dict(spec, tfb_version=tfb.__version__, tree_sha1=sha, generated=time.strftime('%Y-%m-%d %H:%M'))
    srcs = ' '.join(f'{USER}/{s}' for s in spec['sources'])
    head = [
        '# ' + '=' * 118,
        '# GENERATED FILE - DO NOT EDIT. Built by consolidation/bundle.py from the tfb package',
        f"#   tfb {tfb.__version__} | modules tree sha1 {sha} | generated {full['generated']}",
        f"# Parity kernel {spec['slug']} ({spec['priority']}, ~{spec['minutes']} min on Kaggle CPU): {spec['desc']}",
        f"# Stages {spec['stages']} | pick {spec['pick']} | recipe {spec['recipe']} | overrides {json.dumps(spec['overrides'])}",
        f"# Kernel sources to attach ({len(spec['sources'])}, plus the competition data):",
    ]
    line = '#  '
    for s in spec['sources']:
        if len(line) + len(s) + 9 > 120:
            head.append(line); line = '#  '
        line += f' {USER}/{s}'
    head += [line,
             f"# Push: python kaggle_jobs/kjob.py push {spec['slug']} consolidation/kaggle_parity/{spec['slug']}.py {srcs}",
             '# Output: PARITY lines + PARITY SUMMARY in the log; /kaggle/working/parity_<slug>.json; stage outputs in /kaggle/working/tfb_work/',
             '# ' + '=' * 118,
             'import glob, json, os, sys, tempfile, time', '',
             f'SPEC = json.loads(r\'\'\'{json.dumps(full)}\'\'\')', '',
             'MODULES = {}']
    body = [f'MODULES[{p!r}] = {literal(open(os.path.join(HERE, p)).read())}' for p in mods]
    text = '\n'.join(head) + '\n' + '\n'.join(body) + RUNTIME_MAIN
    # self-check: the literals reproduce every module byte for byte and the file parses
    tree = ast.parse(text)
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            k = ast.literal_eval(node.targets[0].slice if not isinstance(node.targets[0].slice, ast.Index) else node.targets[0].slice.value)
            got[k] = ast.literal_eval(node.value)
    assert got == {p: open(os.path.join(HERE, p)).read() for p in mods}, 'inlined module differs from the package'
    path = os.path.join(OUT_DIR, f"{spec['slug']}.py")
    with open(path, 'w') as f:
        f.write(text)
    return path, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    only = set(x for x in a.only.split(',') if x)
    done = []
    for spec in SPECS:
        if only and spec['slug'] not in only:
            continue
        path, full = generate(spec)
        done.append(full)
        print(f"wrote {os.path.relpath(path, HERE)}  ({os.path.getsize(path) // 1024} KB, {len(spec['sources'])} kernel sources)")
    if not only:
        with open(os.path.join(OUT_DIR, 'sources.json'), 'w') as f:
            json.dump({s['slug']: dict(priority=s['priority'], minutes=s['minutes'], stages=s['stages'], pick=s['pick'], recipe=s['recipe'],
                                       sources=[f'{USER}/{x}' for x in s['sources']], desc=s['desc']) for s in done}, f, indent=1)
        lines = ['# Kaggle parity kernels (GENERATED by consolidation/bundle.py - do not edit)', '',
                 'Each kernel is a single file. Attach the competition data and the kernel sources listed below, then read the',
                 '`PARITY ...` lines and the final `PARITY SUMMARY verdict=...` line of the log.',
                 '- Queue rows and path_flow must be EXACT.',
                 '- State rows report max |d| and RMS; EXACT is expected on the image the original ran on.',
                 '- Milestone-2 kernels each rerun ONE original kernel and compare its outputs with that kernel\'s own files.',
                 '- `~min` = original Kaggle runtime + compare overhead (4 vCPU / 30 GB; 12 h limit; 5 sessions in parallel).', '']
        groups = [('Milestone 1 - already PASS on Kaggle (2026-10-09); nothing to push', lambda x: x['slug'] in M1_PASSED),
                  ('Milestone 1 - re-push after the path_flow fix', lambda x: x['priority'].startswith('re-push')),
                  ('Milestone 1 - optional / done', lambda x: x['slug'] in ('tfb-par-lean1',)),
                  ('Milestone 2 - pseudo windows and Task 2 ongoing', lambda x: x['slug'] in M2_T2),
                  ('Milestone 2 - Task 1 member parts of the lean-1 recipe (also in full)', lambda x: x['priority'] == 'lean1 + full' and x['slug'] not in M2_T2),
                  ('Milestone 2 - Task 1 member parts of the full recipe only', lambda x: x['priority'] == 'full only')]
        for title, sel in groups:
            g = [x for x in done if sel(x)]
            if not g:
                continue
            lines += [f'## {title} ({len(g)} kernels, {sum(x["minutes"] for x in g) / 60:.1f} Kaggle-hours)', '',
                      '| Kernel | Priority | ~min | Stages (pick, recipe) | What it checks | Kernel sources |', '|---|---|---:|---|---|---|']
            for x in g:
                lines.append(f"| `{x['slug']}` | {x['priority']} | {x['minutes']} | {', '.join(x['stages'])} ({x['pick']}, {x['recipe']}) | {x['desc']} | "
                             f"{len(x['sources'])}: {_src_note(x)} |")
            lines.append('')
        lines += ['', '## Push commands', '', '```']
        for s in done:
            lines.append(f"python kaggle_jobs/kjob.py push {s['slug']} consolidation/kaggle_parity/{s['slug']}.py " + ' '.join(f'{USER}/{x}' for x in s['sources']))
        lines += ['```', '']
        with open(os.path.join(OUT_DIR, 'SOURCES.md'), 'w') as f:
            f.write('\n'.join(lines))
        print('wrote kaggle_parity/SOURCES.md, kaggle_parity/sources.json')


M1_PASSED = ('tfb-par-onsetsp', 'tfb-par-onsetpv', 'tfb-par-t1mix', 'tfb-par-t1recon', 'tfb-par-asma2', 'tfb-par-asmb2',
             'tfb-par-chaint2a2', 'tfb-par-chaint2b2')
M2_T2 = ('tfb-par-pseudo', 'tfb-par-occ', 'tfb-par-pvogf', 'tfb-par-ogfa', 'tfb-par-ogfb', 'tfb-par-ogfc', 'tfb-par-ogveto')


def _src_note(s):
    mem = [x for x in s['sources'] if x.startswith('tfb-t1')]
    other = [x for x in s['sources'] if not x.startswith('tfb-t1')]
    note = ', '.join(other)
    if mem and len(mem) <= 2:
        return ', '.join(s['sources'])
    if mem:
        bag = 'full' if len(mem) == len(MEMBERS_FULL) else s['recipe']
        note += f' + {len(mem)} Task 1 member parts of the {bag} recipe (a/b, bspec a/b/c; see the push command)'
    return note


if __name__ == '__main__':
    main()
