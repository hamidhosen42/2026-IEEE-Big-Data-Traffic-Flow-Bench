#!/usr/bin/env python3
"""Level-0 checks (DESIGN 4.7): no Kaggle run, no data.

1. vendored stage files differ from the original kernel source (consolidation/lineage_src/) only in lines marked [TFB]
   (a deleted original line must sit next to a [TFB] line), and contain no /kaggle/ path;
2. every constant in tfb/config.py equals the literal in the original kernel / chain state / lineage_A2.json;
3. lineage_src files still match the SHA-1 recorded in lineage_A2.json.
Run:  python tests/test_static.py   (exit code 1 on any failure; also collectable by pytest)
"""
import ast
import difflib
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)
ROOT = os.path.dirname(CONS)
sys.path.insert(0, CONS)
from tfb import config as C        # noqa: E402

LSRC = f'{CONS}/lineage_src'
LINEAGE = json.load(open(f'{CONS}/lineage_A2.json'))
VENDORED = {                      # stage module -> original kernel source
    't1_mix': 'tfb-i4d.py', 't1_recon': 'tfb-i4.py', 'assemble': 'tfb-j3.py',
    'onset_sp': 'tfb-t2osp.py', 'onset_pv': 'tfb-t2opv.py', 't4_nnls': 'tfb-final.py#cell7',
}


def original_source(name):
    if name.endswith('#cell7'):
        nb = json.load(open(f'{LSRC}/{name.split("#")[0]}'))
        src = nb['cells'][7]['source']
        src = ''.join(src) if isinstance(src, list) else src
        return src if src.endswith('\n') else src + '\n'
    return open(f'{LSRC}/{name}').read()


def module_consts(src, names):
    """values of simple top-level assignments NAME = <constant expression> (tuple targets supported)."""
    out = {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in names:
                out[tgt.id] = eval(compile(ast.Expression(node.value), '<c>', 'eval'), {})
            elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for t, v in zip(tgt.elts, node.value.elts):
                    if isinstance(t, ast.Name) and t.id in names:
                        out[t.id] = eval(compile(ast.Expression(v), '<c>', 'eval'), {})
    return out


def check(cond, msg, fails):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        fails.append(msg)


def test_vendored_diff():
    fails = []
    for mod, orig in VENDORED.items():
        a = original_source(orig).splitlines()
        b = open(f'{CONS}/tfb/stages/{mod}.py').read().splitlines()
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        patched, deleted, bad = 0, 0, []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'insert'):
                for j in range(j1, j2):
                    patched += 1
                    if '[TFB]' not in b[j]:
                        bad.append(f'unmarked line {j + 1}: {b[j][:90]}')
            if tag in ('replace', 'delete'):
                deleted += i2 - i1
                if tag == 'delete':
                    near = [b[j] for j in (j1 - 1, j1) if 0 <= j < len(b)]
                    if not any('[TFB]' in x for x in near):
                        bad.append(f'deleted original lines {i1 + 1}-{i2} without an adjacent [TFB] line')
        kag = [i + 1 for i, line in enumerate(b) if '/kaggle/' in line]
        check(not bad and not kag, f'vendored {mod:9s} vs {orig:22s}: {patched} patched/added lines, {deleted} original lines replaced/removed'
              + (f' | {bad[:3]}' if bad else '') + (f' | /kaggle/ paths on lines {kag}' if kag else ''), fails)
    assert not fails, fails


def test_config_constants():
    fails = []
    i4d = module_consts(original_source('tfb-i4d.py'), {'W_FD', 'W_DARK', 'DARK_MEMBER', 'FD_MEMBERS', 'MEMBERS', 'BASE_KERNEL', 'BASE_FILE'})
    check(i4d['W_FD'] == C.T1_MIX['w_fd'] and i4d['W_DARK'] == C.T1_MIX['w_dark'], f'T1 mix weights W_FD {i4d["W_FD"]} W_DARK {i4d["W_DARK"]}', fails)
    full = C.T1_BAGS['full']
    check(i4d['MEMBERS'] == full['old_fta'] and i4d['FD_MEMBERS'] == full['fd'] and i4d['DARK_MEMBER'] == full['dark'],
          'full recipe bags = tfb-i4d MEMBERS / FD_MEMBERS / DARK_MEMBER', fails)
    lean = C.T1_BAGS['lean1']
    check(set(lean['old_fta']) <= set(full['old_fta']) and set(lean['fd']) <= set(full['fd']) and lean['dark'] == full['dark'],
          'lean1 bags are subsets of the full bags (DESIGN 3.3: old v12s4-s7, FD v13/s10/s11/t1ramp)', fails)
    i4 = module_consts(original_source('tfb-i4.py'), {'DO_RECON', 'DO_TV', 'GATE', 'A', 'LOGR_MAX', 'TV_FREE', 'TV_FREE_A', 'TV_GATE', 'TV_GATE_A',
                                                      'TV_ITERS', 'RUN_GAP_NS', 'W_RANGE', 'KJL_RANGE', 'EXCL_MED', 'EXCL_IQR', 'EXCL_MIN_N',
                                                      'BASE_KERNEL', 'BASE_FILE'})
    R = C.RECON_TV
    pairs = dict(DO_RECON='do_recon', DO_TV='do_tv', GATE='gate', A='a', LOGR_MAX='logr_max', TV_FREE='tv_free', TV_FREE_A='tv_free_a',
                 TV_GATE='tv_gate', TV_GATE_A='tv_gate_a', TV_ITERS='tv_iters', RUN_GAP_NS='run_gap_ns', W_RANGE='w_range',
                 KJL_RANGE='kjl_range', EXCL_MED='excl_med', EXCL_IQR='excl_iqr', EXCL_MIN_N='excl_min_n')
    bad = [k for k, v in pairs.items() if i4[k] != R[v]]
    check(not bad, f'recon / TV constants = tfb-i4 ({len(pairs)} constants){" | differ: " + str(bad) if bad else ""}', fails)
    check((i4['BASE_KERNEL'], i4['BASE_FILE']) == R['base'], f'recon base = tfb-i4 base {R["base"]}', fails)
    st = json.load(open(f'{ROOT}/kaggle_jobs/auto_chain_oct9_state.json'))['recon']
    check([st['gate'], st['a'], st['logr']] == [R['gate'], R['a'], R['logr_max']] and st['tv'] == [R['tv_free'], R['tv_free_a'], R['tv_gate'], R['tv_gate_a']],
          'recon / TV constants = chain state auto_chain_oct9_state.json', fails)
    j3 = module_consts(original_source('tfb-j3.py'), {'PARTS', 'BASE_KERNEL', 'BASE_FILE'})
    check([tuple(p) for p in j3['PARTS']] == C.QUEUE_PARTS_ORIGINAL['B2'], 'B2 private parts = tfb-j3 PARTS', fails)
    check((j3['BASE_KERNEL'], j3['BASE_FILE']) == C.ASSEMBLE['base'], f'assemble base = tfb-j3 base {C.ASSEMBLE["base"]}', fails)
    j4 = module_consts(open(f'{ROOT}/kaggle_jobs/swap_tfb-j4.py').read(), {'PARTS', 'BASE_KERNEL', 'BASE_FILE'})
    check([tuple(p) for p in j4['PARTS']] == C.QUEUE_PARTS_ORIGINAL['A2'] and (j4['BASE_KERNEL'], j4['BASE_FILE']) == C.ASSEMBLE['base'],
          'A2 private parts / base = kaggle_jobs/swap_tfb-j4.py (pushed as tfb-j4 v1)', fails)
    for pick in C.PICKS:
        priv = [(k, f.split('|')[0], c, s) for k, f, c, s in C.QUEUE_PARTS[pick] if s == 'private']
        check(priv == C.QUEUE_PARTS_ORIGINAL[pick] or sorted(priv) == sorted(C.QUEUE_PARTS_ORIGINAL[pick] + ([('tfb-t2opv', 'queue_submission_pv.csv', 'queue_onset', 'private')] if pick == 'A2' else [])),
              f'{pick} private parts of the assembly = original swap parts (+ g4 pv onset for A2)', fails)
    e5 = module_consts(original_source('tfb-e5.py'), {'SRC_KERNEL', 'SRC_FILE', 'COND'})
    g3 = module_consts(original_source('tfb-g3.py'), {'PARTS'})
    g4 = module_consts(original_source('tfb-g4.py'), {'PARTS'})
    val = [p for p in C.QUEUE_PARTS['A2'] if p[3] == 'validation']
    check(val[0][:3] == (e5['SRC_KERNEL'], e5['SRC_FILE'], e5['COND']) and tuple(g3['PARTS'][0]) == val[1],
          'validation parts = tfb-e5 (osp sp4 onset) and tfb-g3 (t2occ fs4occ ongoing)', fails)
    check(tuple(g4['PARTS'][0]) == C.QUEUE_PARTS['A2'][2], 'A2 private onset = tfb-g4 part 1 (t2opv pv)', fails)
    for name in ('tfb-t2osp.py', 'tfb-t2opv.py'):
        s = original_source(name)
        cb = 'CatBoostClassifier(iterations=600, depth=8, learning_rate=0.08, thread_count=4, verbose=0, random_seed=0)'
        check(cb in s and "np.random.default_rng(0).choice(len(_Xo), min(len(_Xo), 1_200_000), replace=False)" in s
              and 'MAX_OG = 300 if SMOKE else 3000' in s, f'{name}: CatBoost params / subsample / MAX_OG = config ONSET', fails)
    pv = module_consts(original_source('tfb-t2opv.py'), {'K_BLEND', 'W_MARCH'})
    sp = module_consts(original_source('tfb-t2osp.py'), {'KS'})
    check(pv['K_BLEND'] == C.ONSET['k_blend'] and pv['W_MARCH'] == C.ONSET['w_march'] and sp['KS'] == C.ONSET['ks_sp'], 'onset K_BLEND / W_MARCH / KS', fails)
    check('lam = 0.05' in original_source('tfb-final.py#cell7') and C.T4['lam'] == 0.05, 'T4 lambda 0.05', fails)
    pk = LINEAGE['picks']['A (tfb-i4, public 0.88176)']['t1_state']
    check(tuple(pk['raw_mix']['old_fta_bag']) == full['old_fta'] and tuple(pk['raw_mix']['fd_bag']) == full['fd']
          and pk['raw_mix']['W_FD'] == C.T1_MIX['w_fd'] and pk['blackout_rows']['W_DARK'] == C.T1_MIX['w_dark'], 'bags / weights = lineage_A2.json picks', fails)
    rt = pk['recon_tv']
    check([rt['gate'], rt['a'], rt['logr_max'], rt['tv_free'], rt['tv_free_anchor'], rt['tv_gate'], rt['tv_gate_anchor'], rt['tv_iters']]
          == [R['gate'], R['a'], R['logr_max'], R['tv_free'], R['tv_free_a'], R['tv_gate'], R['tv_gate_a'], R['tv_iters']], 'recon / TV = lineage_A2.json picks', fails)
    for k, m in C.T1_MEMBERS.items():
        lm = LINEAGE['t1_members'].get(k)
        if lm is None:
            fails.append(f'{k} missing in lineage'); continue
        ok = lm['row_subsample_rng'] == m['row_rng'] and (lm.get('lgb_seeds') is None) == (m['lgb_seeds'] is None)
        ok &= m['lgb_seeds'] is None or tuple(lm['lgb_seeds']) == m['lgb_seeds']
        if not ok:
            check(False, f'T1 member {k} row rng / LightGBM seeds = lineage_A2.json', fails)
    check(True, f'T1 member table ({len(C.T1_MEMBERS)} members): row rng + LightGBM seeds checked against lineage_A2.json', fails)
    for st_, k in (('onset_sp', 'tfb-t2osp'), ('onset_pv', 'tfb-t2opv'), ('t1_mix', 'tfb-i4d'), ('t1_recon', 'tfb-i4')):
        lo = set(LINEAGE['kernels'][k]['outputs'] or [])
        check(set(C.stage_outputs(st_, 'A2')) <= lo, f'stage {st_} outputs are outputs of {k}', fails)
    assert not fails, fails


def test_lineage_sources():
    fails, n = [], 0
    for k, v in LINEAGE['kernels'].items():
        p = f'{ROOT}/{v["source"]}'
        if os.path.exists(p):
            n += 1
            if hashlib.sha1(open(p, 'rb').read()).hexdigest() != v['source_sha1']:
                fails.append(k)
    check(not fails, f'lineage_src SHA-1 = lineage_A2.json for {n} kernels{" | differ: " + str(fails) if fails else ""}', fails)
    assert not fails, fails


if __name__ == '__main__':
    bad = 0
    for t in (test_vendored_diff, test_config_constants, test_lineage_sources):
        try:
            t()
        except AssertionError:
            bad += 1
    print('STATIC', 'FAILED' if bad else 'OK')
    sys.exit(1 if bad else 0)
