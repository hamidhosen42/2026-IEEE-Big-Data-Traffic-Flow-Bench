#!/usr/bin/env python3
"""Level-0 checks (DESIGN 4.7): no Kaggle run, no data.

1. vendored stage files differ from the original kernel source (consolidation/lineage_src/) only in lines marked [TFB]
   (a deleted original line must sit next to a [TFB] line), and contain no /kaggle/ path;
2. every constant in tfb/config.py equals the literal in the original kernel / chain state / lineage_A2.json;
3. lineage_src files still match the SHA-1 recorded in lineage_A2.json / lineage_K9.json.
Run:  python tests/test_static.py   (exit code 1 on any failure; also collectable by pytest)
"""
import ast
import difflib
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)
ROOT = os.path.dirname(CONS)
sys.path.insert(0, CONS)
from tfb import config as C        # noqa: E402

LSRC = f'{CONS}/lineage_src'
LINEAGE = json.load(open(f'{CONS}/lineage_A2.json'))
LINEAGE_K9 = json.load(open(f'{CONS}/lineage_K9.json'))
VENDORED = {                      # stage module -> original kernel source
    't1_mix': 'tfb-i4d.py', 't1_recon': 'tfb-i4.py', 'assemble': 'tfb-j3.py',
    'onset_sp': 'tfb-t2osp.py', 'onset_pv': 'tfb-t2opv.py', 't4_nnls': 'tfb-final.py#cell7',
    # milestone 2 (templates: vs their representative part; every part is checked by tests/test_templates.py)
    'pseudo': 'tfb-t2lab-full.py', 't1_fta': 'tfb-t1v12s4a.py', 't1_fta_v11': 'tfb-t1v11a.py', 't1_fd': 'tfb-t1v13a.py',
    't1_dark': 'tfb-t1darka.py', 't1_ramp': 'tfb-t1rampa.py', 't1_bspec': 'tfb-t1bspeca.py',
    't2_occ': 'tfb-t2occ.py', 't2_pvogf': 'tfb-t2ogpvogf.py', 't2_ogfeat': 'tfb-t2ogfa.py', 't2_ogfc': 'tfb-t2ogfc.py', 't2_ogveto': 'tfb-t2ogveto.py',
    # picks A3 / B3 (k generation)
    't1_mix_k5d': 'tfb-k5d.py', 't1_lrtc': 'tfb-k9.py', 'assemble_k10': 'tfb-k10.py', 't1_catb': 'tfb-t1catba.py', 't1_xgb': 'tfb-t1xgba.py',
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
                    if '[TFB]' not in b[j] and b[j].strip():       # blank separator lines need no marker
                        bad.append(f'unmarked line {j + 1}: {b[j][:90]}')
            if tag in ('replace', 'delete'):
                deleted += i2 - i1
                if tag == 'delete':
                    near = [b[j] for j in (j1 - 1, j1) if 0 <= j < len(b)]
                    if not any('[TFB]' in x for x in near):
                        bad.append(f'deleted original lines {i1 + 1}-{i2} without an adjacent [TFB] line')
        kag = [i + 1 for i, line in enumerate(b) if re.search(r"['\"]/kaggle/", line)]   # a quoted /kaggle path (prose in docstrings is fine)
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
        orig = C.QUEUE_PARTS_ORIGINAL['A2' if pick == 'A3' else pick]     # A3: tfb-k9 inherits tfb-j4's queue rows (no own swap)
        check(priv == orig or sorted(priv) == sorted(orig + ([('tfb-t2opv', 'queue_submission_pv.csv', 'queue_onset', 'private')] if pick in C.A_PICKS else [])),
              f'{pick} private parts of the assembly = original swap parts (+ g4 pv onset for A2 / A3)', fails)
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
        if k in C.k_members():
            continue                                     # k generation: checked against the sources in test_templates / test_config_constants_k9
        lm = LINEAGE['t1_members'].get(k)
        if lm is None:
            fails.append(f'{k} missing in lineage'); continue
        ok = lm['row_subsample_rng'] == m['row_rng'] and (lm.get('lgb_seeds') is None) == (m['lgb_seeds'] is None)
        ok &= m['lgb_seeds'] is None or tuple(lm['lgb_seeds']) == m['lgb_seeds']
        if not ok:
            check(False, f'T1 member {k} row rng / LightGBM seeds = lineage_A2.json', fails)
    check(True, f'T1 member table ({len(C.T1_MEMBERS) - len(C.k_members())} A2 / B2 members): row rng + LightGBM seeds checked against lineage_A2.json', fails)
    for st_, k in (('onset_sp', 'tfb-t2osp'), ('onset_pv', 'tfb-t2opv'), ('t1_mix', 'tfb-i4d'), ('t1_recon', 'tfb-i4')):
        lo = set(LINEAGE['kernels'][k]['outputs'] or [])
        check(set(C.stage_outputs(st_, 'A2')) <= lo, f'stage {st_} outputs are outputs of {k}', fails)
    assert not fails, fails


def test_config_constants_k9():
    """constants of the A3 / B3 stages = the literals of tfb-k5d / tfb-k9 / tfb-k10 and the chain bases (lineage_K9.json)."""
    fails = []
    k5d = module_consts(original_source('tfb-k5d.py'), {'BASE_KERNEL', 'BASE_FILE', 'OUT_NAME', 'CAND_MEMBER', 'CAND_PARTS', 'W_CAND', 'CATB_MEMBER',
                                                       'CATB_PARTS', 'W_CATB', 'W_FD', 'W_DARK', 'DARK_MEMBER', 'FD_MEMBERS', 'MEMBERS'})
    M = C.T1_MIX_K5D
    check([k5d[k] for k in ('W_FD', 'W_DARK', 'OUT_NAME', 'CATB_MEMBER', 'CATB_PARTS', 'W_CATB', 'CAND_MEMBER', 'CAND_PARTS', 'W_CAND')]
          == [M[k] for k in ('w_fd', 'w_dark', 'out_name', 'catb_member', 'catb_parts', 'w_catb', 'cand_member', 'cand_parts', 'w_cand')],
          'T1 mix K5D constants (W_FD, W_DARK, CatBoost / XGBoost members, parts, weights) = tfb-k5d', fails)
    full = C.T1_BAGS['full']
    check(k5d['MEMBERS'] == full['old_fta'] and k5d['FD_MEMBERS'] == full['fd'] and k5d['DARK_MEMBER'] == full['dark'],
          'full recipe bags = tfb-k5d MEMBERS / FD_MEMBERS / DARK_MEMBER', fails)
    check(M['base'] == C.T1_MIX['base'] and (k5d['BASE_KERNEL'], k5d['BASE_FILE']) == ('tfb-k3', 'submission_FDRTV.csv'),
          'tfb-k5d base tfb-k3 (queue / path rows only) -> T4 skeleton as for tfb-i4d', fails)
    k9 = module_consts(original_source('tfb-k9.py'), {'DO_RECON', 'DO_TV', 'GATE', 'A', 'LOGR_MAX', 'TV_FREE', 'TV_FREE_A', 'TV_GATE', 'TV_GATE_A',
                                                      'TV_ITERS', 'RUN_GAP_NS', 'W_RANGE', 'KJL_RANGE', 'EXCL_MED', 'EXCL_IQR', 'EXCL_MIN_N',
                                                      'BASE_KERNEL', 'BASE_FILE', 'W_LRTC', 'LRTC_RANKS', 'LRTC_ITERS', 'W_BLK', 'BLK_V_RANKS',
                                                      'BLK_Q_RANKS', 'LRTC_MIN_OBS', 'WRITE_FDR', 'WRITE_TV'})
    cfg = C.stage_cfg('t1_lrtc', 'A3', 'full')
    pairs = dict(DO_RECON='do_recon', DO_TV='do_tv', GATE='gate', A='a', LOGR_MAX='logr_max', TV_FREE='tv_free', TV_FREE_A='tv_free_a',
                 TV_GATE='tv_gate', TV_GATE_A='tv_gate_a', TV_ITERS='tv_iters', RUN_GAP_NS='run_gap_ns', W_RANGE='w_range',
                 KJL_RANGE='kjl_range', EXCL_MED='excl_med', EXCL_IQR='excl_iqr', EXCL_MIN_N='excl_min_n', W_LRTC='w_lrtc',
                 LRTC_RANKS='lrtc_ranks', LRTC_ITERS='lrtc_iters', W_BLK='w_blk', BLK_V_RANKS='blk_v_ranks', BLK_Q_RANKS='blk_q_ranks',
                 LRTC_MIN_OBS='lrtc_min_obs')
    bad = [k for k, v in pairs.items() if k9[k] != cfg[v]]
    check(not bad and (k9['BASE_KERNEL'], k9['BASE_FILE']) == tuple(cfg['base']) and not k9['WRITE_FDR'] and not k9['WRITE_TV'],
          f'steps M / B / R / T constants + base = tfb-k9 ({len(pairs)} constants){" | differ: " + str(bad) if bad else ""}', fails)
    k5 = module_consts(original_source('tfb-k5.py'), set(pairs) - {'W_LRTC', 'LRTC_RANKS', 'LRTC_ITERS', 'W_BLK', 'BLK_V_RANKS', 'BLK_Q_RANKS', 'LRTC_MIN_OBS'})
    check(all(k5[k] == k9[k] for k in k5), 'R / T constants of tfb-k9 = tfb-k5 = tfb-i4 (RECON_TV)', fails)
    k10 = module_consts(original_source('tfb-k10.py'), {'PARTS', 'BASE_KERNEL', 'BASE_FILE', 'OUT_NAME'})
    check([tuple(p) for p in k10['PARTS']] == C.QUEUE_PARTS_ORIGINAL['B3'] and (k10['BASE_KERNEL'], k10['BASE_FILE']) == C.ASSEMBLE_BASE['B3']
          and k10['OUT_NAME'] == C.PICK_KERNEL['B3'][1], 'B3 private parts / base / name = tfb-k10', fails)
    check(C.ASSEMBLE_BASE['A3'] == C.ASSEMBLE_BASE['B3'] == C.PICK_REFERENCE['A3'], 'A3 / B3 assembly base = tfb-k9/submission_FDRTV.csv', fails)
    # the chain that carries tfb-j4's queue rows into tfb-k9 (each kernel's base) and its queue-row invariance gates
    chain = [('tfb-k9.py', ('tfb-k5d', 'submission_K5D.csv')), ('tfb-k5d.py', ('tfb-k3', 'submission_FDRTV.csv')),
             ('tfb-k3.py', ('tfb-k3d', 'submission_K3D.csv')), ('tfb-k3d.py', ('tfb-j4', 'submission_J4.csv'))]
    ok = True
    for f, b in chain:
        c = module_consts(original_source(f), {'BASE_KERNEL', 'BASE_FILE'})
        ok &= (c['BASE_KERNEL'], c['BASE_FILE']) == b and "assert (out.loc[isq, 'queue_pred'].to_numpy() == base.loc[isq, 'queue_pred'].to_numpy()).all()" in original_source(f)
    check(ok, 'base chain tfb-j4 -> k3d -> k3 -> k5d -> k9, each kernel asserts queue_pred == its base', fails)
    j4 = module_consts(original_source('tfb-j4.py'), {'PARTS', 'BASE_KERNEL', 'BASE_FILE'})
    check([tuple(p) for p in j4['PARTS']] == C.QUEUE_PARTS_ORIGINAL['A2'] and C.QUEUE_PARTS['A3'] == C.QUEUE_PARTS['A2'] and C.QUEUE_PARTS['B3'] == C.QUEUE_PARTS['B2'],
          'A3 queue parts = A2 (tfb-j4 rows), B3 = B2', fails)
    for k in ('tfb-t1catb', 'tfb-t1catb2', 'tfb-t1xgb'):
        for part in 'abcd':
            src = original_source(f'{k}{part}.py')
            sel = re.search(r"^PANELS = (\[[^\]]*\])\s+# t1 ?(?:catb|xgb) part " + part + "$", src, flags=re.M)
            if sel is None or eval(sel.group(1)) != C.t1_part_panels(k, part):
                check(False, f'{k}{part} panels = QUAD_PANELS', fails)
    check(True, 'k-generation member panels (12 parts) = QUAD_PANELS[part]', fails)
    for k in C.k_members():
        m = C.T1_MEMBERS[k]
        rt = [LINEAGE_K9['kernels'][f'{k}{p}']['runtime_s'] for p in 'abcd']
        check(rt == [C.T1_PART_RUNTIME_S[f'{k}{p}'] for p in 'abcd'], f'{k}: part runtimes = lineage_K9.json {rt}', fails)
    assert not fails, fails


def test_lineage_sources():
    fails, n = [], 0
    for k, v in list(LINEAGE['kernels'].items()) + list(LINEAGE_K9['kernels'].items()):
        p = f'{ROOT}/{v["source"]}'
        if os.path.exists(p):
            n += 1
            if hashlib.sha1(open(p, 'rb').read()).hexdigest() != v['source_sha1']:
                fails.append(k)
    check(not fails, f'lineage_src SHA-1 = lineage_A2.json / lineage_K9.json for {n} kernels{" | differ: " + str(fails) if fails else ""}', fails)
    assert not fails, fails


if __name__ == '__main__':
    bad = 0
    for t in (test_vendored_diff, test_config_constants, test_config_constants_k9, test_lineage_sources):
        try:
            t()
        except AssertionError:
            bad += 1
    print('STATIC', 'FAILED' if bad else 'OK')
    sys.exit(1 if bad else 0)
