#!/usr/bin/env python3
"""Level-0 check of the family templates: every template rendered with one part's constants equals that part's ORIGINAL
kernel source (consolidation/lineage_src/<part>.py), compared as ASTs (comments / docstrings / formatting ignored).

Rendering (template side): `CFG['k']` / `CFG['k'][i]` / `TFB.cfg['k']` become the part's constant, an
`if CFG['k'] is not None:` block is kept or dropped by that constant, and the two context lines
(`from tfb.runtime import ctx as TFB`, `CFG = TFB.cfg`) are removed.
Normalisation (original side): only the declared I/O substitutions of the vendoring - REL / OUT / SMOKE, the
pseudo-window glob, the part's panel selection (after checking that the original expression selects exactly the
configured panels).
Covers the 43 Task 1 member parts (8 templates: 31 parts of A2 / B2 + the 12 CatBoost / XGBoost parts of A3 / B3) and
tfb-t2ogfa / tfb-t2ogfb (t2_ogfeat).
Run: python tests/test_templates.py   (exit code 1 on failure; also collectable by pytest)
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = os.path.dirname(HERE)
sys.path.insert(0, CONS)
from tfb import config as C        # noqa: E402

LSRC = f'{CONS}/lineage_src'


class Render(ast.NodeTransformer):
    def __init__(self, cfg):
        self.cfg = cfg

    def _key(self, node):
        """'k' if node is CFG['k'] or TFB.cfg['k']."""
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            v = node.value
            if (isinstance(v, ast.Name) and v.id == 'CFG') or (isinstance(v, ast.Attribute) and v.attr == 'cfg'
                                                               and isinstance(v.value, ast.Name) and v.value.id == 'TFB'):
                return node.slice.value
        return None

    @staticmethod
    def _const(val):
        return ast.parse(repr(val), mode='eval').body

    def visit_Subscript(self, node):
        k = self._key(node)
        if k is not None and k != 'panels':
            return self._const(self.cfg[k])
        if isinstance(node.slice, ast.Constant) and self._key(node.value) not in (None, 'panels'):
            return self._const(self.cfg[self._key(node.value)][node.slice.value])
        return self.generic_visit(node)

    def visit_If(self, node):
        t = node.test
        if (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Is, ast.IsNot))
                and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value is None and self._key(t.left)):
            keep = (self.cfg[self._key(t.left)] is None) == isinstance(t.ops[0], ast.Is)
            return [self.visit(n) for n in node.body] if keep else []
        return self.generic_visit(node)


def strip(tree):
    body = tree.body
    while body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]                                  # leading docstrings (ramp2 / bspec c prepend a second one): no-op statements
    out = []
    for n in body:
        if isinstance(n, ast.ImportFrom) and n.module == 'tfb.runtime':
            continue
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'CFG':
            continue
        out.append(n)
    tree.body = out
    return tree


IO = [  # (regex on the original source, replacement) = the declared I/O substitutions
    (r"^REL = glob\.glob\('/kaggle/input/\*\*/kaggle_public', recursive=True\)\[0\]$", "REL = TFB.rel"),
    (r"^OUT = '/kaggle/working'$", "OUT = TFB.out"),
    (r"^SMOKE = False$", "SMOKE = TFB.smoke"),
    (r"^    fs = glob\.glob\(f'/kaggle/input/\*\*/pseudo_\{panel\}\.npz', recursive=True\)$", "    fs = TFB.kglob('tfb-t2lab-full', f'pseudo_{panel}.npz')"),
    (r"^    ref = sorted\(glob\.glob\('/kaggle/input/\*\*/tfb-t2occ/\*\*/queue_submission_fs4occ\.csv', recursive=True\)\)$",
     "    ref = TFB.kglob('tfb-t2occ', 'queue_submission_fs4occ.csv')"),
]


def normalise_original(src, cfg):
    for pat, rep in IO:
        src = re.sub(pat, rep, src, flags=re.M)
    m = list(re.finditer(r"^PANELS = (?!\[')(.*?)(\s+#.*)?$", src, flags=re.M))      # the part's panel selection line
    if m:
        expr = m[0].group(1)
        sel = eval(expr, {'PANELS': list(C.PANELS_ALL)})
        assert sel == cfg['panels'], f'panel selection {expr} = {sel} != configured {cfg["panels"]}'
        src = src[:m[0].start()] + "PANELS = list(CFG['panels'])" + src[m[0].end():]
    elif cfg.get('family') in ('catb', 'xgb'):                                       # k generation: a literal list per part
        m2 = list(re.finditer(r"^PANELS = (\[[^\]]*\])\s+# t1 ?(?:catb|xgb) part [abcd]$", src, flags=re.M))
        assert len(m2) == 1 and eval(m2[0].group(1)) == cfg['panels'], f'panel line of the part != configured {cfg["panels"]}'
        src = src[:m2[0].start()] + "PANELS = list(CFG['panels'])" + src[m2[0].end():]
    elif cfg.get('panels') == ['D7_I405_N']:                                         # part c: PANELS = ['D7_I405_N']
        src = re.sub(r"^PANELS = \['D7_I405_N'\](.*)$", "PANELS = list(CFG['panels'])", src, count=1, flags=re.M)
    return strip(ast.parse(src))


def render_template(path, cfg):
    tree = strip(ast.parse(open(path).read()))
    tree = Render(cfg).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def check(cond, msg, fails):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        fails.append(msg)


def test_t1_templates():
    fails = []
    for kernel in C.t1_member_kernels('full', 'A3'):          # A3 / B3 bags = the A2 / B2 bags + catb, catb2, xgb
        cfg = C.t1_part_cfg(kernel)
        tpl = f'{CONS}/tfb/stages/t1_{cfg["family"]}.py'
        a = ast.dump(render_template(tpl, cfg))
        b = ast.dump(normalise_original(open(f'{LSRC}/{kernel}.py').read(), cfg))
        check(a == b, f'template t1_{cfg["family"]:8s} rendered for {kernel:14s} == original source (AST)', fails)
    assert not fails, fails


def test_t2_ogfeat_template():
    fails = []
    for kernel, cfg in (('tfb-t2ogfa', dict(C.stage_cfg('ogfeat_a', 'B2', 'full'), lgb_repro=None)),     # lgb_repro None = the original
                        ('tfb-t2ogfb', dict(C.stage_cfg('ogfeat_b', 'B2', 'full'), lgb_repro=None))):  # setting (the only addition)
        a = ast.dump(render_template(f'{CONS}/tfb/stages/t2_ogfeat.py', cfg))
        b = ast.dump(normalise_original(open(f'{LSRC}/{kernel}.py').read(), dict(cfg, panels=None)))
        check(a == b, f'template t2_ogfeat rendered for {kernel} == original source (AST)', fails)
    assert not fails, fails


if __name__ == '__main__':
    bad = 0
    for t in (test_t1_templates, test_t2_ogfeat_template):
        try:
            t()
        except AssertionError:
            bad += 1
    print('TEMPLATES', 'FAILED' if bad else 'OK')
    sys.exit(1 if bad else 0)
