"""TRACK ogmodel: local test of kaggle_jobs/t2_ogmodel_v23.py with stubs for the Kaggle file loading (pattern of
exp_v21_localtest.py).  Tiny models, 2 panels; nothing is submitted, no Kaggle API.
A. every function of t2_model_v22.py that v23 does not intentionally change is AST-identical in v23.
B. g72_flags(window index) on the real validation / private / train window indexes == ogmodel_g72.index_gaps (lab).
C. select_windows: every non-run-start F03 window is flagged, run-starts are flagged at the per-panel flip share.
D. end-to-end main() of v23 AND of v22 with the same stubs / tiny settings: output files, onset rows 0, every ongoing
   window has a prediction, and v23's control variant fs_np is identical to v22's fs_np.
Usage (through run_heavy): ogmodel_v23_localtest.py [panels=D7_I210_W,D12_I5_S]"""
import ast, json, os, pickle, sys, types, warnings
import numpy as np
import pandas as pd
import lab
import ogmodel_lib as og
import ogmodel_g72 as g72

ROOT = lab.ROOT
V23 = f'{ROOT}/kaggle_jobs/t2_ogmodel_v23.py'; V22 = f'{ROOT}/kaggle_jobs/t2_model_v22.py'
warnings.simplefilter('ignore'); np.seterr(all='ignore')
panels = (sys.argv[1] if len(sys.argv) > 1 else 'D7_I210_W,D12_I5_S').split(',')


def load_kernel(path, name):
    src = open(path).read()
    line = "REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]"
    assert src.count(line) == 1 and src.rstrip().endswith('\nmain()')
    src = src.replace(line, "REL = '/nonexistent/kaggle_public'").rstrip()[:-len('main()')]
    mod = types.ModuleType(name); exec(compile(src, path, 'exec'), mod.__dict__)
    return mod


# ---------------------------------------------------------------- A
f22 = {n.name: ast.dump(n) for n in ast.parse(open(V22).read()).body if isinstance(n, ast.FunctionDef)}
f23 = {n.name: ast.dump(n) for n in ast.parse(open(V23).read()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
same = sorted(k for k in f22 if f23.get(k) == f22[k]); diff = sorted(k for k in f22 if k in f23 and f23[k] != f22[k]); new = sorted(k for k in f23 if k not in f22)
print('A. identical to v22:', same); print('A. changed:', diff, '| new:', new)
assert set(diff) <= {'select_windows', 'panel_rows', 'load_split', 'predict_split', 'print_footprint', 'control_check', 'main'}, diff
assert set(new) == {'add_flag', 'g72_flags', 'SplitModel'}, new
K = load_kernel(V23, 't2v23')
assert K.PARAMS == lab.prod.PARAMS and K.PRIOR_COLS == lab.prod.PRIOR_COLS and K.ID_DROP == lab.prod.ID_DROP and K.SMOKE is False
print('A. switches: G72', K.G72, 'QSPLIT', K.QSPLIT, 'G72_RS_SHARE', K.G72_RS_SHARE)

# ---------------------------------------------------------------- B
tot = {}
for split in ('train', 'validation', 'private'):
    n = 0
    for p in lab.PANELS:
        wi = lab.window_index(p, split)
        fl, gap = K.g72_flags(wi)
        ref = g72.index_gaps(p, split)
        assert np.array_equal(fl > 0, ref.g72.to_numpy()) and np.array_equal(np.minimum(gap, 10 ** 8), np.minimum(ref.gap.to_numpy(), 10 ** 8)), (p, split)
        n += int(fl[(wi.condition == 'queue_ongoing').to_numpy()].sum())
        assert fl[(wi.condition == 'queue_onset').to_numpy()].sum() == 0
    tot[split] = n
print('B. g72 ongoing windows per split (kernel function == lab):', tot)
assert tot == {'train': 9, 'validation': 8, 'private': 12}, tot


# ---------------------------------------------------------------- C / D
def stub(Kx, out, with_flag):
    Kx.PANELS = list(panels); Kx.N_BIG = 30; Kx.MAX_OG = 250; Kx.OUT = out; os.makedirs(out, exist_ok=True)
    Kx.PARAMS['num_threads'] = int(os.environ.get('T2LAB_THREADS', 4))
    Kx.load_panel = lambda panel: dict(lab.load_panel(panel, cache=False))
    Kx.read_official = lambda panel, P: lab.official_windows(lab.load_panel(panel))
    Kx.build_windows = lambda P_, global_spacing=True, dense_ongoing=False: pickle.load(open(f'{lab.CACHE}/windows/{P_["panel"]}__denseall.pkl', 'rb'))

    def load_split(P, panel, split):
        R = lab.split_rows(lab.load_panel(panel), split, P['prior_all']); wi = R.pop('wi'); R.pop('dow')
        if with_flag:                                   # the two v23 lines of load_split
            R['g72'], gap = Kx.g72_flags(wi)
            Kx.add_flag(R, R['g72'] if Kx.G72 else np.zeros(R['W'], np.float32))
        rows_ = []
        for w, fo in zip(wi.window_id, wi.forecast_origin):
            t0 = pd.Timestamp(fo)
            for h in range(1, 7):
                ts = (t0 + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ')
                rows_ += [(w, ts, l) for l in P['links']]
        tpl = pd.DataFrame(rows_, columns=['window_id', 'timestamp', 'link_id']); tpl['queue_pred'] = '0'
        return wi, tpl, R
    Kx.load_split = load_split


OUT23 = f'{og.RES}/v23_localtest'; OUT22 = f'{og.RES}/v22_localtest'
stub(K, OUT23, True)
for panel in panels:
    P = dict(lab.load_panel(panel, cache=False)); st = K.select_windows(P)
    rs = P['ex_rs']; fl = P['ex_g72'] > 0
    assert fl[~rs].all() and len(fl) == len(P['exw'])
    wsum_rs = K.W_RS * fl[rs].sum(); wsum_mid = K.W_DENSE * (~rs).sum()
    print(f'C. {panel}: F03 windows {len(rs)} (run-starts {int(rs.sum())}), flip {st["g72_flip"]}, run-starts flagged {int(fl[rs].sum())}, '
          f'run-start weight share among flagged windows {wsum_rs / (wsum_rs + wsum_mid):.3f}')
    lab.drop_panel(panel)
K.main()
K22 = load_kernel(V22, 't2v22'); stub(K22, OUT22, False); K22.main()
out = {}
wi_on = set(); wi_g = set()
for p in panels:
    for sp in ('validation', 'private'):
        w = lab.window_index(p, sp); wi_on |= set(w.window_id[w.condition == 'queue_onset'])
        fl, _ = K.g72_flags(w); wi_g |= set(w.window_id[fl > 0])
ref = pd.read_csv(f'{OUT22}/queue_submission_fs_np.csv')
for v in K.VARIANTS:
    s = pd.read_csv(f'{OUT23}/queue_submission_{v}.csv'); out[v] = dict(rows=len(s), pos=int(s.queue_pred.sum()), cols=list(s.columns))
    assert list(s.columns) == ['window_id', 'timestamp', 'link_id', 'queue_pred'] and len(s) == len(ref) and s.queue_pred.isin([0, 1]).all()
    assert s[s.window_id.isin(wi_on)].queue_pred.sum() == 0 and s[~s.window_id.isin(wi_on)].groupby('window_id').queue_pred.sum().min() >= 1
    assert not s.duplicated(['window_id', 'timestamp', 'link_id']).any()
print('D. files:', out)
a = pd.read_csv(f'{OUT23}/queue_submission_fs_np.csv')
same_ctl = bool((a.queue_pred.to_numpy() == ref.queue_pred.to_numpy()).all() and (a.window_id == ref.window_id).all())
print('D. control: v23 fs_np identical to v22 fs_np (same stubs, tiny models):', same_ctl); assert same_ctl
b = pd.read_csv(f'{OUT23}/queue_submission_g4.csv'); r4 = pd.read_csv(f'{OUT22}/queue_submission_fs4.csv')
og_rows = ~b.window_id.isin(wi_on)
for nm, m in (('g72 windows', og_rows & b.window_id.isin(wi_g)), ('other ongoing windows', og_rows & ~b.window_id.isin(wi_g))):
    x, y = b.queue_pred.to_numpy()[m.to_numpy()], r4.queue_pred.to_numpy()[m.to_numpy()]
    print(f'D. g4 (tiny) vs v22 fs4 (tiny) | {nm}: windows {b.window_id[m].nunique()}, rows changed {int((x != y).sum())}/{len(x)}, Jaccard {(x & y).sum() / max((x | y).sum(), 1):.4f}')
z = np.load(f'{OUT23}/t2v23_probs.npz'); print('D. probs keys', len(z.files), z.files[:6])
print('ALL LOCAL TESTS PASSED')
