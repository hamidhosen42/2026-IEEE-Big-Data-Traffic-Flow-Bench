"""Local test of kaggle_jobs/t2_model_v21.py (and its _smoke twin) against the lab, with stubs for Kaggle file loading.
A. shared functions are AST-identical to t2_model_v20.py; smoke file differs only in SMOKE = True.
B. per panel: faithful candidates / run-starts / F03 rest / CUR windows == lab (candidates, runstart, faith3, denseall).
C. end-to-end main() on 2 panels with tiny models (stubs: load_panel, read_official, load_split from the local npz).
Usage: exp_v21_localtest.py [B panels comma list | all] [e2e 0/1]
"""
import ast, json, os, sys, types, pickle, warnings
import numpy as np, pandas as pd
import lab

ROOT = lab.ROOT
V21 = f'{ROOT}/kaggle_jobs/t2_model_v21.py'; V21S = f'{ROOT}/kaggle_jobs/t2_model_v21_smoke.py'; V20 = lab.PROD_FILE
RES = f'{ROOT}/analysis/t2lab/results/v21'; os.makedirs(RES, exist_ok=True)
warnings.simplefilter('ignore')
np.seterr(all='ignore')
report = {}


def load_kernel(path):
    src = open(path).read()
    line = "REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]"
    assert src.count(line) == 1 and src.rstrip().endswith('\nmain()')
    src = src.replace(line, "REL = '/nonexistent/kaggle_public'")
    src = src.rstrip()[:-len('main()')]
    mod = types.ModuleType('t2v21'); exec(compile(src, path, 'exec'), mod.__dict__)
    return mod


# ---------------------------------------------------------------- A
t20 = {n.name: ast.dump(n) for n in ast.parse(open(V20).read()).body if isinstance(n, ast.FunctionDef)}
t21 = {n.name: ast.dump(n) for n in ast.parse(open(V21).read()).body if isinstance(n, ast.FunctionDef)}
same = sorted(k for k in t20 if k in t21 and t20[k] == t21[k]); diff = sorted(k for k in t20 if k in t21 and t20[k] != t21[k])
miss = sorted(k for k in t20 if k not in t21)
print('A. functions identical to v20:', same); print('A. differing:', diff, '| not in v21:', miss)
need = {'load_panel', 'ramp_cube', 'add_ramp', 'established', 'iou', 'build_windows', 'ffill_time', 'shift_links', 'dist_to_queued',
        'window_features', 'prior_table', 'prior_feats', 'make_rows', 'to_matrix', 'ongoing_keep', 'decode', 'predict_windows'}
assert need <= set(same), need - set(same)
K = load_kernel(V21)
assert K.PARAMS == lab.prod.PARAMS and K.PRIOR_COLS == lab.prod.PRIOR_COLS and K.ID_DROP == lab.prod.ID_DROP
assert (K.H, K.F, K.SPACING) == (lab.H, lab.F, lab.SPACING)
a, b = open(V21).read().splitlines(), open(V21S).read().splitlines()
dl = [(x, y) for x, y in zip(a, b) if x != y]
print('A. smoke diff lines:', dl); assert len(a) == len(b) and dl == [('SMOKE = False', 'SMOKE = True')]
report['A'] = dict(identical=same, differing=diff, smoke_diff=dl)

# ---------------------------------------------------------------- B
arg = sys.argv[1] if len(sys.argv) > 1 else 'D7_I10_W,D12_I5_S'
panels = lab.PANELS if arg == 'all' else [p for p in arg.split(',') if p]
report['B'] = {}
for panel in panels:
    P = dict(lab.load_panel(panel, cache=False))
    K.build_windows = lambda P_, global_spacing=True, dense_ongoing=False, _p=panel: pickle.load(
        open(f'{lab.CACHE}/windows/{_p}__denseall.pkl', 'rb'))          # cached output of the identical production function
    st = K.select_windows(P)
    _, og = lab.candidates(lab.load_panel(panel))
    S = lab.selector_stats(lab.load_panel(panel), 't36')
    pers_lab = S['pers'][('fill', 'all')]
    r = dict(cand_equal=bool((P['cand'] == og).all()), n_cand=int(og.sum()),
             pers_maxdiff=float(np.nanmax(np.abs(P['pers'] - pers_lab))), pers_nan_equal=bool((np.isnan(P['pers']) == np.isnan(pers_lab)).all()))
    rs = lab.window_set(panel, 'runstart'); f3 = lab.window_set(panel, 'faith3')
    mine_rs = {(w[0], w[1]) for w, k in zip(P['exw'], P['ex_rs']) if k}
    mine_rest = {(w[0], w[1]) for w, k in zip(P['exw'], P['ex_rs']) if not k}
    lab_rs = set(zip(rs.d, rs.t)); lab_f3 = set(zip(f3.d, f3.t))
    r.update(runstart_equal=mine_rs == lab_rs, n_runstart=len(lab_rs), rest_subset_faith3=mine_rest <= (lab_f3 - lab_rs),
             n_rest=len(mine_rest), n_faith3_minus_rs=len(lab_f3 - lab_rs),
             rest_complete=(mine_rest == lab_f3 - lab_rs) if len(lab_f3 - lab_rs) <= K.MAX_OG else len(mine_rest) == K.MAX_OG,
             no_dup=len(set(P['exw'])) == len(P['exw']))
    tw = pickle.load(open(f'{lab.CACHE}/windows/{panel}__denseall.pkl', 'rb'))
    ogw = [w for w in tw if w[2] == 'queue_ongoing']; onw = [w for w in tw if w[2] == 'queue_onset']
    if len(ogw) > K.MAX_OG:
        ogw = [ogw[i] for i in sorted(np.random.default_rng(0).choice(len(ogw), K.MAX_OG, replace=False))]
    r['cur_equal_v20_draw'] = P['trainw'] == onw + ogw
    off = [w for w in lab.official_windows(lab.load_panel(panel)) if w[2] == 'queue_ongoing']
    r['official_faithful'] = int(sum(P['cand'][w[0], w[1]] for w in off)); r['official_runstart'] = int(sum(P['rs'][w[0], w[1]] for w in off))
    r['stats'] = st
    print('B.', panel, {k: v for k, v in r.items() if k != 'stats'}, flush=True)
    assert r['cand_equal'] and r['runstart_equal'] and r['rest_subset_faith3'] and r['rest_complete'] and r['cur_equal_v20_draw'] and r['no_dup']
    assert r['pers_maxdiff'] < 1e-9 and r['pers_nan_equal']
    report['B'][panel] = r
    lab.drop_panel(panel)
    json.dump(report, open(f'{RES}/localtest.json', 'w'), indent=1, default=str)
print('B. total official ongoing faithful', sum(r['official_faithful'] for r in report['B'].values()),
      'run-starts', sum(r['official_runstart'] for r in report['B'].values()), 'of', 5 * len(report['B']))

# ---------------------------------------------------------------- C
if (sys.argv[2] if len(sys.argv) > 2 else '1') == '1':
    K = load_kernel(V21)
    K.PANELS = ['D7_I10_W', 'D7_I210_E']; K.N_BIG = 30; K.MAX_OG = 250; K.OUT = RES
    K.PARAMS['num_threads'] = int(os.environ.get('T2LAB_THREADS', 4))
    K.load_panel = lambda panel: dict(lab.load_panel(panel, cache=False))
    K.read_official = lambda panel, P: lab.official_windows(lab.load_panel(panel))
    K.build_windows = lambda P_, global_spacing=True, dense_ongoing=False: pickle.load(
        open(f'{lab.CACHE}/windows/{P_["panel"]}__denseall.pkl', 'rb'))

    def load_split(P, panel, split):
        R = lab.split_rows(lab.load_panel(panel), split, P['prior_all']); wi = R.pop('wi'); R.pop('dow')
        rows_ = []
        for w, fo in zip(wi.window_id, wi.forecast_origin):
            t0 = pd.Timestamp(fo)
            for h in range(1, 7):
                ts = (t0 + pd.Timedelta(minutes=5 * h)).strftime('%Y-%m-%dT%H:%M:%SZ')
                rows_ += [(w, ts, l) for l in P['links']]
        tpl = pd.DataFrame(rows_, columns=['window_id', 'timestamp', 'link_id']); tpl['queue_pred'] = '0'
        return wi, tpl, R
    K.load_split = load_split
    K.main()
    out = {}
    for v in K.VARIANTS:
        s = pd.read_csv(f'{RES}/queue_submission_{v}.csv'); out[v] = dict(rows=len(s), pos=int(s.queue_pred.sum()), cols=list(s.columns))
    print('C. files:', out)
    z = np.load(f'{RES}/t2v21_probs.npz'); print('C. probs keys', len(z.files), z.files[:8])
    fp = pd.read_csv(f'{RES}/t2v21_footprint.csv')
    c2 = pd.read_csv(f'{RES}/queue_submission_cur2.csv'); wi_on = set()
    for p in K.PANELS:
        for sp in ('validation', 'private'):
            w = lab.window_index(p, sp); wi_on |= set(w.window_id[w.condition == 'queue_onset'])
    for v in K.VARIANTS:
        s = pd.read_csv(f'{RES}/queue_submission_{v}.csv')
        assert s[s.window_id.isin(wi_on)].queue_pred.sum() == 0 and len(s) == len(c2) and s.queue_pred.isin([0, 1]).all()
        assert s[~s.window_id.isin(wi_on)].groupby('window_id').queue_pred.sum().min() >= 1
    report['C'] = dict(files=out, footprint_rows=len(fp), onset_rows_zero=True)
    json.dump(report, open(f'{RES}/localtest.json', 'w'), indent=1, default=str)
print('ALL LOCAL TESTS PASSED')
