"""Local end-to-end SMOKE run of kaggle_jobs/t1_catb_member_a.py on D12_I405_S (the panel with a local train release).
A temporary kaggle_public (argv[1]) gets symlinked corridors (the private month doubles as the validation month: no local
validation release) and Task 1 templates rebuilt from the masked layers
(template = eligible cells blanked in the masked layer, as the official templates); the kernel runs with SMOKE = True
(every 9th day, CatBoost 60 / 30 iterations), PANELS = [D12_I405_S], OUT = argv[1]/out; peak_gb patched for macOS.
Usage: t1catb_kernel_smoke.py <TMPDIR> [part=a]"""
import glob, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
TMP = os.path.abspath(sys.argv[1]); PART = sys.argv[2] if len(sys.argv) > 2 else 'a'
PANEL = 'D12_I405_S'
REL = os.path.join(TMP, 'kaggle_public'); OUT = os.path.join(TMP, 'out')
os.makedirs(REL, exist_ok=True); os.makedirs(OUT, exist_ok=True)
C0 = os.path.join(ROOT, 'data', 'kaggle_public', 'corridors', PANEL); C1 = f'{REL}/corridors/{PANEL}'
if os.path.islink(f'{REL}/corridors'):
    os.remove(f'{REL}/corridors')
os.makedirs(C1, exist_ok=True)
for sub, frm in (('network', 'network'), ('train', 'train'), ('private', 'private'), ('validation', 'private')):
    if not os.path.exists(f'{C1}/{sub}'):          # no local validation release: the private month stands in for it
        os.symlink(f'{C0}/{frm}', f'{C1}/{sub}')
for split in ('train', 'validation', 'private'):
    d = f'{REL}/task1/{PANEL}/{split}'; os.makedirs(d, exist_ok=True)
    if os.path.exists(f'{d}/sample_submission_state.csv'):
        continue
    parts = []
    for f in sorted(glob.glob(f'{REL}/corridors/{PANEL}/{split}/mainline_states_masked/*/*.parquet')):
        m = pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'is_score_eligible'])
        m = m[(m.is_score_eligible == 1) & m.speed_kmh.isna()]
        parts.append(m.assign(mask_regime=re.search(r'mask_regime=(R\d)', f).group(1))[['timestamp', 'station_id', 'link_id', 'mask_regime']])
    t = pd.concat(parts, ignore_index=True)
    t['timestamp'] = pd.to_datetime(t.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    t.to_csv(f'{d}/sample_submission_state.csv', index=False)
    print(split, 'template rows', len(t), flush=True)
src = open(os.path.join(ROOT, 'kaggle_jobs', f't1_catb_member_{PART}.py')).read()
rep = [("REL = glob.glob('/kaggle/input/**/kaggle_public', recursive=True)[0]", f"REL = {REL!r}"),
       ("OUT = '/kaggle/working'", f"OUT = {OUT!r}"),
       ("SMOKE = False", "SMOKE = True"),
       ("    PANELS = ['D7_I10_W']", f"    PANELS = [{PANEL!r}]"),
       ("ru_maxrss / 1024 ** 2", "ru_maxrss / 1024 ** 3")]
for a, b in rep:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, f't1_catb_member_{PART}[smoke]', 'exec'), {'__name__': '__main__'})
S = pd.read_csv(f'{OUT}/state_submission_FTA.csv')
assert len(S) > 0 and S[['speed_kmh', 'flow_vph']].notna().all().all()
print('SMOKE OK: state rows', len(S), 'speed', S.speed_kmh.describe()[['min', 'mean', 'max']].round(2).to_dict(),
      '| holdout', pd.read_csv(f'{OUT}/t1_valholdout.csv').round(4).to_dict('records'))
