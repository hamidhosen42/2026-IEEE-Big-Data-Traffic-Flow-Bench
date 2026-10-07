"""verify_occ_static.py : no training.  (a) leave-one-panel-out / drop-best-panel deltas of the occ track's own OOF frames
(np, id single arms; np+id and bag of 4 re-blended from their saved probabilities), (b) the same on their pseudo-March og4 frame."""
import numpy as np
import pandas as pd
import lab
import occ_lib as oc
import exp_firstcross_lib as fc


def bt(a, b, drop=None):
    if drop:
        a = a[~a.panel.isin(drop)]; b = b[~b.panel.isin(drop)]
    a, b = fc.paired(a, b)
    r = lab.paired_bootstrap(a, b)
    return f'{r["diff"]:+.4f} [{r["lo"]:+.4f}, {r["hi"]:+.4f}] n={r["n_windows"]}'


def lopo(a, b, label):
    a, b = fc.paired(a, b)
    out = []
    for p in sorted(a.panel.unique()):
        aa = a[a.panel != p]; bb = b[b.panel != p]
        out.append((p, round(lab.aggregate(bb) - lab.aggregate(aa), 4)))
    lab.log(f'{label} leave-one-panel-out deltas: {out} | min {min(v for _, v in out):+.4f}')


S = 'offlike'
fr = {}
fr['af_npid'] = oc.blend_frame(['af:af_all_np255', 'af:af_all_id255'], S, 'af_npid')
fr['occ_npid'] = oc.blend_frame(['occ_np255', 'occ_id255'], S, 'occ_npid')
fr['af_bag4'] = oc.blend_frame(['af:af_all_np255', 'af:af_all_id255', 'seed1_np255', 'seed1_id255'], S, 'af_bag4')
fr['occ_bag4'] = oc.blend_frame(['occ_np255', 'occ_id255', 'occs1_np255', 'occs1_id255'], S, 'occ_bag4')
for c in ('np', 'id'):
    a = oc.load_arm(f'af:af_all_{c}255'); b = oc.load_arm(f'occ_{c}255')
    fr[f'af_{c}'] = a[a.set == S]; fr[f'occ_{c}'] = b[b.set == S]
for c in ('np', 'id', 'npid', 'bag4'):
    a, b = fr[f'af_{c}'], fr[f'occ_{c}']
    lab.log(f'[{S}] {c}: all {bt(a, b)} | drop D12_I5_S {bt(a, b, ["D12_I5_S"])} | drop D12_I5_S + D7_I10_E {bt(a, b, ["D12_I5_S", "D7_I10_E"])}')
    rs = a.runstart.astype(bool) if 'runstart' in a else None
    if rs is not None:
        lab.log(f'[{S}] {c} runstart only, drop D12_I5_S: {bt(a[a.runstart.astype(bool)], b[b.runstart.astype(bool)], ["D12_I5_S"])}')
    lopo(a, b, f'[{S}] {c}')
# pseudo-March og4 frame of the occ track
pm = pd.read_pickle(f'{oc.RES}/pmarch/pmarch_og4.pkl')
for s in ('offlike_m', 'runstart_m', 'faith3_m'):
    g = pm[pm.set == s]
    for c in ('npid', 'npid1', 'bag4'):
        a = g[g.arm == f'af_{c}']; b = g[g.arm == f'occ_{c}']
        a2, b2 = fc.paired(a, b)
        best = (b2.groupby('panel').iou.mean() - a2.groupby('panel').iou.mean()).idxmax()
        lab.log(f'PM [{s}] {c}: all {bt(a, b)} | drop best panel {best} {bt(a, b, [best])}')
    if s != 'faith3_m':
        lopo(g[g.arm == 'af_bag4'], g[g.arm == 'occ_bag4'], f'PM [{s}] bag4')
lab.log('STATIC DONE')
