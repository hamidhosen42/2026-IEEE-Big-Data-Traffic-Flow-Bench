"""verify_t1dark_selbias.py - is the t1dark gain an artefact of selecting evaluation events on their own evaluation labels?
Lab: an evaluation blackout is selected only if a RELEASED (= evaluation) cell in T+1..T+6 is queued (v <= 0.6 vf).
Deployment: the organizer selects on the true state, scored cells are a random hash subset -> scored cells are not the
conditioning set. Emulation: split each event's evaluation cells randomly into O1 (selection) / O2 (scoring); keep the
event only if O1 has a queued cell in k = 1..6; score base vs dark on O2 only (O2 is then not conditioned on by the queue
criterion). Repeated over R random splits. Works on the candidate's files (bagB/bagD) and on verify_t1dark_exp outputs."""
import glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESC = os.path.join(HERE, 'results', 't1dark')
RESV = os.path.join(HERE, 'results', 'verify_t1dark')
CACHE = os.path.join(ROOT, 'data', 't2local', 'cache', 't1dark')
R = 20


def cand_frames():
    out = []
    for p in ['D7_I405_S', 'D7_I210_W', 'D12_I5_S', 'D7_I10_E']:
        meta = np.load(f'{CACHE}/{p}_meta.npz', allow_pickle=True)
        for sp in ('validation', 'private'):
            sim = meta[f'{sp}_sim']
            for f in (0, 1):
                z = np.load(f'{RESC}/{p}_f{f}.npz', allow_pickle=True)
                f2 = f'{RESC}/{p}_f{f}_s10.npz'
                z2 = np.load(f2, allow_pickle=True) if os.path.exists(f2) else None
                k = f'{sp}_blk'
                if z2 is not None:
                    vb = 0.5 * (z[f'base|{k}|v'].astype(float) + z2[f'base|{k}|v']); vd = 0.5 * (z[f'dark|{k}|v'].astype(float) + z2[f'dark|{k}|v'])
                else:
                    vb = 0.5 * (z[f'base|{k}|v'].astype(float) + z[f'base2|{k}|v']); vd = 0.5 * (z[f'dark|{k}|v'].astype(float) + z[f'dark2|{k}|v'])
                ev = z[f'{k}|ev']
                out.append(pd.DataFrame(dict(pm=f'{p}|{sp}', ev=ev, k=z[f'{k}|t'] - sim[ev, 1], cond=sim[ev, 2], vt=z[f'{k}|vt'],
                                             vf=z[f'{k}|vf'], vb=vb, vd=vd)))
    return pd.concat(out, ignore_index=True)


def ver_frames(tag=''):
    out = []
    for f in sorted(glob.glob(f'{RESV}/*_f[01]{tag}.npz')):
        if 'smoke' in f:
            continue
        z = np.load(f, allow_pickle=True)
        p = os.path.basename(f).rsplit('_f', 1)[0]
        for sp in ('validation', 'private'):
            pre = f'{sp}|std|blk'
            if f'{pre}|ev' not in z.files:
                continue
            out.append(pd.DataFrame(dict(pm=f'{p}|{sp}', ev=z[f'{pre}|ev'], k=z[f'{pre}|k'], cond=z[f'{pre}|cond'], vt=z[f'{pre}|vt'],
                                         vf=z[f'{pre}|vf'], vb=z[f'{pre}|base|v'].astype(float), vd=z[f'{pre}|dark|v'].astype(float))))
    return pd.concat(out, ignore_index=True) if out else None


def analyse(A, name):
    A = A.copy()
    A['q'] = A.vt <= 0.6 * A.vf
    A['eb'] = (A.vb - A.vt) ** 2; A['ed'] = (A.vd - A.vt) ** 2
    full = np.sqrt(A.ed.mean()) - np.sqrt(A.eb.mean())
    print(f'\n[{name}] cells {len(A)} events {A.groupby(["pm", "ev"]).ngroups} | lab conditioning (all cells): '
          f'speed {np.sqrt(A.eb.mean()):.3f} -> {np.sqrt(A.ed.mean()):.3f} d {full:+.3f}')
    rng = np.random.default_rng(12345)
    res = []
    for r in range(R):
        o1 = rng.random(len(A)) < 0.5
        sel = A[o1 & A.q & A.k.between(1, 6)].groupby(['pm', 'ev']).size().index
        keyed = pd.MultiIndex.from_frame(A[['pm', 'ev']])
        kept = keyed.isin(sel)
        o2 = ~o1
        B_ = A[o2 & kept]; C_ = A[o2]
        res.append(dict(n_kept=len(sel), d_o2_kept=np.sqrt(B_.ed.mean()) - np.sqrt(B_.eb.mean()),
                        rb=np.sqrt(B_.eb.mean()), rd=np.sqrt(B_.ed.mean()),
                        d_o2_all=np.sqrt(C_.ed.mean()) - np.sqrt(C_.eb.mean()),
                        d_o2_kept_onset=np.sqrt(B_[B_.cond == 0].ed.mean()) - np.sqrt(B_[B_.cond == 0].eb.mean()),
                        d_o2_kept_k16=np.sqrt(B_[B_.k <= 6].ed.mean()) - np.sqrt(B_[B_.k <= 6].eb.mean()),
                        d_o2_all_k16=np.sqrt(C_[C_.k <= 6].ed.mean()) - np.sqrt(C_[C_.k <= 6].eb.mean())))
    T = pd.DataFrame(res)
    print('  deployment-like (select on O1, score O2), mean over', R, 'splits:')
    print('   kept events %.1f of %d | O2 kept: speed %.3f -> %.3f d %+.3f (sd over splits %.3f) | O2 all events d %+.3f'
          % (T.n_kept.mean(), A.groupby(['pm', 'ev']).ngroups, T.rb.mean(), T.rd.mean(), T.d_o2_kept.mean(), T.d_o2_kept.std(),
             T.d_o2_all.mean()))
    print('   onset kept d %+.3f | k1..6 kept d %+.3f vs k1..6 all d %+.3f' % (T.d_o2_kept_onset.mean(), T.d_o2_kept_k16.mean(),
                                                                         T.d_o2_all_k16.mean()))
    # how strongly is the eval set conditioned: share of events whose queue criterion rests on <= 3 queued eval cells
    g = A[A.k.between(1, 6)].groupby(['pm', 'ev']).q.sum()
    print('   queued eval cells in k1..6 per event: median %d, share of events with <= 3: %.2f, <= 10: %.2f'
          % (g.median(), (g <= 3).mean(), (g <= 10).mean()))
    for lo, hi in ((1, 3), (4, 10), (11, 10 ** 6)):
        evs = g[(g >= lo) & (g <= hi)].index
        S = A[pd.MultiIndex.from_frame(A[['pm', 'ev']]).isin(evs)]
        print('   events with %d..%s queued eval cells: n %d, d %+.3f (base %.3f)' % (lo, hi if hi < 10 ** 6 else 'inf', len(evs),
              np.sqrt(S.ed.mean()) - np.sqrt(S.eb.mean()), np.sqrt(S.eb.mean())))


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'cand'
    if which in ('cand', 'both'):
        analyse(cand_frames(), 'candidate files, 2-seed bags (bagB -> bagD)')
    if which in ('ver', 'both'):
        A = ver_frames()
        if A is not None:
            analyse(A, 'verifier files, single seed (base -> dark)')
            for pm, g in A.groupby('pm'):
                print('   ', pm, 'd %+.3f' % (np.sqrt(((g.vd - g.vt) ** 2).mean()) - np.sqrt(((g.vb - g.vt) ** 2).mean())))
