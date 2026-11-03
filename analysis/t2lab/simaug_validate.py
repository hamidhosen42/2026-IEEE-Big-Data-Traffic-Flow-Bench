"""simaug step 1: CTM realism on TRAIN days (train split only).
Per panel: simulate NDAYS train days (spread over the 273), compare the simulated queue state (V <= 0.6 vf) with the observed one
on observed cells.  Metrics:
  iou_pm        pooled cell IoU 12:00-20:00, sim vs obs (same day)
  iou_pm_shuf   same, sim of another day vs obs (day-specific skill reference)
  share_corr    Pearson corr across links of the queued share (all slots)
  onset_hit     official-like onset windows (offlike, cond onset) on the simulated days: hit if the sim has a queued cell within
                +-2 links of a true T+1..T+6 queued cell during [T-6, T+12]
  onset_hit_prior  same with 'queued' = all-train prior >= 0.05 at that link/slot (climatology reference, uses all train days)
  sim_onsets_per_day / obs_onsets_per_day   number of distinct onset events"""
import sys, time, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import lab, simaug_ctm as ctm

NDAYS = int(os.environ.get('NDAYS', 40))
panels = sys.argv[1:] or lab.PANELS
OUT = f'{lab.ROOT}/analysis/t2lab/results/simaug'
rows = []


def onset_events(Q, E):
    """count onset events: queued cell whose 5x5 (link +-2, slots t-6..t-1) neighbourhood had no queue."""
    D, T, L = Q.shape; n = 0
    from scipy.ndimage import maximum_filter
    prev = maximum_filter(Q.astype(np.uint8), size=(1, 6, 5), origin=(0, 3, 0)) > 0   # covers t-? approx
    return int((Q & ~prev).sum())


for panel in panels:
    t0 = time.time()
    P = lab.load_panel(panel); net = ctm.load_net(panel)
    if os.environ.get('CALIB') == '1':      # effective capacity of the link downstream of each observed queue head (train truth)
        Vt, Ft = P['V'], P['Fq']; Qt = np.isfinite(Vt) & (Vt <= P['cut']); Et = np.isfinite(Vt)
        hd = Qt[:, :, :-1] & Et[:, :, 1:] & ~Qt[:, :, 1:]
        for l in np.where(hd.sum((0, 1)) >= int(os.environ.get('MINHEAD', 30)))[0]:
            dis = Ft[:, :, l + 1][hd[:, :, l]]
            net['cap'][l + 1] = min(net['cap'][l + 1], float(np.nanquantile(dis, float(os.environ.get('CQ', 0.9)))))
    D = len(P['days']); days = np.unique(np.linspace(0, D - 1, NDAYS).round().astype(int))
    z = np.load(P['npz'])
    V = P['V'][days]; Fq = P['Fq'][days]; ON = P['ON'][days]; OFF = P['OFF'][days]
    RS = float(os.environ.get('RAMPSCALE', 1.0)); Lk = len(P['cut'])
    Vs, Fs, cl = ctm.simulate(net, V, Fq, ON, OFF, m_on=np.full(Lk, RS), m_off=np.full(Lk, RS))
    cut = P['cut']; E = np.isfinite(V); Qo = E & (V <= cut); Qs = Vs <= cut
    pm = slice(144, 240)
    def iou(a, b, m):
        a = a & m; b = b & m; u = (a | b).sum(); return np.nan if u == 0 else (a & b).sum() / u
    io = iou(Qs[:, pm], Qo[:, pm], E[:, pm])
    sh = np.roll(np.arange(len(days)), 7)
    io_sh = iou(Qs[sh][:, pm], Qo[:, pm], E[:, pm])
    so = np.where(E, Qo, np.nan); sho = np.nanmean(so, (0, 1)); shs = Qs.mean((0, 1))
    corr = np.corrcoef(sho, shs)[0, 1] if shs.std() > 0 else np.nan
    prec = (Qs & Qo & E).sum() / max((Qs & E).sum(), 1); rec = (Qs & Qo & E).sum() / max((Qo).sum(), 1)
    # onset windows
    W = lab.window_set(panel, 'offlike'); W = W[(W.cond == lab.ONSET) & W.d.isin(days)]
    pos = {d: i for i, d in enumerate(days)}
    prior = lab.prior_table(P, np.ones(D, bool))     # [3, 288, L]
    hit = hitp = 0; n = 0; L = len(cut)
    for d, t in zip(W.d, W.t):
        i = pos[d]; tr = Qo[i, t + 1:t + 7].any(0)
        if not tr.any():
            continue
        n += 1
        ll = np.where(tr)[0]; nb = np.zeros(L, bool)
        for l in ll:
            nb[max(l - 2, 0):l + 3] = True
        hit += Qs[i, max(t - 6, 0):t + 13][:, nb].any()
        cls = int(P['dow'][d] >= 5)
        hitp += (prior[cls, max(t - 6, 0):t + 13][:, nb] >= 0.05).any()
    r = dict(panel=panel, L=L, cells=cl['C'], days=len(days), iou_pm=io, iou_pm_shuf=io_sh, prec=prec, rec=rec,
             share_obs=float(np.nanmean(so)), share_sim=float(Qs.mean()), share_corr=corr,
             onset_n=n, onset_hit=hit / max(n, 1), onset_hit_prior=hitp / max(n, 1), sec=time.time() - t0)
    # top queued links obs vs sim
    r['top_obs'] = np.argsort(-sho)[:6].tolist(); r['top_sim'] = np.argsort(-shs)[:6].tolist()
    rows.append(r); lab.log(r)
    np.savez_compressed(f'{OUT}/val_{panel}.npz', days=days, Qs=Qs, Vs=Vs.astype(np.float16))
    lab.drop_panel(panel)
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(df.round(3).to_string(index=False))
df.to_csv(f'{OUT}/validate_{os.environ.get("TAG", "v1")}.csv', index=False)
