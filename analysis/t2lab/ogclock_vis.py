"""TRACK ogclock: masked-layer visibility facts (released validation / private cubes) for the train-side emulation."""
import numpy as np, pandas as pd
import lab
rows = []
for panel in lab.PANELS:
    z = np.load(f'{lab.DATA}/t2data_{panel}.npz', allow_pickle=False)
    Et = np.isfinite(z['train_V'])
    for split in ('validation', 'private'):
        days = np.array([str(x) for x in z[f'{split}_days']]); o = np.argsort(days)
        Vm = z[f'{split}_Vm'][o].astype(np.float32); Em = z[f'{split}_Em'][o]; Vis = np.isfinite(Vm)
        assert not (Vis & ~Em).any()
        blank = (Em.sum(2) >= 5) & (Vis.sum(2) == 0)
        nb = ~blank[:, :, None] & Em
        vis = Vis[nb].mean()
        # temporal / spatial dependence of visibility among eligible non-blank cells
        a, b = Vis[:, :-1], Vis[:, 1:]; m = nb[:, :-1] & nb[:, 1:]
        p_tt = (a & b)[m].sum() / max(a[m].sum(), 1)
        a2, b2 = Vis[:, :, :-1], Vis[:, :, 1:]; m2 = nb[:, :, :-1] & nb[:, :, 1:]
        p_ss = (a2 & b2)[m2].sum() / max(a2[m2].sum(), 1)
        vday = np.array([Vis[d][nb[d]].mean() for d in range(len(days))])
        # official window histories: raw Vh vs masked layer at the same slots
        wi = lab.window_index(panel, split); org = pd.to_datetime(wi.forecast_origin, utc=True)
        dd = org.dt.strftime('%Y-%m-%d').map({d: i for i, d in enumerate(days[o])}).to_numpy(); tt = (org.dt.hour * 12 + org.dt.minute // 5).to_numpy()
        Vh = z[f'{split}_Vh'].astype(np.float32)
        hm = np.stack([Vm[d, t - 12:t] for d, t in zip(dd, tt)]); he = np.stack([Em[d, t - 12:t] for d, t in zip(dd, tt)])
        both = np.isfinite(hm) & np.isfinite(Vh)
        rows.append(dict(panel=panel, split=split, Em_share=Em.mean(), trainE=Et.mean(), blank_slots=int(blank.sum()), vis=vis,
                         vis_day_min=vday.min(), vis_day_max=vday.max(), p_vis_next_t=p_tt, p_vis_next_l=p_ss,
                         Vh_cov=np.isfinite(Vh).mean(), Vm_cov_in_hist=np.isfinite(hm).mean(), Em_in_hist=he.mean(),
                         hist_same=float(np.abs(hm[both] - Vh[both]).max()) if both.any() else np.nan,
                         blank_in_hist=float(np.mean([blank[d, t - 12:t].any() for d, t in zip(dd, tt)])),
                         blank_before_hist=float(np.mean([blank[d, :t - 12].any() for d, t in zip(dd, tt)]))))
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(D.round(3).to_string(index=False))
print(D.groupby('split')[['vis', 'p_vis_next_t', 'p_vis_next_l', 'Em_share', 'Vh_cov', 'Vm_cov_in_hist']].mean().round(3))
D.to_csv(f'{lab.ROOT}/analysis/t2lab/results/ogclock/vis_facts.csv', index=False)
