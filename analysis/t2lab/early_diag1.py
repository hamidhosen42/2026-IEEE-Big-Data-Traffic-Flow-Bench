"""early track diagnostic (light): per-window view of truth / base prediction / train prior / previous-day evidence."""
import sys
import numpy as np, pandas as pd
import lab, early_lib as el
F = lab.F
cat = el.load_cat()
panels = sys.argv[1].split(',') if len(sys.argv) > 1 else ['D7_I10_W', 'D12_I5_S', 'D7_I405_S', 'D12_I5_N']
for panel in panels:
    b = el.march_bundle(panel)
    evd = el.Evidence(b)
    widx = np.arange(b['W']); p = el.predict_items(cat, b, widx)
    iou, npred = el.iou_items(b, widx, p)
    print(f'\n===== {panel}  L={b["L"]}  windows={b["W"]}  base IoU mean {iou.mean():.3f}')
    for i in widx:
        d, t = int(b['d'][i]), int(b['t'][i]); s6 = t + 6; c = int(b['wk'][d])
        P6 = np.zeros((F, b['L']), np.float32); P6[F - 1] = p[i]
        pred = lab.decode(P6.ravel()).reshape(F, b['L'])
        tr = np.where(b['truth'][i][5] & b['ev'][i][5])[0]; pr = np.where(pred[5])[0]
        tr_early = int((b['truth'][i][:5] & b['ev'][i][:5]).sum())
        pt = b['prior_tr'][c, s6]
        ev_days = []
        for dd in range(max(d - 4, 0), d):
            O, e, X = evd.box(np.array([dd]), s6, -6, 6)
            def sh(links):
                return '/'.join(f'{int(O[0, l])}:{int(e[0, l])}' for l in links[:4])
            top = np.argsort(-O[0])[:3]; top = top[O[0, top] > 0]
            ev_days.append(f'd{dd}({"we" if b["wk"][dd] else "wd"}) T[{sh(tr)}] P[{sh(pr)}] topQ{top.tolist()}')
        print(f'd{d:2d} {"we" if c else "wd"} t={t:3d} ({t // 12:02d}:{t % 12 * 5:02d}) iou {iou[i]:.2f} truth{tr.tolist()} (+{tr_early} early) pred{pr.tolist()} '
              f'prior@truth {np.round(pt[tr], 2).tolist()} prior@pred {np.round(pt[pr], 2).tolist()} p@truth {np.round(p[i][tr], 2).tolist()} p@pred {np.round(p[i][pr], 2).tolist()}')
        for s in ev_days:
            print('        ', s)
