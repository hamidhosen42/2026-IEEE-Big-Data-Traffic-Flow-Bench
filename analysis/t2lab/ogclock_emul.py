"""TRACK ogclock: sensitivity of the ck features to the masked-layer visibility (train offlike windows of 2 panels):
emulated thinning (per-day share from the released days) vs no thinning (raw day) vs fixed share 0.5; plus the distribution of the
features on train offlike (emulated) vs pseudo-March offlike_m (real masked layer, new scenario)."""
import numpy as np, pandas as pd
import lab
import ogclock_lib as ck

pd.set_option('display.width', 250)
out = []
for panel in ('D7_I405_S', 'D12_I5_N'):
    P = lab.load_panel(panel)
    rc = lab.rows(panel, 'offlike', lab.ONGOING)
    Vh, Fh, VT, d, t = ck.train_inputs(P, rc.win)
    THIN, rates = ck.thin_cube(P)
    A = np.asarray(ck.build_fe(panel, 'offlike'))
    Bf = ck.fe_train(P, np.ones_like(THIN), d, t, Vh, VT)                              # raw day (no masking)
    rng = np.random.default_rng(5); T5 = rng.random(THIN.shape) < 0.5
    C5 = ck.fe_train(P, T5, d, t, Vh, VT)                                                # every day at share 0.5
    q = rc.keep5[:, 4] & (rc.h == 1)
    wi, li = rc.wi[q], rc.li[q]
    pm = ck.build_pm(panel)['offlike_m']
    PMf = pm['FE'][pm['meta'][:, 0], pm['meta'][:, 1]][pm['meta'][:, 2] == 1]
    for j, c in enumerate(ck.CK_ALL):
        a, b, c5 = A[wi, li, j], Bf[wi, li, j], C5[wi, li, j]
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c5)
        pmv = PMf[:, j]
        out.append(dict(panel=panel, feat=c, corr_raw=np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan,
                        corr_05=np.corrcoef(c5[ok], b[ok])[0, 1] if ok.sum() > 2 and c5[ok].std() > 0 and b[ok].std() > 0 else np.nan,
                        mean_emul=np.nanmean(a), mean_raw=np.nanmean(b), mean_05=np.nanmean(c5), mean_pmarch=np.nanmean(pmv),
                        nan_emul=np.isnan(a).mean(), nan_pm=np.isnan(pmv).mean()))
    lab.drop_panel(panel)
D = pd.DataFrame(out)
print(D.round(3).to_string(index=False))
D.to_csv(f'{ck.RES}/emul_sensitivity.csv', index=False)
lab.log('DONE')
