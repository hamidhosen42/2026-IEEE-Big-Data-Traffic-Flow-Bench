"""t1div: U-Net T1 member (EXP_T1_UNET) on the t1bspec pseudo-holdout - yeashusemwal task1_unet2.py recipe.

Model (as task1_unet2.py): 2-D U-Net (3 levels, GroupNorm + GELU) over a day grid (stations x 288 slots), 13 input
channels (masked v / vf, masked q / cap, occupancy (absent in the reduced replica -> 0), observed flag, weekday profile
v / vf and q / cap, vf / 120, lanes / 6, log1p(length), sin / cos time of day, mask rate, weekday / 6), output = residual on
top of the profile. Loss on target cells: 0.54 (dv / 25)^2 + 0.46 (dq_lane / 600)^2 + DENS_W * (per-lane density error)^2
(dk = (qp / max(vp, 5) - qt / max(vt, 5)) / lanes, DENS_W 0.05). Training: the 273 train days of the 4 lab panels
jointly (padded to 152 stations), fresh random masks each epoch (rate 0.2 / 0.3 / 0.5 on eligible cells) + with prob 0.6
one or two 18-slot full blackouts; weekday profile leave-one-day-out (from the released-like masked train view); AdamW
1e-3, OneCycle, grad clip 1, batch 6, last epoch kept (no holdout days needed: OneCycle anneals to ~0).
Inference: on each panel / fold / month, the t1bspec evaluation view (evaluation blackouts + ordinary eval cells hidden),
profile = full train profile (t1bspec meta hv / hq), rate = organizer regime rate; predictions at the evaluation cells in
the t1bspec order, clipped like the LightGBM arms ([3, 1.05 vf] km/h, >= 20 vph).
Usage: t1div_unet.py [epochs=30] [width=32] [seed=50] [dens=0.05] [device=mps] [tag=]
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1ramp_exp import prod, load_panel, train_cube, month_cube, SPLITS, GAP  # noqa: E402
import t1div_lib as L  # noqa: E402

T = 288
MMAX = 152
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def static(st):
    return dict(vf=st.vf.to_numpy(np.float32), cap=st.cap.to_numpy(np.float32), lanes=st.lanes.to_numpy(np.float32),
                L=st.length_km.to_numpy(np.float32))


def make_x(V, Q, obs, pv, pq, wd, rate, S):
    """V, Q, obs, pv, pq: [stations, 288] -> (13, stations, 288) float32 (task1_unet2.make_x; occupancy channel 0)."""
    M = V.shape[0]; vf = S['vf'][:, None]; cap = S['cap'][:, None]; tt = 2 * np.pi * np.arange(T) / T
    x = np.zeros((13, M, T), np.float32)
    x[0] = np.where(obs, V / vf, 0); x[1] = np.where(obs, Q / cap, 0); x[3] = obs
    x[4] = pv / vf; x[5] = pq / cap
    x[6] = vf / 120; x[7] = S['lanes'][:, None] / 6; x[8] = np.log1p(S['L'])[:, None]
    x[9] = np.sin(tt)[None]; x[10] = np.cos(tt)[None]; x[11] = rate; x[12] = wd / 6
    return x


def loo_profiles(mv, mq, dow):
    """leave-one-day-out weekday x slot x station mean of the masked train view; fallback = full mean, then global."""
    D = mv.shape[0]
    Sv = np.zeros((7,) + mv.shape[1:]); Sq = Sv.copy(); Cv = Sv.copy(); Cq = Sv.copy()
    fv, fq = np.isfinite(mv), np.isfinite(mq)
    for w in range(7):
        s = dow == w
        Sv[w] = np.where(fv[s], mv[s], 0).sum(0); Cv[w] = fv[s].sum(0); Sq[w] = np.where(fq[s], mq[s], 0).sum(0); Cq[w] = fq[s].sum(0)
    gv, gq = np.nanmean(mv), np.nanmean(mq)
    PV = np.zeros(mv.shape, np.float32); PQ = PV.copy()
    for d in range(D):
        w = dow[d]
        cv = Cv[w] - fv[d]; cq = Cq[w] - fq[d]
        with np.errstate(all='ignore'):
            pv = (Sv[w] - np.where(fv[d], mv[d], 0)) / cv; pq = (Sq[w] - np.where(fq[d], mq[d], 0)) / cq
        PV[d] = np.where(cv > 0, pv, gv); PQ[d] = np.where(cq > 0, pq, gq)
    return PV, PQ


def build_unet(width):
    import torch, torch.nn as nn, torch.nn.functional as F

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())

    class UNet(nn.Module):
        def __init__(s, w=width):
            super().__init__(); s.e1 = blk(13, w); s.e2 = blk(w, 2 * w); s.e3 = blk(2 * w, 4 * w); s.b = blk(4 * w, 8 * w)
            s.u3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2); s.d3 = blk(8 * w, 4 * w); s.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2); s.d2 = blk(4 * w, 2 * w)
            s.u1 = nn.ConvTranspose2d(2 * w, w, 2, 2); s.d1 = blk(2 * w, w); s.o = nn.Conv2d(w, 2, 1)

        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); b = s.b(F.max_pool2d(e3, 2))
            d3 = s.d3(torch.cat([s.u3(b), e3], 1)); d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            return torch.stack([x[:, 4], x[:, 5]], 1) + s.o(d1)
    return UNet()


def load_train(panel):
    z, x, st = load_panel(panel)
    Tc = train_cube(panel, z, x)
    mv = np.where(Tc['target'], np.nan, Tc['V']); mq = np.where(Tc['target'], np.nan, Tc['Q'])
    PV, PQ = loo_profiles(mv, mq, Tc['dow'])
    tr = lambda A: np.ascontiguousarray(np.transpose(A, (0, 2, 1)))           # [D, S, 288]
    return dict(panel=panel, z=z, x=x, st=st, S=static(st), M=len(st), V=tr(Tc['V']).astype(np.float16), Q=tr(Tc['Q']).astype(np.float16),
                el=tr(Tc['elig']), PV=tr(PV).astype(np.float16), PQ=tr(PQ).astype(np.float16), dow=Tc['dow'])


def synth(P, d, rng):
    V = P['V'][d].astype(np.float32); Q = P['Q'][d].astype(np.float32); el = P['el'][d]
    rate = float(rng.choice([0.2, 0.3, 0.5])); m = el & (rng.random(el.shape) < rate)
    gap = np.zeros(T, bool)
    if rng.random() < 0.6:
        for s in rng.integers(20, 250, rng.integers(1, 3)):
            gap[s:s + GAP] = True
    obs = np.isfinite(V) & np.isfinite(Q) & ~m & ~gap[None]
    tgt = el & ~obs
    return make_x(V, Q, obs, P['PV'][d].astype(np.float32), P['PQ'][d].astype(np.float32), P['dow'][d], rate, P['S']), V, Q, tgt


def main():
    kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
    epochs, width, seed = int(kw.get('epochs', 30)), int(kw.get('width', 32)), int(kw.get('seed', 50))
    dens_w, dev, tag, bs = float(kw.get('dens', 0.05)), kw.get('device', 'mps'), kw.get('tag', ''), int(kw.get('bs', 6))
    import torch
    torch.set_num_threads(int(os.environ.get('T2LAB_THREADS', 4)))
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    Ps = [load_train(p) for p in L.PANELS]
    log('loaded', [(P['panel'], P['V'].shape) for P in Ps])
    net = build_unet(width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    items = [(pi, d) for pi, P in enumerate(Ps) for d in range(P['V'].shape[0])]
    steps_per = len(items) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps_per)
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(items)); losses = []
        for s in range(steps_per):
            X = np.zeros((bs, 13, MMAX, T), np.float32); Y = np.zeros((bs, 2, MMAX, T), np.float32)
            W = np.zeros((bs, 2, MMAX, T), np.float32); Sx = np.zeros((bs, 4, MMAX, T), np.float32)
            for k, j in enumerate(order[s * bs:(s + 1) * bs]):
                pi, d = items[j]; P = Ps[pi]; M = P['M']; St = P['S']
                x, V, Q, tgt = synth(P, d, rng)
                X[k, :, :M] = x
                Y[k, 0, :M] = np.where(tgt, V / St['vf'][:, None], 0); Y[k, 1, :M] = np.where(tgt, Q / St['cap'][:, None], 0)
                W[k, 0, :M] = np.where(tgt, 0.54 * (St['vf'][:, None] / 25) ** 2, 0)
                W[k, 1, :M] = np.where(tgt, 0.46 * (St['cap'][:, None] / St['lanes'][:, None] / 600) ** 2, 0)
                Sx[k, 0, :M] = St['vf'][:, None]; Sx[k, 1, :M] = St['cap'][:, None]; Sx[k, 2, :M] = St['lanes'][:, None]; Sx[k, 3, :M] = tgt
            X, Y, W, Sx = [torch.tensor(a, device=dev) for a in (X, Y, W, Sx)]
            out = net(X).float()
            loss = ((out - Y) ** 2 * W).sum() / (W > 0).sum().clamp(min=1)
            m = Sx[:, 3] > 0
            if dens_w > 0:
                vp = (out[:, 0] * Sx[:, 0]).clamp(min=5.0); qp = out[:, 1] * Sx[:, 1]
                vt = (Y[:, 0] * Sx[:, 0]).clamp(min=5.0); qt = Y[:, 1] * Sx[:, 1]
                dk = (qp / vp - qt / vt) / Sx[:, 2].clamp(min=1)
                loss = loss + dens_w * (dk[m] ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            losses.append(loss.item())
            if ep == 0 and s == 9:
                log(f'10 steps done; est. {(time.time() - T0) / 10 * steps_per * epochs / 3600:.2f} h total (incl. load)')
        log(f'epoch {ep}: loss {np.mean(losses):.5f}')
    torch.save(net.state_dict(), f'{L.RES}/unet{tag}.pt')
    net.eval()
    # ---- inference on the t1bspec evaluation views
    for P in Ps:
        panel = P['panel']; meta = L.load_meta(panel); St = P['S']; vf = St['vf']; ln = St['lanes']
        hv, hq = meta['hv'], meta['hq']
        for fold in (0, 1):
            out = {}
            saved = np.load(f'{L.BSPEC_RES}/{panel}_f{fold}.npz', allow_pickle=True)
            for split in SPLITS:
                Mc = month_cube(panel, P['z'], P['x'], split); obs = Mc['obs']
                EM = L.eval_masks(Mc, meta, split, fold)
                gone = EM['HO'] | EM['EB']
                mv = np.where(gone, np.nan, Mc['mv']); mq = np.where(gone, np.nan, Mc['mq'])
                ob = np.isfinite(mv) & np.isfinite(mq)
                D = mv.shape[0]; PRV = np.zeros((D, T, P['M']), np.float32); PRQ = PRV.copy()
                for d0 in range(0, D, 4):
                    dd = range(d0, min(D, d0 + 4))
                    X = np.zeros((len(dd), 13, MMAX, T), np.float32)
                    for k, d in enumerate(dd):
                        X[k, :, :P['M']] = make_x(mv[d].T, mq[d].T, ob[d].T, hv[Mc['dow'][d]].T, hq[Mc['dow'][d]].T, Mc['dow'][d],
                                                  prod.RATE[Mc['reg'][d]], St)
                    with torch.no_grad():
                        o = net(torch.tensor(X, device=dev)).float().cpu().numpy()[:, :, :P['M']]
                    for k, d in enumerate(dd):
                        PRV[d] = (o[k, 0] * vf[:, None]).T; PRQ[d] = (o[k, 1] * St['cap'][:, None]).T
                for kind, m in L.eval_cells(EM, obs):
                    k = f'{split}_{kind}'
                    d_, t_, s_ = np.nonzero(m)
                    assert np.array_equal(d_, saved[f'{k}|d']) and np.array_equal(s_, saved[f'{k}|s']), k
                    out[f'{k}|v'] = np.clip(PRV[m], 3, 1.05 * vf[s_]).astype(np.float32)
                    out[f'{k}|q'] = np.maximum(PRQ[m], 20).astype(np.float32)
                    vt = saved[f'{k}|vt']
                    log(panel, fold, k, 'unet speed rmse', round(float(np.sqrt(np.mean((out[f'{k}|v'] - vt) ** 2))), 3),
                        '| base', round(float(np.sqrt(np.mean((saved[f'base|{k}|v'] - vt) ** 2))), 3))
            np.savez_compressed(f'{L.RES}/{panel}_f{fold}_unet{tag}.npz', **out,
                                info=json.dumps(dict(epochs=epochs, width=width, seed=seed, dens=dens_w)))
    log('done')


if __name__ == '__main__':
    main()
