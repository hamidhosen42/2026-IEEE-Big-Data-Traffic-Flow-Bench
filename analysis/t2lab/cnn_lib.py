"""Track `cnn`: compact 1-D convolutional ongoing model (ensemble member next to the F03 trees).

Nothing here edits lab.py or any existing cache.  New caches live under data/t2local/cache/cnn/.

Inputs per window (all from data timestamped <= T, same information as the production tree features):
  * 12 history slots of vr = V / v_cut and fr = flow / capacity (raw with a mask, and forward-filled),
  * the slot-T speed through the production visibility simulation (62 % of the observed cells; for the evaluation sets
    the draw is the production draw rng(len(windows)) so the network sees exactly the cells the trees see),
  * queue-state channels (history, last-valid state at T), age of the last observation,
  * ramp flows of slots T-3..T-1 and T-12..T-4 (production add_ramp formulas),
  * static link attributes (lanes, capacity per lane, capacity ratios to both neighbours, length),
  * two window-level scalars (share of queued links now, share of links queued in the history).
  No prior / tod / dow channel (the network is an "identity-kept, prior-free" member like the `id` tree family).
Labels: Q at T+1..T+6 on observed cells (E), loss restricted to the production keep region (cells outside it are
scored as 0 exactly as for the tree arms).
Model: fully convolutional over the link axis (time slots are channels), residual dilated blocks, 6 logits per link.
"""
import gc, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import lab

H, F = lab.H, lab.F
THREADS = int(os.environ.get('T2LAB_THREADS', 4))
torch.set_num_threads(THREADS)
CDIR = f'{lab.CACHE}/cnn'
TDIR = f'{CDIR}/tensors'
RES = f'{lab.ROOT}/analysis/t2lab/results/cnn'
for _d in (CDIR, TDIR, RES):
    os.makedirs(_d, exist_ok=True)
P_VIS = 0.62                      # production slot-T visibility simulation
DEVICE = os.environ.get('CNN_DEVICE', 'cpu')     # 'mps' = Apple GPU (the job still runs under the run_heavy semaphore; CPU threads stay capped)
CAP = 1500                        # F03: non-run-start faith3 windows per panel
EVALS = ('official', 'offlike', 'pseudo')
PANELS = os.environ['CNN_PANELS'].split(',') if os.environ.get('CNN_PANELS') else list(lab.PANELS)     # smoke runs only


# ------------------------------------------------------------------ arrays
def static_channels(P):
    cap = P['cap']; lanes = P['lanes']
    st = np.stack([lanes / 4.0 - 1.0, cap / lanes / 1700.0 - 1.0, np.r_[cap[1:], cap[-1]] / cap - 1.0, np.r_[cap[0], cap[:-1]] / cap - 1.0,
                   np.log(np.maximum(P['length'], 0.01) / 0.25) / 2.0])
    return np.clip(st, -3, 3).astype(np.float32)                  # [5, L]


def ramp_arrays(ON, OFF, d, t, cap):
    """production add_ramp quantities: on3, off3 (mean of slots T-3..T-1 / capacity), on12 (mean of T-12..T-4): [W, 3, L], NaN = missing."""
    import warnings
    capl = cap[None, :]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        on3 = np.nanmean(ON[d[:, None], t[:, None] + np.arange(-3, 0)[None]], 1) / capl
        off3 = np.nanmean(OFF[d[:, None], t[:, None] + np.arange(-3, 0)[None]], 1) / capl
        on12 = np.nanmean(ON[d[:, None], t[:, None] + np.arange(-H, -3)[None]], 1) / capl
    return np.stack([on3, off3, on12], 1).astype(np.float32)


def hist_arrays(P, V, Fq, d, t):
    """vr, fr [W, 12, L] and the TRUE slot-T ratio vT [W, L] (NaN = missing; the visibility simulation is applied later)."""
    idx = t[:, None] + np.arange(-H, 0)[None]
    vr = V[d[:, None], idx] / P['cut'][None, None, :]
    fr = Fq[d[:, None], idx] / P['cap'][None, None, :]
    vT = V[d, t] / P['cut'][None, :]
    return vr.astype(np.float32), fr.astype(np.float32), vT.astype(np.float32)


def prep(panel, name, refresh=False):
    """tensors of every ongoing window of a cached lab window set (aligned with lab.rows(panel, name).win)."""
    f = f'{TDIR}/{panel}__{name}.npz'
    if os.path.exists(f) and not refresh:
        z = np.load(f)
        return {k: z[k] for k in z.files}
    P = lab.load_panel(panel)
    path = f'{lab.CACHE}/rows/{panel}__{name}__{lab.ONGOING}'
    if not os.path.exists(f'{path}/meta.json'):
        lab.rows(panel, name)
    win = pd.read_pickle(f'{path}/win.pkl'); z = np.load(f'{path}/aux.npz')
    W, L = len(win), len(P['links'])
    d = win.d.to_numpy().astype(np.int64); t = win.t.to_numpy().astype(np.int64)
    wi, li, h, keep5 = z['wi'], z['li'], z['h'], z['keep5']
    keep = np.zeros((W, 5, L), bool); m1 = h == 1
    for k in range(5):
        m = m1 & keep5[:, k]; keep[wi[m], k, li[m]] = True
    vr, fr, vT = hist_arrays(P, P['V'], P['Fq'], d, t)
    out = dict(d=d, t=t, fold=win.fold.to_numpy().astype(np.int8), runstart=win.runstart.to_numpy().astype(bool),
               vr=vr.astype(np.float16), fr=fr.astype(np.float16), vT=vT.astype(np.float16),
               ramp=ramp_arrays(P['ON'], P['OFF'], d, t, P['cap']).astype(np.float16),
               Y=z['truth'], M=z['ev'], keep=keep, qlast=z['qlast'],
               vis=np.random.default_rng(W).random((W, L)) < P_VIS)        # the production draw of make_rows for this window list
    np.savez(f, **out)
    return out


def _cap_idx(idx, cap, seed=0):
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


class PanelData:
    """arrays of a list of windows of one panel (float32 on load)."""

    def __init__(self, panel, parts):
        """parts: list of (prep dict, window index array, weight array)."""
        cat = lambda key: np.concatenate([p[key][ix] for p, ix, _ in parts])
        self.panel = panel
        self.vr = torch.from_numpy(cat('vr').astype(np.float32)); self.fr = torch.from_numpy(cat('fr').astype(np.float32))
        self.vT = torch.from_numpy(cat('vT').astype(np.float32)); self.ramp = torch.from_numpy(cat('ramp').astype(np.float32))
        self.Y = torch.from_numpy(cat('Y')); self.M = torch.from_numpy(cat('M'))
        self.vis = torch.from_numpy(cat('vis'))
        self.w = torch.from_numpy(np.concatenate([w for _, _, w in parts]).astype(np.float32))
        self.static = torch.from_numpy(static_channels(lab.load_panel(panel)))
        self.n = len(self.w); self.L = self.vr.shape[2]
        self.keep = None

    def set_keep(self, keep):
        self.keep = torch.from_numpy(keep)


def train_panels(k, cap=CAP, wd=0.3, panels=None, sets='F03'):
    """F03 windows outside fold k (k = 4: all): run-starts (weight 1.0) + non-run-start faith3 windows (<= cap per panel, rng(0),
    weight wd) -- the same windows as exp_firstcross_lib.ARMS['F03'].  keep = production keep region under the fold-k prior."""
    out = []
    for panel in panels or PANELS:
        f3 = prep(panel, 'faith3'); rs = prep(panel, 'runstart')
        i3 = _cap_idx(np.where(~f3['runstart'])[0], cap); i3 = i3[f3['fold'][i3] != k]
        ir = np.where(rs['fold'] != k)[0]
        pd_ = PanelData(panel, [(f3, i3, np.full(len(i3), wd, np.float32)), (rs, ir, np.ones(len(ir), np.float32))])
        pd_.set_keep(np.concatenate([f3['keep'][i3][:, k], rs['keep'][ir][:, k]]))
        out.append(pd_)
    return out


def eval_panel(panel, name, k, widx=None):
    """evaluation windows of fold k (or the given window indices) with the production slot-T draw; returns (PanelData, widx)."""
    z = prep(panel, name)
    widx = np.where(z['fold'] == k)[0] if widx is None else np.asarray(widx)
    if not len(widx):
        return None, widx
    pd_ = PanelData(panel, [(z, widx, np.ones(len(widx), np.float32))])
    pd_.set_keep(z['keep'][widx][:, k])
    return pd_, widx


# ------------------------------------------------------------------ channels (torch)
N_CH = 12 * 5 + 1 + 6 + 6 + 5 + 2


def assemble(vr, fr, vT, ramp, static, vis=None, hist_drop=0.0, gen=None):
    """channels [B, N_CH, L].  vr, fr [B, 12, L] (NaN = missing), vT [B, L] true slot-T ratio, vis [B, L] bool visibility of
    slot T (None: use vT as given), ramp [B, 3, L] (NaN = missing), static [5, L].  hist_drop: extra i.i.d. dropout of
    history cells (augmentation only)."""
    B, _, L = vr.shape
    if hist_drop > 0:
        drop = torch.rand(vr.shape, generator=gen) < hist_drop
        vr = torch.where(drop, torch.full_like(vr, float('nan')), vr); fr = torch.where(drop, torch.full_like(fr, float('nan')), fr)
    if vis is not None:
        vT = torch.where(vis, vT, torch.full_like(vT, float('nan')))
    m = torch.isfinite(vr)
    cl = lambda a: a.clamp(0.0, 2.5) - 1.0
    v0 = torch.where(m, cl(vr), torch.zeros_like(vr))
    f0 = torch.where(m, fr.clamp(0.0, 1.5), torch.zeros_like(fr))
    qh = (m & (vr <= 1.0)).float()
    vf = vr.clone(); ff = fr.clone()
    age = torch.where(m[:, 0], torch.zeros(B, L), torch.full((B, L), float(H)))
    for i in range(1, H):
        nan_i = ~m[:, i]
        vf[:, i] = torch.where(nan_i, vf[:, i - 1], vf[:, i]); ff[:, i] = torch.where(nan_i, ff[:, i - 1], ff[:, i])
        age = torch.where(nan_i, age + 1.0, torch.zeros_like(age))
    mf = torch.isfinite(vf)
    vf0 = torch.where(mf, cl(vf), torch.zeros_like(vf))
    hasT = torch.isfinite(vT)
    last = torch.where(hasT, vT, vf[:, -1]); has_last = torch.isfinite(last)
    qlast = (has_last & (last <= 1.0)).float()
    last0 = torch.where(has_last, cl(last), torch.zeros_like(last))
    ageT = torch.where(hasT, torch.zeros_like(age), (age + 1.0).clamp(max=float(H))) / float(H)
    fl = ff[:, -1]; fl0 = torch.where(torch.isfinite(fl), fl.clamp(0.0, 1.5), torch.zeros_like(fl))
    rm = torch.isfinite(ramp)
    r0 = torch.where(rm, (ramp * 10.0).clamp(0.0, 3.0), torch.zeros_like(ramp))
    g1 = qlast.mean(1, keepdim=True).expand(B, L); g2 = qh.amax(1).mean(1, keepdim=True).expand(B, L)
    x = torch.cat([v0, m.float(), vf0, f0, qh,
                   fl0[:, None],
                   torch.where(hasT, cl(vT), torch.zeros_like(vT))[:, None], hasT.float()[:, None], last0[:, None], qlast[:, None],
                   has_last.float()[:, None], ageT[:, None],
                   r0, rm.float(),
                   static[None].expand(B, -1, -1),
                   g1[:, None], g2[:, None]], 1)
    return x, qlast


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, c, dil, k=3, groups=8, drop=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, c); self.c1 = nn.Conv1d(c, c, k, padding=dil * (k // 2), dilation=dil)
        self.n2 = nn.GroupNorm(groups, c); self.c2 = nn.Conv1d(c, c, k, padding=dil * (k // 2), dilation=dil)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        h = self.c1(Fn.silu(self.n1(x)))
        h = self.c2(self.drop(Fn.silu(self.n2(h))))
        return x + h


class Net(nn.Module):
    def __init__(self, cin=N_CH, c=64, dils=(1, 2, 4, 8, 1, 2, 4, 8), drop=0.05):
        super().__init__()
        self.stem = nn.Conv1d(cin, c, 3, padding=1)
        self.blocks = nn.ModuleList([Block(c, d, drop=drop) for d in dils])
        self.norm = nn.GroupNorm(8, c)
        self.head = nn.Conv1d(c, F, 1)
        self.pers = nn.Parameter(torch.tensor(2.0))            # persistence shortcut: logit += pers * (2 qlast - 1)

    def forward(self, x, qlast):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head(Fn.silu(self.norm(h))) + self.pers * (2.0 * qlast[:, None] - 1.0)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ training
def batches(pds, bs, rng):
    out = []
    for i, p in enumerate(pds):
        perm = rng.permutation(p.n)
        out += [(i, perm[j:j + bs]) for j in range(0, p.n, bs)]
    order = rng.permutation(len(out))
    return [out[j] for j in order]


def train(pds, epochs=12, bs=32, lr=2e-3, wd=1e-2, seed=0, c=64, dils=(1, 2, 4, 8, 1, 2, 4, 8), drop=0.05, hist_drop=0.0,
          ema_decay=0.998, max_minutes=None, log=lab.log, tag='', device=None):
    """returns the EMA model.  Loss: BCE on observed cells inside the keep region, weighted by the window weight."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
    dev = torch.device(device or DEVICE)
    model = Net(c=c, dils=dils, drop=drop); ema = Net(c=c, dils=dils, drop=drop); ema.load_state_dict(model.state_dict())
    model.to(dev); ema.to(dev)
    for p in ema.parameters():
        p.requires_grad_(False)
    n_steps = epochs * sum((p.n + bs - 1) // bs for p in pds)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps, pct_start=0.08, anneal_strategy='cos')
    log(f'{tag} train: {sum(p.n for p in pds)} windows, {n_steps} steps, {n_params(model)} parameters, threads {torch.get_num_threads()}, device {dev}')
    t0 = time.time(); step = 0
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for i, ix in batches(pds, bs, rng):
            p = pds[i]; ix = torch.from_numpy(ix)
            vis = torch.rand((len(ix), p.L), generator=gen) < P_VIS                 # fresh slot-T visibility draw every step
            x, ql = assemble(p.vr[ix], p.fr[ix], p.vT[ix], p.ramp[ix], p.static, vis, hist_drop, gen)
            logit = model(x.to(dev), ql.to(dev))
            mask = ((p.M[ix] & p.keep[ix][:, None, :]).float() * p.w[ix][:, None, None]).to(dev)
            loss = (Fn.binary_cross_entropy_with_logits(logit, p.Y[ix].float().to(dev), reduction='none') * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            with torch.no_grad():
                dcy = min(ema_decay, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
            tot += loss.item(); nb += 1
        log(f'{tag} epoch {ep + 1}/{epochs}: loss {tot / max(nb, 1):.4f}, {time.time() - t0:.0f}s')
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            log(f'{tag} stopped by the time limit after epoch {ep + 1}'); break
    ema.eval()
    return ema.to('cpu')


@torch.no_grad()
def predict(model, p, bs=128, use_vis=True):
    """probabilities [n, 6, L] (float32), zero outside the keep region (as the tree arms)."""
    model.eval(); out = np.zeros((p.n, F, p.L), np.float32)
    dev = next(model.parameters()).device
    for j in range(0, p.n, bs):
        ix = torch.arange(j, min(j + bs, p.n))
        x, ql = assemble(p.vr[ix], p.fr[ix], p.vT[ix], p.ramp[ix], p.static, p.vis[ix] if use_vis else None)
        pr = torch.sigmoid(model(x.to(dev), ql.to(dev))).cpu()
        if p.keep is not None:
            pr = pr * p.keep[ix][:, None, :].float()
        out[j:j + len(ix)] = pr.numpy()
    return out


def tree_ref(set_name, panel):
    """reference: mean of the cached OOF probabilities of fc_F03_np255 and fc_F03_id255."""
    a, da = lab.load_probs('fc_F03_np255', set_name, panel); b, db = lab.load_probs('fc_F03_id255', set_name, panel)
    return 0.5 * (a + b), da & db
