"""t1lwrproj step 2: B1 conservation-aware projection on the t1lab D12_I405_S replica (4 pseudo-months A-D, real bag members).

Base = production chain: mix bag (0.5 FD + 0.5 no-FD, seeds 0-2) -> recon (gate 0.6, a 0.75, logr 0.5) -> TV (production constants).
B1 (on top of the base): per pseudo-month, unknowns y = N = k*L at the Task 1 target cells (released cells fixed), minimise
    sum_target (y - y0)^2 + lam * sum_pairs (y(t+1) - y(t) - c(t))^2 + mu * sum_gate (y - N_FD(v))^2
over the consecutive eligible pairs touching a target (exactly the pairs of the LWR proxy), c(t) = dt*(q_up(t) - q(t) + on - off)
(ramps: none released locally for D12_I405_S train -> 0; their r with dN is ~0 on the 8 other panels, see t1lwrproj_diag.py).
The density change is mapped back like the production TV: gate cells (v < 0.6 vf) split it 0.75 to speed, free cells put it on flow,
|log r| <= 0.5.
Flux variants:  real   c from the current field (released truth + base predictions), upstream = index-1 (o+) or index+1 (o-)
                oracle c from the TRUE flows everywhere (information upper bound of the conservation term)
                zero   c = 0 (pure quadratic smoothing of N; control that separates smoothing from conservation information)
                trueDN c = true dN (sanity check of the solver; not attainable)
Headroom rows: truth at the targets (oracle that fixes only the missing cells), base before/after recon+TV.
Usage: t1lwrproj_exp.py"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t1lab as L  # noqa: E402

RES = os.path.join(ROOT, 'analysis', 't2lab', 'results', 't1lwrproj')
os.makedirs(RES, exist_ok=True)
DT = 5.0 / 60.0
SEEDS = [0, 1, 2]


class B1:
    def __init__(self, M):
        self.M = M
        Z = M.Z
        self.L = M.L_s
        self.ln = M.lanes_s
        nd, T, S = M.shape
        self.idx = -np.ones(M.shape, np.int64); self.idx[M.tm] = np.arange(M.tm.sum())
        d, t, s = np.nonzero(M.touch)                                  # pair (d, t, s) -> (d, t+1, s)
        self.pd, self.pt, self.ps = d, t, s
        i0 = self.idx[d, t, s]; i1 = self.idx[d, t + 1, s]
        n = len(d); rows = np.arange(n)
        m1 = i1 >= 0; m0 = i0 >= 0
        self.A = sp.csr_matrix((np.concatenate([np.ones(m1.sum()), -np.ones(m0.sum())]),
                                (np.concatenate([rows[m1], rows[m0]]), np.concatenate([i1[m1], i0[m0]]))), shape=(n, M.tm.sum()))
        self.m0, self.m1 = m0, m1
        P = M.post(); self.P = P
        self.vf_row = P.vf; self.vf_ok = P.sw.vf_ok

    def field(self, v, q):
        M = self.M
        V = M.V0.copy(); Q = M.Q0.copy(); V[M.tm] = v; Q[M.tm] = q
        return V, Q

    def flux(self, Q, orient):
        """c at every pair start: dt*(q_up - q_own); NaN upstream (corridor end / ineligible) -> 0 (no constraint information)."""
        up = np.full_like(Q, np.nan)
        if orient > 0:
            up[..., 1:] = Q[..., :-1]
        else:
            up[..., :-1] = Q[..., 1:]
        c = DT * (up - Q)
        c = c[self.pd, self.pt, self.ps]
        return np.where(np.isfinite(c), c, 0.0)

    def solve(self, v, q, c, lam, mu=0.0):
        M = self.M
        V, Q = self.field(v, q)
        N = Q / np.clip(V, 1, None) * self.L[None, None, :]
        y0 = N[M.tm]
        # fixed parts of each pair
        d, t, s = self.pd, self.pt, self.ps
        fix1 = np.where(self.m1, 0.0, N[d, t + 1, s]); fix0 = np.where(self.m0, 0.0, N[d, t, s])
        b = c - fix1 + fix0
        n = len(y0)
        H = sp.identity(n, format='csr') + lam * (self.A.T @ self.A)
        rhs = y0 + lam * (self.A.T @ b)
        if mu > 0:
            g = self.vf_ok & (v < 0.6 * self.vf_row)
            yfd = self.P.kfd(v) * self.ln[M.s_of] * self.L[M.s_of]
            H = H + sp.diags(mu * g.astype(float))
            rhs = rhs + mu * np.where(g, yfd, 0.0)
        y = spl.spsolve(H.tocsc(), rhs)
        r = np.exp(np.clip(np.log(np.maximum(y, 1e-9) / np.maximum(y0, 1e-9)), -0.5, 0.5))
        gg = self.vf_ok & (v < 0.6 * self.vf_row)
        a = np.where(gg, 0.75, 0.0)
        vn = np.where(gg, np.clip(v * r ** (-a), 3, 1.05 * self.vf_row), v)
        qn = np.maximum(q * r ** (1 - a), 20.0)
        return vn, qn


def main():
    T0 = time.time()
    Z = L.Data()
    out = []
    stats = {}
    for nm in ('A', 'B', 'C', 'D'):
        M = L.Month(Z, nm)
        v0, q0 = L.load_bag(nm, 'mix', SEEDS)
        vb, qb = L.run_pipeline(M, v0, q0, L.PROD_PIPE)
        P = B1(M)
        arms = {'truth': (M.vt, M.qt), 'bag_raw': (v0, q0), 'PROD': (vb, qb)}
        Vb, Qb = P.field(vb, qb)
        Qt = M.Q0
        dNt = M.dNt[P.pd, P.pt, P.ps]
        flux = {'real+': P.flux(Qb, 1), 'real-': P.flux(Qb, -1), 'oracle+': P.flux(Qt, 1), 'oracle-': P.flux(Qt, -1),
                'zero': np.zeros(len(P.pd)), 'trueDN': np.where(np.isfinite(dNt), dNt, 0.0)}
        for fk, c in flux.items():
            for lam in (0.03, 0.1, 0.3, 1.0):
                arms[f'B1[{fk},l{lam}]'] = P.solve(vb, qb, c, lam)
        for mu in (0.1, 0.3):
            arms[f'B1[real+,l0.1,mu{mu}]'] = P.solve(vb, qb, flux['real+'], 0.1, mu)
            arms[f'B1[fdonly,mu{mu}]'] = P.solve(vb, qb, flux['zero'], 0.0, mu)
        for a, (pv, pq) in arms.items():
            stats.setdefault(a, {})[nm] = M.stats(pv, pq)
        stats.setdefault('_reg', {})[nm] = M.reg
        L.log(f'month {nm}: {len(arms)} arms, targets {M.tm.sum()}, pairs {len(P.pd)}')
    np.savez(f'{RES}/stats_D12_I405_S.npz', **{f'{a}|{nm}': s for a, d in stats.items() if a != '_reg' for nm, s in d.items()},
             **{f'_reg|{nm}': r for nm, r in stats['_reg'].items()})
    months = ('A', 'B', 'C', 'D')
    W = {nm: L.boot_weights(stats['_reg'][nm], B=200, seed=1) for nm in months}
    print('\nlevels (regime means) per arm, mean over the 4 pseudo-months; dJ vs PROD per month; wins; worst regime dJ')
    print(f'{"arm":28s} {"S_state":>8s} {"LWR":>7s} {"S_FD":>7s} {"J":>8s} | {"dJ mean":>9s} {"SE":>7s} | dJ A..D | win | min reg dJ')
    for a in stats:
        if a == '_reg':
            continue
        ag = [L.aggregate(stats[a][nm], stats['_reg'][nm]) for nm in months]
        dj, se, regd = [], [], []
        for nm in months:
            d0, s0 = L.paired_delta(stats['PROD'][nm], stats[a][nm], stats['_reg'][nm], W[nm])
            dj.append(d0); se.append(s0)
            gb = L.aggregate(stats['PROD'][nm], stats['_reg'][nm]); ga = L.aggregate(stats[a][nm], stats['_reg'][nm])
            for r in L.REGS:
                if f'S_state_{r}' in ga:
                    regd.append(0.35 * (ga[f'S_state_{r}'] - gb[f'S_state_{r}']) + 0.10 * (ga[f'LWR_{r}'] - gb[f'LWR_{r}'])
                                + 0.05 * (ga[f'S_FD_{r}'] - gb[f'S_FD_{r}']))
        line = (f'{a:28s} {np.mean([g["S_state"] for g in ag]):8.5f} {np.mean([g["LWR"] for g in ag]):7.4f} '
                f'{np.mean([g["S_FD"] for g in ag]):7.4f} {np.mean([g["J"] for g in ag]):8.5f} | {np.mean(dj):+9.5f} '
                f'{np.sqrt(np.sum(np.square(se))) / 4:7.5f} | ' + ' '.join(f'{x:+.5f}' for x in dj)
                + f' | {sum(x > 0 for x in dj)}/4 | {min(regd):+.5f}')
        print(line, flush=True)
        out.append(line)
    open(f'{RES}/report_D12_I405_S.txt', 'w').write('\n'.join(out) + '\n')
    L.log('done', f'{time.time() - T0:.0f}s')


if __name__ == '__main__':
    main()
