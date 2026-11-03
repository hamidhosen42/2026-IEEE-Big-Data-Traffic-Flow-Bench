"""Experimental posterior-over-trajectories queue decoder.

Not a trained/validated model. Uses only out-of-fold marginal probabilities and
TRAIN-split trajectories from disjoint folds. A panel-local empirical bank is
translated spatially and used as correlated possible outcomes.

No post-origin data are read for the evaluated window. Never use validation or
private future truth to construct or choose this bank.
"""
from __future__ import annotations
import numpy as np


def baseline_decode(p: np.ndarray) -> np.ndarray:
    """Byte-equivalent logic to consolidation/tfb/stages/t2_occ.py:decode."""
    flat = np.asarray(p, dtype=np.float64).ravel()
    if not len(flat) or not np.isfinite(flat).all() or np.min(flat) < 0 or np.max(flat) > 1:
        raise ValueError('probabilities must be finite and in [0,1]')
    order = np.argsort(-flat)
    ps = flat[order]
    cs = np.cumsum(ps)
    j = int(np.argmax(cs / (np.arange(1, len(ps)+1) + (cs[-1]-cs))))
    out = np.zeros(len(ps), dtype=bool)
    out[order[:j+1]] = True
    return out.reshape(np.asarray(p).shape)


def iou(pred: np.ndarray, true: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Organizer-style score for one window; count only eligible cells."""
    valid = np.ones_like(pred, dtype=bool) if valid is None else np.asarray(valid, bool)
    pred = np.asarray(pred, bool) & valid
    true = np.asarray(true, bool) & valid
    union = np.count_nonzero(pred | true)
    return float(np.count_nonzero(pred & true) / union) if union else 1.0


def shift_trajectory(x: np.ndarray, shift: int) -> np.ndarray:
    """Spatial shift, with zero padding rather than wraparound."""
    x = np.asarray(x, dtype=bool)
    if x.ndim != 2:
        raise ValueError('trajectory must be 2-D [horizon, link]')
    out = np.zeros_like(x)
    n = x.shape[1]
    if 0 <= shift < n:
        out[:, shift:] = x[:, :n-shift]
    elif -n < shift < 0:
        out[:, :n+shift] = x[:, -shift:]
    return out


def _anchor(x: np.ndarray, condition: str) -> int:
    # For onset, use the last step because the five earlier steps are empty.
    weight = np.asarray(x[-1] if condition == 'queue_onset' else x.sum(axis=0), dtype=float)
    return int(np.argmax(weight))



def causal_bank_mask(eval_fold: int, eval_day: int, bank_folds: np.ndarray,
                     bank_days: np.ndarray, eligible: np.ndarray, truth: np.ndarray,
                     *, condition: str) -> np.ndarray:
    """Only older TRAIN days outside the target fold; no time-travel labels."""
    bank_folds=np.asarray(bank_folds); bank_days=np.asarray(bank_days)
    eligible=np.asarray(eligible, bool); truth=np.asarray(truth, bool)
    mask=(bank_folds!=eval_fold)&(bank_days<eval_day)
    mask &= eligible.mean(axis=(1,2))>=.65
    mask &= truth.sum(axis=(1,2))>0
    if condition=='queue_onset':
        mask &= ~truth[:,:-1].any(axis=(1,2))
    elif condition!='queue_ongoing':
        raise ValueError(condition)
    return mask

def correlated_decode(p: np.ndarray, bank: np.ndarray, *, condition: str,
                      temperature: float = 0.02, anchors: int = 3,
                      bank_limit: int = 128, posterior_limit: int = 64,
                      margin: float = 0.005) -> tuple[np.ndarray, dict]:
    """Returns (prediction, diagnostic), without using the target window's label.

    Candidate construction: translate historical TRAIN trajectories to top-k
    probability anchors (same panel); posterior weights: Gibbs kernel on
    mean-square discrepancy with marginal P6; choose candidate maximizing
    weighted expected IoU over sampled possible whole trajectories. Baseline
    is included in the candidate set and has an expected-IoU margin guard.

    Fixed hyperparameters: optimize only on TRAIN OOF; do not tune using April.
    """
    p = np.asarray(p, dtype=np.float64)
    pred0 = baseline_decode(p)
    bank = np.asarray(bank, dtype=bool)
    if p.ndim != 2 or bank.ndim != 3 or bank.shape[1:] != p.shape:
        raise ValueError('expected p[6,L], bank[N,6,L]')
    if condition not in {'queue_onset','queue_ongoing'}:
        raise ValueError(condition)
    if temperature <= 0 or not 1 <= anchors <= p.shape[1] or posterior_limit < 1 or bank_limit < 1:
        raise ValueError('invalid hyperparameters')
    if len(bank) == 0:
        return pred0, {'changed': False, 'reason': 'no_bank'}
    if condition == 'queue_onset':
        if np.any(bank[:, :-1]):
            raise ValueError('onset bank has positives before last horizon step')
        p = p.copy()
        p[:-1] = 0
        pred0 = baseline_decode(p)
    activity = p[-1] if condition == 'queue_onset' else p.sum(axis=0)
    target_links = np.argsort(-activity, kind='stable')[:anchors]
    # Deterministically thin the template bank; do not fit any hidden labels.
    if len(bank) > bank_limit:
        bank = bank[np.linspace(0, len(bank)-1, bank_limit, dtype=int)]
    candidates = []
    for b in bank:
        if not b.any():
            continue
        origin = _anchor(b, condition)
        for link in target_links:
            t = shift_trajectory(b, int(link)-origin)
            if t.any():
                candidates.append(t)
    if not candidates:
        return pred0, {'changed': False, 'reason': 'empty_templates'}
    cand = np.unique(np.asarray(candidates, dtype=bool).reshape(len(candidates), -1), axis=0)
    if len(cand) == 0:
        return pred0, {'changed': False, 'reason': 'empty_candidates'}
    pp = p.ravel()
    dist = np.mean((cand.astype(np.float64)-pp[None,:])**2, axis=1)
    ix = np.argsort(dist, kind='stable')[:posterior_limit]
    samples = cand[ix]
    d = dist[ix]
    logw = -(d - np.min(d))/temperature
    w = np.exp(logw - np.max(logw)); w /= np.sum(w)
    choices = np.vstack([pred0.reshape(1,-1), samples])
    choices = np.unique(choices, axis=0)
    inter = choices.astype(np.float64) @ samples.astype(np.float64).T
    union = choices.sum(axis=1)[:,None] + samples.sum(axis=1)[None,:] - inter
    expected = np.sum(w[None,:] * np.divide(inter, union, out=np.ones_like(inter), where=union>0), axis=1)
    base_ix = np.flatnonzero(np.all(choices == pred0.ravel(), axis=1))
    assert len(base_ix) == 1
    base_score = float(expected[base_ix[0]])
    best_ix = int(np.argmax(expected))
    best_score = float(expected[best_ix])
    change = bool(best_score > base_score + margin)
    return (choices[best_ix] if change else pred0.ravel()).reshape(p.shape).astype(bool), {
        'changed': change, 'reason': 'higher_posterior_expected_IoU' if change else 'baseline_margin',
        'n_samples': int(len(samples)), 'base_expected': base_score,
        'best_expected': best_score, 'posterior_margin': best_score-base_score,
    }
