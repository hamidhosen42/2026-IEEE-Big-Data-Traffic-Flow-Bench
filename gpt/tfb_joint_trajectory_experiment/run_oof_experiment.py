"""Evaluate a new correlated Task-2 decoder on EXISTING OOF train cache.

No retraining, no Kaggle calls, no leaderboard optimization. Reads only
analysis/t2lab train cache, never March/April withheld truth or post-T features.

Example:
 python run_oof_experiment.py --cache-root data/t2local/cache --arm A_np255 \
     --set offlike --bank-set faith3 --condition queue_ongoing --output results.json

NOTE: Current lab's RowCache truth is observed/eligible-only proxy truth;
therefore this is a diagnostic, not official IoU. A train-fold OOF guard
ensures the candidate bank never includes the evaluated fold's labels.
"""
from __future__ import annotations
import argparse, glob, json, pathlib, sys
import numpy as np
import pandas as pd
from joint_decoder import correlated_decode, baseline_decode, iou, causal_bank_mask


def group_score(df):
    if df.empty:
        return float('nan')
    pc = df.groupby(['family','panel','cond'], observed=True).iou.mean()
    return float(pc.groupby(level=[0,1]).mean().groupby(level=0).mean().mean())


def bootstrapped_delta(df, repeats=1000, seed=41):
    """Resample entire calendar days within each panel, not individual cells."""
    rng = np.random.default_rng(seed)
    groups = [(a, list(g.groupby('d'))) for a,g in df.groupby(['family','panel','cond'])]
    scores = []
    for _ in range(repeats):
        records = []
        for (family,panel,cond), daygroups in groups:
            js = rng.integers(0,len(daygroups),len(daygroups))
            part = pd.concat([daygroups[j][1] for j in js], ignore_index=True)
            records.append(dict(family=family,panel=panel,cond=cond,
                                delta=float((part.new-part.base).mean())))
        acc = pd.DataFrame(records)
        scores.append(float(acc.groupby(['family','panel']).delta.mean().groupby('family').mean().mean()))
    return dict(lo=float(np.quantile(scores,.025)),hi=float(np.quantile(scores,.975)),
                p_positive=float(np.mean(np.asarray(scores)>0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-root', required=True, type=pathlib.Path)
    ap.add_argument('--arm', default='A_np255')
    ap.add_argument('--set', default='offlike')
    ap.add_argument('--bank-set', default='faith3')
    ap.add_argument('--condition', choices=['queue_ongoing','queue_onset'],default='queue_ongoing')
    ap.add_argument('--temperature', type=float, default=.02)
    ap.add_argument('--anchors', type=int, default=3)
    ap.add_argument('--bank-limit',type=int,default=128)
    ap.add_argument('--posterior-limit',type=int,default=64)
    ap.add_argument('--margin',type=float,default=.005)
    ap.add_argument('--max-windows',type=int,default=None,help='Debug only; cannot establish pass gate')
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('oof_joint_results.json'))
    args=ap.parse_args()
    if args.set not in {'official','offlike','pseudo'}:
        ap.error('only TRAIN-derived sets official/offlike/pseudo supported')
    root=args.cache_root
    files=sorted(root.glob(f'oof/{args.arm}/probs__{args.set}__*.npz'))
    if not files:
        raise FileNotFoundError(f'No OOF probabilities: {root}/oof/{args.arm}/probs__{args.set}__*.npz')
    records=[]
    for f in files:
        panel=f.stem.split(f'probs__{args.set}__',1)[-1]
        path=root/'rows'/f'{panel}__{args.set}__{args.condition}'
        bankpath=root/'rows'/f'{panel}__{args.bank_set}__{args.condition}'
        if not path.joinpath('aux.npz').exists() or not bankpath.joinpath('aux.npz').exists():
            raise FileNotFoundError(f'Missing TRAIN row caches for panel {panel}. Build lab.rows before running.')
        P=np.load(f); A=np.load(path/'aux.npz'); B=np.load(bankpath/'aux.npz')
        win=pd.read_pickle(path/'win.pkl'); bwin=pd.read_pickle(bankpath/'win.pkl')
        pp=P['P6'].astype(np.float64); done=P['done'].astype(bool)
        if pp.shape != A['truth'].shape or len(done)!=len(win) or len(bwin)!=len(B['truth']):
            raise AssertionError(f'{panel}: OOF shape mismatch')
        if not (np.array_equal(P['d'],win.d.to_numpy()) and np.array_equal(P['t'],win.t.to_numpy())):
            raise AssertionError(f'{panel}: OOF origin mismatch')
        # Fail closed: evaluation must be OOF, and each fold's bank cannot use same fold.
        unique_folds=np.unique(win.fold.to_numpy())
        if len(unique_folds)<2:
            raise AssertionError(f'{panel}: at least 2 evaluation folds required')
        bank_truth=B['truth'].astype(bool) & B['ev'].astype(bool)
        bank_eligible=B['ev'].astype(bool)
        bank_folds=bwin.fold.to_numpy()
        bank_days=bwin.d.to_numpy()
        selected=np.flatnonzero(done)
        if args.max_windows is not None: selected=selected[:args.max_windows]
        for i in selected:
            fold=int(win.fold.iloc[i]); d=int(win.d.iloc[i])
            # Causal historical bank: no same-fold or same/future-day horizon labels.
            bb=causal_bank_mask(fold,d,bank_folds,bank_days,bank_eligible,bank_truth,condition=args.condition)
            # For the first chronological block, no historical bank is correct: baseline fallback.
            # Never allow future TRAIN labels simply to populate a bank.
            valid=A['ev'][i].astype(bool); truth=A['truth'][i].astype(bool)
            base=baseline_decode(pp[i]); new, diag=correlated_decode(pp[i],bank_truth[bb],
                  condition=args.condition,temperature=args.temperature,anchors=args.anchors,
                  bank_limit=args.bank_limit,posterior_limit=args.posterior_limit,margin=args.margin)
            records.append(dict(panel=panel,family=panel.rsplit('_',1)[0],cond=args.condition,
                d=d,t=int(win.t.iloc[i]),fold=fold,
                base=iou(base,truth,valid),new=iou(new,truth,valid),
                changed=int(diag['changed']),bank_count=int(np.count_nonzero(bb))))
        print(f'{panel}: evaluated {np.count_nonzero(done)} OOF windows',flush=True)
    df=pd.DataFrame(records)
    if df.empty: raise AssertionError('No scored train windows')
    base=group_score(df.assign(iou=df.base)); new=group_score(df.assign(iou=df.new))
    diff=new-base; panels=df.groupby('panel').apply(lambda g: float((g.new-g.base).mean()), include_groups=False)
    ci=bootstrapped_delta(df) if len(df)>10 else None
    # For a single condition, organizer total coefficient = 0.30 / 2 = 0.15.
    comp_delta=.15*diff
    # Predeclared deployment criteria for a research hypothesis, not official validation.
    gate=(args.max_windows is None and len(panels)>=8 and diff>=.01 and (panels>0).sum()>=7
          and ci is not None and ci['lo']>0)
    result=dict(experiment='train_OOF_joint_trajectory',train_only=True,arm=args.arm,set=args.set,
       bank_set=args.bank_set,condition=args.condition,windows=len(df),panels=len(panels),
       config=dict(temperature=args.temperature,anchors=args.anchors,bank_limit=args.bank_limit,
                   posterior_limit=args.posterior_limit,margin=args.margin),
       base_iou=base,new_iou=new,delta_condition_iou=diff,
       delta_composite_if_other_tasks_unchanged=comp_delta,
       changed_windows=int(df.changed.sum()),panel_deltas=panels.to_dict(),day_block_bootstrap=ci,
       diagnostic_gate='PASS' if gate else 'FAIL',
       deployment_gate='UNVERIFIED: requires chronological model-training audit, March transfer, and post-T rerun parity',
       restrictions=['TRAIN-only template bank from days strictly before eval origin and excluding its fold',
                     'Existing lab OOF P6 may be trained using future train folds: decoder-only offline diagnostic, not end-to-end causal proof',
                     'Official leaderboard cannot be predicted from proxy OOF',
                     'No March/April model or threshold selection',
                     'No data after evaluated origin T enters its features (existing P6 only)'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True))
    df.to_csv(args.output.with_suffix('.windows.csv'),index=False)
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    sys.exit(main())
