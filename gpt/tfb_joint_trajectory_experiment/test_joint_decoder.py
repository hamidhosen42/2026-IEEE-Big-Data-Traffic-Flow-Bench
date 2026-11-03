"""Unit/synthetic tests; NOT evidence of leaderboard improvement."""
import numpy as np
from joint_decoder import baseline_decode, correlated_decode, iou, shift_trajectory, causal_bank_mask


def test_baseline_matches_production_reference():
    rng=np.random.default_rng(2026)
    for _ in range(30):
        p=rng.random((6,21))
        order=np.argsort(-p.ravel()); ps=p.ravel()[order]; cs=np.cumsum(ps)
        j=np.argmax(cs/(np.arange(1,len(ps)+1)+(cs[-1]-cs)))
        expected=np.zeros(len(ps),bool); expected[order[:j+1]]=1
        assert np.array_equal(baseline_decode(p).ravel(),expected)


def test_shift_zero_pad_and_truth_scoring():
    x=np.zeros((6,7),bool); x[-1,1:3]=True
    y=shift_trajectory(x,4)
    assert y[-1,5:].tolist()==[True,True] and y.sum()==2
    assert shift_trajectory(x,-9).sum()==0
    assert iou(x,x)==1
    assert iou(x, np.zeros_like(x))==0


def test_joint_onset_validity_and_no_labels_used():
    p=np.zeros((6,20),float); p[-1,11]=.67; p[-1,12]=.28; p[-1,9]=.24
    bank=np.zeros((6,6,20),bool)
    for i in range(6): bank[i,-1,3+i:5+i]=True
    a,info=correlated_decode(p,bank,condition='queue_onset')
    assert a.shape==p.shape and a.dtype==bool and not a[:-1].any()
    assert 0<=info.get('posterior_margin',0)<=1


def test_no_bank_is_original_baseline():
    rng=np.random.default_rng(4); p=rng.random((6,6))
    out,diag=correlated_decode(p,np.zeros((0,6,6),bool),condition='queue_ongoing')
    assert np.array_equal(out,baseline_decode(p)) and not diag['changed']


def test_disjoint_future_labels_not_needed_for_decode():
    # Exact same P6 + frozen bank produces identical prediction, regardless
    # of how true unseen horizon is subsequently defined.
    rng=np.random.default_rng(5)
    p=rng.random((6,13)); bank=rng.random((25,6,13))>.80
    a,_=correlated_decode(p,bank,condition='queue_ongoing')
    truth1=rng.random((6,13))>.9
    truth2=~truth1
    b,_=correlated_decode(p,bank,condition='queue_ongoing')
    assert np.array_equal(a,b)

def test_causal_bank_rejects_same_day_future_fold():
    truth=np.zeros((6,6,12),bool); truth[:,-1,4]=True
    ev=np.ones_like(truth)
    days=np.array([0,2,3,4,5,6]); folds=np.array([0,0,1,1,2,2])
    mask=causal_bank_mask(1,4,folds,days,ev,truth,condition='queue_onset')
    assert np.array_equal(np.flatnonzero(mask), np.array([0,1]))

if __name__=='__main__':
    test_baseline_matches_production_reference()
    test_shift_zero_pad_and_truth_scoring()
    test_joint_onset_validity_and_no_labels_used()
    test_no_bank_is_original_baseline()
    test_disjoint_future_labels_not_needed_for_decode()
    test_causal_bank_rejects_same_day_future_fold()
    print('6 synthetic/unit tests PASS (not competition performance)')
