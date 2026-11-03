"""Build a completely synthetic lookalike lab cache and run the actual CLI.
This is an integration test, not a real TrafficFlowBench experiment.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys, tempfile
import numpy as np
import pandas as pd


def main():
    rng=np.random.default_rng(112)
    with tempfile.TemporaryDirectory() as temp:
        root=pathlib.Path(temp)/'cache'
        for panel in ['D7_I10_E','D7_I10_W']:
            W,H,L=32,6,22
            truth=np.zeros((W,H,L),bool); prob=np.zeros((W,H,L),np.float32)
            for i in range(W):
                center=int(rng.integers(4,18))
                for h in range(H):
                    width=min(4,max(1,h//2+1))
                    truth[i,h,max(0,center-width):min(L,center+1)]=True
                # Intentionally noisy but future-truth-generated synthetic P6 for engineering smoke.
                prob[i] = np.clip(.13+ .62*truth[i]+ rng.normal(0,.22,(H,L)),0,1)
            ev=(rng.random((W,H,L))<.87)
            d=np.arange(W,dtype=int); t=np.full(W,120); fold=d*4//W
            win=pd.DataFrame({'d':d,'t':t,'fold':fold,'cond':'queue_ongoing'})
            opath=root/'oof'/'smoke'; opath.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(opath/f'probs__offlike__{panel}.npz',P6=prob.astype(np.float16),done=np.ones(W,bool),d=d,t=t)
            for name in ['offlike','faith3']:
                p=root/'rows'/f'{panel}__{name}__queue_ongoing';p.mkdir(parents=True)
                np.savez_compressed(p/'aux.npz', truth=truth,ev=ev)
                win.to_pickle(p/'win.pkl')
        output=pathlib.Path(temp)/'result.json'
        cmd=[sys.executable,str(pathlib.Path(__file__).parent/'run_oof_experiment.py'),
             '--cache-root',str(root),'--arm','smoke','--set','offlike','--bank-set','faith3',
             '--condition','queue_ongoing','--output',str(output)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        if result.returncode: raise AssertionError(result.stderr+'\n'+result.stdout)
        report=json.loads(output.read_text())
        assert report['windows']==64 and report['panels']==2
        assert report['diagnostic_gate']=='FAIL' and report['deployment_gate'].startswith('UNVERIFIED')
        assert -1 <=report['delta_condition_iou']<=1
        print('Integration PASS: 64 purely synthetic windows; correct cache parsing, OOF scoring, daily bootstrap, JSON.')
        print('Gate blocked deployment for incomplete two-panel synthetic fixture:', report['diagnostic_gate'])

if __name__=='__main__': main()
