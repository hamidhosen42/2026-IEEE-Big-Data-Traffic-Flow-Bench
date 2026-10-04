"""Probability blends of cached arms, scored with the production decode and stored as arms (results.pkl) for the report.
Usage: exp_firstcross_blend.py <armA> <armB> <w_B[,w_B...]> [out prefix]     blend = (1 - w_B) * A + w_B * B
Light (no training)."""
import json, os, sys
import numpy as np
import pandas as pd
import lab
import exp_firstcross_lib as fc

a, b = sys.argv[1], sys.argv[2]
ws = [float(x) for x in sys.argv[3].split(',')]
prefix = sys.argv[4] if len(sys.argv) > 4 else f'fc_BL__{a}__{b}'
pd.set_option('display.width', 250)
for w in ws:
    name = f'{prefix}__w{int(round(w * 100)):03d}'
    df = pd.concat([fc.blend_frame([a, b], [1 - w, w], s, name) for s in fc.EVALS], ignore_index=True)
    os.makedirs(fc.arm_dir(name), exist_ok=True)
    df.to_pickle(f'{fc.arm_dir(name)}/results.pkl')
    json.dump(dict(arm=name, blend=[a, b], weights=[1 - w, w]), open(f'{fc.arm_dir(name)}/config.json', 'w'))
    print(name, {s: (round(lab.aggregate(df[df.set == s]), 4), int((df.set == s).sum())) for s in fc.EVALS}, flush=True)
