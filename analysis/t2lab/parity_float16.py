"""Parity of the local npz loader against the production loader on raw files. Raw train parquet of the Task 2 panels is not
on this machine; D12_I405_S (raw files local, not a Task 2 panel) is used to measure the only lossy step of the export:
speed stored as float16. Production load_panel is executed from the ast-extracted code with REL -> data/kaggle_public."""
import numpy as np, lab
lab.prod.REL = f'{lab.ROOT}/data/kaggle_public'
P = lab.prod.load_panel('D12_I405_S')
V = P['V']; cut = P['cut']; E = P['E']; Q = P['Q']
V16 = V.astype(np.float16).astype(np.float32)
Q16 = np.isfinite(V16) & (V16 <= cut[None, None, :])
flip = Q16 != Q
print('days', V.shape, 'observed cells', int(E.sum()), 'queued', int(Q.sum()), 'queue-state flips after float16 round trip', int(flip.sum()),
      'share of observed', flip.sum() / E.sum(), 'share of queued', flip.sum() / Q.sum(), 'max abs speed error', float(np.nanmax(np.abs(V16 - V))))
w = lab.prod.build_windows(P, global_spacing=True)
P2 = dict(P, V=V16, Q=Q16)
w2 = lab.prod.build_windows(P2, global_spacing=True)
print('pseudo windows float32', len(w), 'float16', len(w2), 'common', len(set(w) & set(w2)))
