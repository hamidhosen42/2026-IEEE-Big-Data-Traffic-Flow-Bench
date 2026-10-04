"""Decide the 2026-10-04 chain's S5 slot once S4 has been scored (or at 20:40 UTC at the latest), unless S5 already has an override.
  - if more FD-feature Task 1 members are complete than the chain base uses -> T1 mix with all of them (W_FD 0.5)
  - otherwise -> T1 mix with the members already used, FD weight 0.7 instead of 0.5
Log: s5_planner.log."""
import json, os, re, sys, time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN_F = os.path.expanduser('~/.kaggle/access_token')
while not os.path.exists(TOKEN_F):
    time.sleep(60)
import kjob  # noqa: E402

LOG = open(os.path.join(HERE, 's5_planner.log'), 'a', buffering=1)
STATE_F = os.path.join(HERE, 'auto_chain_oct4b_state.json')
OV = os.path.join(HERE, 'auto_chain_oct4b_override_S5.json')
FDM_ALL = ('tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11', 'tfb-t1v13s12', 'tfb-t1v13s13')
LAST = dt.datetime(2026, 10, 4, 19, 30)   # fallback time: a better S5 override may be written by hand before this


def log(*a):
    LOG.write(f'[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] ' + ' '.join(str(x) for x in a) + '\n')


def state():
    return json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}


def complete(k):
    for _ in range(30):
        try:
            return str(kjob.status(k).get('status', '')).lower().endswith('complete')
        except Exception:
            time.sleep(30)
    return False


log('S5 planner started')
while dt.datetime.utcnow() < LAST and not os.path.exists(OV):
    time.sleep(60)
if os.path.exists(OV) or 'S5' in {d['slot'] for d in state().get('done', [])}:
    log('S5 already decided -> nothing to do'); sys.exit(0)
st = state()
used = st.get('fd_used') or ['tfb-t1v13', 'tfb-t1v13s10', 'tfb-t1v13s11']
fdm = [k for k in FDM_ALL if complete(k + 'a') and complete(k + 'b')]
if len(fdm) > len(used):
    ov = {'type': 't1mix', 'w_fd': 0.5, 'fd_members': fdm, 'thr': 0.0001,
          'desc': f'EXP_COMBO_S5: T1 = 0.5 FD-feature bag ({len(fdm)}) + 0.5 old bag(8), recon + TV'}
else:
    ov = {'type': 't1mix', 'w_fd': 0.7, 'fd_members': used, 'thr': 0.0001,
          'desc': f'EXP_COMBO_S5: T1 = 0.7 FD-feature bag ({len(used)}) + 0.3 old bag(8), recon + TV'}
tmp = OV + '.tmp'
json.dump(ov, open(tmp, 'w'), indent=1); os.replace(tmp, OV)
log('S5 override written', json.dumps(ov))
