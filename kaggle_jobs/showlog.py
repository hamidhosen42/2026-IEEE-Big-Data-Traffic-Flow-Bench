import json, sys
raw = open(sys.argv[1]).read()
try:
    L = json.loads(raw); txt = ''.join(x.get('data', '') for x in L)
except Exception:
    txt = raw
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
print(txt[-n:])
