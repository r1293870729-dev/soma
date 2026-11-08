"""Check main-signal identity; --formal additionally fails on unfinished fields.

This check is not independent scientific acceptance or a complete manual validator.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from soma_b import validate_main_signal
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/signal_b_final.json')
p.add_argument('--formal',action='store_true');a=p.parse_args()
d=json.loads(a.config.read_text(encoding='utf8'))
try:validate_main_signal(d['signal'])
except (KeyError,ValueError) as e:print('INVALID:',e);raise SystemExit(2)
print('Main signal locked: B / mass / fixed reference / empirical edge.')
if a.formal:
    errors=[]
    for key,val in d.get('numeric_parameters',{}).items():
        if val is None:errors.append('numeric_parameters.'+key)
    for key,val in d.get('statistics',{}).items():
        if val is None:errors.append('statistics.'+key)
    if not d.get('formal_test_unlocked'):errors.append('formal_test_unlocked')
    if not d.get('numeric_parameters'):errors.append('numeric_parameters missing')
    if not d.get('statistics'):errors.append('statistics missing')
    if d.get('gates',{}).get('A0')!='accepted':errors.append('Gate A0')
    if errors:
        print('FORMAL BLOCKED:',', '.join(errors));raise SystemExit(2)
    print('Metadata preflight passed; independent task review still required.')
