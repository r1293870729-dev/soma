#!/usr/bin/env python3
"""Check freeze metadata without selecting parameters or launching final tests.
The supplied draft must exit 2. Passing metadata is not scientific acceptance.
"""
import argparse
import json
from pathlib import Path

def check(p: dict) -> list[str]:
    errors=[]
    def require(key,obj=p):
        value=obj.get(key)
        if value is None or value=='' or value==[] or value=={}:
            errors.append(f'author-input-needed: {key}')
    if p.get('status')!='frozen': errors.append('protocol status is not frozen')
    if p.get('final_test_unlocked') is not True: errors.append('final test remains locked')
    if p.get('scope_decisions_approved') is not True: errors.append('scope not approved')
    for key in ('approved_by','frozen_at','data_role_manifest_sha256','code_commit','environment_sha256'):
        require(key)
    q=p.get('target_question')
    if q not in ('Q1','Q2','Q3'): errors.append('target_question must be Q1/Q2/Q3')
    if p.get('signal',{}).get('definition')!='fixed-white-empirical-edge-mass':
        errors.append('signal differs from manual; explicit protocol revision needed')
    for key in ('W','alpha','beta','energy_floor','edge_windows','numeric_tolerance','readout_sha256'):
        require(key,p.get('signal',{}))
    for key in ('highest_unit_definition','planned_highest_units','nested_seeds','stopping_rule','failure_rerun_policy','endpoints','comparisons','families','bootstrap_draws','sign_flip_draws'):
        require(key,p.get('statistics',{}))
    needed=['A0']+(['A'] if q in ('Q2','Q3') else [])+(['B'] if q=='Q3' else [])
    for gate in needed:
        if p.get('gates',{}).get(gate)!='accepted': errors.append(f'Gate {gate} is not accepted')
    if p.get('author_inputs'): errors.append('unresolved author inputs remain')
    return errors

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();p=json.loads(args.protocol.read_text(encoding='utf-8'))
    errors=check(p)
    print(json.dumps({'permitted_by_metadata_check':not errors,'errors':errors,'notice':'Not a security sandbox or substitute for scientific review.'},ensure_ascii=False,indent=2))
    raise SystemExit(2 if errors else 0)
if __name__=='__main__':main()
