#!/usr/bin/env python3
"""Verify delivered originals and optionally the full local file manifest."""
import argparse,hashlib,json
from pathlib import Path

def sha256(p:Path)->str:
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def verify(root:Path,full:bool=False)->dict:
    name='file_manifest.json' if full else 'uploads.json'
    manifest=root/'registries'/name
    if not manifest.is_file():
        return {'checked':0,'passed':False,'errors':[{'path':str(manifest),'problem':'Missing manifest; use the full local bundle before full verification'}]}
    entries=json.loads(manifest.read_text(encoding='utf-8'))
    errors=[]
    for entry in entries:
        path=entry.get('repository_path',entry.get('path'));p=(root/path).resolve()
        if root.resolve() not in p.parents:
            errors.append({'path':path,'problem':'unsafe path'});continue
        if not p.is_file(): errors.append({'path':path,'problem':'missing; full bundle/LFS may be required'})
        elif p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:
            errors.append({'path':path,'problem':'size/hash mismatch; check for an LFS pointer'})
    return {'checked':len(entries),'errors':errors,'passed':not errors}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--full',action='store_true');a=ap.parse_args()
    result=verify(a.root,a.full);print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__':main()
