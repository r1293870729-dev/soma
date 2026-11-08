"""Recompute final B from saved pre-update residuals, with no training side effects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .signal import BMonitor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--residuals',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stride',type=int,default=1,help='Record every N samples; no samples skipped in the ring')
    a=p.parse_args()
    if a.stride < 1:p.error('--stride must be positive')
    m=BMonitor.load(a.reference)
    if m.samples_seen:p.error('Replay from start requires a reference-only asset, not a resumed monitor')
    r=np.load(a.residuals,allow_pickle=False)
    if r.ndim!=2 or len(r)<1:p.error('Residuals must be a nonempty N x d numeric .npy')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8') as out:
        for i in range(len(r)):
            row=m.update(r[i:i+1])
            if (i+1)%a.stride==0 or i+1==len(r):
                out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\\n')
    print(f'Replayed {len(r)} residuals as {m.reference_id()} -> {a.out}')
if __name__=='__main__':main()
