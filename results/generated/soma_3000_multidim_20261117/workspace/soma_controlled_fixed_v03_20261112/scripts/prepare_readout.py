"""Download only the selected official weights; no dataset/model training."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'
SHA256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
target=ROOT/'assets/readouts/resnet18-f37072fd.pth'
if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_suffix('.pth.partial')
    with urllib.request.urlopen(URL,timeout=45) as source,part.open('wb') as dest:
        for block in iter(lambda:source.read(1024*1024),b''):dest.write(block)
    if hashlib.sha256(part.read_bytes()).hexdigest()!=SHA256:raise ValueError('Official weight hash mismatch')
    part.replace(target)
if hashlib.sha256(target.read_bytes()).hexdigest()!=SHA256:raise ValueError('Existing weight hash mismatch')
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=SHA256),indent=2))
