"""Release idle-sleep guard when monitoring stops, fails, or becomes stale."""
from pathlib import Path
import time,json
ROOT=Path(__file__).resolve().parents[1];start=time.monotonic()
while time.monotonic()-start<36*3600:
    file=ROOT/'reports/LIVE_PROGRESS.json'
    if (ROOT/'reports/MONITOR_FAILED.json').exists():break
    if file.exists():
        if json.loads(file.read_text()).get('status') in ['complete','failed']:break
        if time.time()-file.stat().st_mtime>600:break
    time.sleep(20)
