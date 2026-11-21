"""Hold idle-sleep prevention only while this experiment monitor is alive."""
from pathlib import Path
import json,time
root=Path(__file__).resolve().parents[1];started=time.monotonic()
while time.monotonic()-started<24*3600:
    live=root/'reports/LIVE_PROGRESS.json'
    if (root/'reports/monitor_failed.json').exists():break
    if live.exists():
        state=json.loads(live.read_text())
        if state.get('status') in ['complete','failed']:break
        if time.time()-live.stat().st_mtime>600:break
    time.sleep(20)
print('Experiment idle-sleep guard released.',flush=True)
