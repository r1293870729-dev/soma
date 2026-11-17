"""Keep a local progress file current without importing torch or changing workers."""
import runpy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
while True:
    runpy.run_path(str(ROOT/'scripts/reconstruction_status.py'),run_name='__main__')
    if (ROOT/'results/reconstructed_B_001/job_failures.txt').exists():break
    time.sleep(15)
