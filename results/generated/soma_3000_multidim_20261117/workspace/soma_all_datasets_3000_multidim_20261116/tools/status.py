"""Terminal progress bar, reads monitor output only; no training imports."""
from pathlib import Path
import json,time,argparse
ROOT=Path(__file__).resolve().parents[1]
def show():
    p=ROOT/'reports/progress.json'
    if not p.exists():print('Waiting for monitor');return
    v=json.loads(p.read_text());print('SOMA 3000-step multi-dimension run · '+v['status'])
    for e in v['entries']:
        n=e['complete'];total=e['planned'];bar='█'*int(28*n/total)+'░'*(28-int(28*n/total));print(f'{e["dataset"]:12} [{bar}] {n:2}/{total:2} '+e['status'])
        for a in e['active']:print('  '+a)
    u=v['new_optimizer_updates'];total=v['planned_new_optimizer_updates'];print(f'New optimizer updates: {u:,}/{total:,} ({100*u/total:.1f}%)')
    print('Progress heartbeat: '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(v['updated_at'])))
    if v['failures']:print('FAILURES: '+json.dumps(v['failures'],ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    try:
        while True:
            if args.watch:print('\033[2J\033[H',end='')
            show()
            if not args.watch:break
            time.sleep(5)
    except KeyboardInterrupt:pass
