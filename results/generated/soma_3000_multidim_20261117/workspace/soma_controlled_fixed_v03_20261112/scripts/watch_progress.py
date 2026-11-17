"""Read-only terminal progress; closing this viewer does not stop training."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.b_weight_sensitivity.progress import bar

try:
    while True:
        meta_path=ROOT/'reports/latest_execution.json'
        if meta_path.exists():
            meta=json.loads(meta_path.read_text());run=meta['experiment_id']
            path=ROOT/'results/generated'/run/'progress.json'
            print('\033[2J\033[H',end='')
            print('SOMA v0.3 controlled | 固定 rec=1, KL=0.001')
            print('实验 ID:',run)
            if path.exists():
                data=json.loads(path.read_text())
                print('总体',bar(data.get('completed_runs',0),data.get('total_runs',3)))
                print('当前',bar(data.get('step',0),data.get('total_steps',3000)))
                print('阶段:',data.get('phase'),'seed:',data.get('unit',{}).get('seed'))
            receipt=ROOT/'reports'/('exit_'+run+'.json')
            if receipt.exists():
                result=json.loads(receipt.read_text());print('流程状态:',result['status'])
                print('报告:',ROOT/'reports/controlled'/run)
                break
        else:
            print('等待 controlled 运行启动...',flush=True)
        sys.stdout.flush();time.sleep(2)
except KeyboardInterrupt:
    print('\n已退出进度查看；训练进程不受影响。')
